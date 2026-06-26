"""Addressable recovery references for compressed tool output.

When Compresr's tool-output API returns compressed text with inline drop
markers (``[5 lines removed]``, ``[54 tokens dropped]``, ...), we rewrite each
marker into an ADDRESSABLE reference the agent can recover with its NATIVE
``read_file``/``grep`` tools — either the whole cached original or a surgical
slice::

    [compresr: 5 lines omitted · .compresr/cache/a3f9 L40-44 · Read(offset=40,limit=5) or Grep to recover]

The mechanism does NOT trust marker counts/units. The primary path anchors the
compressed output's verbatim KEPT lines against the cached original and derives
the dropped line range from every original span skipped between anchors. That
works identically for line markers, token markers, and silent drops, and is
robust to ~6% whitespace drift from the compressor (normalized line matching).

This is what makes the compression *lossless-by-recovery*: every dropped byte
is one ``read_file(offset, limit)`` away.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

# Workspace-relative directory that holds cached originals. Hermes's native
# read_file/grep tools resolve relative paths against the task's tracked cwd,
# so a reference pointing here is directly recoverable by the model.
CACHE_DIR_NAME = ".compresr/cache"

# Hermes's read_file path truncates any single line longer than this (see
# tools/tool_output_limits.py DEFAULT_MAX_LINE_LENGTH) and appends "... [truncated]".
# A dropped line longer than this is therefore NOT byte-exact recoverable, so we
# refuse to drop any span containing one — we keep it inline instead. This is the
# guard that preserves the lossless-by-recovery invariant for long lines.
MAX_RECOVERABLE_LINE_CHARS = 2000

# The LLM-visible recoverable placeholder. Offsets are 1-indexed to match
# Hermes's read_file_tool(path, offset>=1, limit) contract exactly.
#   [compresr: 5 lines omitted · .compresr/cache/a3f9 L40-44 · Read(offset=40,limit=5) or Grep to recover]
ADDRESSABLE_REF_FORMAT = (
    "[compresr: {count} lines omitted · {path} L{start}-{end} · "
    "Read(offset={offset},limit={count}) or Grep to recover]"
)

# Drop-verb vocabulary shared by both marker regexes. Excludes "omitted" (the
# verb our own references use) so a reference is never read as a drop marker.
_DROP_VERB = r"removed|dropped|truncated|elided|snipped|redacted|hidden|stripped|cut"

# Counted drop marker ("[N lines removed]", "[N tokens dropped]", ...). Any unit,
# any verb — toc_latte_v2 emits tokens, not lines. Count is only a tie-breaker for
# a trailing gap; real line ranges come from anchoring.
_placeholder_re = re.compile(
    r"\[\s*(\d+)\s+"
    r"(?:lines?|tokens?|chars?|characters?|words?|bytes?)?\s*"
    rf"(?:{_DROP_VERB})[^\]]*\]",
    re.IGNORECASE,
)

# General "content dropped here" marker — treated purely as a gap signal; the
# line range comes from anchoring, so unit/count never matter.
_drop_marker_re = re.compile(
    rf"\[[^\]]*\b(?:{_DROP_VERB})\b[^\]]*\]",
    re.IGNORECASE,
)


@dataclass
class Gap:
    """A contiguous run of original lines a placeholder stands in for.

    ``start``/``end`` are 0-based, inclusive line indices into the original.
    """

    start: int
    end: int
    count: int


def normalize_line(s: str) -> str:
    """Trim and collapse internal whitespace runs to a single space.

    Cosmetic reformatting by the compressor (re-indentation, CRLF, collapsed
    runs) must not defeat anchoring.
    """
    return " ".join(s.split())


def _find_run(hay: List[str], seg: List[str], start: int) -> int:
    """First index >= ``start`` where ``seg`` appears consecutively in ``hay``, else -1."""
    if not seg or start < 0:
        return -1
    last = len(hay) - len(seg)
    for i in range(start, last + 1):
        if hay[i : i + len(seg)] == seg:
            return i
    return -1


def build_reference(cache_path: str, gap: Gap) -> str:
    """Render an addressable recovery reference for ``gap`` of the cached file at
    ``cache_path`` (an absolute path, so recovery survives a later ``cd``)."""
    # L-range is 0-based inclusive (matches the stored file's line indices for
    # debuggability); the Read offset is 1-indexed (start + 1) for the tool.
    return ADDRESSABLE_REF_FORMAT.format(
        count=gap.count,
        path=cache_path,
        start=gap.start,
        end=gap.end,
        offset=gap.start + 1,
    )


def _whole_file_gap(original: str) -> Gap:
    total = original.count("\n") + 1
    return Gap(start=0, end=total - 1, count=total)


def _split_trim_boundary(s: str) -> List[str]:
    """Split a kept segment into lines, dropping only blank lines ADJACENT to
    placeholder boundaries (join artifacts) while preserving internal blanks so
    anchoring stays aligned with the original.
    """
    lines = s.split("\n")
    while lines and lines[0].strip() == "":
        lines = lines[1:]
    while lines and lines[-1].strip() == "":
        lines = lines[:-1]
    return lines


def _anchor_walk(
    cache_path: str, original: str, compressed: str, build_out: bool
) -> Tuple[str, List[Gap], bool, bool]:
    """Shared engine for :func:`infer_gaps` and :func:`rewrite_placeholders`.

    Walks the compressed lines, drops marker-only lines (any unit/verb), anchors
    each verbatim content line against the original, and derives a Gap from every
    original range skipped between anchors (plus a trailing gap). When
    ``build_out`` is true, also returns the compressed content with each gap
    replaced by an addressable reference and paraphrased lines preserved.

    Returns ``(out, gaps, anchored, reordered)``.
    """
    o_raw = original.split("\n")
    o_norm = [normalize_line(l) for l in o_raw]

    out_lines: List[str] = []
    gaps: List[Gap] = []
    anchored = False
    reordered = False
    cursor = 0
    remaining_counts = Counter(o_norm)

    def advance_cursor(new_cursor: int) -> None:
        nonlocal cursor
        if new_cursor <= cursor:
            return
        for line in o_norm[cursor:new_cursor]:
            remaining_counts[line] -= 1
        cursor = new_cursor

    def emit_gap(idx: int) -> None:
        if idx > cursor:
            span = o_raw[cursor:idx]
            if any(len(l) > MAX_RECOVERABLE_LINE_CHARS for l in span):
                # Contains a line the recovery read path can't return verbatim —
                # keep the span inline rather than emit an unrecoverable reference.
                if build_out:
                    out_lines.extend(span)
            else:
                g = Gap(start=cursor, end=idx - 1, count=idx - cursor)
                gaps.append(g)
                if build_out:
                    out_lines.append(build_reference(cache_path, g))
            advance_cursor(idx)

    for raw_cl in compressed.split("\n"):
        # Strip inline drop markers before anchoring: toc_latte_v2 fuses them onto
        # the end of a kept line ("…kept text[54 tokens dropped]"), which otherwise
        # fails to anchor and leaks the marker into output.
        had_marker = bool(_drop_marker_re.search(raw_cl))
        cl = _drop_marker_re.sub("", raw_cl) if had_marker else raw_cl
        n = normalize_line(cl)
        if n == "":
            # Marker-only line → drop it (gap is anchored); keep genuinely blank
            # original lines to stay aligned.
            if build_out and not had_marker:
                out_lines.append(cl)
            continue
        if remaining_counts[n] > 1:
            # A repeated single-line anchor can point at the wrong original
            # span. Mark the output as low-confidence so callers can fail open
            # instead of emitting confidently wrong recovery references.
            reordered = True
            if build_out:
                out_lines.append(cl)
            continue
        idx = _find_run(o_norm, [n], cursor)
        if idx < 0:
            # Not verbatim-after-cursor.
            if _find_run(o_norm, [n], 0) >= 0:
                # Same content elsewhere → reordered anchor; flag low-confidence.
                reordered = True
            elif had_marker:
                # Marker-bearing line matching nothing in the original is a
                # fabricated hybrid (wrong-ts + marker). Pure gap signal: drop it.
                continue
            # Otherwise a genuine paraphrase: keep as-is, do NOT fabricate a gap.
            if build_out:
                out_lines.append(cl)
            continue
        emit_gap(idx)
        if build_out:
            out_lines.append(cl)
        advance_cursor(idx + 1)
        anchored = True

    # Trailing gap: the tail of the original was dropped after the last anchor.
    if anchored and cursor < len(o_raw):
        span = o_raw[cursor:]
        if any(len(l) > MAX_RECOVERABLE_LINE_CHARS for l in span):
            if build_out:
                out_lines.extend(span)
        else:
            g = Gap(start=cursor, end=len(o_raw) - 1, count=len(o_raw) - cursor)
            gaps.append(g)
            if build_out:
                out_lines.append(build_reference(cache_path, g))

    out = "\n".join(out_lines) if build_out else ""
    return out, gaps, anchored, reordered


def infer_gaps(original: str, compressed: str) -> Tuple[List[Gap], bool, bool]:
    """Recover the original line ranges dropped from an extractive compressed
    output, by anchoring its verbatim kept lines. ``ok`` is false when nothing
    could be anchored (fully paraphrased/degenerate) — fall back to whole-file.
    """
    _, gaps, anchored, reordered = _anchor_walk("", original, compressed, False)
    return gaps, anchored, reordered


def align_placeholders(
    original: str, compressed: str
) -> Tuple[List[Gap], bool, bool]:
    """Map each counted drop marker (``[N lines removed]``, ``[N tokens dropped]``,
    ...) back to a concrete line range.

    The compressed text is verbatim KEPT segments separated by markers; we anchor
    each kept segment in order and the run between two anchors is the removed gap.
    Returns ``(gaps, ok, reordered)``; ok=false if a kept segment can't be located.
    """
    o_raw = original.split("\n")
    o_norm = [normalize_line(l) for l in o_raw]

    # Parts: alternating kept text and gap counts, split on the strict marker.
    parts: List[dict] = []
    last = 0
    for m in _placeholder_re.finditer(compressed):
        parts.append({"kept": compressed[last : m.start()], "is_gap": False})
        parts.append({"gap": int(m.group(1)), "is_gap": True})
        last = m.end()
    parts.append({"kept": compressed[last:], "is_gap": False})

    gaps: List[Gap] = []
    reordered = False
    cursor = 0
    pending_gap_start: int | None = None
    pending_gap_count = 0

    def close_pending_gap(end: int) -> None:
        nonlocal pending_gap_start, pending_gap_count
        if pending_gap_start is None:
            return
        if end >= pending_gap_start:
            count = end - pending_gap_start + 1
            gaps.append(Gap(start=pending_gap_start, end=end, count=count))
        pending_gap_start = None
        pending_gap_count = 0

    for p in parts:
        if p["is_gap"]:
            if pending_gap_start is None:
                pending_gap_start = cursor
            pending_gap_count += p["gap"]
            continue
        seg = _split_trim_boundary(p["kept"])
        if not seg:
            continue
        seg_n = [normalize_line(l) for l in seg]
        at = _find_run(o_norm, seg_n, cursor)
        if at < 0:
            at = _find_run(o_norm, seg_n, 0)
            if at < 0:
                return gaps, False, reordered
            reordered = True
        close_pending_gap(at - 1)
        cursor = at + len(seg_n)
    if pending_gap_start is not None:
        end = min(len(o_raw) - 1, pending_gap_start + pending_gap_count - 1)
        close_pending_gap(end)
    return gaps, True, reordered


def rewrite_placeholders(
    cache_path: str, original: str, compressed: str
) -> Tuple[str, List[Gap], bool]:
    """Turn compressed tool output into text carrying addressable recovery refs.

    Primary path anchors verbatim kept lines and replaces every skipped original
    range with a surgical reference. Falls back to strict line-count alignment,
    then to whole-file references (ok=false) for degenerate/paraphrased output.
    Returns ``(rewritten, gaps, ok)``.
    """
    out, gaps, anchored, anchor_reordered = _anchor_walk(
        cache_path, original, compressed, True
    )
    if anchored and not anchor_reordered:
        return out, gaps, True
    ag, aok, align_reordered = align_placeholders(original, compressed)
    if align_reordered:
        return original, [], False
    if aok and ag:
        counter = {"i": 0}

        def _repl(_m: "re.Match[str]") -> str:
            i = counter["i"]
            counter["i"] += 1
            gap = ag[i] if i < len(ag) else _whole_file_gap(original)
            return build_reference(cache_path, gap)

        rewritten = _placeholder_re.sub(_repl, compressed)
        return rewritten, ag, True

    if anchor_reordered:
        return original, [], False

    if _drop_marker_re.search(compressed) is None:
        return original, [], False

    # Safe fallback: replace every drop marker (any format) with a whole-file ref.
    whole = build_reference(cache_path, _whole_file_gap(original))
    rewritten = _drop_marker_re.sub(whole, compressed)
    return rewritten, [_whole_file_gap(original)], False
