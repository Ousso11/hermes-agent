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
    "[{count} lines omitted (Read offset={offset})]"
)

# Strict, line-counted removal marker ("[N lines removed ...]"). Used only by
# the marker-count alignment fallback, where the count denotes whole lines.
_placeholder_re = re.compile(r"\[\s*(\d+)\s+lines?\s+removed[^\]]*\]", re.IGNORECASE)

# The GENERAL family of "content was dropped here" markers a compressor may
# emit, regardless of unit (tokens/lines/chars/words) or verb. The anchor-based
# rewriter treats these purely as a SIGNAL that a gap exists; the recovered line
# range comes from anchoring, so the marker's unit/count never matters. It
# deliberately excludes "omitted" — the verb our own references use — so a
# reference is never mistaken for a drop marker.
_drop_marker_re = re.compile(
    r"\[[^\]]*\b(?:removed|dropped|truncated|elided|snipped|redacted|hidden|stripped|cut)\b[^\]]*\]",
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

    def emit_gap(idx: int) -> None:
        nonlocal cursor
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
            cursor = idx

    for cl in compressed.split("\n"):
        # A line that is ONLY a drop marker (any unit/verb) is removed; the gap
        # it implies is recovered from the anchor skip instead.
        if _drop_marker_re.search(cl) and _drop_marker_re.sub("", cl).strip() == "":
            continue
        n = normalize_line(cl)
        if n == "":
            if build_out:
                out_lines.append(cl)
            continue
        idx = _find_run(o_norm, [n], cursor)
        if idx < 0:
            # Not verbatim-after-cursor: a paraphrase or a reordered anchor. Keep
            # it as-is and do NOT fabricate a gap (avoids mis-anchoring).
            if _find_run(o_norm, [n], 0) >= 0:
                reordered = True
            if build_out:
                out_lines.append(cl)
            continue
        emit_gap(idx)
        if build_out:
            out_lines.append(cl)
        cursor = idx + 1
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
    """Map each strict ``[N lines removed]`` marker back to a concrete line range.

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
    for p in parts:
        if p["is_gap"]:
            end = cursor + p["gap"] - 1
            if end >= len(o_raw):
                end = len(o_raw) - 1
            gaps.append(Gap(start=cursor, end=end, count=p["gap"]))
            cursor += p["gap"]
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
        cursor = at + len(seg_n)
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
    out, gaps, anchored, _ = _anchor_walk(cache_path, original, compressed, True)
    if anchored:
        return out, gaps, True

    ag, aok, _ = align_placeholders(original, compressed)
    if aok and ag:
        counter = {"i": 0}

        def _repl(_m: "re.Match[str]") -> str:
            i = counter["i"]
            counter["i"] += 1
            gap = ag[i] if i < len(ag) else _whole_file_gap(original)
            return build_reference(cache_path, gap)

        rewritten = _placeholder_re.sub(_repl, compressed)
        return rewritten, ag, True

    # Safe fallback: replace every drop marker (any format) with a whole-file ref.
    whole = build_reference(cache_path, _whole_file_gap(original))
    rewritten = _drop_marker_re.sub(whole, compressed)
    return rewritten, [_whole_file_gap(original)], False
