"""Tests for the tool_output_compresr plugin: recovery anchoring and the
transform_tool_result hook gating/fail-open behavior.

These cover the load-bearing claim — compression is lossless *by recovery*: the
addressable references map back to the exact dropped original line ranges.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugins.tool_output_compresr import recover  # noqa: E402
from plugins.tool_output_compresr import ToolOutputCompressor  # noqa: E402

ORIGINAL = "\n".join(
    [
        "line0 alpha",
        "line1 bravo",
        "line2 charlie",
        "line3 delta",
        "line4 echo",
        "line5 foxtrot",
    ]
)


def _recover_gap(original, gap):
    """Reconstruct the dropped span from the cached original, as the agent would
    via read_file(offset=gap.start+1, limit=gap.count)."""
    lines = original.split("\n")
    return lines[gap.start : gap.end + 1]


def _assert_gap_is_read_addressable(rewritten, gap, cache_path):
    assert f"{cache_path} L{gap.start}-{gap.end}" in rewritten
    assert f"Read(offset={gap.start + 1},limit={gap.count})" in rewritten


def _shell_file_ops_for_tmp_path(tmp_path):
    from tools.file_operations import ShellFileOperations

    class LocalEnv:
        cwd = str(tmp_path)

        def execute(self, command, cwd=None, timeout=None, stdin_data=None):
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd or self.cwd,
                input=stdin_data,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            return {
                "output": (result.stdout or "") + (result.stderr or ""),
                "returncode": result.returncode,
            }

    return ShellFileOperations(LocalEnv(), cwd=str(tmp_path))


def test_anchor_roundtrip_with_marker():
    compressed = "\n".join(
        ["line0 alpha", "line1 bravo", "[2 lines removed]", "line4 echo", "line5 foxtrot"]
    )
    rewritten, gaps, ok = recover.rewrite_placeholders(
        ".compresr/cache/abc123", ORIGINAL, compressed
    )
    assert ok
    assert len(gaps) == 1
    g = gaps[0]
    assert (g.start, g.end, g.count) == (2, 3, 2)
    # The dropped span is exactly recoverable.
    assert _recover_gap(ORIGINAL, g) == ["line2 charlie", "line3 delta"]
    # The reference is addressable and offset is 1-indexed (start+1).
    assert ".compresr/cache/abc123 L2-3" in rewritten
    assert "Read(offset=3,limit=2)" in rewritten
    assert "line4 echo" in rewritten  # kept content preserved verbatim


def test_anchor_silent_drop_no_marker():
    """Even with NO drop marker, anchoring infers the gap from the skip."""
    compressed = "\n".join(["line0 alpha", "line1 bravo", "line4 echo", "line5 foxtrot"])
    _, gaps, ok = recover.rewrite_placeholders(".compresr/cache/id1", ORIGINAL, compressed)
    assert ok
    assert len(gaps) == 1
    assert (gaps[0].start, gaps[0].end) == (2, 3)


def test_anchor_tolerates_whitespace_drift():
    """~6% whitespace reformatting by the compressor must not defeat anchoring."""
    compressed = "\n".join(
        ["line0   alpha", "line1\tbravo", "[tokens dropped]", "line4 echo", "line5 foxtrot"]
    )
    _, gaps, ok = recover.rewrite_placeholders(".compresr/cache/id2", ORIGINAL, compressed)
    assert ok
    assert (gaps[0].start, gaps[0].end) == (2, 3)


def test_trailing_gap():
    compressed = "\n".join(["line0 alpha", "line1 bravo"])
    _, gaps, ok = recover.rewrite_placeholders(".compresr/cache/id3", ORIGINAL, compressed)
    assert ok
    assert gaps[-1].end == 5  # tail dropped after last anchor


def test_fully_paraphrased_falls_back_to_whole_file():
    compressed = "a totally rewritten summary with no verbatim lines [content truncated]"
    rewritten, gaps, ok = recover.rewrite_placeholders(".compresr/cache/id4", ORIGINAL, compressed)
    assert not ok
    assert gaps == [recover._whole_file_gap(ORIGINAL)]
    assert ".compresr/cache/id4" in rewritten


def test_long_line_span_kept_inline_not_dropped():
    """A dropped span containing a line longer than the recovery read path can
    return verbatim (~2000 chars) must be kept INLINE, not turned into an
    unrecoverable reference — preserving lossless-by-recovery for long lines."""
    long_line = "DATA " + "x" * (recover.MAX_RECOVERABLE_LINE_CHARS + 500)
    lines = [f"line {i} anchor text" for i in range(20)]
    lines[10] = long_line
    original = "\n".join(lines)
    compressed = "\n".join(lines[:9] + ["[4 lines removed]"] + lines[13:])
    rewritten, gaps, ok = recover.rewrite_placeholders(".compresr/cache/ll", original, compressed)
    assert long_line in rewritten           # kept verbatim, recoverable in-context
    assert all(g.start > 12 or g.end < 9 for g in gaps)  # the long span isn't a gap


def test_reference_uses_given_absolute_path():
    """The reference embeds the path it is handed verbatim, so callers can pass an
    ABSOLUTE (cd-proof) cache path and recovery survives a later ``cd``."""
    compressed = "\n".join(
        ["line0 alpha", "line1 bravo", "[2 lines removed]", "line4 echo", "line5 foxtrot"]
    )
    abs_path = "/work/sub/.compresr/cache/abc123"
    rewritten, _, _ = recover.rewrite_placeholders(abs_path, ORIGINAL, compressed)
    assert f"{abs_path} L2-3" in rewritten
    assert "Read(offset=3,limit=2)" in rewritten


def test_reference_never_matches_drop_marker():
    """Our own 'omitted' reference must not be re-detected as a drop marker by
    EITHER the general signal regex or the broadened counted-marker regex."""
    ref = recover.build_reference(".compresr/cache/x", recover.Gap(0, 4, 5))
    assert recover._drop_marker_re.search(ref) is None
    assert recover._placeholder_re.search(ref) is None


def test_ambiguous_repeated_anchor_fails_open():
    """Repeated single-line anchors can point at the wrong span, so recovery
    should fail open instead of emitting a confidently wrong line reference."""
    original = "\n".join(
        [
            "start",
            "repeat",
            "first hidden payload",
            "repeat",
            "second hidden payload",
            "end",
        ]
    )
    compressed = "\n".join(["start", "repeat", "[content dropped]", "end"])
    rewritten, gaps, ok = recover.rewrite_placeholders(
        ".compresr/cache/ambiguous", original, compressed
    )
    assert not ok
    assert gaps == []
    assert rewritten == original


def test_anchor_accepts_repeated_line_that_is_unique_after_cursor():
    original = "\n".join(["same", "keep", "same", "tail"])
    compressed = "\n".join(["keep", "same", "tail"])

    rewritten, gaps, ok = recover.rewrite_placeholders(
        ".compresr/cache/repeated-after-cursor", original, compressed
    )

    assert ok
    assert [(g.start, g.end, g.count) for g in gaps] == [(0, 0, 1)]
    assert ".compresr/cache/repeated-after-cursor L0-0" in rewritten


def test_strict_marker_fallback_handles_repeated_kept_lines():
    original = "\n".join(
        ["header", "repeat", "payload a", "payload b", "repeat", "footer"]
    )
    compressed = "\n".join(["header", "repeat", "[2 lines removed]", "repeat", "footer"])

    rewritten, gaps, ok = recover.rewrite_placeholders(
        ".compresr/cache/repeated-strict", original, compressed
    )

    assert ok
    assert [(g.start, g.end, g.count) for g in gaps] == [(2, 3, 2)]
    assert ".compresr/cache/repeated-strict L2-3" in rewritten
    assert "Read(offset=3,limit=2)" in rewritten


def test_strict_marker_fallback_extends_bad_count_to_next_anchor():
    original = "\n".join(
        ["header", "repeat", "payload a", "payload b", "repeat", "footer"]
    )
    compressed = "\n".join(["header", "repeat", "[1 lines removed]", "repeat", "footer"])

    rewritten, gaps, ok = recover.rewrite_placeholders(
        ".compresr/cache/bad-count", original, compressed
    )

    assert ok
    assert [(g.start, g.end, g.count) for g in gaps] == [(2, 3, 2)]
    assert ".compresr/cache/bad-count L2-3" in rewritten
    assert "Read(offset=3,limit=2)" in rewritten


def test_inline_token_marker_anchors_and_does_not_leak():
    """toc_latte_v2 attaches its marker to the END of a kept line
    ("...kept[N tokens dropped]"), not on its own line. The kept text must still
    anchor (so the gap boundary is correct) and the marker must not leak into the
    rewritten output. Regression for the live-API round-trip."""
    original = "\n".join(["a one", "b two", "c three", "d four", "e five"])
    # Keeps a/b verbatim, marker attached to "b two", drops c/d/e.
    compressed = "a one\nb two[3 tokens dropped]"

    rewritten, gaps, ok = recover.rewrite_placeholders(
        ".compresr/cache/inline", original, compressed
    )

    assert ok
    assert "[3 tokens dropped]" not in rewritten  # stripped, no leak into context
    assert "b two" in rewritten  # kept line preserved, clean
    assert [(g.start, g.end, g.count) for g in gaps] == [(2, 4, 3)]  # c/d/e, not b
    assert "Read(offset=3,limit=3)" in rewritten


def test_midline_marker_hybrid_is_dropped_not_corrupting():
    """Live toc_latte_v2 sometimes fuses a WRONG prefix + the marker onto the next
    kept line ("[wrong-ts][N tokens dropped] real line"). That hybrid matches
    nothing verbatim; it must be dropped (pure gap signal), not emitted as a
    corrupted line — and the real span recovered by anchoring around it. Regression
    for the real-Hermes round-trip on a dense log."""
    original = "\n".join([f"L{i} val{i}" for i in range(8)])  # L0..L7
    compressed = "\n".join([
        "L0 val0", "L1 val1",
        "WRONGPREFIX[4 tokens dropped] L6 val6",  # hybrid: matches nothing verbatim
        "L7 val7",
    ])

    rewritten, gaps, ok = recover.rewrite_placeholders(
        ".compresr/cache/hybrid", original, compressed
    )

    assert ok
    assert "[4 tokens dropped]" not in rewritten   # marker stripped
    assert "WRONGPREFIX" not in rewritten          # corrupted hybrid dropped
    assert [(g.start, g.end) for g in gaps] == [(2, 6)]  # L2..L6 recovered as a gap
    assert "Read(offset=3,limit=5)" in rewritten


def test_strict_marker_fallback_handles_token_markers():
    """Abstractive output emits ``[N tokens dropped]``, not ``[N lines removed]``.
    The marker-count fallback must still split on it and emit an addressable Read
    reference — the count's unit is irrelevant, the gap's line range is anchored.
    (Regression for: strict regex only matched 'lines removed' → 0 token matches.)
    """
    original = "\n".join(
        ["header", "repeat", "payload a", "payload b", "repeat", "footer"]
    )
    compressed = "\n".join(
        ["header", "repeat", "[2 tokens dropped]", "repeat", "footer"]
    )

    rewritten, gaps, ok = recover.rewrite_placeholders(
        ".compresr/cache/token-strict", original, compressed
    )

    assert ok
    assert [(g.start, g.end, g.count) for g in gaps] == [(2, 3, 2)]
    assert ".compresr/cache/token-strict L2-3" in rewritten
    assert "Read(offset=3,limit=2)" in rewritten


def test_multiple_inline_token_markers_emit_exact_read_ranges():
    original = "\n".join(
        [
            "header",
            "keep alpha",
            "hidden one",
            "hidden two",
            "keep beta",
            "hidden three",
            "hidden four",
            "footer",
        ]
    )
    compressed = "\n".join(
        [
            "header",
            "keep alpha[2 tokens dropped]",
            "keep beta[2 tokens dropped]",
            "footer",
        ]
    )
    cache_path = ".compresr/cache/multi-inline"

    rewritten, gaps, ok = recover.rewrite_placeholders(cache_path, original, compressed)

    assert ok
    assert "tokens dropped" not in rewritten
    assert [(g.start, g.end, g.count) for g in gaps] == [(2, 3, 2), (5, 6, 2)]
    for gap in gaps:
        _assert_gap_is_read_addressable(rewritten, gap, cache_path)


def test_hook_skips_small_output():
    c = ToolOutputCompressor()
    c.enabled, c.api_key = True, "cmp_test"
    assert c.on_transform_tool_result(tool_name="grep", args={}, result="tiny") is None


def test_hook_compresses_large_output(monkeypatch):
    from plugins.tool_output_compresr import cache

    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 5
    # The cache write goes through Hermes's file-ops backend; capture it here so
    # the test doesn't spin up a real environment.
    stored = {}
    monkeypatch.setattr(
        cache, "store_original",
        lambda cid, content, task_id="default", **_: stored.update({cid: content})
        or cache.relative_cache_path(cid),
    )
    # Mock the API: pretend it kept the first 2 lines and dropped the rest.
    monkeypatch.setattr(
        c._client,
        "compress",
        lambda **kw: ("line0 alpha\nline1 bravo\n[4 lines removed]", {"tokens_saved": 4}),
    )
    out = c.on_transform_tool_result(
        tool_name="grep", args={"pattern": "line"}, result=ORIGINAL, tool_call_id="tc1"
    )
    assert out is not None
    assert "[compresr:" in out
    # The EXACT text the API compressed was persisted so the reference resolves.
    cache_id = c._cache_id(ORIGINAL)
    assert cache_id in stored


def test_hook_rewrites_inline_token_markers_to_recovery_refs(monkeypatch):
    from plugins.tool_output_compresr import cache

    original = "\n".join(
        [
            "header status=ok",
            "keep alpha result",
            "hidden diagnostic one",
            "hidden diagnostic two",
            "hidden diagnostic three",
            "keep beta result",
            "footer complete",
        ]
    )
    compressed = "\n".join(
        [
            "header status=ok",
            "keep alpha result[3 tokens dropped]",
            "keep beta result",
            "footer complete",
        ]
    )
    cache_path = ".compresr/cache/inline-hook"
    stored = {}

    def store_original(cid, content, task_id="default", **_):
        stored[cid] = (content, task_id)
        return cache_path

    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 1
    monkeypatch.setattr(cache, "store_original", store_original)
    monkeypatch.setattr(
        c._client,
        "compress",
        lambda **kw: (compressed, {"tokens_saved": 12}),
    )

    out = c.on_transform_tool_result(
        tool_name="terminal",
        args={"command": "show synthetic diagnostics"},
        result=original,
        task_id="task-inline",
        tool_call_id="tc-inline",
    )

    assert out is not None
    assert "tokens dropped" not in out
    assert f"{cache_path} L2-4" in out
    assert "Read(offset=3,limit=3)" in out
    assert stored[c._cache_id(original)] == (original, "task-inline")


def test_hook_failopen_on_api_error(monkeypatch):
    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 5

    def _boom(**kw):
        raise RuntimeError("HTTP 500")

    monkeypatch.setattr(c._client, "compress", _boom)
    # Fail-open → original content, called_api False → hook returns None.
    out = c.on_transform_tool_result(
        tool_name="grep", args={"pattern": "x"}, result=ORIGINAL, tool_call_id="tc2"
    )
    assert out is None


def test_hook_failopen_on_cache_write_failure(monkeypatch):
    """If persisting the original fails, references would dangle — so the hook
    must fail open to the original output rather than emit them."""
    from plugins.tool_output_compresr import cache

    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 5
    monkeypatch.setattr(
        cache, "store_original", lambda cid, content, task_id="default", **_: None
    )
    monkeypatch.setattr(
        c._client,
        "compress",
        lambda **kw: ("line0 alpha\nline1 bravo\n[4 lines removed]", {"tokens_saved": 4}),
    )
    out = c.on_transform_tool_result(
        tool_name="grep", args={"pattern": "x"}, result=ORIGINAL, tool_call_id="tc4"
    )
    assert out is None


def test_hook_failopen_when_no_recovery_reference(monkeypatch):
    """If compression ever yields output carrying no recovery reference (degenerate
    / fully-paraphrased, no anchors, no marker), the hook must not hand back
    silently-lossy text — it fails open to the original."""
    from plugins.tool_output_compresr import cache

    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 5
    monkeypatch.setattr(
        cache,
        "store_original",
        lambda cid, content, task_id="default", **_: cache.relative_cache_path(cid),
    )
    monkeypatch.setattr(
        c._client,
        "compress",
        lambda **kw: ("a totally rewritten summary with no verbatim lines", {"tokens_saved": 1}),
    )
    out = c.on_transform_tool_result(
        tool_name="grep", args={"pattern": "x"}, result=ORIGINAL, tool_call_id="tc5"
    )
    assert out is None


def test_hook_failopen_on_ambiguous_recovery(monkeypatch):
    """If recovery references would be ambiguous, the transform hook leaves the
    original result in context instead of returning compressed text."""
    from plugins.tool_output_compresr import cache

    original = "\n".join(
        [
            "start",
            "repeat",
            "first hidden payload",
            "repeat",
            "second hidden payload",
            "end",
        ]
    )
    compressed = "\n".join(["start", "repeat", "[content dropped]", "end"])
    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 1
    monkeypatch.setattr(
        cache,
        "store_original",
        lambda cid, content, task_id="default", **_: cache.relative_cache_path(cid),
    )
    monkeypatch.setattr(
        c._client,
        "compress",
        lambda **kw: (compressed, {"tokens_saved": 10}),
    )

    assert c.on_transform_tool_result(
        tool_name="grep", args={"pattern": "repeat"}, result=original, tool_call_id="tc6"
    ) is None


def test_tool_output_api_key_is_env_only(monkeypatch, tmp_path):
    monkeypatch.delenv("COMPRESR_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "compresr:\n"
        "  api_key: cmp_from_config\n"
        "  tool_output_enabled: true\n"
        "  tool_output_min_tokens: 7\n"
        "  tool_output_max_cache_mb: 3\n",
        encoding="utf-8",
    )

    c = ToolOutputCompressor()
    assert c.api_key == ""
    assert c.enabled is True
    assert c.min_tokens == 7
    assert c.max_cache_mb == 3
    assert not c.active


def test_backend_cache_prune_uses_backend_exec(tmp_path):
    from plugins.tool_output_compresr.cache import _prune_cache_dir_via_backend

    old = tmp_path / "old"
    newer = tmp_path / "newer"
    current = tmp_path / "current"
    old.write_text("x" * 10, encoding="utf-8")
    newer.write_text("y" * 10, encoding="utf-8")
    current.write_text("z" * 10, encoding="utf-8")
    os.utime(old, (1, 1))
    os.utime(newer, (2, 2))
    os.utime(current, (3, 3))

    file_ops = _shell_file_ops_for_tmp_path(tmp_path)
    _prune_cache_dir_via_backend(file_ops, str(tmp_path), 20, str(current))

    assert not old.exists()
    assert newer.exists()
    assert current.exists()
    assert sum(path.stat().st_size for path in tmp_path.iterdir() if path.is_file()) <= 20


def test_backend_cache_prune_disabled_leaves_files(tmp_path):
    from plugins.tool_output_compresr.cache import _prune_cache_dir_via_backend

    old = tmp_path / "old"
    current = tmp_path / "current"
    old.write_text("x" * 10, encoding="utf-8")
    current.write_text("z" * 10, encoding="utf-8")

    file_ops = _shell_file_ops_for_tmp_path(tmp_path)
    _prune_cache_dir_via_backend(file_ops, str(tmp_path), 0, str(current))

    assert old.exists()
    assert current.exists()


def test_backend_cache_prune_without_exec_does_not_touch_local_path(tmp_path):
    from plugins.tool_output_compresr.cache import _prune_cache_dir_via_backend

    old = tmp_path / "old"
    current = tmp_path / "current"
    old.write_text("x" * 10, encoding="utf-8")
    current.write_text("z" * 10, encoding="utf-8")

    _prune_cache_dir_via_backend(object(), str(tmp_path), 1, str(current))

    assert old.exists()
    assert current.exists()


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
