"""Regression tests for three fixes to the Compresr integration.

Each test pins a specific bug so it cannot silently reappear:

* **F1** — the exact post-footer size gate used to ``unlink`` the content-addressed
  cache file when it failed. Because the cache key is a hash of the *content*, a
  concurrent (or prior) compression of identical content could already have handed
  that path to the model as a recovery pointer; deleting it dangled the pointer and
  broke the recover-verbatim guarantee. The gate must now fail open WITHOUT deleting.
* **F2** — ``_is_recovery_read`` matched a bare ``"cache/compresr"`` substring in any
  arg, so an ordinary grep/read that merely mentioned it (e.g. working on this repo)
  was misread as a recovery and skipped compression. It must now match the full
  ``cache/compresr/tool-output`` segment, while genuine recovery paths still skip.
* **F3** — plugin ``__init__`` coerced numeric config/env with bare ``int()``/
  ``float()``; a typo like ``COMPRESR_TIMEOUT=abc`` raised, which the loader swallowed
  into a silent whole-feature disable. Bad values must now fall back to defaults.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

import plugins.tool_output_compresr as toc  # noqa: E402
from plugins.tool_output_compresr import ToolOutputCompressor, cache, compress  # noqa: E402
from plugins.tool_output_compresr.compress import compress_tool_output  # noqa: E402
from plugins.context_engine.compresr import CompresrContextEngine  # noqa: E402

# ~240 tokens — comfortably above the min_tokens gates used below.
ORIGINAL = "\n".join(f"line{i} payload" for i in range(400))
COMPRESSED = "line0 payload\n[many lines removed]"

# An agent-visible path long enough that the real footer (which embeds the path
# twice) costs more than the nominal FOOTER_TOKEN_BUDGET, so the exact gate can
# actually fire. store_original returns this; the host file that must survive is
# whatever cache_file_path resolves to — in reality these differ (agent-visible
# vs host path), which is exactly the long-remote-home case F1 rode on.
_LONG_VISIBLE_PATH = "/root/.hermes/cache/compresr/tool-output/" + ("a" * 560)


class _FakeClient:
    def __init__(self, out=COMPRESSED):
        self.out = out

    def compress(self, **kw):
        return self.out, {}


# --------------------------------------------------------------------------- #
# F1 — the gate-fail path must not delete a content-addressed cache file
# --------------------------------------------------------------------------- #
def test_f1_gate_fail_does_not_delete_cache_file(tmp_path, monkeypatch):
    real_file = tmp_path / "host-cache-file"
    real_file.write_text("VERBATIM ORIGINAL")
    monkeypatch.setattr(cache, "store_original", lambda *a, **k: _LONG_VISIBLE_PATH)
    monkeypatch.setattr(cache, "cache_file_path", lambda cid: real_file)

    base = "x" * 4000  # ~1000 tokens
    body = "x" * 3400  # passes the cheap pre-filter, fails the exact gate via the long path
    out, info = compress_tool_output(
        query="q", content=base, tool_name="grep", cache_id="c",
        client=_FakeClient(out=body), task_id="t",
    )
    assert out == base                                   # failed open
    assert info["error"] == "not smaller after footer"
    assert real_file.exists()                            # F1: NOT deleted
    assert real_file.read_text() == "VERBATIM ORIGINAL"  # verbatim original intact


def test_f1_second_call_does_not_dangle_first_calls_pointer(tmp_path, monkeypatch):
    """The original failure: two compressions of identical content share one
    content-addressed cache file. Call A wins and returns a footer pointing at it;
    call B over the same content fails the exact gate. B must not delete A's file."""
    real_file = tmp_path / "host-cache-file"

    def _store(cid, content, task_id="default", **_):
        real_file.write_text(content)   # host write, as the real store_original does
        return _LONG_VISIBLE_PATH        # agent-visible path handed to the model

    monkeypatch.setattr(cache, "store_original", _store)
    monkeypatch.setattr(cache, "cache_file_path", lambda cid: real_file)

    base = "x" * 4000
    # A: small body → net win → returns a footer that names the shared cache file.
    out_a, info_a = compress_tool_output(
        query="q", content=base, tool_name="grep", cache_id="shared",
        client=_FakeClient(out="y" * 2000), task_id="t",
    )
    assert info_a["shortened"] is True
    assert info_a["cache_path"] in out_a
    assert real_file.exists()

    # B: larger body over identical content → fails the exact gate.
    out_b, info_b = compress_tool_output(
        query="q", content=base, tool_name="grep", cache_id="shared",
        client=_FakeClient(out="z" * 3400), task_id="t",
    )
    assert out_b == base
    assert info_b["error"] == "not smaller after footer"

    # F1: A's recovery pointer still resolves to the verbatim original.
    assert real_file.exists()
    assert real_file.read_text() == base


# --------------------------------------------------------------------------- #
# F2 — recovery guard matches the full segment, not a bare substring
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("args", [
    {"pattern": "cache/compresr"},                       # grepping for the string
    {"query": "find TODOs under cache/compresr"},        # query mentioning it
    {"file_path": "/work/src/cache/compresr_notes.md"},  # a file whose path contains it
])
def test_f2_ordinary_call_mentioning_substring_is_still_compressed(tmp_path, monkeypatch, args):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 5
    monkeypatch.setattr(c._client, "compress", lambda **kw: (COMPRESSED, {}))
    monkeypatch.setattr(
        cache, "store_original",
        lambda cid, content, task_id="default", **_: f"{tmp_path}/cache/compresr/tool-output/{cid}",
    )
    before = c.recoveries
    out = c.on_transform_tool_result(tool_name="search_files", args=args, result=ORIGINAL)
    assert out is not None and compress.FOOTER_MARKER in out  # compressed, not skipped
    assert c.recoveries == before                            # recoveries stat not inflated


def test_f2_genuine_recovery_path_still_skips(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    c = ToolOutputCompressor()
    c.enabled, c.api_key, c.min_tokens = True, "cmp_test", 5
    called = {"n": 0}

    def _spy(**kw):
        called["n"] += 1
        return COMPRESSED, {}

    monkeypatch.setattr(c._client, "compress", _spy)
    before = c.recoveries
    # A container-translated recovery path carries the full cache/compresr/tool-output segment.
    out = c.on_transform_tool_result(
        tool_name="read_file",
        args={"file_path": "/root/.hermes/cache/compresr/tool-output/deadbeef"},
        result=ORIGINAL,
    )
    assert out is None                    # verbatim passthrough
    assert c.recoveries == before + 1     # correctly counted as a recovery
    assert called["n"] == 0               # never hit the API


# --------------------------------------------------------------------------- #
# F3 — malformed numeric config/env falls back to defaults instead of raising
# --------------------------------------------------------------------------- #
_TOOL_OUTPUT_DEFAULTS = {
    "COMPRESR_TOOL_OUTPUT_MIN_TOKENS": ("min_tokens", toc._DEFAULT_MIN_TOKENS),
    "COMPRESR_TOOL_OUTPUT_TIMEOUT": ("timeout", toc._DEFAULT_TIMEOUT),
    "COMPRESR_TOOL_OUTPUT_MAX_CACHE_MB": ("max_cache_mb", toc._DEFAULT_MAX_CACHE_MB),
    "COMPRESR_TOOL_OUTPUT_TARGET_RATIO": ("target_ratio", toc._DEFAULT_TARGET_RATIO),
}


@pytest.mark.parametrize("var", list(_TOOL_OUTPUT_DEFAULTS))
def test_f3_bad_tool_output_numeric_env_falls_back(monkeypatch, var):
    attr, default = _TOOL_OUTPUT_DEFAULTS[var]
    monkeypatch.setenv(var, "not-a-number")
    c = ToolOutputCompressor()  # must not raise
    assert getattr(c, attr) == default


def test_f3_bad_context_engine_timeout_falls_back(monkeypatch):
    monkeypatch.setenv("COMPRESR_TIMEOUT", "abc")
    eng = CompresrContextEngine()  # must not raise
    assert eng.compresr_timeout == 60


def test_f3_bad_context_engine_ratio_override_becomes_none(monkeypatch):
    monkeypatch.setenv("COMPRESR_TARGET_RATIO", "abc")
    eng = CompresrContextEngine()  # must not raise
    assert eng.compresr_ratio_override is None


def test_f3_register_still_succeeds_with_bad_env(monkeypatch):
    """The end goal of F3: a typo'd tunable must not silently disable the feature.
    Both plugins must still register their hook / engine."""
    monkeypatch.setenv("COMPRESR_TIMEOUT", "abc")
    monkeypatch.setenv("COMPRESR_TOOL_OUTPUT_TIMEOUT", "abc")
    hooks, engines = [], []
    ctx = type("Ctx", (), {
        "register_hook": lambda self, name, cb: hooks.append(name),
        "register_context_engine": lambda self, e: engines.append(e),
    })()

    import plugins.context_engine.compresr as ce

    toc.register(ctx)                                 # must not raise
    ce.register(ctx)                                  # must not raise

    assert "transform_tool_result" in hooks
    assert len(engines) == 1
