"""Contract test for the context_engine/compresr compaction engine.

The engine couples to the parent ``ContextCompressor``'s PRIVATE seam
(``_generate_summary`` + summary-prefix helpers). This test pins that contract
so an upstream refactor fails loudly in CI instead of silently degrading the
plugin to a no-op:

  * the override's signature matches the parent's,
  * a successful API call returns the prefixed compressed body,
  * an API failure returns None (so inherited compress() falls back) and trips
    the error counter + cooldown.
"""

import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.context_compressor import ContextCompressor  # noqa: E402
from plugins.context_engine.compresr import CompresrContextEngine  # noqa: E402

TURNS = [
    {"role": "user", "content": "Refactor the auth module and fix the login bug."},
    {"role": "assistant", "content": "Edited auth.py; tests pass. Token TTL set to 3600."},
]


def _engine():
    e = CompresrContextEngine()
    e.compresr_api_key = "cmp_test"  # avoid the unset-key warning path
    return e


def test_override_signature_matches_parent():
    parent = inspect.signature(ContextCompressor._generate_summary).parameters
    child = inspect.signature(CompresrContextEngine._generate_summary).parameters
    assert list(parent) == list(child), (
        "parent _generate_summary signature changed — update the override"
    )


def test_success_returns_prefixed_body():
    e = _engine()
    e._call_compresr = lambda context, query: ("KEPT: login bug, TTL 3600", {"tokens_saved": 42})
    out = e._generate_summary(TURNS, focus_topic="login bug")
    assert out is not None
    # Carries the standard summary prefix so iterative re-compaction recognizes it.
    assert out == e._with_summary_prefix(e._strip_summary_prefix(out))
    assert "login bug" in out
    assert e.compresr_calls == 1
    assert e._previous_summary  # continuity state updated


def test_failure_falls_back_to_none():
    e = _engine()

    def _boom(context, query):
        raise RuntimeError("HTTP 500")

    e._call_compresr = _boom
    out = e._generate_summary(TURNS, focus_topic="anything")
    assert out is None  # → inherited compress() uses its deterministic handoff
    assert e.compresr_errors == 1
    assert e._summary_failure_cooldown_until > 0  # cooldown tripped


def test_empty_compression_is_treated_as_failure():
    e = _engine()
    e._call_compresr = lambda context, query: ("", {})
    assert e._generate_summary(TURNS, focus_topic="x") is None
    assert e.compresr_errors == 1


def test_ratio_mapping_keep_to_nx():
    e = _engine()
    e.summary_target_ratio = 0.2  # keep ~20%
    assert e._target_compression_ratio() == 5.0  # → 5x
    e.compresr_ratio_override = 0.8
    assert e._target_compression_ratio() == 0.8  # explicit override wins


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
