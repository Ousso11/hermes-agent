"""Compresr (YC W26) context engine for Hermes.

A drop-in replacement for the built-in ``compressor`` context engine. Instead
of asking an auxiliary LLM to write an abstractive summary of the middle of the
conversation, this engine sends those turns to Compresr's query-specific
compression API (``latte_v1`` / ``latte_v2``), which scores spans against the
current goal and keeps only the answer-bearing ones.

Design: we subclass :class:`ContextCompressor` and override *only* the
"compress the middle window" step (``_generate_summary``) plus ``update_model``
(to recompute token budgets after the real model is known). All of the
surrounding machinery — old-tool-result pruning, head/tail protection, boundary
alignment, tool_call/tool_result pair sanitization, historical-media stripping,
and anti-thrashing accounting — is inherited verbatim. That makes an A/B against
the built-in engine a clean comparison of the *core compaction strategy* only.

Activate via ``~/.hermes/config.yaml``::

    context:
      engine: compresr

Configuration is read from environment first for secrets, then from an optional
``compresr:`` block in config.yaml for non-secret settings:

    COMPRESR_API_KEY        (required, .env only)  cmp_... key
    COMPRESR_BASE_URL       default https://api.compresr.ai/api
    COMPRESR_MODEL          default latte_v2  (latte_v1 | latte_v2)
    COMPRESR_TARGET_RATIO   optional Compresr ratio override (see semantics below)
    COMPRESR_TIMEOUT        default 60 (seconds)
    COMPRESR_COARSE         latte_v1 only: "1"/"0" paragraph vs token-level
    COMPRESR_DISABLE_PLACEHOLDERS  "1" to drop the "[N tokens dropped]" markers

target_compression_ratio semantics (Compresr canonical): a value in (0, 1] is a
*removal* fraction (0.8 removes ~80%); a value > 1 is an Nx target (5 -> ~1/5).
Hermes's ``compression.target_ratio`` is a *keep* fraction (0.2 keeps ~20%), so
when no explicit COMPRESR_TARGET_RATIO is set we map keep-fraction f to the Nx
factor 1/f (0.2 keep -> 5x).
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.context_compressor import ContextCompressor
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.compresr.ai/api"
_DEFAULT_MODEL = "latte_v2"
_DEFAULT_TIMEOUT = 60
# Generic query used when no recent user focus can be derived. Compresr requires
# a non-empty query, so we always send something meaningful.
_FALLBACK_QUERY = (
    "Preserve the key facts, decisions, file paths, commands, results, and open "
    "tasks needed to continue this work."
)


def _as_int(v: Any) -> int:
    """Tolerant int coercion for server-supplied stats — never crash on 'N/A'."""
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def _read_config_block() -> Dict[str, Any]:
    """Best-effort read of a ``compresr:`` block from config.yaml.

    Environment variables take precedence; this only fills gaps. Never raises.
    """
    cfg_path = get_hermes_home() / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml

        with open(cfg_path, encoding="utf-8-sig") as f:
            data = yaml.safe_load(f) or {}
        block = data.get("compresr", {})
        return block if isinstance(block, dict) else {}
    except Exception as e:  # pragma: no cover - config is optional
        logger.debug("compresr: could not read config block: %s", e)
        return {}


class CompresrContextEngine(ContextCompressor):
    """Context engine that compacts via Compresr's query-specific API."""

    def __init__(self, **kwargs: Any) -> None:
        cfg = _read_config_block()

        def _opt(env_key: str, cfg_key: str, default: Any) -> Any:
            val = os.environ.get(env_key)
            if val is not None and val != "":
                return val
            if cfg_key in cfg and cfg[cfg_key] not in (None, ""):
                return cfg[cfg_key]
            return default

        self.compresr_api_key = os.environ.get("COMPRESR_API_KEY", "")
        self.compresr_base_url = str(
            _opt("COMPRESR_BASE_URL", "base_url", _DEFAULT_BASE_URL)
        ).rstrip("/")
        self.compresr_model = str(_opt("COMPRESR_MODEL", "model", _DEFAULT_MODEL))
        self.compresr_timeout = int(_opt("COMPRESR_TIMEOUT", "timeout", _DEFAULT_TIMEOUT))
        self.compresr_coarse = str(_opt("COMPRESR_COARSE", "coarse", "")).lower() in (
            "1", "true", "yes",
        )
        self.compresr_disable_placeholders = str(
            _opt("COMPRESR_DISABLE_PLACEHOLDERS", "disable_placeholders", "")
        ).lower() in ("1", "true", "yes")
        _ratio_override = _opt("COMPRESR_TARGET_RATIO", "target_ratio", "")
        self.compresr_ratio_override: Optional[float] = (
            float(_ratio_override) if str(_ratio_override).strip() else None
        )

        # Cumulative API stats for get_status() / benchmarking.
        self.compresr_calls = 0
        self.compresr_errors = 0
        self.compresr_tokens_saved = 0
        self.compresr_last_duration_ms = 0

        # Construct the parent with a placeholder model + explicit context length
        # so __init__ never makes a network/model-metadata lookup. agent_init
        # always calls update_model() right after, which sets the real values.
        kwargs.setdefault("model", "compresr-placeholder")
        kwargs.setdefault("config_context_length", 200_000)
        # Compresr is an external service. If it is unavailable, preserve the
        # current transcript instead of inserting a deterministic placeholder
        # handoff and dropping the middle window.
        kwargs.setdefault("abort_on_summary_failure", True)
        super().__init__(**kwargs)

        if not self.compresr_api_key:
            logger.warning(
                "compresr: COMPRESR_API_KEY is not set — compaction will fail and "
                "compression will be aborted without dropping context. Set it in "
                "~/.hermes/.env."
            )

    # -- Identity ----------------------------------------------------------

    @property
    def name(self) -> str:
        return "compresr"

    def is_available(self) -> bool:
        """Used by discover_context_engines() for the availability check."""
        return bool(self.compresr_api_key)

    # -- Token budgets must be recomputed when the real model arrives ------

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: Any = "",
        provider: str = "",
        api_mode: str = "",
        max_tokens: int | None = None,
    ) -> None:
        # The parent's update_model already computes threshold_tokens (with the
        # small-context carve-out that keeps the trigger below the window for
        # ctx <= 64K) plus tail_token_budget and max_summary_tokens. Re-flooring
        # to MINIMUM_CONTEXT_LENGTH here pushed the threshold to >= the window
        # for small contexts, so compaction could never fire (regressing #14690).
        # There is nothing left for the child to recompute — we forward every
        # parameter (incl. max_tokens) verbatim so the override stays a pure
        # signature-compatible pass-through as the parent evolves.
        super().update_model(
            model, context_length, base_url, api_key, provider, api_mode, max_tokens
        )

    # -- Ratio mapping -----------------------------------------------------

    def _target_compression_ratio(self) -> float:
        """Resolve the Compresr ``target_compression_ratio`` to send.

        Explicit override wins. Otherwise map Hermes's keep-fraction
        ``summary_target_ratio`` (e.g. 0.2 = keep ~20%) to a Compresr Nx factor
        (1/0.2 = 5x). Clamp to Compresr's documented [.., 200] range.
        """
        if self.compresr_ratio_override is not None:
            return self.compresr_ratio_override
        keep = self.summary_target_ratio or 0.2
        keep = min(max(keep, 0.01), 0.95)
        return round(min(1.0 / keep, 200.0), 3)

    # -- The one override that matters: compress the middle via Compresr ---

    def _generate_summary(
        self,
        turns_to_summarize: List[Dict[str, Any]],
        focus_topic: Optional[str] = None,
    ) -> Optional[str]:
        """Compress the middle window with Compresr instead of an LLM summary.

        Returns the compressed text (with the standard summary prefix so
        iterative re-compaction recognizes it), or None on failure — which lets
        the inherited ``compress()`` fall back to its deterministic handoff.
        """
        now = time.monotonic()
        if now < self._summary_failure_cooldown_until:
            logger.debug("compresr: in failure cooldown, skipping")
            return None

        context = self._serialize_for_summary(turns_to_summarize)
        if not context.strip():
            return None

        query = (focus_topic or "").strip() or _FALLBACK_QUERY

        # Preserve continuity: fold the prior handoff summary into the context so
        # Compresr can carry forward still-relevant facts across compactions.
        if self._previous_summary:
            context = (
                "[PRIOR CONTEXT SUMMARY]\n"
                + self._previous_summary
                + "\n\n[NEW CONVERSATION TURNS]\n"
                + context
            )

        try:
            compressed, stats = self._call_compresr(context, query)
        except Exception as e:
            self.compresr_errors += 1
            self._last_summary_error = f"compresr: {e}"
            # Mirror the base's failure behavior: brief cooldown to avoid
            # hammering a failing endpoint on every turn.
            self._summary_failure_cooldown_until = time.monotonic() + 30.0
            logger.warning("compresr: compression failed (%s) — falling back", e)
            return None

        if not compressed or not compressed.strip():
            self.compresr_errors += 1
            self._last_summary_error = "compresr: empty compressed_context"
            # Back off like the exception path, so a persistently-empty endpoint
            # isn't re-hit on every compaction.
            self._summary_failure_cooldown_until = time.monotonic() + 30.0
            return None

        self.compresr_calls += 1
        self.compresr_tokens_saved += _as_int(stats.get("tokens_saved"))
        self.compresr_last_duration_ms = _as_int(stats.get("duration_ms"))
        logger.info(
            "compresr: %s %s tokens -> %s tokens (saved %s, %sms server)",
            self.compresr_model,
            stats.get("original_tokens", "?"),
            stats.get("compressed_tokens", "?"),
            stats.get("tokens_saved", "?"),
            stats.get("duration_ms", "?"),
        )

        body = self._strip_summary_prefix(compressed)
        self._previous_summary = body
        return self._with_summary_prefix(body)

    # -- Compresr API call -------------------------------------------------

    def _call_compresr(self, context: str, query: str) -> tuple[str, Dict[str, Any]]:
        """POST to /compress/question-specific/ and return (text, stats).

        Raises on transport/HTTP/parse errors so the caller can fall back.
        """
        if not self.compresr_api_key:
            raise RuntimeError("COMPRESR_API_KEY not set")

        payload: Dict[str, Any] = {
            "context": context,
            "query": query,
            "compression_model_name": self.compresr_model,
            "target_compression_ratio": self._target_compression_ratio(),
        }
        if self.compresr_model == "latte_v1":
            payload["coarse"] = self.compresr_coarse
        if self.compresr_disable_placeholders:
            payload["disable_placeholders"] = True

        url = f"{self.compresr_base_url}/compress/question-specific/"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-API-Key": self.compresr_api_key,
                "User-Agent": "hermes-compresr-engine/0.1",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.compresr_timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            raise RuntimeError(f"HTTP {e.code}: {detail or e.reason}") from e

        parsed = json.loads(raw)
        if not parsed.get("success", False):
            raise RuntimeError(
                f"API error: {parsed.get('message') or parsed.get('error')}"
            )
        d = parsed.get("data") or {}
        compressed = d.get("compressed_context", "")
        return compressed, d

    # -- Status (extends base with Compresr counters) ---------------------

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update(
            {
                "engine": "compresr",
                "compresr_model": self.compresr_model,
                "compresr_calls": self.compresr_calls,
                "compresr_errors": self.compresr_errors,
                "compresr_tokens_saved": self.compresr_tokens_saved,
                "compresr_last_duration_ms": self.compresr_last_duration_ms,
            }
        )
        return status


def register(ctx: Any) -> None:
    """Plugin entry point — called by the context-engine loader."""
    ctx.register_context_engine(CompresrContextEngine())
