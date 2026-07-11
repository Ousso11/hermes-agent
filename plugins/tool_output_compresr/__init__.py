"""Compresr tool-output compression plugin for Hermes.

Compresses large tool outputs (verbose ``grep``/``read_file``/``execute_code``
dumps) *as they arrive*, on the ``transform_tool_result`` hook — before they
bloat the context window — using Compresr's query-specific tool-output API. The
API's compressed text is passed through UNCHANGED (its own ``[N tokens removed]``
markers included); we append a footer pointing the agent at the full verbatim
original in Hermes's managed cache, recoverable with a plain
``read_file``/``search_files``. Lossy summary + recoverable source; fail-open.

This is the per-turn complement to ``plugins/context_engine/compresr`` (which
compresses at compaction time). The two compose: this is a pre-filter that
shrinks each output once, so compaction has less residual to summarize.

Activation (opt-in, off by default):

    # ~/.hermes/config.yaml
    compresr:
      tool_output_enabled: true

    # ~/.hermes/.env
    COMPRESR_API_KEY=cmp_...

Config (env first for secrets, then a ``compresr:`` block in config.yaml for
non-secret settings):

    COMPRESR_API_KEY                    (required, .env only) cmp_... key
    COMPRESR_BASE_URL                   default https://api.compresr.ai/api
    COMPRESR_TOOL_OUTPUT_ENABLED        "1" to enable (or compresr.tool_output_enabled)
    COMPRESR_TOOL_OUTPUT_MODEL          default toc_latte_v2
    COMPRESR_TOOL_OUTPUT_MIN_TOKENS     default 1500 (skip smaller outputs)
    COMPRESR_TOOL_OUTPUT_TIMEOUT        default 30 (seconds)
    COMPRESR_TOOL_OUTPUT_MAX_CACHE_MB   default 256 (0 disables pruning)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from hermes_constants import get_hermes_home

from . import cache
from .client import CompresrToolOutputClient, DEFAULT_TOOL_OUTPUT_MODEL
from .compress import FOOTER_MARKER, compress_tool_output, count_tokens

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.compresr.ai/api"
_DEFAULT_MIN_TOKENS = 1500
_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_CACHE_MB = 256
_DEFAULT_TARGET_RATIO = 2.0
_FALLBACK_QUERY = (
    "Preserve the facts, paths, identifiers, errors, and results in this tool "
    "output that are needed to continue the task."
)
# Args we mine, in priority order, to reconstruct the tool's intent as a query.
_QUERY_ARG_KEYS = ("query", "pattern", "command", "q", "search", "regex", "url")
_PATH_ARG_KEYS = ("file_path", "path", "file", "filename", "directory")

# Tool → key holding the plain-text payload inside a JSON envelope. Compressing
# the raw envelope wastes ratio on JSON syntax and escaped "\n", and caches an
# unreadable escaped blob; unwrapping compresses (and caches) the real content,
# so a recovery read_file returns clean text.
_UNWRAPPABLE_JSON_TOOLS: Dict[str, str] = {
    "read_file": "content",
    "terminal": "output",
    "execute_code": "output",
    # SearchResult.to_dict(densify=True) puts the payload under "matches_text"
    # (never a top-level "content"), so that is the key to unwrap.
    "search_files": "matches_text",
}


def _try_unwrap_json_tool_result(
    tool_name: str, result: str
) -> Tuple[Optional[str], Optional[Callable[[str], str]]]:
    key = _UNWRAPPABLE_JSON_TOOLS.get(tool_name)
    if key is None:
        return None, None
    if not result.lstrip().startswith("{"):
        return None, None
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, ValueError):
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    inner = parsed.get(key)
    if not isinstance(inner, str) or not inner.strip():
        return None, None

    def splice(new_inner: str) -> str:
        out = dict(parsed)
        out[key] = new_inner
        return json.dumps(out, ensure_ascii=False)

    return inner, splice


def _read_config_block() -> Dict[str, Any]:
    """Best-effort read of the ``compresr:`` block from config.yaml. Never raises."""
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
        logger.debug("tool_output_compresr: could not read config block: %s", e)
        return {}


def _as_bool(v: Any) -> bool:
    return str(v).lower() in ("1", "true", "yes", "on")


class ToolOutputCompressor:
    """Holds config + clients and implements the transform hook."""

    def __init__(self) -> None:
        cfg = _read_config_block()

        def _opt(env_key: str, cfg_key: str, default: Any) -> Any:
            val = os.environ.get(env_key)
            if val is not None and val != "":
                return val
            if cfg_key in cfg and cfg[cfg_key] not in (None, ""):
                return cfg[cfg_key]
            return default

        self.api_key = os.environ.get("COMPRESR_API_KEY", "")
        self.base_url = str(
            _opt("COMPRESR_BASE_URL", "base_url", _DEFAULT_BASE_URL)
        ).rstrip("/")
        self.enabled = _as_bool(
            _opt("COMPRESR_TOOL_OUTPUT_ENABLED", "tool_output_enabled", "")
        )
        self.model = str(
            _opt("COMPRESR_TOOL_OUTPUT_MODEL", "tool_output_model", DEFAULT_TOOL_OUTPUT_MODEL)
        )
        self.min_tokens = int(
            _opt("COMPRESR_TOOL_OUTPUT_MIN_TOKENS", "tool_output_min_tokens", _DEFAULT_MIN_TOKENS)
        )
        self.timeout = int(
            _opt("COMPRESR_TOOL_OUTPUT_TIMEOUT", "tool_output_timeout", _DEFAULT_TIMEOUT)
        )
        self.max_cache_mb = int(
            _opt(
                "COMPRESR_TOOL_OUTPUT_MAX_CACHE_MB",
                "tool_output_max_cache_mb",
                _DEFAULT_MAX_CACHE_MB,
            )
        )
        self.target_ratio = float(
            _opt(
                "COMPRESR_TOOL_OUTPUT_TARGET_RATIO",
                "tool_output_target_ratio",
                _DEFAULT_TARGET_RATIO,
            )
        )

        self._client = CompresrToolOutputClient(
            api_key=self.api_key, base_url=self.base_url, model=self.model, timeout=self.timeout
        )
        # Cumulative stats for /usage and benchmarking.
        self.calls = 0
        self.errors = 0
        self.tokens_saved = 0
        self.recoveries = 0
        self._cooldown_until = 0.0

    # -- helpers -----------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.api_key)

    @staticmethod
    def _derive_query(tool_name: str, args: Any) -> str:
        """Reconstruct the tool's intent as a Compresr query.

        For search-class tools (grep/web_search) the args ARE an excellent query;
        for read/exec tools they are a weaker but still useful proxy.
        """
        if isinstance(args, dict):
            for k in _QUERY_ARG_KEYS:
                v = args.get(k)
                if isinstance(v, str) and v.strip():
                    return f"{tool_name}: {v.strip()}"[:600]
            for k in _PATH_ARG_KEYS:
                v = args.get(k)
                if isinstance(v, str) and v.strip():
                    return f"Relevant content of {v.strip()} for the current task"[:600]
        if tool_name:
            return f"Relevant output of the {tool_name} call for the current task"
        return _FALLBACK_QUERY

    @staticmethod
    def _is_recovery_read(args: Any) -> bool:
        """True if *args* points at a file under our compresr cache.

        A recovery ``read_file`` (or ``search_files``) targeting a cached
        original must NOT be re-compressed: the cached file is the verbatim
        source and is >= min_tokens but carries no FOOTER_MARKER, so without
        this guard the hook re-fires and returns a lossy summary instead of the
        exact original the agent asked to recover.

        Backend-agnostic: the cache path always contains the segment
        ``cache/compresr/tool-output`` — true for the host path AND the
        container-translated ``/root/.hermes/...`` path. We first try a resolved
        comparison against the real cache root, then fall back to a substring
        check on ``cache/compresr`` so remote/translated paths still match.
        """
        if not isinstance(args, dict):
            return False
        try:
            cache_root = str(cache.get_cache_root().resolve())
        except Exception:
            cache_root = ""
        for v in args.values():
            if not isinstance(v, str) or not v:
                continue
            if "cache/compresr" in v:
                return True
            if cache_root:
                try:
                    if str(Path(v).resolve()).startswith(cache_root):
                        return True
                except (OSError, ValueError):
                    pass
        return False

    @staticmethod
    def _cache_id(content: str) -> str:
        # Hash the CONTENT (not the tool_call_id) so two different outputs can
        # never collide onto one cache file; identical outputs safely dedupe.
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    # -- the hook ----------------------------------------------------------

    def on_transform_tool_result(
        self,
        tool_name: str = "",
        args: Any = None,
        result: Any = None,
        task_id: str = "",
        tool_call_id: str = "",
        status: str = "",
        **_: Any,
    ) -> Optional[str]:
        """Return a compressed replacement string, or None to leave unchanged.

        Fail-open: any error, cooldown, small output, or error-status result
        returns None so the model sees the original.
        """
        if not self.active or not isinstance(result, str):
            return None
        if status and status not in ("", "ok", "success"):
            return None  # don't mangle error results
        if FOOTER_MARKER in result:
            return None  # already compressed by us — idempotent
        if self._is_recovery_read(args):
            # The agent is recovering a cached original via read_file/search_files;
            # return it verbatim rather than re-compressing it into a lossy summary.
            self.recoveries += 1
            return None
        if count_tokens(result) < self.min_tokens:
            return None  # too small to be worth a round-trip
        now = time.monotonic()
        if now < self._cooldown_until:
            return None

        query = self._derive_query(tool_name, args)

        inner_text, splice = _try_unwrap_json_tool_result(tool_name, result)
        compress_target = inner_text if inner_text is not None else result
        if inner_text is not None and count_tokens(inner_text) < self.min_tokens:
            return None

        cache_id = self._cache_id(compress_target)
        try:
            out, info = compress_tool_output(
                query=query,
                content=compress_target,
                tool_name=tool_name,
                cache_id=cache_id,
                client=self._client,
                task_id=task_id,
                max_cache_mb=self.max_cache_mb,
                target_ratio=self.target_ratio,
            )
        except Exception as e:  # compress is already fail-open, but be defensive
            self.errors += 1
            self._cooldown_until = now + 30.0
            logger.warning("tool_output_compresr: hook error (%s)", e)
            return None

        if info.get("error"):
            self.errors += 1
            self._cooldown_until = now + 30.0
        if not info.get("called_api"):
            return None  # API failed → leave the original output unchanged
        if not info.get("shortened"):
            # The API didn't produce a meaningfully shorter output (or the cache
            # write / backend-visibility check failed) — leave the original
            # unchanged rather than spend context on a footer for no gain.
            return None
        saved = max(0, info.get("base_tokens", 0) - info.get("out_tokens", 0))
        self.calls += 1
        self.tokens_saved += saved
        logger.info(
            "tool_output_compresr: %s %d→%d tokens (saved %d, unwrapped=%s, cache=%s)",
            tool_name,
            info.get("base_tokens", 0),
            info.get("out_tokens", 0),
            saved,
            splice is not None,
            info.get("cache_path", "?"),
        )
        if splice is not None:
            # Fold the footer into the JSON envelope's inner text so the result
            # stays valid JSON for the tool that produced it.
            return splice(out)
        return out

    def get_status(self) -> Dict[str, Any]:
        return {
            "plugin": "tool_output_compresr",
            "active": self.active,
            "model": self.model,
            "min_tokens": self.min_tokens,
            "max_cache_mb": self.max_cache_mb,
            "calls": self.calls,
            "errors": self.errors,
            "tokens_saved": self.tokens_saved,
            "recoveries": self.recoveries,
        }


def register(ctx: Any) -> None:
    """Plugin entry point — called by the Hermes plugin loader."""
    try:
        cache.ensure_cache_root()
    except Exception as e:
        logger.warning("tool_output_compresr: could not initialize cache root: %s", e)
    compressor = ToolOutputCompressor()
    ctx.register_hook("transform_tool_result", compressor.on_transform_tool_result)
    if not compressor.api_key:
        logger.info(
            "tool_output_compresr: loaded but COMPRESR_API_KEY is unset — inactive."
        )
    elif not compressor.enabled:
        logger.info(
            "tool_output_compresr: loaded but disabled — set compresr.tool_output_enabled: true."
        )
