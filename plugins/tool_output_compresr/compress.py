"""Tool-output compression for Hermes — one lean path.

Above the size threshold (enforced by the caller via ``min_tokens``) we ALWAYS
call the Compresr API, persist the original tool output to the workspace cache,
and rewrite the API's inline placeholders into addressable ``.compresr/cache``
references the agent can ``Read``/``Grep`` back. Lossless by recovery.

Fail-open: any API error returns the original content unchanged.

(Earlier versions ran a multi-tier "cascade" — deterministic cleanup + an
economics gate that sometimes skipped the API. That was dropped for simplicity;
the threshold is the only gate. Re-introduce tiers here later if needed.)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from . import cache
from .client import CompresrToolOutputClient
from .recover import rewrite_placeholders

logger = logging.getLogger(__name__)


def count_tokens(s: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token) for gating."""
    return (len(s) + 3) // 4


def compress_tool_output(
    query: str,
    content: str,
    tool_name: str,
    cache_id: str,
    client: CompresrToolOutputClient,
    task_id: str = "default",
    max_cache_mb: int = 256,
) -> Tuple[str, Dict[str, Any]]:
    """Compress via Compresr, store the original, rewrite placeholders to refs.

    Returns ``(output_text, info)``. Never raises: an API failure falls back to
    the original content with ``called_api`` False so the caller leaves the tool
    output unchanged.
    """
    base_tok = count_tokens(content)
    info: Dict[str, Any] = {"called_api": False, "base_tokens": base_tok}

    try:
        compressed, stats = client.compress(
            tool_output=content, query=query, tool_name=tool_name, coarse=True
        )
    except Exception as e:  # fail-open to the original tool output
        logger.warning("tool_output_compresr: API failed (%s) — leaving original", e)
        info["error"] = str(e)
        info["out_tokens"] = base_tok
        return content, info

    # Persist the EXACT text the API compressed so anchored references resolve,
    # and reference it by ABSOLUTE path so recovery survives a later `cd`. If the
    # write failed we get None back and fail open to the original tool output
    # rather than emit references to a file that was never written.
    cache_path = cache.store_original(cache_id, content, task_id, max_cache_mb=max_cache_mb)
    if cache_path is None:
        info["error"] = "cache write failed"
        info["out_tokens"] = base_tok
        return content, info

    rewritten, gaps, ok = rewrite_placeholders(cache_path, content, compressed)

    out_tok = count_tokens(rewritten)

    if gaps:
        header = (
            f"[compresr: Output compressed. Original saved to {cache_path}]\n"
            f"[Tokens: {base_tok} cached, {out_tok} non-cached]\n"
            f"[Use read_file(offset, limit) or search_files to recover omitted lines]\n\n"
        )
        rewritten = header + rewritten
        out_tok = count_tokens(rewritten)

    info.update(
        {
            "called_api": True,
            "anchored_ok": ok,
            "gaps": len(gaps),
            "out_tokens": out_tok,
            "api_stats": stats,
        }
    )
    return rewritten, info
