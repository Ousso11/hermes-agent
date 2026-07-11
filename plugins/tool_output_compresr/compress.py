"""Tool-output compression for Hermes — compress, cache, point back.

Above the size threshold (enforced by the caller via ``min_tokens``) we call the
Compresr API, persist the verbatim original to Hermes's managed cache, and return
the API's compressed text UNCHANGED with a short footer that points the agent at
the full original (recoverable with ``read_file``/``search_files``). Lossy summary
+ recoverable source — the API's own inline markers (``[N tokens removed]`` …) are
left as-is; we do NOT rewrite them into line-level references.

Fail-open: any API error, or an output that isn't meaningfully shorter than the
original, returns the original content unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from . import cache
from .client import CompresrToolOutputClient

logger = logging.getLogger(__name__)

# Sentinel that marks our own footer, so the hook never re-compresses an output
# it already processed.
FOOTER_MARKER = "[compresr:recover]"

# The recovery footer itself costs tokens (~83 for a typical cache path). We gate
# on the compressed body PLUS this budget so a marginal compression can never come
# back net-larger than the original once the footer is appended.
FOOTER_TOKEN_BUDGET = 90


def count_tokens(s: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token) for gating."""
    return (len(s) + 3) // 4


def _footer(path: str, base_tok: int, out_tok: int) -> str:
    saved = max(0, base_tok - out_tok)
    pct = int(round(100 * saved / base_tok)) if base_tok else 0
    return (
        f"\n\n{FOOTER_MARKER} Tool output compressed {base_tok}→{out_tok} tokens "
        f"(~{pct}% saved). The full verbatim original is cached at {path} — "
        f"if you need exact details that were summarized away, recover them with "
        f'read_file("{path}") or search_files.'
    )


def compress_tool_output(
    query: str,
    content: str,
    tool_name: str,
    cache_id: str,
    client: CompresrToolOutputClient,
    task_id: str = "default",
    max_cache_mb: int = 256,
    target_ratio: float = 2.0,
    cache_content: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Compress via Compresr, store the original, append a recovery footer.

    Returns ``(output_text, info)``. Never raises: an API failure (or output that
    isn't meaningfully shorter) falls back to the original content with
    ``shortened`` False so the caller leaves the tool output unchanged.

    ``cache_content`` is the text persisted to the recovery cache; it defaults to
    ``content`` but may differ (e.g. a de-numbered copy of already-line-numbered
    read_file output, so a later ``read_file`` on the cache re-adds exactly one
    gutter instead of showing a doubled ``M|N|`` prefix). The API call and the
    size gate always use ``content``.
    """
    if cache_content is None:
        cache_content = content
    base_tok = count_tokens(content)
    info: Dict[str, Any] = {
        "called_api": False,
        "base_tokens": base_tok,
        "out_tokens": base_tok,
        "shortened": False,
    }

    try:
        compressed, stats = client.compress(
            tool_output=content, query=query, tool_name=tool_name,
            coarse=True, target_ratio=target_ratio,
        )
    except Exception as e:  # fail-open to the original tool output
        logger.warning("tool_output_compresr: API failed (%s) — leaving original", e)
        info["error"] = str(e)
        return content, info

    info["called_api"] = True
    info["api_stats"] = stats

    # A whitespace-only (or empty) compressed body is truthy at the client layer
    # but would silently replace the tool output with near-nothing + a footer.
    # Treat it as a failure and fail open to the original.
    if not compressed or not compressed.strip():
        info["error"] = "empty compressed output"
        return content, info

    # Only cache + point back if the API actually shortened the output ONCE the
    # recovery footer is accounted for; otherwise the footer would push the
    # returned value net-larger than the original for no gain. Gate BEFORE caching.
    body_tok = count_tokens(compressed)
    if body_tok + FOOTER_TOKEN_BUDGET >= base_tok:
        return content, info

    # Persist the exact original so the pointer resolves. store_original returns an
    # agent-visible path (or None if the active backend can't prove one) — on None
    # we fail open rather than point at a file the agent cannot read.
    cache_path = cache.store_original(cache_id, cache_content, task_id, max_cache_mb=max_cache_mb)
    if cache_path is None:
        info["error"] = "cache write failed"
        return content, info

    # Footer text reports the honest post-footer size (body + footer budget), so
    # it never understates the returned output's real token cost.
    reported_out_tok = body_tok + FOOTER_TOKEN_BUDGET
    out = compressed + _footer(cache_path, base_tok, reported_out_tok)
    out_tok = count_tokens(out)
    info.update(
        {
            "shortened": True,
            "out_tokens": out_tok,
            "saved": max(0, base_tok - out_tok),
            "cache_path": cache_path,
        }
    )
    return out, info
