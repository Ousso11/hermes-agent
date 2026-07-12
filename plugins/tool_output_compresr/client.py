"""Thin client for Compresr's tool-output compression endpoint.

``POST /compress/tool-output/`` with the tool output, the query (tool intent),
and the tool name. ``disable_placeholders`` stays FALSE so the API keeps its own
inline drop markers (``[N tokens removed]``), passed through verbatim alongside
the recovery pointer. stdlib-only (urllib) to keep the plugin import-light.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple

# The tool-output endpoint only accepts toc_* models (latte_v2 → HTTP 422).
DEFAULT_TOOL_OUTPUT_MODEL = "toc_latte_v2"

# Max chars of a server error body to surface in the raised RuntimeError.
_ERR_DETAIL_MAXLEN = 300


class CompresrToolOutputClient:
    DEFAULT_SOURCE = "gateway:hermes"
    USER_AGENT = "hermes-tool-output/0.1"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.compresr.ai/api",
        model: str = DEFAULT_TOOL_OUTPUT_MODEL,
        timeout: int = 30,
        source: str = DEFAULT_SOURCE,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.source = source

    def compress(
        self,
        tool_output: str,
        query: str,
        tool_name: str,
        target_ratio: float = 0.0,
        coarse: bool = True,
        disable_placeholders: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """Return ``(compressed_output, data)``. Raises on transport/HTTP/API error."""
        if not self.api_key:
            raise RuntimeError("COMPRESR_API_KEY not set")
        if not tool_output:
            raise RuntimeError("tool_output is required")

        payload: Dict[str, Any] = {
            "tool_output": tool_output,
            "query": query,
            "tool_name": tool_name or "unknown",
            "compression_model_name": self.model,
            "source": self.source,
            "disable_placeholders": disable_placeholders,
            "coarse": coarse,
        }
        if target_ratio and target_ratio > 0:
            payload["target_compression_ratio"] = target_ratio

        req = urllib.request.Request(
            f"{self.base_url}/compress/tool-output/",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
                "User-Agent": self.USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:_ERR_DETAIL_MAXLEN]
            except Exception:
                pass
            raise RuntimeError(f"HTTP {e.code}: {detail or e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"connection error: {e.reason}") from e

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"non-JSON response ({e}); body prefix: {raw[:200]!r}") from e
        if not parsed.get("success", False):
            raise RuntimeError(f"API error: {parsed.get('message') or parsed.get('error')}")
        d = parsed.get("data") or {}
        compressed = d.get("compressed_output", "")
        if not compressed:
            raise RuntimeError("API returned empty compressed_output")
        return compressed, d
