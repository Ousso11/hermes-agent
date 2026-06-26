"""Persist original tool outputs so recovery references resolve.

The addressable references emitted by :mod:`recover` point at
``.compresr/cache/<id>``. We persist each original there through Hermes's own
file-operations layer (``_get_file_ops(task_id).write_file``) — the SAME backend
the agent's ``read_file``/``write_file`` tools use — rather than touching the
local filesystem directly.

That matters for correctness, not just tidiness: when the agent runs in a
non-local environment (docker/modal sandbox, remote terminal), the cache write
lands in that same environment and resolves through the same per-task cwd, so the
reference the model later ``read_file``s is guaranteed to point at the file we
wrote. A direct local-disk write would silently miss the sandbox.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Set

from .recover import CACHE_DIR_NAME

logger = logging.getLogger(__name__)

# ``.compresr`` — the parent of the cache dir; carries a self-ignoring .gitignore.
_CACHE_ROOT = CACHE_DIR_NAME.split("/")[0]

# Tasks for which we've already dropped the ignore file (write it at most once).
_gitignore_done: Set[str] = set()


def relative_cache_path(cache_id: str) -> str:
    """Workspace-relative path the recovery reference points at."""
    return f"{CACHE_DIR_NAME}/{cache_id}"


def _prune_cache_dir_via_backend(
    file_ops: object,
    cache_dir: str,
    max_bytes: int,
    keep_path: str,
) -> None:
    """Prune in the same backend that owns the cached files."""
    if max_bytes <= 0:
        return
    exec_fn = getattr(file_ops, "_exec", None)
    escape = getattr(file_ops, "_escape_shell_arg", None)
    if not callable(exec_fn) or not callable(escape):
        logger.debug(
            "tool_output_compresr: skipping backend cache prune for %s; "
            "file_ops has no shell execution helpers",
            cache_dir,
        )
        return

    snippet = """
import pathlib
import sys
import traceback

try:
    cache_dir = pathlib.Path(sys.argv[1])
    max_bytes = int(sys.argv[2])
    keep_path = pathlib.Path(sys.argv[3]).resolve()
    entries = []
    total = 0
    for path in cache_dir.iterdir():
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        size = int(stat.st_size)
        total += size
        entries.append((float(stat.st_mtime), str(path), size))
    if total <= max_bytes:
        raise SystemExit(0)
    for _, path_str, size in sorted(entries):
        path = pathlib.Path(path_str)
        try:
            if path.resolve() == keep_path:
                continue
            path.unlink()
            total -= size
        except OSError:
            continue
        if total <= max_bytes:
            break
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
"""
    last_result = None
    for python_cmd in ("python3", "python"):
        command = (
            f"{python_cmd} -c {escape(snippet)} "
            f"{escape(cache_dir)} {escape(str(max_bytes))} {escape(keep_path)}"
        )
        result = exec_fn(command, timeout=30)
        if getattr(result, "exit_code", None) == 0:
            return
        last_result = result
    logger.debug(
        "tool_output_compresr: backend cache prune failed for %s: %s",
        cache_dir,
        getattr(last_result, "stdout", "") if last_result is not None else "",
    )

def store_original(
    cache_id: str,
    content: str,
    task_id: str = "default",
    max_cache_mb: int = 256,
) -> Optional[str]:
    """Persist ``content`` under ``cache_id`` via Hermes's file-ops backend.

    Returns the ABSOLUTE path written, or ``None`` if the write failed. We resolve
    to an absolute path (against the task's cwd at write time) so the recovery
    reference survives a later ``cd``: Hermes resolves *relative* read paths against
    the live terminal cwd, which moves, but an absolute path does not. On write
    failure we return ``None`` so the caller fails open instead of emitting a
    reference to a file that was never written.
    """
    from tools.file_tools import _get_file_ops, _resolve_path_for_task

    ops = _get_file_ops(task_id)
    rel = relative_cache_path(cache_id)

    # Keep the cache out of the user's VCS regardless of their workspace.
    if task_id not in _gitignore_done:
        try:
            ops.write_file(f"{_CACHE_ROOT}/.gitignore", "*\n")
            _gitignore_done.add(task_id)
        except Exception as e:  # pragma: no cover - cosmetic, never fatal
            logger.debug("tool_output_compresr: could not write cache .gitignore: %s", e)

    try:
        ops.write_file(rel, content)
    except Exception as e:
        logger.warning("tool_output_compresr: cache write failed for %s: %s", rel, e)
        return None
    try:
        resolved = _resolve_path_for_task(rel, task_id)
    except Exception:  # pragma: no cover - resolver is defensive; fall back to rel
        return rel
    try:
        _prune_cache_dir_via_backend(
            ops,
            str(Path(resolved).parent),
            max(0, int(max_cache_mb)) * 1024 * 1024,
            str(resolved),
        )
    except Exception as e:  # pragma: no cover - pruning must never break recovery
        logger.debug("tool_output_compresr: cache prune failed: %s", e)
    return str(resolved)
