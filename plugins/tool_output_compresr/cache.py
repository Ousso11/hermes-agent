"""Persist original tool outputs so recovery references resolve.

The canonical cache lives under ``HERMES_HOME/cache/compresr/tool-output``. We
write originals there on the host, then translate the path to an agent-visible
location only when the active backend can prove one — otherwise callers fail open
rather than emit a recovery reference the agent can't read.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_SUBDIR = Path("cache") / "compresr" / "tool-output"
_CONTAINER_HERMES_HOME = "/root/.hermes"  # HERMES_HOME inside Docker/Modal backends
_DIR_MODE = 0o700
_FILE_MODE = 0o600
_MB = 1024 * 1024

# Serialize write+prune across threads: subagents share one cache dir, so a
# concurrent prune must never race a sibling that just handed out a footer path.
_STORE_LOCK = threading.Lock()

# Pin recently-written entries against size-based eviction for this long, so a
# footer path just returned to the model survives a parallel prune.
_PRUNE_PIN_SECONDS = 300.0


def get_cache_root() -> Path:
    """Return the host-side cache root for compressed tool outputs."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / _CACHE_SUBDIR


def ensure_cache_root() -> Path:
    """Create the cache root with restrictive permissions if needed."""
    root = get_cache_root()
    root.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
    try:
        os.chmod(root, _DIR_MODE)
    except OSError:
        pass
    return root


def cache_file_path(cache_id: str) -> Path:
    """Host-side file path for *cache_id* — the single source of truth, used by
    :func:`store_original`."""
    return get_cache_root() / cache_id


def _get_active_env(task_id: str):
    try:
        from tools.terminal_tool import get_active_env as _get

        return _get(task_id)
    except Exception:
        return None


def _agent_visible_cache_path(cache_path: Path, task_id: str) -> Optional[str]:
    """Translate *cache_path* to the path the active backend can read.

    Local backends read the host path directly; non-local backends need a
    concrete mounted/synced path. Return ``None`` (→ caller fails open) if we
    can't establish one.
    """
    active_env = _get_active_env(task_id)

    if active_env is not None:
        env_name = active_env.__class__.__name__
        if env_name == "LocalEnvironment":
            try:
                return str(cache_path.resolve())
            except OSError:
                return str(cache_path)
        if env_name == "SingularityEnvironment" or "singularity" in env_name.lower():
            return None

        remote_home = getattr(active_env, "_remote_home", None)
        if isinstance(remote_home, str) and remote_home.strip():
            container_base = f"{remote_home.rstrip('/')}/.hermes"
        elif env_name in {"DockerEnvironment", "ModalEnvironment"}:
            container_base = _CONTAINER_HERMES_HOME
        else:
            return None
    else:
        backend = (os.getenv("TERMINAL_ENV") or "local").strip().lower() or "local"
        if backend == "local":
            try:
                return str(cache_path.resolve())
            except OSError:
                return str(cache_path)
        if backend in {"docker", "modal"}:
            container_base = _CONTAINER_HERMES_HOME
        else:
            return None

    try:
        from tools.credential_files import map_cache_path_to_container

        return map_cache_path_to_container(str(cache_path), container_base=container_base)
    except Exception as e:  # pragma: no cover - translation is best effort
        logger.debug("tool_output_compresr: cache path mapping failed: %s", e)
        return None


def _force_sync_visible_cache(cache_path: Path, task_id: str) -> bool:
    """Best-effort force sync for backends that stage files into a remote FS."""
    active_env = _get_active_env(task_id)
    if active_env is None:
        return True

    env_name = active_env.__class__.__name__
    if env_name == "LocalEnvironment":
        return True
    if env_name == "SingularityEnvironment" or "singularity" in env_name.lower():
        return True

    sync_manager = None
    for attr in ("_sync_manager", "sync_manager", "_file_sync_manager"):
        candidate = getattr(active_env, attr, None)
        if candidate is not None and callable(getattr(candidate, "sync", None)):
            sync_manager = candidate
            break
    if sync_manager is None:
        return True

    # raise_on_error surfaces transport failures that sync() would otherwise
    # swallow (leaving a dangling footer); fall back for managers without it.
    try:
        try:
            sync_manager.sync(force=True, raise_on_error=True)
        except TypeError:
            sync_manager.sync(force=True)
        return True
    except Exception as e:
        # Don't unlink cache_path: it is content-addressed, so a sibling may
        # already hold this path. Leave it for the pruner to reclaim.
        logger.warning("tool_output_compresr: force sync failed for %s: %s", cache_path, e)
        return False


def _prune_cache_dir(cache_dir: str, max_bytes: int, keep_path: str) -> None:
    """Best-effort size-based prune of the host-side cache root."""
    if max_bytes <= 0:
        return
    root = Path(cache_dir)
    keep = Path(keep_path)
    now = time.time()
    try:
        entries = []
        total = 0
        for path in root.iterdir():
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            size = int(stat.st_size)
            total += size
            entries.append((float(stat.st_mtime), path, size))
        if total <= max_bytes:
            return
        for mtime, path, size in sorted(entries):
            try:
                if path.resolve() == keep.resolve():
                    continue
                if now - mtime < _PRUNE_PIN_SECONDS:  # pin recent footer targets
                    continue
                path.unlink()
                total -= size
            except OSError:
                continue
            if total <= max_bytes:
                break
    except Exception as e:  # pragma: no cover - pruning must never break recovery
        logger.debug("tool_output_compresr: cache prune failed for %s: %s", cache_dir, e)


def store_original(
    cache_id: str,
    content: str,
    task_id: str = "default",
    max_cache_mb: int = 256,
) -> Optional[str]:
    """Persist ``content`` under ``cache_id`` on the host cache root.

    Returns an agent-visible path when one can be proven, else ``None`` (caller
    must fail open).
    """
    root = ensure_cache_root()
    cache_path = cache_file_path(cache_id)
    try:
        cache_path.write_text(content, encoding="utf-8")
        try:
            os.chmod(cache_path, _FILE_MODE)
        except OSError:
            pass
    except Exception as e:
        logger.warning("tool_output_compresr: cache write failed for %s: %s", cache_path, e)
        return None

    if not _force_sync_visible_cache(cache_path, task_id):
        return None

    visible_path = _agent_visible_cache_path(cache_path, task_id)
    if visible_path is None:
        # Content-addressed: don't unlink — a sibling may already hold this path.
        logger.warning(
            "tool_output_compresr: cache path not visible to the active backend: %s",
            cache_path,
        )
        return None

    try:
        with _STORE_LOCK:
            _prune_cache_dir(str(root), max(0, int(max_cache_mb)) * _MB, str(cache_path))
    except Exception as e:  # pragma: no cover - pruning must never break recovery
        logger.debug("tool_output_compresr: cache prune failed: %s", e)
    return visible_path
