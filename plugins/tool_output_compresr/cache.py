"""Persist original tool outputs so recovery references resolve.

The canonical cache lives under ``HERMES_HOME/cache/compresr/tool-output``.
We write originals there on the host, then translate the path to an
agent-visible location only when the active backend can prove one. If that
authority cannot be established, callers fail open instead of emitting a
recovery reference that points at something the agent cannot read.
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

# Serialize write+prune across threads: subagents share one cache dir, and a
# concurrent prune must never race a sibling that just handed a footer path to
# the model.
_STORE_LOCK = threading.Lock()

# Recently-written entries are pinned against size-based eviction for this many
# seconds so a footer path just returned to the model can't be reclaimed by a
# parallel prune before the model reads it.
_PRUNE_PIN_SECONDS = 300.0


def get_cache_root() -> Path:
    """Return the host-side cache root for compressed tool outputs."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / _CACHE_SUBDIR


def ensure_cache_root() -> Path:
    """Create the cache root with restrictive permissions if needed."""
    root = get_cache_root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    return root


def cache_file_path(cache_id: str) -> Path:
    """Return the host-side file path for *cache_id*."""
    return get_cache_root() / cache_id


def relative_cache_path(cache_id: str) -> str:
    """Return the canonical host-side cache path for *cache_id*."""
    return str(cache_file_path(cache_id))


def _get_active_env(task_id: str):
    try:
        from tools.terminal_tool import get_active_env as _get

        return _get(task_id)
    except Exception:
        return None


def _agent_visible_cache_path(cache_path: Path, task_id: str) -> Optional[str]:
    """Translate *cache_path* to the path the active backend can read.

    Local backends can read the host path directly. Non-local backends need a
    concrete mounted/synced path. If we cannot establish that path, return
    ``None`` so the caller can fail open.
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
            container_base = "/root/.hermes"
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
            container_base = "/root/.hermes"
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

    # Ask the manager to surface transport failures. FileSyncManager.sync()
    # otherwise catches errors internally and returns None, so a failed upload
    # would leave store_original returning a remote path whose file was never
    # uploaded (a dangling footer). raise_on_error=True re-raises, letting us
    # fail open. Fall back gracefully for managers without the kwarg.
    try:
        try:
            sync_manager.sync(force=True, raise_on_error=True)
        except TypeError:
            sync_manager.sync(force=True)
        return True
    except Exception as e:
        try:
            cache_path.unlink(missing_ok=True)
        except OSError:
            pass
        logger.warning(
            "tool_output_compresr: force sync failed for %s: %s", cache_path, e
        )
        return False


def _prune_cache_dir_via_backend(
    _file_ops: object | None,
    cache_dir: str,
    max_bytes: int,
    keep_path: str,
) -> None:
    """Best-effort prune of the host-side cache root."""
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
                # Pin recently-written entries: a footer path just handed to
                # the model must survive a concurrent prune long enough to be
                # read back, even if the dir is over budget.
                if now - mtime < _PRUNE_PIN_SECONDS:
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

    Returns an agent-visible path when one can be proven, or ``None`` if the
    write or path translation failed. Callers must fail open on ``None``.
    """
    root = ensure_cache_root()
    cache_path = root / cache_id
    try:
        cache_path.write_text(content, encoding="utf-8")
        try:
            os.chmod(cache_path, 0o600)
        except OSError:
            pass
    except Exception as e:
        logger.warning("tool_output_compresr: cache write failed for %s: %s", cache_path, e)
        return None

    if not _force_sync_visible_cache(cache_path, task_id):
        return None

    visible_path = _agent_visible_cache_path(cache_path, task_id)
    if visible_path is None:
        try:
            cache_path.unlink(missing_ok=True)
        except OSError:
            pass
        logger.warning(
            "tool_output_compresr: cache path not visible to the active backend: %s",
            cache_path,
        )
        return None

    # Serialize prune across threads: subagents share this dir, so a concurrent
    # prune must not race a sibling's scan. Pin-recent (see the pruner) already
    # protects the file we're about to hand back as a footer path.
    try:
        with _STORE_LOCK:
            _prune_cache_dir_via_backend(
                None,
                str(root),
                max(0, int(max_cache_mb)) * 1024 * 1024,
                str(cache_path),
            )
    except Exception as e:  # pragma: no cover - pruning must never break recovery
        logger.debug("tool_output_compresr: cache prune failed: %s", e)
    return visible_path
