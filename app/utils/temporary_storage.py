"""Private process storage with OS locks to reap crash leftovers safely.

Only application-marked, unlocked runtime directories are removed. A running
process holds its owner lock for its entire lifetime; another process leaves
that directory alone, even when a dataset is idle.
"""
import atexit
import os
from pathlib import Path
import shutil
import tempfile
import threading

from app.utils.logger import get_logger

logger = get_logger(__name__)
MARKER = b"AI_DATA_ANALYST_RUNTIME_V1\n"
_runtime = None
_owner = None
_initialization_lock = threading.Lock()


def _lock(handle):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def reap_orphaned_runtimes(parent):
    """Inspect only our marked directories, never follow a symlink outside temp."""
    parent = Path(parent).resolve()
    removed = 0
    for root in parent.glob("analyst-runtime-*"):
        owner_path = root / "owner.lock"
        if root.is_symlink() or not root.is_dir() or root.resolve().parent != parent or owner_path.is_symlink():
            continue
        try:
            with owner_path.open("r+b") as owner:
                if owner.read(len(MARKER) + 1) != MARKER:
                    continue
                try:
                    _lock(owner)
                except OSError:
                    continue  # A live process owns the files.
                _unlock(owner)
            # Runtime names are never reused; an unlocked marked owner is dead.
            shutil.rmtree(root)
            removed += 1
        except (OSError, FileNotFoundError):
            logger.warning("Temporary storage cleanup deferred")
    return removed


def _shutdown():
    global _owner, _runtime
    if _owner is not None:
        _owner.close()
        _owner = None
    if _runtime is not None:
        try:
            parent = Path(tempfile.gettempdir()).resolve()
            if not _runtime.is_symlink() and _runtime.resolve().parent == parent and _runtime.name.startswith("analyst-runtime-"):
                shutil.rmtree(_runtime)
        except OSError:
            pass  # The next process can reap a marked leftover.
        _runtime = None


def runtime_directory():
    global _runtime, _owner
    with _initialization_lock:
        if _runtime is None:
            parent = Path(tempfile.gettempdir())
            removed = reap_orphaned_runtimes(parent)
            _runtime = Path(tempfile.mkdtemp(prefix="analyst-runtime-", dir=parent))
            try:
                _owner = (_runtime / "owner.lock").open("w+b")
                _owner.write(MARKER)
                _owner.flush()
                _lock(_owner)
            except Exception:
                _shutdown()
                raise
            atexit.register(_shutdown)
            logger.info("Private temporary storage initialized | orphan_runtimes_removed=%d", removed)
        return _runtime
