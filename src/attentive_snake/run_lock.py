"""One training process per run directory, wherever it was started: an operating-system lock on ``training.lock``.

The kernel releases the lock when the process ends, however it ends, so a lock can never be
stale. The file itself stays behind: deleting it could let two processes hold "the" lock on two
different files. It holds the process id of the last holder, for display only.

Without ``fcntl`` (Windows) nothing is locked and no run is ever reported as locked.
"""

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from attentive_snake.run_dir import lock_path

try:
    import fcntl
except ImportError:
    fcntl = None

# ``lock_holder`` holds a shared lock for an instant, so a free lock can look taken: try for a moment.
_ACQUIRE_ATTEMPTS = 20
_ACQUIRE_PAUSE_SECONDS = 0.1


class RunLocked(RuntimeError):
    """Another process is training this run."""


@contextmanager
def hold_run_lock(run_dir: Path) -> Iterator[None]:
    """Hold the run's lock inside the block, creating the directory if needed; ``RunLocked`` if another process has it."""
    if fcntl is None:
        yield
        return
    path = lock_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        _acquire(descriptor, run_dir)
        os.ftruncate(descriptor, 0)
        os.pwrite(descriptor, str(os.getpid()).encode(), 0)
        yield
    finally:
        os.close(descriptor)  # releases the lock


def lock_holder(run_dir: Path) -> int | None:
    """The process id training the run (0 if it has not written its id yet), or ``None`` if no process holds the lock."""
    if fcntl is None:
        return None
    try:
        descriptor = os.open(lock_path(run_dir), os.O_RDONLY)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            text = os.pread(descriptor, 32, 0).decode(errors="replace").strip()
            return int(text) if text.isdecimal() else 0
        return None
    finally:
        os.close(descriptor)


def _acquire(descriptor: int, run_dir: Path) -> None:
    for attempt in range(_ACQUIRE_ATTEMPTS):
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if attempt < _ACQUIRE_ATTEMPTS - 1:
                time.sleep(_ACQUIRE_PAUSE_SECONDS)
    raise RunLocked(f"{run_dir} is already training (process {lock_holder(run_dir)}).")
