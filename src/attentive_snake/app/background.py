"""Background jobs in threads of the server, at most one running per key, with progress; for work that takes seconds to minutes.

The work is a function ``work(progress) -> result``; it calls ``progress(done, total)`` as it
goes and returns a JSON-able result. Status (``state`` running / finished / failed, progress,
result or error) stays available until the next job of the same key starts. Threads are
daemons: a server that exits does not wait for them.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

type Progress = Callable[[int, int], None]
type Work = Callable[[Progress], Any]
JobState = Literal["running", "finished", "failed"]


class JobBusy(RuntimeError):
    """A job with this key is still running."""


@dataclass
class _Job:
    request: dict[str, Any]
    total: int
    done: int = 0
    state: JobState = "running"
    result: Any = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "request": self.request,
            "progress": {"done": self.done, "total": self.total},
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class BackgroundJobs:
    def __init__(self) -> None:
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()

    def start(self, key: str, work: Work, request: dict[str, Any], total: int) -> dict[str, Any]:
        """Run ``work`` in a new thread; ``JobBusy`` if ``key``'s previous job is still running. Returns its status."""
        with self._lock:
            previous = self._jobs.get(key)
            if previous is not None and previous.state == "running":
                raise JobBusy(f"{key} is busy (started {time.strftime('%H:%M:%S', time.localtime(previous.started_at))}).")
            job = self._jobs[key] = _Job(request, total)
            status = job.describe()
        threading.Thread(target=self._run, args=(job, work), name=f"job {key}", daemon=True).start()
        return status

    def status(self, key: str) -> dict[str, Any] | None:
        """The latest job of ``key`` (running or not), or ``None`` if none was started."""
        with self._lock:
            job = self._jobs.get(key)
            return None if job is None else job.describe()

    def _run(self, job: _Job, work: Work) -> None:
        def progress(done: int, total: int) -> None:
            with self._lock:
                job.done, job.total = done, total

        try:
            result = work(progress)
        except Exception as error:  # reported to the client, not raised in a thread nobody joins
            outcome = {"state": "failed", "error": f"{type(error).__name__}: {error}"}
        else:
            outcome = {"state": "finished", "result": result}
        with self._lock:
            for name, value in outcome.items():
                setattr(job, name, value)
            job.finished_at = time.time()
