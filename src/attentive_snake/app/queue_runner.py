"""Starting the training queue's entries: automatically when a slot is free (``tick``), or on request (``run_now``).

The rule: at most ``queue.max_parallel`` jobs started by ``tick`` run at once; whenever fewer do,
the oldest waiting entry starts. Jobs the user started directly, and entries started with
``run_now``, take no slot and never hold the queue back. An entry that cannot be started stays
in the queue as failed, with the reason, and the next one is tried.

``tick`` is cheap and safe to call at any time, from any thread: after a change to the queue,
and regularly (``queue_ticker``) to notice that a job has ended. The runner also remembers what
became of every entry it started (``outcome``), for callers that submitted one and wait for it.
That memory is the server's, like the jobs themselves: it does not survive a restart.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from attentive_snake.app.jobs import TrainingJobs
from attentive_snake.app.launch_plan import LaunchedRun
from attentive_snake.app.launch_request import LaunchRequest
from attentive_snake.app.queue_entry import QueueEntry
from attentive_snake.app.run_lookup import run_job_key
from attentive_snake.app.training_queue import TrainingQueue

OutcomeState = Literal["queued", "failed_to_start", "running", "finished", "failed", "stopped", "ended", "unknown"]
_WAITING = ("queued", "running")


@dataclass(frozen=True)
class EntryOutcome:
    """
    Attributes:
        state: ``queued`` (not started yet), ``failed_to_start`` (still in the queue, with ``error``),
            ``running``, then how its job ended: ``finished``, ``failed`` or ``stopped``; ``ended`` if
            the run was launched again since, so only that it ended is known; ``unknown`` for an id
            that was removed from the queue, never existed, or was queued before the server restarted.
        run_dir: The run's directory, once the entry has started.
        error: Why it could not be started (``failed_to_start`` only).
    """

    state: OutcomeState
    run_dir: Path | None = None
    error: str | None = None

    @property
    def done(self) -> bool:
        """Nothing more will happen to this entry by itself."""
        return self.state not in _WAITING


@dataclass(frozen=True)
class _Started:
    entry: QueueEntry
    run_dir: Path
    pid: int
    uses_slot: bool


def _reason(error: Exception) -> str:
    """The message of ``error``: an API error's ``detail`` if it has one."""
    return str(getattr(error, "detail", None) or error) or type(error).__name__


class QueueRunner:
    def __init__(self, queue: TrainingQueue, jobs: TrainingJobs, launch: Callable[[LaunchRequest], LaunchedRun]) -> None:
        """``launch`` starts a request's training job in ``jobs`` or raises (``launch_plan.start_launch``)."""
        self._queue = queue
        self._jobs = jobs
        self._launch = launch
        self._started: dict[str, _Started] = {}  # entry id -> its job
        self._lock = threading.Lock()  # one start at a time, so slots are counted before each

    def tick(self) -> None:
        """Start waiting entries, oldest first, while a slot is free."""
        with self._lock:
            while self._slots_in_use() < self._queue.max_parallel:
                entry = self._queue.claim_next()
                if entry is None:
                    return
                try:
                    self._start(entry, uses_slot=True)
                except Exception:  # recorded on the entry; the queue goes on
                    continue

    def run_now(self, entry_id: str) -> LaunchedRun:
        """Start a waiting (or failed) entry at once, whatever is running; ``LookupError`` if there is none.

        If it cannot be started the entry is kept as failed and the error is raised.
        """
        with self._lock:
            return self._start(self._queue.claim(entry_id), uses_slot=False)

    def running(self) -> list[QueueEntry]:
        """The entries started here whose jobs are running, oldest start first."""
        with self._lock:
            return [started.entry for started in self._started.values() if self._job_state(started) == "running"]

    def outcome(self, entry_id: str) -> EntryOutcome:
        """What became of the entry ``entry_id``."""
        with self._lock:
            started = self._started.get(entry_id)
            if started is not None:
                return EntryOutcome(self._job_state(started), started.run_dir)
        for entry in self._queue.entries():
            if entry.id == entry_id:
                return EntryOutcome("failed_to_start", error=entry.error) if entry.state == "failed" else EntryOutcome("queued")
        with self._lock:  # it may have started between the two looks
            started = self._started.get(entry_id)
            return EntryOutcome("unknown") if started is None else EntryOutcome(self._job_state(started), started.run_dir)

    def _start(self, entry: QueueEntry, uses_slot: bool) -> LaunchedRun:
        try:
            launched = self._launch(entry.request)
        except Exception as error:
            self._queue.mark_failed(entry.id, _reason(error))
            raise
        self._started[entry.id] = _Started(entry, launched.run_dir, launched.status["pid"], uses_slot)
        self._queue.discard(entry.id)
        return launched

    def _slots_in_use(self) -> int:
        return sum(1 for started in self._started.values() if started.uses_slot and self._job_state(started) == "running")

    def _job_state(self, started: _Started) -> OutcomeState:
        """The state of the job this runner started; ``ended`` if the run's latest job is a later one."""
        status = self._jobs.status(run_job_key(started.run_dir))
        if status is None or status["pid"] != started.pid:
            return "ended"
        return status["state"]
