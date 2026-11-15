"""The training queue: launch requests waiting to start, in order. It never starts anything itself.

Thread-safe: API requests and the scheduler (``queue_runner``) use one queue concurrently, and
every method is one atomic step. Starting an entry is two steps for its caller: ``claim`` it
(nobody else can then claim or remove it, and its name stays reserved), then ``discard`` it once
its job runs or ``mark_failed`` it. So an entry is started at most once. Every change is saved
to the storage before the method returns.
"""

import dataclasses
import threading

from attentive_snake.app.launch_request import LaunchRequest
from attentive_snake.app.queue_entry import QueueEntry, new_entry
from attentive_snake.app.queue_storage import MemoryStorage, QueueState, QueueStorage

_RESERVING = ("queued", "starting")  # a failed entry does not hold its name: it is not going to start by itself


class NameReserved(RuntimeError):
    """An entry for this run of this game is already waiting in the queue."""


class TrainingQueue:
    def __init__(self, storage: QueueStorage | None = None) -> None:
        self._storage = storage or MemoryStorage()
        self._lock = threading.Lock()
        state = self._storage.load()
        # A "starting" entry of a server that died never got its job: it waits again.
        self._entries = [dataclasses.replace(e, state="queued") if e.state == "starting" else e for e in state.entries]
        self._max_parallel = state.max_parallel

    def add(self, request: LaunchRequest) -> QueueEntry:
        """Append ``request``; ``NameReserved`` if an entry for the same run of the same game is waiting."""
        with self._lock:
            if self._reserved(request.game, request.run):
                raise NameReserved(f"run {request.run!r} is already in the training queue.")
            entry = new_entry(request)
            self._entries.append(entry)
            self._save()
            return entry

    def entries(self) -> list[QueueEntry]:
        """Every entry, oldest first."""
        with self._lock:
            return list(self._entries)

    def is_reserved(self, game: str, run: str) -> bool:
        """Whether a waiting entry will create or resume run ``run`` of ``game``."""
        with self._lock:
            return self._reserved(game, run)

    def remove(self, entry_id: str) -> QueueEntry:
        """Take a waiting or failed entry out; ``LookupError`` if there is none (or it is starting right now)."""
        with self._lock:
            entry = self._find(entry_id, ("queued", "failed"))
            self._entries.remove(entry)
            self._save()
            return entry

    def claim(self, entry_id: str) -> QueueEntry:
        """Mark a waiting or failed entry as starting; ``LookupError`` if there is none (or it is starting already)."""
        with self._lock:
            return self._replace(self._find(entry_id, ("queued", "failed")), state="starting", error=None)

    def claim_next(self) -> QueueEntry | None:
        """Mark the oldest waiting entry as starting and return it; ``None`` if nothing waits."""
        with self._lock:
            waiting = next((entry for entry in self._entries if entry.state == "queued"), None)
            return None if waiting is None else self._replace(waiting, state="starting")

    def discard(self, entry_id: str) -> None:
        """Drop a claimed entry: its job has started."""
        with self._lock:
            self._entries.remove(self._find(entry_id, ("starting",)))
            self._save()

    def mark_failed(self, entry_id: str, error: str) -> None:
        """Keep a claimed entry, in its place, as failed with ``error``."""
        with self._lock:
            self._replace(self._find(entry_id, ("starting",)), state="failed", error=error)

    @property
    def max_parallel(self) -> int:
        """How many queue-started jobs may run at once (0: nothing starts by itself)."""
        with self._lock:
            return self._max_parallel

    def set_max_parallel(self, value: int) -> None:
        if value < 0:
            raise ValueError(f"max_parallel must be 0 or more, got {value}.")
        with self._lock:
            self._max_parallel = value
            self._save()

    def _reserved(self, game: str, run: str) -> bool:
        return any(e.state in _RESERVING and (e.request.game, e.request.run) == (game, run) for e in self._entries)

    def _find(self, entry_id: str, states: tuple[str, ...]) -> QueueEntry:
        for entry in self._entries:
            if entry.id == entry_id and entry.state in states:
                return entry
        raise LookupError(f"the training queue has no {' or '.join(states)} entry {entry_id!r}.")

    def _replace(self, entry: QueueEntry, **changes) -> QueueEntry:
        changed = dataclasses.replace(entry, **changes)
        self._entries[self._entries.index(entry)] = changed
        self._save()
        return changed

    def _save(self) -> None:
        self._storage.save(QueueState(tuple(self._entries), self._max_parallel))
