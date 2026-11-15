"""The training queue's file: ``<runs root>/queue.json``, so the queue survives a restart of the app.

::

    {"version": 1, "max_parallel": 1, "entries": [{"id", "request": {...}, "queued_at", "state", "error"}, ...]}

Entries are in queue order. The file is replaced atomically (written beside it, then renamed), so
a crash never leaves half a file. A file that cannot be read is moved aside (``queue.json.unreadable``)
and the queue starts empty: a damaged queue must not keep the app from starting.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from attentive_snake.app.queue_entry import QueueEntry, entry_from_json, entry_to_json

QUEUE_FILENAME = "queue.json"
VERSION = 1
DEFAULT_MAX_PARALLEL = 1


@dataclass(frozen=True)
class QueueState:
    """Everything the queue remembers: its entries in order and how many queue-started jobs may run at once."""

    entries: tuple[QueueEntry, ...] = ()
    max_parallel: int = DEFAULT_MAX_PARALLEL


class QueueStorage(Protocol):
    def load(self) -> QueueState: ...

    def save(self, state: QueueState) -> None: ...


def queue_path(runs_dir: Path) -> Path:
    return Path(runs_dir) / QUEUE_FILENAME


@dataclass
class MemoryStorage:
    """Keeps the state in memory only (tests, and a queue nobody needs after the process ends)."""

    state: QueueState = field(default_factory=QueueState)

    def load(self) -> QueueState:
        return self.state

    def save(self, state: QueueState) -> None:
        self.state = state


class QueueFile:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def load(self) -> QueueState:
        """The saved state; an empty queue if there is no file or it cannot be read."""
        if not self._path.is_file():
            return QueueState()
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            entries = tuple(entry_from_json(entry) for entry in data["entries"])
            return QueueState(entries, int(data["max_parallel"]))
        except (OSError, ValueError, TypeError, KeyError):
            os.replace(self._path, self._path.with_name(f"{self._path.name}.unreadable"))
            return QueueState()

    def save(self, state: QueueState) -> None:
        data = {"version": VERSION, "max_parallel": state.max_parallel, "entries": [entry_to_json(e) for e in state.entries]}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(f"{self._path.name}.tmp")
        temporary.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(temporary, self._path)
