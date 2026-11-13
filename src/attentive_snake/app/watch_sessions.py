"""A small registry of live ``WatchSession`` objects keyed by id; the oldest is evicted when full."""

import threading
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


class UnknownSession(LookupError):
    """No session with this id (never created, removed, or evicted)."""


@dataclass
class _Entry:
    session: Any
    lock: threading.Lock = field(default_factory=threading.Lock)


class WatchSessions:
    def __init__(self, capacity: int = 4) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1.")
        self._capacity = capacity
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, session: Any) -> str:
        """Register ``session``; returns its new id."""
        session_id = uuid.uuid4().hex
        with self._lock:
            self._entries[session_id] = _Entry(session)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
        return session_id

    @contextmanager
    def use(self, session_id: str) -> Iterator[Any]:
        """The session, held exclusively for the ``with`` block. Raises ``UnknownSession``."""
        with self._lock:
            entry = self._get(session_id)
            self._entries.move_to_end(session_id)
        with entry.lock:
            yield entry.session

    def remove(self, session_id: str) -> None:
        """Forget the session. Raises ``UnknownSession``."""
        with self._lock:
            self._get(session_id)
            del self._entries[session_id]

    def _get(self, session_id: str) -> _Entry:
        try:
            return self._entries[session_id]
        except KeyError:
            raise UnknownSession(session_id) from None

    def __len__(self) -> int:
        return len(self._entries)
