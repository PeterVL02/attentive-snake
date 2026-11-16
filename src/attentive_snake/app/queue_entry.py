"""One entry of the training queue: a launch request that has not started yet, and why if it could not."""

import dataclasses
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from attentive_snake.app.launch_request import LaunchRequest, request_from_json, request_to_json

EntryState = Literal["queued", "starting", "failed"]


@dataclass(frozen=True)
class QueueEntry:
    """
    Attributes:
        id: Unique among all entries ever queued.
        request: What to start.
        queued_at: Seconds since the epoch.
        state: ``queued`` (waiting), ``starting`` (claimed by whoever is starting it right now)
            or ``failed`` (it could not be started; ``error`` says why). An entry that started
            leaves the queue: it is an ordinary run from then on.
        error: Why the entry could not be started (``failed`` only).
    """

    id: str
    request: LaunchRequest
    queued_at: float
    state: EntryState = "queued"
    error: str | None = None


def new_entry(request: LaunchRequest) -> QueueEntry:
    return QueueEntry(uuid.uuid4().hex[:12], request, time.time())


def entry_to_json(entry: QueueEntry) -> dict[str, Any]:
    return {**dataclasses.asdict(entry), "request": request_to_json(entry.request)}


def entry_from_json(data: dict[str, Any]) -> QueueEntry:
    """The entry written by ``entry_to_json`` (``TypeError`` / ``KeyError`` if it is not one)."""
    return QueueEntry(**{**data, "request": request_from_json(data["request"])})
