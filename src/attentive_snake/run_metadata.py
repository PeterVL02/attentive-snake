"""The history of a run (``run.json``): when it was started, resumed or forked, with optional descriptions.

Not configuration (that is ``config.toml``): a log of what was done to the run::

    {"events": [
        {"timestamp": "2026-11-14T10:00:00+00:00", "action": "start", "description": "baseline",
         "env_steps_at_action": 0},
        {"timestamp": "...", "action": "resume", "description": null, "env_steps_at_action": 500000},
    ]}

A forked run's first event has ``"action": "fork"`` and a ``"parent"`` entry:
``{"run": name, "checkpoint": file name, "env_steps": steps of that checkpoint}``.
"""

import dataclasses
import datetime
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from attentive_snake.run_dir import metadata_path

type RunAction = Literal["start", "resume", "fork"]
ACTIONS: tuple[str, ...] = ("start", "resume", "fork")


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Lineage:
    """Where a forked run came from."""

    run: str
    checkpoint: str
    env_steps: int


@dataclass(frozen=True)
class RunEvent:
    """One start, resume or fork of a run."""

    action: RunAction
    env_steps_at_action: int
    description: str | None = None
    parent: Lineage | None = None
    timestamp: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action {self.action!r}; valid actions: {', '.join(ACTIONS)}.")
        if (self.action == "fork") != (self.parent is not None):
            raise ValueError("a fork event needs a parent, and only a fork event has one.")


def clean_description(text: str | None) -> str | None:
    """``text`` with surrounding whitespace removed, or ``None`` if nothing is left."""
    text = (text or "").strip()
    return text or None


def event_to_json(event: RunEvent) -> dict[str, Any]:
    data = dataclasses.asdict(event)
    if data["parent"] is None:
        del data["parent"]
    return data


def event_from_json(data: dict[str, Any]) -> RunEvent:
    parent = data.get("parent")
    return RunEvent(
        action=data["action"],
        env_steps_at_action=int(data["env_steps_at_action"]),
        description=data.get("description"),
        parent=None if parent is None else Lineage(str(parent["run"]), str(parent["checkpoint"]), int(parent["env_steps"])),
        timestamp=str(data["timestamp"]),
    )


def read_events(run_dir: Path) -> list[RunEvent]:
    """The run's events, oldest first (empty if it has no ``run.json``); ``ValueError`` if the file is malformed."""
    path = metadata_path(run_dir)
    if not path.is_file():
        return []
    try:
        return [event_from_json(item) for item in json.loads(path.read_text(encoding="utf-8"))["events"]]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is not a valid run history: {error}") from error


def append_event(run_dir: Path, event: RunEvent) -> None:
    """Add ``event`` at the end of the run's history (the file is replaced atomically)."""
    path = metadata_path(run_dir)
    events = [*read_events(run_dir), event]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps({"events": [event_to_json(e) for e in events]}, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def latest_description(events: list[RunEvent]) -> str | None:
    """The most recent non-empty description, or ``None``."""
    return next((event.description for event in reversed(events) if event.description), None)


def lineage(events: list[RunEvent]) -> Lineage | None:
    """The parent of a forked run (from its fork event), or ``None`` for a run started from scratch."""
    return next((event.parent for event in events if event.parent is not None), None)
