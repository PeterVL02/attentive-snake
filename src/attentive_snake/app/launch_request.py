"""A launch request: everything needed to start one training job later, as plain JSON-able data.

The same request starts a job right away (``launch_plan.start_launch``) or waits in the training
queue (``training_queue``). It holds names, never paths, and experiment tables, never an
``ExperimentConfig``, so it can be written to a file and read back by another server process.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

LaunchKind = Literal["new", "fork", "resume"]


@dataclass(frozen=True)
class LaunchRequest:
    """
    Attributes:
        kind: ``new`` (a new run), ``fork`` (a new run from a parent's checkpoint) or ``resume``.
        game: The game of the run (and of a fork's parent).
        run: Name of the run to create (new, fork) or to resume.
        config: Experiment tables ``{section: {field: value}}`` of the new run (new, fork).
        description: Saved in the run's run.json.
        parent: The run a fork starts from.
        checkpoint: The parent's checkpoint file a fork starts from; ``None`` = its latest when the job starts.
        total_env_steps: Resume: train until this many env steps (``None``: the run's target).
        keep_checkpoints: Resume: change ``train.keep_checkpoints`` (``None``: keep it).
        evaluation: Resume: changed ``[evaluation]`` settings (``interval`` / ``games`` / ``seed``).
    """

    kind: LaunchKind
    game: str
    run: str
    config: dict[str, dict[str, Any]] | None = None
    description: str | None = None
    parent: str | None = None
    checkpoint: str | None = None
    total_env_steps: int | None = None
    keep_checkpoints: bool | None = None
    evaluation: dict[str, int] = field(default_factory=dict)


def new_run(game: str, run: str, config: dict[str, dict[str, Any]], description: str | None = None) -> LaunchRequest:
    return LaunchRequest("new", game, run, config=config, description=description)


def fork(
    game: str,
    parent: str,
    run: str,
    config: dict[str, dict[str, Any]],
    checkpoint: str | None = None,
    description: str | None = None,
) -> LaunchRequest:
    return LaunchRequest("fork", game, run, config=config, description=description, parent=parent, checkpoint=checkpoint)


def resume(
    game: str,
    run: str,
    total_env_steps: int | None = None,
    keep_checkpoints: bool | None = None,
    description: str | None = None,
    evaluation: dict[str, int] | None = None,
) -> LaunchRequest:
    return LaunchRequest(
        "resume",
        game,
        run,
        description=description,
        total_env_steps=total_env_steps,
        keep_checkpoints=keep_checkpoints,
        evaluation=dict(evaluation or {}),
    )


def request_to_json(request: LaunchRequest) -> dict[str, Any]:
    return dataclasses.asdict(request)


def request_from_json(data: dict[str, Any]) -> LaunchRequest:
    """The request written by ``request_to_json`` (``TypeError`` for unknown or missing keys)."""
    return LaunchRequest(**data)
