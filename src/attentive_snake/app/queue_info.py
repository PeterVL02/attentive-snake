"""What the API says about a training-queue entry: its request in short, without the whole config."""

from typing import Any

from attentive_snake.app.queue_entry import QueueEntry
from attentive_snake.config.experiment_io import experiment_from_tables


def _config_facts(tables: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    """Model kind, algorithm and training target of the entry's config; ``None`` each if it has none (a resume)."""
    try:
        config = experiment_from_tables(tables) if tables is not None else None
    except (ValueError, TypeError):  # valid when it was queued; the code may have changed since
        config = None
    if config is None:
        return {"model_kind": None, "algorithm": None, "total_env_steps": None}
    return {"model_kind": config.model.kind, "algorithm": config.train.algorithm, "total_env_steps": config.train.total_env_steps}


def entry_info(entry: QueueEntry) -> dict[str, Any]:
    """``id``, ``game``, ``run``, ``kind`` (new / fork / resume), ``parent`` and ``checkpoint`` (fork), ``description``,
    ``queued_at``, ``state``, ``error``, and ``model_kind`` / ``algorithm`` / ``total_env_steps`` where known."""
    request = entry.request
    facts = _config_facts(request.config)
    if request.kind == "resume":
        facts["total_env_steps"] = request.total_env_steps
    return {
        "id": entry.id,
        "game": request.game,
        "run": request.run,
        "kind": request.kind,
        "parent": request.parent,
        "checkpoint": request.checkpoint,
        "description": request.description,
        "queued_at": entry.queued_at,
        "state": entry.state,
        "error": entry.error,
        **facts,
    }
