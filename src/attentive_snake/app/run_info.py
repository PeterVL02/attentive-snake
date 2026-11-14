"""Read-only facts about a run directory: config, checkpoints, metrics, history (run.json) and the tail of its log."""

import dataclasses
from pathlib import Path

from attentive_snake.app.config_json import dataclass_to_json
from attentive_snake.config import load_experiment
from attentive_snake.run_dir import config_path, list_checkpoints, metrics_path
from attentive_snake.run_metadata import event_to_json, latest_description, lineage, read_events
from attentive_snake.training import checkpoint_steps
from attentive_snake.training.metrics import read_metrics

LOG_FILENAME = "train.log"
_TAIL_BYTES = 64 * 1024


def log_path(run_dir: Path) -> Path:
    return Path(run_dir) / LOG_FILENAME


def checkpoints(run_dir: Path) -> list[dict]:
    """``[{"name", "steps"}]``, ascending steps."""
    return [{"name": path.name, "steps": checkpoint_steps(path)} for path in list_checkpoints(run_dir)]


def metrics(run_dir: Path) -> list[dict]:
    path = metrics_path(run_dir)
    return read_metrics(path) if path.is_file() else []


def config_summary(run_dir: Path) -> dict:
    """Game, model kind, game rules and training target from config.toml (``None`` values if it cannot be read).

    ``width`` / ``height`` are the rules' fields of that name for grid games such as Snake
    (``None`` for games without them); the current Runs tab shows them.
    """
    try:
        config = load_experiment(config_path(run_dir))
    except (OSError, ValueError, TypeError):
        return {"game": None, "kind": None, "rules": None, "width": None, "height": None, "total_env_steps": None}
    return {
        "game": config.game_name,
        "kind": config.model.kind,
        "rules": dataclass_to_json(config.game),
        "width": getattr(config.game, "width", None),
        "height": getattr(config.game, "height", None),
        "total_env_steps": config.train.total_env_steps,
    }


def history(run_dir: Path) -> dict:
    """``{"events": [...], "description": latest or None, "parent": lineage or None}`` from run.json.

    A run without run.json (or with an unreadable one) has no events.
    """
    try:
        events = read_events(run_dir)
    except ValueError:
        events = []
    parent = lineage(events)
    return {
        "events": [event_to_json(event) for event in events],
        "description": latest_description(events),
        "parent": None if parent is None else dataclasses.asdict(parent),
    }


def log_tail(run_dir: Path, lines: int = 200) -> str:
    """The last ``lines`` lines of the run's train.log ('' if there is none)."""
    path = log_path(run_dir)
    if not path.is_file():
        return ""
    with path.open("rb") as file:
        file.seek(max(0, path.stat().st_size - _TAIL_BYTES))
        text = file.read().decode("utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])
