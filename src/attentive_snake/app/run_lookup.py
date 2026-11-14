"""Resolving user-supplied run and checkpoint names to paths, rejecting anything outside the runs directory."""

from pathlib import Path

from attentive_snake.app.errors import conflict, invalid_input_as_422, not_found
from attentive_snake.run_dir import config_path, list_checkpoints, validate_run_name


def run_name(name: str) -> str:
    """The validated name (422 if it is not a valid run name)."""
    with invalid_input_as_422():
        return validate_run_name(name)


def existing_run_dir(runs_dir: Path, name: str) -> Path:
    """The directory of an existing run (one with a config.toml); 422 for bad names, 404 for missing runs."""
    run_dir = runs_dir / run_name(name)
    if not config_path(run_dir).is_file():
        raise not_found(f"no run named {name!r}.")
    return run_dir


def checkpoint_in_run(run_dir: Path, checkpoint: str) -> Path:
    """The checkpoint file of ``run_dir`` named ``checkpoint`` (a bare file name); 404 otherwise."""
    for path in list_checkpoints(run_dir):
        if path.name == checkpoint:
            return path
    raise not_found(f"run {run_dir.name!r} has no checkpoint {checkpoint!r}.")


def latest_checkpoint_in_run(run_dir: Path) -> Path:
    """The newest checkpoint file of ``run_dir``; 409 if it has none yet."""
    checkpoints = list_checkpoints(run_dir)
    if not checkpoints:
        raise conflict(f"run {run_dir.name!r} has no checkpoints yet.")
    return checkpoints[-1]
