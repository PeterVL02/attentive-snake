"""Resolving user-supplied game, run and checkpoint names to paths, rejecting anything outside the runs directory.

Runs live in ``<runs root>/<game>/<run>/``. A training job is known by ``job_key(game, run)``.
"""

from pathlib import Path

from attentive_snake.app.errors import conflict, invalid_input_as_422, not_found
from attentive_snake.app.game_lookup import known_game
from attentive_snake.run_dir import config_path, game_runs_dir, list_checkpoints, validate_run_name


def run_name(name: str) -> str:
    """The validated name (422 if it is not a valid run name)."""
    with invalid_input_as_422():
        return validate_run_name(name)


def job_key(game: str, name: str) -> str:
    """How ``TrainingJobs`` knows the job of run ``name`` of ``game`` (run names are unique per game only)."""
    return f"{game}/{name}"


def run_job_key(run_dir: Path) -> str:
    """``job_key`` of a run directory ``<runs root>/<game>/<run>``."""
    return job_key(run_dir.parent.name, run_dir.name)


def game_runs(runs_dir: Path, game: str) -> Path:
    """The directory of ``game``'s runs; 404 for an unknown game."""
    return game_runs_dir(runs_dir, known_game(game).name)


def new_run_dir(runs_dir: Path, game: str, name: str) -> Path:
    """Where run ``name`` of ``game`` goes (not checked to be free); 404 unknown game, 422 bad name."""
    return game_runs(runs_dir, game) / run_name(name)


def existing_run_dir(runs_dir: Path, game: str, name: str) -> Path:
    """The directory of an existing run (one with a config.toml); 422 for bad names, 404 for missing games or runs."""
    run_dir = new_run_dir(runs_dir, game, name)
    if not config_path(run_dir).is_file():
        raise not_found(f"no {game} run named {name!r}.")
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
