"""Starting training jobs for new, resumed and forked runs (the steps between a validated request and a subprocess)."""

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from attentive_snake.app.errors import conflict, invalid_input_as_422
from attentive_snake.app.jobs import JobConflict, TrainingJobs
from attentive_snake.app.paths import AppPaths
from attentive_snake.app.run_info import log_path
from attentive_snake.app.run_lookup import run_job_key
from attentive_snake.app.train_command import fork_command, new_run_command, resume_command
from attentive_snake.config import ExperimentConfig, load_experiment, save_experiment
from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.config.fork_rules import require_forkable
from attentive_snake.run_dir import config_path


def parse_experiment(tables: Mapping[str, Any]) -> ExperimentConfig:
    """The config described by ``tables`` (422 with the validation message if it is invalid)."""
    with invalid_input_as_422():
        return experiment_from_tables(tables)


def require_fork_compatible(parent_dir: Path, config: ExperimentConfig) -> None:
    """422 naming every setting ``config`` changed that a fork of ``parent_dir`` must keep."""
    with invalid_input_as_422():
        require_forkable(load_experiment(config_path(parent_dir)), config)


def start_new_run(
    paths: AppPaths, jobs: TrainingJobs, run_dir: Path, config: ExperimentConfig, description: str | None = None
) -> dict:
    """Create ``run_dir`` (409 if it exists), write its config.toml and launch train.py on it."""
    _create_run(run_dir, config)
    command = new_run_command(paths.train_script, run_dir, config_path(run_dir), description)
    return _launch(jobs, run_dir, command, "new")


def fork_run(
    paths: AppPaths,
    jobs: TrainingJobs,
    parent_checkpoint: Path,
    run_dir: Path,
    config: ExperimentConfig,
    description: str | None = None,
) -> dict:
    """Create ``run_dir`` (409 if it exists) and launch train.py --fork from ``parent_checkpoint`` on it.

    Validate the config with ``require_fork_compatible`` first.
    """
    _create_run(run_dir, config)
    parent_dir = parent_checkpoint.parent
    command = fork_command(paths.train_script, run_dir, parent_dir, parent_checkpoint.name, config_path(run_dir), description)
    return _launch(jobs, run_dir, command, "fork")


def resume_run(
    paths: AppPaths,
    jobs: TrainingJobs,
    run_dir: Path,
    total_steps: int | None,
    keep_checkpoints: bool | None = None,
    description: str | None = None,
    evaluation: Mapping[str, int] | None = None,
) -> dict:
    """Launch train.py --resume on ``run_dir`` (409 if it is already training); ``evaluation`` as in ``resume_command``."""
    command = resume_command(paths.train_script, run_dir, total_steps, keep_checkpoints, description, evaluation)
    return _launch(jobs, run_dir, command, "resume")


def _create_run(run_dir: Path, config: ExperimentConfig) -> None:
    """Make the run directory (409 if it exists) and write its config.toml with the directory as checkpoint_dir."""
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise conflict(f"run {run_dir.name!r} already exists; choose another name, or resume it.") from None
    config = dataclasses.replace(config, train=dataclasses.replace(config.train, checkpoint_dir=run_dir))
    save_experiment(config, config_path(run_dir))


def _launch(jobs: TrainingJobs, run_dir: Path, command: list[str], mode: str) -> dict:
    try:
        return jobs.launch(run_job_key(run_dir), command, log_path(run_dir), mode)
    except JobConflict as error:
        raise conflict(str(error)) from error
