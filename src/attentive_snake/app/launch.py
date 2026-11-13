"""Starting training jobs for new and existing runs (the steps between a validated request and a subprocess)."""

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from attentive_snake.app.errors import conflict, invalid_input_as_422
from attentive_snake.app.jobs import JobConflict, TrainingJobs
from attentive_snake.app.paths import AppPaths
from attentive_snake.app.run_info import log_path
from attentive_snake.app.train_command import new_run_command, resume_command
from attentive_snake.config import ExperimentConfig, save_experiment
from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.run_dir import config_path


def parse_experiment(tables: Mapping[str, Any]) -> ExperimentConfig:
    """The config described by ``tables`` (422 with the validation message if it is invalid)."""
    with invalid_input_as_422():
        return experiment_from_tables(tables)


def start_new_run(paths: AppPaths, jobs: TrainingJobs, run_dir: Path, config: ExperimentConfig) -> dict:
    """Create ``run_dir`` (409 if it exists), write its config.toml and launch train.py on it."""
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise conflict(f"run {run_dir.name!r} already exists; resume it instead.") from None
    config = dataclasses.replace(config, train=dataclasses.replace(config.train, checkpoint_dir=run_dir))
    save_experiment(config, config_path(run_dir))
    command = new_run_command(paths.train_script, run_dir, config_path(run_dir))
    return _launch(jobs, run_dir, command, "new")


def resume_run(paths: AppPaths, jobs: TrainingJobs, run_dir: Path, total_steps: int | None) -> dict:
    """Launch train.py --resume on ``run_dir`` (409 if it is already training)."""
    return _launch(jobs, run_dir, resume_command(paths.train_script, run_dir, total_steps), "resume")


def _launch(jobs: TrainingJobs, run_dir: Path, command: list[str], mode: str) -> dict:
    try:
        return jobs.launch(run_dir.name, command, log_path(run_dir), mode)
    except JobConflict as error:
        raise conflict(str(error)) from error
