"""Runs of a game: list, detail, start, resume, fork (continue under a new name with changes) and stop.

Runs are addressed by game and name: ``?game=`` (default ``snake``) on every route with a
run name in the path and on the list. A new run's game is its config's ``[game] name``.
"""

from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from attentive_snake.app import run_info
from attentive_snake.app.config_json import experiment_to_json
from attentive_snake.app.errors import conflict
from attentive_snake.app.evaluation_info import recorded_evaluations
from attentive_snake.app.game_lookup import DEFAULT_GAME, GameParam
from attentive_snake.app.jobs import TrainingJobs
from attentive_snake.app.launch import fork_run, parse_experiment, require_fork_compatible, resume_run, start_new_run
from attentive_snake.app.paths import AppPaths
from attentive_snake.app.run_lookup import (
    checkpoint_in_run,
    existing_run_dir,
    game_runs,
    job_key,
    latest_checkpoint_in_run,
    new_run_dir,
)
from attentive_snake.config import load_experiment
from attentive_snake.run_dir import config_path, list_runs
from attentive_snake.run_metadata import clean_description

DESCRIPTION_LENGTH = 2000


class NewRun(BaseModel):
    name: str
    config: dict[str, dict[str, Any]] = Field(description="experiment tables: {section: {field: value}}; game = config.game.name")
    description: str | None = Field(default=None, max_length=DESCRIPTION_LENGTH, description="saved in run.json")


class Resume(BaseModel):
    total_env_steps: int | None = Field(default=None, ge=1, description="train until this many env steps")
    keep_checkpoints: bool | None = Field(default=None, description="change train.keep_checkpoints (None: keep it)")
    description: str | None = Field(default=None, max_length=DESCRIPTION_LENGTH, description="saved in run.json")
    evaluation_interval: int | None = Field(default=None, ge=0, description="change evaluation.interval (0: off; None: keep it)")
    evaluation_games: int | None = Field(default=None, ge=1, description="change evaluation.games (None: keep it)")
    evaluation_seed: int | None = Field(default=None, description="change evaluation.seed (None: keep it)")

    def evaluation_changes(self) -> dict[str, int]:
        """The changed ``[evaluation]`` settings, ``{"interval" | "games" | "seed": value}``."""
        values = {"interval": self.evaluation_interval, "games": self.evaluation_games, "seed": self.evaluation_seed}
        return {key: value for key, value in values.items() if value is not None}


class Fork(BaseModel):
    name: str = Field(description="name of the new run (of the same game)")
    checkpoint: str | None = Field(default=None, description="checkpoint file of the parent (None: its latest)")
    config: dict[str, dict[str, Any]] = Field(description="experiment tables of the new run")
    description: str | None = Field(default=None, max_length=DESCRIPTION_LENGTH, description="saved in run.json")


def make_router(paths: AppPaths, jobs: TrainingJobs) -> APIRouter:
    router = APIRouter(prefix="/api/runs")

    def status(game: str, name: str) -> dict:
        return jobs.status(job_key(game, name)) or {"state": "idle"}

    def summary(game: str, name: str) -> dict:
        run_dir = game_runs(paths.runs_dir, game) / name
        checkpoints = run_info.checkpoints(run_dir)
        metrics = run_info.metrics(run_dir)
        history = run_info.history(run_dir)
        checkpoint_steps = checkpoints[-1]["steps"] if checkpoints else 0
        logged_steps = int(metrics[-1].get("env_steps", 0)) if metrics else 0
        return {
            "name": name,
            "status": status(game, name),
            "checkpoint_steps": checkpoint_steps,
            "env_steps": max(checkpoint_steps, logged_steps),
            **run_info.config_summary(run_dir),
            "game": game,
            "latest_metrics": metrics[-1] if metrics else None,
            "description": history["description"],
            "parent": history["parent"],
        }

    @router.get("")
    def list_all(game: GameParam = DEFAULT_GAME) -> list[dict]:
        return [summary(game, name) for name in list_runs(game_runs(paths.runs_dir, game))]

    @router.get("/{name}")
    def detail(name: str, game: GameParam = DEFAULT_GAME, log_lines: int = Query(200, ge=1, le=5000)) -> dict:
        run_dir = existing_run_dir(paths.runs_dir, game, name)
        return {
            **summary(game, run_dir.name),
            "config": experiment_to_json(load_experiment(config_path(run_dir))),
            "checkpoints": run_info.checkpoints(run_dir),
            "metrics": run_info.metrics(run_dir),
            "log": run_info.log_tail(run_dir, log_lines),
            "history": run_info.history(run_dir)["events"],
            "evaluations": recorded_evaluations(run_dir),
        }

    @router.post("", status_code=201)
    def start(request: NewRun) -> dict:
        config = parse_experiment(request.config)
        run_dir = new_run_dir(paths.runs_dir, config.game_name, request.name)
        job = start_new_run(paths, jobs, run_dir, config, clean_description(request.description))
        return {"name": run_dir.name, "game": config.game_name, "status": job}

    @router.post("/{name}/resume")
    def resume(name: str, request: Resume, game: GameParam = DEFAULT_GAME) -> dict:
        run_dir = existing_run_dir(paths.runs_dir, game, name)
        description = clean_description(request.description)
        job = resume_run(
            paths, jobs, run_dir, request.total_env_steps, request.keep_checkpoints, description, request.evaluation_changes()
        )
        return {"name": run_dir.name, "game": game, "status": job}

    @router.post("/{name}/fork", status_code=201)
    def fork(name: str, request: Fork, game: GameParam = DEFAULT_GAME) -> dict:
        parent_dir = existing_run_dir(paths.runs_dir, game, name)
        child_dir = new_run_dir(paths.runs_dir, game, request.name)
        config = parse_experiment(request.config)
        require_fork_compatible(parent_dir, config)
        if request.checkpoint is None:
            checkpoint = latest_checkpoint_in_run(parent_dir)
        else:
            checkpoint = checkpoint_in_run(parent_dir, request.checkpoint)
        description = clean_description(request.description)
        job = fork_run(paths, jobs, checkpoint, child_dir, config, description)
        return {"name": child_dir.name, "game": game, "status": job}

    @router.post("/{name}/stop")
    def stop(name: str, game: GameParam = DEFAULT_GAME) -> dict:
        run_dir = existing_run_dir(paths.runs_dir, game, name)
        try:
            return {"name": run_dir.name, "game": game, "status": jobs.stop(job_key(game, run_dir.name))}
        except LookupError as error:
            raise conflict(str(error)) from error

    return router
