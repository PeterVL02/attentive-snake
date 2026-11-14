"""Runs: list, detail, start, resume, fork (continue under a new name with changes) and stop."""

from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from attentive_snake.app.config_json import experiment_to_json
from attentive_snake.app.errors import conflict
from attentive_snake.app.jobs import TrainingJobs
from attentive_snake.app.launch import fork_run, parse_experiment, require_fork_compatible, resume_run, start_new_run
from attentive_snake.app.paths import AppPaths
from attentive_snake.app import run_info
from attentive_snake.app.run_lookup import checkpoint_in_run, existing_run_dir, latest_checkpoint_in_run, run_name
from attentive_snake.config import load_experiment
from attentive_snake.run_dir import config_path, list_runs
from attentive_snake.run_metadata import clean_description

DESCRIPTION_LENGTH = 2000


class NewRun(BaseModel):
    name: str
    config: dict[str, dict[str, Any]] = Field(description="experiment tables: {section: {field: value}}")
    description: str | None = Field(default=None, max_length=DESCRIPTION_LENGTH, description="saved in run.json")


class Resume(BaseModel):
    total_env_steps: int | None = Field(default=None, ge=1, description="train until this many env steps")
    keep_checkpoints: bool | None = Field(default=None, description="change train.keep_checkpoints (None: keep it)")
    description: str | None = Field(default=None, max_length=DESCRIPTION_LENGTH, description="saved in run.json")


class Fork(BaseModel):
    name: str = Field(description="name of the new run")
    checkpoint: str | None = Field(default=None, description="checkpoint file of the parent (None: its latest)")
    config: dict[str, dict[str, Any]] = Field(description="experiment tables of the new run")
    description: str | None = Field(default=None, max_length=DESCRIPTION_LENGTH, description="saved in run.json")


def make_router(paths: AppPaths, jobs: TrainingJobs) -> APIRouter:
    router = APIRouter(prefix="/api/runs")

    def status(name: str) -> dict:
        return jobs.status(name) or {"state": "idle"}

    def summary(name: str) -> dict:
        run_dir = paths.runs_dir / name
        checkpoints = run_info.checkpoints(run_dir)
        metrics = run_info.metrics(run_dir)
        history = run_info.history(run_dir)
        checkpoint_steps = checkpoints[-1]["steps"] if checkpoints else 0
        logged_steps = int(metrics[-1].get("env_steps", 0)) if metrics else 0
        return {
            "name": name,
            "status": status(name),
            "checkpoint_steps": checkpoint_steps,
            "env_steps": max(checkpoint_steps, logged_steps),
            **run_info.config_summary(run_dir),
            "latest_metrics": metrics[-1] if metrics else None,
            "description": history["description"],
            "parent": history["parent"],
        }

    @router.get("")
    def list_all() -> list[dict]:
        if not paths.runs_dir.is_dir():
            return []
        return [summary(name) for name in list_runs(paths.runs_dir)]

    @router.get("/{name}")
    def detail(name: str, log_lines: int = Query(200, ge=1, le=5000)) -> dict:
        run_dir = existing_run_dir(paths.runs_dir, name)
        return {
            **summary(run_dir.name),
            "config": experiment_to_json(load_experiment(config_path(run_dir))),
            "checkpoints": run_info.checkpoints(run_dir),
            "metrics": run_info.metrics(run_dir),
            "log": run_info.log_tail(run_dir, log_lines),
            "history": run_info.history(run_dir)["events"],
        }

    @router.post("", status_code=201)
    def start(request: NewRun) -> dict:
        name = run_name(request.name)
        config = parse_experiment(request.config)
        job = start_new_run(paths, jobs, paths.runs_dir / name, config, clean_description(request.description))
        return {"name": name, "status": job}

    @router.post("/{name}/resume")
    def resume(name: str, request: Resume) -> dict:
        run_dir = existing_run_dir(paths.runs_dir, name)
        description = clean_description(request.description)
        job = resume_run(paths, jobs, run_dir, request.total_env_steps, request.keep_checkpoints, description)
        return {"name": run_dir.name, "status": job}

    @router.post("/{name}/fork", status_code=201)
    def fork(name: str, request: Fork) -> dict:
        parent_dir = existing_run_dir(paths.runs_dir, name)
        new_name = run_name(request.name)
        config = parse_experiment(request.config)
        require_fork_compatible(parent_dir, config)
        if request.checkpoint is None:
            checkpoint = latest_checkpoint_in_run(parent_dir)
        else:
            checkpoint = checkpoint_in_run(parent_dir, request.checkpoint)
        description = clean_description(request.description)
        job = fork_run(paths, jobs, checkpoint, paths.runs_dir / new_name, config, description)
        return {"name": new_name, "status": job}

    @router.post("/{name}/stop")
    def stop(name: str) -> dict:
        run_dir = existing_run_dir(paths.runs_dir, name)
        try:
            return {"name": run_dir.name, "status": jobs.stop(run_dir.name)}
        except LookupError as error:
            raise conflict(str(error)) from error

    return router

