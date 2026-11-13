"""Runs: list, detail, start, resume and stop."""

from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from attentive_snake.app.config_json import experiment_to_json
from attentive_snake.app.errors import conflict
from attentive_snake.app.jobs import TrainingJobs
from attentive_snake.app.launch import parse_experiment, resume_run, start_new_run
from attentive_snake.app.paths import AppPaths
from attentive_snake.app import run_info
from attentive_snake.app.run_lookup import existing_run_dir, run_name
from attentive_snake.config import load_experiment
from attentive_snake.run_dir import config_path, list_runs


class NewRun(BaseModel):
    name: str
    config: dict[str, dict[str, Any]] = Field(description="experiment tables: {section: {field: value}}")


class Resume(BaseModel):
    total_env_steps: int | None = Field(default=None, ge=1, description="train until this many env steps")


def make_router(paths: AppPaths, jobs: TrainingJobs) -> APIRouter:
    router = APIRouter(prefix="/api/runs")

    def status(name: str) -> dict:
        return jobs.status(name) or {"state": "idle"}

    def summary(name: str) -> dict:
        run_dir = paths.runs_dir / name
        checkpoints = run_info.checkpoints(run_dir)
        metrics = run_info.metrics(run_dir)
        checkpoint_steps = checkpoints[-1]["steps"] if checkpoints else 0
        logged_steps = int(metrics[-1].get("env_steps", 0)) if metrics else 0
        return {
            "name": name,
            "status": status(name),
            "checkpoint_steps": checkpoint_steps,
            "env_steps": max(checkpoint_steps, logged_steps),
            **run_info.config_summary(run_dir),
            "latest_metrics": metrics[-1] if metrics else None,
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
        }

    @router.post("", status_code=201)
    def start(request: NewRun) -> dict:
        name = run_name(request.name)
        config = parse_experiment(request.config)
        job = start_new_run(paths, jobs, paths.runs_dir / name, config)
        return {"name": name, "status": job}

    @router.post("/{name}/resume")
    def resume(name: str, request: Resume) -> dict:
        run_dir = existing_run_dir(paths.runs_dir, name)
        return {"name": run_dir.name, "status": resume_run(paths, jobs, run_dir, request.total_env_steps)}

    @router.post("/{name}/stop")
    def stop(name: str) -> dict:
        run_dir = existing_run_dir(paths.runs_dir, name)
        try:
            return {"name": run_dir.name, "status": jobs.stop(run_dir.name)}
        except LookupError as error:
            raise conflict(str(error)) from error

    return router

