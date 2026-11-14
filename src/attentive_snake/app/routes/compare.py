"""Comparing runs of one game: metrics, evaluations and config of several runs in one call, plus the config fields that differ."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from attentive_snake.app import run_info
from attentive_snake.app.config_diff import differing_fields
from attentive_snake.app.evaluation_info import recorded_evaluations
from attentive_snake.app.jobs import TrainingJobs
from attentive_snake.app.parameter_count import run_parameter_count
from attentive_snake.app.paths import AppPaths
from attentive_snake.app.game_lookup import DEFAULT_GAME
from attentive_snake.app.run_lookup import existing_run_dir, job_key
from attentive_snake.run_dir import config_path

MAX_RUNS = 12


class Comparison(BaseModel):
    game: str = Field(default=DEFAULT_GAME, description="every run compared is a run of this game")
    runs: list[str] = Field(min_length=1, max_length=MAX_RUNS)


def make_router(paths: AppPaths, jobs: TrainingJobs) -> APIRouter:
    router = APIRouter(prefix="/api")

    def run_entry(game: str, name: str) -> dict:
        run_dir = existing_run_dir(paths.runs_dir, game, name)
        return {
            "name": run_dir.name,
            "status": jobs.status(job_key(game, run_dir.name)) or {"state": "idle"},
            **run_info.config_summary(run_dir),
            "config": run_info.config_json(run_dir),  # None if unreadable: see config_error
            "parameters": run_parameter_count(config_path(run_dir)),
            "metrics": run_info.metrics(run_dir),
            "evaluations": recorded_evaluations(run_dir),
            "description": run_info.history(run_dir)["description"],
        }

    @router.post("/compare")
    def compare(request: Comparison) -> dict:
        runs = [run_entry(request.game, name) for name in dict.fromkeys(request.runs)]
        configs = {run["name"]: run["config"] for run in runs if run["config"] is not None}
        return {"game": request.game, "runs": runs, "differences": differing_fields(configs)}

    return router
