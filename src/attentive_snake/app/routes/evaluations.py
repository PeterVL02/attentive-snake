"""Evaluating a checkpoint of a run on demand: start it in the background (one at a time per run) and poll it.

``POST /api/runs/{name}/evaluate?game=`` starts, ``GET`` the same path reports
``{"state": "idle" | "running" | "finished" | "failed", "progress": {"done", "total"},
"result": {"evaluation", "reproduces", "saved"}, ...}``. Recorded evaluations are part of
the run detail (``/api/runs/{name}``).
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from attentive_snake.app.background import BackgroundJobs, JobBusy
from attentive_snake.app.errors import conflict, invalid_input_as_422
from attentive_snake.app.evaluation_info import evaluation_settings, evaluation_work
from attentive_snake.app.game_lookup import DEFAULT_GAME, GameParam
from attentive_snake.app.paths import AppPaths
from attentive_snake.app.run_lookup import checkpoint_in_run, existing_run_dir, job_key, latest_checkpoint_in_run

MAX_GAMES = 10_000


class EvaluateRequest(BaseModel):
    checkpoint: str | None = Field(default=None, description="checkpoint file name; default: the latest")
    games: int | None = Field(default=None, ge=1, le=MAX_GAMES, description="default: the run's evaluation.games")
    seed: int | None = Field(default=None, description="default: the run's evaluation.seed")
    max_episode_steps: int | None = Field(default=None, ge=1, description="default: the run's evaluation.max_episode_steps")
    save: bool = Field(default=False, description="append the result to the run's evaluations.jsonl (as a manual evaluation)")


def make_router(paths: AppPaths, jobs: BackgroundJobs) -> APIRouter:
    router = APIRouter(prefix="/api/runs")

    @router.post("/{name}/evaluate", status_code=202)
    def start(name: str, request: EvaluateRequest, game: GameParam = DEFAULT_GAME) -> dict:
        run_dir = existing_run_dir(paths.runs_dir, game, name)
        if request.checkpoint is None:
            checkpoint = latest_checkpoint_in_run(run_dir)
        else:
            checkpoint = checkpoint_in_run(run_dir, request.checkpoint)
        with invalid_input_as_422():
            settings = evaluation_settings(
                run_dir, games=request.games, seed=request.seed, max_episode_steps=request.max_episode_steps
            )
        described = {"checkpoint": checkpoint.name, **settings.to_json(), "save": request.save}
        try:
            return jobs.start(job_key(game, run_dir.name), evaluation_work(run_dir, checkpoint, settings, request.save), described, settings.games)
        except JobBusy as error:
            raise conflict(f"an evaluation of {run_dir.name!r} is already running; wait for it to finish. ({error})") from error

    @router.get("/{name}/evaluate")
    def status(name: str, game: GameParam = DEFAULT_GAME) -> dict:
        run_dir = existing_run_dir(paths.runs_dir, game, name)
        return jobs.status(job_key(game, run_dir.name)) or {"state": "idle"}

    return router
