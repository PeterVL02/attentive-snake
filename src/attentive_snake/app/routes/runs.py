"""Runs of a game: list, detail, start, resume, fork (continue under a new name with changes) and stop.

Runs are addressed by game and name: ``?game=`` (default ``snake``) on every route with a
run name in the path and on the list. A new run's game is its config's ``[game] name``.
"""

from fastapi import APIRouter, Query

from attentive_snake.app import run_info
from attentive_snake.app.errors import conflict
from attentive_snake.app.evaluation_info import recorded_evaluations
from attentive_snake.app.game_lookup import DEFAULT_GAME, GameParam
from attentive_snake.app.jobs import TrainingJobs
from attentive_snake.app.launch_bodies import Fork, NewRun, Resume
from attentive_snake.app.launch_request import LaunchRequest
from attentive_snake.app.paths import AppPaths
from attentive_snake.app.run_lookup import existing_run_dir, game_runs, job_key
from attentive_snake.app.training_launcher import TrainingLauncher
from attentive_snake.run_dir import list_runs


def make_router(paths: AppPaths, jobs: TrainingJobs, launcher: TrainingLauncher) -> APIRouter:
    router = APIRouter(prefix="/api/runs")

    def started(request: LaunchRequest) -> dict:
        """Start ``request`` now and describe the job."""
        launched = launcher.start(request)
        return {"name": launched.run_dir.name, "game": request.game, "status": launched.status}

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
            "config": run_info.config_json(run_dir),  # None if unreadable: see config_error
            "checkpoints": run_info.checkpoints(run_dir),
            "metrics": run_info.metrics(run_dir),
            "log": run_info.log_tail(run_dir, log_lines),
            "history": run_info.history(run_dir)["events"],
            "evaluations": recorded_evaluations(run_dir),
        }

    @router.post("", status_code=201)
    def start(request: NewRun) -> dict:
        return started(request.launch())

    @router.post("/{name}/resume")
    def resume(name: str, request: Resume, game: GameParam = DEFAULT_GAME) -> dict:
        return started(request.launch(game, name))

    @router.post("/{name}/fork", status_code=201)
    def fork(name: str, request: Fork, game: GameParam = DEFAULT_GAME) -> dict:
        return started(request.launch(game, name))

    @router.post("/{name}/stop")
    def stop(name: str, game: GameParam = DEFAULT_GAME) -> dict:
        run_dir = existing_run_dir(paths.runs_dir, game, name)
        try:
            return {"name": run_dir.name, "game": game, "status": jobs.stop(job_key(game, run_dir.name))}
        except LookupError as error:
            raise conflict(str(error)) from error

    return router
