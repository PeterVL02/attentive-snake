"""The reward search's ``TrainingBackend`` in the app: member runs go through the training queue, answers come from disk and jobs.

So the owner's "runs at once" setting governs member runs like any other queued run, and a
member is an ordinary run of its game (Runs tab). ``progress`` looks at the queue, then at the
jobs, then at the run directory, in that order: the queue keeps an entry reserved until its job
runs, so no run is ever seen as missing in between.
"""

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from attentive_snake.app import launch_request
from attentive_snake.app.jobs import TrainingJobs
from attentive_snake.app.paths import AppPaths
from attentive_snake.app.run_lookup import job_key
from attentive_snake.app.run_status import TRAINING_STATES, run_status
from attentive_snake.app.training_launcher import TrainingLauncher
from attentive_snake.config import load_experiment
from attentive_snake.config.experiment_io import experiment_to_tables
from attentive_snake.evaluation.log import read_evaluations
from attentive_snake.reward_search.backend import FinalEvaluation, LaunchRefused, RunProgress
from attentive_snake.run_dir import config_path, evaluations_path, game_runs_dir, is_run_dir


class QueueBackend:
    def __init__(self, paths: AppPaths, jobs: TrainingJobs, launcher: TrainingLauncher) -> None:
        self._paths = paths
        self._jobs = jobs
        self._launcher = launcher

    def run_tables(self, game: str, run: str) -> dict[str, dict[str, Any]]:
        run_dir = self._run_dir(game, run)
        if not is_run_dir(run_dir):
            raise LookupError(f"no {game} run named {run!r}.")
        return experiment_to_tables(load_experiment(config_path(run_dir)))

    def enqueue_fork(
        self, game: str, parent: str, checkpoint: str | None, run: str, tables: dict[str, dict[str, Any]], description: str
    ) -> None:
        self._enqueue(launch_request.fork(game, parent, run, tables, checkpoint, description))

    def enqueue_resume(self, game: str, run: str) -> None:
        self._enqueue(launch_request.resume(game, run))

    def progress(self, game: str, run: str) -> RunProgress:
        if self._launcher.queue.is_reserved(game, run):
            return "waiting"
        if run_status(self._paths.runs_dir, self._jobs, game, run)["state"] in TRAINING_STATES:
            return "training"
        return "ended" if is_run_dir(self._run_dir(game, run)) else "missing"

    def last_training_evaluation(self, game: str, run: str) -> FinalEvaluation | None:
        made_while_training = [e for e in read_evaluations(evaluations_path(self._run_dir(game, run))) if e.source == "training"]
        if not made_while_training:
            return None
        last = made_while_training[-1]
        return FinalEvaluation(last.env_steps, last.checkpoint, dict(last.statistics))

    def cancel(self, game: str, run: str) -> None:
        queue = self._launcher.queue
        for entry in queue.entries():
            if (entry.request.game, entry.request.run) == (game, run) and entry.state != "starting":
                try:
                    queue.remove(entry.id)
                except LookupError:  # started or removed meanwhile
                    pass
        try:
            self._jobs.stop(job_key(game, run))
        except LookupError:  # no running job
            pass

    def _run_dir(self, game: str, run: str) -> Path:
        return game_runs_dir(self._paths.runs_dir, game) / run

    def _enqueue(self, request: launch_request.LaunchRequest) -> None:
        try:
            self._launcher.enqueue(request)
        except HTTPException as error:
            raise LaunchRefused(str(error.detail)) from error
