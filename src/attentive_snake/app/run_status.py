"""What a run is doing now: the app's own job for it, or a training process the app did not start.

The state is a ``JobState``, ``"idle"`` (no job and no process), or ``"external"`` (a process
started elsewhere holds the run's lock, with its ``pid``). The app neither stops nor resumes
an external run.
"""

from pathlib import Path

from attentive_snake.app.jobs import TrainingJobs
from attentive_snake.app.run_lookup import job_key
from attentive_snake.run_dir import game_runs_dir
from attentive_snake.run_lock import lock_holder

TRAINING_STATES = ("running", "external")


def run_status(runs_dir: Path, jobs: TrainingJobs, game: str, name: str) -> dict:
    """The status of run ``name`` of ``game``; its ``state`` is one of ``TRAINING_STATES`` while anything trains it."""
    status = jobs.status(job_key(game, name))
    if status is not None and status["state"] == "running":
        return status
    holder = lock_holder(game_runs_dir(runs_dir, game) / name)
    if holder is not None:
        return {"state": "external", "pid": holder}
    return status or {"state": "idle"}
