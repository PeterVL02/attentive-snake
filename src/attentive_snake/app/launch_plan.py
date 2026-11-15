"""From a launch request to a running job: validate it against the runs on disk, then start it.

``plan_launch`` is every check that can be made without side effects; a request that passes it
now can still fail later (a parent's checkpoint may be deleted while the request waits in the
queue), so ``start_launch`` plans again at the moment it starts. Errors are the API's:
``HTTPException`` 404 (unknown game, run or checkpoint), 409 (the run exists, is training, or the
parent has no checkpoint) and 422 (invalid name or config).
"""

from dataclasses import dataclass
from pathlib import Path

from attentive_snake.app.errors import conflict, invalid_input_as_422
from attentive_snake.app.jobs import TrainingJobs
from attentive_snake.app.launch import fork_run, parse_experiment, require_fork_compatible, resume_run, start_new_run
from attentive_snake.app.launch_request import LaunchRequest
from attentive_snake.app.paths import AppPaths
from attentive_snake.app.run_lookup import checkpoint_in_run, existing_run_dir, latest_checkpoint_in_run, new_run_dir
from attentive_snake.config import ExperimentConfig


@dataclass(frozen=True)
class LaunchPlan:
    """A validated request with its names resolved: ``config`` for new and fork, ``checkpoint`` (the parent's) for fork."""

    request: LaunchRequest
    run_dir: Path
    config: ExperimentConfig | None = None
    checkpoint: Path | None = None


@dataclass(frozen=True)
class LaunchedRun:
    """A started job: the run's directory and the job's status as ``TrainingJobs`` describes it."""

    run_dir: Path
    status: dict


def plan_launch(paths: AppPaths, request: LaunchRequest) -> LaunchPlan:
    """The plan of ``request`` if it could start right now; nothing is created or changed."""
    if request.kind == "resume":
        return LaunchPlan(request, existing_run_dir(paths.runs_dir, request.game, request.run))
    config = parse_experiment(request.config or {})
    if request.kind == "new":
        with invalid_input_as_422():
            if config.game_name != request.game:
                raise ValueError(f"the config is for {config.game_name!r}, the request for {request.game!r}.")
        return LaunchPlan(request, _free_run_dir(paths, request), config)
    parent_dir = existing_run_dir(paths.runs_dir, request.game, request.parent or "")
    run_dir = new_run_dir(paths.runs_dir, request.game, request.run)
    require_fork_compatible(parent_dir, config)
    if request.checkpoint is None:
        checkpoint = latest_checkpoint_in_run(parent_dir)
    else:
        checkpoint = checkpoint_in_run(parent_dir, request.checkpoint)
    return LaunchPlan(request, _free_run_dir(paths, request, run_dir), config, checkpoint)


def start_launch(paths: AppPaths, jobs: TrainingJobs, request: LaunchRequest) -> LaunchedRun:
    """Validate ``request`` and start its training job."""
    plan = plan_launch(paths, request)
    description = request.description
    if plan.config is None:
        status = resume_run(
            paths, jobs, plan.run_dir, request.total_env_steps, request.keep_checkpoints, description, request.evaluation
        )
    elif plan.checkpoint is None:
        status = start_new_run(paths, jobs, plan.run_dir, plan.config, description)
    else:
        status = fork_run(paths, jobs, plan.checkpoint, plan.run_dir, plan.config, description)
    return LaunchedRun(plan.run_dir, status)


def _free_run_dir(paths: AppPaths, request: LaunchRequest, run_dir: Path | None = None) -> Path:
    """Where the new run goes; 409 if something is there already (404 unknown game, 422 bad name)."""
    run_dir = run_dir or new_run_dir(paths.runs_dir, request.game, request.run)
    if run_dir.exists():
        raise conflict(f"run {run_dir.name!r} already exists; choose another name, or resume it.")
    return run_dir
