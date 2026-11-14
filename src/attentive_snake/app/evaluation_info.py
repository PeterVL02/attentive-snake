"""Evaluations of a run for the UI: the recorded ones (with whether their checkpoint still exists), and running a new one."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from attentive_snake.config import load_experiment
from attentive_snake.evaluation.checkpoint_evaluation import evaluate_checkpoint
from attentive_snake.evaluation.log import append_evaluation, read_evaluations
from attentive_snake.evaluation.matching import find_reproduction
from attentive_snake.evaluation.settings import EvalSettings, settings_from_config
from attentive_snake.run_dir import config_path, evaluations_path, list_checkpoints
from attentive_snake.training.runtime import resolve_device


def recorded_evaluations(run_dir: Path) -> list[dict[str, Any]]:
    """Every evaluation in the run's evaluations.jsonl, oldest first, each with ``checkpoint_available``.

    A checkpoint is gone when ``train.keep_checkpoints = false`` pruned it (or it was deleted
    by hand); such an evaluation can no longer be reproduced.
    """
    present = {path.name for path in list_checkpoints(run_dir)}
    return [
        {**evaluation.to_json(), "checkpoint_available": evaluation.checkpoint in present}
        for evaluation in read_evaluations(evaluations_path(run_dir))
    ]


def evaluation_settings(run_dir: Path, **changes: int | None) -> EvalSettings:
    """The run's ``[evaluation]`` settings on its training device, with the given changes (``None``: keep)."""
    config = load_experiment(config_path(run_dir))
    return settings_from_config(config.evaluation, str(resolve_device(config.train.device)), **changes)


def evaluation_work(run_dir: Path, checkpoint: Path, settings: EvalSettings, save: bool) -> Callable[[Callable[[int, int], None]], dict]:
    """The work of one on-demand evaluation, for ``BackgroundJobs``: evaluate, compare with the records, optionally save."""

    def work(progress: Callable[[int, int], None]) -> dict[str, Any]:
        evaluation = evaluate_checkpoint(run_dir, checkpoint, settings, progress)
        reproduction = find_reproduction(read_evaluations(evaluations_path(run_dir)), evaluation)
        if save:
            append_evaluation(evaluations_path(run_dir), evaluation)
        return {
            "evaluation": evaluation.to_json(),
            "reproduces": None if reproduction is None else reproduction.to_json(),
            "saved": save,
        }

    return work
