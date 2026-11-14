"""Evaluating one checkpoint of a run directory after training (``scripts/evaluate.py`` and the app)."""

from pathlib import Path

from attentive_snake.config import load_experiment
from attentive_snake.evaluation.evaluate import evaluate_policy
from attentive_snake.evaluation.player import Progress
from attentive_snake.evaluation.record import Evaluation
from attentive_snake.evaluation.settings import EvalSettings
from attentive_snake.factory import make_env, make_policy
from attentive_snake.games.registry import get_game
from attentive_snake.run_dir import config_path
from attentive_snake.training.checkpoint import load_checkpoint


def evaluate_checkpoint(run_dir: Path, checkpoint: Path, settings: EvalSettings, progress: Progress | None = None) -> Evaluation:
    """Load the run's config and ``checkpoint``'s weights and evaluate them; the result is a ``"manual"`` evaluation."""
    config = load_experiment(config_path(run_dir))
    policy = make_policy(config, make_env(config))
    env_steps = load_checkpoint(checkpoint, policy, map_location=settings.device)
    policy.to(settings.device)
    return evaluate_policy(
        policy,
        get_game(config.game_name),
        lambda: make_env(config),
        settings,
        source="manual",
        env_steps=env_steps,
        checkpoint=Path(checkpoint).name,
        progress=progress,
    )
