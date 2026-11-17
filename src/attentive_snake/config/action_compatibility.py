"""What a game with continuous actions cannot be trained with, checked from the config alone (no env is built).

Three parts of an experiment need a discrete action set: an algorithm without a head for
continuous actions (DQN: one value per action), ``env.history_actions`` (one one-hot channel per
action) and the model kinds marked ``KindInfo.needs_discrete_actions`` (the world models: their
predictor is conditioned on one action index). ``ExperimentConfig`` refuses such a combination
when it is built, so the app answers 422 and the CLI stops before training starts.
"""

from typing import Any

from attentive_snake.games.registry import get_game
from attentive_snake.models.registry import KIND_INFO
from attentive_snake.training.algorithms import get_algorithm


def continuous_action_problems(game_name: str, rules: Any, algorithm: str, model_kind: str, history_actions: bool) -> list[str]:
    """Plain-language reasons why this combination cannot train (empty if it can, or if the game's actions are discrete)."""
    game = get_game(game_name)
    if not game.has_continuous_actions(rules):
        return []
    where = f"{game.title} with these rules has continuous actions (a vector of numbers, not one of a few choices)"
    problems = []
    trainer = get_algorithm(algorithm)
    if not trainer.trains_continuous_actions:
        problems.append(f"{where}, but {trainer.label} learns one value per action and needs a discrete action set: train it with an algorithm for continuous actions (PPO), or turn continuous actions off in [game].")
    if history_actions:
        problems.append(f"{where}, but env.history_actions shows the model one channel per action and needs a discrete action set: turn history_actions off.")
    if model_kind in KIND_INFO and KIND_INFO[model_kind].needs_discrete_actions:
        problems.append(f"{where}, but the {model_kind} model predicts where each of a few actions leads and needs a discrete action set: choose another model kind.")
    return problems


def require_compatible_actions(game_name: str, rules: Any, algorithm: str, model_kind: str, history_actions: bool) -> None:
    """``ValueError`` naming every problem of ``continuous_action_problems``."""
    problems = continuous_action_problems(game_name, rules, algorithm, model_kind, history_actions)
    if problems:
        raise ValueError(" ".join(problems))
