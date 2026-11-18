"""Which games an algorithm's observation normalisation can be used on, checked from the config alone (no env is built).

Running statistics rescale numbers of different sizes. Boards and screens are already on a fixed
scale, and a mean per cell would erase what the cell's position means, so only vector games qualify.
``ExperimentConfig`` refuses anything else when it is built: the app answers 422, the CLI stops before training.
"""

from typing import Any

from attentive_snake.games.registry import get_game
from attentive_snake.training.algorithms import get_algorithm


def require_normalizable_observations(game_name: str, algorithm: str, settings: Any) -> None:
    """``ValueError`` if ``settings`` of ``algorithm`` normalise the observations of a game that is not a vector game."""
    game = get_game(game_name)
    if get_algorithm(algorithm).normalizes_observations(settings) and game.observation_kind != "vector":
        raise ValueError(
            f"normalize_observations only works for games described by a list of numbers, but {game.title} shows the model "
            f"{game.observation_kind} observations, which are already on a fixed scale: turn normalize_observations off."
        )
