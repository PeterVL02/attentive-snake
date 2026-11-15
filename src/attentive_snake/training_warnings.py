"""A game's own warnings about the config a run is about to be trained with (app and CLI)."""

from attentive_snake.config import ExperimentConfig
from attentive_snake.games.registry import get_game


def training_warnings(config: ExperimentConfig) -> tuple[str, ...]:
    """Plain-language warnings from the config's game (none if it has no ``training_warnings``)."""
    check = get_game(config.game_name).training_warnings
    if check is None:
        return ()
    return tuple(check(config.game, config.reward, config.train.gamma))
