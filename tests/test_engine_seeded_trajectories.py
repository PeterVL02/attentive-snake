"""Pins the engine's random-number use: with default rules a seed gives exactly these games.

The digests were recorded before apple placement was made configurable; a change here means
existing seeds no longer give the same games.
"""

import hashlib
import random

import pytest

from attentive_snake.games.snake.config import SnakeConfig
from attentive_snake.games.snake.engine import SnakeGame
from attentive_snake.games.snake.types import Action


def _digest(config: SnakeConfig, seed: int, steps: int = 600) -> str:
    game = SnakeGame(config, seed=seed)
    chooser = random.Random(seed + 1000)  # actions come from their own stream
    record = [repr(game.state.food)]
    for _ in range(steps):
        result = game.step(Action(chooser.randrange(4)))
        record.append(f"{result.event.value}:{result.state.food!r}")
        if result.state.done:
            record.append(repr(game.reset().food))
    return hashlib.sha256("|".join(record).encode()).hexdigest()[:16]


CASES = [
    (SnakeConfig(width=8, height=8, max_steps_without_food=128), 0, "7a96fccd1c0585ee"),
    (SnakeConfig(width=8, height=8, max_steps_without_food=128), 7, "bbf2978f424d513d"),
    (SnakeConfig(width=10, height=10), 3, "c7e0ab393c94da32"),
    (SnakeConfig(width=4, height=3, initial_length=2, max_steps_without_food=40), 5, "fdac31de7ff3a50c"),
]


@pytest.mark.parametrize("config,seed,expected", CASES)
def test_seeded_games_are_unchanged(config, seed, expected):
    assert _digest(config, seed) == expected
