"""Where a new apple lands: one choice among the free cells. Pure Python, no engine state."""

import math
import random
from collections.abc import Sequence

from attentive_snake.games.snake.config import SnakeConfig
from attentive_snake.games.snake.types import Position


class FoodPlacer:
    """Picks the cell of a new apple from the free cells with the engine's own random generator.

    ``"uniform"`` draws exactly like ``rng.choice(free)`` (so seeds give the same games as before
    the rule existed); ``"center"`` weights each cell by ``exp(-d^2 / (2 * food_spread^2))``,
    d being its distance to the board's centre point.
    """

    def __init__(self, config: SnakeConfig) -> None:
        self._centred = config.food_distribution == "center"
        cx, cy = (config.width - 1) / 2, (config.height - 1) / 2
        two_variances = 2 * config.food_spread**2
        self._weights = {
            (x, y): math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / two_variances)
            for y in range(config.height)
            for x in range(config.width)
        }

    def choose(self, free: Sequence[Position], rng: random.Random) -> Position:
        """One of ``free`` (not empty)."""
        if not self._centred:
            return rng.choice(free)
        return rng.choices(free, weights=[self._weights[cell] for cell in free])[0]
