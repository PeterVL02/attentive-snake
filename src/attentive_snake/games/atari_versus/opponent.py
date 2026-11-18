"""The fixed opponent of evaluation and watching: a player that presses a uniformly random action every step.

It has its own generator, seeded with the game's seed, so a seeded game is the same game every
time (evaluations are reproducible) and the global random streams are never touched.
"""

import numpy as np


class RandomOpponent:
    """Uniform over ``num_actions`` (indices of the player's action set)."""

    def __init__(self, num_actions: int) -> None:
        self._num_actions = num_actions
        self._rng: np.random.Generator | None = None

    def reset(self, seed: int | None) -> None:
        """Seeded: the same choices again for the same seed. Without a seed: carry on (an OS seed the first time)."""
        if seed is not None or self._rng is None:
            self._rng = np.random.default_rng(seed)

    def act(self) -> int:
        return int(self._rng.integers(self._num_actions))
