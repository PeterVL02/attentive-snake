"""What a game's actions are: one of ``count`` choices (discrete) or a vector of real numbers within bounds (continuous).

Part of the env contract (``env.types``). A ``GameEnv`` whose actions are continuous carries
``action_space: ContinuousActions``; an env without the attribute has ``DiscreteActions(num_actions)``
(every game before continuous actions existed, unchanged). Which one an env has may depend on its
rules (Lunar Lander's ``continuous``), as ``GameEnv.action_labels`` may; ``GameSpec.continuous_actions``
answers the same question from the rules alone, for config validation without building an env.

``num_actions`` keeps one meaning for both: the width of the policy's action output (the number of
choices, or the dimension of the action vector). The names of the actions, or of the dimensions,
stay where they are: ``GameEnv.action_labels`` / ``GameSpec.action_labels``.

Numpy only, like the rest of ``env``. ``env_action`` turns what a policy produced into what the
game's ``step`` takes; ``HistoryEnv`` and ``SyncVecEnv`` call it, so a game never sees an
out-of-bounds continuous action.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DiscreteActions:
    """One of ``count`` actions, chosen by index; ``step`` takes a Python ``int``."""

    count: int
    continuous = False

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of one action as a policy produces it: a scalar index."""
        return ()

    def env_action(self, action: Any) -> int:
        return int(action)


@dataclass(frozen=True)
class ContinuousActions:
    """A vector of ``len(low)`` real numbers, number ``i`` in ``[low[i], high[i]]``; ``step`` takes float32 ``(dimension,)``.

    A policy may produce numbers outside the bounds (a Gaussian has no edges); ``env_action``
    clips them, which is the action the game plays.
    """

    low: tuple[float, ...]
    high: tuple[float, ...]
    continuous = True

    def __post_init__(self) -> None:
        if not self.low or len(self.low) != len(self.high):
            raise ValueError("continuous actions need one low and one high bound per dimension (at least one dimension).")
        if any(lo >= hi for lo, hi in zip(self.low, self.high)):
            raise ValueError("every low bound of continuous actions must be below its high bound.")

    @property
    def dimension(self) -> int:
        return len(self.low)

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.dimension,)

    def env_action(self, action: Any) -> np.ndarray:
        vector = np.asarray(action, dtype=np.float32).reshape(self.shape)
        return np.clip(vector, np.asarray(self.low, dtype=np.float32), np.asarray(self.high, dtype=np.float32))


type ActionSpace = DiscreteActions | ContinuousActions


def action_space_of(env: Any) -> ActionSpace:
    """``env.action_space`` if it has one, else ``DiscreteActions(env.num_actions)``."""
    space = getattr(env, "action_space", None)
    return space if space is not None else DiscreteActions(env.num_actions)
