"""n-step returns read from a ring of one-step transitions: how far a sampled transition's target looks ahead.

The replay storage keeps what really happened, one step per slot. ``n_step_windows`` follows
each sampled slot's env for up to ``n`` steps (``training.slot_sequences``: a later step of an
env is ``stride`` slots on; the walk stops after a step that ended its episode and at the
newest step stored for that env).

A window of ``k`` steps gives the return ``r_0 + gamma r_1 + ... + gamma^(k-1) r_(k-1)``, the
slot of its last step (whose next observation and done flag the target bootstraps from) and
the discount ``gamma^k`` of that bootstrap. ``n = 1`` is the one-step transition itself.
"""

from dataclasses import dataclass

import numpy as np

from attentive_snake.training.slot_sequences import slot_sequences


@dataclass(frozen=True)
class Horizon:
    """How far targets look ahead: ``n`` steps, discounted by ``gamma`` per step."""

    n: int
    gamma: float

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be at least 1.")


@dataclass(frozen=True)
class Windows:
    """One window per sampled slot."""

    returns: np.ndarray  # float32 (B,), the discounted rewards of the window's steps
    last: np.ndarray  # int64 (B,), the slot of the window's last step
    discounts: np.ndarray  # float32 (B,), gamma ** (steps in the window)


def n_step_windows(
    rewards: np.ndarray, ends: np.ndarray, starts: np.ndarray, later_steps: np.ndarray, stride: int, horizon: Horizon
) -> Windows:
    """The windows starting at the slots ``starts`` of a ring holding ``rewards`` and ``ends`` (episode ended with that step).

    ``later_steps[i]`` = how many later steps of the same env the ring holds after ``starts[i]``.
    """
    sequences = slot_sequences(ends, starts, later_steps, stride, horizon.n)
    returns = rewards[starts].astype(np.float64)
    for k in range(1, horizon.n):
        returns += np.where(sequences.present[:, k], horizon.gamma**k * rewards[sequences.slots[:, k]], 0.0)
    steps = sequences.present.sum(axis=1)
    return Windows(returns.astype(np.float32), sequences.slots[:, -1], np.power(horizon.gamma, steps).astype(np.float32))
