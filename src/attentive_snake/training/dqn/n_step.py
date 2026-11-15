"""n-step returns read from a ring of one-step transitions: how far a sampled transition's target looks ahead.

The replay storage keeps what really happened, one step per slot, written one vector step
(``stride`` = one slot per env) after another. A step of the same env is therefore always
``stride`` slots further on (modulo the ring), whatever the capacity. ``n_step_windows``
walks that way from each sampled slot for up to ``n`` steps and stops early

* after a step that ended its episode (the next slot of that env is another episode), and
* at the newest step stored for that env (``later_steps``; nothing newer exists yet, and a slot
  beyond it holds the env's oldest data, not its future).

A window of ``k`` steps gives the return ``r_0 + gamma r_1 + ... + gamma^(k-1) r_(k-1)``, the
slot of its last step (whose next observation and done flag the target bootstraps from) and
the discount ``gamma^k`` of that bootstrap. ``n = 1`` is the one-step transition itself.
"""

from dataclasses import dataclass

import numpy as np


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
    capacity = len(rewards)
    returns = rewards[starts].astype(np.float64)
    last = np.array(starts, dtype=np.int64)
    steps = np.ones(len(last), dtype=np.int64)
    going = np.ones(len(last), dtype=bool)
    for k in range(1, horizon.n):
        going &= ~ends[last] & (later_steps >= k)
        slots = (starts + k * stride) % capacity
        returns += np.where(going, horizon.gamma**k * rewards[slots], 0.0)
        last = np.where(going, slots, last)
        steps += going
    return Windows(returns.astype(np.float32), last, np.power(horizon.gamma, steps).astype(np.float32))
