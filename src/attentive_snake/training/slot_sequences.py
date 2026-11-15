"""Steps of one env in a row, found in storage that keeps a vector env's steps one vector step after another.

Such storage (DQN's replay ring, PPO's flattened rollout) writes one slot per env per vector
step, in env order. A later step of the same env is therefore always ``stride`` slots (the
number of envs) further on, modulo the storage size. ``slot_sequences`` walks that way from
each starting slot for up to ``length`` steps and stops early

* after a step that ended its episode (the next slot of that env is another episode), and
* at the newest step stored for that env (``later_steps``: nothing newer exists yet, and in a
  ring a slot beyond it holds the env's oldest data, not its future).

Numpy only; the callers decide what they read from the slots (n-step returns: ``dqn/n_step.py``;
the steps a model is unrolled along: ``dqn/replay_storage.py``, ``ppo/sequences.py``).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SlotSequences:
    """One sequence per starting slot. Column 0 is the starting slot itself and always present."""

    slots: np.ndarray  # int64 (B, length): the slot of each step; where a step is not present, the sequence's last present slot
    present: np.ndarray  # bool (B, length): the step exists and belongs to the starting step's episode; once False it stays False


def later_steps_in_ring(starts: np.ndarray, next_slot: int, capacity: int, stride: int) -> np.ndarray:
    """How many later steps of the same env a ring holds after each slot of ``starts``; ``next_slot`` is the slot written next."""
    return ((next_slot - 1 - starts) % capacity) // stride


def slot_sequences(ends: np.ndarray, starts: np.ndarray, later_steps: np.ndarray, stride: int, length: int) -> SlotSequences:
    """The sequences starting at the slots ``starts`` of storage whose slot ``i`` ended its episode where ``ends[i]``.

    ``later_steps[i]`` = how many later steps of the same env the storage holds after ``starts[i]``.
    """
    if length < 1:
        raise ValueError("length must be at least 1.")
    capacity = len(ends)
    slots = np.empty((len(starts), length), dtype=np.int64)
    present = np.ones((len(starts), length), dtype=bool)
    slots[:, 0] = starts
    for k in range(1, length):
        present[:, k] = present[:, k - 1] & ~ends[slots[:, k - 1]] & (later_steps >= k)
        slots[:, k] = np.where(present[:, k], (starts + k * stride) % capacity, slots[:, k - 1])
    return SlotSequences(slots, present)
