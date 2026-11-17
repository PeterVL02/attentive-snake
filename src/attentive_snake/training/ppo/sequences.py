"""The steps that followed every step of a PPO rollout, for an auxiliary objective that reads several in a row.

A rollout flattened over (steps, envs) keeps one slot per env per step, in env order, so a
later step of an env is ``num_envs`` slots on (``training.slot_sequences``). A sequence stops
at an episode end and at the end of the rollout: what follows the last collected step is not
in the buffer.
"""

import numpy as np
import torch

from attentive_snake.models.auxiliary import TransitionSequences
from attentive_snake.training.ppo.rollout import RolloutBuffer
from attentive_snake.training.slot_sequences import slot_sequences


def rollout_sequences(buffer: RolloutBuffer, length: int) -> TransitionSequences:
    """``length`` steps from each step of the rollout on, one row per flattened step (as ``flatten_rollout`` orders them).

    The buffer must have kept its next observations. ``terminal`` is ``dones``: PPO bootstraps
    from no step that ended its episode.
    """
    if buffer.next_observations is None:
        raise ValueError("sequences need a rollout that kept its next observations.")
    dones = buffer.dones.flatten()
    steps, envs = buffer.rollout_length, buffer.num_envs
    starts = np.arange(steps * envs)
    later_steps = steps - 1 - starts // envs
    found = slot_sequences(dones.cpu().numpy() > 0.5, starts, later_steps, envs, length)
    slots = torch.as_tensor(found.slots, device=dones.device)
    return TransitionSequences(
        actions=buffer.actions.flatten(0, 1)[slots],
        next_observations=buffer.next_observations.flatten(0, 1)[slots],
        valid=torch.as_tensor(found.present.astype(np.float32), device=dones.device),
        terminal=dones[slots],
    )
