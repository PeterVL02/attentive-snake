"""Bootstrapping truncated episodes: the reward of a step that hit a limit gets ``gamma * V(final observation)``.

A truncated episode did not end by the game's rules; it would have gone on. Adding the
discounted value of its last observation to the last reward, while the step stays
``done``, gives ``compute_gae`` exactly the target of a bootstrapped transition
(``r + gamma * V(s')``) without changing GAE or the loss. The value comes from the final
observation in ``VecEnvStep.final_observations``, never from the auto-reset one.
"""

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from attentive_snake.env.types import VecEnvStep

ValueFunction = Callable[[Tensor], Tensor]  # (B, *obs_shape) -> (B,)


def truncated_envs(step: VecEnvStep) -> list[int]:
    """Indices of the envs whose episode was cut by a limit in ``step`` (a rules ending wins over a cut)."""
    indices = np.flatnonzero(np.logical_and(step.truncated, np.logical_not(step.terminated)))
    return [int(i) for i in indices]


class TruncationBootstrap:
    """Adds ``gamma * V(final observation)`` to the rewards of truncated steps; other rewards pass unchanged.

    The caller supplies the value function, so it is evaluated in the caller's mode (no
    grad, the policy's collection mode); all truncated envs of a step share one forward pass.
    """

    def __init__(self, gamma: float) -> None:
        self._gamma = gamma

    @torch.no_grad()
    def rewards(self, rewards: Tensor, step: VecEnvStep, value_of: ValueFunction) -> Tensor:
        """``rewards`` (N,) of ``step`` with the bootstrap added for its truncated envs (unchanged if there are none)."""
        indices = truncated_envs(step)
        if not indices:
            return rewards
        missing = [i for i in indices if i not in step.final_observations]
        if missing:
            raise ValueError(f"truncated envs {missing} have no final observation in the VecEnvStep.")
        finals = np.stack([step.final_observations[i] for i in indices]).astype(np.float32, copy=False)
        values = value_of(torch.as_tensor(finals, device=rewards.device))
        corrected = rewards.clone()
        corrected[torch.as_tensor(indices, device=rewards.device)] += self._gamma * values.to(rewards.dtype)
        return corrected
