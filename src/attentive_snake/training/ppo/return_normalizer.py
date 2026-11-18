"""Scaling PPO's training rewards by the running spread of the discounted return (Gymnasium's ``NormalizeReward``, CleanRL)."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from attentive_snake.models.running_statistics import RunningStatistics

CLIP = 10.0  # scaled rewards are cut to +-CLIP
_EPSILON = 1e-8


class ReturnNormalizer:
    """Divides each reward by the running standard deviation of the discounted return; the mean is not subtracted.

    Every environment slot keeps its discounted return so far, which restarts at an episode's
    end. Only the variance's statistics are saved: a resumed run's environments start fresh
    episodes, so their returns start from 0 as well. Scores and episode summaries never pass
    through here; they stay in the game's own units.
    """

    def __init__(self, num_envs: int, gamma: float, device: torch.device) -> None:
        self._gamma = gamma
        self._returns = torch.zeros(num_envs, dtype=torch.float64, device=device)
        self._statistics = RunningStatistics().to(device)

    @torch.no_grad()
    def __call__(self, rewards: Tensor, dones: Tensor) -> Tensor:
        """The scaled ``rewards`` (N,) of one step whose episode ends are ``dones`` (N,)."""
        raw = rewards.to(self._returns.dtype)
        self._returns = self._gamma * self._returns + raw
        self._statistics.update(self._returns)
        scaled = (raw / (self._statistics.variance + _EPSILON).sqrt()).clamp(-CLIP, CLIP)
        self._returns = self._returns * (1.0 - dones.to(self._returns.dtype))
        return scaled.to(rewards.dtype)

    def state_dict(self) -> dict[str, Any]:
        return dict(self._statistics.state_dict())

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._statistics.load_state_dict(state)
