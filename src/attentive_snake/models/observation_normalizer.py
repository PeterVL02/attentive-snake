"""Rescaling vector observations by running statistics, as a part of the policy.

Living in the policy (``BasePolicy.observation_normalizer``, outside ``policy.net``) means every
reader of the policy sees the same input without knowing about it: acting, the truncation
bootstrap, evaluation, watching and attention all go through ``forward`` / ``attention_maps``, and
the statistics are saved and restored with the weights. Only the learner calls ``update``.

One observation is K stacked frames of F numbers, of which the last ``action_channels`` are a
one-hot action. Each of the game's numbers has one mean and variance, shared by all K frames so
that differences between frames keep their meaning; the action channels pass through unchanged.
The statistics learn from the newest frame only: older positions hold zeros before an episode's
start, which are not observations of the game.
"""

import torch
from torch import Tensor, nn

from attentive_snake.models.running_statistics import RunningStatistics
from attentive_snake.observation_spec import ObservationSpec

CLIP = 10.0  # normalised numbers are cut to +-CLIP, as in Gymnasium's NormalizeObservation and CleanRL
_EPSILON = 1e-8


class ObservationNormalizer(nn.Module):
    """``(number - mean) / std``, clipped, for observations ``(..., K * F)`` of a vector game."""

    def __init__(self, spec: ObservationSpec) -> None:
        super().__init__()
        if spec.kind != "vector":
            raise ValueError(f"only vector observations can be normalised, not {spec.kind} ones.")
        self._frames = (spec.history_length, spec.frame_channels)
        self._game_numbers = spec.base_channels
        self.statistics = RunningStatistics((spec.base_channels,))

    def forward(self, observations: Tensor) -> Tensor:
        frames = observations.unflatten(-1, self._frames)
        game, actions = frames[..., : self._game_numbers], frames[..., self._game_numbers :]
        scale = (self.statistics.variance + _EPSILON).sqrt()
        scaled = ((game.to(scale.dtype) - self.statistics.mean) / scale).clamp(-CLIP, CLIP)
        return torch.cat([scaled.to(observations.dtype), actions], dim=-1).flatten(-2)

    def update(self, observations: Tensor) -> None:
        """Learn from a batch ``(B, K * F)``: the game's numbers of each newest frame."""
        self.statistics.update(observations[:, -self._frames[1] :][:, : self._game_numbers])
