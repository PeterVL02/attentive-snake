"""How a stacked observation splits into frames, game channels and actions (see ``ObservationSpec``).

Frames are concatenated along axis 1 (after the batch axis), oldest first. Within a frame
the first ``spec.base_channels`` channels come from the game; the trailing
``spec.action_channels`` channels, if any, are a constant one-hot of the action that
produced the frame (all zero for the reset frame). Works for grids ``(B, K*F, H, W)``
and vectors ``(B, K*F)`` alike.
"""

import torch
from torch import Tensor

from attentive_snake.observation_spec import ObservationSpec


def split_frames(obs: Tensor, spec: ObservationSpec) -> Tensor:
    """(B, K*F, *rest) -> (B, K, F, *rest), oldest frame first."""
    return obs.unflatten(1, (spec.history_length, spec.frame_channels))


def base_planes(frames: Tensor, spec: ObservationSpec) -> Tensor:
    """(B, K, F, *rest) -> (B, K, base_channels, *rest): the game's own channels of every frame."""
    return frames[:, :, : spec.base_channels]


def action_ids(frames: Tensor, spec: ObservationSpec, num_actions: int) -> Tensor:
    """(B, K, F, *rest) -> (B, K) action index of each frame; ``num_actions`` means "no action".

    The action channels are constant, so any single position carries the action; an
    all-zero stack (reset frame, or a frame before the episode began) is "no action".
    """
    planes = frames[:, :, spec.base_channels :]
    planes = planes.reshape(*planes.shape[:3], -1)[..., 0]  # (B, K, A): one position per channel
    none = planes.sum(-1) == 0
    return torch.where(none, torch.full_like(none, num_actions, dtype=torch.long), planes.argmax(-1))
