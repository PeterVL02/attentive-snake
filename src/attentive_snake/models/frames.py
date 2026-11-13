"""How a stacked observation ``(B, K*F, H, W)`` splits into frames, board planes and actions.

Frames are concatenated along channels oldest first, newest last. Within a frame
the first ``BOARD_CHANNELS`` channels are the board (head, body, food); an optional
``num_actions`` constant planes follow, one-hot of the action that produced the frame
(all zero for the reset frame).
"""

from dataclasses import dataclass

import torch
from torch import Tensor

BOARD_CHANNELS = 3


@dataclass(frozen=True)
class FrameSpec:
    history_length: int  # K
    frame_channels: int  # F
    num_actions: int

    @classmethod
    def from_observation(
        cls, observation_shape: tuple[int, int, int], num_actions: int, history_length: int
    ) -> "FrameSpec":
        channels = observation_shape[0]
        if history_length < 1 or channels % history_length != 0:
            raise ValueError(
                f"observation channels ({channels}) must be a multiple of history_length ({history_length})"
            )
        return cls(history_length, channels // history_length, num_actions)

    @property
    def has_actions(self) -> bool:
        return self.frame_channels == BOARD_CHANNELS + self.num_actions

    def require_board_frames(self) -> None:
        """For models that split board from action planes: the frame must be one of the two known layouts."""
        if self.frame_channels != BOARD_CHANNELS and not self.has_actions:
            raise ValueError(
                f"frame has {self.frame_channels} channels; expected {BOARD_CHANNELS} (board) or "
                f"{BOARD_CHANNELS + self.num_actions} (board + {self.num_actions} action planes)"
            )


def split_frames(obs: Tensor, spec: FrameSpec) -> Tensor:
    """(B, K*F, H, W) -> (B, K, F, H, W), oldest frame first."""
    return obs.unflatten(1, (spec.history_length, spec.frame_channels))


def board_planes(frames: Tensor) -> Tensor:
    """(B, K, F, H, W) -> (B, K, 3, H, W)."""
    return frames[:, :, :BOARD_CHANNELS]


def action_ids(frames: Tensor, num_actions: int) -> Tensor:
    """(B, K, F, H, W) -> (B, K) action index of each frame; ``num_actions`` means "no action".

    The planes are constant, so any single cell carries the action; an all-zero
    plane stack (reset frame, or a frame before the episode began) is "no action".
    """
    planes = frames[:, :, BOARD_CHANNELS:, 0, 0]  # (B, K, A)
    none = planes.sum(-1) == 0
    return torch.where(none, torch.full_like(none, num_actions, dtype=torch.long), planes.argmax(-1))
