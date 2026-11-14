"""What a policy needs to know about its input: the contract between environments and models.

Pure data (no numpy, no torch). ``attentive_snake.factory`` builds an ``ObservationSpec``
from the environment and the game, and passes it to the policy; models never look at a
game or an environment themselves.

Layout of one observation (batch axis not included), frames oldest first along axis 0::

    grid / image : (K * F, H, W)
    vector       : (K * F,)

K is ``history_length`` and F is ``frame_channels``. Within a frame the first
``base_channels`` channels are the game's own observation; the last ``action_channels``
channels (0 or the number of actions) are a one-hot of the action that produced the frame,
constant over the spatial axes (all zero for the first frame of an episode).
"""

from dataclasses import dataclass
from typing import Literal

type ObservationKind = Literal["grid", "vector", "image"]
OBSERVATION_KINDS: tuple[str, ...] = ("grid", "vector", "image")

_RANK = {"grid": 3, "image": 3, "vector": 1}


@dataclass(frozen=True)
class ObservationSpec:
    """Shape and meaning of a (possibly stacked) observation.

    Attributes:
        kind: ``"grid"`` (a small board, one cell per position, e.g. Snake), ``"vector"``
            (a flat list of numbers, e.g. Lunar Lander) or ``"image"`` (screen pixels).
        shape: The full observation shape without the batch axis (see the module docstring).
        history_length: Number of stacked frames K (1 = no history).
        action_channels: Trailing one-hot action channels per frame (0 = none).
    """

    kind: ObservationKind
    shape: tuple[int, ...]
    history_length: int = 1
    action_channels: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(n) for n in self.shape))
        if self.kind not in _RANK:
            raise ValueError(f"unknown observation kind {self.kind!r}; valid kinds: {', '.join(OBSERVATION_KINDS)}.")
        if len(self.shape) != _RANK[self.kind]:
            raise ValueError(f"a {self.kind} observation needs {_RANK[self.kind]} axes, got shape {self.shape}.")
        if self.history_length < 1 or self.shape[0] % self.history_length != 0:
            raise ValueError(
                f"observation channels ({self.shape[0]}) must be a multiple of history_length ({self.history_length})."
            )
        if not 0 <= self.action_channels < self.frame_channels:
            raise ValueError(
                f"action_channels ({self.action_channels}) must be at least 0 and fewer than the "
                f"{self.frame_channels} channels of a frame."
            )

    @property
    def frame_channels(self) -> int:
        """F: channels of one frame, action channels included."""
        return self.shape[0] // self.history_length

    @property
    def base_channels(self) -> int:
        """Channels of one frame that come from the game (F minus the action channels)."""
        return self.frame_channels - self.action_channels

    @property
    def has_actions(self) -> bool:
        return self.action_channels > 0

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """``(H, W)`` for grids and images, ``()`` for vectors."""
        return self.shape[1:]
