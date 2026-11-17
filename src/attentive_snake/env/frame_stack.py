"""One observer's observations: each new frame gets its action channels and joins the history of the last frames."""

import numpy as np

from attentive_snake.env.frame import ActionChannels
from attentive_snake.env.history import FrameHistory


class FrameStack:
    """Frames ``frame_shape`` -> observations ``shape``: ``history_length`` frames, each with ``action_channels`` one-hot channels.

    ``reset`` starts an episode from its first frame (no action yet), ``push`` adds the frame
    the action ``action`` produced. Both return a fresh array (see ``FrameHistory``).
    """

    def __init__(self, frame_shape: tuple[int, ...], history_length: int, action_channels: int) -> None:
        self._actions = ActionChannels(tuple(frame_shape), action_channels)
        self._history = FrameHistory(history_length, self._actions.shape)
        self.frame_channels = self._actions.shape[0]
        self.shape: tuple[int, ...] = self._history.shape

    def reset(self, frame: np.ndarray) -> np.ndarray:
        return self._history.reset(self._actions.build(frame))

    def push(self, frame: np.ndarray, action: int | np.ndarray) -> np.ndarray:
        return self._history.push(self._actions.build(frame, action))
