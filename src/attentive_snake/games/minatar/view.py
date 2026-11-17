"""What the model sees of a MinAtar game as a ``"board"`` view: one coloured, named layer per visible channel.

A channel keeps its colour whether or not the motion channels are shown (colours follow the
title's full channel list), so Breakout's ball looks the same with ``trails`` on and off.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from attentive_snake.views.board_layers import board_layer, layered_board_payload

# Ten mid-tone colours that read on the app's light and dark boards (MinAtar titles have at most 10 channels).
PALETTE: tuple[str, ...] = (
    "#4e79a7", "#e15759", "#76b7b2", "#59a14f", "#edc948",
    "#b07aa1", "#ff9da7", "#9c755f", "#f28e2b", "#bab0ac",
)  # fmt: skip


def channel_colors(all_channels: Sequence[str]) -> dict[str, str]:
    """``{channel name: colour}`` by the channel's place in the title's full list."""
    return {name: PALETTE[index % len(PALETTE)] for index, name in enumerate(all_channels)}


def board_view(observation: np.ndarray, names: Sequence[str], colors: dict[str, str]) -> dict[str, Any]:
    """The board payload of one ``(channels, height, width)`` observation whose channels are ``names``."""
    _, height, width = observation.shape
    layers = [board_layer(name, colors[name], ((x, y) for y, x in np.argwhere(plane > 0))) for name, plane in zip(names, observation)]
    return layered_board_payload(width, height, layers)
