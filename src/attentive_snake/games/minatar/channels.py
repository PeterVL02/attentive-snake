"""Which of a title's channels the model sees: all of them, or all but the motion channels (``trails = false``).

Pure: works on the package's ``{channel name: index}`` table, no emulator import.
"""

from collections.abc import Mapping

import numpy as np

from attentive_snake.games.minatar.titles import Title


def channel_names(channels: Mapping[str, int]) -> tuple[str, ...]:
    """The package's channel names in channel order."""
    return tuple(sorted(channels, key=channels.__getitem__))


def visible_channels(channels: Mapping[str, int], title: Title, trails: bool) -> tuple[str, ...]:
    """The names of the channels the model sees, in channel order; ``ValueError`` if the title names an unknown one."""
    missing = [name for name in title.motion_channels if name not in channels]
    if missing:
        raise ValueError(f"MinAtar {title.label} has no channel(s) {', '.join(missing)}; it has {', '.join(channel_names(channels))}.")
    return tuple(name for name in channel_names(channels) if trails or name not in title.motion_channels)


def channel_indices(channels: Mapping[str, int], names: tuple[str, ...]) -> np.ndarray:
    """The package's indices of ``names``, in their order (for picking them out of a state)."""
    return np.array([channels[name] for name in names], dtype=np.intp)
