"""The MinAtar titles the adapter offers: one table row each, nothing else is written per title.

``key`` is the value of ``[game] title`` and the name of the package's module
(``minatar.environments.<key>``), ``label`` the name people know. ``motion_channels`` names the
title's channels (``Env.channels`` of the package) that only tell how things move: the "trail"
channels marking where a moving object was one step earlier (Breakout, Asterix, Seaquest), the
five trail channels of Freeway whose choice tells a car's speed, and the two channels of Space
Invaders telling which way the aliens march. ``trails = false`` drops exactly these; a test
checks every name against the installed package.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Title:
    key: str
    label: str
    motion_channels: tuple[str, ...]


TITLES: tuple[Title, ...] = (
    Title("breakout", "Breakout", ("trail",)),
    Title("space_invaders", "Space Invaders", ("alien_left", "alien_right")),
    Title("freeway", "Freeway", ("speed1", "speed2", "speed3", "speed4", "speed5")),
    Title("seaquest", "Seaquest", ("trail",)),
    Title("asterix", "Asterix", ("trail",)),
)

TITLE_KEYS: tuple[str, ...] = tuple(title.key for title in TITLES)
DEFAULT_TITLE = "breakout"


def get_title(key: str) -> Title:
    """The title called ``key``; ``ValueError`` listing the known ones otherwise."""
    for title in TITLES:
        if title.key == key:
            return title
    raise ValueError(f"unknown MinAtar title {key!r}; known titles: {', '.join(TITLE_KEYS)}.")
