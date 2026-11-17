"""The two-player Atari titles the adapter offers: one table row each, nothing else is written per title.

``key`` is the value of ``[game] title``, ``rom`` the ROM file ``ale-py`` ships (``ale_py/roms/<rom>.bin``),
``mode`` the title's two-player game mode (the numbers PettingZoo's Atari environments use),
``label`` the name people know, ``players`` how each seat's player looks on the screen, seat 0
first (found by moving one player at a time; ``tests/test_atari_versus.py`` checks the sides).
Another two-player title of the Arcade Learning Environment is one more row.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VersusTitle:
    key: str
    rom: str
    mode: int
    label: str
    players: tuple[str, str]


TITLES: tuple[VersusTitle, ...] = (
    VersusTitle("pong", "pong", 4, "Pong", ("green paddle on the right", "orange paddle on the left")),
    VersusTitle("combat_tank", "combat", 9, "Combat: Tank", ("green tank on the left", "pink tank on the right")),
    VersusTitle("combat_plane", "combat", 15, "Combat: Plane", ("orange plane", "green plane")),
    VersusTitle("boxing", "boxing", 2, "Boxing", ("white boxer", "black boxer")),
)

TITLE_KEYS: tuple[str, ...] = tuple(title.key for title in TITLES)
DEFAULT_TITLE = "pong"


def get_title(key: str) -> VersusTitle:
    """The title called ``key``; ``ValueError`` listing the known ones otherwise."""
    for title in TITLES:
        if title.key == key:
            return title
    raise ValueError(f"unknown two-player Atari title {key!r}; known titles: {', '.join(TITLE_KEYS)}.")
