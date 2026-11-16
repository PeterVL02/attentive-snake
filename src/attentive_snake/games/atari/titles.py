"""The Atari titles the adapter offers: one table row each, nothing else is written per title.

``key`` is the value of ``[game] title``, ``env_id`` the Gymnasium id (``ale-py`` ships the ROMs),
``label`` the name people know. Any other single-player ALE title is one more row.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Title:
    key: str
    env_id: str
    label: str


TITLES: tuple[Title, ...] = (
    Title("pong", "ALE/Pong-v5", "Pong"),
    Title("ms_pacman", "ALE/MsPacman-v5", "Ms. Pac-Man"),
    Title("montezuma_revenge", "ALE/MontezumaRevenge-v5", "Montezuma's Revenge"),
    Title("pitfall", "ALE/Pitfall-v5", "Pitfall!"),
    Title("private_eye", "ALE/PrivateEye-v5", "Private Eye"),
    Title("venture", "ALE/Venture-v5", "Venture"),
)

TITLE_KEYS: tuple[str, ...] = tuple(title.key for title in TITLES)
DEFAULT_TITLE = "pong"


def get_title(key: str) -> Title:
    """The title called ``key``; ``ValueError`` listing the known ones otherwise."""
    for title in TITLES:
        if title.key == key:
            return title
    raise ValueError(f"unknown Atari title {key!r}; known titles: {', '.join(TITLE_KEYS)}.")
