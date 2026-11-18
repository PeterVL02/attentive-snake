"""Which game modes and difficulties a title offers, and refusing the others when a config is built.

The ROM itself is asked, so the answer cannot go stale and no list is kept per title. That needs
``ale-py``, imported here on first use and only for rules that choose a mode or a difficulty:
rules that leave both unset never load the emulator. A refusal is a ``ValueError``, which the app
answers with 422 and ``scripts/train.py`` prints as a usage error.
"""

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class TitleVariants:
    modes: tuple[int, ...]
    difficulties: tuple[int, ...]


@lru_cache
def available_variants(title: str) -> TitleVariants:
    """What the ROM of ``title`` (a key of ``titles.TITLES``, which is also the ROM's name in ``ale-py``) offers."""
    import ale_py
    from ale_py import roms

    ale_py.ALEInterface.setLoggerMode(ale_py.LoggerMode.Error)
    ale = ale_py.ALEInterface()
    ale.loadROM(roms.get_rom_path(title))
    return TitleVariants(tuple(int(mode) for mode in ale.getAvailableModes()), tuple(int(level) for level in ale.getAvailableDifficulties()))


def require_available_variant(title: str, mode: int | None, difficulty: int | None) -> None:
    """``ValueError`` naming the title, each unavailable value and the available ones; ``None`` is always fine.

    The message says "game", not "title": the app's form puts an error next to the first field whose name it contains.
    """
    if mode is None and difficulty is None:
        return
    variants = available_variants(title)
    problems = []
    if mode is not None and mode not in variants.modes:
        problems.append(f"Atari game {title!r} has no mode {mode}; available modes: {_listed(variants.modes)}.")
    if difficulty is not None and difficulty not in variants.difficulties:
        problems.append(f"Atari game {title!r} has no difficulty {difficulty}; available difficulties: {_listed(variants.difficulties)}.")
    if problems:
        raise ValueError(" ".join(problems))


def _listed(values: tuple[int, ...]) -> str:
    return ", ".join(str(value) for value in values)
