"""Making the two-player emulator of a title: ``multi_agent_ale_py`` (PettingZoo's fork of the Arcade
Learning Environment, whose ``act`` takes one action per player) with the ROM that ``ale-py`` ships.

Both are imported here, on first use, so that importing the game package costs nothing. A ROM
is (re)loaded on every seeded start: that is how the emulator takes a new random seed.
"""

from importlib import resources

from attentive_snake.games.atari_versus.titles import VersusTitle

_SEED_LIMIT = 2**31  # the emulator's seed is a C int


def new_emulator():
    """An empty two-player emulator (``multi_agent_ale_py.ALEInterface``) that logs only errors."""
    import multi_agent_ale_py

    multi_agent_ale_py.ALEInterface.setLoggerMode(multi_agent_ale_py.ALEInterface.Logger.Error)
    return multi_agent_ale_py.ALEInterface()


def load_title(ale, title: VersusTitle, sticky_actions: float, seed: int) -> None:
    """Load ``title``'s ROM into ``ale`` with ``seed`` and its two-player mode; the game starts afresh."""
    ale.setInt(b"random_seed", int(seed) % _SEED_LIMIT)
    ale.setFloat(b"repeat_action_probability", float(sticky_actions))
    ale.loadROM(str(resources.files("ale_py") / "roms" / f"{title.rom}.bin"))
    ale.setMode(title.mode)
    ale.reset_game()
