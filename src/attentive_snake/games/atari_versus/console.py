"""One two-player title running on the emulator: start a game, hold one action per player, read the screen."""

from collections.abc import Sequence

import numpy as np

from attentive_snake.games.atari_versus.config import VersusRules
from attentive_snake.games.atari_versus.emulator import load_title, new_emulator
from attentive_snake.games.atari_versus.titles import get_title

PLAYERS = 2


def os_seed() -> int:
    """A seed from the operating system's entropy (the global random streams are not touched)."""
    return int(np.random.SeedSequence().generate_state(1)[0])


class Console:
    """The emulator of ``rules.title`` with its rules' frame skip, sticky actions, action set and colours.

    ``action_ids`` are the emulator's numbers of the actions a player may choose (all 18, or the
    title's own with ``full_action_space`` off), the same for both players. ``hold`` presses one
    action per player (an emulator number each) for ``frame_skip`` frames, as single-player Atari
    does, and keeps the last two screens; ``screen`` is the brighter of them at each pixel.
    """

    def __init__(self, rules: VersusRules) -> None:
        self._rules, self._title = rules, get_title(rules.title)
        self._ale = new_emulator()
        load_title(self._ale, self._title, rules.sticky_actions, 0)  # to read the action set and the screen size
        self._started = False
        self.action_ids = self._ale.getLegalActionSet() if rules.full_action_space else self._ale.getMinimalActionSet()
        height, width, _ = self._ale.getScreenRGB().shape
        self.screen_shape = (height, width)
        self._read = self._ale.getScreenGrayscale if rules.grayscale else self._ale.getScreenRGB
        self._screens = [np.zeros((height, width) if rules.grayscale else (height, width, 3), dtype=np.uint8) for _ in range(2)]

    def start(self, seed: int | None) -> None:
        """A new game: from ``seed``, or without one the emulator's next game (from an OS seed the first time)."""
        if seed is not None or not self._started:
            load_title(self._ale, self._title, self._rules.sticky_actions, os_seed() if seed is None else seed)
        else:
            self._ale.reset_game()
        self._started = True
        self._read(self._screens[1])
        self._screens[0][...] = self._screens[1]

    def hold(self, action_ids: Sequence[int]) -> tuple[list[float], bool]:
        """Press one action per player for ``frame_skip`` frames (fewer if the game ends): the points each won, and whether it ended."""
        actions = np.asarray(action_ids, dtype=np.intc)
        points, last = np.zeros(PLAYERS, dtype=np.float64), self._rules.frame_skip - 1
        for frame in range(last + 1):
            points += self._ale.act(actions)
            if self._ale.game_over():
                break
            if frame == last - 1:
                self._read(self._screens[0])
        self._read(self._screens[1])
        if last == 0:
            self._screens[0][...] = self._screens[1]
        return [float(value) for value in points], bool(self._ale.game_over())

    def screen(self) -> np.ndarray:
        """``(H, W)`` grayscale or ``(H, W, 3)`` colour uint8: the brighter of the last two frames at each pixel."""
        return np.maximum(*self._screens)

    def rgb(self) -> np.ndarray:
        """The full colour screen now, ``(H, W, 3)`` uint8."""
        return self._ale.getScreenRGB()
