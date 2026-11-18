"""An Atari title as a ``GameEnv``: one preprocessed screen picture per step, discrete joystick actions
(or with ``AtariRules.continuous`` a continuous joystick: ``joystick``).

One step holds the chosen action for ``frame_skip`` screen frames (the emulator applies sticky
actions on each), sums their points, and shows the brighter of the last two frames at each
pixel, shrunk to ``screen_size`` squared (``ale.screen.ScreenShrinker``): ``(1, S, S)`` grayscale
or ``(3, S, S)`` colour, float32 in [0, 1]. History is added outside (``HistoryEnv``).
``view()`` is the console's full colour screen; nothing is rendered for training.
"""

from typing import Any

import numpy as np

from attentive_snake.ale.actions import action_labels
from attentive_snake.ale.screen import ScreenShrinker
from attentive_snake.env.types import EnvStep
from attentive_snake.games.atari.config import AtariRules
from attentive_snake.games.atari.emulator import make_emulator
from attentive_snake.games.atari.joystick import CONTINUOUS_ACTIONS, CONTINUOUS_LABELS, joystick_press
from attentive_snake.games.atari.outcome import step_event
from attentive_snake.games.atari.reward import AtariReward, atari_reward
from attentive_snake.views.image import image_payload


class AtariEnv:
    """One Atari game; ``score`` is the episode's raw points, ``reward`` the clipped / scaled one.

    ``action_labels`` names this env's actions (a title's own list when ``full_action_space`` is off).
    With ``continuous`` rules it carries ``action_space`` (stick push, direction, fire) and every one
    of the 18 joystick actions can be reached, whatever ``full_action_space`` says.
    """

    def __init__(self, rules: AtariRules, reward: AtariReward, seed: int | None = None) -> None:
        self._emulator = make_emulator(rules)
        self._ale = self._emulator.ale
        legal = rules.full_action_space or rules.continuous
        self._actions = self._ale.getLegalActionSet() if legal else self._ale.getMinimalActionSet()
        self._rules = rules
        self._reward = reward
        self._seed = seed  # used by the first reset that is given no seed
        height, width = self._ale.getScreenDims()
        self._read = self._ale.getScreenGrayscale if rules.grayscale else self._ale.getScreenRGB
        self._screens = [np.zeros((height, width) if rules.grayscale else (height, width, 3), dtype=np.uint8) for _ in range(2)]
        self._shrink = ScreenShrinker(height, width, rules.screen_size)
        self.observation_shape = (1 if rules.grayscale else 3, rules.screen_size, rules.screen_size)
        self.num_actions = len(self._actions)
        self.action_labels = action_labels([action.name for action in self._actions])
        self._continuous = rules.continuous
        if rules.continuous:
            self.action_space, self.num_actions, self.action_labels = CONTINUOUS_ACTIONS, CONTINUOUS_ACTIONS.dimension, CONTINUOUS_LABELS
            self._by_name = {action.name: action for action in self._actions}
        self._points, self._steps, self._lives = 0.0, 0, 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        seed, self._seed = (seed if seed is not None else self._seed), None
        self._emulator.reset(seed=seed)
        self._points, self._steps, self._lives = 0.0, 0, self._ale.lives()
        self._read(self._screens[1])
        return self._shrink(self._screens[1])

    def step(self, action: int | np.ndarray) -> EnvStep:
        points, game_over = self._hold(*self._press(action))
        self._points += points
        self._steps += 1
        lives_lost = max(self._lives - self._ale.lives(), 0)
        self._lives = self._ale.lives()
        timed_out = not game_over and self._steps >= self._rules.max_episode_steps
        return EnvStep(
            observation=self._shrink(np.maximum(*self._screens)),
            reward=atari_reward(self._reward, points, lives_lost),
            terminated=game_over,
            truncated=timed_out,
            event=step_event(points, lives_lost, game_over, timed_out),
            score=self._points,
        )

    def _press(self, action: int | np.ndarray) -> tuple:
        """What ``ale.act`` is given: the chosen action, or the stick's joystick action and its strength."""
        if not self._continuous:
            return (self._actions[int(action)],)
        name, strength = joystick_press(action)
        return self._by_name[name], strength

    def _hold(self, *press) -> tuple[float, bool]:
        """Press (``ale.act(*press)``) for ``frame_skip`` frames (fewer if the game ends): the points won, and whether it ended.

        Leaves the last two screens in ``self._screens`` (twice the last one with ``frame_skip = 1``).
        """
        points, last = 0.0, self._rules.frame_skip - 1
        for frame in range(last + 1):
            points += self._ale.act(*press)
            if self._ale.game_over():
                break
            if frame == last - 1:
                self._read(self._screens[0])
        self._read(self._screens[1])
        if last == 0:
            self._screens[0][...] = self._screens[1]
        return float(points), bool(self._ale.game_over())

    def view(self) -> dict[str, Any]:
        return image_payload(self._ale.getScreenRGB())
