"""A MinAtar title as a ``GameEnv``: a 10 x 10 grid with one 0/1 channel per kind of object, discrete actions.

The package's state is ``(10, 10, channels)`` booleans; one frame here is ``(channels, 10, 10)``
float32, the channels in the package's order minus the motion channels when ``trails`` is off
(``channels.visible_channels``; their names are ``channel_names``). History is added outside
(``HistoryEnv``). ``reset(seed)`` reseeds the title and its sticky actions, and forgets the last
action (the package keeps it across games), so a seeded game always plays out the same way.
"""

from typing import Any

import numpy as np

from attentive_snake.env.types import EnvStep
from attentive_snake.games.minatar.actions import action_labels
from attentive_snake.games.minatar.channels import channel_indices, channel_names, visible_channels
from attentive_snake.games.minatar.config import MinAtarRules
from attentive_snake.games.minatar.emulator import make_emulator
from attentive_snake.games.minatar.outcome import step_event
from attentive_snake.games.minatar.reward import MinAtarReward, minatar_reward
from attentive_snake.games.minatar.titles import get_title
from attentive_snake.games.minatar.view import board_view, channel_colors

FULL_ACTIONS = tuple(range(6))


class MinAtarEnv:
    """One MinAtar game; ``score`` is the episode's points, ``reward`` the scaled one.

    ``action_labels`` names this env's actions (the title's own list unless ``full_action_space``),
    ``channel_names`` the channels of a frame.
    """

    def __init__(self, rules: MinAtarRules, reward: MinAtarReward, seed: int | None = None) -> None:
        self._game = make_emulator(rules)
        channels = self._game.env.channels
        self.channel_names = visible_channels(channels, get_title(rules.title), rules.trails)
        self._visible = channel_indices(channels, self.channel_names)
        self._colors = channel_colors(channel_names(channels))
        self._actions = FULL_ACTIONS if rules.full_action_space else tuple(self._game.minimal_action_set())
        self.num_actions = len(self._actions)
        self.action_labels = action_labels(self._actions)
        self.observation_shape = (len(self.channel_names), *self._game.state_shape()[:2])
        self._rules, self._reward = rules, reward
        self._seed = seed  # used by the first reset that is given no seed
        self._points, self._steps = 0.0, 0
        self._observation = np.zeros(self.observation_shape, dtype=np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        seed, self._seed = (seed if seed is not None else self._seed), None
        self._game.seed(seed)  # None keeps the current random state
        self._game.reset()
        self._game.last_action = 0
        self._points, self._steps = 0.0, 0
        return self._observe()

    def step(self, action: int | np.ndarray) -> EnvStep:
        points, game_over = self._game.act(self._actions[int(action)])
        points, game_over = float(points), bool(game_over)
        self._points += points
        self._steps += 1
        timed_out = not game_over and self._steps >= self._rules.max_episode_steps
        return EnvStep(
            observation=self._observe(),
            reward=minatar_reward(self._reward, points, game_over),
            terminated=game_over,
            truncated=timed_out,
            event=step_event(points, game_over, timed_out),
            score=self._points,
        )

    def _observe(self) -> np.ndarray:
        state = self._game.state()  # (height, width, all channels) bool
        self._observation = np.ascontiguousarray(state[:, :, self._visible].transpose(2, 0, 1), dtype=np.float32)
        return self._observation.copy()

    def view(self) -> dict[str, Any]:
        return board_view(self._observation, self.channel_names, self._colors)
