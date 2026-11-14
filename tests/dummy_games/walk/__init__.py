"""A tiny 1-D grid game used to prove that a game is drag-and-drop: walk right to the goal.

It lives only in the test tree; tests make it visible by adding ``tests/dummy_games`` to
``attentive_snake.games.__path__``. Nothing outside this directory names it.
Observation: ``(2, 1, length)``: channel 0 the walker, channel 1 the goal (the last cell).
Actions: 0 = left, 1 = right.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from attentive_snake.env.types import EnvStep
from attentive_snake.games.spec import GameSpec
from attentive_snake.views.board import board_payload


@dataclass(frozen=True)
class WalkRules:
    length: int = field(default=5, metadata={"help": "Number of cells; the goal is the last one."})
    max_steps: int = field(default=12, metadata={"help": "Steps before the episode is cut off."})


@dataclass(frozen=True)
class WalkReward:
    goal: float = field(default=1.0, metadata={"help": "Reward for reaching the goal."})
    step: float = field(default=-0.01, metadata={"help": "Reward for every other step."})


class WalkEnv:
    num_actions = 2

    def __init__(self, rules: WalkRules, reward: WalkReward, seed: int | None = None) -> None:
        self._rules, self._reward = rules, reward
        self.observation_shape = (2, 1, rules.length)
        self._position = self._steps = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._position = self._steps = 0
        return self._observation()

    def step(self, action: int) -> EnvStep:
        self._steps += 1
        self._position = min(max(self._position + (1 if action == 1 else -1), 0), self._rules.length - 1)
        won = self._position == self._rules.length - 1
        cut = not won and self._steps >= self._rules.max_steps
        event = "won" if won else "too_slow" if cut else "moved"
        reward = self._reward.goal if won else self._reward.step
        return EnvStep(self._observation(), reward, won, cut, event, float(self._position))

    def view(self) -> dict[str, Any]:
        return board_payload(self._rules.length, 1, [(self._position, 0)], (self._rules.length - 1, 0))

    def _observation(self) -> np.ndarray:
        obs = np.zeros(self.observation_shape, dtype=np.float32)
        obs[0, 0, self._position] = 1.0
        obs[1, 0, -1] = 1.0
        return obs


GAME = GameSpec(
    name="walk",
    title="Walk",
    description="Walk right until you reach the goal.",
    observation_kind="grid",
    view_kind="board",
    rules_type=WalkRules,
    reward_type=WalkReward,
    make_env=WalkEnv,
    locked_rules={"length": "it fixes the input size"},
)
