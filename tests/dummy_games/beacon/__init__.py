"""A tiny game with the ``"image"`` view kind and no reward table, for the app's tests: drift to the beacon.

Like ``walk`` it lives only in the test tree (tests add ``tests/dummy_games`` to
``attentive_snake.games.__path__``). The observation is a 1-D grid ``(2, 1, length)``
(channel 0 the player, channel 1 the beacon in the last cell), so every policy kind can
train on it; the view is a small RGB picture, ``CELL_PX`` pixels per cell.
Actions: 0 = left, 1 = right.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from attentive_snake.env.types import EnvStep
from attentive_snake.games.spec import GameSpec
from attentive_snake.views.image import image_payload

CELL_PX = 4


@dataclass(frozen=True)
class BeaconRules:
    length: int = field(default=5, metadata={"help": "Number of cells; the beacon is in the last one."})
    max_steps: int = field(default=12, metadata={"help": "Steps before the episode is cut off."})


@dataclass(frozen=True)
class BeaconReward:
    beacon: float = field(default=1.0, metadata={"help": "Reward for reaching the beacon."})
    step: float = field(default=-0.01, metadata={"help": "Reward for every other step."})


class BeaconEnv:
    num_actions = 2

    def __init__(self, rules: BeaconRules, reward: BeaconReward, seed: int | None = None) -> None:
        self._rules, self._reward = rules, reward
        self.observation_shape = (2, 1, rules.length)
        self._position = self._steps = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._position = self._steps = 0
        return self._observation()

    def step(self, action: int) -> EnvStep:
        self._steps += 1
        self._position = min(max(self._position + (1 if action == 1 else -1), 0), self._rules.length - 1)
        arrived = self._position == self._rules.length - 1
        cut = not arrived and self._steps >= self._rules.max_steps
        event = "arrived" if arrived else "too_slow" if cut else "drifted"
        reward = self._reward.beacon if arrived else self._reward.step
        return EnvStep(self._observation(), reward, arrived, cut, event, float(self._position))

    def view(self) -> dict[str, Any]:
        picture = np.zeros((CELL_PX, CELL_PX * self._rules.length, 3), dtype=np.uint8)
        picture[:, -CELL_PX:] = (240, 200, 40)
        picture[:, self._position * CELL_PX : (self._position + 1) * CELL_PX] = (60, 160, 230)
        return image_payload(picture)

    def _observation(self) -> np.ndarray:
        obs = np.zeros(self.observation_shape, dtype=np.float32)
        obs[0, 0, self._position] = 1.0
        obs[1, 0, -1] = 1.0
        return obs


GAME = GameSpec(
    name="beacon",
    title="Beacon",
    description="Drift right until you reach the beacon.",
    observation_kind="grid",
    view_kind="image",
    rules_type=BeaconRules,
    reward_type=BeaconReward,
    make_env=BeaconEnv,
    locked_rules={"length": "it fixes the input size"},
)
