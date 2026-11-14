"""Hiding the velocities from the policy (``LanderRules.hide_velocities``).

Gymnasium's observation is ``x, y, vx, vy, angle, angular velocity, left leg, right leg``.
With the rule on, the three velocities are zero in what the policy sees: a policy can then
only learn them by remembering earlier frames (position and angle over time). The true
state still drives everything else (physics, reward, outcome, score).
"""

from collections.abc import Callable

import numpy as np

from attentive_snake.games.lunar_lander.config import LanderRules

VELOCITY_INDICES = (2, 3, 5)  # vx, vy, angular velocity

type Visible = Callable[[np.ndarray], np.ndarray]


def _unchanged(observation: np.ndarray) -> np.ndarray:
    return observation


def _without_velocities(observation: np.ndarray) -> np.ndarray:
    seen = observation.copy()
    seen[list(VELOCITY_INDICES)] = 0.0
    return seen


def visible_part(rules: LanderRules) -> Visible:
    """The function from the true observation to what the policy sees."""
    return _without_velocities if rules.hide_velocities else _unchanged
