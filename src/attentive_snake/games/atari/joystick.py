"""Continuous joystick actions for Atari, as ALE's own continuous mode reads them (``ale_py.env.AtariEnv(continuous=True)``).

An action is three numbers: how far the stick is pushed (``radius`` in [0, 1]), in which
direction (``angle`` in [-pi, pi], 0 = right, pi/2 = up) and the fire button (``fire`` in [0, 1]).
The stick counts as pushed left/right (up/down) when its horizontal (vertical) part is beyond
``THRESHOLD``, fire as pressed above it, giving one of the 18 joystick actions (named as
``actions.ALE_ACTION_NAMES``); the push is also passed to the emulator as the paddle strength.
Numpy only, no emulator import.
"""

import math

import numpy as np

from attentive_snake.env.action_space import ContinuousActions

THRESHOLD = 0.5  # ALE's default continuous_action_threshold
CONTINUOUS_ACTIONS = ContinuousActions(low=(0.0, -math.pi, 0.0), high=(1.0, math.pi, 1.0))
CONTINUOUS_LABELS = ("stick push", "stick direction", "fire")


def _axis(value: float, negative: str, positive: str) -> str:
    return negative if value < -THRESHOLD else positive if value > THRESHOLD else ""


def joystick_press(action: np.ndarray) -> tuple[str, float]:
    """``(radius, angle, fire)`` -> (the ALE name of the joystick action, the paddle strength)."""
    radius, angle, fire = (float(number) for number in action)
    x, y = radius * math.cos(angle), radius * math.sin(angle)
    name = _axis(y, "DOWN", "UP") + _axis(x, "LEFT", "RIGHT") + ("FIRE" if fire > THRESHOLD else "")
    return name or "NOOP", radius
