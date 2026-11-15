"""Names of the joystick actions, as the emulator orders them.

``FULL_ACTION_LABELS`` are the 18 actions of ``full_action_space`` (the same in every title);
``action_labels`` names whatever set an emulator offers (a title's own, shorter list otherwise).
"""

from collections.abc import Sequence

ALE_ACTION_NAMES: tuple[str, ...] = (
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT",
    "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE",
)  # fmt: skip

FULL_ACTION_LABELS: tuple[str, ...] = (
    "nothing", "fire", "up", "right", "left", "down", "up-right", "up-left", "down-right", "down-left",
    "up + fire", "right + fire", "left + fire", "down + fire", "up-right + fire", "up-left + fire", "down-right + fire", "down-left + fire",
)  # fmt: skip

_LABEL = dict(zip(ALE_ACTION_NAMES, FULL_ACTION_LABELS))


def action_labels(ale_names: Sequence[str]) -> tuple[str, ...]:
    """Our labels for the emulator's action names (``ale_py.Action`` names), in the same order."""
    return tuple(_LABEL[name] for name in ale_names)
