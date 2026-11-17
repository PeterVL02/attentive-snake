"""Names of MinAtar's actions, as the package orders them (``Env.action_map``: n, l, u, r, d, f).

``FULL_ACTION_LABELS`` are the 6 actions every title accepts; ``action_labels`` names whatever
subset an env offers (a title's ``minimal_action_set()`` otherwise).
"""

from collections.abc import Sequence

FULL_ACTION_LABELS: tuple[str, ...] = ("nothing", "left", "up", "right", "down", "fire")


def action_labels(actions: Sequence[int]) -> tuple[str, ...]:
    """Our labels for the package's action numbers ``actions``, in the same order."""
    return tuple(FULL_ACTION_LABELS[action] for action in actions)
