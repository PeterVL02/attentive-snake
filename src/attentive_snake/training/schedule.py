"""Schedules as pure functions of training progress."""


def linear_decay(initial: float, update: int, num_updates: int) -> float:
    """Value for 0-based ``update``: ``initial`` at the first update, decaying to ``initial / num_updates`` at the last."""
    return initial * (1.0 - update / num_updates)


def crossed_multiple(previous_env_steps: int, env_steps: int, interval: int) -> bool:
    """Whether going from ``previous_env_steps`` to ``env_steps`` passed a multiple of ``interval`` (> 0)."""
    return env_steps // interval > previous_env_steps // interval
