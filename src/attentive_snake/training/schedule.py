"""Learning-rate schedules as pure functions."""


def linear_decay(initial: float, update: int, num_updates: int) -> float:
    """Value for 0-based ``update``: ``initial`` at the first update, decaying to ``initial / num_updates`` at the last."""
    return initial * (1.0 - update / num_updates)
