"""Exploration schedule of DQN as a pure function of training progress."""


def linear_epsilon(start: float, end: float, fraction: float, env_steps: int, total_env_steps: int) -> float:
    """``start`` at step 0, falling linearly to ``end`` at ``fraction * total_env_steps`` env steps, ``end`` after."""
    decay_steps = fraction * total_env_steps
    if decay_steps <= 0 or env_steps >= decay_steps:
        return end
    return start + (end - start) * env_steps / decay_steps
