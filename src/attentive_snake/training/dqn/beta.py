"""Annealing of prioritized replay's importance-sampling exponent as a pure function of training progress."""


def linear_beta(start: float, env_steps: int, total_env_steps: int) -> float:
    """``start`` at step 0, rising linearly to 1 at ``total_env_steps`` env steps (1 from there on)."""
    if total_env_steps <= 0 or env_steps >= total_env_steps:
        return 1.0
    return start + (1.0 - start) * env_steps / total_env_steps
