"""When training evaluates: each time the env-step count passes a multiple of the interval, and at the end."""


def evaluation_due(previous_env_steps: int, env_steps: int, interval: int, is_last: bool) -> bool:
    """Whether the update that went from ``previous_env_steps`` to ``env_steps`` ends with an evaluation.

    ``interval`` 0 never evaluates. An update crossing several multiples evaluates once.
    """
    if interval <= 0:
        return False
    return is_last or env_steps // interval > previous_env_steps // interval
