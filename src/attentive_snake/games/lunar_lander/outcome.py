"""What a Lunar Lander step did, in plain words, derived from what Gymnasium exposes.

Gymnasium's ``LunarLander`` ends an episode (``terminated``) when the lander body touches
the ground or leaves the screen (``game_over`` / ``|x| >= 1``, reward -100), or when Box2D
puts the lander to sleep because it came to rest (``lander.awake`` is false, reward +100;
this wins if both happen). The ``TimeLimit`` wrapper truncates at ``max_episode_steps``.
"""

FLYING, LANDED, CRASHED, FLEW_OFF, TIMEOUT = "flying", "landed", "crashed", "flew_off", "timeout"


def step_event(lander, x: float, terminated: bool, truncated: bool) -> str:
    """The event of one step; ``lander`` is the unwrapped Gymnasium env, ``x`` the new horizontal position."""
    if terminated:
        if not lander.lander.awake:
            return LANDED  # at rest anywhere, not necessarily between the flags
        return FLEW_OFF if abs(x) >= 1.0 else CRASHED
    return TIMEOUT if truncated else FLYING
