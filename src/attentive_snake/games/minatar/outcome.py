"""What a MinAtar step did, in plain words.

The last step's event is the episode's outcome: ``game_over`` (the title ended the game: the
ball lost, the player hit, Freeway's clock run out) or ``timeout`` (``MinAtarRules.max_episode_steps``).
"""

PLAYING, SCORED, GAME_OVER, TIMEOUT = "playing", "scored", "game_over", "timeout"


def step_event(points: float, game_over: bool, timed_out: bool) -> str:
    """The event of one step: an ending first, then points won."""
    if game_over:
        return GAME_OVER
    if timed_out:
        return TIMEOUT
    return SCORED if points > 0 else PLAYING
