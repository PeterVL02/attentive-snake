"""What an Atari step did, in plain words, from what the emulator reports.

The last step's event is the episode's outcome: ``game_over`` (the title ended the game: last
life lost, or in Pong one side reached 21) or ``timeout`` (``AtariRules.max_episode_steps``).
"""

PLAYING, SCORED, LOST_POINTS, LIFE_LOST, GAME_OVER, TIMEOUT = "playing", "scored", "lost_points", "life_lost", "game_over", "timeout"


def step_event(points: float, lives_lost: int, game_over: bool, timed_out: bool) -> str:
    """The event of one step: an ending first, then a lost life, then points won or lost."""
    if game_over:
        return GAME_OVER
    if timed_out:
        return TIMEOUT
    if lives_lost > 0:
        return LIFE_LOST
    if points > 0:
        return SCORED
    return LOST_POINTS if points < 0 else PLAYING
