"""What a step of a two-player match did for one player, in plain words.

During play: ``scored`` (the player won points), ``conceded`` (the other one did) or ``playing``.
The last step's event is the match's outcome for that player: ``won``, ``lost`` or ``draw`` by
its margin when the title ended the match, or ``timeout`` (``VersusRules.max_episode_steps``).
"""

PLAYING, SCORED, CONCEDED = "playing", "scored", "conceded"
WON, LOST, DRAW, TIMEOUT = "won", "lost", "draw", "timeout"


def seat_event(points: float, margin: float, game_over: bool, timed_out: bool) -> str:
    """The event of one player's step that won ``points``, with ``margin`` its points minus the other's so far."""
    if game_over:
        return WON if margin > 0 else LOST if margin < 0 else DRAW
    if timed_out:
        return TIMEOUT
    return SCORED if points > 0 else CONCEDED if points < 0 else PLAYING
