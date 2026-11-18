"""What a game with several players (seats) needs from ``[env]``, checked from the config alone (no env is built).

Self-play training (``GameSpec.make_match``) fills the parallel environments with whole
matches, one slot per seat, so ``env.num_envs`` must be a multiple of the game's ``seats``.
``ExperimentConfig`` refuses anything else when it is built, so the app answers 422 and the
CLI stops before training starts.
"""

from attentive_snake.games.registry import get_game


def seat_problems(game_name: str, num_envs: int) -> list[str]:
    """Plain-language reasons why ``num_envs`` cannot hold whole matches of the game (empty if it can)."""
    game = get_game(game_name)
    if num_envs % game.seats == 0:
        return []
    return [
        f"{game.title} is played by {game.seats} players and trains by self-play: every match fills {game.seats} of the "
        f"parallel games (one per player), so env.num_envs must be a multiple of {game.seats}, not {num_envs}."
    ]


def require_whole_matches(game_name: str, num_envs: int) -> None:
    """``ValueError`` naming every problem of ``seat_problems``."""
    problems = seat_problems(game_name, num_envs)
    if problems:
        raise ValueError(" ".join(problems))
