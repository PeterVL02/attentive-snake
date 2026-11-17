"""What a MinAtar reward scheme pays in each kind of step, for the app's Rewards tab."""

from attentive_snake.games.minatar.reward import MinAtarReward

_DIGITS = 9  # hides float noise


def reward_table(config: MinAtarReward, example_score: int = 1) -> list[dict]:
    """Rows ``{"situation", "reward", "breakdown": {term: value}}``, one per kind of step.

    ``example_score`` is the points of the scoring row (MinAtar mostly pays 1 at a time).
    """

    def row(situation: str, points: float, game_over: bool) -> dict:
        breakdown = {"points": round(config.scale * points, _DIGITS), "game_over": config.game_over if game_over else 0.0}
        return {"situation": situation, "reward": round(sum(breakdown.values()), _DIGITS), "breakdown": breakdown}

    return [
        row("A step without points", 0, False),
        row(f"Score {example_score} point{'s' if example_score != 1 else ''}", example_score, False),
        row("Lose the game (ball lost, player hit)", 0, True),
        row("Cut off by max_episode_steps", 0, False),
    ]
