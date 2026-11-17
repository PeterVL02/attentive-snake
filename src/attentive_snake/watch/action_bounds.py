"""The range of each number of a continuous action, for drawing the mean action against it when watching."""

from attentive_snake.env.action_space import action_space_of
from attentive_snake.env.types import GameEnv


def action_bounds(env: GameEnv) -> dict[str, list[float]] | None:
    """``{"low": [...], "high": [...]}`` per dimension for continuous actions; ``None`` for discrete ones."""
    space = action_space_of(env)
    if not space.continuous:
        return None
    return {"low": list(space.low), "high": list(space.high)}
