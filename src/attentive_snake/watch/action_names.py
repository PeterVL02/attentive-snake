"""Names of a game's actions, for showing what the network says about each one."""

from attentive_snake.env.types import GameEnv
from attentive_snake.games.spec import GameSpec


def action_names(game: GameSpec, env: GameEnv) -> list[str]:
    """The env's own ``action_labels`` if it has them (a game whose actions depend on its rules, see
    ``GameEnv``), else the game's ``action_labels``, else ``"action 0"``, ..."""
    labels = getattr(env, "action_labels", None) or game.action_labels
    if labels is not None:
        return list(labels)
    return [f"action {i}" for i in range(env.num_actions)]
