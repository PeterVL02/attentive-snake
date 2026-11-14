"""Names of a game's actions, for showing what the network says about each one."""

from attentive_snake.games.spec import GameSpec


def action_names(game: GameSpec, num_actions: int) -> list[str]:
    """The game's ``action_labels``, else ``"action 0"``, ..."""
    if game.action_labels is not None:
        return list(game.action_labels)
    return [f"action {i}" for i in range(num_actions)]
