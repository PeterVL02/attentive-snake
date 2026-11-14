"""Which settings a run continued from another run's checkpoint (a fork) must share with its parent.

The parent's weights must fit the new network and keep their meaning, so the game itself,
the whole ``[model]``, the observation layout (``env.history_length``,
``env.history_actions``) and the game rules its ``GameSpec.locked_rules`` names (for
Snake: the board size and the start length) are locked. Everything else (rewards,
training settings, the number of parallel games, the other rules) may change.
"""

import dataclasses

from attentive_snake.config.experiment import GAME_NAME_KEY, ExperimentConfig
from attentive_snake.games.registry import DEFAULT_GAME, get_game
from attentive_snake.models.config import ModelConfig

GAME_KEY = f"game.{GAME_NAME_KEY}"
_GAME_REASON = "the parent's weights were trained on that game"
_MODEL_REASON = "the parent's weights only fit the same network"
_INPUT_REASON = "it changes the observation the parent's weights were trained on"
_LOCKED_ENV = {"history_length": _INPUT_REASON, "history_actions": _INPUT_REASON}


def locked_fields(game: str = DEFAULT_GAME) -> dict[str, str]:
    """``{"section.field": reason}`` for every setting a fork of a ``game`` run must take unchanged from its parent."""
    model = {f"model.{field.name}": _MODEL_REASON for field in dataclasses.fields(ModelConfig)}
    rules = {f"game.{name}": reason for name, reason in get_game(game).locked_rules.items()}
    env = {f"env.{name}": reason for name, reason in _LOCKED_ENV.items()}
    return {GAME_KEY: _GAME_REASON} | model | rules | env


def locked_field_changes(parent: ExperimentConfig, child: ExperimentConfig) -> list[str]:
    """The locked ``"section.field"`` names whose value differs between ``parent`` and ``child`` (empty if none).

    A different game is reported alone (``["game.name"]``): the other fields cannot be compared.
    """
    if parent.game_name != child.game_name:
        return [GAME_KEY]
    return [key for key in locked_fields(parent.game_name) if _value(parent, key) != _value(child, key)]


def require_forkable(parent: ExperimentConfig, child: ExperimentConfig) -> None:
    """``ValueError`` naming each locked field that ``child`` changed, with the reason it is locked."""
    changed = locked_field_changes(parent, child)
    if changed:
        reasons = locked_fields(parent.game_name)
        details = "; ".join(
            f"{key} ({_value(parent, key)} -> {_value(child, key)}): {reasons[key]}" for key in changed
        )
        raise ValueError(f"a continued run must keep these settings of its parent: {details}.")


def _value(config: ExperimentConfig, key: str) -> object:
    if key == GAME_KEY:
        return config.game_name
    section, name = key.split(".")
    return getattr(getattr(config, section), name)
