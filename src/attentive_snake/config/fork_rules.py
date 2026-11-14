"""Which settings a run continued from another run's checkpoint (a fork) must share with its parent.

The parent's weights must fit the new network and keep their meaning, so the
whole ``[model]``, the board size, the snake's start length and the observation
layout are locked. Everything else (rewards, training settings, the number of
parallel games, the starvation limit) may change.
"""

import dataclasses

from attentive_snake.config.experiment import ExperimentConfig
from attentive_snake.models.config import ModelConfig

_MODEL_REASON = "the parent's weights only fit the same network"
_INPUT_REASON = "it changes the observation the parent's weights were trained on"

_LOCKED_GAME = {
    "width": "the board size fixes the network's input size",
    "height": "the board size fixes the network's input size",
    "initial_length": "it changes the situations the parent's weights were trained on",
}
_LOCKED_ENV = {"history_length": _INPUT_REASON, "history_actions": _INPUT_REASON}


def locked_fields() -> dict[str, str]:
    """``{"section.field": reason}`` for every setting a fork must take unchanged from its parent."""
    model = {f"model.{field.name}": _MODEL_REASON for field in dataclasses.fields(ModelConfig)}
    game = {f"game.{name}": reason for name, reason in _LOCKED_GAME.items()}
    env = {f"env.{name}": reason for name, reason in _LOCKED_ENV.items()}
    return model | game | env


def locked_field_changes(parent: ExperimentConfig, child: ExperimentConfig) -> list[str]:
    """The locked ``"section.field"`` names whose value differs between ``parent`` and ``child`` (empty if none)."""
    return [key for key in locked_fields() if _value(parent, key) != _value(child, key)]


def require_forkable(parent: ExperimentConfig, child: ExperimentConfig) -> None:
    """``ValueError`` naming each locked field that ``child`` changed, with the reason it is locked."""
    changed = locked_field_changes(parent, child)
    if changed:
        reasons = locked_fields()
        details = "; ".join(
            f"{key} ({_value(parent, key)} -> {_value(child, key)}): {reasons[key]}" for key in changed
        )
        raise ValueError(f"a continued run must keep these settings of its parent: {details}.")


def _value(config: ExperimentConfig, key: str) -> object:
    section, name = key.split(".")
    return getattr(getattr(config, section), name)
