"""The rule for which settings a forked run may change."""

import dataclasses

import pytest

from attentive_snake.config import ExperimentConfig
from attentive_snake.config.fork_rules import locked_field_changes, locked_fields, require_forkable
from attentive_snake.models.config import ModelConfig


def changed(config: ExperimentConfig, section: str, **values) -> ExperimentConfig:
    return dataclasses.replace(config, **{section: dataclasses.replace(getattr(config, section), **values)})


def test_locked_fields_cover_the_model_board_and_observation_layout() -> None:
    locked = locked_fields()
    assert {f"model.{field.name}" for field in dataclasses.fields(ModelConfig)} <= set(locked)
    assert {"game.width", "game.height", "game.initial_length", "env.history_length", "env.history_actions"} <= set(locked)
    assert not {key for key in locked if key.startswith(("reward.", "train."))}
    assert "env.num_envs" not in locked and "game.max_steps_without_food" not in locked
    assert all(reason for reason in locked.values())


@pytest.mark.parametrize(
    "section, values",
    [
        ("reward", {"food": 5.0, "step": 0.0}),
        ("train", {"learning_rate": 1e-4, "total_env_steps": 123, "entropy_coef": 0.0, "seed": 3, "keep_checkpoints": False}),
        ("env", {"num_envs": 3}),
        ("game", {"max_steps_without_food": 7}),
    ],
)
def test_changeable_settings_pass(section: str, values: dict) -> None:
    parent = ExperimentConfig()
    child = changed(parent, section, **values)
    assert locked_field_changes(parent, child) == []
    require_forkable(parent, child)


@pytest.mark.parametrize(
    "section, values, key",
    [
        ("model", {"kind": "cnn"}, "model.kind"),
        ("model", {"d_model": 32}, "model.d_model"),
        ("model", {"conv_channels": 8}, "model.conv_channels"),
        ("game", {"width": 6}, "game.width"),
        ("game", {"height": 6}, "game.height"),
        ("game", {"initial_length": 2}, "game.initial_length"),
        ("env", {"history_length": 2}, "env.history_length"),
        ("env", {"history_actions": True}, "env.history_actions"),
    ],
)
def test_locked_settings_are_named_in_the_error(section: str, values: dict, key: str) -> None:
    parent = ExperimentConfig()
    child = changed(parent, section, **values)
    assert locked_field_changes(parent, child) == [key]
    with pytest.raises(ValueError, match=key.replace(".", r"\.")):
        require_forkable(parent, child)


def test_several_changes_are_all_reported() -> None:
    parent = ExperimentConfig()
    child = changed(changed(parent, "game", width=6), "model", num_layers=1)
    assert set(locked_field_changes(parent, child)) == {"game.width", "model.num_layers"}
