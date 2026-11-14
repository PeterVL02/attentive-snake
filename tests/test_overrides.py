"""section.key=value overrides of an experiment config."""

import pytest

from attentive_snake.config import ExperimentConfig
from attentive_snake.config.overrides import apply_overrides, parse_override


@pytest.mark.parametrize(
    "text, expected",
    [
        ("train.learning_rate=1e-4", ("train", "learning_rate", 1e-4)),
        ("reward.food = 2", ("reward", "food", 2)),
        ("train.anneal_lr=false", ("train", "anneal_lr", False)),
        ('train.device="cuda:0"', ("train", "device", "cuda:0")),
        ("train.device=cpu", ("train", "device", "cpu")),
    ],
)
def test_parse_override(text, expected) -> None:
    assert parse_override(text) == expected


@pytest.mark.parametrize("text", ["learning_rate=1", "train.learning_rate", "=1", "train.=1", ".x=1", "a.b.c=1"])
def test_parse_override_rejects_malformed(text) -> None:
    with pytest.raises(ValueError, match="section.key=value"):
        parse_override(text)


def test_apply_overrides_changes_only_the_named_settings() -> None:
    base = ExperimentConfig()
    config = apply_overrides(base, [("train", "learning_rate", 1e-4), ("reward", "food", 2), ("game", "max_steps_without_food", 9)])
    assert config.train.learning_rate == 1e-4 and config.reward.food == 2.0
    assert config.game.max_steps_without_food == 9
    assert config.model == base.model and config.env == base.env
    assert apply_overrides(base, []) == base


def test_apply_overrides_validates() -> None:
    with pytest.raises(ValueError, match="no_such_key"):
        apply_overrides(ExperimentConfig(), [("train", "no_such_key", 1)])
    with pytest.raises(ValueError, match="section"):
        apply_overrides(ExperimentConfig(), [("nope", "x", 1)])
    with pytest.raises(ValueError, match="learning_rate"):
        apply_overrides(ExperimentConfig(), [("train", "learning_rate", -1.0)])
