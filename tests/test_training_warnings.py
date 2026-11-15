"""A game's warnings about the config it is trained with: Snake's step-cost rule and the generic lookup."""

import dataclasses

import pytest

from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.games.snake.reward import SnakeRewardConfig
from attentive_snake.games.snake.warnings import step_cost_warnings
from attentive_snake.games.validation import spec_problems
from attentive_snake.games.registry import get_game
from attentive_snake.training_warnings import training_warnings


def warnings_for(gamma: float = 0.99, **reward) -> tuple[str, ...]:
    return step_cost_warnings(None, SnakeRewardConfig(**reward), gamma)


def test_default_reward_warns_with_the_numbers():
    (text,) = warnings_for()
    assert "step cost of -0.01" in text and "gamma 0.99" in text
    assert "-1.00 against a death penalty of -1.00" in text and "-0.001" in text


@pytest.mark.parametrize(
    "gamma, reward",
    [(0.99, {"step": -0.001}), (0.99, {"step": 0.0}), (0.99, {"death": -2.0}), (0.99, {"death": 0.0}), (0.9, {})],
)
def test_no_warning(gamma: float, reward: dict):
    assert warnings_for(gamma, **reward) == ()


def test_gamma_closer_to_one_warns():
    assert warnings_for(0.999, death=-2.0)  # 0.01 / 0.001 = 10 >= 2
    assert warnings_for(1.0, step=-0.0001)
    assert warnings_for(1.0, step=0.0) == ()


def test_generic_lookup_per_game():
    snake = experiment_from_tables({})
    assert len(training_warnings(snake)) == 1
    assert training_warnings(dataclasses.replace(snake, reward=SnakeRewardConfig(step=-0.001))) == ()
    assert training_warnings(experiment_from_tables({"game": {"name": "lunar_lander"}})) == ()


def test_spec_validation_of_the_field():
    snake = get_game("snake")
    assert spec_problems(snake, "snake") == []
    assert spec_problems(dataclasses.replace(snake, training_warnings="nope"), "snake") == ["training_warnings must be callable"]
    assert get_game("lunar_lander").training_warnings is None
