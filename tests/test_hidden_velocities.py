"""Lunar Lander's ``hide_velocities`` rule: zeros for the velocity numbers, the true state for everything else."""

import dataclasses

import numpy as np
import pytest

from attentive_snake.config import ExperimentConfig, load_experiment, save_experiment
from attentive_snake.config.fork_rules import locked_field_changes, locked_fields
from attentive_snake.config.schema import experiment_schema
from attentive_snake.games.lunar_lander.config import LanderRules
from attentive_snake.games.lunar_lander.game import GAME, OBSERVATION_LABELS
from attentive_snake.games.lunar_lander.hidden_velocities import VELOCITY_INDICES
from attentive_snake.games.lunar_lander.reward import LanderReward
from attentive_snake.games.validation import spec_problems


def make(hide: bool, seed: int = 0):
    return GAME.make_env(LanderRules(hide_velocities=hide), LanderReward(), seed)


def rollout(env, actions):
    """Reset and play ``actions`` (cycled) to the end of the episode: (observations, steps)."""
    observations, steps = [env.reset(5)], []
    while not steps or not (steps[-1].terminated or steps[-1].truncated):
        steps.append(env.step(actions[len(steps) % len(actions)]))
        observations.append(steps[-1].observation)
    return observations, steps


def test_the_velocity_indices_are_the_velocities_in_the_labels():
    assert [OBSERVATION_LABELS[i] for i in VELOCITY_INDICES] == ["vx", "vy", "angular velocity"]


def test_hidden_velocities_are_zero_and_everything_else_is_unchanged():
    shown, _ = rollout(make(False), [2, 0, 1, 3])
    hidden, _ = rollout(make(True), [2, 0, 1, 3])
    shown, hidden = np.array(shown), np.array(hidden)
    assert shown.shape == hidden.shape and hidden.dtype == np.float32
    assert np.abs(shown[:, list(VELOCITY_INDICES)]).sum() > 0  # the velocities are not zero by themselves
    assert (hidden[:, list(VELOCITY_INDICES)] == 0).all()
    kept = [i for i in range(8) if i not in VELOCITY_INDICES]
    assert np.array_equal(shown[:, kept], hidden[:, kept])


def test_the_true_state_still_drives_reward_score_and_outcome():
    _, shown = rollout(make(False), [2, 0, 1, 3])
    _, hidden = rollout(make(True), [2, 0, 1, 3])
    assert [(s.reward, s.score, s.event, s.terminated) for s in shown] == [(s.reward, s.score, s.event, s.terminated) for s in hidden]
    assert hidden[-1].event in {"landed", "crashed", "flew_off", "timeout"}


def test_default_leaves_observations_identical_to_gymnasium():
    assert LanderRules().hide_velocities is False
    env = make(False, seed=3)
    first = env.reset(3)
    assert np.array_equal(first, GAME.make_env(LanderRules(), LanderReward(), 3).reset(3)) and first[2:4].any()


def test_the_view_still_renders_with_hidden_velocities():
    env = make(True)
    env.reset(0)
    assert env.view()["width"] == 600


def test_spec_is_still_valid_and_labels_unchanged():
    assert spec_problems(GAME, "lunar_lander") == [] and len(GAME.observation_labels) == 8


def test_config_round_trip_and_schema(tmp_path):
    config = dataclasses.replace(ExperimentConfig(game_name="lunar_lander", game=LanderRules(hide_velocities=True), reward=LanderReward()))
    save_experiment(config, tmp_path / "c.toml")
    assert load_experiment(tmp_path / "c.toml").game.hide_velocities is True
    row = {r["name"]: r for r in experiment_schema("lunar_lander")["sections"]["game"]}["hide_velocities"]
    assert (row["type"], row["default"]) == ("bool", False) and "zeros" in row["description"]


def test_locked_on_fork():
    assert "game.hide_velocities" in locked_fields("lunar_lander")
    parent = ExperimentConfig(game_name="lunar_lander", game=LanderRules(), reward=LanderReward())
    child = dataclasses.replace(parent, game=LanderRules(hide_velocities=True))
    assert locked_field_changes(parent, child) == ["game.hide_velocities"]
