"""Bootstrapping truncated episodes (``train.bootstrap_truncated``): the final observation in
``VecEnvStep``, the reward correction, its effect on GAE, the collector with the flag on and off,
and the config field."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from attentive_snake.config import load_experiment, save_experiment
from attentive_snake.config.experiment_io import experiment_from_tables
from attentive_snake.config.fork_rules import locked_fields
from attentive_snake.config.layout import default_config_path
from attentive_snake.config.overrides import apply_overrides, parse_override
from attentive_snake.env.history_env import HistoryEnv
from attentive_snake.env.types import VecEnvStep
from attentive_snake.env.vec_env import SyncVecEnv
from attentive_snake.models.types import PolicyOutput
from attentive_snake.training import TrainConfig
from attentive_snake.training.ppo.collector import RolloutCollector
from attentive_snake.training.ppo.rollout import RolloutBuffer, compute_gae
from attentive_snake.training.ppo.truncation import TruncationBootstrap, truncated_envs
from dummy_games.walk import WalkEnv, WalkReward, WalkRules
from test_flag_off_reference import TABLES, short_training

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
RIGHT, LEFT = 1, 0


def walk_env(length: int = 4, max_steps: int = 10, history: int = 2) -> HistoryEnv:
    return HistoryEnv(WalkEnv(WalkRules(length, max_steps), WalkReward()), history, history_actions=True)


# --------------------------------------------------------------------------- VecEnvStep.final_observations


def test_sync_vec_env_keeps_the_last_observation_of_each_finished_episode() -> None:
    # env 0 walks right and wins (terminated) at step 3; env 1 walks left and is cut (truncated) at step 2.
    vec = SyncVecEnv([walk_env(max_steps=10), walk_env(max_steps=2)])
    twins = [walk_env(max_steps=10), walk_env(max_steps=2)]  # the same games stepped by hand, without auto-reset
    vec.reset(seed=0)
    for twin in twins:
        twin.reset()
    actions = np.array([RIGHT, LEFT])

    first = vec.step(actions)
    for twin, action in zip(twins, actions, strict=True):
        twin.step(int(action))
    assert first.final_observations == {}

    second = vec.step(actions)
    twins[0].step(RIGHT)
    cut = twins[1].step(LEFT)
    assert cut.truncated and second.truncated[1] and not second.terminated[1]
    assert set(second.final_observations) == {1}
    np.testing.assert_array_equal(second.final_observations[1], cut.observation)  # both stacked frames + actions
    assert second.final_observations[1].shape == vec.observation_shape
    np.testing.assert_array_equal(second.observations[1], walk_env(max_steps=2).reset())  # already the next episode
    assert not np.array_equal(second.final_observations[1], second.observations[1])

    third = vec.step(actions)
    won = twins[0].step(RIGHT)
    assert won.terminated and third.terminated[0]
    assert set(third.final_observations) == {0}
    np.testing.assert_array_equal(third.final_observations[0], won.observation)


# --------------------------------------------------------------------------- the reward correction


def make_step(observations: np.ndarray, terminated, truncated, finals: dict[int, np.ndarray]) -> VecEnvStep:
    n = observations.shape[0]
    return VecEnvStep(
        observations, np.zeros(n, np.float32), np.array(terminated), np.array(truncated), (), finals
    )


class RecordingValue:
    """Value function ``V(obs) = sum(obs)``, recording each batch it is asked for."""

    def __init__(self) -> None:
        self.batches: list[torch.Tensor] = []

    def __call__(self, observations: torch.Tensor) -> torch.Tensor:
        self.batches.append(observations.clone())
        return observations.flatten(1).sum(1)


def test_only_truncated_steps_get_gamma_times_the_value_of_their_final_observation() -> None:
    observations = np.full((4, 2), 100.0, dtype=np.float32)  # the reset observations: must not be used
    finals = {i: np.full(2, float(i + 1), dtype=np.float32) for i in range(3)}
    # env 0 truncated, env 1 terminated, env 2 both (the rules' ending wins), env 3 still playing
    step = make_step(observations, [False, True, True, False], [True, False, True, False], finals)
    assert truncated_envs(step) == [0]
    value = RecordingValue()
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])

    corrected = TruncationBootstrap(gamma=0.5).rewards(rewards, step, value)

    torch.testing.assert_close(corrected, torch.tensor([1.0 + 0.5 * 2.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(rewards, torch.tensor([1.0, 2.0, 3.0, 4.0]))  # the input is not modified
    assert len(value.batches) == 1 and value.batches[0].tolist() == [[1.0, 1.0]]


def test_several_truncated_envs_share_one_forward_pass() -> None:
    finals = {0: np.array([1.0], np.float32), 2: np.array([3.0], np.float32)}
    step = make_step(np.zeros((3, 1), np.float32), [False] * 3, [True, False, True], finals)
    value = RecordingValue()
    corrected = TruncationBootstrap(gamma=0.9).rewards(torch.zeros(3), step, value)
    torch.testing.assert_close(corrected, torch.tensor([0.9, 0.0, 2.7]))
    assert len(value.batches) == 1 and value.batches[0].shape == (2, 1)


def test_no_truncation_means_no_forward_pass_and_the_same_rewards() -> None:
    step = make_step(np.zeros((2, 1), np.float32), [True, False], [False, False], {0: np.ones(1, np.float32)})
    value = RecordingValue()
    rewards = torch.tensor([1.0, 2.0])
    assert TruncationBootstrap(gamma=0.9).rewards(rewards, step, value) is rewards
    assert value.batches == []


def test_a_truncated_env_without_final_observation_is_an_error() -> None:
    step = make_step(np.zeros((2, 1), np.float32), [False, False], [False, True], {})
    with pytest.raises(ValueError, match=r"\[1\]"):
        TruncationBootstrap(gamma=0.9).rewards(torch.zeros(2), step, RecordingValue())


def test_gae_bootstraps_the_truncated_step_by_hand() -> None:
    # One env, two steps; the second is truncated with V(final) = 4. gamma = 0.9, lambda = 0.8.
    values = torch.tensor([[1.0], [2.0]])
    dones = torch.tensor([[0.0], [1.0]])
    bootstrap = torch.tensor([10.0])  # value of the next episode's first observation: must not leak in
    step = make_step(np.zeros((1, 1), np.float32), [False], [True], {0: np.array([4.0], np.float32)})
    rewards = torch.stack([torch.tensor([1.0]), TruncationBootstrap(0.9).rewards(torch.tensor([0.5]), step, RecordingValue())])

    advantages, returns = compute_gae(rewards, values, dones, bootstrap, gamma=0.9, gae_lambda=0.8)

    # t=1: delta = 0.5 + 0.9*4 - 2 = 2.1                         -> A = 2.1, return = 4.1 = r + gamma V(final)
    # t=0: delta = 1 + 0.9*2 - 1 = 1.8, A = 1.8 + 0.72*2.1 = 3.312
    torch.testing.assert_close(advantages[:, 0], torch.tensor([3.312, 2.1]))
    torch.testing.assert_close(returns[:, 0], torch.tensor([4.312, 4.1]))
    # Without the correction (truncation as termination) the last return would be 0.5.
    plain, _ = compute_gae(torch.tensor([[1.0], [0.5]]), values, dones, bootstrap, gamma=0.9, gae_lambda=0.8)
    torch.testing.assert_close(plain[1, 0], torch.tensor(-1.5))


# --------------------------------------------------------------------------- the collector


class ProbePolicy(nn.Module):
    """Uniform logits; ``V(obs) = sum(obs) + bias``. Records the mode and grad state of every forward pass."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.25))
        self.calls: list[tuple[int, bool, bool]] = []  # (batch, training, grad enabled)

    def forward(self, observations: torch.Tensor) -> PolicyOutput:
        self.calls.append((observations.shape[0], self.training, torch.is_grad_enabled()))
        batch = observations.shape[0]
        return PolicyOutput(torch.zeros(batch, 2), observations.flatten(1).sum(1) + self.bias)


def collect(truncation: TruncationBootstrap | None, rollout_length: int = 12):
    vec = SyncVecEnv([walk_env(max_steps=3), walk_env(max_steps=5), walk_env(max_steps=4)])
    buffer = RolloutBuffer.allocate(rollout_length, vec.num_envs, vec.observation_shape, torch.device("cpu"))
    torch.manual_seed(0)
    collector = RolloutCollector(vec, buffer, vec.reset(seed=0), truncation)
    steps: list[VecEnvStep] = []
    step = vec.step
    vec.step = lambda actions: steps.append(step(actions)) or steps[-1]
    policy = ProbePolicy()
    policy.train()
    result = collector.collect(policy)
    return buffer, result, steps, policy


def test_collector_adds_the_bootstrap_only_with_the_flag_and_draws_the_same_actions() -> None:
    off_buffer, off_result, steps, off_policy = collect(None)
    on_buffer, on_result, _, on_policy = collect(TruncationBootstrap(gamma=0.9))

    assert torch.equal(on_buffer.actions, off_buffer.actions)  # no extra RNG draw
    assert torch.equal(on_buffer.dones, off_buffer.dones) and torch.equal(on_buffer.values, off_buffer.values)
    expected = off_buffer.rewards.clone()
    truncations = 0
    for t, step in enumerate(steps):
        for i in truncated_envs(step):
            expected[t, i] += 0.9 * (float(step.final_observations[i].sum()) + 0.25)
            truncations += 1
    assert truncations > 0
    torch.testing.assert_close(on_buffer.rewards, expected)
    # Logged episodes keep the game's own rewards.
    assert on_result.finished_episodes == off_result.finished_episodes

    rollout_length = off_buffer.rollout_length
    assert len(off_policy.calls) == rollout_length + 1  # one per step + the bootstrap: no extra pass when off
    steps_with_truncation = sum(1 for step in steps if truncated_envs(step))
    assert len(on_policy.calls) == rollout_length + 1 + steps_with_truncation
    assert all(not training and not grad for _, training, grad in on_policy.calls)
    assert on_policy.training  # mode restored


def test_trainer_with_the_flag_trains_differently_from_without() -> None:
    off, _ = short_training()
    tables = {**TABLES, "train": {**TABLES["train"], "bootstrap_truncated": True}}
    on, truncations = short_training(tables)
    assert truncations > 0
    assert any(not torch.equal(off[name], on[name]) for name in off)


# --------------------------------------------------------------------------- config


def test_config_field_defaults_round_trip_and_is_not_locked(tmp_path) -> None:
    assert TrainConfig().bootstrap_truncated is False
    old = experiment_from_tables({"game": {"name": "lunar_lander"}, "train": {"gamma": 0.99}})  # an old config
    assert old.train.bootstrap_truncated is False

    config = apply_overrides(old, [parse_override("train.bootstrap_truncated=true")])
    save_experiment(config, tmp_path / "config.toml")
    assert "bootstrap_truncated = true" in (tmp_path / "config.toml").read_text()
    assert load_experiment(tmp_path / "config.toml").train.bootstrap_truncated is True

    for game in ("snake", "lunar_lander"):
        assert "train.bootstrap_truncated" not in locked_fields(game)


def test_default_configs_bootstrap_lunar_lander_time_limits_but_not_snake_starvation() -> None:
    assert load_experiment(default_config_path(CONFIGS, "lunar_lander")).train.bootstrap_truncated is True
    assert load_experiment(default_config_path(CONFIGS, "snake")).train.bootstrap_truncated is False
