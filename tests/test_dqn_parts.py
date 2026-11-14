"""DQN's parts one by one: replay buffer, next observations across auto-reset, epsilon, collector, target network, loss, learner schedules."""

import numpy as np
import pytest
import torch
from torch import nn

from attentive_snake.env.types import EpisodeSummary, VecEnvStep
from attentive_snake.models.types import QValues
from attentive_snake.training.algorithm import LearnerSetup
from attentive_snake.training.config import TrainConfig
from attentive_snake.training.dqn.beta import linear_beta
from attentive_snake.training.dqn.collector import EpsilonGreedyCollector, transition_dones
from attentive_snake.training.next_observations import next_observations
from attentive_snake.training.dqn.config import DqnConfig
from attentive_snake.training.dqn.epsilon import linear_epsilon
from attentive_snake.training.dqn.learner import DqnLearner
from attentive_snake.training.dqn.loss import dqn_loss, dqn_targets
from attentive_snake.training.dqn.replay import Transitions
from attentive_snake.training.dqn.target import TargetNetwork
from attentive_snake.training.dqn.uniform_replay import UniformReplayBuffer

CPU = torch.device("cpu")


def transitions(start: int, count: int, shape=(2,)) -> Transitions:
    """``count`` transitions whose every number is its serial number ``start + i``."""
    serial = np.arange(start, start + count, dtype=np.float32)
    return Transitions(
        observations=np.broadcast_to(serial.reshape(-1, *([1] * len(shape))), (count, *shape)).copy(),
        actions=serial.astype(np.int64),
        rewards=serial,
        next_observations=np.broadcast_to((serial + 0.5).reshape(-1, *([1] * len(shape))), (count, *shape)).copy(),
        dones=np.zeros(count, dtype=np.float32),
    )


# --------------------------------------------------------------------------- replay buffer


def test_uniform_buffer_wraps_around_and_keeps_the_latest() -> None:
    buffer = UniformReplayBuffer(5, (2,), np.random.default_rng(0))
    assert len(buffer) == 0
    buffer.add(transitions(0, 3))
    assert len(buffer) == 3
    buffer.add(transitions(3, 4))  # 7 added, capacity 5: serials 2..6 remain
    assert len(buffer) == 5
    sample = buffer.sample(200, CPU)
    assert set(sample.rewards.tolist()) == {2.0, 3.0, 4.0, 5.0, 6.0}
    # Every field of one slot stays one transition.
    assert torch.equal(sample.observations[:, 0], sample.rewards) and torch.equal(sample.actions.float(), sample.rewards)
    assert torch.equal(sample.next_observations[:, 1], sample.rewards + 0.5)


def test_uniform_sample_shapes_weights_and_indices() -> None:
    buffer = UniformReplayBuffer(10, (3, 4, 4), np.random.default_rng(1))
    buffer.add(transitions(0, 6, shape=(3, 4, 4)))
    sample = buffer.sample(8, CPU)
    assert sample.observations.shape == sample.next_observations.shape == (8, 3, 4, 4)
    assert sample.observations.dtype == torch.float32 and sample.actions.dtype == torch.int64
    assert sample.rewards.shape == sample.dones.shape == sample.weights.shape == (8,)
    assert torch.equal(sample.weights, torch.ones(8)) and sample.indices.shape == (8,)
    assert np.all((0 <= sample.indices) & (sample.indices < 6))
    buffer.update_priorities(sample.indices, np.ones(8))  # a no-op for uniform sampling
    again = UniformReplayBuffer(10, (3, 4, 4), np.random.default_rng(1))
    again.add(transitions(0, 6, shape=(3, 4, 4)))
    assert np.array_equal(again.sample(8, CPU).indices, sample.indices)  # same seed, same batches
    with pytest.raises(ValueError, match="empty"):
        UniformReplayBuffer(4, (1,), np.random.default_rng(0)).sample(2, CPU)


# --------------------------------------------------------------------------- transitions across auto-reset


def vec_step(observations, terminated, truncated, finals) -> VecEnvStep:
    n = len(observations)
    return VecEnvStep(
        observations=np.asarray(observations, dtype=np.float32).reshape(n, 1),
        rewards=np.zeros(n, dtype=np.float32),
        terminated=np.asarray(terminated),
        truncated=np.asarray(truncated),
        finished_episodes=(),
        final_observations={i: np.asarray([value], dtype=np.float32) for i, value in finals.items()},
    )


def test_next_observation_is_the_final_one_never_the_reset_one() -> None:
    # env 0 goes on, env 1 ended by its rules, env 2 was cut by a limit; 1 and 2 already show a new episode (0.0).
    step = vec_step([5.0, 0.0, 0.0], [False, True, False], [False, False, True], {1: 9.0, 2: 7.0})
    assert next_observations(step)[:, 0].tolist() == [5.0, 9.0, 7.0]
    assert step.observations[:, 0].tolist() == [5.0, 0.0, 0.0]  # the env's own array is not changed


def test_dones_count_cuts_as_ends_unless_bootstrapping() -> None:
    step = vec_step([0.0] * 4, [False, True, False, True], [False, False, True, True], {1: 1.0, 2: 2.0, 3: 3.0})
    assert transition_dones(step, bootstrap_truncated=False).tolist() == [0.0, 1.0, 1.0, 1.0]
    assert transition_dones(step, bootstrap_truncated=True).tolist() == [0.0, 1.0, 0.0, 1.0]  # a rules ending wins


class CountdownVecEnv:
    """Each env's observation is its step within the episode; episodes last ``length`` steps and are cut (truncated)."""

    observation_shape = (1,)
    num_actions = 3

    def __init__(self, num_envs: int = 2, length: int = 3) -> None:
        self.num_envs, self._length = num_envs, length
        self._t = np.zeros(num_envs)
        self.actions: list[np.ndarray] = []

    def reset(self, seed=None):
        self._t[:] = 0
        return self._t.reshape(-1, 1).astype(np.float32)

    def step(self, actions):
        self.actions.append(np.asarray(actions))
        self._t += 1
        ended = self._t >= self._length
        finals = {int(i): np.array([self._t[i]], dtype=np.float32) for i in np.flatnonzero(ended)}
        episodes = tuple(EpisodeSummary(int(i), float(self._t[i]), int(self._t[i]), 1.0, "timeout") for i in np.flatnonzero(ended))
        self._t[ended] = 0
        return VecEnvStep(
            observations=self._t.reshape(-1, 1).astype(np.float32),
            rewards=np.ones(self.num_envs, dtype=np.float32),
            terminated=np.zeros(self.num_envs, dtype=bool),
            truncated=ended,
            finished_episodes=episodes,
            final_observations=finals,
        )


class FixedQ(nn.Module):
    """Q-values that prefer action 2 everywhere, plus one trainable number."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([0.0, 0.5, 1.0]))
        self.scale = nn.Linear(1, 1)

    def forward(self, observations):
        return QValues(self.bias + 0.0 * self.scale(observations))


def test_collector_stores_true_next_states_and_explores_by_epsilon() -> None:
    env = CountdownVecEnv(num_envs=2, length=3)
    collector = EpsilonGreedyCollector(env, env.reset(), np.random.default_rng(0), CPU, bootstrap_truncated=False)
    policy = FixedQ().train()
    steps = [collector.step(policy, epsilon=0.0) for _ in range(3)]
    assert policy.training  # mode restored
    assert all(np.all(step.transitions.actions == 2) for step in steps)  # greedy
    observed = [(s.transitions.observations[0, 0], s.transitions.next_observations[0, 0], s.transitions.dones[0]) for s in steps]
    assert observed == [(0.0, 1.0, 0.0), (1.0, 2.0, 0.0), (2.0, 3.0, 1.0)]  # the cut step keeps its final state, 3
    assert len(steps[2].finished_episodes) == 2
    after_reset = collector.step(policy, epsilon=0.0).transitions
    assert after_reset.observations[0, 0] == 0.0  # the next transition starts from the reset observation
    explored = np.concatenate([collector.step(policy, epsilon=1.0).transitions.actions for _ in range(100)])
    assert set(explored.tolist()) == {0, 1, 2}


# --------------------------------------------------------------------------- epsilon


def test_epsilon_falls_linearly_then_stays() -> None:
    assert linear_epsilon(1.0, 0.1, 0.5, 0, 1000) == 1.0
    assert linear_epsilon(1.0, 0.1, 0.5, 250, 1000) == pytest.approx(0.55)
    assert linear_epsilon(1.0, 0.1, 0.5, 500, 1000) == 0.1
    assert linear_epsilon(1.0, 0.1, 0.5, 900, 1000) == 0.1
    assert linear_epsilon(1.0, 0.1, 0.0, 0, 1000) == 0.1  # no decay period: the end value from the start


# --------------------------------------------------------------------------- target network


def test_target_network_is_a_frozen_copy_until_synced() -> None:
    online = nn.Linear(2, 2)
    target = TargetNetwork(online)
    assert not any(p.requires_grad for p in target.network.parameters()) and not target.network.training
    with torch.no_grad():
        online.weight.add_(1.0)
    assert not torch.equal(target.network.weight, online.weight)
    target.sync(online)
    assert torch.equal(target.network.weight, online.weight)
    restored = TargetNetwork(nn.Linear(2, 2))
    restored.load_state_dict(target.state_dict())
    assert torch.equal(restored.network.weight, online.weight)


# --------------------------------------------------------------------------- loss


def test_double_dqn_targets_and_loss_by_hand() -> None:
    q_values = torch.tensor([[1.0, 2.0], [0.0, 3.0], [4.0, 0.5]], requires_grad=True)
    actions = torch.tensor([1, 0, 0])
    rewards = torch.tensor([1.0, -1.0, 0.5])
    # sample 0 goes on; sample 1 terminal; sample 2 cut by a limit and treated as an end (done = 1)
    dones = torch.tensor([0.0, 1.0, 1.0])
    next_online = torch.tensor([[0.0, 9.0], [5.0, 0.0], [1.0, 2.0]])  # picks actions 1, 0, 1
    next_target = torch.tensor([[3.0, 2.0], [7.0, 7.0], [8.0, 6.0]])
    gamma = 0.5
    # Double: values of the online's picks under the target: 2.0 (s0); s1, s2 are done.
    double = dqn_targets(rewards, dones, next_target, next_online, gamma)
    assert double.tolist() == [1.0 + 0.5 * 2.0, -1.0, 0.5]
    # Plain DQN: the target's own max, 3.0.
    assert dqn_targets(rewards, dones, next_target, None, gamma).tolist() == [2.5, -1.0, 0.5]
    # The same cut sample bootstrapped (done = 0) uses the target's value of the online pick (6.0).
    assert dqn_targets(rewards, torch.zeros(3), next_target, next_online, gamma)[2] == 0.5 + 0.5 * 6.0

    weights = torch.tensor([1.0, 2.0, 0.5])
    terms = dqn_loss(q_values, actions, rewards, dones, next_target, next_online, gamma, weights)
    # chosen = [2.0, 0.0, 4.0]; td = [0.0, 1.0, 3.5]; Huber = [0, 0.5, 3.0]
    assert terms.td_errors.tolist() == [0.0, 1.0, 3.5]
    assert float(terms.loss.detach()) == pytest.approx((0.0 + 2.0 * 0.5 + 0.5 * 3.0) / 3)
    assert float(terms.q_mean) == pytest.approx(2.0)
    terms.loss.backward()
    assert q_values.grad[:, 1].tolist()[1:] == [0.0, 0.0] and q_values.grad[0].tolist() == [0.0, 0.0]


# --------------------------------------------------------------------------- learner schedules


class TinyQ(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 3)

    def forward(self, observations):
        return QValues(self.linear(observations))


def learner(num_envs: int = 4, **dqn) -> tuple[DqnLearner, CountdownVecEnv]:
    torch.manual_seed(0)
    env = CountdownVecEnv(num_envs=num_envs, length=5)
    settings = DqnConfig(**({"buffer_size": 100, "batch_size": 4, "learning_starts": 8} | dqn))
    train = TrainConfig(algorithm="dqn", total_env_steps=400, device="cpu")
    return DqnLearner(LearnerSetup(TinyQ(), env, train, settings, CPU)), env


def test_gradient_steps_follow_train_frequency_in_env_steps_after_learning_starts() -> None:
    dqn, _ = learner(num_envs=4, train_frequency=2, gradient_steps=3, learning_starts=8)
    counts = []
    for i in range(4):
        dqn.iterate(env_steps=4 * i)
        counts.append(dqn.report()["gradient_steps"])
    # 4 env steps per iteration = 2 rounds of 3; nothing before the buffer holds 8 transitions.
    assert counts == [0.0, 6.0, 12.0, 18.0]
    slow, _ = learner(num_envs=4, train_frequency=10, learning_starts=0)
    for i in range(5):
        slow.iterate(env_steps=4 * i)
    assert slow.report()["gradient_steps"] == 2.0  # passed 10 and 20 by env step 20


def test_target_syncs_every_interval_and_epsilon_decays() -> None:
    dqn, _ = learner(num_envs=4, target_update_interval=8, learning_starts=0, epsilon_fraction=0.1)
    synced = []
    for i in range(4):
        before = dqn.extra_state()["target"]["linear.weight"].clone()
        dqn.iterate(env_steps=4 * i)
        synced.append(not torch.equal(before, dqn.extra_state()["target"]["linear.weight"]))
    assert synced == [False, True, False, True]
    assert torch.equal(dqn.extra_state()["target"]["linear.weight"], dqn.policy.linear.weight)
    record = dqn.report()
    assert record["epsilon"] == pytest.approx(linear_epsilon(1.0, 0.05, 0.1, 12, 400))
    assert {"loss", "q_mean", "td_error", "replay_size"} <= set(record) and record["replay_size"] == 16.0
    assert set(dqn.report()) == {"epsilon", "replay_size", "gradient_steps"}  # no gradient step since the last report


def test_prioritized_learner_trains_reports_priorities_and_anneals_beta() -> None:
    dqn, _ = learner(num_envs=4, learning_starts=8, prioritized=True, priority_beta_start=0.4)
    for i in range(4):
        dqn.iterate(env_steps=4 * i)
    record = dqn.report()
    assert record["gradient_steps"] > 0 and record["priority_max"] > 0 and record["priority_mean"] > 0
    assert record["priority_beta"] == pytest.approx(linear_beta(0.4, 16, 400))  # after 16 of 400 env steps
    assert learner(prioritized=False)[0].report().keys() == {"epsilon", "replay_size", "gradient_steps"}
