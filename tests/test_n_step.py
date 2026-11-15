"""n-step targets by hand: episode ends inside the window, truncation with and without bootstrapping, two envs interleaved, the ring's boundaries."""

import numpy as np
import pytest
import torch
from torch import nn

from attentive_snake.models.auxiliary import AuxiliaryLoss
from attentive_snake.models.types import QValues
from attentive_snake.training.auxiliary import AuxiliaryTraining
from attentive_snake.training.dqn.collector import EpsilonGreedyCollector
from attentive_snake.training.dqn.config import DqnConfig
from attentive_snake.training.dqn.n_step import Horizon, n_step_windows
from attentive_snake.training.dqn.prioritized_replay import PrioritizedReplayBuffer
from attentive_snake.training.dqn.replay import Transitions
from attentive_snake.training.dqn.replay_storage import ReplayStorage
from attentive_snake.training.dqn.target import TargetNetwork
from attentive_snake.training.dqn.uniform_replay import UniformReplayBuffer
from attentive_snake.training.dqn.update import dqn_gradient_step
from test_dqn_parts import CPU, CountdownVecEnv, FixedQ, learner

GAMMA = 0.5
THREE_STEPS = Horizon(3, GAMMA)
ENV0_REWARDS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]  # env 0 dies (terminated) at its step 2
ENV1_REWARD = 3.0  # env 1 is cut by a limit (truncated) at its step 1


def vector_step(t: int, bootstrap_truncated: bool) -> Transitions:
    """Step ``t`` of two envs: slot order env 0, env 1. Observations are ``10 * env + t``; the next one is ``+ 1``."""
    died, cut = t == 2, t == 1
    return Transitions(
        observations=np.array([[t], [10 + t]], dtype=np.float32),
        actions=np.array([0, 1]),
        rewards=np.array([ENV0_REWARDS[t], ENV1_REWARD], dtype=np.float32),
        next_observations=np.array([[t + 1], [10 + t + 1]], dtype=np.float32),  # the true next state, also at an end
        dones=np.array([float(died), float(cut and not bootstrap_truncated)], dtype=np.float32),
        ends=np.array([died, cut]),
    )


def filled(capacity: int, bootstrap_truncated: bool = False, horizon: Horizon = THREE_STEPS) -> ReplayStorage:
    storage = ReplayStorage(capacity, (1,), horizon)
    for t in range(6):
        storage.write(vector_step(t, bootstrap_truncated))
    return storage


def windows(storage: ReplayStorage, slots: list[int]):
    """(returns, bootstrap observation, done, discount) of the windows starting at ``slots``."""
    sample = storage.gather(np.array(slots), np.ones(len(slots), dtype=np.float32), CPU)
    return list(zip(sample.rewards.tolist(), sample.bootstrap_observations[:, 0].tolist(), sample.dones.tolist(), sample.discounts.tolist()))


def test_windows_stop_at_an_episode_end_and_at_the_newest_step() -> None:
    storage = filled(capacity=12)  # nothing overwritten: step t of env e sits in slot 2 * t + e
    # env 0: dies at t = 2. Windows from t = 0, 1, 2 end there (shorter and shorter), nothing is bootstrapped.
    assert windows(storage, [0, 2, 4]) == [
        (1.0 + 0.5 * 2.0 + 0.25 * 4.0, 3.0, 1.0, 0.125),
        (2.0 + 0.5 * 4.0, 3.0, 1.0, 0.25),
        (4.0, 3.0, 1.0, 0.5),
    ]
    # env 0's next episode (t = 3, 4, 5): a full window, then two cut short by the newest step stored (t = 5),
    # which bootstrap from the state after it (6) with gamma ** (their length).
    assert windows(storage, [6, 8, 10]) == [
        (8.0 + 0.5 * 16.0 + 0.25 * 32.0, 6.0, 0.0, 0.125),
        (16.0 + 0.5 * 32.0, 6.0, 0.0, 0.25),
        (32.0, 6.0, 0.0, 0.5),
    ]


@pytest.mark.parametrize("bootstrap_truncated", [False, True])
def test_a_truncated_step_ends_the_window_and_is_bootstrapped_only_if_asked(bootstrap_truncated: bool) -> None:
    storage = filled(capacity=12, bootstrap_truncated=bootstrap_truncated)
    done = 0.0 if bootstrap_truncated else 1.0
    # env 1 (odd slots) is cut at t = 1: the window from t = 0 ends there and never reaches the next episode's rewards.
    # Its bootstrap state is the cut step's own final observation (12), two steps on: gamma ** 2.
    assert windows(storage, [1, 3]) == [(3.0 + 0.5 * 3.0, 12.0, done, 0.25), (3.0, 12.0, done, 0.5)]
    # After the cut, env 1 runs on: a full three-step window from t = 2, never mixed with env 0's rewards.
    assert windows(storage, [5]) == [(3.0 + 1.5 + 0.75, 15.0, 0.0, 0.125)]


def test_the_one_step_next_observation_is_kept_beside_the_bootstrap_observation() -> None:
    sample = filled(capacity=12).gather(np.array([0, 6, 5]), np.ones(3, dtype=np.float32), CPU)
    assert sample.observations[:, 0].tolist() == [0.0, 3.0, 12.0]
    assert sample.next_observations[:, 0].tolist() == [1.0, 4.0, 13.0]  # one step on, whatever n is
    assert sample.bootstrap_observations[:, 0].tolist() == [3.0, 6.0, 15.0]
    assert sample.actions.tolist() == [0, 0, 1]


@pytest.mark.parametrize("capacity", [8, 7])  # a multiple of the two envs, and not
def test_windows_follow_an_env_across_the_ring_wrap_and_never_into_overwritten_slots(capacity: int) -> None:
    storage = filled(capacity)  # 12 transitions written: number i = 2 * t + env sits in slot i % capacity
    slot = lambda t, env: (2 * t + env) % capacity
    assert slot(5, 0) < slot(3, 0)  # env 0's window from t = 3 crosses the end of the arrays
    assert windows(storage, [slot(3, 0), slot(4, 0), slot(5, 0)]) == [
        (8.0 + 0.5 * 16.0 + 0.25 * 32.0, 6.0, 0.0, 0.125),
        (16.0 + 0.5 * 32.0, 6.0, 0.0, 0.25),
        (32.0, 6.0, 0.0, 0.5),  # the newest step: the slots after it hold the env's oldest steps, not its future
    ]
    assert windows(storage, [slot(4, 1), slot(5, 1)]) == [(3.0 + 1.5, 16.0, 0.0, 0.25), (3.0, 16.0, 0.0, 0.5)]


def test_one_step_is_the_stored_transition_and_n_must_be_positive() -> None:
    one = filled(capacity=12, horizon=Horizon(1, GAMMA))
    sample = one.gather(np.arange(12), np.ones(12, dtype=np.float32), CPU)
    assert sample.rewards.tolist() == [reward for t in range(6) for reward in (ENV0_REWARDS[t], ENV1_REWARD)]
    assert sample.bootstrap_observations is sample.next_observations and set(sample.discounts.tolist()) == {0.5}
    assert sample.dones.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0] + [0.0] * 6
    with pytest.raises(ValueError, match="at least 1"):
        Horizon(0, GAMMA)
    with pytest.raises(ValueError, match="n_step must be positive"):
        DqnConfig(n_step=0)


def test_the_pure_function_on_plain_arrays() -> None:
    rewards = np.array([1.0, 10.0, 2.0, 20.0, 4.0, 40.0], dtype=np.float32)  # two envs interleaved, three steps
    ends = np.array([False, True, False, False, False, False])  # env 1's first step ended its episode
    starts = np.array([0, 1, 3])
    result = n_step_windows(rewards, ends, starts, later_steps=np.array([2, 2, 1]), stride=2, horizon=THREE_STEPS)
    assert result.returns.tolist() == [1.0 + 1.0 + 1.0, 10.0, 20.0 + 20.0]
    assert result.last.tolist() == [4, 1, 5] and result.discounts.tolist() == [0.125, 0.5, 0.25]


def test_n_step_writes_must_all_have_one_transition_per_env() -> None:
    storage = filled(capacity=12)
    lone = vector_step(0, False)
    with pytest.raises(ValueError, match="same number of transitions"):
        storage.write(Transitions(lone.observations[:1], lone.actions[:1], lone.rewards[:1], lone.next_observations[:1], lone.dones[:1], lone.ends[:1]))


@pytest.mark.parametrize("prioritized", [False, True])
def test_both_buffers_sample_n_step_windows(prioritized: bool) -> None:
    rng = np.random.default_rng(0)
    buffer = PrioritizedReplayBuffer(12, (1,), rng, 0.6, 1e-6, lambda: 1.0, THREE_STEPS) if prioritized else UniformReplayBuffer(12, (1,), rng, THREE_STEPS)
    for t in range(6):
        buffer.add(vector_step(t, False))
    sample = buffer.sample(64, CPU)
    expected = dict(zip(range(12), windows(filled(capacity=12), list(range(12)))))
    drawn = list(zip(sample.rewards.tolist(), sample.bootstrap_observations[:, 0].tolist(), sample.dones.tolist(), sample.discounts.tolist()))
    assert drawn == [expected[int(slot)] for slot in sample.indices]
    buffer.update_priorities(sample.indices, np.ones(64))  # priorities stay per slot


def test_the_collector_marks_every_episode_end_bootstrapped_or_not() -> None:
    env = CountdownVecEnv(num_envs=2, length=3)  # every episode is cut after 3 steps
    collector = EpsilonGreedyCollector(env, env.reset(), np.random.default_rng(0), CPU, bootstrap_truncated=True)
    steps = [collector.step(FixedQ(), epsilon=0.0).transitions for _ in range(3)]
    assert [step.ends.tolist() for step in steps] == [[False, False], [False, False], [True, True]]
    assert [step.dones.tolist() for step in steps] == [[0.0, 0.0]] * 3  # bootstrapped: not done, but the window must stop


# --------------------------------------------------------------------------- the gradient step


class LinearQ(nn.Module):
    """Q(s) = [s, 2 s] and a record of the next observations ``trained_values`` was asked about."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.asked: list[torch.Tensor] = []

    def forward(self, observations):
        return QValues(observations * self.weight)

    def trained_values(self, observations, actions, next_observations):
        self.asked.append(next_observations)
        return None


class RecordingObjective:
    """An auxiliary objective that only records the next observations it is given."""

    def __init__(self) -> None:
        self.seen: list[torch.Tensor] = []

    def loss(self, batch):
        self.seen.append(batch.next_observations)
        return AuxiliaryLoss(torch.zeros(()), {})

    def after_step(self) -> None: ...


def test_the_gradient_step_bootstraps_from_the_window_end_and_keeps_one_step_states_for_the_policy() -> None:
    policy = LinearQ()
    objective = RecordingObjective()
    sample = filled(capacity=12).gather(np.array([0, 6, 5]), np.ones(3, dtype=np.float32), CPU)
    stats = dqn_gradient_step(policy, TargetNetwork(policy), torch.optim.SGD(policy.parameters(), lr=0.0), sample, True, 10.0, AuxiliaryTraining(objective))
    # Q(s, a) = [0, 3, 2 * 12]; targets = return + gamma ** k * (1 - done) * max Q(bootstrap state) = 2 * state:
    # slot 0: 3 (died);  slot 6: 24 + 0.125 * 12;  slot 5: 5.25 + 0.125 * 30.
    assert stats.td_errors.tolist() == [0.0 - 3.0, 3.0 - 25.5, 24.0 - 9.0]
    assert policy.asked[0][:, 0].tolist() == objective.seen[0][:, 0].tolist() == [1.0, 4.0, 13.0]  # the true next states


def test_a_learner_trains_with_n_step_targets_prioritized_or_not() -> None:
    for prioritized in (False, True):
        dqn, _ = learner(num_envs=4, learning_starts=8, n_step=3, prioritized=prioritized)
        for i in range(6):
            dqn.iterate(env_steps=4 * i)
        record = dqn.report()
        assert record["gradient_steps"] > 0 and np.isfinite(record["loss"])
