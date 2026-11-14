"""Proportional prioritized replay: priorities of new and overwritten slots, sampling frequencies, importance weights, beta."""

import numpy as np
import pytest
import torch

from attentive_snake.training.dqn.beta import linear_beta
from attentive_snake.training.dqn.config import DqnConfig
from attentive_snake.training.dqn.prioritized_replay import PrioritizedReplayBuffer
from attentive_snake.training.dqn.replay_storage import ReplayStorage
from attentive_snake.training.dqn.uniform_replay import UniformReplayBuffer
from test_dqn_parts import CPU, transitions


def buffer(capacity: int = 8, alpha: float = 1.0, beta: float = 1.0, epsilon: float = 1e-6, seed: int = 0) -> PrioritizedReplayBuffer:
    return PrioritizedReplayBuffer(capacity, (2,), np.random.default_rng(seed), alpha, epsilon, lambda: beta)


def test_new_transitions_get_the_largest_priority_so_far() -> None:
    replay = buffer(capacity=6, alpha=1.0)
    replay.add(transitions(0, 3))
    assert replay._tree[np.arange(3)].tolist() == [1.0, 1.0, 1.0]  # nothing updated yet: 1
    replay.update_priorities(np.array([0, 1]), np.array([4.0, 0.5]))  # priorities 4 + eps, 0.5 + eps
    replay.add(transitions(3, 2))
    assert replay._tree[np.array([3, 4])].tolist() == [pytest.approx(4.000001)] * 2
    assert replay.statistics()["priority_max"] == pytest.approx(4.000001)


def test_wrapping_around_overwrites_the_priority_with_the_slot() -> None:
    replay = buffer(capacity=4, alpha=1.0)
    replay.add(transitions(0, 4))
    replay.update_priorities(np.arange(4), np.array([0.1, 0.1, 0.1, 9.0]))
    replay.add(transitions(4, 2))  # overwrites slots 0 and 1 at the maximum (9)
    assert replay._tree[np.arange(4)].tolist() == [pytest.approx(9.000001)] * 2 + [pytest.approx(0.100001), pytest.approx(9.000001)]
    sample = replay.sample(64, CPU)
    assert set(sample.rewards.tolist()) <= {4.0, 5.0, 2.0, 3.0} and len(replay) == 4


def test_sampling_frequencies_follow_the_priorities() -> None:
    replay = buffer(capacity=4, alpha=1.0, epsilon=0.0 + 1e-9, seed=7)
    replay.add(transitions(0, 4))
    replay.update_priorities(np.arange(4), np.array([1.0, 2.0, 3.0, 4.0]))
    counts = np.zeros(4)
    for _ in range(300):
        np.add.at(counts, replay.sample(40, CPU).indices, 1)
    assert counts / counts.sum() == pytest.approx([0.1, 0.2, 0.3, 0.4], abs=0.02)


def test_sampling_is_stratified_one_draw_per_segment() -> None:
    replay = buffer(capacity=4, alpha=1.0)
    replay.add(transitions(0, 4))  # equal priorities: segment k of 4 holds exactly slot k
    assert replay.sample(4, CPU).indices.tolist() == [0, 1, 2, 3]


def test_importance_weights_by_hand() -> None:
    replay = buffer(capacity=4, alpha=1.0, beta=0.5, epsilon=1e-9)
    replay.add(transitions(0, 4))
    replay.update_priorities(np.arange(4), np.array([1.0, 1.0, 2.0, 4.0]))  # total 8, P = 1/8, 1/8, 2/8, 4/8
    sample = replay.sample(4, CPU)  # stratified over segments of 2: slots 0, 1, 2 or 3 as the prefix falls
    probability = np.array([1, 1, 2, 4])[sample.indices] / 8
    raw = (4 * probability) ** -0.5  # (N * P) ** -beta
    assert sample.weights.tolist() == pytest.approx((raw / raw.max()).tolist(), rel=1e-6)
    assert float(sample.weights.max()) == pytest.approx(1.0)
    # The rarest slot (P = 1/8) weighs 1; the most likely (P = 1/2) weighs (1/4) ** 0.5 = 0.5 of it.
    lookup = dict(zip(sample.indices.tolist(), sample.weights.tolist(), strict=True))
    if 0 in lookup and 3 in lookup:
        assert lookup[3] == pytest.approx(0.5 * lookup[0], rel=1e-6)


def test_beta_one_with_alpha_zero_is_uniform_with_weights_of_one() -> None:
    replay = buffer(capacity=10, alpha=0.0, beta=1.0)
    replay.add(transitions(0, 7))
    replay.update_priorities(np.arange(7), np.array([5.0, 0.0, 1.0, 2.0, 9.0, 0.3, 0.0]))  # alpha 0: every priority is 1
    sample = replay.sample(32, CPU)
    assert sample.weights.tolist() == pytest.approx([1.0] * 32)


def test_a_zero_td_error_keeps_the_transition_samplable() -> None:
    replay = buffer(capacity=3, alpha=0.6, epsilon=1e-6, seed=1)
    replay.add(transitions(0, 3))
    replay.update_priorities(np.arange(3), np.zeros(3))
    assert replay._tree.total > 0 and (replay._tree[np.arange(3)] > 0).all()
    assert set(replay.sample(300, CPU).indices.tolist()) == {0, 1, 2}


def test_sample_carries_transitions_and_indices_and_empty_buffers_refuse() -> None:
    replay = buffer(capacity=5)
    with pytest.raises(ValueError, match="empty"):
        replay.sample(2, CPU)
    replay.add(transitions(0, 5))
    sample = replay.sample(6, CPU)
    assert torch.equal(sample.observations[:, 0], sample.rewards) and sample.weights.dtype == torch.float32
    assert (replay._storage.gather(sample.indices, np.ones(6), CPU).rewards == sample.rewards).all()


def test_statistics_report_beta_and_priorities() -> None:
    replay = buffer(capacity=4, alpha=1.0, beta=0.7)
    replay.add(transitions(0, 2))
    assert replay.statistics() == {"priority_beta": 0.7, "priority_mean": 1.0, "priority_max": 1.0}
    assert UniformReplayBuffer(4, (2,), np.random.default_rng(0)).statistics() == {}


def test_linear_beta_rises_to_one_at_the_end() -> None:
    assert linear_beta(0.4, 0, 1000) == 0.4 and linear_beta(0.4, 500, 1000) == pytest.approx(0.7)
    assert linear_beta(0.4, 1000, 1000) == 1.0 and linear_beta(0.4, 5000, 1000) == 1.0 and linear_beta(0.4, 0, 0) == 1.0


def test_storage_writes_a_ring_and_returns_the_slots() -> None:
    storage = ReplayStorage(4, (2,))
    assert storage.write(transitions(0, 3)).tolist() == [0, 1, 2]
    assert storage.write(transitions(3, 3)).tolist() == [3, 0, 1] and len(storage) == 4


def test_config_validates_the_priority_settings() -> None:
    DqnConfig(prioritized=True, priority_alpha=0.0, priority_beta_start=1.0)
    for bad in ({"priority_alpha": -0.1}, {"priority_beta_start": 1.5}, {"priority_epsilon": 0.0}):
        with pytest.raises(ValueError, match="priority_"):
            DqnConfig(**bad)
