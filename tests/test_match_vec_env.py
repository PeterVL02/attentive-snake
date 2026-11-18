"""MatchVecEnv: self-play matches as slots of one VecEnv, on a tiny numpy match (no emulator)."""

import numpy as np
import pytest
import torch
from torch import nn

from attentive_snake.env.history_env import HistoryEnv
from attentive_snake.env.match_vec_env import MatchVecEnv
from attentive_snake.env.types import EnvStep
from attentive_snake.env.vec_env import env_seed
from attentive_snake.models.types import PolicyOutput
from attentive_snake.training.ppo.collector import RolloutCollector
from attentive_snake.training.ppo.rollout import RolloutBuffer, compute_gae


class Race:
    """Two seats race to ``goal``: action 1 moves a seat one step, 0 waits. Frame: (2,) = (seat, own position).

    Ends (terminated) when a seat reaches the goal, or (truncated) after ``max_steps``. Reward +1 to
    the seat that arrives first, -1 to the other. ``seeds`` records every reset's seed.
    """

    num_seats = 2
    num_actions = 2
    observation_shape = (2,)

    def __init__(self, goal: int = 3, max_steps: int = 10, uneven: bool = False) -> None:
        self.goal, self.max_steps, self.uneven = goal, max_steps, uneven
        self.seeds: list[int | None] = []

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.positions, self.steps = [0, 0], 0
        return self._frames()

    def step(self, actions):
        self.steps += 1
        for seat, action in enumerate(actions):
            self.positions[seat] += int(action)
        won = [position >= self.goal for position in self.positions]
        over = any(won)
        cut = not over and self.steps >= self.max_steps
        reward = [0.0, 0.0] if won[0] == won[1] else [1.0 if won[0] else -1.0, 1.0 if won[1] else -1.0]
        frames = self._frames()
        return tuple(
            EnvStep(frames[seat], reward[seat], over and not (self.uneven and seat == 1), cut, "end" if over or cut else "run", float(self.positions[seat]))
            for seat in range(2)
        )

    def view(self):
        return {"width": 1, "height": 1}

    def _frames(self):
        return np.array([[0.0, self.positions[0]], [1.0, self.positions[1]]], dtype=np.float32)


def test_slots_are_seats_of_matches_and_matches_get_derived_seeds():
    matches = [Race(), Race()]
    vec = MatchVecEnv(matches)
    assert (vec.num_envs, vec.num_seats, vec.observation_shape, vec.num_actions) == (4, 2, (2,), 2)
    observations = vec.reset(seed=7)
    assert observations.tolist() == [[0, 0], [1, 0], [0, 0], [1, 0]]  # slot m * 2 + s: seat s of match m
    assert [match.seeds for match in matches] == [[env_seed(7, 0)], [env_seed(7, 1)]]
    step = vec.step(np.array([1, 0, 0, 1]))
    assert step.observations.tolist() == [[0, 1], [1, 0], [0, 0], [1, 1]]  # each slot's action went to its own seat


def test_a_finished_match_ends_and_restarts_all_its_slots_together():
    vec = MatchVecEnv([Race(goal=2), Race(goal=5)])
    vec.reset(seed=0)
    vec.step(np.array([1, 0, 1, 1]))
    step = vec.step(np.array([1, 0, 1, 1]))  # match 0: seat 0 arrives
    assert step.terminated.tolist() == [True, True, False, False]
    assert step.rewards.tolist() == [1.0, -1.0, 0.0, 0.0]
    assert sorted(step.final_observations) == [0, 1]
    assert step.final_observations[0].tolist() == [0, 2] and step.final_observations[1].tolist() == [1, 0]
    assert step.observations[:2].tolist() == [[0, 0], [1, 0]]  # both seats start the next match
    summaries = {summary.env_index: summary for summary in step.finished_episodes}
    assert sorted(summaries) == [0, 1] and summaries[0].total_reward == 1.0 and summaries[1].total_reward == -1.0
    assert summaries[0].length == summaries[1].length == 2


def test_truncation_reaches_every_seat():
    vec = MatchVecEnv([Race(goal=9, max_steps=2)])
    vec.reset(seed=0)
    vec.step(np.array([0, 0]))
    step = vec.step(np.array([0, 0]))
    assert step.truncated.tolist() == [True, True] and not step.terminated.any()
    assert sorted(step.final_observations) == [0, 1]


def test_seats_ending_differently_is_refused():
    vec = MatchVecEnv([Race(goal=1, uneven=True)])
    vec.reset(seed=0)
    with pytest.raises(ValueError, match="same step"):
        vec.step(np.array([1, 0]))


def test_each_seat_has_its_own_history_and_action_channels():
    vec = MatchVecEnv([Race()], history_length=2, history_actions=True)
    assert vec.observation_shape == (8,) and vec.frame_channels == 4 and vec.action_channels == 2
    vec.reset(seed=0)
    observation = vec.step(np.array([1, 0])).observations
    assert observation[0].tolist() == [0, 0, 0, 0, 0, 1, 0, 1]  # seat 0: empty first frame, then moved with action 1
    assert observation[1].tolist() == [1, 0, 0, 0, 1, 0, 1, 0]  # seat 1: waited (action 0)


def test_the_same_history_as_a_single_env():
    """A seat's observations are built exactly as HistoryEnv builds a single game's."""

    class OneSeat:
        observation_shape, num_actions = (2,), 2

        def __init__(self):
            self.race = Race()

        def reset(self, seed=None):
            return self.race.reset(seed)[0]

        def step(self, action):
            return self.race.step([action, 0])[0]

        def view(self):
            return {}

    single = HistoryEnv(OneSeat(), history_length=3, history_actions=True)
    vec = MatchVecEnv([Race()], history_length=3, history_actions=True)
    assert np.array_equal(single.reset(0), vec.reset(0)[0])
    for action in (1, 0, 1):
        assert np.array_equal(single.step(action).observation, vec.step(np.array([action, 0])).observations[0])


class SeatZeroRaces(nn.Module):
    """Seat 0 always moves, seat 1 always waits (the observation's first number is the seat)."""

    def forward(self, observations):
        moves = 100.0 * (1.0 - 2.0 * observations[:, 0])
        return PolicyOutput(logits=torch.stack([-moves, moves], dim=1), value=torch.zeros(len(observations)))


def test_a_rollout_keeps_each_seat_in_its_own_column_and_gae_does_not_mix_them():
    vec = MatchVecEnv([Race(goal=2), Race(goal=3)])
    buffer = RolloutBuffer.allocate(6, vec.num_envs, vec.observation_shape, torch.device("cpu"))
    RolloutCollector(vec, buffer, vec.reset(seed=0)).collect(SeatZeroRaces())
    assert buffer.actions.T.tolist() == [[1] * 6, [0] * 6, [1] * 6, [0] * 6]
    assert buffer.rewards.T.tolist() == [[0, 1, 0, 1, 0, 1], [0, -1, 0, -1, 0, -1], [0, 0, 1, 0, 0, 1], [0, 0, -1, 0, 0, -1]]
    assert buffer.dones.T.tolist() == [[0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 0, 0, 1], [0, 0, 1, 0, 0, 1]]
    advantages, _ = compute_gae(buffer.rewards, buffer.values, buffer.dones, torch.zeros(4), gamma=0.5, gae_lambda=1.0)
    assert advantages.T[0].tolist() == [0.5, 1.0, 0.5, 1.0, 0.5, 1.0]  # each episode's own reward, discounted
    assert torch.equal(advantages.T[1], -advantages.T[0]) and torch.equal(advantages.T[3], -advantages.T[2])


def test_matches_must_agree():
    class Wide(Race):
        observation_shape = (3,)

    with pytest.raises(ValueError, match="same seats"):
        MatchVecEnv([Race(), Wide()])
    with pytest.raises(ValueError, match="at least one"):
        MatchVecEnv([])
