"""Epsilon-greedy play in a ``VecEnv``, one vector step at a time, producing replay transitions."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from attentive_snake.env.types import EpisodeSummary, VecEnv, VecEnvStep
from attentive_snake.models.acting import greedy_actions
from attentive_snake.training.dqn.replay import Transitions
from attentive_snake.training.next_observations import next_observations


@dataclass(frozen=True)
class CollectedStep:
    transitions: Transitions
    finished_episodes: tuple[EpisodeSummary, ...]


def transition_dones(step: VecEnvStep, bootstrap_truncated: bool) -> np.ndarray:
    """1.0 where nothing is bootstrapped: a rules ending, or a cut by a limit unless ``bootstrap_truncated``."""
    cut = np.logical_and(step.truncated, np.logical_not(step.terminated))
    ended = np.logical_or(step.terminated, np.logical_and(cut, not bootstrap_truncated))
    return ended.astype(np.float32)


class EpsilonGreedyCollector:
    """Steps a ``VecEnv``: each env takes a random action with probability epsilon, else the greedy one.

    Owns the current observation of every env between calls, so consecutive steps continue
    the same episodes. The policy acts in ``eval()`` mode without gradients; its previous
    mode is restored. Random choices come from ``rng`` only.
    """

    def __init__(
        self,
        env: VecEnv,
        initial_observations: np.ndarray,
        rng: np.random.Generator,
        device: torch.device,
        bootstrap_truncated: bool,
    ) -> None:
        self._env = env
        self._observations = np.asarray(initial_observations, dtype=np.float32)
        self._rng = rng
        self._device = device
        self._bootstrap_truncated = bootstrap_truncated

    @torch.no_grad()
    def step(self, policy: nn.Module, epsilon: float) -> CollectedStep:
        actions = self._actions(policy, epsilon)
        step = self._env.step(actions)
        transitions = Transitions(
            observations=self._observations,
            actions=actions,
            rewards=np.asarray(step.rewards, dtype=np.float32),
            next_observations=next_observations(step),
            dones=transition_dones(step, self._bootstrap_truncated),
        )
        self._observations = np.asarray(step.observations, dtype=np.float32)
        return CollectedStep(transitions, tuple(step.finished_episodes))

    def _actions(self, policy: nn.Module, epsilon: float) -> np.ndarray:
        was_training = policy.training
        policy.eval()
        try:
            greedy = greedy_actions(policy(torch.as_tensor(self._observations, device=self._device))).cpu().numpy()
        finally:
            policy.train(was_training)
        explore = self._rng.random(self._env.num_envs) < epsilon
        random = self._rng.integers(0, self._env.num_actions, size=self._env.num_envs)
        return np.where(explore, random, greedy).astype(np.int64)
