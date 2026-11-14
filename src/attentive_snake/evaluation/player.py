"""Playing N complete games with a policy's greedy action at every step (most likely action, or highest Q-value), batched over parallel environments.

Game ``k`` (0-based) starts with ``reset(seed=env_seed(settings.seed, k))`` on whichever of
the ``parallel_games`` environments is free; environments are taken in a fixed order and
play is deterministic, so the same policy and settings always play the same games in the
same batches. Nothing here samples: no random stream is used.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from attentive_snake.env.types import SingleEnv
from attentive_snake.env.vec_env import env_seed
from attentive_snake.episode_metrics import STEP_LIMIT, EpisodeRecord
from attentive_snake.evaluation.settings import EvalSettings
from attentive_snake.models.acting import greedy_actions

type EnvFactory = Callable[[], SingleEnv]
type Progress = Callable[[int, int], None]  # (games finished, games in total)


@dataclass
class _Game:
    """A game in progress on one environment."""

    index: int
    observation: np.ndarray
    events: list[str] = field(default_factory=list)
    total_reward: float = 0.0
    score: float = 0.0

    def record(self, outcome: str) -> EpisodeRecord:
        return EpisodeRecord(self.score, len(self.events), self.total_reward, outcome, tuple(self.events))


@torch.no_grad()
def play_games(policy: nn.Module, make_env: EnvFactory, settings: EvalSettings, progress: Progress | None = None) -> list[EpisodeRecord]:
    """``settings.games`` records, in game order. The policy runs in eval mode on ``settings.device``
    (it must already be there); its train/eval mode is restored afterwards.

    A game that has not ended after ``settings.max_episode_steps`` steps is stopped with the
    outcome ``STEP_LIMIT``.
    """
    envs = [make_env() for _ in range(min(settings.parallel_games, settings.games))]
    was_training = policy.training
    policy.eval()
    try:
        return _play(policy, envs, settings, progress or (lambda done, total: None))
    finally:
        policy.train(was_training)


def _play(policy: nn.Module, envs: list[SingleEnv], settings: EvalSettings, progress: Progress) -> list[EpisodeRecord]:
    device = torch.device(settings.device)
    records: list[EpisodeRecord | None] = [None] * settings.games
    upcoming = iter(range(settings.games))

    def start(env: SingleEnv) -> _Game | None:
        index = next(upcoming, None)
        return None if index is None else _Game(index, env.reset(seed=env_seed(settings.seed, index)))

    slots = [start(env) for env in envs]
    finished = 0
    while any(slot is not None for slot in slots):
        active = [i for i, slot in enumerate(slots) if slot is not None]
        observations = torch.as_tensor(np.stack([slots[i].observation for i in active]), dtype=torch.float32, device=device)
        actions = greedy_actions(policy(observations)).tolist()
        for i, action in zip(active, actions):
            game, step = slots[i], envs[i].step(int(action))
            game.observation, game.score = step.observation, float(step.score)
            game.total_reward += float(step.reward)
            game.events.append(step.event)
            ended = step.terminated or step.truncated
            if ended or len(game.events) >= settings.max_episode_steps:
                records[game.index] = game.record(step.event if ended else STEP_LIMIT)
                finished += 1
                progress(finished, settings.games)
                slots[i] = start(envs[i])
    return records
