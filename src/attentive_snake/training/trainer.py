"""Orchestration of a PPO run: collect -> GAE -> update -> log -> checkpoint -> hooks."""

import dataclasses
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from attentive_snake.env.types import VecEnv
from attentive_snake.training.checkpoint import checkpoint_path, prune_checkpoints, save_checkpoint
from attentive_snake.training.collector import RolloutCollector
from attentive_snake.training.config import TrainConfig
from attentive_snake.training.hooks import UpdateEnd, UpdateHook
from attentive_snake.training.metrics import ConsoleLogger, EpisodeStatsWindow, MetricsLogger
from attentive_snake.training.ppo import ppo_update
from attentive_snake.training.rollout import RolloutBuffer, compute_gae, flatten_rollout
from attentive_snake.training.runtime import resolve_device, seed_everything
from attentive_snake.training.schedule import linear_decay


class Trainer:
    """Trains any policy ``nn.Module`` (obs -> ``PolicyOutput``) on any ``VecEnv`` with PPO.

    Seeding happens in the constructor, before the optimizer and buffers are
    created; the policy's initial weights are whatever the caller built (seed
    before constructing the model for fully reproducible runs). The policy is
    moved to the configured device in place.

    Resuming: ``train`` starts at update ``env_steps // steps_per_update``, so
    a trainer whose policy, optimizer and ``env_steps`` were restored (see
    ``attentive_snake.training.resume.resume_from_checkpoint``) continues the
    run: update numbers, logged env steps, checkpoint names and the learning
    rate all carry on from there. The learning-rate schedule is a pure
    function of the update index and ``config.total_env_steps``; if the total
    is raised on resume, the schedule is stretched over the new total (the LR
    is what it would have been had the run been planned with the new total
    from the start), so it may jump up at the resume point. Not restored: the
    environments (they start fresh episodes), RNG streams (re-seeded from
    ``config.seed``) and the episode-statistics window, so a resumed run is
    not bit-identical to an uninterrupted one.

    Hooks (e.g. periodic evaluation, ``attentive_snake.evaluation.hook``) are called after
    every update, in order, with an ``UpdateEnd``; they decide themselves whether to act.
    """

    def __init__(
        self,
        policy: nn.Module,
        env: VecEnv,
        config: TrainConfig,
        logger: MetricsLogger | None = None,
        hooks: Sequence[UpdateHook] = (),
    ) -> None:
        seed_everything(config.seed)
        self.config = config
        self.device = resolve_device(config.device)
        self.policy = policy.to(self.device)
        self.env = env
        self.logger = logger if logger is not None else ConsoleLogger()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate, eps=1e-5)
        self.env_steps = 0
        self._hooks = tuple(hooks)
        self._saved_at: int | None = None  # env steps of the last checkpoint this trainer saved
        self._shuffle_generator = torch.Generator().manual_seed(config.seed)
        self._buffer = RolloutBuffer.allocate(
            config.rollout_length, env.num_envs, tuple(env.observation_shape), self.device
        )
        self._collector = RolloutCollector(env, self._buffer, env.reset(seed=config.seed))
        self._episode_stats = EpisodeStatsWindow()

    @property
    def steps_per_update(self) -> int:
        return self.config.rollout_length * self.env.num_envs

    @property
    def num_updates(self) -> int:
        return max(1, self.config.total_env_steps // self.steps_per_update)

    @property
    def completed_updates(self) -> int:
        return self.env_steps // self.steps_per_update

    def train(self) -> list[dict[str, float]]:
        """Run the remaining updates; returns one metrics record per update run (empty if already done)."""
        history = []
        for update in range(self.completed_updates, self.num_updates):
            previous_env_steps = self.env_steps
            record = self._train_one_update(update)
            history.append(record)
            is_last = update == self.num_updates - 1
            if (update + 1) % self.config.log_interval == 0 or is_last:
                self.logger.log(self.env_steps, record)
            if self.config.checkpoint_dir is not None and ((update + 1) % self.config.checkpoint_interval == 0 or is_last):
                self.save_checkpoint()
            self._call_hooks(previous_env_steps, is_last)
        return history

    def save_checkpoint(self) -> Path:
        assert self.config.checkpoint_dir is not None
        path = checkpoint_path(self.config.checkpoint_dir, self.env_steps)
        save_checkpoint(path, self.policy, self.optimizer, self.env_steps)
        self._saved_at = self.env_steps
        if not self.config.keep_checkpoints:
            prune_checkpoints(self.config.checkpoint_dir, keep=path)
        return path

    def checkpoint_now(self) -> Path | None:
        """The checkpoint of the current weights, saved unless it was just saved; ``None`` without a checkpoint directory."""
        if self.config.checkpoint_dir is None:
            return None
        if self._saved_at == self.env_steps:
            return checkpoint_path(self.config.checkpoint_dir, self.env_steps)
        return self.save_checkpoint()

    def _call_hooks(self, previous_env_steps: int, is_last: bool) -> None:
        end = UpdateEnd(previous_env_steps, self.env_steps, is_last, self.policy, self.device, self.checkpoint_now)
        for hook in self._hooks:
            hook(end)

    def _train_one_update(self, update: int) -> dict[str, float]:
        start = time.perf_counter()
        learning_rate = self._set_learning_rate(update)

        collected = self._collector.collect(self.policy)
        self.env_steps += self.steps_per_update
        self._episode_stats.add(collected.finished_episodes)

        advantages, returns = compute_gae(
            self._buffer.rewards,
            self._buffer.values,
            self._buffer.dones,
            collected.bootstrap_value,
            self.config.gamma,
            self.config.gae_lambda,
        )
        batch = flatten_rollout(self._buffer, advantages, returns)
        stats = ppo_update(self.policy, self.optimizer, batch, self.config, self._shuffle_generator)

        elapsed = time.perf_counter() - start
        return {
            "update": float(update + 1),
            "env_steps": float(self.env_steps),
            **self._episode_stats.summary(),
            **dataclasses.asdict(stats),
            "learning_rate": learning_rate,
            "steps_per_second": self.steps_per_update / elapsed,
        }

    def _set_learning_rate(self, update: int) -> float:
        if self.config.anneal_lr:
            learning_rate = linear_decay(self.config.learning_rate, update, self.num_updates)
        else:
            learning_rate = self.config.learning_rate
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate
