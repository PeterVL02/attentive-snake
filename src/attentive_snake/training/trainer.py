"""The training loop every algorithm shares: learning-rate schedule -> learner iteration -> log -> checkpoint -> hooks."""

import time
from collections.abc import Sequence
from pathlib import Path

import torch

from attentive_snake.training.algorithm import Learner
from attentive_snake.training.checkpoint import checkpoint_path, prune_checkpoints, read_checkpoint, save_checkpoint
from attentive_snake.training.config import TrainConfig
from attentive_snake.training.hooks import UpdateEnd, UpdateHook
from attentive_snake.training.metrics import ConsoleLogger, EpisodeStatsWindow, MetricsLogger
from attentive_snake.training.schedule import crossed_multiple, linear_decay


class Trainer:
    """Drives a ``Learner`` (any algorithm) until ``config.total_env_steps``; build it with ``training.build.build_trainer``.

    One iteration consumes ``learner.steps_per_iteration`` env steps; the run has
    ``max(1, total_env_steps // steps_per_iteration)`` iterations. Before each, the learning
    rate is set from the iteration index (linear decay when ``anneal_lr``). Metrics are
    logged, and a checkpoint saved, each time the env-step count passes a multiple of
    ``log_interval`` / ``checkpoint_interval`` and after the last iteration. Hooks (e.g.
    periodic evaluation, ``attentive_snake.evaluation.hook``) are called after every
    iteration, in order, with an ``UpdateEnd``; they decide themselves whether to act.

    Resuming (``resume``): iterations, logged env steps, checkpoint names and the learning
    rate carry on from the checkpoint's step count. The learning-rate schedule is a pure
    function of the iteration and ``total_env_steps``; a raised total stretches it (the
    LR may jump up at the resume point). Not restored: the environments (they start fresh
    episodes), RNG streams (re-seeded from ``config.seed``), the episode-statistics window
    and whatever the learner keeps outside ``extra_state`` (e.g. DQN's replay buffer), so a
    resumed run is not bit-identical to an uninterrupted one.
    """

    def __init__(
        self,
        learner: Learner,
        config: TrainConfig,
        device: torch.device,
        logger: MetricsLogger | None = None,
        hooks: Sequence[UpdateHook] = (),
    ) -> None:
        self.learner = learner
        self.config = config
        self.device = device
        self.logger = logger if logger is not None else ConsoleLogger()
        self.env_steps = 0
        self._hooks = tuple(hooks)
        self._saved_at: int | None = None  # env steps of the last checkpoint this trainer saved
        self._episode_stats = EpisodeStatsWindow()

    @property
    def policy(self) -> torch.nn.Module:
        return self.learner.policy

    @property
    def steps_per_iteration(self) -> int:
        return self.learner.steps_per_iteration

    @property
    def num_iterations(self) -> int:
        return max(1, self.config.total_env_steps // self.steps_per_iteration)

    @property
    def completed_iterations(self) -> int:
        return self.env_steps // self.steps_per_iteration

    def train(self) -> list[dict[str, float]]:
        """Run the remaining iterations; returns the logged metrics records (empty if already done)."""
        logged = []
        last_log = (self.env_steps, time.perf_counter())
        for iteration in range(self.completed_iterations, self.num_iterations):
            previous_env_steps = self.env_steps
            learning_rate = self._set_learning_rate(iteration)
            self._episode_stats.add(self.learner.iterate(self.env_steps))
            self.env_steps += self.steps_per_iteration
            is_last = iteration == self.num_iterations - 1
            if is_last or crossed_multiple(previous_env_steps, self.env_steps, self.config.log_interval):
                record = self._record(iteration, learning_rate, last_log)
                self.logger.log(self.env_steps, record)
                logged.append(record)
                last_log = (self.env_steps, time.perf_counter())
            if self.config.checkpoint_dir is not None and (
                is_last or crossed_multiple(previous_env_steps, self.env_steps, self.config.checkpoint_interval)
            ):
                self.save_checkpoint()
            self._call_hooks(previous_env_steps, is_last)
        return logged

    def save_checkpoint(self) -> Path:
        assert self.config.checkpoint_dir is not None
        path = checkpoint_path(self.config.checkpoint_dir, self.env_steps)
        save_checkpoint(path, self.policy, self.learner.optimizer, self.env_steps, self.learner.extra_state())
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

    def restore(self, path: Path, keep_step_count: bool) -> int:
        """Load the policy, optimizer and learner state of a checkpoint; returns its env steps.

        ``keep_step_count`` continues the checkpoint's run (resume); otherwise the step
        counter stays where it is (0 for a fork: a new run with its own schedule).
        """
        state = read_checkpoint(path, map_location=self.device)
        self.policy.load_state_dict(state["policy"])
        self.learner.optimizer.load_state_dict(state["optimizer"])
        self.learner.load_extra_state(state.get("learner", {}))
        steps = int(state["env_steps"])
        if keep_step_count:
            self.env_steps = steps
        return steps

    def _record(self, iteration: int, learning_rate: float, last_log: tuple[int, float]) -> dict[str, float]:
        steps_since, seconds = self.env_steps - last_log[0], time.perf_counter() - last_log[1]
        return {
            "iteration": float(iteration + 1),
            "env_steps": float(self.env_steps),
            **self._episode_stats.summary(),
            **self.learner.report(),
            "learning_rate": learning_rate,
            "steps_per_second": steps_since / seconds if seconds > 0 else 0.0,
        }

    def _call_hooks(self, previous_env_steps: int, is_last: bool) -> None:
        end = UpdateEnd(previous_env_steps, self.env_steps, is_last, self.policy, self.device, self.checkpoint_now)
        for hook in self._hooks:
            hook(end)

    def _set_learning_rate(self, iteration: int) -> float:
        if self.config.anneal_lr:
            learning_rate = linear_decay(self.config.learning_rate, iteration, self.num_iterations)
        else:
            learning_rate = self.config.learning_rate
        for group in self.learner.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate
