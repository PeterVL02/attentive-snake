"""A policy playing Snake one step at a time, with optional attention; no windows, no pygame."""

from pathlib import Path

import numpy as np
import torch

from attentive_snake.config import load_experiment
from attentive_snake.factory import make_env, make_policy
from attentive_snake.game.types import GameState
from attentive_snake.models.registry import KIND_INFO
from attentive_snake.run_dir import config_path
from attentive_snake.training.checkpoint import latest_checkpoint, load_checkpoint
from attentive_snake.watch.actions import choose_action
from attentive_snake.watch.attention_panel import (
    AttentionPanel,
    attention_panel,
    check_attention_options,
    num_attention_layers,
)
from attentive_snake.watch.frame import WatchFrame


class WatchSession:
    """Runs the run directory's policy (CPU) in its environment, one step per ``step()``.

    Option changes (``set_options``) apply to the next frame; ``current()`` rebuilds the
    present frame with the current options, e.g. to redraw the heatmap of a paused game.
    """

    def __init__(
        self,
        run_dir: Path,
        checkpoint: Path | None = None,
        sample: bool = False,
        attention: bool = True,
        layer: int = -1,
        frame: int = -1,
        seed: int | None = None,
    ) -> None:
        config = load_experiment(config_path(run_dir))
        checkpoint = checkpoint if checkpoint is not None else latest_checkpoint(run_dir)
        if checkpoint is None:
            raise FileNotFoundError(f"no checkpoint found in {run_dir}")
        self._env = make_env(config, seed=seed)
        self._policy = make_policy(config, self._env)
        env_steps = load_checkpoint(checkpoint, self._policy)
        self._policy.eval()
        self.info = {
            "kind": config.model.kind,
            "num_layers": num_attention_layers(self._policy),
            "history_length": config.env.history_length,
            "attention_view": KIND_INFO[config.model.kind].attention_view,
            "checkpoint": str(checkpoint),
            "env_steps": env_steps,
            "width": config.game.width,
            "height": config.game.height,
        }
        self._sample, self._attention, self._layer, self._frame = sample, attention, layer, frame
        check_attention_options(self._policy, layer, frame)
        self._episode = 0
        self._observation = np.zeros(0)
        self._event: str | None = None
        self._done = False
        self.reset()

    @property
    def state(self) -> GameState:
        return self._env.state

    def set_options(
        self,
        *,
        sample: bool | None = None,
        attention: bool | None = None,
        layer: int | None = None,
        frame: int | None = None,
    ) -> None:
        new_layer = self._layer if layer is None else layer
        new_frame = self._frame if frame is None else frame
        check_attention_options(self._policy, new_layer, new_frame)
        self._layer, self._frame = new_layer, new_frame
        if sample is not None:
            self._sample = sample
        if attention is not None:
            self._attention = attention

    def reset(self) -> WatchFrame:
        self._observation = self._env.reset()
        self._event, self._done = None, False
        return self.current()

    def step(self) -> WatchFrame:
        if self._done:
            self.reset()
        step = self._env.step(choose_action(self._policy, self._tensor(), self._sample))
        self._observation = step.observation
        self._event = step.event.value
        self._done = step.terminated or step.truncated
        if self._done:
            self._episode += 1
        return self.current()

    def current(self) -> WatchFrame:
        """The present frame, with attention for the observation the next action is chosen from."""
        state = self.state
        panel = self._panel()
        return WatchFrame(
            width=state.width,
            height=state.height,
            snake=[[x, y] for x, y in state.snake],
            food=None if state.food is None else list(state.food),
            score=state.score,
            steps=state.steps,
            event=self._event,
            done=self._done,
            heatmap=None if panel.heatmap is None else panel.heatmap.tolist(),
            frame_weights=None if panel.frame_weights is None else panel.frame_weights.tolist(),
            caption=panel.caption,
            episode=self._episode,
        )

    def _panel(self) -> AttentionPanel:
        if not self._attention or self._done:  # nothing is chosen from a finished game's observation
            return AttentionPanel()
        return attention_panel(self._policy, self._tensor(), self._layer, self._frame)

    def _tensor(self) -> torch.Tensor:
        return torch.from_numpy(self._observation).unsqueeze(0)
