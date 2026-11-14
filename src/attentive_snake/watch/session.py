"""A policy playing its game one step at a time, with optional attention; no windows, no pygame."""

from pathlib import Path

import torch

from attentive_snake.config import load_experiment
from attentive_snake.factory import make_env, make_policy, observation_spec
from attentive_snake.games.registry import get_game
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
from attentive_snake.watch.element_names import element_names
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
        game = get_game(config.game_name)
        self._view_kind = game.view_kind
        self._policy = make_policy(config, self._env)
        env_steps = load_checkpoint(checkpoint, self._policy)
        self._policy.eval()
        self._sample, self._attention, self._layer, self._frame = sample, attention, layer, frame
        check_attention_options(self._policy, layer, frame)
        self._episode = 0
        first = self.reset()  # GameEnv contract: view() only after a reset
        self.info = {
            "game": config.game_name,
            "view_kind": self._view_kind,
            "observation_shape": list(self._env.observation_shape),
            "kind": config.model.kind,
            "num_layers": num_attention_layers(self._policy),
            "history_length": config.env.history_length,
            "attention_view": KIND_INFO[config.model.kind].attention_view,
            "element_names": element_names(game, observation_spec(config, self._env)),  # vector games only
            "checkpoint": str(checkpoint),
            "env_steps": env_steps,
            "width": first.view["width"],  # of the view: cells for "board", pixels for "image"
            "height": first.view["height"],
        }

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
        self._score, self._steps = 0.0, 0
        return self.current()

    def step(self) -> WatchFrame:
        if self._done:
            self.reset()
        step = self._env.step(choose_action(self._policy, self._tensor(), self._sample))
        self._observation = step.observation
        self._event = step.event
        self._score = step.score
        self._steps += 1
        self._done = step.terminated or step.truncated
        if self._done:
            self._episode += 1
        return self.current()

    def current(self) -> WatchFrame:
        """The present frame, with attention for the observation the next action is chosen from."""
        panel = self._panel()
        return WatchFrame(
            view_kind=self._view_kind,
            view=self._env.view(),
            score=self._score,
            steps=self._steps,
            event=self._event,
            done=self._done,
            heatmap=None if panel.heatmap is None else panel.heatmap.tolist(),
            frame_weights=None if panel.frame_weights is None else panel.frame_weights.tolist(),
            caption=panel.caption,
            episode=self._episode,
            element_weights=None if panel.element_weights is None else panel.element_weights.tolist(),
        )

    def _panel(self) -> AttentionPanel:
        if not self._attention or self._done:  # nothing is chosen from a finished game's observation
            return AttentionPanel()
        return attention_panel(self._policy, self._tensor(), self._layer, self._frame)

    def _tensor(self) -> torch.Tensor:
        return torch.from_numpy(self._observation).unsqueeze(0)
