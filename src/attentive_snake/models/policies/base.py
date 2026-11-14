"""Shared skeleton of every policy: input validation, head initialisation, forward.

``BasePolicy`` subclasses build ``self.net`` (a ``ModuleDict`` that must contain
``policy_head`` and ``value_head``) and say how an observation becomes pooled
features. ``AttentivePolicy`` adds the transformer tail (class token, blocks, norm,
class-token pool, heads) for subclasses that only have to produce input tokens.
"""

import torch
from torch import Tensor, nn

from attentive_snake.models.blocks.transformer import (
    ClassToken,
    ClassTokenPool,
    PolicyHead,
    TransformerBlock,
    ValueHead,
    run_blocks,
)
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.frames import FrameSpec
from attentive_snake.models.token_layout import TokenLayout
from attentive_snake.models.types import PolicyOutput

POLICY_INIT_GAIN = 0.01  # near-uniform initial policy
VALUE_INIT_GAIN = 1.0


def init_head(linear: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(linear.weight, gain=gain)
    nn.init.zeros_(linear.bias)


def output_heads(d_model: int, num_actions: int) -> dict[str, nn.Module]:
    return {"policy_head": PolicyHead(d_model, num_actions), "value_head": ValueHead(d_model)}


def transformer_tail(config: ModelConfig, num_actions: int) -> dict[str, nn.Module]:
    """cls -> blocks -> norm -> pool -> heads, shared by all attention policies."""
    c = config
    return {
        "cls": ClassToken(c.d_model),
        "blocks": nn.ModuleList(
            [TransformerBlock(c.d_model, c.num_heads, c.mlp_ratio, c.dropout) for _ in range(c.num_layers)]
        ),
        "norm": nn.LayerNorm(c.d_model),
        "pool": ClassTokenPool(),
        **output_heads(c.d_model, num_actions),
    }


class BasePolicy(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        observation_shape: tuple[int, int, int],
        num_actions: int,
        history_length: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.observation_shape = tuple(observation_shape)
        self.num_actions = num_actions
        self.history_length = history_length
        self.frame_spec = FrameSpec.from_observation(self.observation_shape, num_actions, history_length)
        self.net = nn.ModuleDict(self._build_net())
        init_head(self.net["policy_head"].linear, POLICY_INIT_GAIN)
        init_head(self.net["value_head"].linear, VALUE_INIT_GAIN)

    def _build_net(self) -> dict[str, nn.Module]:
        raise NotImplementedError

    def _pooled(self, obs: Tensor, need_weights: bool) -> tuple[Tensor, list[Tensor]]:
        """Pooled features (B, D) and attention maps (empty unless ``need_weights``)."""
        raise NotImplementedError

    def forward(self, obs: Tensor) -> PolicyOutput:
        self._check_observation(obs)
        pooled, _ = self._pooled(obs, need_weights=False)
        return PolicyOutput(logits=self.net["policy_head"](pooled), value=self.net["value_head"](pooled))

    def _check_observation(self, obs: Tensor) -> None:
        if obs.ndim != 4 or tuple(obs.shape[1:]) != self.observation_shape:
            raise ValueError(
                f"expected observations of shape (B, C, H, W) with (C, H, W) = "
                f"{self.observation_shape}, got {tuple(obs.shape)}"
            )


class AttentivePolicy(BasePolicy):
    """Policies whose body is tokens -> class token -> transformer blocks."""

    def _build_front(self) -> dict[str, nn.Module]:
        """Modules that turn observations into tokens (before the class token)."""
        raise NotImplementedError

    def _tokens(self, obs: Tensor) -> Tensor:
        """Observation -> input tokens (B, N, D), without the class token."""
        raise NotImplementedError

    @property
    def token_layout(self) -> TokenLayout:
        """Where frames, cells and action tokens sit on the token axis of the attention maps."""
        raise NotImplementedError

    def _build_net(self) -> dict[str, nn.Module]:
        return {**self._build_front(), **transformer_tail(self.config, self.num_actions)}

    def _pooled(self, obs: Tensor, need_weights: bool) -> tuple[Tensor, list[Tensor]]:
        net = self.net
        tokens, maps = run_blocks(net["blocks"], net["cls"](self._tokens(obs)), need_weights)
        return net["pool"](net["norm"](tokens)), maps

    @torch.no_grad()
    def attention_maps(self, obs: Tensor) -> tuple[Tensor, ...]:
        """Per-block attention, each (B, heads, T, T), T = ``token_layout.num_tokens``.

        Runs in eval mode (no dropout); the previous train/eval mode is restored.
        """
        self._check_observation(obs)
        was_training = self.training
        self.eval()
        try:
            _, maps = self._pooled(obs, need_weights=True)
        finally:
            self.train(was_training)
        return tuple(maps)
