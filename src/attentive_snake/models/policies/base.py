"""Shared skeleton of every policy: input validation, head initialisation, forward.

Every policy is built as ``Policy(config, observation_spec, num_actions)``: the input
layout comes from the environment (via the factory) as an ``ObservationSpec``, never from
``ModelConfig``. A policy class lists the observation kinds it can read in
``observation_kinds``; building it for another kind is a ``ValueError``.

``BasePolicy`` subclasses build ``self.net`` (a ``ModuleDict`` that must contain
``policy_head`` and ``value_head``) and say how an observation becomes pooled
features. ``AttentivePolicy`` adds the transformer tail (class token, blocks, norm,
class-token pool, heads) for subclasses that only have to produce input tokens.
"""

from typing import ClassVar

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
from attentive_snake.models.token_layout import TokenLayout
from attentive_snake.models.types import PolicyOutput
from attentive_snake.observation_spec import ObservationSpec

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


def require_action_layout(spec: ObservationSpec, num_actions: int) -> None:
    """For policies that read actions from the frames: the action channels are absent or one per action."""
    if spec.action_channels not in (0, num_actions):
        raise ValueError(
            f"frames have {spec.action_channels} action channels; expected 0 or one per action ({num_actions})."
        )


class BasePolicy(nn.Module):
    observation_kinds: ClassVar[tuple[str, ...]] = ("grid",)

    def __init__(self, config: ModelConfig, observation_spec: ObservationSpec, num_actions: int) -> None:
        super().__init__()
        if observation_spec.kind not in self.observation_kinds:
            raise ValueError(
                f"model kind {config.kind!r} cannot read {observation_spec.kind} observations "
                f"(it reads: {', '.join(self.observation_kinds)})."
            )
        self.config = config
        self.observation_spec = observation_spec
        self.observation_shape = observation_spec.shape
        self.num_actions = num_actions
        self.history_length = observation_spec.history_length
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
        if tuple(obs.shape[1:]) != self.observation_shape or obs.ndim != 1 + len(self.observation_shape):
            raise ValueError(
                f"expected observations of shape (B, *{self.observation_shape}), got {tuple(obs.shape)}"
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
