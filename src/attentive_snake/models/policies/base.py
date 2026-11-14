"""Shared skeleton of every policy: input validation, the head, forward.

Every policy is built as ``Policy(config, observation_spec, num_actions, head)``: the input
layout comes from the environment (via the factory) as an ``ObservationSpec``, never from
``ModelConfig``; ``head`` names a ``models.heads.HEADS`` entry (the training algorithm's
choice: actor-critic for PPO, Q-values for DQN). A policy class lists the observation kinds it can read in
``observation_kinds``; building it for another kind is a ``ValueError``.

The part that reads the observation (the input stage) comes from
``models.input_stages.catalog`` by observation kind (``_input_stage``), so a policy says
*what* it needs (tokens, a feature vector, one vector per frame) and never branches on
the kind itself; its ``observation_kinds`` are the kinds its stage tables cover.

``BasePolicy`` subclasses build the trunk of ``self.net`` (a ``ModuleDict``; the head's
modules are appended after it) and say how an observation becomes pooled ``d_model``
features. ``AttentivePolicy`` adds the transformer tail (class token, blocks, norm,
class-token pool) for subclasses that only have to produce input tokens.
"""

from collections.abc import Mapping
from typing import ClassVar

import torch
from torch import Tensor, nn

from attentive_snake.models.blocks.transformer import ClassToken, ClassTokenPool, TransformerBlock, run_blocks
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.heads import DEFAULT_HEAD, HEADS, Head, get_head
from attentive_snake.models.input_stages.catalog import StageBuilder
from attentive_snake.models.input_stages.stage import Stage, StageInput
from attentive_snake.models.token_layout import TokenLayout
from attentive_snake.models.types import PolicyOutput, QValues
from attentive_snake.observation_spec import ObservationSpec


def transformer_tail(config: ModelConfig) -> dict[str, nn.Module]:
    """cls -> blocks -> norm -> pool, shared by all attention policies (the head follows)."""
    c = config
    return {
        "cls": ClassToken(c.d_model),
        "blocks": nn.ModuleList(
            [TransformerBlock(c.d_model, c.num_heads, c.mlp_ratio, c.dropout) for _ in range(c.num_layers)]
        ),
        "norm": nn.LayerNorm(c.d_model),
        "pool": ClassTokenPool(),
    }


def require_action_layout(spec: ObservationSpec, num_actions: int) -> None:
    """For policies that read actions from the frames: the action channels are absent or one per action."""
    if spec.action_channels not in (0, num_actions):
        raise ValueError(
            f"frames have {spec.action_channels} action channels; expected 0 or one per action ({num_actions})."
        )


class BasePolicy(nn.Module):
    observation_kinds: ClassVar[tuple[str, ...]] = ()
    heads: ClassVar[Mapping[str, Head]] = HEADS  # the head forms by name; a kind may bring its own (same names)

    def __init__(
        self, config: ModelConfig, observation_spec: ObservationSpec, num_actions: int, head: str = DEFAULT_HEAD
    ) -> None:
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
        self.head_name = head
        self._head = get_head(head, self.heads)
        trunk = self._build_net()
        self.net = nn.ModuleDict({**trunk, **self._head.modules(config.d_model, num_actions)})

    def _build_net(self) -> dict[str, nn.Module]:
        """The trunk: observation -> pooled ``d_model`` features (the head is added after it)."""
        raise NotImplementedError

    def _input_stage(self, table: Mapping[str, StageBuilder], source: StageInput) -> Stage:
        """The stage of ``table`` for this policy's observation kind, reading ``source``."""
        return table[self.observation_spec.kind](source, self.config)

    def _pooled(self, obs: Tensor, need_weights: bool) -> tuple[Tensor, list[Tensor]]:
        """Pooled features (B, D) and attention maps (empty unless ``need_weights``)."""
        raise NotImplementedError

    def forward(self, obs: Tensor) -> PolicyOutput | QValues:
        self._check_observation(obs)
        pooled, _ = self._pooled(obs, need_weights=False)
        return self._head.output(self.net, pooled)

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
        return {**self._build_front(), **transformer_tail(self.config)}

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
