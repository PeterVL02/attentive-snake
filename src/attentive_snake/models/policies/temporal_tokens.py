"""The input tokens of ``temporal_transformer``: every position of each history frame, then one token per past action.

One function, two callers: ``TemporalTransformerPolicy._tokens`` runs it over the policy's own
``net``, and ``TemporalTokens`` (an ``nn.Module`` holding the very same modules) runs it for code
that needs the observation -> tokens step as a module of its own (``jepa_temporal``'s embedding
encoder, which the prediction objective deep-copies). Token order and what each token holds: see
``policies.temporal_transformer``.
"""

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from attentive_snake.models.frames import action_ids, base_planes, split_frames
from attentive_snake.models.input_stages.stage import run_stage
from attentive_snake.observation_spec import ObservationSpec


def temporal_tokens(
    modules: Mapping[str, nn.Module], stage_names: Sequence[str], spec: ObservationSpec, num_actions: int, obs: Tensor
) -> Tensor:
    """Observations -> tokens (B, K*P (+ K), D): ``modules[stage_names]`` (one frame's token stage) on every frame,
    ``modules["temporal"]`` (per-frame embedding), and with action channels ``modules["action_embed"]`` tokens."""
    frames = split_frames(obs, spec)  # (B, K, F, *spatial)
    batch = frames.shape[0]
    positions = run_stage(modules, stage_names, base_planes(frames, spec).flatten(0, 1))  # (B*K, P, D)
    positions = modules["temporal"](positions.unflatten(0, (batch, spec.history_length)))
    tokens = positions.flatten(1, 2)  # (B, K*P, D)
    if spec.has_actions:
        actions = modules["temporal"](modules["action_embed"](action_ids(frames, spec, num_actions)))
        tokens = torch.cat([tokens, actions], dim=1)
    return tokens


def temporal_token_names(stage_names: Sequence[str], spec: ObservationSpec) -> tuple[str, ...]:
    """The names of the modules ``temporal_tokens`` reads."""
    return (*stage_names, "temporal", *(("action_embed",) if spec.has_actions else ()))


class TemporalTokens(nn.Module):
    """``temporal_tokens`` as a module: holds (shares, does not copy) the modules it reads from ``modules``."""

    def __init__(
        self, modules: Mapping[str, nn.Module], stage_names: Sequence[str], spec: ObservationSpec, num_actions: int
    ) -> None:
        super().__init__()
        self.parts = nn.ModuleDict({name: modules[name] for name in temporal_token_names(stage_names, spec)})
        self.stage_names = tuple(stage_names)
        self.spec = spec
        self.num_actions = num_actions

    def forward(self, obs: Tensor) -> Tensor:
        return temporal_tokens(self.parts, self.stage_names, self.spec, self.num_actions, obs)
