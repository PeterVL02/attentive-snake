"""Attention across positions and time (``kind = "temporal_transformer"``).

Children of ``self.net``, in order:
embed -> positional -> temporal -> [action_embed] -> cls -> blocks -> norm -> pool -> {policy_head, value_head}

Token order (see ``token_layout``), T = 1 + K*P (+ K), P = positions per frame:

    [ class | frame 0 positions | ... | frame K-1 positions | action 0 .. action K-1 ]

Positions are the cells of a grid (H*W, row-major) or the game's elements of a vector
(``base_channels`` numbers, in order). Frame 0 is the oldest, frame K-1 the newest. Each
position token embeds the frame's game channels there (``TOKEN_STAGES`` of one frame:
``embed`` + ``positional``), plus a learned per-frame (age) embedding.
When frames carry action channels there is one action token per frame: the embedding of
that frame's action (a learned "no action" entry for all-zero channels) plus the same
per-frame embedding. Without action channels there are no action tokens (and no action
parameters). Frames before the episode began are all-zero, not masked.
"""

import torch
from torch import Tensor, nn

from attentive_snake.models.blocks.temporal import ActionEmbedding, TemporalEmbedding
from attentive_snake.models.frames import action_ids, base_planes, split_frames
from attentive_snake.models.input_stages.catalog import TOKEN_STAGES, readable_kinds
from attentive_snake.models.input_stages.stage import one_frame, run_stage
from attentive_snake.models.policies.base import AttentivePolicy, require_action_layout
from attentive_snake.models.token_layout import TokenLayout


class TemporalTransformerPolicy(AttentivePolicy):
    observation_kinds = readable_kinds(TOKEN_STAGES)

    def _build_front(self) -> dict[str, nn.Module]:
        require_action_layout(self.observation_spec, self.num_actions)
        d = self.config.d_model
        stage = self._input_stage(TOKEN_STAGES, one_frame(self.observation_spec))
        self._stage_names, self._positions = tuple(stage.modules), stage.num_tokens
        front: dict[str, nn.Module] = {**stage.modules, "temporal": TemporalEmbedding(self.history_length, d)}
        if self.observation_spec.has_actions:
            front["action_embed"] = ActionEmbedding(self.num_actions, d)
        return front

    def _tokens(self, obs: Tensor) -> Tensor:
        net = self.net
        frames = split_frames(obs, self.observation_spec)  # (B, K, F, *spatial)
        batch = frames.shape[0]
        positions = run_stage(net, self._stage_names, base_planes(frames, self.observation_spec).flatten(0, 1))  # (B*K, P, D)
        positions = net["temporal"](positions.unflatten(0, (batch, self.history_length)))
        tokens = positions.flatten(1, 2)  # (B, K*P, D)
        if self.observation_spec.has_actions:
            actions = net["temporal"](net["action_embed"](action_ids(frames, self.observation_spec, self.num_actions)))
            tokens = torch.cat([tokens, actions], dim=1)
        return tokens

    @property
    def token_layout(self) -> TokenLayout:
        return TokenLayout(
            num_frames=self.history_length,
            tokens_per_frame=self._positions,
            num_action_tokens=self.history_length if self.observation_spec.has_actions else 0,
        )
