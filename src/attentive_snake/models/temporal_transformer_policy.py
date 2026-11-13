"""Attention across space and time (``kind = "temporal_transformer"``).

Children of ``self.net``, in order:
embed -> positional -> temporal -> [action_embed] -> cls -> blocks -> norm -> pool -> {policy_head, value_head}

Token order (see ``token_layout``), T = 1 + K*H*W (+ K):

    [ class | frame 0 cells (H*W, row-major) | ... | frame K-1 cells | action 0 .. action K-1 ]

Frame 0 is the oldest, frame K-1 the newest. Each cell token embeds the frame's 3 board
channels, plus a spatial position embedding and a learned per-frame (age) embedding.
When frames carry action planes there is one action token per frame: the embedding of
that frame's action (a learned "no action" entry for all-zero planes) plus the same
per-frame embedding. Without action planes there are no action tokens (and no action
parameters). Frames before the episode began are all-zero cells, not masked.
"""

import torch
from torch import Tensor, nn

from attentive_snake.models.blocks import CellEmbedding, PositionalEmbedding2D
from attentive_snake.models.frames import BOARD_CHANNELS, action_ids, board_planes, split_frames
from attentive_snake.models.policy_base import AttentivePolicy
from attentive_snake.models.temporal_blocks import ActionEmbedding, TemporalEmbedding
from attentive_snake.models.token_layout import TokenLayout


class TemporalTransformerPolicy(AttentivePolicy):
    def _build_front(self) -> dict[str, nn.Module]:
        self.frame_spec.require_board_frames()
        d = self.config.d_model
        _, height, width = self.observation_shape
        front: dict[str, nn.Module] = {
            "embed": CellEmbedding(BOARD_CHANNELS, d),
            "positional": PositionalEmbedding2D(height, width, d),
            "temporal": TemporalEmbedding(self.history_length, d),
        }
        if self.frame_spec.has_actions:
            front["action_embed"] = ActionEmbedding(self.num_actions, d)
        return front

    def _tokens(self, obs: Tensor) -> Tensor:
        net = self.net
        frames = split_frames(obs, self.frame_spec)  # (B, K, F, H, W)
        batch = frames.shape[0]
        cells = net["embed"](board_planes(frames).flatten(0, 1))  # (B*K, H*W, D)
        cells = net["temporal"](net["positional"](cells).unflatten(0, (batch, self.history_length)))
        tokens = cells.flatten(1, 2)  # (B, K*H*W, D)
        if self.frame_spec.has_actions:
            actions = net["temporal"](net["action_embed"](action_ids(frames, self.num_actions)))
            tokens = torch.cat([tokens, actions], dim=1)
        return tokens

    @property
    def token_layout(self) -> TokenLayout:
        _, height, width = self.observation_shape
        return TokenLayout(
            num_frames=self.history_length,
            tokens_per_frame=height * width,
            num_action_tokens=self.history_length if self.frame_spec.has_actions else 0,
        )
