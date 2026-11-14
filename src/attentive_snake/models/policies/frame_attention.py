"""CNN per frame, attention across frames (``kind = "frame_attention"``).

Children of ``self.net``, in order:
frame_encoder -> [action_embed] -> temporal -> cls -> blocks -> norm -> pool -> {policy_head, value_head}

``frame_encoder`` (shared by all frames) encodes each frame's 3 board channels into one
``d_model`` vector (ConvEncoder + FlattenReadout). That vector gets the frame's action
embedding added (when action planes exist; a learned "no action" entry for all-zero
planes) and a learned per-frame embedding.

Token order (see ``token_layout``), T = 1 + K:  [ class | frame 0 | ... | frame K-1 ]
Frame 0 is the oldest, frame K-1 the newest.

Readout is the class token rather than the newest frame's token: the same pool as the
other attention policies, and it can mix any frames, with its attention over frames
directly showing which frame the decision draws on (the newest frame can still be
attended by the class token trivially, so nothing is lost).
"""

from torch import Tensor, nn

from attentive_snake.models.blocks.conv import ConvEncoder, FlattenReadout
from attentive_snake.models.blocks.temporal import ActionEmbedding, TemporalEmbedding
from attentive_snake.models.config import ModelConfig
from attentive_snake.models.frames import BOARD_CHANNELS, action_ids, board_planes, split_frames
from attentive_snake.models.policies.base import AttentivePolicy
from attentive_snake.models.token_layout import TokenLayout


class FrameEncoder(nn.Module):
    """One board frame -> one vector: (N, 3, H, W) -> (N, D)."""

    def __init__(self, config: ModelConfig, height: int, width: int) -> None:
        super().__init__()
        self.conv = ConvEncoder(BOARD_CHANNELS, config.conv_channels, config.conv_layers, config.kernel_size)
        self.readout = FlattenReadout(config.conv_channels, height, width, config.d_model)

    def forward(self, boards: Tensor) -> Tensor:
        return self.readout(self.conv(boards))


class FrameAttentionPolicy(AttentivePolicy):
    def _build_front(self) -> dict[str, nn.Module]:
        self.frame_spec.require_board_frames()
        _, height, width = self.observation_shape
        front: dict[str, nn.Module] = {"frame_encoder": FrameEncoder(self.config, height, width)}
        if self.frame_spec.has_actions:
            front["action_embed"] = ActionEmbedding(self.num_actions, self.config.d_model)
        front["temporal"] = TemporalEmbedding(self.history_length, self.config.d_model)
        return front

    def _tokens(self, obs: Tensor) -> Tensor:
        net = self.net
        frames = split_frames(obs, self.frame_spec)  # (B, K, F, H, W)
        batch = frames.shape[0]
        tokens = net["frame_encoder"](board_planes(frames).flatten(0, 1)).unflatten(0, (batch, self.history_length))
        if self.frame_spec.has_actions:
            tokens = tokens + net["action_embed"](action_ids(frames, self.num_actions))
        return net["temporal"](tokens)

    @property
    def token_layout(self) -> TokenLayout:
        return TokenLayout(num_frames=self.history_length, tokens_per_frame=1)
