"""Conv stem + transformer (``kind = "hybrid"``).

Children of ``self.net``, in order:
stem -> project -> tokens -> positional -> cls -> blocks -> norm -> pool -> {policy_head, value_head}

``stem`` is a ConvEncoder, so each cell's feature already summarises its neighbourhood;
``project`` (1x1 conv) maps to ``d_model``; ``tokens`` turns cells into tokens.
Token order: [class token, cell 0 .. cell H*W-1] (row-major); see ``token_layout``.
"""

from torch import Tensor, nn

from attentive_snake.models.blocks import PositionalEmbedding2D
from attentive_snake.models.conv_blocks import ConvEncoder, GridToTokens
from attentive_snake.models.policy_base import AttentivePolicy
from attentive_snake.models.token_layout import TokenLayout


class HybridPolicy(AttentivePolicy):
    def _build_front(self) -> dict[str, nn.Module]:
        c = self.config
        channels, height, width = self.observation_shape
        return {
            "stem": ConvEncoder(channels, c.conv_channels, c.conv_layers, c.kernel_size),
            "project": nn.Conv2d(c.conv_channels, c.d_model, kernel_size=1),
            "tokens": GridToTokens(),
            "positional": PositionalEmbedding2D(height, width, c.d_model),
        }

    def _tokens(self, obs: Tensor) -> Tensor:
        net = self.net
        return net["positional"](net["tokens"](net["project"](net["stem"](obs))))

    @property
    def token_layout(self) -> TokenLayout:
        _, height, width = self.observation_shape
        return TokenLayout(num_frames=1, tokens_per_frame=height * width)
