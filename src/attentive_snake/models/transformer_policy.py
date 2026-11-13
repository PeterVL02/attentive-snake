"""Actor-critic transformer over grid cells (``kind = "transformer"``).

Children of ``self.net``, in order:
embed -> positional -> cls -> blocks (each applied in turn) -> norm -> pool -> {policy_head, value_head}

Every cell is one token built from all its channels; with history (K > 1) the
stacked frames simply appear as extra channels of each cell.
Token order: [class token, cell 0 .. cell H*W-1] (row-major); see ``token_layout``.

Pooling uses a learned class token rather than mean pooling: the snake head is a
single cell out of H*W, and a mean would dilute it to 1/(H*W) of the signal, whereas
the class token can attend to the cells that matter (head, nearby body, food).
"""

from torch import Tensor, nn

from attentive_snake.models.blocks import CellEmbedding, PositionalEmbedding2D
from attentive_snake.models.policy_base import AttentivePolicy
from attentive_snake.models.token_layout import TokenLayout


class SnakeTransformerPolicy(AttentivePolicy):
    def _build_front(self) -> dict[str, nn.Module]:
        channels, height, width = self.observation_shape
        return {
            "embed": CellEmbedding(channels, self.config.d_model),
            "positional": PositionalEmbedding2D(height, width, self.config.d_model),
        }

    def _tokens(self, obs: Tensor) -> Tensor:
        return self.net["positional"](self.net["embed"](obs))

    @property
    def token_layout(self) -> TokenLayout:
        _, height, width = self.observation_shape
        return TokenLayout(num_frames=1, tokens_per_frame=height * width)
