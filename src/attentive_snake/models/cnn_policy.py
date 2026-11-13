"""Pure convolutional actor-critic baseline (``kind = "cnn"``).

Children of ``self.net``: encoder -> readout -> {policy_head, value_head}

``encoder``: ``conv_layers`` same-padded convs (``conv_channels`` wide, residual after the first).
``readout``: flatten the whole feature map, linear to ``d_model``, GELU. Flatten-then-linear
was chosen over global mean pooling because the policy depends on *where* the head and
food are (a mean would erase position and dilute the one head cell), and it is possible
because the grid size is fixed per model. Stacked history frames are plain input channels.
There is no attention, hence no ``attention_maps``.
"""

from torch import Tensor, nn

from attentive_snake.models.conv_blocks import ConvEncoder, FlattenReadout
from attentive_snake.models.policy_base import BasePolicy, output_heads


class CnnPolicy(BasePolicy):
    def _build_net(self) -> dict[str, nn.Module]:
        c = self.config
        channels, height, width = self.observation_shape
        return {
            "encoder": ConvEncoder(channels, c.conv_channels, c.conv_layers, c.kernel_size),
            "readout": FlattenReadout(c.conv_channels, height, width, c.d_model),
            **output_heads(c.d_model, self.num_actions),
        }

    def _pooled(self, obs: Tensor, need_weights: bool) -> tuple[Tensor, list[Tensor]]:
        return self.net["readout"](self.net["encoder"](obs)), []
