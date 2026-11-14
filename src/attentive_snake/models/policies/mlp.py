"""Plain multilayer perceptron (``kind = "mlp"``): the honest baseline for vector games.

Children of ``self.net``: encoder -> {policy_head, value_head}

``encoder`` flattens the whole observation (all frames and action numbers; a grid is
simply flattened) and applies ``num_layers`` fully connected layers ``d_model`` wide (the
first widening, the rest residual), GELU after each. No attention, no structure assumed.
"""

import math

from torch import Tensor, nn

from attentive_snake.models.blocks.vector import MlpEncoder
from attentive_snake.models.policies.base import BasePolicy, output_heads


class MlpPolicy(BasePolicy):
    observation_kinds = ("grid", "vector")

    def _build_net(self) -> dict[str, nn.Module]:
        c = self.config
        return {
            "encoder": MlpEncoder(math.prod(self.observation_shape), c.d_model, c.num_layers),
            **output_heads(c.d_model, self.num_actions),
        }

    def _pooled(self, obs: Tensor, need_weights: bool) -> tuple[Tensor, list[Tensor]]:
        return self.net["encoder"](obs), []
