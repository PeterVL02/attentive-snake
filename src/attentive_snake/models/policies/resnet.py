"""Residual network (``kind = "resnet"``): the ``cnn`` layout with two-convolution residual blocks.

Children of ``self.net``: encoder -> readout -> head (policy_head + value_head, or q_head: models.heads)
(from ``RESIDUAL_STAGES``).

Grid: ``encoder`` is a conv stem (``conv_channels`` wide) and ``conv_layers`` residual
blocks of two convolutions with a skip connection, all the same width and without
downsampling (boards are tiny); ``readout`` flattens the whole feature map, linear to
``d_model``, GELU, as in the ``cnn`` kind.
Vector: ``encoder`` is a linear embedding and ``conv_layers`` residual MLP blocks
(LayerNorm, linear, GELU, linear, plus the skip; SimBa, arXiv 2410.09754), ``readout`` a
linear layer to ``d_model``. Stacked history frames are plain input channels / numbers.
No attention, hence no ``attention_maps``.
"""

from torch import Tensor, nn

from attentive_snake.models.input_stages.catalog import RESIDUAL_STAGES, readable_kinds
from attentive_snake.models.input_stages.stage import run_stage, whole_observation
from attentive_snake.models.policies.base import BasePolicy


class ResnetPolicy(BasePolicy):
    observation_kinds = readable_kinds(RESIDUAL_STAGES)

    def _build_net(self) -> dict[str, nn.Module]:
        stage = self._input_stage(RESIDUAL_STAGES, whole_observation(self.observation_spec))
        self._stage_names = tuple(stage.modules)
        return dict(stage.modules)

    def _pooled(self, obs: Tensor, need_weights: bool) -> tuple[Tensor, list[Tensor]]:
        return run_stage(self.net, self._stage_names, obs), []
