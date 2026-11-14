"""Actor-critic baseline without attention (``kind = "cnn"``).

Children of ``self.net``: encoder -> readout -> head (policy_head + value_head, or q_head: models.heads) (from ``FEATURE_STAGES``).

Grid: ``encoder`` is ``conv_layers`` same-padded convs (``conv_channels`` wide, residual
after the first); ``readout`` flattens the whole feature map, linear to ``d_model``, GELU.
Flatten-then-linear was chosen over global mean pooling because the policy depends on
*where* the head and food are (a mean would erase position and dilute the one head cell),
and it is possible because the grid size is fixed per model.
Vector: a convolution over an arbitrary order of numbers is meaningless, so ``encoder`` is
an MLP of the same width and depth and ``readout`` a linear layer to ``d_model``.
Stacked history frames are plain input channels / numbers. There is no attention, hence
no ``attention_maps``.
"""

from torch import Tensor, nn

from attentive_snake.models.input_stages.catalog import FEATURE_STAGES, readable_kinds
from attentive_snake.models.input_stages.stage import run_stage, whole_observation
from attentive_snake.models.policies.base import BasePolicy


class CnnPolicy(BasePolicy):
    observation_kinds = readable_kinds(FEATURE_STAGES)

    def _build_net(self) -> dict[str, nn.Module]:
        stage = self._input_stage(FEATURE_STAGES, whole_observation(self.observation_spec))
        self._stage_names = tuple(stage.modules)
        return dict(stage.modules)

    def _pooled(self, obs: Tensor, need_weights: bool) -> tuple[Tensor, list[Tensor]]:
        return run_stage(self.net, self._stage_names, obs), []
