"""One ``d_model`` vector per history frame, shared by the policies that reason across frames.

``frame_vector_modules`` makes the children of ``self.net`` that do it (``frame_encoder``
and, when the frames carry action channels, ``action_embed``); ``frame_vectors`` runs them.
"""

from collections import OrderedDict

from torch import Tensor, nn

from attentive_snake.models.blocks.temporal import ActionEmbedding
from attentive_snake.models.frames import action_ids, base_planes, split_frames
from attentive_snake.models.input_stages.stage import Stage
from attentive_snake.models.policies.base import require_action_layout
from attentive_snake.observation_spec import ObservationSpec


class FrameEncoder(nn.Sequential):
    """One frame -> one vector: (N, C, *spatial) -> (N, D), the frame stage's modules in order."""

    def __init__(self, stage: Stage) -> None:
        super().__init__(OrderedDict(stage.modules))


def frame_vector_modules(stage: Stage, spec: ObservationSpec, num_actions: int, d_model: int) -> dict[str, nn.Module]:
    """``frame_encoder`` (from the one-frame ``stage``) and, if frames carry actions, ``action_embed``."""
    require_action_layout(spec, num_actions)
    modules: dict[str, nn.Module] = {"frame_encoder": FrameEncoder(stage)}
    if spec.has_actions:
        modules["action_embed"] = ActionEmbedding(num_actions, d_model)
    return modules


def frame_vectors(net: nn.ModuleDict, obs: Tensor, spec: ObservationSpec, num_actions: int) -> Tensor:
    """(B, K*F, *spatial) -> (B, K, D): every frame encoded, plus its action's embedding if actions exist."""
    frames = split_frames(obs, spec)  # (B, K, F, *spatial)
    batch = frames.shape[0]
    vectors = net["frame_encoder"](base_planes(frames, spec).flatten(0, 1)).unflatten(0, (batch, spec.history_length))
    if spec.has_actions:
        vectors = vectors + net["action_embed"](action_ids(frames, spec, num_actions))
    return vectors
