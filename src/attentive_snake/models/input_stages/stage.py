"""``Stage``: named modules that read the observation, and ``StageInput``: what they are given."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from torch import Tensor, nn

from attentive_snake.observation_spec import ObservationSpec


@dataclass(frozen=True)
class StageInput:
    """An input ``(N, values * channels, *spatial)``: ``values`` stacked copies (frames) of a ``channels``-channel frame.

    For a grid every cell has ``values * channels`` numbers; for a vector every one of the
    ``channels`` elements has ``values`` numbers (its value in each stacked frame).
    """

    values: int
    channels: int
    spatial: tuple[int, ...]


def whole_observation(spec: ObservationSpec) -> StageInput:
    """The full stacked observation (all frames, game and action channels)."""
    return StageInput(spec.history_length, spec.frame_channels, spec.spatial_shape)


def one_frame(spec: ObservationSpec) -> StageInput:
    """One frame's game channels only (actions are read separately)."""
    return StageInput(1, spec.base_channels, spec.spatial_shape)


@dataclass(frozen=True)
class Stage:
    """Modules applied in order. Their names become keys of the policy's ``net`` and so of
    its checkpoints: renaming one breaks every saved model.

    ``num_tokens`` is the number of tokens a token stage makes per input (cells of a grid,
    elements of a vector); 0 for stages that make one feature vector.
    """

    modules: Mapping[str, nn.Module]
    num_tokens: int = 0


def run_stage(net: nn.ModuleDict, names: Iterable[str], x: Tensor) -> Tensor:
    """Apply ``net[name]`` for each name in turn."""
    for name in names:
        x = net[name](x)
    return x
