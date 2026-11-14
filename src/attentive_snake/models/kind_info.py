"""Plain facts about one policy kind, for UIs and docs (no behaviour)."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

type AttentionView = Literal["grid", "frames", "weights"]

TRANSFORMER_FIELDS = ("d_model", "num_heads", "num_layers", "mlp_ratio", "dropout")
CONV_FIELDS = ("conv_channels", "conv_layers", "kernel_size")


@dataclass(frozen=True)
class InputReading:
    """How a kind reads one observation kind.

    Attributes:
        description: One or two plain sentences on what the kind does with such observations.
        recommended: Whether the kind suits such observations (a sensible first choice).
        unused_fields: Fields of ``KindInfo.fields`` that have no effect for such observations
            (e.g. ``kernel_size`` when there is nothing to convolve).
    """

    description: str
    recommended: bool
    unused_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class KindInfo:
    """What a ``ModelConfig.kind`` is and which settings it reacts to.

    Attributes:
        fields: Names of ``ModelConfig`` fields (besides ``kind``) that change this kind.
        uses_history: Frames/actions get their own tokens (history is time, not extra channels).
        attention_view: What attention can be shown: ``"grid"`` (one heatmap of the
            positions), ``"frames"`` (a heatmap of the positions per frame), ``"weights"``
            (one weight per frame) or ``None`` (no attention). Positions are board cells for
            grid observations; for vector observations "grid" and "frames" mean one weight
            per element (number) of the observation (``WatchFrame.element_weights``).
        inputs: ``{observation kind: InputReading}`` for every observation kind the kind
            can read (the keys equal the policy class's ``observation_kinds``).
    """

    fields: tuple[str, ...]
    uses_history: bool
    attention_view: AttentionView | None
    inputs: Mapping[str, InputReading]

    @property
    def observation_kinds(self) -> tuple[str, ...]:
        return tuple(self.inputs)

    def fields_for(self, observation_kind: str) -> tuple[str, ...]:
        """``fields`` that matter for ``observation_kind``."""
        unused = self.inputs[observation_kind].unused_fields if observation_kind in self.inputs else ()
        return tuple(name for name in self.fields if name not in unused)
