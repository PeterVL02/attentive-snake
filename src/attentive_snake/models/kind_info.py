"""Plain facts about one policy kind, for UIs and docs (no behaviour)."""

from dataclasses import dataclass
from typing import Literal

type AttentionView = Literal["grid", "frames", "weights"]

TRANSFORMER_FIELDS = ("d_model", "num_heads", "num_layers", "mlp_ratio", "dropout")
CONV_FIELDS = ("conv_channels", "conv_layers", "kernel_size")


@dataclass(frozen=True)
class KindInfo:
    """What a ``ModelConfig.kind`` is and which settings it reacts to.

    Attributes:
        description: One or two plain sentences on the architecture.
        fields: Names of ``ModelConfig`` fields (besides ``kind``) that change this kind.
        uses_history: Frames/actions get their own tokens (history is time, not extra channels).
        attention_view: What attention can be shown: ``"grid"`` (one board heatmap),
            ``"frames"`` (a board heatmap per frame), ``"weights"`` (one weight per frame)
            or ``None`` (no attention).
    """

    description: str
    fields: tuple[str, ...]
    uses_history: bool
    attention_view: AttentionView | None
