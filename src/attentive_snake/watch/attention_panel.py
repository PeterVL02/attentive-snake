"""Which attention picture to show for a policy, decided from what the policy exposes.

A policy with ``attention_maps`` and ``token_layout`` has attention; whether its tokens are
cells (a board heatmap of one frame) or whole frames (one weight per frame) follows from
``token_layout.tokens_per_frame``. Nothing here looks at class names.
"""

from dataclasses import dataclass

import numpy as np
from torch import Tensor, nn

from attentive_snake.viz.attention import cls_attention_frames, cls_attention_over_frames


@dataclass(frozen=True)
class AttentionPanel:
    heatmap: np.ndarray | None = None  # (H, W) in [0, 1]
    frame_weights: np.ndarray | None = None  # (K,), oldest first
    caption: str = ""


def has_attention(policy: nn.Module) -> bool:
    return hasattr(policy, "attention_maps") and hasattr(policy, "token_layout")


def num_attention_layers(policy: nn.Module) -> int:
    return policy.config.num_layers if has_attention(policy) else 0


def num_history_frames(policy: nn.Module) -> int:
    return policy.token_layout.num_frames if has_attention(policy) else 1


def check_attention_options(policy: nn.Module, layer: int, frame: int) -> None:
    """``ValueError`` if ``layer`` / ``frame`` do not index the policy's layers / frames (no attention: anything goes)."""
    if not has_attention(policy):
        return
    layers, frames = num_attention_layers(policy), num_history_frames(policy)
    if not -layers <= layer < layers:
        raise ValueError(f"layer {layer} out of range for {layers} layer(s) (use -{layers}..{layers - 1})")
    if not -frames <= frame < frames:
        raise ValueError(f"frame {frame} out of range for {frames} frame(s) (use -{frames}..{frames - 1})")


def attention_panel(policy: nn.Module, observation: Tensor, layer: int, frame: int) -> AttentionPanel:
    """Attention for one observation (1, C, H, W): a cell heatmap of one frame, or one weight per frame."""
    if not has_attention(policy):
        return AttentionPanel(caption="model has no attention")
    layout = policy.token_layout
    maps = policy.attention_maps(observation)
    if layout.tokens_per_frame == 1:
        weights = cls_attention_over_frames(maps, layout.num_frames, layer=layer)
        return AttentionPanel(frame_weights=weights, caption=f"layer {layer}, attention per frame")
    _, height, width = policy.observation_shape
    frames = cls_attention_frames(maps, layout.num_frames, height, width, layer=layer)
    return AttentionPanel(heatmap=frames[frame], caption=f"layer {layer}, frame {frame}")
