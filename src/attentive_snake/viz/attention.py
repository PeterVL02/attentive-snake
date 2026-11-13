"""Turn per-layer attention maps into (height, width) grids for display.

Token layout: index 0 is the class token, 1.. are grid cells in row-major order
(``cls_attention_frames`` / ``cls_attention_over_frames`` take the layout explicitly).
"""

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor


def _normalise(grid: np.ndarray) -> np.ndarray:
    peak = grid.max()
    return (grid / peak if peak > 0 else grid).astype(np.float32)


def cls_attention_grid(
    maps: Sequence[Tensor],
    height: int,
    width: int,
    layer: int = -1,
    batch_index: int = 0,
) -> np.ndarray:
    """Class-token attention to each cell in one layer, head-averaged, scaled to max 1."""
    weights = maps[layer][batch_index].detach().float().mean(dim=0)  # (T, T)
    cells = weights[0, 1:].cpu().numpy()
    return _normalise(cells.reshape(height, width))


def attention_rollout(
    maps: Sequence[Tensor], height: int, width: int, batch_index: int = 0
) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema): class-token flow to cells across all layers."""
    first = maps[0][batch_index].detach().float().mean(dim=0)
    identity = torch.eye(first.shape[-1], device=first.device)
    rollout = identity
    for layer_maps in maps:
        attn = layer_maps[batch_index].detach().float().mean(dim=0)
        attn = 0.5 * attn + 0.5 * identity  # residual connection
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rollout = attn @ rollout
    cells = rollout[0, 1:].cpu().numpy()
    return _normalise(cells.reshape(height, width))


def _cls_row(maps: Sequence[Tensor], layer: int, batch_index: int) -> np.ndarray:
    """Head-averaged attention of the class token (token 0) to every token: (T,)."""
    return maps[layer][batch_index].detach().float().mean(dim=0)[0].cpu().numpy()


def cls_attention_frames(
    maps: Sequence[Tensor],
    num_frames: int,
    height: int,
    width: int,
    layer: int = -1,
    batch_index: int = 0,
    offset: int = 1,
) -> np.ndarray:
    """Class-token attention to the cells of each frame: (K, H, W), oldest frame first.

    Cell tokens of frame k start at ``offset + k * H * W`` (row-major). Scaled by one
    common maximum, so brighter frames really receive more attention.
    """
    cells = _cls_row(maps, layer, batch_index)[offset : offset + num_frames * height * width]
    return _normalise(cells.reshape(num_frames, height, width))


def cls_attention_over_frames(
    maps: Sequence[Tensor],
    num_frames: int,
    layer: int = -1,
    batch_index: int = 0,
    offset: int = 1,
) -> np.ndarray:
    """Class-token attention to the one token of each frame, renormalised to sum 1: (K,)."""
    weights = _cls_row(maps, layer, batch_index)[offset : offset + num_frames]
    total = weights.sum()
    return (weights / total if total > 0 else weights).astype(np.float32)
