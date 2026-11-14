"""Embeddings that add time and action information to tokens."""

import torch
from torch import Tensor, nn


class TemporalEmbedding(nn.Module):
    """Adds a learned embedding per frame to tokens shaped (B, K, ..., D); frame 0 is the oldest."""

    def __init__(self, num_frames: int, d_model: int) -> None:
        super().__init__()
        self.frame = nn.Parameter(torch.randn(num_frames, d_model) * 0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        extra = (1,) * (tokens.ndim - 3)
        return tokens + self.frame.view(1, self.frame.shape[0], *extra, self.frame.shape[1])


class ActionEmbedding(nn.Module):
    """Action index -> vector; index ``num_actions`` is the learned "no action" entry. (B, K) -> (B, K, D)."""

    def __init__(self, num_actions: int, d_model: int) -> None:
        super().__init__()
        self.table = nn.Embedding(num_actions + 1, d_model)

    def forward(self, ids: Tensor) -> Tensor:
        return self.table(ids)
