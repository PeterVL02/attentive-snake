"""Small single-purpose building blocks of the transformer policies.

Token layout: ``(batch, tokens, d_model)``. Grid tokens are in row-major order.
"""

import torch
from torch import Tensor, nn


class CellEmbedding(nn.Module):
    """Each grid cell's channel vector becomes one token: (B,C,H,W) -> (B,H*W,D)."""

    def __init__(self, in_channels: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)

    def forward(self, obs: Tensor) -> Tensor:
        tokens = obs.flatten(2).transpose(1, 2)  # (B, H*W, C), row-major
        return self.proj(tokens)


class PositionalEmbedding2D(nn.Module):
    """Adds a learned row embedding plus a learned column embedding to each cell token."""

    def __init__(self, height: int, width: int, d_model: int) -> None:
        super().__init__()
        self.row = nn.Parameter(torch.randn(height, 1, d_model) * 0.02)
        self.col = nn.Parameter(torch.randn(1, width, d_model) * 0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        grid = (self.row + self.col).flatten(0, 1)  # (H*W, D), row-major
        return tokens + grid


class ClassToken(nn.Module):
    """Prepends a learned summary token: (B,N,D) -> (B,N+1,D)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        cls = self.token.expand(tokens.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1)


class SelfAttention(nn.Module):
    """Multi-head self-attention over the token axis."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self, tokens: Tensor, need_weights: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        """Returns (output, per-head weights (B, heads, T, T) or None)."""
        return self.attn(
            tokens,
            tokens,
            tokens,
            need_weights=need_weights,
            average_attn_weights=False,
        )


class FeedForward(nn.Module):
    """Position-wise MLP: D -> D*ratio -> D."""

    def __init__(self, d_model: int, mlp_ratio: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * mlp_ratio)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_model * mlp_ratio, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.drop(self.fc2(self.act(self.fc1(tokens))))


class TransformerBlock(nn.Module):
    """Pre-norm residual block: x + Attn(LN(x)), then x + FFN(LN(x))."""

    def __init__(
        self, d_model: int, num_heads: int, mlp_ratio: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = SelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, mlp_ratio, dropout)

    def forward(
        self, tokens: Tensor, need_weights: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        """Returns (tokens, attention weights or None when ``need_weights`` is False)."""
        attended, weights = self.attention(self.norm1(tokens), need_weights)
        tokens = tokens + attended
        return tokens + self.feed_forward(self.norm2(tokens)), weights


def run_blocks(
    blocks: nn.ModuleList, tokens: Tensor, need_weights: bool = False
) -> tuple[Tensor, list[Tensor]]:
    """Apply transformer blocks in turn; returns (tokens, per-block attention maps, empty unless asked)."""
    maps: list[Tensor] = []
    for block in blocks:
        tokens, weights = block(tokens, need_weights)
        if weights is not None:
            maps.append(weights)
    return tokens, maps


class ClassTokenPool(nn.Module):
    """Selects the class token as the pooled state summary: (B,N+1,D) -> (B,D)."""

    def forward(self, tokens: Tensor) -> Tensor:
        return tokens[:, 0]
