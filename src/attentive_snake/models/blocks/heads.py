"""Output layers that turn pooled features (B, d_model) into what a head returns."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PolicyHead(nn.Module):
    """Pooled features -> action logits (B, num_actions)."""

    def __init__(self, d_model: int, num_actions: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, num_actions)

    def forward(self, pooled: Tensor) -> Tensor:
        return self.linear(pooled)


class LogStd(nn.Module):
    """A learned log standard deviation per action dimension, the same for every observation: pooled (B, D) -> (B, dimension)."""

    def __init__(self, dimension: int, initial: float = 0.0) -> None:
        super().__init__()
        self.log_std = nn.Parameter(torch.full((dimension,), initial))

    def forward(self, pooled: Tensor) -> Tensor:
        return self.log_std.expand(pooled.shape[0], -1)


class ValueHead(nn.Module):
    """Pooled features -> scalar state value (B,)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, pooled: Tensor) -> Tensor:
        return self.linear(pooled).squeeze(-1)


class QHead(nn.Module):
    """Pooled features -> one Q-value (expected discounted return) per action (B, num_actions)."""

    def __init__(self, d_model: int, num_actions: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, num_actions)

    def forward(self, pooled: Tensor) -> Tensor:
        return self.linear(pooled)


class DuelingStream(nn.Module):
    """One stream of a dueling head: pooled features -> a hidden layer of the same width -> ``outputs`` numbers (B, outputs)."""

    def __init__(self, d_model: int, outputs: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, outputs)

    def forward(self, pooled: Tensor) -> Tensor:
        return self.linear(F.relu(self.hidden(pooled)))


class EmbeddingScore(nn.Module):
    """Embeddings -> one score each, on the last axis: (..., d_model) -> (...)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.linear(embeddings).squeeze(-1)
