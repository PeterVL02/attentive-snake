"""Output layers that turn pooled features (B, d_model) into what a head returns."""

from torch import Tensor, nn


class PolicyHead(nn.Module):
    """Pooled features -> action logits (B, num_actions)."""

    def __init__(self, d_model: int, num_actions: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, num_actions)

    def forward(self, pooled: Tensor) -> Tensor:
        return self.linear(pooled)


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


class EmbeddingScore(nn.Module):
    """Embeddings -> one score each, on the last axis: (..., d_model) -> (...)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.linear(embeddings).squeeze(-1)
