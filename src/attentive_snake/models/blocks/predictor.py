"""The action-conditioned predictor of a world model: embedding and action -> predicted embedding of the next state.

``ActionPredictor`` works on embeddings only (never on observations), so it can be applied
again to its own predictions (multi-step lookahead).
"""

import torch
from torch import Tensor, nn

from attentive_snake.models.blocks.vector import ResidualLinear


class ActionPredictor(nn.Module):
    """``z' = z + body([z, embedding(a)])``: the residual change one action makes to an embedding.

    ``body`` is a widening linear layer from ``2 * d_model`` to ``d_model`` with a GELU,
    ``layers - 1`` residual linear layers and a linear output. Residual, because one step
    changes a state only a little: the predictor learns the change, and an untrained one
    predicts "nothing changes" plus noise rather than an unrelated vector.
    """

    def __init__(self, d_model: int, num_actions: int, layers: int) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.action_embedding = nn.Embedding(num_actions, d_model)
        self.body = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            *[ResidualLinear(d_model) for _ in range(layers - 1)],
            nn.Linear(d_model, d_model),
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        """Every action at once: (B, D) -> (B, num_actions, D)."""
        batch, width = embeddings.shape
        actions = torch.arange(self.num_actions, device=embeddings.device)
        return self.step(
            embeddings.unsqueeze(1).expand(batch, self.num_actions, width).reshape(-1, width),
            actions.repeat(batch),
        ).view(batch, self.num_actions, width)

    def step(self, embeddings: Tensor, actions: Tensor) -> Tensor:
        """One given action per embedding: (B, D), (B,) -> (B, D)."""
        joined = torch.cat([embeddings, self.action_embedding(actions.long())], dim=-1)
        return embeddings + self.body(joined)
