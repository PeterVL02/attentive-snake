"""The per-token predictor of a world model: a token set and an action -> the predicted token set of the next state.

Where ``blocks.predictor.ActionPredictor`` predicts one vector per state, ``TokenPredictor``
predicts every token (one per board cell or observed number, plus the class token). A step
of a board game changes few cells, so most tokens only have to be copied: the predictor is
residual and learns the change. It reads token sets only (never observations), so it can be
applied again to its own predictions (multi-step lookahead).
"""

import torch
from torch import Tensor, nn

from attentive_snake.models.blocks.transformer import TransformerBlock, run_blocks


class TokenPredictor(nn.Module):
    """``tokens' = tokens + out(norm(blocks(tokens + embedding(a))))``: the residual change one action makes to every token.

    The action's learned embedding is added to every token, so each token (each cell) knows the
    action without having to attend to it; ``layers`` transformer blocks let the cells exchange
    what changes (the head moves into a neighbour, the tail leaves, an apple appears); a final
    norm and linear layer give the change.
    """

    def __init__(self, d_model: int, num_actions: int, layers: int, num_heads: int, mlp_ratio: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.action_embedding = nn.Embedding(num_actions, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, mlp_ratio, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        """Every action at once: (B, T, D) -> (B, num_actions, T, D)."""
        batch, count, width = tokens.shape
        actions = torch.arange(self.num_actions, device=tokens.device)
        repeated = tokens.unsqueeze(1).expand(batch, self.num_actions, count, width).reshape(-1, count, width)
        return self.step(repeated, actions.repeat(batch)).view(batch, self.num_actions, count, width)

    def step(self, tokens: Tensor, actions: Tensor) -> Tensor:
        """One given action per token set: (B, T, D), (B,) -> (B, T, D)."""
        conditioned = tokens + self.action_embedding(actions.long()).unsqueeze(1)
        changed, _ = run_blocks(self.blocks, conditioned)
        return tokens + self.out(self.norm(changed))
