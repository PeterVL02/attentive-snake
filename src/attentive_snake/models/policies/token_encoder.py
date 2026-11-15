"""``TokenEncoder``: an attention policy's observation -> pooled-feature path as one ``nn.Module``.

For code that needs the encoder as a module of its own (the jepa prediction objective makes
an EMA copy of it). It holds the very modules it is given, so it shares their weights with
the policy; it is never registered on the policy, so the policy's state dict is unchanged.
"""

from collections.abc import Mapping, Sequence

from torch import Tensor, nn

from attentive_snake.models.policies.base import pool_tokens

TAIL_NAMES = ("cls", "blocks", "norm", "pool")  # the modules of base.transformer_tail


class TokenEncoder(nn.Module):
    """``front`` (applied in order: observation -> tokens), then class token -> blocks -> norm -> pool from ``tail``."""

    def __init__(self, front: Sequence[nn.Module], tail: Mapping[str, nn.Module]) -> None:
        super().__init__()
        self.front = nn.Sequential(*front)
        self.tail = nn.ModuleDict({name: tail[name] for name in TAIL_NAMES})

    def forward(self, obs: Tensor) -> Tensor:
        pooled, _ = pool_tokens(self.tail, self.front(obs), need_weights=False)
        return pooled
