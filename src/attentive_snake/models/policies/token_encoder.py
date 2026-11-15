"""An attention policy's observation -> embedding path as one ``nn.Module``.

For code that needs the encoder as a module of its own (the jepa prediction objective makes
an EMA copy of it). Each holds the very modules it is given, so it shares their weights with
the policy; it is never registered on the policy, so the policy's state dict is unchanged.

* ``TokenSetEncoder``: observation -> the encoded token set (B, 1 + N, D), class token first
  (``jepa_tokens``: its state embedding is every token).
* ``TokenEncoder``: the same, then the class-token pool -> (B, D) (``jepa_attention``, ``jepa_temporal``).
"""

from collections.abc import Mapping, Sequence
from typing import ClassVar

from torch import Tensor, nn

from attentive_snake.models.policies.base import encode_tokens

TOKEN_SET_NAMES = ("cls", "blocks", "norm")  # the modules of base.transformer_tail before the pool
TAIL_NAMES = (*TOKEN_SET_NAMES, "pool")


class TokenSetEncoder(nn.Module):
    """``front`` (applied in order: observation -> tokens), then class token -> blocks -> norm from ``tail``."""

    names: ClassVar[tuple[str, ...]] = TOKEN_SET_NAMES

    def __init__(self, front: Sequence[nn.Module], tail: Mapping[str, nn.Module]) -> None:
        super().__init__()
        self.front = nn.Sequential(*front)
        self.tail = nn.ModuleDict({name: tail[name] for name in self.names})

    def forward(self, obs: Tensor) -> Tensor:
        tokens, _ = encode_tokens(self.tail, self.front(obs), need_weights=False)
        return tokens


class TokenEncoder(TokenSetEncoder):
    """``TokenSetEncoder``, then ``tail["pool"]`` (the class token): observation -> (B, D)."""

    names: ClassVar[tuple[str, ...]] = TAIL_NAMES

    def forward(self, obs: Tensor) -> Tensor:
        return self.tail["pool"](super().forward(obs))
