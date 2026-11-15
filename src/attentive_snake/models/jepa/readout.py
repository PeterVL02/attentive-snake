"""What the decoders of a jepa kind read from a state embedding (current or predicted).

A jepa kind's state embedding is either one vector per state (``jepa``, ``jepa_attention``:
``(..., D)``) or a token set (``jepa_tokens``: ``(..., T, D)``, class token first). The value
and policy decoders always read one ``(..., D)`` vector per state; a readout turns the
embedding into it.
"""

from torch import Tensor


def whole(embedding: Tensor) -> Tensor:
    """The embedding is the vector: (..., D) -> (..., D)."""
    return embedding


def class_token(tokens: Tensor) -> Tensor:
    """The class token of a token set: (..., T, D) -> (..., D)."""
    return tokens[..., 0, :]
