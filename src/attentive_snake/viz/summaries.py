"""One-line summaries and categories for ``nn.Module`` types.

Responsibility: turn a single module (not its children) into a short string.
Custom classes can be added with :func:`register_summary`.
"""

from collections.abc import Callable

from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _NormBase

SummaryFn = Callable[[nn.Module], str]

MAX_SUMMARY_LEN = 60

_REGISTRY: dict[type, SummaryFn] = {}
_ATOMIC: set[type] = set()


def register_summary(
    module_type: type, fn: SummaryFn, *, atomic: bool = False
) -> None:
    """Register ``fn`` as the summary for ``module_type`` (and subclasses).

    ``atomic=True`` asks the tree builder not to recurse into its children.
    """
    _REGISTRY[module_type] = fn
    if atomic:
        _ATOMIC.add(module_type)
    else:
        _ATOMIC.discard(module_type)


def _lookup(module: nn.Module) -> type | None:
    for cls in type(module).__mro__:
        if cls in _REGISTRY:
            return cls
    return None


def is_atomic(module: nn.Module) -> bool:
    """True if the module should be drawn as one box despite having children."""
    cls = _lookup(module)
    return cls is not None and cls in _ATOMIC


def _truncate(text: str, limit: int = MAX_SUMMARY_LEN) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def summarize(module: nn.Module) -> str:
    """One-line summary of the key hyperparameters of ``module``."""
    cls = _lookup(module)
    if cls is not None:
        return _truncate(_REGISTRY[cls](module))
    try:
        return _truncate(module.extra_repr())
    except Exception:  # a broken custom extra_repr must not kill the diagram
        return ""


def category_of(module: nn.Module) -> str:
    """Coarse category: linear/conv/attention/norm/activation/other."""
    if isinstance(module, nn.MultiheadAttention):
        return "attention"
    if isinstance(module, (nn.Linear, nn.Bilinear)):
        return "linear"
    if isinstance(module, _ConvNd):
        return "conv"
    if isinstance(module, (_NormBase, nn.LayerNorm, nn.GroupNorm, nn.RMSNorm)):
        return "norm"
    if type(module).__module__ == "torch.nn.modules.activation":
        return "activation"
    return "other"


def _tuple_or_int(value: object) -> str:
    if isinstance(value, (tuple, list)):
        if len(set(value)) == 1:
            return str(value[0])
        return "(" + ",".join(str(v) for v in value) + ")"
    return str(value)


def _conv(m: _ConvNd) -> str:
    return (
        f"{m.in_channels} → {m.out_channels}, k{_tuple_or_int(m.kernel_size)} "
        f"s{_tuple_or_int(m.stride)} p{_tuple_or_int(m.padding)}"
    )


def _mha(m: nn.MultiheadAttention) -> str:
    return f"d={m.embed_dim}, heads={m.num_heads}"


def _encoder_layer(m: nn.TransformerEncoderLayer) -> str:
    attn = m.self_attn
    return f"d={attn.embed_dim}, heads={attn.num_heads}, ff={m.linear1.out_features}"


def _num_features(m: nn.Module) -> str:
    return str(m.num_features)  # type: ignore[attr-defined]


register_summary(nn.Linear, lambda m: f"{m.in_features} → {m.out_features}")  # type: ignore[attr-defined]
register_summary(_ConvNd, _conv)  # type: ignore[arg-type]
register_summary(nn.Embedding, lambda m: f"{m.num_embeddings} × {m.embedding_dim}")  # type: ignore[attr-defined]
register_summary(nn.MultiheadAttention, _mha, atomic=True)  # type: ignore[arg-type]
register_summary(nn.TransformerEncoderLayer, _encoder_layer)  # type: ignore[arg-type]
register_summary(nn.LayerNorm, lambda m: _tuple_or_int(m.normalized_shape))  # type: ignore[attr-defined]
register_summary(nn.GroupNorm, lambda m: f"{m.num_groups} groups, {m.num_channels} ch")  # type: ignore[attr-defined]
register_summary(_NormBase, _num_features)
register_summary(nn.modules.dropout._DropoutNd, lambda m: f"p={m.p}")  # type: ignore[attr-defined]
