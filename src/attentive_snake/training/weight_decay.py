"""Which parameters of a network weight decay shrinks, and which it leaves alone.

Decayed: the weight matrices of the layers that transform their input: linear layers,
convolutions, attention (its input projection) and recurrent layers. Left alone: every bias,
every normalisation layer's scale and shift, and everything that is looked up or added rather
than multiplied with an input: embedding tables, positional / per-frame embeddings, class
tokens. Pulling those towards zero does not make the function simpler, it only erases what
they store.

The rule is a whitelist by the type of the module that owns the parameter, so a parameter of
a kind of module this file has never seen is not decayed.
"""

from dataclasses import dataclass

from torch import nn

DECAYED_LAYERS: tuple[type[nn.Module], ...] = (nn.Linear, nn.modules.conv._ConvNd, nn.MultiheadAttention, nn.RNNBase)


@dataclass(frozen=True)
class DecayGroups:
    """A network's parameters split by whether weight decay applies; each list is in ``named_parameters()`` order."""

    decayed: tuple[nn.Parameter, ...]
    exempt: tuple[nn.Parameter, ...]

    def ordered(self) -> tuple[nn.Parameter, ...]:
        """All parameters, decayed first: the order an optimizer built from both groups numbers them in."""
        return self.decayed + self.exempt


def decayed_names(module: nn.Module) -> set[str]:
    """Names (as in ``module.named_parameters()``) of the parameters weight decay applies to."""
    return {
        f"{prefix}.{name}" if prefix else name
        for prefix, owner in module.named_modules()
        if isinstance(owner, DECAYED_LAYERS)
        for name, parameter in owner.named_parameters(recurse=False)
        if parameter.ndim >= 2
    }


def decay_groups(module: nn.Module) -> DecayGroups:
    """Every parameter of ``module`` exactly once, in the group it belongs to."""
    decayed = decayed_names(module)
    named = list(module.named_parameters())
    return DecayGroups(
        tuple(parameter for name, parameter in named if name in decayed),
        tuple(parameter for name, parameter in named if name not in decayed),
    )
