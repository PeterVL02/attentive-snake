"""Introspection only: build an immutable tree from an ``nn.Module``.

The tree reflects *containment and registration order*, not data flow.
"""

from dataclasses import dataclass, replace
from typing import Literal

from torch import nn

from attentive_snake.viz.summaries import category_of, is_atomic, summarize

Kind = Literal["leaf", "sequential", "list", "dict", "custom"]


@dataclass(frozen=True)
class ModuleNode:
    name: str
    type_name: str
    kind: Kind
    summary: str
    category: str
    own_params: int
    total_params: int
    children: tuple["ModuleNode", ...] = ()
    repeat: int = 1  # >1 when this node stands for a run of identical siblings


def resolve_root(module: nn.Module) -> nn.Module:
    """Return ``module.net`` if it is an ``nn.Module``, else ``module``."""
    net = getattr(module, "net", None)
    return net if isinstance(net, nn.Module) else module


def _kind_of(module: nn.Module, has_children: bool) -> Kind:
    if not has_children:
        return "leaf"
    if isinstance(module, nn.Sequential):
        return "sequential"
    if isinstance(module, nn.ModuleList):
        return "list"
    if isinstance(module, nn.ModuleDict):
        return "dict"
    return "custom"


def build_tree(
    module: nn.Module, max_depth: int | None = None, name: str = "net"
) -> ModuleNode:
    """Build a :class:`ModuleNode` tree.

    ``max_depth`` counts levels below the root (0 = root only); nodes at the
    limit keep their parameter totals but lose their children.
    """
    expand = not is_atomic(module) and (max_depth is None or max_depth > 0)
    children: tuple[ModuleNode, ...] = ()
    if expand:
        next_depth = None if max_depth is None else max_depth - 1
        children = tuple(
            build_tree(child, next_depth, child_name)
            for child_name, child in module.named_children()
        )
    return ModuleNode(
        name=name,
        type_name=type(module).__name__,
        kind=_kind_of(module, bool(children)),
        summary=summarize(module),
        category=category_of(module),
        own_params=sum(p.numel() for p in module.parameters(recurse=False)),
        total_params=sum(p.numel() for p in module.parameters()),
        children=children,
    )


def _same_structure(a: ModuleNode, b: ModuleNode) -> bool:
    return replace(a, name="") == replace(b, name="")


def collapse_repeats(node: ModuleNode) -> ModuleNode:
    """Merge runs of structurally identical consecutive siblings.

    A run of ``n`` nodes becomes one node named ``first..last`` with
    ``repeat=n``; its parameter counts stay per instance.
    """
    merged: list[ModuleNode] = []
    run: list[ModuleNode] = []

    def flush() -> None:
        if len(run) == 1:
            merged.append(run[0])
        elif run:
            merged.append(
                replace(run[0], name=f"{run[0].name}..{run[-1].name}", repeat=len(run))
            )
        run.clear()

    for child in (collapse_repeats(c) for c in node.children):
        if run and not _same_structure(run[0], child):
            flush()
        run.append(child)
    flush()
    return replace(node, children=tuple(merged))
