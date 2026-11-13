"""Render a ModuleNode tree as a Mermaid flowchart.

Conventions (also emitted as comments in the output):
  * Containers are subgraphs; leaves are boxes coloured by category.
  * Solid arrows ``-->``: children of Sequential / ModuleList, applied in order.
  * Dotted arrows ``-.->``: children of ModuleDict / custom modules, shown in
    registration order only; this is NOT data flow (skips and branches in
    forward() are invisible to the module tree).
"""

from collections.abc import Iterator
from itertools import count

from attentive_snake.viz.tree import ModuleNode

CATEGORIES = {
    "linear": "fill:#dbeafe,stroke:#2563eb,color:#111",
    "conv": "fill:#dcfce7,stroke:#16a34a,color:#111",
    "attention": "fill:#fee2e2,stroke:#dc2626,color:#111",
    "norm": "fill:#fef9c3,stroke:#ca8a04,color:#111",
    "activation": "fill:#f3e8ff,stroke:#9333ea,color:#111",
    "other": "fill:#f3f4f6,stroke:#6b7280,color:#111",
    "container": "fill:#ffffff,stroke:#9ca3af,stroke-dasharray:4 3,color:#111",
}
_ORDERED_KINDS = {"sequential", "list"}


def escape_label(text: str) -> str:
    """Escape text for use inside a double-quoted Mermaid label."""
    for char, entity in (("#", "#35;"), ('"', "#quot;"), ("<", "#lt;"), (">", "#gt;")):
        text = text.replace(char, entity)
    return text


def _label(node: ModuleNode, with_params: bool = True) -> str:
    title = f"{node.name}: {node.type_name}"
    if node.repeat > 1:
        title += f" × {node.repeat}"
    lines = [escape_label(title)]
    if node.summary:
        lines.append(escape_label(node.summary))
    if with_params:
        suffix = " each" if node.repeat > 1 else ""
        lines.append(f"{node.total_params:,} params{suffix}")
    return "<br/>".join(lines)


def _emit(node: ModuleNode, ids: Iterator[int], lines: list[str], indent: str) -> str:
    """Append mermaid lines for ``node`` and return its id."""
    node_id = f"n{next(ids)}"
    if not node.children:
        lines.append(f'{indent}{node_id}["{_label(node)}"]:::{node.category}')
        return node_id
    lines.append(f'{indent}subgraph {node_id}["{_label(node)}"]')
    child_ids = [_emit(c, ids, lines, indent + "  ") for c in node.children]
    arrow = "-->" if node.kind in _ORDERED_KINDS else "-.->"
    for a, b in zip(child_ids, child_ids[1:]):
        lines.append(f"{indent}  {a} {arrow} {b}")
    lines.append(f"{indent}end")
    lines.append(f"{indent}class {node_id} container")
    return node_id


def render_mermaid(node: ModuleNode) -> str:
    lines = [
        "flowchart TD",
        "%% Solid arrow: applied in order (Sequential / ModuleList).",
        "%% Dotted arrow: registration order only (ModuleDict / custom) - not data flow.",
    ]
    lines += [f"classDef {name} {style}" for name, style in CATEGORIES.items()]
    _emit(node, count(), lines, "")
    return "\n".join(lines) + "\n"
