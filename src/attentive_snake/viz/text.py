"""Render a ModuleNode tree as an indented Unicode tree."""

from attentive_snake.viz.tree import ModuleNode

LEGEND = (
    "Legend: [seq]/[list] children are applied in order; [dict]/[custom] children "
    "are listed in registration order only (true data flow is not known)."
)
_TAGS = {"sequential": "[seq]", "list": "[list]", "dict": "[dict]", "custom": "[custom]"}


def _line(node: ModuleNode) -> str:
    parts = [f"{node.name}: {node.type_name}"]
    if node.kind in _TAGS:
        parts.append(_TAGS[node.kind])
    if node.repeat > 1:
        parts.append(f"× {node.repeat}")
    if node.summary:
        parts.append(f"({node.summary})")
    suffix = " each" if node.repeat > 1 else ""
    parts.append(f"{node.total_params:,} params{suffix}")
    return " ".join(parts)


def _lines(node: ModuleNode, prefix: str, is_last: bool, is_root: bool) -> list[str]:
    if is_root:
        lines, child_prefix = [_line(node)], ""
    else:
        lines = [prefix + ("└── " if is_last else "├── ") + _line(node)]
        child_prefix = prefix + ("    " if is_last else "│   ")
    last = len(node.children) - 1
    for i, child in enumerate(node.children):
        lines.extend(_lines(child, child_prefix, i == last, False))
    return lines


def render_text(node: ModuleNode) -> str:
    return "\n".join([*_lines(node, "", True, True), "", LEGEND])
