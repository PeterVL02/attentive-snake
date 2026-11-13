"""Façade: nn.Module -> diagram string / file."""

from pathlib import Path
from typing import Literal

from torch import nn

from attentive_snake.viz.html import render_html
from attentive_snake.viz.mermaid import render_mermaid
from attentive_snake.viz.text import render_text
from attentive_snake.viz.tree import ModuleNode, build_tree, collapse_repeats, resolve_root

Format = Literal["text", "mermaid", "html"]

_SUFFIX_FORMATS: dict[str, Format] = {
    ".txt": "text",
    ".mmd": "mermaid",
    ".md": "mermaid",
    ".html": "html",
}


def _tree(model: nn.Module, max_depth: int | None, collapse: bool) -> ModuleNode:
    tree = build_tree(resolve_root(model), max_depth=max_depth)
    return collapse_repeats(tree) if collapse else tree


def _render(tree: ModuleNode, fmt: Format) -> str:
    if fmt == "text":
        return render_text(tree)
    if fmt == "mermaid":
        return render_mermaid(tree)
    if fmt == "html":
        return render_html(render_mermaid(tree))
    raise ValueError(f"unknown format {fmt!r}; expected text, mermaid or html")


def diagram(
    model: nn.Module,
    fmt: Format = "text",
    max_depth: int | None = None,
    collapse_repeats: bool = True,
) -> str:
    """Diagram ``model.net`` if it is a module, else ``model`` itself."""
    return _render(_tree(model, max_depth, collapse_repeats), fmt)


def save_diagram(
    model: nn.Module,
    path: str | Path,
    max_depth: int | None = None,
    collapse_repeats: bool = True,
) -> Path:
    """Write a diagram; format is inferred from the suffix (.txt/.mmd/.md/.html)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in _SUFFIX_FORMATS:
        raise ValueError(f"unsupported suffix {suffix!r}; use one of {sorted(_SUFFIX_FORMATS)}")
    content = diagram(model, _SUFFIX_FORMATS[suffix], max_depth, collapse_repeats)
    if suffix == ".md":
        content = f"```mermaid\n{content}```\n"
    path.write_text(content, encoding="utf-8")
    return path
