import re

import pytest
from torch import nn

from attentive_snake.viz import diagram, register_summary, save_diagram
from attentive_snake.viz.mermaid import escape_label, render_mermaid
from attentive_snake.viz.summaries import summarize
from attentive_snake.viz.tree import build_tree, collapse_repeats


class Block(nn.Module):
    def __init__(self, d=16, heads=2):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, d))


class Transformerish(nn.Module):
    def __init__(self, n_blocks=3):
        super().__init__()
        self.net = nn.ModuleDict(
            {
                "embed": nn.Embedding(100, 16),
                "blocks": nn.ModuleList(Block() for _ in range(n_blocks)),
                "head": nn.Linear(16, 4),
            }
        )


def mlp():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def convnet():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.Flatten(), nn.Linear(16, 2)
    )


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def check_mermaid(src):
    lines = [ln.strip() for ln in src.splitlines()]
    assert lines[0] == "flowchart TD"
    opens = sum(ln.startswith("subgraph ") for ln in lines)
    assert opens == sum(ln == "end" for ln in lines)
    defined = re.findall(r"^(?:subgraph )?(n\d+)\[", "\n".join(lines), re.M)
    assert len(defined) == len(set(defined))
    for ln in lines:
        m = re.fullmatch(r"(n\d+) (?:-->|-\.->) (n\d+)", ln)
        if m:
            assert m.group(1) in defined and m.group(2) in defined
    # no raw quotes inside labels
    for ln in lines:
        if "[" in ln and not ln.startswith("%%"):
            body = ln[ln.index('["') + 2 : ln.rindex('"]')]
            assert '"' not in body
    return defined


def test_mlp_text_and_summaries():
    out = diagram(mlp())
    assert "0: Linear (4 → 8)" in out
    assert "[seq]" in out and "2: Linear (8 → 2)" in out


def test_conv_summary():
    assert summarize(nn.Conv2d(3, 16, 3, padding=1)) == "3 → 16, k3 s1 p1"
    assert summarize(nn.LayerNorm(64)) == "64"
    assert summarize(nn.Dropout(0.1)) == "p=0.1"
    assert summarize(nn.Embedding(100, 64)) == "100 × 64"
    assert summarize(nn.MultiheadAttention(64, 4)) == "d=64, heads=4"
    assert "16" in diagram(convnet()) and "BatchNorm2d" in diagram(convnet())


def test_extra_repr_fallback_and_registry():
    class Odd(nn.Module):
        def extra_repr(self):
            return "x" * 200

    assert len(summarize(Odd())) <= 60

    class Mine(nn.Module):
        pass

    register_summary(Mine, lambda m: "custom!")
    assert summarize(Mine()) == "custom!"


def test_net_attribute_vs_none():
    model = Transformerish()
    assert diagram(model).startswith("net: ModuleDict")
    plain = Block()
    assert diagram(plain).startswith("net: Block")
    assert "attn: MultiheadAttention" in diagram(plain)


def test_param_counts():
    model = Transformerish()
    tree = build_tree(model.net)
    assert tree.total_params == nparams(model)

    def walk(n):
        assert n.total_params >= n.own_params
        if n.children:
            assert n.total_params == n.own_params + sum(c.total_params for c in n.children)
        for c in n.children:
            walk(c)

    walk(tree)


def test_mha_is_atomic_and_encoder_handled():
    tree = build_tree(nn.MultiheadAttention(16, 2))
    assert tree.kind == "leaf" and tree.total_params == nparams(nn.MultiheadAttention(16, 2))
    enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(16, 2, 32), num_layers=3)
    out = diagram(enc)
    assert "× 3" in out and "TransformerEncoderLayer" in out and "heads=2" in out
    check_mermaid(diagram(enc, "mermaid"))


def test_collapse_repeats():
    tree = build_tree(Transformerish(6).net)
    blocks = next(c for c in tree.children if c.name == "blocks")
    assert len(blocks.children) == 6
    collapsed = collapse_repeats(tree)
    cb = next(c for c in collapsed.children if c.name == "blocks")
    assert len(cb.children) == 1 and cb.children[0].repeat == 6
    assert cb.children[0].name == "0..5"
    assert collapsed.total_params == tree.total_params
    assert "× 6" in diagram(Transformerish(6))
    assert "× 6" not in diagram(Transformerish(6), collapse_repeats=False)


def test_collapse_keeps_different_siblings():
    tree = collapse_repeats(build_tree(nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 8))))
    assert [c.repeat for c in tree.children] == [2, 1]


def test_max_depth():
    model = Transformerish(2)
    t0 = build_tree(model.net, max_depth=0)
    assert t0.children == () and t0.total_params == nparams(model)
    t1 = build_tree(model.net, max_depth=1)
    assert all(not c.children for c in t1.children)
    assert t1.total_params == nparams(model)


def test_mermaid_structure():
    for model in (mlp(), convnet(), Transformerish(3), Block()):
        for collapse in (True, False):
            check_mermaid(diagram(model, "mermaid", collapse_repeats=collapse))
    src = diagram(Transformerish(3), "mermaid")
    assert "-.->" in src and "-->" in src
    for cat in ("linear", "attention", "norm", "activation"):
        assert f":::{cat}" in src


def test_mermaid_escaping():
    assert escape_label('a "b" <c> #d') == "a #quot;b#quot; #lt;c#gt; #35;d"
    node = build_tree(nn.Sequential(nn.Linear(2, 2)), name='we"ird[name]')
    check_mermaid(render_mermaid(node))


def test_html_wraps_mermaid():
    out = diagram(mlp(), "html")
    assert out.startswith("<!doctype html>")
    assert "mermaid@" in out and 'class="mermaid"' in out
    assert "flowchart TD" in out and "-->" not in out.replace("--&gt;", "")


def test_unknown_format():
    with pytest.raises(ValueError):
        diagram(mlp(), "png")  # type: ignore[arg-type]


@pytest.mark.parametrize("suffix", [".txt", ".mmd", ".md", ".html"])
def test_save_each_suffix(tmp_path, suffix):
    path = save_diagram(Transformerish(2), tmp_path / f"m{suffix}")
    content = path.read_text(encoding="utf-8")
    assert content.strip()
    if suffix == ".md":
        assert content.startswith("```mermaid\nflowchart TD") and content.rstrip().endswith("```")
    if suffix == ".mmd":
        assert content.startswith("flowchart TD")
    if suffix == ".html":
        assert "<html" in content


def test_save_bad_suffix(tmp_path):
    with pytest.raises(ValueError):
        save_diagram(mlp(), tmp_path / "m.png")
