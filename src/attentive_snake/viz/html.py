"""Wrap Mermaid source in a standalone HTML page."""

from html import escape

MERMAID_VERSION = "11.4.0"
_MERMAID_URL = (
    f"https://cdn.jsdelivr.net/npm/mermaid@{MERMAID_VERSION}/dist/mermaid.esm.min.mjs"
)

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>body {{ font-family: sans-serif; margin: 1.5rem; }}</style>
</head>
<body>
<h1>{title}</h1>
<pre class="mermaid">
{source}
</pre>
<script type="module">
import mermaid from "{url}";
mermaid.initialize({{ startOnLoad: true, securityLevel: "strict" }});
</script>
</body>
</html>
"""


def render_html(mermaid_source: str, title: str = "Model architecture") -> str:
    return _TEMPLATE.format(
        title=escape(title), source=escape(mermaid_source), url=_MERMAID_URL
    )
