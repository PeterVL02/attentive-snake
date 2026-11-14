// Rendering Mermaid source to SVG with Mermaid loaded lazily from a pinned CDN build.

const MERMAID_URL = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs";

let loading = null;
let counter = 0;

function loadMermaid() {
  loading ??= import(MERMAID_URL).then(({ default: mermaid }) => {
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: dark ? "dark" : "neutral", flowchart: { subGraphTitleMargin: { top: 8, bottom: 30 } } });
    return mermaid;
  });
  loading.catch(() => (loading = null));
  return loading;
}

/** Render `source` into `container`; throws if Mermaid cannot be loaded (e.g. offline) or parsed. */
export async function renderMermaid(container, source) {
  const mermaid = await loadMermaid();
  counter += 1;
  const { svg } = await mermaid.render(`mermaid-${counter}`, source);
  container.innerHTML = svg; // produced by Mermaid with securityLevel "strict" from our own diagram source
}
