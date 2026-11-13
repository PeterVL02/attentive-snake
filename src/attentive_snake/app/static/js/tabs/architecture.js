// Tab 3: the policy network of the New-training form or of an existing run.

import { get, post } from "../api.js";
import { use } from "../bus.js";
import { formatNumber, h, message, replace } from "../dom.js";
import { renderMermaid } from "../mermaid_view.js";
import { fillRunSelect } from "../run_options.js";

const CAVEAT =
  "The diagram shows how modules are nested and registered, not the true data flow: solid arrows connect " +
  "children that are applied in order (Sequential / ModuleList); dotted arrows only follow registration order, " +
  "so skips and branches inside forward() are not visible.";

let els = {};

function source() {
  return els.root.querySelector('input[name="arch-source"]:checked').value;
}

async function render() {
  message(els.status, "info", "Building the model…");
  replace(els.result);
  let data;
  try {
    if (source() === "form") {
      const form = use("formConfig");
      if (!form) throw new Error("The New-training form is not ready yet.");
      if (form.errors) throw new Error("The New-training form has invalid fields; fix them there first.");
      data = await post("/api/architecture", form.tables);
    } else {
      if (!els.run.value) throw new Error("Choose a run.");
      data = await get(`/api/runs/${encodeURIComponent(els.run.value)}/architecture`);
    }
  } catch (error) {
    message(els.status, "error", error.message);
    return;
  }
  message(els.status, "", "");
  const diagram = h("div", { class: "mermaid-out" }, h("div", { class: "muted small", text: "Rendering diagram…" }));
  replace(
    els.result,
    h(
      "div",
      { class: "stats", style: "margin-bottom:16px" },
      h("div", { class: "stat" }, h("div", { class: "k", text: "kind" }), h("div", { class: "v", text: data.kind })),
      h("div", { class: "stat" }, h("div", { class: "k", text: "parameters" }), h("div", { class: "v", text: formatNumber(data.parameters) })),
      h("div", { class: "stat" }, h("div", { class: "k", text: "observation (C×H×W)" }), h("div", { class: "v", text: data.observation_shape.join("×") })),
    ),
    h("p", { class: "muted small", text: CAVEAT }),
    h("div", { class: "panel" }, h("h3", { text: "Diagram" }), diagram),
    h("div", { class: "panel" }, h("h3", { text: "Module tree" }), h("pre", { class: "tree", text: data.text })),
  );
  try {
    await renderMermaid(diagram, data.mermaid);
  } catch (error) {
    replace(diagram, h("div", { class: "message info", text: `The diagram could not be drawn (${error.message || error}). Mermaid is loaded from cdn.jsdelivr.net, so it needs internet access; the module tree below has the same content.` }));
  }
}

export function init(root) {
  els.root = root;
  els.run = h("select", { id: "arch-run", "aria-label": "run" });
  els.status = h("div", { class: "message", role: "status" });
  els.result = h("div", {});
  const radio = (value, text) =>
    h("label", { class: "row", style: "gap:6px" }, h("input", { type: "radio", name: "arch-source", value, checked: value === "form", onchange: render }), text);
  replace(
    root,
    h("h2", { text: "Architecture" }),
    h("p", { class: "lead", text: "The network a config builds (on the CPU, untrained), with its parameter count." }),
    h(
      "div",
      { class: "panel" },
      h("div", { class: "row" }, radio("form", "New-training form"), radio("run", "Existing run:"), els.run, h("button", { type: "button", class: "primary", text: "Show", onclick: render })),
      h("div", { style: "margin-top:10px" }, els.status),
    ),
    els.result,
  );
  els.run.addEventListener("change", () => {
    els.root.querySelector('input[value="run"]').checked = true;
    render();
  });
}

export async function show(params = {}) {
  try {
    await fillRunSelect(els.run, params.run || els.run.value);
  } catch (error) {
    message(els.status, "error", error.message);
  }
  if (params.source) els.root.querySelector(`input[value="${params.source}"]`).checked = true;
  if (params.source || !els.result.hasChildNodes()) render();
}
