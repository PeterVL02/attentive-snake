// Attention weights as bar charts: one vertical bar per history frame, or one labelled horizontal
// bar per observation element (vector games). Bars are scaled so the largest weight fills the bar.

import { h } from "./dom.js";

const FRAME_BAR_PX = 70;

function empty() {
  return h("div", { class: "muted small", text: "No weights for this frame." });
}

/** Bars for `weights` per frame, oldest first (the last is "now"). */
export function frameWeightBars(weights) {
  if (!weights) return empty();
  const peak = Math.max(...weights, 1e-9);
  return h(
    "div",
    { class: "weights" },
    weights.map((w, i) =>
      h(
        "div",
        { class: "bar", title: `frame ${i + 1}: ${w.toFixed(3)}` },
        h("span", { text: w.toFixed(2) }),
        h("div", { style: `height:${Math.max(1, (FRAME_BAR_PX * w) / peak)}px` }),
        h("span", { text: i === weights.length - 1 ? "now" : `t-${weights.length - 1 - i}` }),
      ),
    ),
  );
}

/** Labelled horizontal bars for `weights` per observation element; `names` may be missing or short. */
export function elementWeightBars(weights, names) {
  if (!weights) return empty();
  const peak = Math.max(...weights, 1e-9);
  return h(
    "div",
    { class: "element-weights", role: "list" },
    weights.map((w, i) => {
      const name = names?.[i] ?? `element ${i}`;
      return h(
        "div",
        { class: "element-row", role: "listitem", title: `${name}: ${w.toFixed(3)}` },
        h("span", { class: "element-name", text: name }),
        h("div", { class: "element-track" }, h("div", { class: "element-fill", style: `width:${((100 * w) / peak).toFixed(1)}%` })),
        h("span", { class: "element-value", text: w.toFixed(2) }),
      );
    }),
  );
}
