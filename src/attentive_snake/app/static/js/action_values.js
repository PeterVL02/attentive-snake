// What the network says about each action of the observation the next action is chosen from: action
// probabilities and the state value (actor-critic, PPO) or one Q-value per action (DQN). The readout's
// `kind` and `label` come from the server; nothing here knows the algorithm.

import { h } from "./dom.js";

/** Bar widths in [0, 1]: probabilities as they are; other values (Q-values may be negative) spread from the lowest to the highest. */
function widths(readout) {
  const values = readout.values;
  if (readout.kind === "probabilities") return values;
  const low = Math.min(...values);
  const span = Math.max(...values) - low;
  return values.map((v) => (span > 0 ? (v - low) / span : 1));
}

function format(readout, value) {
  return readout.kind === "probabilities" ? `${(100 * value).toFixed(1)} %` : value.toFixed(3);
}

/** Labelled bars for `readout` ({kind, label, values, state_value}) with `names` per action; the greedy action is marked. */
export function actionValueBars(readout, names) {
  if (!readout) return h("div", { class: "muted small", text: "The episode is over: no action to choose." });
  const best = readout.values.indexOf(Math.max(...readout.values));
  const fill = widths(readout);
  return h(
    "div",
    {},
    h(
      "div",
      { class: "element-weights action-values", role: "list", "aria-label": readout.label },
      readout.values.map((value, i) => {
        const name = names?.[i] ?? `action ${i}`;
        return h(
          "div",
          { class: `element-row${i === best ? " best-action" : ""}`, role: "listitem", title: `${name}: ${format(readout, value)}${i === best ? " (greedy choice)" : ""}` },
          h("span", { class: "element-name", text: i === best ? `${name} ◂` : name }),
          h("div", { class: "element-track" }, h("div", { class: "element-fill", style: `width:${(100 * fill[i]).toFixed(1)}%` })),
          h("span", { class: "element-value", text: format(readout, value) }),
        );
      }),
    ),
    readout.state_value === null
      ? null
      : h("div", { class: "muted small", style: "margin-top:6px", text: `state value (expected discounted return from here): ${readout.state_value.toFixed(3)}` }),
  );
}

/** One line on how to read the bars of `readout`'s kind. */
export function actionValueNote(readout) {
  if (readout?.kind === "q_values") {
    return "Q-values: the discounted return the network expects after each action (in training reward units). The greedy action (◂) has the highest; bars run from the lowest to the highest.";
  }
  return "The policy's probability of each action; the greedy action (◂) is the most likely. The state value is the value head's estimate of the discounted return from here.";
}
