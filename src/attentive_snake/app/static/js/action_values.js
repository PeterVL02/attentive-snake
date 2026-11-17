// What the network says about each action of the observation the next action is chosen from: action
// probabilities and the state value (actor-critic, PPO), one Q-value per action (DQN), or for continuous
// actions the mean action per number with its spread ("action_vector", drawn between the game's bounds).
// The readout's `kind` and `label` come from the server; nothing here knows the algorithm or the game.

import { h } from "./dom.js";

const isVector = (readout) => readout.kind === "action_vector";

/** Bar widths in [0, 1]: probabilities as they are; an action vector from each number's low to its high bound
 * (clipped there, as the game clips it); other values (Q-values may be negative) spread from the lowest to the highest. */
function widths(readout, bounds) {
  const values = readout.values;
  if (readout.kind === "probabilities") return values;
  if (isVector(readout) && bounds) {
    return values.map((v, i) => Math.min(1, Math.max(0, (v - bounds.low[i]) / (bounds.high[i] - bounds.low[i]))));
  }
  const low = Math.min(...values);
  const span = Math.max(...values) - low;
  return values.map((v) => (span > 0 ? (v - low) / span : 1));
}

function format(readout, value, i) {
  if (readout.kind === "probabilities") return `${(100 * value).toFixed(1)} %`;
  if (isVector(readout)) return `${value.toFixed(2)} ± ${readout.spread[i].toFixed(2)}`;
  return value.toFixed(3);
}

/** Labelled bars for `readout` ({kind, label, values, state_value, spread?}) with `names` per action (or per number of
 * an action vector, drawn within `bounds` {low, high}); the greedy action is marked (an action vector has none to mark). */
export function actionValueBars(readout, names, bounds = null) {
  if (!readout) return h("div", { class: "muted small", text: "The episode is over: no action to choose." });
  const best = isVector(readout) ? -1 : readout.values.indexOf(Math.max(...readout.values));
  const fill = widths(readout, bounds);
  return h(
    "div",
    {},
    h(
      "div",
      { class: "element-weights action-values", role: "list", "aria-label": readout.label },
      readout.values.map((value, i) => {
        const name = names?.[i] ?? (isVector(readout) ? `number ${i}` : `action ${i}`);
        return h(
          "div",
          { class: `element-row${i === best ? " best-action" : ""}`, role: "listitem", title: `${name}: ${format(readout, value, i)}${i === best ? " (greedy choice)" : ""}` },
          h("span", { class: "element-name", text: i === best ? `${name} ◂` : name }),
          h("div", { class: "element-track" }, h("div", { class: "element-fill", style: `width:${(100 * fill[i]).toFixed(1)}%` })),
          h("span", { class: "element-value", text: format(readout, value, i) }),
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
  if (readout && isVector(readout)) {
    return "The mean action: the numbers the policy plays when not sampling, each ± its spread (the standard deviation it samples with). Bars run from each number's lowest to its highest allowed value; the game clips a number beyond them.";
  }
  if (readout?.kind === "q_values") {
    return "Q-values: the discounted return the network expects after each action (in training reward units). The greedy action (◂) has the highest; bars run from the lowest to the highest.";
  }
  return "The policy's probability of each action; the greedy action (◂) is the most likely. The state value is the value head's estimate of the discounted return from here.";
}
