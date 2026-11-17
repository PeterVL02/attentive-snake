// The reward searches of the selected game (a list) and one search in full: per generation, each member's values of the
// searched reward fields, its state, its score and the winner, with Stop. Labels come from the game's metric definitions.

import { post } from "./api.js";
import { badge, formatNumber, h, message, replace } from "./dom.js";
import { formatMetric, metricDefinitions } from "./evaluation_metrics.js";
import { navigate } from "./bus.js";

const MEMBER_STATES = { planned: "planned", waiting: "waiting in the queue", training: "training", done: "done", failed: "failed" };

function objectiveDefinition(key) {
  return metricDefinitions().find((d) => d.key === key) ?? { key, label: key, format: "number", unit: "" };
}

/** The list: one row per search, the chosen one marked; `onOpen(name)` when a row's name is clicked. */
export function searchList(searches, chosen, onOpen) {
  if (searches.length === 0) return h("p", { class: "muted small", text: "No reward searches for this game yet." });
  return h(
    "table",
    {},
    h("thead", {}, h("tr", {}, ["search", "from run", "searched fields", "generations", "best so far", "status"].map((text) => h("th", { text })))),
    h("tbody", {}, searches.map((s) => {
      const definition = objectiveDefinition(s.objective);
      return h(
        "tr",
        { "aria-current": String(s.name === chosen) },
        h("td", {}, h("button", { type: "button", class: "link", text: s.name, onclick: () => onOpen(s.name) })),
        h("td", { class: "mono", text: s.parent }),
        h("td", { text: s.fields.join(", ") }),
        h("td", { class: "num", text: `${s.generations_judged} / ${s.generations}` }),
        h("td", { class: "num", text: s.best_score === null ? "–" : `${definition.label}: ${formatMetric(definition, s.best_score)}` }),
        h("td", {}, badge(s.status)),
      );
    })),
  );
}

function memberRow(member, fields, definition, isWinner) {
  const runLink = h("button", { type: "button", class: "link mono", text: member.run, onclick: () => navigate("runs", { run: member.run }) });
  return h(
    "tr",
    {},
    h("td", { class: "num", text: member.index === 0 ? "0 (unchanged)" : String(member.index) }),
    h("td", {}, runLink),
    fields.map((field) => h("td", { class: "num", text: formatNumber(member.reward[field]) })),
    h("td", {}, h("span", { text: MEMBER_STATES[member.state] ?? member.state }), member.error ? h("div", { class: "small", text: member.error }) : null),
    h("td", { class: "num", text: member.state === "done" ? formatMetric(definition, member.score) : "–" }),
    h("td", { text: isWinner ? "winner" : "" }),
  );
}

/** One search in full ({summary fields, state}); `onChanged()` after Stop. */
export function searchDetail(detail, onChanged) {
  const { state } = detail;
  const fields = state.spec.fields;
  const definition = objectiveDefinition(state.objective);
  const note = h("div", { class: "message", role: "alert" });
  const stop = h("button", { type: "button", class: "danger", text: "Stop search" });
  stop.hidden = state.status !== "running";
  stop.addEventListener("click", async () => {
    stop.disabled = true;
    try {
      await post(`/api/reward-search/${encodeURIComponent(state.spec.name)}/stop`);
      onChanged();
    } catch (error) {
      message(note, "error", error.message);
      stop.disabled = false;
    }
  });
  const settings = state.spec.settings;
  const direction = state.higher_is_better ? "higher is better" : "lower is better";
  const generations = state.generations.map((generation) =>
    h(
      "div",
      { style: "margin-top:12px" },
      h("h4", { text: `Generation ${generation.index}: from ${generation.parent} (${generation.checkpoint ?? "latest checkpoint"})` }),
      h(
        "table",
        {},
        h("thead", {}, h("tr", {}, ["member", "run", ...fields, "state", definition.label, ""].map((text) => h("th", { text })))),
        h("tbody", {}, generation.members.map((m) => memberRow(m, fields, definition, generation.winner === m.index))),
      ),
    ),
  );
  return h(
    "div",
    { class: "panel" },
    h("div", { class: "row spread" }, h("h3", { text: `Search ${state.spec.name}`, style: "margin:0" }), h("div", { class: "row" }, badge(state.status), stop)),
    h("p", {
      class: "muted small",
      text: `Starts from ${state.spec.parent} (${state.spec.checkpoint}). ${settings.population} members x ${settings.generations} generations x ${formatNumber(settings.steps_per_member)} env steps. Judged by ${definition.label} (${direction}) over ${settings.evaluation_games} greedy games.`,
    }),
    state.error ? h("div", { class: "message error", text: state.error }) : null,
    h("p", { class: "small", text: `Starting reward: ${fields.map((f) => `${f} = ${formatNumber(state.start_reward[f])}`).join(", ")}.` }),
    generations.length === 0 ? h("p", { class: "muted small", text: "The first generation is being planned." }) : generations,
    note,
  );
}
