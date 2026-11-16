// The form that starts a reward search of the selected game: a run and its checkpoint, the reward fields to search
// (ticked, with the run's current values), and the settings. Built from /api/reward-search/form; nothing game-specific.

import { get, post } from "./api.js";
import { formatNumber, h, message, replace } from "./dom.js";
import { attachServerError, fieldControl, readFields } from "./fields.js";
import { gameUrl } from "./game.js";
import { fillRunSelect, setCheckpointOptions } from "./run_options.js";

/** {el, load()}: `load()` (re)fills it for `game`; `onStarted(name)` is called after a search has started. */
export function searchForm(game, { onStarted }) {
  const name = h("input", { type: "text", id: "search-name", spellcheck: "false", placeholder: "e.g. step-death" });
  const run = h("select", { id: "search-run" });
  const checkpoint = h("select", { id: "search-checkpoint" });
  const fieldList = h("div", { class: "search-fields" });
  const settingsBox = h("div", { class: "fields" });
  const start = h("button", { type: "button", class: "primary", text: "Start search" });
  const note = h("div", { class: "message", role: "alert" });
  let boxes = []; // [{name, input}]
  let widgets = [];
  let rows = []; // the game's searchable reward fields

  async function showRun() {
    replace(fieldList, h("p", { class: "muted small", text: "Loading the run…" }));
    if (!run.value) {
      replace(fieldList, h("p", { class: "muted small", text: "Choose a run first." }));
      return;
    }
    const detail = await get(gameUrl(`/api/runs/${encodeURIComponent(run.value)}?log_lines=1`, game));
    setCheckpointOptions(checkpoint, detail.checkpoints);
    const current = detail.config?.reward ?? {};
    const ticked = new Set(boxes.filter((b) => b.input.checked).map((b) => b.name));
    boxes = rows.map((row) => ({ name: row.name, input: h("input", { type: "checkbox", id: `search-field-${row.name}`, checked: ticked.has(row.name) }) }));
    replace(
      fieldList,
      rows.length === 0 ? h("p", { class: "muted small", text: "This game has no reward coefficients to search." }) : null,
      rows.map((row, i) =>
        h(
          "div",
          { class: "field checkbox" },
          h("label", { for: boxes[i].input.id }, boxes[i].input, ` ${row.name} `, h("span", { class: "muted small", text: `(now ${formatNumber(current[row.name] ?? row.default)})` })),
          h("div", { class: "description", text: row.description }),
        ),
      ),
    );
  }

  async function submit() {
    message(note, "", "");
    const { values, errors } = readFields(widgets);
    const fields = boxes.filter((b) => b.input.checked).map((b) => b.name);
    if (errors) return;
    if (!name.value.trim()) return message(note, "error", "Give the search a name.");
    if (fields.length === 0) return message(note, "error", "Tick at least one reward field to search.");
    start.disabled = true;
    try {
      const body = { name: name.value.trim(), parent: run.value, checkpoint: checkpoint.value || null, fields, settings: values };
      const started = await post(gameUrl("/api/reward-search", game), body);
      message(note, "info", `Started ${started.name}. Its runs wait in the training queue and appear in the Runs tab as they start.`);
      onStarted(started.name);
    } catch (error) {
      if (!attachServerError(widgets, error.message)) message(note, "error", error.message);
    } finally {
      start.disabled = false;
    }
  }

  async function load() {
    try {
      const form = await get(gameUrl("/api/reward-search/form", game));
      rows = form.fields;
      widgets = form.settings.map((spec) => fieldControl(spec));
      replace(settingsBox, widgets.map((w) => w.el));
      await fillRunSelect(run, game, run.value);
      await showRun();
    } catch (error) {
      message(note, "error", `Could not load the form: ${error.message}`);
    }
  }

  run.addEventListener("change", () => showRun().catch((error) => message(note, "error", error.message)));
  start.addEventListener("click", submit);
  const el = h(
    "div",
    { class: "panel" },
    h("h3", { text: "Start a reward search" }),
    h("div", { class: "row" },
      h("label", { for: "search-name", class: "label", text: "name" }), name,
      h("label", { for: "search-run", class: "label", text: "start from run" }), run,
      h("label", { for: "search-checkpoint", class: "label", text: "checkpoint" }), checkpoint),
    h("h4", { text: "Reward fields to search" }),
    fieldList,
    h("h4", { text: "Settings" }),
    settingsBox,
    h("div", { class: "row" }, start),
    note,
  );
  return { el, load };
}
