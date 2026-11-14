// "Start a new run from a checkpoint": a dialog that starts a new run of the same game from a
// checkpoint of an existing run (a fork), with changed rewards or training settings.

import { ApiError, get, post } from "./api.js";
import { formatNumber, h, message } from "./dom.js";
import { attachServerError, descriptionField, fieldControl, readFields } from "./fields.js";
import { gameUrl } from "./game.js";
import { loadSchema } from "./schema.js";

const PARENT_VALUES = ""; // reward-scheme option value: keep the parent's reward values
const HIDDEN = new Set(["train.checkpoint_dir"]); // set by the server to the new run's folder

function clone(config) {
  return JSON.parse(JSON.stringify(config));
}

function suggestName(parent, taken) {
  const base = parent.replace(/-v\d+$/, "");
  for (let n = 2; ; n += 1) {
    const name = `${base}-v${n}`;
    if (!taken.has(name)) return name;
  }
}

function same(a, b) {
  return JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
}

/**
 * Open the dialog for starting a new run of `game` from run `parent` (its detail from /api/runs/{name})
 * at `checkpoint` ({name, steps}); `onStarted(name)` is called once the new run has been launched.
 */
export async function openForkForm(game, parent, checkpoint, onStarted) {
  const [{ schema }, rules, schemes, runs] = await Promise.all([
    loadSchema(game),
    get(gameUrl("/api/fork-rules", game)),
    get(gameUrl("/api/rewards", game)).then((r) => r.schemes).catch(() => []),
    get(gameUrl("/api/runs", game)).catch(() => []),
  ]);
  const locked = rules.locked;
  const config = parent.config;
  const editable = []; // [section, widget]
  const lockedWidgets = [];

  function widget(section, spec) {
    const key = `${section}.${spec.name}`;
    const w = fieldControl(spec, config[section][spec.name]);
    if (key in locked) {
      w.input.disabled = true;
      w.el.classList.add("locked");
      w.el.querySelector(".description").textContent = `Locked: ${locked[key]}.`;
      lockedWidgets.push(w);
    } else {
      editable.push([section, w]);
      w.onChange(updateNote);
    }
    return w;
  }

  function fieldsOf(section, filter = () => true) {
    return schema.sections[section].filter((spec) => !HIDDEN.has(`${section}.${spec.name}`) && filter(spec)).map((spec) => widget(section, spec));
  }

  const usedModelFields = new Set(["kind", ...(schema.model_kinds[config.model.kind]?.fields ?? [])]);
  const rewardWidgets = fieldsOf("reward");
  const trainWidgets = [...fieldsOf("train"), ...fieldsOf(config.train.algorithm)];
  const evaluationWidgets = fieldsOf("evaluation");
  const runWidgets = [...fieldsOf("env", (s) => !(`env.${s.name}` in locked)), ...fieldsOf("game", (s) => !(`game.${s.name}` in locked))];
  const lockedShown = [
    ...fieldsOf("model", (s) => usedModelFields.has(s.name)),
    ...fieldsOf("game", (s) => `game.${s.name}` in locked),
    ...fieldsOf("env", (s) => `env.${s.name}` in locked),
  ];

  const taken = new Set([parent.name, ...runs.map((r) => r.name)]);
  const nameWidget = fieldControl(
    { name: "new run name", type: "str", nullable: false, description: `A new directory under runs/${game}/; must differ from every existing run of this game.` },
    suggestName(parent.name, taken),
  );
  const description = descriptionField();
  const note = h("div", { class: "message", role: "status" });
  const formError = h("div", { class: "message", role: "alert" });

  const schemeSelect = h(
    "select",
    { id: "fork-reward-scheme" },
    h("option", { value: PARENT_VALUES, text: `${parent.name}'s values` }),
    schemes.map((s) => h("option", { value: s, text: s })),
  );
  schemeSelect.addEventListener("change", async () => {
    try {
      const values = schemeSelect.value === PARENT_VALUES ? config.reward : await get(gameUrl(`/api/rewards/${encodeURIComponent(schemeSelect.value)}`, game));
      for (const w of rewardWidgets) w.set(values[w.spec.name]);
      updateNote();
    } catch (error) {
      message(formError, "error", error.message);
    }
  });

  /** The new run's config: the parent's, with every editable field read from the form. */
  function readConfig() {
    const tables = clone(config);
    let errors = 0;
    for (const [section, w] of editable) {
      const result = readFields([w]);
      errors += result.errors;
      if (!result.errors) tables[section][w.spec.name] = result.values[w.spec.name];
    }
    return { tables, errors };
  }

  function changedFields() {
    return editable.filter(([section, w]) => {
      const result = w.get();
      return !result.error && !same(result.value, config[section][w.spec.name]);
    });
  }

  function updateNote() {
    const changed = changedFields();
    if (changed.length === 0) {
      message(note, "info", "No settings changed yet. Starting now copies this checkpoint into a new run with the same settings; to simply train the same run longer, close this and use \u201cResume this run\u201d.");
    } else {
      message(note, "info", `Changed: ${changed.map(([section, w]) => `${section}.${w.spec.name}`).join(", ")}.`);
    }
  }

  const startButton = h("button", { type: "button", class: "primary", text: "Start new run" });
  const cancelButton = h("button", { type: "button", text: "Cancel" });
  const panel = (title, ...children) => h("section", { class: "panel" }, h("h3", { text: title }), ...children);
  const dialog = h(
    "dialog",
    { class: "fork-dialog", "aria-labelledby": "fork-title" },
    h(
      "div",
      { class: "dialog-body" },
      h("h2", { id: "fork-title", text: `New run from ${parent.name}` }),
      h("p", {
        class: "lead",
        text: `A new run starts from the weights and optimizer state of ${parent.name} at ${formatNumber(checkpoint.steps)} env steps (${checkpoint.name}). It counts its own steps from 0 with its own total_env_steps, and its learning-rate schedule starts again from learning_rate. ${parent.name} itself is not changed.`,
      }),
      panel("New run", h("div", { class: "fields" }, nameWidget.el), description.el),
      panel(
        "Reward",
        h("div", { class: "row", style: "margin-bottom:12px" }, h("label", { for: "fork-reward-scheme", class: "label", text: "fill from scheme" }), schemeSelect),
        h("div", { class: "fields" }, rewardWidgets.map((w) => w.el)),
      ),
      panel("Training", h("div", { class: "fields" }, [...trainWidgets, ...runWidgets].map((w) => w.el))),
      panel(
        "Evaluation",
        h("p", { class: "muted small", text: "Greedy games played while the new run trains; its evaluations go to its own evaluations.jsonl (the parent's are not copied)." }),
        h("div", { class: "fields" }, evaluationWidgets.map((w) => w.el)),
      ),
      h(
        "details",
        { class: "panel" },
        h("summary", { text: "Locked settings (taken from the parent)" }),
        h("p", { class: "muted small", text: "These must stay as they are: the parent's weights only fit this network, these game rules and this observation." }),
        h("div", { class: "fields" }, lockedShown.map((w) => w.el)),
      ),
      note,
      h("div", { class: "row dialog-actions" }, startButton, cancelButton),
      formError,
    ),
  );

  async function start() {
    message(formError, "", "");
    nameWidget.setError("");
    const name = nameWidget.get();
    const { tables, errors } = readConfig();
    if (!name.error && taken.has(name.value)) name.error = name.value === parent.name ? "Must differ from the parent's name." : "A run with this name already exists.";
    if (name.error || errors) {
      nameWidget.setError(name.error);
      message(formError, "error", "Some fields need attention (marked in red).");
      return;
    }
    startButton.disabled = true;
    startButton.textContent = "Starting…";
    try {
      const body = { name: name.value, checkpoint: checkpoint.name, config: tables, description: description.get() };
      const result = await post(gameUrl(`/api/runs/${encodeURIComponent(parent.name)}/fork`, game), body);
      dialog.close();
      onStarted(result.name);
    } catch (error) {
      const text = error.message;
      if (error instanceof ApiError && (error.status === 409 || /name/.test(text))) nameWidget.setError(text);
      else attachServerError([...editable.map(([, w]) => w), ...lockedWidgets], text);
      message(formError, "error", text);
    } finally {
      startButton.disabled = false;
      startButton.textContent = "Start new run";
    }
  }

  startButton.addEventListener("click", start);
  cancelButton.addEventListener("click", () => dialog.close());
  dialog.addEventListener("close", () => dialog.remove());
  document.body.append(dialog);
  updateNote();
  dialog.showModal();
}
