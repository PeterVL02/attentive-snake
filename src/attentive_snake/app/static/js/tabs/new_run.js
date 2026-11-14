// Tab 1: configure and start a new training run.

import { ApiError, get, post } from "../api.js";
import { navigate, provide } from "../bus.js";
import { debounce, formatSteps, h, message, replace } from "../dom.js";
import { attachServerError, fieldControl, readFields } from "../fields.js";
import { rewardTable } from "../reward_table.js";
import { loadSchema } from "../schema.js";

const COMMON_TRAIN = [
  ["train", "total_env_steps"],
  ["env", "num_envs"],
  ["train", "learning_rate"],
  ["train", "anneal_lr"],
  ["train", "entropy_coef"],
  ["train", "seed"],
  ["train", "device"],
];
const HIDDEN = new Set(["train.checkpoint_dir", "model.kind"]); // set by the app / by the kind dropdown
const STATE_FIELDS = ["history_length", "history_actions"];
const DEFAULT_SCHEME = "default"; // configs/rewards/default.toml: the single source of the default reward values
const BUILT_IN = ""; // option value used only when no scheme can be listed: the code defaults (same values)

let schema;
let defaults;
let root;
const widgets = {}; // "section.name" -> widget
let kindSelect;
let kindDescription;
let modelFields;
let explain;
let nameWidget;
let nameEdited = false;
let rewardSelect;
let rewardPreview;
let rewardValues = null;
let formError;
let startButton;
let summary;

function sectionWidgets(section, names) {
  return names.map((name) => {
    const spec = schema.sections[section].find((f) => f.name === name);
    const widget = fieldControl(spec, defaults[section][name]);
    widgets[`${section}.${name}`] = widget;
    widget.onChange(updateSummary);
    return widget;
  });
}

function allWidgets() {
  return Object.values(widgets);
}

function kindInfo() {
  return schema.model_kinds[kindSelect.value];
}

function explainState() {
  const info = kindInfo();
  const n = widgets["env.history_length"].get().value;
  const actions = widgets["env.history_actions"].get().value;
  if (!Number.isInteger(n) || n < 1) {
    explain.textContent = "Enter a history length of at least 1.";
    return;
  }
  const actionNote = actions ? " Each frame also carries which action produced it." : "";
  let text;
  if (n === 1) {
    text = `The model sees only the current board.${actionNote}`;
    if (info.uses_history) text += " This model is built to look across time; with one frame it has nothing to compare.";
  } else if (info.uses_history) {
    text = `The model sees the last ${n} boards as a sequence (oldest to newest) and can attend across time.${actionNote}`;
  } else {
    text = `The last ${n} boards are stacked as extra input channels of each cell; this model has no notion of their order in time.${actionNote} The recommended setting for this model is a single frame.`;
  }
  explain.textContent = text;
}

function showKindFields() {
  const info = kindInfo();
  kindDescription.textContent = info.description;
  for (const spec of schema.sections.model) {
    if (spec.name === "kind") continue;
    widgets[`model.${spec.name}`].el.hidden = !info.fields.includes(spec.name);
  }
}

function onKindChange() {
  showKindFields();
  const recommended = kindInfo().recommended_env;
  widgets["env.history_length"].set(recommended.history_length);
  widgets["env.history_actions"].set(recommended.history_actions);
  explainState();
  updateSummary();
}

/** Keep the model panel as tall as the kind with the most fields, so switching kinds does not shift the page. */
function reserveModelHeight() {
  if (!modelFields.offsetParent) return; // not visible: nothing to measure
  const current = kindSelect.value;
  modelFields.style.minHeight = "";
  let tallest = 0;
  for (const kind of Object.keys(schema.model_kinds)) {
    kindSelect.value = kind;
    showKindFields();
    tallest = Math.max(tallest, modelFields.offsetHeight);
  }
  kindSelect.value = current;
  showKindFields();
  modelFields.style.minHeight = `${tallest}px`;
}

function suggestedName() {
  const w = widgets["game.width"].get().value;
  const ht = widgets["game.height"].get().value;
  return `${kindSelect.value}-${w}x${ht}`;
}

function updateSummary() {
  if (!nameEdited) nameWidget.set(suggestedName());
  const steps = widgets["train.total_env_steps"].get().value;
  const w = widgets["game.width"].get().value;
  const ht = widgets["game.height"].get().value;
  summary.textContent = `${kindSelect.value} on a ${w}×${ht} board for ${formatSteps(steps)} env steps.`;
}

/** The form as experiment tables: {tables, errors}; field errors are shown inline. */
function currentTables() {
  const tables = { game: {}, model: { kind: kindSelect.value }, train: {}, env: {} };
  let errors = 0;
  for (const [key, widget] of Object.entries(widgets)) {
    const [section, name] = key.split(".");
    const result = readFields([widget]);
    errors += result.errors;
    if (!result.errors) tables[section][name] = result.values[name];
  }
  tables.reward = rewardValues ? { ...rewardValues } : { ...defaults.reward };
  return { tables, errors };
}

async function loadReward() {
  const name = rewardSelect.value;
  try {
    rewardValues = name === BUILT_IN ? { ...defaults.reward } : await get(`/api/rewards/${encodeURIComponent(name)}`);
    const rows = await post("/api/rewards/preview", { reward: rewardValues });
    replace(rewardPreview, rewardTable(rows));
  } catch (error) {
    replace(rewardPreview, h("div", { class: "message error", text: error.message }));
  }
}

async function fillRewardSchemes() {
  const previous = rewardSelect.value;
  let schemes = [];
  try {
    schemes = (await get("/api/rewards")).schemes;
  } catch {
    /* the default is still usable */
  }
  replace(
    rewardSelect,
    schemes.length === 0 ? h("option", { value: BUILT_IN, text: "built-in default values" }) : null,
    schemes.map((s) => h("option", { value: s, text: s })),
  );
  if (schemes.includes(previous)) rewardSelect.value = previous;
  else if (schemes.includes(DEFAULT_SCHEME)) rewardSelect.value = DEFAULT_SCHEME;
  await loadReward();
}

async function start() {
  message(formError, "", "");
  nameWidget.setError("");
  const { tables, errors } = currentTables();
  const nameResult = nameWidget.get();
  if (nameResult.error || errors) {
    nameWidget.setError(nameResult.error);
    message(formError, "error", "Some fields need attention (marked in red).");
    return;
  }
  startButton.disabled = true;
  startButton.textContent = "Starting…";
  try {
    const result = await post("/api/runs", { name: nameResult.value, config: tables });
    nameEdited = false;
    navigate("runs", { run: result.name });
  } catch (error) {
    const text = error.message;
    if (error instanceof ApiError && (error.status === 409 || /name/.test(text))) nameWidget.setError(text);
    else attachServerError(allWidgets(), text);
    message(formError, "error", text);
  } finally {
    startButton.disabled = false;
    startButton.textContent = "Start training";
  }
}

function panel(title, ...children) {
  return h("div", { class: "panel" }, h("h3", { text: title }), ...children);
}

function build() {
  const nameSpec = { name: "run name", type: "str", nullable: false, description: "Directory under runs/: letters, digits, '.', '_' and '-'." };
  nameWidget = fieldControl(nameSpec, "");
  nameWidget.onChange(() => (nameEdited = true));

  const kindSpec = schema.sections.model.find((f) => f.name === "kind");
  kindSelect = h("select", { id: "kind-select" }, kindSpec.choices.map((k) => h("option", { value: k, text: k })));
  kindSelect.value = defaults.model.kind;
  kindSelect.addEventListener("change", onKindChange);
  kindDescription = h("p", { class: "kind-description" });
  const modelNames = schema.sections.model.map((f) => f.name).filter((n) => n !== "kind");
  modelFields = h("div", { class: "fields" }, sectionWidgets("model", modelNames).map((w) => w.el));

  const stateWidgets = sectionWidgets("env", STATE_FIELDS);
  explain = h("div", { class: "explain", "aria-live": "polite" });
  stateWidgets.forEach((w) => w.onChange(explainState));

  const gameWidgets = sectionWidgets("game", schema.sections.game.map((f) => f.name));

  const common = COMMON_TRAIN.flatMap(([section, name]) => sectionWidgets(section, [name]));
  const commonKeys = new Set(COMMON_TRAIN.map(([s, n]) => `${s}.${n}`));
  const advanced = [
    ...sectionWidgets("train", schema.sections.train.map((f) => f.name).filter((n) => !commonKeys.has(`train.${n}`) && !HIDDEN.has(`train.${n}`))),
    ...sectionWidgets("env", schema.sections.env.map((f) => f.name).filter((n) => !commonKeys.has(`env.${n}`) && !STATE_FIELDS.includes(n))),
  ];

  rewardSelect = h("select", { id: "reward-scheme" });
  rewardSelect.addEventListener("change", loadReward);
  rewardPreview = h("div", {}, h("div", { class: "muted small", text: "Loading…" }));

  formError = h("div", { class: "message", role: "alert" });
  startButton = h("button", { class: "primary", type: "button", text: "Start training", onclick: start });
  summary = h("p", { class: "muted small" });

  const left = h(
    "div",
    {},
    panel("Run", h("div", { class: "fields" }, nameWidget.el)),
    panel(
      "Model",
      h("div", { class: "field" }, h("label", { for: "kind-select", text: "kind" }), kindSelect),
      kindDescription,
      modelFields,
    ),
    panel(
      "State representation",
      h("p", { class: "muted small", text: "What one observation contains. Set to the recommendation when you change the model; you can still edit it." }),
      h("div", { class: "fields" }, stateWidgets.map((w) => w.el)),
      explain,
    ),
    panel("Game", h("div", { class: "fields" }, gameWidgets.map((w) => w.el))),
    panel(
      "Training",
      h("div", { class: "fields" }, common.map((w) => w.el)),
      h("details", { class: "advanced" }, h("summary", { text: "Advanced" }), h("div", { class: "fields" }, advanced.map((w) => w.el))),
    ),
  );
  const right = h(
    "div",
    { class: "sticky" },
    panel(
      "Reward scheme",
      h("div", { class: "row" }, rewardSelect, h("button", { type: "button", class: "link", text: "Edit schemes", onclick: () => navigate("rewards") })),
      h("p", { class: "muted small", text: "What this scheme pays:" }),
      rewardPreview,
    ),
    panel(
      "Start",
      summary,
      h("div", { class: "row" }, startButton, h("button", { type: "button", text: "Preview architecture", onclick: () => navigate("architecture", { source: "form" }) })),
      h("div", { style: "margin-top:12px" }, formError),
    ),
  );
  replace(
    root,
    h("h2", { text: "New training" }),
    h("p", { class: "lead", text: "Pick a model, adjust what you need and start. Every field shows its default; the run's full config is saved to runs/<name>/config.toml." }),
    h("div", { class: "columns" }, left, right),
  );
  onKindChange();
  // Keep the loaded defaults for history even if they differ from the kind's recommendation.
  widgets["env.history_length"].set(defaults.env.history_length);
  widgets["env.history_actions"].set(defaults.env.history_actions);
  explainState();
  window.addEventListener("resize", debounce(reserveModelHeight, 200));
}

export async function init(element) {
  root = element;
  replace(root, h("div", { class: "empty", text: "Loading the config schema…" }));
  try {
    ({ schema, defaults } = await loadSchema());
  } catch (error) {
    replace(root, h("div", { class: "message error", text: `Could not load the schema: ${error.message}` }));
    return;
  }
  build();
  provide("formConfig", currentTables);
  await fillRewardSchemes();
}

export function show() {
  if (!kindSelect) return;
  requestAnimationFrame(reserveModelHeight);
  fillRewardSchemes();
}
