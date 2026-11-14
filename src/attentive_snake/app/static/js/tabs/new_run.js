// Tab 1: configure and start a new training run of the selected game.

import { ApiError, get, post } from "../api.js";
import { algorithmPicker } from "../algorithm_picker.js";
import { navigate, provide } from "../bus.js";
import { debounce, formatSteps, h, message, replace } from "../dom.js";
import { attachServerError, descriptionField, fieldControl, readFields } from "../fields.js";
import { currentGame, gameUrl } from "../game.js";
import { modelPicker } from "../model_picker.js";
import { rewardTable } from "../reward_table.js";
import { boardSize, rulesSummary } from "../rules_summary.js";
import { loadAlgorithmDefaults, loadSchema } from "../schema.js";

const COMMON_TRAIN = [
  ["train", "total_env_steps"],
  ["env", "num_envs"],
  ["train", "learning_rate"],
  ["train", "anneal_lr"],
  ["train", "seed"],
  ["train", "device"],
  ["train", "keep_checkpoints"],
];
const HIDDEN = new Set(["train.checkpoint_dir", "model.kind", "train.algorithm"]); // set by the app / by the kind and algorithm dropdowns
const ALGORITHM_SHARED = ["env.num_envs"]; // shared settings outside [train] that an algorithm's defaults may set
const STATE_FIELDS = ["history_length", "history_actions"];
const DEFAULT_SCHEME = "default"; // configs/<game>/rewards/default.toml: the single source of the default reward values
const BUILT_IN = ""; // option value used only when no scheme can be listed: the code defaults (same values)

let root;
let game = null; // info of the game the form is built for (null while loading)
let schema;
let defaults;
let widgets = {}; // "section.name" -> widget
let picker;
let algorithms; // the algorithm dropdown
let algorithmPanels = {}; // algorithm -> its settings panel
let algorithmToken = 0; // a defaults load that finds it changed (newer switch, other game) is dropped
let modelFields;
let explain;
let nameWidget;
let nameEdited = false;
let description;
let rewardSelect;
let rewardPreview;
let rewardValues = null;
let formError;
let startButton;
let summary;

function sectionWidgets(section, names) {
  return names.map((name) => {
    const spec = schema.sections[section].find((f) => f.name === name);
    const widget = fieldControl(spec, defaults[section]?.[name] ?? spec.default);
    widgets[`${section}.${name}`] = widget;
    widget.onChange(updateSummary);
    return widget;
  });
}

function allWidgets() {
  return Object.values(widgets);
}

function kindInfo() {
  return schema.model_kinds[picker.value()];
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
    text = `The model sees only the current observation.${actionNote}`;
    if (info.uses_history) text += " This model is built to look across time; with one frame it has nothing to compare.";
  } else if (info.uses_history) {
    text = `The model sees the last ${n} observations as a sequence (oldest to newest) and can attend across time.${actionNote}`;
  } else {
    text = `The last ${n} observations are stacked as extra input channels; this model has no notion of their order in time.${actionNote} The recommended setting for this model is a single frame.`;
  }
  explain.textContent = text;
}

function showKindFields() {
  const info = kindInfo();
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
  if (!modelFields?.offsetParent) return; // not built or not visible: nothing to measure
  const current = picker.value();
  modelFields.style.minHeight = "";
  let tallest = 0;
  for (const kind of Object.keys(schema.model_kinds)) {
    picker.select.value = kind;
    showKindFields();
    tallest = Math.max(tallest, modelFields.offsetHeight);
  }
  picker.set(current);
  showKindFields();
  modelFields.style.minHeight = `${tallest}px`;
}

function gameRules() {
  const rules = {};
  for (const spec of schema.sections.game) rules[spec.name] = widgets[`game.${spec.name}`].get().value;
  return rules;
}

function suggestedName() {
  const size = boardSize(gameRules());
  const algorithm = algorithms.value() === defaults.train.algorithm ? "" : `-${algorithms.value()}`;
  return size ? `${picker.value()}${algorithm}-${size[0]}x${size[1]}` : `${picker.value()}${algorithm}`;
}

function updateSummary() {
  if (!nameEdited) nameWidget.set(suggestedName());
  const steps = widgets["train.total_env_steps"].get().value;
  const label = schema.algorithms[algorithms.value()].label;
  summary.textContent = `${picker.value()} trained with ${label} on ${game.title} (${rulesSummary(gameRules())}) for ${formatSteps(steps)} env steps.`;
}

function showAlgorithmPanel() {
  for (const [name, panelEl] of Object.entries(algorithmPanels)) panelEl.hidden = name !== algorithms.value();
}

/** Show the chosen algorithm's settings and fill them, [train] and num_envs with the game's defaults for it. */
async function onAlgorithmChange() {
  const token = ++algorithmToken;
  const algorithm = algorithms.value();
  showAlgorithmPanel();
  updateSummary();
  let values;
  try {
    values = algorithm === defaults.train.algorithm ? defaults : await loadAlgorithmDefaults(game.name, algorithm);
  } catch (error) {
    if (token === algorithmToken) message(formError, "error", `Could not load the ${algorithm} defaults: ${error.message}`);
    return;
  }
  if (token !== algorithmToken) return;
  for (const [key, widget] of Object.entries(widgets)) {
    const [section, name] = key.split(".");
    const applies = section === "train" || section === algorithm || ALGORITHM_SHARED.includes(key);
    if (applies && values[section] && name in values[section]) widget.set(values[section][name]);
  }
  updateSummary();
}

/** The form as experiment tables: {tables, errors}; field errors are shown inline. Null while loading. */
function currentTables() {
  if (!game) return null;
  const algorithm = algorithms.value();
  const tables = { game: { name: game.name }, model: { kind: picker.value() }, train: { algorithm }, [algorithm]: {}, env: {}, evaluation: {} };
  let errors = 0;
  for (const [key, widget] of Object.entries(widgets)) {
    const [section, name] = key.split(".");
    if (!(section in tables)) continue; // another algorithm's settings
    const result = readFields([widget]);
    errors += result.errors;
    if (!result.errors) tables[section][name] = result.values[name];
  }
  tables.reward = rewardValues ? { ...rewardValues } : { ...defaults.reward };
  return { tables, errors };
}

async function loadReward() {
  const forGame = game;
  const name = rewardSelect.value;
  try {
    const values = name === BUILT_IN ? { ...defaults.reward } : await get(gameUrl(`/api/rewards/${encodeURIComponent(name)}`, forGame.name));
    if (forGame !== game) return;
    rewardValues = values;
    if (!forGame.has_reward_table) return;
    const rows = await post(gameUrl("/api/rewards/preview", forGame.name), { reward: values });
    if (forGame === game) replace(rewardPreview, rewardTable(rows));
  } catch (error) {
    if (forGame === game) replace(rewardPreview, h("div", { class: "message error", text: error.message }));
  }
}

async function fillRewardSchemes() {
  const forGame = game;
  const previous = rewardSelect.value;
  let schemes = [];
  try {
    schemes = (await get(gameUrl("/api/rewards", forGame.name))).schemes;
  } catch {
    /* the default is still usable */
  }
  if (forGame !== game) return;
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
    const result = await post("/api/runs", { name: nameResult.value, config: tables, description: description.get() });
    nameEdited = false;
    description.set("");
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

function rewardPanel() {
  const body = game.has_reward_table
    ? [h("p", { class: "muted small", text: "What this scheme pays:" }), rewardPreview]
    : [h("p", { class: "muted small", text: `${game.title} has no reward preview; open Rewards to see and edit the scheme's values.` })];
  return panel(
    "Reward scheme",
    h("div", { class: "row" }, rewardSelect, h("button", { type: "button", class: "link", text: "Edit schemes", onclick: () => navigate("rewards") })),
    ...body,
  );
}

function build() {
  widgets = {};
  nameEdited = false;
  rewardValues = null;
  const nameSpec = { name: "run name", type: "str", nullable: false, description: `Directory under runs/${game.name}/: letters, digits, '.', '_' and '-'.` };
  nameWidget = fieldControl(nameSpec, "");
  nameWidget.onChange(() => (nameEdited = true));
  description = descriptionField();

  picker = modelPicker(schema.model_kinds);
  picker.set(defaults.model.kind);
  picker.onChange(onKindChange);
  const modelNames = schema.sections.model.map((f) => f.name).filter((n) => n !== "kind");
  modelFields = h("div", { class: "fields" }, sectionWidgets("model", modelNames).map((w) => w.el));

  const stateWidgets = sectionWidgets("env", STATE_FIELDS);
  explain = h("div", { class: "explain", "aria-live": "polite" });
  stateWidgets.forEach((w) => w.onChange(explainState));

  const gameWidgets = sectionWidgets("game", schema.sections.game.map((f) => f.name));

  algorithms = algorithmPicker(schema.algorithms);
  algorithms.set(defaults.train.algorithm);
  algorithms.onChange(onAlgorithmChange);
  const common = COMMON_TRAIN.flatMap(([section, name]) => sectionWidgets(section, [name]));
  const commonKeys = new Set(COMMON_TRAIN.map(([s, n]) => `${s}.${n}`));
  const advanced = [
    ...sectionWidgets("train", schema.sections.train.map((f) => f.name).filter((n) => !commonKeys.has(`train.${n}`) && !HIDDEN.has(`train.${n}`))),
    ...sectionWidgets("env", schema.sections.env.map((f) => f.name).filter((n) => !commonKeys.has(`env.${n}`) && !STATE_FIELDS.includes(n))),
  ];
  algorithmPanels = Object.fromEntries(
    Object.entries(schema.algorithms).map(([name, info]) => [
      name,
      panel(
        `${info.label} settings`,
        h("p", { class: "muted small", text: `The [${info.section}] section: only used when training with ${info.label}.` }),
        h("div", { class: "fields" }, sectionWidgets(info.section, schema.sections[info.section].map((f) => f.name)).map((w) => w.el)),
      ),
    ]),
  );

  const evaluationWidgets = sectionWidgets("evaluation", schema.sections.evaluation.map((f) => f.name));

  rewardSelect = h("select", { id: "reward-scheme", "aria-label": "reward scheme" });
  rewardSelect.addEventListener("change", loadReward);
  rewardPreview = h("div", {}, h("div", { class: "muted small", text: "Loading…" }));

  formError = h("div", { class: "message", role: "alert" });
  startButton = h("button", { class: "primary", type: "button", text: "Start training", onclick: start });
  summary = h("p", { class: "muted small" });

  const left = h(
    "div",
    {},
    panel("Run", h("div", { class: "fields" }, nameWidget.el), description.el),
    panel("Model", picker.el, modelFields),
    panel(
      "State representation",
      h("p", { class: "muted small", text: "What one observation contains. Set to the recommendation when you change the model; you can still edit it." }),
      h("div", { class: "fields" }, stateWidgets.map((w) => w.el)),
      explain,
    ),
    panel(
      `Game: ${game.title}`,
      gameWidgets.length
        ? h("div", { class: "fields" }, gameWidgets.map((w) => w.el))
        : h("p", { class: "muted small", text: "This game has no settings." }),
    ),
    panel(
      "Training",
      algorithms.el,
      h("p", { class: "muted small", text: "Settings every algorithm shares. Changing the algorithm fills them, num_envs and its own settings with this game's defaults for it." }),
      h("div", { class: "fields" }, common.map((w) => w.el)),
      h("details", { class: "advanced" }, h("summary", { text: "Advanced" }), h("div", { class: "fields" }, advanced.map((w) => w.el))),
    ),
    ...Object.values(algorithmPanels),
    panel(
      "Evaluation",
      h("p", { class: "muted small", text: `Every so often the run plays complete games without exploration (always its best action: the most likely one, or the highest Q-value), saves a checkpoint and records ${game.title}'s statistics in runs/${game.name}/<name>/evaluations.jsonl. The Runs tab charts them and can reproduce them later.` }),
      h("div", { class: "fields" }, evaluationWidgets.map((w) => w.el)),
    ),
  );
  const right = h(
    "div",
    { class: "sticky" },
    rewardPanel(),
    panel(
      "Start",
      summary,
      h("div", { class: "row" }, startButton, h("button", { type: "button", text: "Preview architecture", onclick: () => navigate("architecture", { source: "form" }) })),
      h("div", { style: "margin-top:12px" }, formError),
    ),
  );
  replace(
    root,
    h("h2", { text: `New training: ${game.title}` }),
    h("p", { class: "lead", text: `Pick a model, adjust what you need and start. Every field shows its default; the run's full config is saved to runs/${game.name}/<name>/config.toml.` }),
    h("div", { class: "columns" }, left, right),
  );
  showAlgorithmPanel();
  onKindChange();
  // Keep the loaded defaults for history even if they differ from the kind's recommendation.
  widgets["env.history_length"].set(defaults.env.history_length);
  widgets["env.history_actions"].set(defaults.env.history_actions);
  explainState();
  requestAnimationFrame(reserveModelHeight);
}

/** Load the selected game's schema and defaults and build the form for it (a newer call wins). */
async function load() {
  const target = currentGame();
  game = null;
  modelFields = null;
  algorithmToken += 1;
  replace(root, h("div", { class: "empty", text: `Loading the ${target.title} config schema…` }));
  let loaded;
  try {
    loaded = await loadSchema(target.name);
  } catch (error) {
    if (currentGame() === target) replace(root, h("div", { class: "message error", text: `Could not load the schema: ${error.message}` }));
    return;
  }
  if (currentGame() !== target) return; // another game was selected meanwhile; its load() builds the form
  ({ schema, defaults } = loaded);
  game = target;
  build();
  await fillRewardSchemes();
}

export async function init(element) {
  root = element;
  provide("formConfig", currentTables);
  window.addEventListener("resize", debounce(reserveModelHeight, 200));
  await load();
}

export function show() {
  if (!game) return;
  requestAnimationFrame(reserveModelHeight);
  fillRewardSchemes();
}

export function reset() {
  load(); // eagerly, so the Architecture tab's "New-training form" is the selected game's
}
