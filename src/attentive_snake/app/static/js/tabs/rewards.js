// Tab 5: write and edit the selected game's reward schemes (configs/<game>/rewards/*.toml), with a
// live preview of what they pay for games that have one.

import { del, get, post, put } from "../api.js";
import { debounce, h, message, replace } from "../dom.js";
import { attachServerError, fieldControl, readFields } from "../fields.js";
import { currentGame, gameUrl } from "../game.js";
import { rewardTable } from "../reward_table.js";
import { loadSchema } from "../schema.js";

let root;
let game = null; // info of the game the editor is built for (null while loading)
let defaults;
let els = {};
let widgets = [];
let current = null; // name of the scheme being edited, null for a new one
let protectedNames = [];
let dirty = false;

function url(path = "") {
  return gameUrl(`/api/rewards${path}`, game.name);
}

function schemeFile(name) {
  return `configs/${game.name}/rewards/${name}.toml`;
}

function setDirty(value) {
  dirty = value;
  els.dirty.hidden = !dirty;
}

async function refreshPreview() {
  if (!game.has_reward_table) return;
  const forGame = game;
  const { values, errors } = readFields(widgets);
  if (errors) {
    replace(els.preview, h("div", { class: "muted small", text: "Fix the marked fields to see the preview." }));
    return;
  }
  const score = Number(els.exampleScore.value);
  try {
    const rows = await post(url("/preview"), { reward: values, example_score: Number.isInteger(score) && score >= 0 ? score : 5 });
    if (forGame === game) replace(els.preview, rewardTable(rows));
  } catch (error) {
    if (forGame !== game) return;
    attachServerError(widgets, error.message);
    replace(els.preview, h("div", { class: "message error", text: error.message }));
  }
}

const schedulePreview = debounce(() => game && refreshPreview(), 250);

function renderList(names) {
  replace(
    els.list,
    names.length === 0 ? h("li", { class: "empty", text: "No saved schemes." }) : null,
    names.map((name) =>
      h("li", {}, h("button", { type: "button", "aria-current": String(name === current), onclick: () => open(name) }, h("span", { class: "name", text: name }), protectedNames.includes(name) ? h("span", { class: "muted small", text: "built-in" }) : null)),
    ),
  );
}

async function refreshList() {
  const forGame = game;
  const data = await get(url());
  if (forGame !== game) return [];
  protectedNames = data.protected;
  renderList(data.schemes);
  return data.schemes;
}

function confirmDiscard() {
  return !dirty || confirm("Discard the unsaved changes to this scheme?");
}

function showEditor(name, values) {
  current = name;
  widgets.forEach((w) => {
    w.set(values[w.spec.name]);
    w.setError("");
  });
  els.title.textContent = name ?? "New scheme";
  els.save.disabled = name === null;
  els.delete.disabled = name === null || protectedNames.includes(name);
  els.delete.title = protectedNames.includes(name) ? "Built-in schemes cannot be deleted." : "";
  message(els.status, "", "");
  setDirty(false);
  refreshPreview();
}

async function open(name) {
  if (!confirmDiscard()) return;
  const forGame = game;
  try {
    const values = await get(url(`/${encodeURIComponent(name)}`));
    if (forGame !== game) return;
    showEditor(name, values);
    await refreshList();
  } catch (error) {
    if (forGame === game) message(els.status, "error", error.message);
  }
}

async function saveAs(name) {
  const { values, errors } = readFields(widgets);
  if (errors) {
    message(els.status, "error", "Some fields need attention.");
    return;
  }
  if (!name) {
    message(els.status, "error", "Enter a name for the scheme.");
    return;
  }
  const schemes = await refreshList().catch(() => []);
  if (name !== current && schemes.includes(name) && !confirm(`Overwrite the existing scheme “${name}”?`)) return;
  if (protectedNames.includes(name) && !confirm(`“${name}” is the built-in scheme used as a starting point. Overwrite it?`)) return;
  try {
    await put(url(`/${encodeURIComponent(name)}`), values);
    current = name;
    setDirty(false);
    await refreshList();
    showEditor(name, values);
    message(els.status, "info", `Saved ${schemeFile(name)}.`);
    els.newName.value = "";
  } catch (error) {
    attachServerError(widgets, error.message);
    message(els.status, "error", error.message);
  }
}

async function remove() {
  if (!current || !confirm(`Delete the scheme “${current}”? This removes ${schemeFile(current)}.`)) return;
  try {
    await del(url(`/${encodeURIComponent(current)}`));
    const name = current;
    setDirty(false);
    const schemes = await refreshList();
    if (schemes.length) await open(schemes[0]);
    message(els.status, "info", `Deleted ${name}.`);
  } catch (error) {
    message(els.status, "error", error.message);
  }
}

function previewPanel() {
  if (!game.has_reward_table) {
    return h(
      "div",
      { class: "panel" },
      h("h3", { text: "What this scheme pays" }),
      h("p", { class: "muted small", text: `${game.title} has no preview of what a scheme pays; the field descriptions say what each term rewards.` }),
    );
  }
  return h(
    "div",
    { class: "panel" },
    h("div", { class: "row spread" }, h("h3", { text: "What this scheme pays", style: "margin:0" }), h("label", { class: "row", style: "gap:6px" }, h("span", { class: "small muted", text: "example score" }), els.exampleScore)),
    h("p", { class: "muted small", text: "One step of each kind and the reward it would get, split into the terms that produce it." }),
    els.preview,
  );
}

function build(schema) {
  current = null;
  dirty = false;
  widgets = schema.sections.reward.map((spec) => fieldControl(spec, defaults.reward[spec.name]));
  widgets.forEach((w) =>
    w.onChange(() => {
      setDirty(true);
      schedulePreview();
    }),
  );
  els = {
    list: h("ul", { class: "run-list" }),
    title: h("h3", { style: "margin:0" }),
    dirty: h("span", { class: "badge", text: "unsaved changes", hidden: true }),
    status: h("div", { class: "message", role: "status" }),
    preview: h("div", {}),
    exampleScore: h("input", { type: "number", id: "example-score", min: "0", step: "1", value: "5", style: "width:90px" }),
    newName: h("input", { type: "text", id: "scheme-name", placeholder: "new name", spellcheck: "false", "aria-label": "new scheme name" }),
  };
  els.save = h("button", { type: "button", class: "primary", text: "Save", onclick: () => saveAs(current) });
  els.saveAs = h("button", { type: "button", text: "Save as", onclick: () => saveAs(els.newName.value.trim()) });
  els.delete = h("button", { type: "button", class: "danger", text: "Delete", onclick: remove });
  els.exampleScore.addEventListener("input", schedulePreview);
  replace(
    root,
    h("h2", { text: `Reward schemes: ${game.title}` }),
    h("p", { class: "lead", text: `A scheme is the [reward] section of an experiment, saved as configs/${game.name}/rewards/<name>.toml and offered in New training. All terms are added together.` }),
    h(
      "div",
      { class: "columns narrow-left" },
      h(
        "div",
        { class: "panel sticky" },
        h("div", { class: "row spread" }, h("h3", { text: "Schemes", style: "margin:0" }), h("button", { type: "button", class: "clear", text: "New", onclick: () => confirmDiscard() && showEditor(null, defaults.reward) })),
        els.list,
      ),
      h(
        "div",
        {},
        h(
          "div",
          { class: "panel" },
          h("div", { class: "row spread", style: "margin-bottom:12px" }, h("div", { class: "row" }, els.title, els.dirty), h("div", { class: "row" }, els.save, els.delete)),
          h("div", { class: "fields" }, widgets.map((w) => w.el)),
          h("div", { class: "row", style: "margin-top:16px" }, els.newName, els.saveAs),
          h("div", { style: "margin-top:10px" }, els.status),
        ),
        previewPanel(),
      ),
    ),
  );
}

/** Load the selected game's reward schema and schemes and build the editor (a newer call wins). */
async function load() {
  const target = currentGame();
  game = null;
  replace(root, h("div", { class: "empty", text: "Loading…" }));
  let loaded;
  try {
    loaded = await loadSchema(target.name);
  } catch (error) {
    if (currentGame() === target) replace(root, h("div", { class: "message error", text: error.message }));
    return;
  }
  if (currentGame() !== target) return;
  game = target;
  defaults = loaded.defaults;
  build(loaded.schema);
  try {
    const schemes = await refreshList();
    if (game !== target) return;
    if (schemes.length) await open(schemes.includes("default") ? "default" : schemes[0]);
    else showEditor(null, defaults.reward);
  } catch (error) {
    if (game === target) message(els.status, "error", error.message);
  }
}

export async function init(element) {
  root = element;
  await load();
}

export function show() {
  if (game) refreshList().catch(() => {});
}

export function reset() {
  load(); // unsaved edits of the previous game's scheme are dropped
}
