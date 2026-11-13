// Tab switching and start-up. Each tab module exports init(panel), optional show(params) and hide().

import { onNavigate } from "./bus.js";
import * as newRun from "./tabs/new_run.js";
import * as runs from "./tabs/runs.js";
import * as compare from "./tabs/compare.js";
import * as architecture from "./tabs/architecture.js";
import * as watch from "./tabs/watch.js";
import * as rewards from "./tabs/rewards.js";
import * as cheatsheet from "./tabs/cheatsheet.js";

const TABS = {
  new: { label: "New training", module: newRun },
  runs: { label: "Runs", module: runs },
  compare: { label: "Compare", module: compare },
  architecture: { label: "Architecture", module: architecture },
  watch: { label: "Watch & attention", module: watch },
  rewards: { label: "Rewards", module: rewards },
  cli: { label: "CLI cheat sheet", module: cheatsheet },
};

const nav = document.getElementById("tabs");
const main = document.getElementById("main");
const ready = {};
let active = null;

for (const [id, tab] of Object.entries(TABS)) {
  const button = document.createElement("button");
  button.type = "button";
  button.setAttribute("role", "tab");
  button.id = `tab-${id}`;
  button.textContent = tab.label;
  button.setAttribute("aria-controls", `panel-${id}`);
  button.addEventListener("click", () => select(id));
  nav.append(button);
  const panel = document.createElement("section");
  panel.id = `panel-${id}`;
  panel.setAttribute("role", "tabpanel");
  panel.hidden = true;
  panel.setAttribute("aria-labelledby", button.id);
  main.append(panel);
  ready[id] = Promise.resolve(tab.module.init(panel));
}

// Arrow keys move between tabs (WAI-ARIA tabs pattern).
nav.addEventListener("keydown", (event) => {
  const ids = Object.keys(TABS);
  const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
  if (!step) return;
  const next = ids[(ids.indexOf(active) + step + ids.length) % ids.length];
  select(next);
  document.getElementById(`tab-${next}`).focus();
});

async function select(id, params = {}) {
  if (!TABS[id]) id = "new";
  if (active && active !== id) TABS[active].module.hide?.();
  active = id;
  for (const other of Object.keys(TABS)) {
    const selected = other === id;
    const button = document.getElementById(`tab-${other}`);
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    document.getElementById(`panel-${other}`).hidden = !selected;
  }
  if (location.hash !== `#${id}`) history.replaceState(null, "", `#${id}`);
  await ready[id];
  TABS[id].module.show?.(params);
}

onNavigate((tab, params) => select(tab, params));
window.addEventListener("hashchange", () => select(location.hash.slice(1)));
select(location.hash.slice(1) || "new");
