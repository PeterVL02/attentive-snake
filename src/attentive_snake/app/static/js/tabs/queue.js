// Tab: the training queue. Its own tab because the queue is one for all games, while the Runs tab shows one game.

import { h, replace } from "../dom.js";
import { queuePanel } from "../queue_panel.js";

const FAST_POLL_MS = 2000; // something waits or trains from the queue
const SLOW_POLL_MS = 8000;

let panel;
let visible = false;
let timer = null;

async function refresh() {
  await panel.refresh();
  clearTimeout(timer);
  if (visible) timer = setTimeout(refresh, panel.isActive() ? FAST_POLL_MS : SLOW_POLL_MS);
}

export function init(element) {
  panel = queuePanel();
  replace(
    element,
    h("h2", { text: "Training queue" }),
    h("p", { class: "lead", text: "Runs waiting to start, oldest first. Add one with “Add to queue” in New training or in the dialog for a new run from a checkpoint." }),
    panel.el,
  );
}

export function show() {
  visible = true;
  refresh();
}

export function hide() {
  visible = false;
  clearTimeout(timer);
}
