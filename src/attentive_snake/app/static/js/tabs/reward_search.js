// Tab: reward searches of the selected game. Start one from a run (reward_search_form.js), see them all and one in full
// (reward_search_view.js). Polls while visible: fast while a search runs.

import { get } from "../api.js";
import { h, message, replace } from "../dom.js";
import { currentGame, gameUrl, isCurrent } from "../game.js";
import { searchForm } from "../reward_search_form.js";
import { searchDetail, searchList } from "../reward_search_view.js";

const FAST_POLL_MS = 3000;
const SLOW_POLL_MS = 15000;

let root;
let form = null;
let listBox;
let detailBox;
let note;
let chosen = null;
let visible = false;
let timer = null;

async function refresh() {
  clearTimeout(timer);
  const game = currentGame().name;
  let running = false;
  try {
    const searches = await get(gameUrl("/api/reward-search", game));
    if (!isCurrent(game)) return;
    running = searches.some((s) => s.status === "running");
    replace(listBox, searchList(searches, chosen, open));
    if (chosen && searches.some((s) => s.name === chosen)) {
      replace(detailBox, searchDetail(await get(`/api/reward-search/${encodeURIComponent(chosen)}`), refresh));
    } else {
      replace(detailBox);
    }
    message(note, "", "");
  } catch (error) {
    message(note, "error", `Could not load the reward searches: ${error.message}`);
  }
  if (visible) timer = setTimeout(refresh, running ? FAST_POLL_MS : SLOW_POLL_MS);
}

function open(name) {
  chosen = name;
  refresh();
}

function build() {
  form = searchForm(currentGame().name, { onStarted: open });
  replace(
    root,
    h("h2", { text: "Reward search" }),
    h("p", {
      class: "lead",
      text: "Let the reward adjust itself. Each generation trains several copies of the best model so far, one on the current reward and the others on changed copies of it; the one that plays best by the game's own score (never by the reward) wins, and its model and reward start the next generation. Member runs go through the training queue and are ordinary runs in the Runs tab.",
    }),
    form.el,
    h("div", { class: "panel" }, h("h3", { text: "Searches" }), listBox, note),
    detailBox,
  );
  form.load();
}

export function init(element) {
  root = element;
  listBox = h("div");
  detailBox = h("div");
  note = h("div", { class: "message", role: "alert" });
}

export function show() {
  visible = true;
  if (form === null) build();
  refresh();
}

export function hide() {
  visible = false;
  clearTimeout(timer);
}

export function reset() {
  form = null;
  chosen = null;
  replace(listBox);
  replace(detailBox);
}
