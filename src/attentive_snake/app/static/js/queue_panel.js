// The training queue (one for all games): the runs waiting to start, in order, with "Run now" and "Remove",
// and how many of them may train at once. Entries are added in New training and in the new-run-from-a-checkpoint dialog.

import { del, get, post, put } from "./api.js";
import { formatSteps, h, message, replace } from "./dom.js";
import { listGames } from "./game.js";

const STATE_LABELS = { queued: "waiting", starting: "starting", failed: "failed" };

function gameTitle(name) {
  return listGames().find((g) => g.name === name)?.title ?? name;
}

/** What the entry will do: a new run, a new run from a parent's checkpoint, or a resume. */
function kindText(entry) {
  if (entry.kind === "fork") return `new run from ${entry.parent} (${entry.checkpoint ?? "its latest checkpoint"})`;
  if (entry.kind === "resume") return "resume";
  return "new run";
}

function settingsText(entry) {
  const steps = entry.total_env_steps ? `${formatSteps(entry.total_env_steps)} steps` : null;
  return [entry.model_kind, entry.algorithm, steps].filter(Boolean).join(" · ") || "–";
}

/**
 * The panel: {el, refresh(), isActive()}. `refresh()` reloads it from /api/queue; `isActive()` says whether something
 * waits or trains (worth polling fast). It refreshes itself after each of its own actions.
 */
export function queuePanel() {
  let active = false;
  let loadFailed = false;
  const parallel = h("input", { type: "number", id: "queue-max-parallel", min: "0", max: "64", step: "1", style: "width:70px" });
  const running = h("div", { class: "muted small" });
  const list = h("div", { style: "margin-top:10px" });
  const note = h("div", { class: "message", role: "alert" });

  /** Run `work` (a request that changes the queue); its error is shown, and `after(result)` runs if it worked. */
  async function act(button, work, after) {
    button.disabled = true;
    message(note, "", "");
    try {
      after(await work());
    } catch (error) {
      message(note, "error", error.message);
    } finally {
      button.disabled = false;
      refresh(); // also after an error: a failed "Run now" leaves the entry marked as failed
    }
  }

  function runNow(entry, button) {
    act(button, () => post(`/api/queue/${encodeURIComponent(entry.id)}/run`), (started) =>
      message(note, "info", `Started ${started.name} (${gameTitle(started.game)}); it trains like any other run, see the Runs tab.`),
    );
  }

  function remove(entry, button) {
    act(button, () => del(`/api/queue/${encodeURIComponent(entry.id)}`), () => message(note, "info", `Removed ${entry.run} from the queue.`));
  }

  function setParallel() {
    const value = Number(parallel.value.trim());
    if (parallel.value.trim() === "" || !Number.isInteger(value) || value < 0) {
      message(note, "error", "Runs at once must be a whole number, 0 or more (0 pauses the queue).");
      return;
    }
    act(parallel, () => put("/api/queue/settings", { max_parallel: value }), () => message(note, "", ""));
  }

  function row(entry, index) {
    const runButton = h("button", { type: "button", text: entry.state === "failed" ? "Try again" : "Run now" });
    const removeButton = h("button", { type: "button", class: "danger", text: "Remove" });
    runButton.addEventListener("click", () => runNow(entry, runButton));
    removeButton.addEventListener("click", () => remove(entry, removeButton));
    const busy = entry.state === "starting";
    runButton.disabled = busy;
    removeButton.disabled = busy;
    return h(
      "tr",
      {},
      h("td", { class: "num", text: String(index + 1) }),
      h("td", {}, h("div", { class: "mono", text: entry.run }), entry.description ? h("div", { class: "muted small", text: entry.description }) : null),
      h("td", { text: gameTitle(entry.game) }),
      h("td", { text: kindText(entry) }),
      h("td", { text: settingsText(entry) }),
      h("td", { class: "small", text: new Date(entry.queued_at * 1000).toLocaleString() }),
      h("td", {}, h("span", { class: `badge ${entry.state}`, text: STATE_LABELS[entry.state] ?? entry.state }), entry.error ? h("div", { class: "small", text: `Could not start: ${entry.error}` }) : null),
      h("td", {}, h("div", { class: "row" }, runButton, removeButton)),
    );
  }

  function render(data) {
    active = data.running.length > 0 || data.entries.some((entry) => entry.state !== "failed");
    if (document.activeElement !== parallel) parallel.value = String(data.max_parallel);
    const paused = data.max_parallel === 0 ? " The queue is paused (runs at once is 0): entries only start with Run now." : "";
    running.textContent =
      (data.running.length > 0
        ? `Training from the queue now: ${data.running.map((job) => `${job.run} (${gameTitle(job.game)})`).join(", ")}.`
        : "Nothing from the queue is training.") + paused;
    if (data.entries.length === 0) {
      replace(list, h("p", { class: "muted small", text: "The queue is empty. “Add to queue” in New training, or in the dialog for a new run from a checkpoint, puts a run here; it starts when the ones before it have ended." }));
      return;
    }
    replace(
      list,
      h(
        "table",
        {},
        h("thead", {}, h("tr", {}, ["#", "run", "game", "what", "settings", "queued", "state", ""].map((text) => h("th", { text })))),
        h("tbody", {}, data.entries.map(row)),
      ),
    );
  }

  async function refresh() {
    try {
      render(await get("/api/queue"));
      if (loadFailed) message(note, "", "");
      loadFailed = false;
    } catch (error) {
      loadFailed = true;
      message(note, "error", `Could not load the training queue: ${error.message}`);
    }
  }

  parallel.addEventListener("change", setParallel);
  const el = h(
    "div",
    { class: "panel" },
    h(
      "div",
      { class: "row spread" },
      h("h3", { text: "Training queue (all games)", style: "margin:0" }),
      h("div", { class: "row" }, h("label", { for: "queue-max-parallel", class: "label", text: "runs at once" }), parallel),
    ),
    h("p", { class: "muted small", text: "Queued runs start by themselves, oldest first, whenever fewer than “runs at once” runs started by the queue are training. Runs you start yourself, or with Run now, do not hold the queue back." }),
    running,
    list,
    h("div", { style: "margin-top:10px" }, note),
  );
  return { el, refresh, isActive: () => active };
}
