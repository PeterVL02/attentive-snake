// "Run evaluation now" for one run: pick a checkpoint, the number of games and the seed, let the server play
// them in the background, show the progress, the result, and whether it reproduces a recorded evaluation.

import { get, post } from "./api.js";
import { formatNumber, h, message, replace } from "./dom.js";
import { metricDefinitions } from "./evaluation_metrics.js";
import { evaluationDetails } from "./evaluation_view.js";
import { gameUrl } from "./game.js";
import { setCheckpointOptions } from "./run_options.js";

const POLL_MS = 1000;

/** [kind, text] for the comparison with a recorded evaluation of the same checkpoint and settings. */
function reproductionMessage(reproduces) {
  if (!reproduces) return ["info", "No recorded evaluation of this checkpoint with these settings to compare with."];
  const when = `the evaluation recorded ${reproduces.source === "training" ? "while training" : "on request"} (${new Date(reproduces.timestamp).toLocaleString()})`;
  if (reproduces.identical) return ["info", `Identical to ${when}: the numbers reproduce exactly.`];
  const device = reproduces.same_device ? "" : " It ran on another device (CPU vs GPU), which can change results slightly.";
  return ["error", `Different from ${when}.${device}`];
}

/** The latest recorded evaluation of `checkpoint` in `evaluations`, preferring one made while training. */
function recordedFor(evaluations, checkpoint) {
  const matching = evaluations.filter((e) => e.checkpoint === checkpoint);
  return matching.filter((e) => e.source === "training").at(-1) ?? matching.at(-1) ?? null;
}

function numberInput(id, min) {
  return h("input", { type: "number", id, min, step: "1", inputmode: "numeric" });
}

/** The control for run `name` of `game`; `update(detail)` refreshes it, `onSaved()` runs after a saved evaluation finished. */
export function evaluationRunner(game, name, { onSaved }) {
  let detail = null;
  let defaultsLoaded = false;
  let timer = null;
  let running = false; // an evaluation of this run is going on on the server
  let reported = null; // finished_at of the job whose result was last handed to onSaved
  const url = gameUrl(`/api/runs/${encodeURIComponent(name)}/evaluate`, game);
  const els = {
    checkpoint: h("select", { id: "eval-checkpoint" }),
    games: numberInput("eval-games", "1"),
    seed: numberInput("eval-seed"),
    save: h("input", { type: "checkbox", id: "eval-save" }),
    run: h("button", { type: "button", class: "primary", text: "Run evaluation now" }),
    hint: h("p", { class: "muted small" }),
    progressBar: h("div", {}),
    progress: null,
    status: h("div", { class: "message", role: "status" }),
    result: h("div", {}),
  };
  els.progress = h("div", { class: "progress", hidden: true }, els.progressBar);

  function prefillFromRecord() {
    const recorded = detail && recordedFor(detail.evaluations, els.checkpoint.value);
    if (!recorded) {
      els.hint.textContent = "This checkpoint has no recorded evaluation; any settings start a new one.";
      return;
    }
    els.games.value = String(recorded.settings.games);
    els.seed.value = String(recorded.settings.seed);
    els.hint.textContent = `Filled in from its recorded evaluation (${formatNumber(recorded.settings.games)} games, seed ${recorded.settings.seed}): running it again should give exactly the same numbers.`;
  }

  function readInputs() {
    const games = Number(els.games.value);
    const seed = Number(els.seed.value);
    if (!Number.isInteger(games) || games < 1) return { error: "The number of games must be a whole number of at least 1." };
    if (els.seed.value.trim() === "" || !Number.isInteger(seed)) return { error: "The seed must be a whole number." };
    return { body: { checkpoint: els.checkpoint.value || null, games, seed, save: els.save.checked } };
  }

  function canRun() {
    return !running && detail !== null && detail.checkpoints.length > 0;
  }

  /** Show a job status; `fresh`: it may be a result this panel has not reported yet (false for the first look). */
  function show(status, fresh = true) {
    running = status.state === "running";
    els.run.disabled = !canRun();
    els.progress.hidden = !running;
    if (running) {
      const { done, total } = status.progress;
      els.progressBar.style.width = `${total ? (100 * done) / total : 0}%`;
      message(els.status, "info", `Playing ${status.request.checkpoint}: ${formatNumber(done)} of ${formatNumber(total)} games finished…`);
      return;
    }
    if (status.state === "failed") {
      message(els.status, "error", `The evaluation failed: ${status.error}`);
      return;
    }
    if (status.state !== "finished") return;
    const [kind, text] = reproductionMessage(status.result.reproduces);
    message(els.status, kind, `${text}${status.result.saved ? " Saved to the run's evaluations." : ""}`);
    replace(els.result, evaluationDetails(metricDefinitions(), status.result.evaluation));
    if (status.result.saved && reported !== status.finished_at) {
      reported = status.finished_at;
      if (fresh) onSaved();
    }
  }

  async function poll(fresh = true) {
    clearTimeout(timer);
    if (!el.isConnected) return; // the panel was replaced (another run or game): stop polling
    try {
      const status = await get(url);
      show(status, fresh);
      if (status.state === "running") timer = setTimeout(poll, POLL_MS);
    } catch (error) {
      message(els.status, "error", error.message);
    }
  }

  async function run() {
    const { body, error } = readInputs();
    if (error) {
      message(els.status, "error", error);
      return;
    }
    els.run.disabled = true;
    replace(els.result);
    try {
      show(await post(url, body));
      reported = null;
      timer = setTimeout(poll, POLL_MS);
    } catch (failure) {
      message(els.status, "error", failure.message);
      els.run.disabled = !canRun();
    }
  }

  els.run.addEventListener("click", run);
  els.checkpoint.addEventListener("change", prefillFromRecord);

  const field = (id, label, input, help) => h("div", { class: "field" }, h("label", { for: id, text: label }), input, h("div", { class: "description", text: help }));
  const el = h(
    "div",
    { class: "evaluation-runner continue-option" },
    h("h4", { text: "Run evaluation now" }),
    h("p", { class: "muted small", text: "Plays complete games with the chosen checkpoint, always taking its best action (the most likely one, or the highest Q-value; no exploration). The same checkpoint, number of games and seed always give the same numbers." }),
    h(
      "div",
      { class: "fields" },
      field("eval-checkpoint", "checkpoint", els.checkpoint, "Which saved weights to evaluate."),
      field("eval-games", "games", els.games, "How many complete games to play."),
      field("eval-seed", "seed", els.seed, "Seed of the games; the same seed plays the same games."),
    ),
    els.hint,
    h("label", { class: "row inline-check", for: "eval-save" }, els.save, h("span", { class: "label", text: "save to the run's evaluations" })),
    h("div", { class: "row", style: "margin-top:10px" }, els.run),
    h("div", { style: "margin-top:10px" }, els.progress),
    h("div", { style: "margin-top:10px" }, els.status),
    els.result,
  );

  function update(d) {
    const first = detail === null;
    detail = d;
    const previous = els.checkpoint.value;
    setCheckpointOptions(els.checkpoint, d.checkpoints);
    els.checkpoint.disabled = d.checkpoints.length === 0;
    if (!defaultsLoaded) {
      els.games.value = String(d.config.evaluation.games);
      els.seed.value = String(d.config.evaluation.seed);
      defaultsLoaded = true;
    }
    if (first || previous !== els.checkpoint.value) prefillFromRecord();
    if (d.checkpoints.length === 0) els.hint.textContent = "No checkpoint to evaluate yet.";
    els.run.disabled = !canRun();
    if (first) poll(false); // pick up an evaluation started earlier (it runs on the server)
  }

  return { el, update };
}
