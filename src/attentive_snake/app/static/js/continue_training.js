// The Runs tab's "Continue training" panel for one run: resume the run itself, or start a new run
// from one of its checkpoints (the fork dialog). Each option says when to use it and why it is disabled.

import { post } from "./api.js";
import { formatNumber, formatSteps, h, message } from "./dom.js";
import { descriptionField } from "./fields.js";
import { openForkForm } from "./fork_form.js";
import { gameUrl } from "./game.js";
import { setCheckpointOptions } from "./run_options.js";

/** Why resuming `detail` is not possible right now, or null. */
function resumeBlocker(detail) {
  if (detail.status.state === "running") return "This run is training right now. Stop it first, or wait until it finishes.";
  if (detail.checkpoints.length === 0) return "Nothing to resume yet: the run has not saved a checkpoint (it saves one every train.checkpoint_interval updates).";
  return null;
}

/** Why a new run cannot start from `detail`'s checkpoints right now, or null. */
function forkBlocker(detail) {
  if (detail.checkpoints.length === 0) return "No checkpoint to start from yet: the run has not saved one.";
  return null;
}

// [evaluation] settings a resume may change: field -> [input id, label, help, minimum].
const EVALUATION_FIELDS = {
  interval: ["resume-eval-interval", "evaluate every (env steps)", "0 = no evaluation while training.", 0],
  games: ["resume-eval-games", "games per evaluation", "Complete greedy games each evaluation plays.", 1],
  seed: ["resume-eval-seed", "evaluation seed", "Same seed and games give the same numbers for the same checkpoint.", null],
};

function option(title, when, ...body) {
  return h("div", { class: "continue-option" }, h("h4", { text: title }), h("p", { class: "muted small", text: when }), ...body);
}

/**
 * The panel for run `name` of `game`. `update(detail)` refreshes it from /api/runs/{name};
 * `onChanged()` is called after a resume, `onForked(newRun)` once a new run was started.
 */
export function continueTrainingPanel(game, name, { onChanged, onForked }) {
  let detail = null;
  let keepLoaded = false;
  const els = {
    steps: h("input", { type: "number", min: "1", step: "1", id: "resume-steps", placeholder: "keep current target" }),
    keep: h("input", { type: "checkbox", id: "resume-keep" }),
    keepNote: h("div", { class: "muted small" }),
    target: h("div", { class: "muted small" }),
    description: descriptionField("Optional. Saved in the run's history (run.json) with this resume."),
    evaluation: Object.fromEntries(
      Object.entries(EVALUATION_FIELDS).map(([key, [id, , , min]]) => [key, h("input", { type: "number", id, step: "1", min: min ?? undefined })]),
    ),
    resume: h("button", { type: "button", class: "primary", text: "Resume this run" }),
    resumeBlocked: h("div", { class: "message info" }),
    checkpoint: h("select", { id: "fork-checkpoint" }),
    fork: h("button", { type: "button", class: "primary", text: "Choose settings…" }),
    forkBlocked: h("div", { class: "message info" }),
    message: h("div", { class: "message", role: "alert" }),
  };

  function updateKeepNote() {
    els.keepNote.textContent = els.keep.checked
      ? ""
      : "Older checkpoints will be deleted each time a new one is saved; only the latest can then be watched or continued from.";
  }

  /** {evaluation_interval: n, ...} for the evaluation settings that differ from the run's, or {error}. */
  function evaluationChanges() {
    const changes = {};
    for (const [key, [, label, , min]] of Object.entries(EVALUATION_FIELDS)) {
      const raw = els.evaluation[key].value.trim();
      const value = Number(raw);
      if (raw === "" || !Number.isInteger(value) || (min !== null && value < min)) {
        return { error: `${label}: enter a whole number${min !== null ? ` of at least ${min}` : ""}.` };
      }
      if (value !== detail.config.evaluation[key]) changes[`evaluation_${key}`] = value;
    }
    return { changes };
  }

  async function resume() {
    const evaluation = evaluationChanges();
    if (evaluation.error) {
      message(els.message, "error", evaluation.error);
      return;
    }
    const raw = els.steps.value.trim();
    const steps = raw === "" ? null : Number(raw);
    if (steps !== null && (!Number.isInteger(steps) || steps < 1)) {
      message(els.message, "error", "Total env steps must be a positive whole number (or empty to keep the current target).");
      return;
    }
    if (steps !== null && steps <= detail.checkpoint_steps) {
      message(els.message, "error", `The run already has ${formatNumber(detail.checkpoint_steps)} steps; choose a larger total.`);
      return;
    }
    els.resume.disabled = true;
    try {
      const body = { total_env_steps: steps, keep_checkpoints: els.keep.checked, description: els.description.get(), ...evaluation.changes };
      await post(gameUrl(`/api/runs/${encodeURIComponent(name)}/resume`, game), body);
      message(els.message, "info", "Resumed.");
      els.steps.value = "";
      els.description.set("");
    } catch (error) {
      message(els.message, "error", error.message);
    }
    onChanged();
  }

  async function fork() {
    const checkpoint = detail.checkpoints.find((c) => c.name === els.checkpoint.value) ?? detail.checkpoints.at(-1);
    try {
      await openForkForm(game, detail, checkpoint, onForked);
    } catch (error) {
      message(els.message, "error", `Could not open the form: ${error.message}`);
    }
  }

  els.resume.addEventListener("click", resume);
  els.fork.addEventListener("click", fork);
  els.keep.addEventListener("change", updateKeepNote);

  const el = h(
    "div",
    { class: "panel" },
    h("h3", { text: "Continue training" }),
    h(
      "div",
      { class: "continue-options" },
      option(
        "Resume this run",
        "Use this to train the same run longer, or to pick it up after it was stopped: same name and settings, from its latest checkpoint.",
        h("div", { class: "field" }, h("label", { for: "resume-steps", text: "train until (total env steps)" }), els.steps, els.target),
        h("label", { class: "row inline-check", for: "resume-keep" }, els.keep, h("span", { class: "label", text: "keep older checkpoints" })),
        els.keepNote,
        h("p", { class: "muted small", text: "Evaluation while training (greedy games recorded in evaluations.jsonl):" }),
        Object.entries(EVALUATION_FIELDS).map(([key, [id, label, help]]) =>
          h("div", { class: "field" }, h("label", { for: id, text: label }), els.evaluation[key], h("div", { class: "description", text: help })),
        ),
        els.description.el,
        h("div", { class: "row", style: "margin-top:10px" }, els.resume),
        els.resumeBlocked,
      ),
      option(
        "Start a new run from a checkpoint",
        "Use this to try other rewards or training settings without changing this run: a new run, with its own name, starts from the weights of the checkpoint you pick.",
        h("div", { class: "field" }, h("label", { for: "fork-checkpoint", text: "checkpoint" }), els.checkpoint),
        h("div", { class: "row", style: "margin-top:10px" }, els.fork),
        els.forkBlocked,
      ),
    ),
    h("div", { style: "margin-top:10px" }, els.message),
  );

  function update(d) {
    detail = d;
    const resumeReason = resumeBlocker(d);
    els.resume.disabled = Boolean(resumeReason);
    els.resumeBlocked.textContent = resumeReason ?? "";
    const forkReason = forkBlocker(d);
    els.fork.disabled = Boolean(forkReason);
    els.forkBlocked.textContent = forkReason ?? "";
    els.checkpoint.disabled = Boolean(forkReason);
    setCheckpointOptions(els.checkpoint, d.checkpoints);
    const total = d.total_env_steps;
    els.target.textContent =
      total && d.checkpoint_steps >= total
        ? `It reached its target of ${formatSteps(total)} steps; enter a larger total to train further.`
        : `Empty keeps the current target (${formatSteps(total)} steps).`;
    if (!keepLoaded) {
      for (const key of Object.keys(EVALUATION_FIELDS)) els.evaluation[key].value = String(d.config.evaluation[key]);
      els.keep.checked = d.config.train.keep_checkpoints !== false;
      keepLoaded = true;
      updateKeepNote();
    }
  }

  return { el, update };
}
