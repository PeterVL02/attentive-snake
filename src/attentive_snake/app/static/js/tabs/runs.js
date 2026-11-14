// Tab 2: the runs, their progress, metrics and history; stop, resume, continue with changes, and jump to watching a checkpoint.

import { get, post } from "../api.js";
import { navigate } from "../bus.js";
import { badge, formatNumber, formatSteps, h, message, replace } from "../dom.js";
import { descriptionField } from "../fields.js";
import { openForkForm } from "../fork_form.js";
import { LineChart, renderLegend, seriesColor } from "../chart.js";
import { monotonic, points } from "../metric_series.js";

const METRICS = [
  { key: "mean_score", label: "mean score", hidden: false },
  { key: "mean_length", label: "mean episode length", hidden: true },
  { key: "mean_return", label: "mean return", hidden: true },
];
const STAT_KEYS = [
  "mean_score", "mean_length", "mean_return", "episodes", "policy_loss", "value_loss", "entropy",
  "approx_kl", "clip_fraction", "explained_variance", "learning_rate", "steps_per_second",
];
const FAST_POLL_MS = 2000;
const SLOW_POLL_MS = 8000;

let root;
let listEl;
let detailEl;
let selected = null;
let visible = false;
let timer = null;
let detail = null;
let chart = null;
const series = METRICS.map((m, i) => ({ ...m, color: null, index: i, points: [] }));
let els = {};

function renderList(runs) {
  if (runs.length === 0) {
    replace(listEl, h("li", { class: "empty", text: "No runs yet. Start one in New training." }));
    return;
  }
  replace(
    listEl,
    runs.map((run) =>
      h(
        "li",
        {},
        h(
          "button",
          { type: "button", "aria-current": String(run.name === selected), onclick: () => select(run.name) },
          h(
            "span",
            { class: "run-label" },
            h("span", { class: "name", text: run.name, title: run.name }),
            run.description ? h("span", { class: "run-description", text: run.description, title: run.description }) : null,
          ),
          badge(run.status.state),
        ),
      ),
    ),
  );
}

function buildDetail() {
  els = {
    title: h("h2", {}),
    badge: h("span"),
    stop: h("button", { type: "button", class: "danger", text: "Stop", onclick: stop }),
    progress: h("div", {}),
    progressText: h("div", { class: "muted small" }),
    resumeSteps: h("input", { type: "number", min: "1", step: "1", id: "resume-steps", placeholder: "keep current target", style: "width: 180px" }),
    resume: h("button", { type: "button", class: "primary", text: "Resume", onclick: resume }),
    keep: h("input", { type: "checkbox", id: "resume-keep" }),
    keepNote: h("div", { class: "muted small" }),
    resumeDescription: descriptionField("Optional. Saved in the run's history (run.json) with this resume."),
    fork: h("button", { type: "button", text: "Continue with changes…", onclick: fork }),
    lineage: h("div", { class: "lineage" }),
    history: h("div", {}),
    actionMessage: h("div", { class: "message", role: "alert" }),
    legend: h("div", { class: "legend" }),
    chartWrap: h("div", { class: "chart-wrap" }),
    stats: h("div", { class: "stats" }),
    checkpoint: h("select", { id: "run-checkpoint" }),
    log: h("pre", { class: "log" }),
    config: h("div", {}),
  };
  replace(
    detailEl,
    h("div", { class: "row spread" }, h("div", { class: "row" }, els.title, els.badge), els.stop),
    els.lineage,
    h(
      "div",
      { class: "panel" },
      h("div", { class: "progress" }, els.progress),
      h("div", { style: "margin-top:6px" }, els.progressText),
      h(
        "div",
        { class: "row", style: "margin-top:14px" },
        h("label", { for: "resume-steps", class: "label", text: "train until (total env steps)" }),
        els.resumeSteps,
        h("label", { class: "row inline-check", for: "resume-keep" }, els.keep, h("span", { class: "label", text: "keep older checkpoints" })),
      ),
      els.keepNote,
      h("div", { style: "margin-top:10px" }, els.resumeDescription.el),
      h(
        "div",
        { class: "row", style: "margin-top:10px" },
        els.resume,
        els.fork,
        h("span", { class: "muted small", text: "Resume continues this run with its settings. Continue with changes starts a new run from the checkpoint selected under Checkpoints." }),
      ),
      h("div", { style: "margin-top:10px" }, els.actionMessage),
    ),
    h("div", { class: "panel" }, h("h3", { text: "Training curve" }), els.chartWrap, els.legend),
    h("div", { class: "panel" }, h("h3", { text: "Latest metrics" }), els.stats),
    h(
      "div",
      { class: "panel" },
      h("h3", { text: "Checkpoints" }),
      h(
        "div",
        { class: "row" },
        els.checkpoint,
        h("button", { type: "button", text: "Watch", onclick: () => navigate("watch", { run: selected, checkpoint: els.checkpoint.value || null }) }),
        h("button", { type: "button", text: "Architecture", onclick: () => navigate("architecture", { source: "run", run: selected }) }),
      ),
    ),
    h("div", { class: "panel" }, h("h3", { text: "History" }), els.history),
    h("div", { class: "panel" }, h("h3", { text: "Log (tail of train.log)" }), els.log),
    h("details", { class: "panel" }, h("summary", { text: "Config (read-only)" }), els.config),
  );
  chart = new LineChart(els.chartWrap, { xFormat: formatSteps, empty: "No metrics logged yet." });
  series.forEach((s) => (s.color = seriesColor(s.index)));
  renderLegend(els.legend, series, () => chart.draw());
  els.keep.addEventListener("change", updateKeepNote);
}

function updateKeepNote() {
  els.keepNote.textContent = els.keep.checked
    ? ""
    : "Older checkpoints will be deleted each time a new one is saved; only the latest can then be watched or continued from.";
}

function renderLineage(parent) {
  if (!parent) return null;
  return [
    "Continued from ",
    h("button", { type: "button", class: "link", text: parent.run, onclick: () => select(parent.run) }),
    ` at ${formatNumber(parent.env_steps)} steps (${parent.checkpoint}).`,
  ];
}

const ACTION_LABELS = { start: "started", resume: "resumed", fork: "continued from another run" };

function renderHistory(events) {
  if (!events.length) return h("p", { class: "muted small", text: "No recorded history (runs started before run.json existed have none)." });
  return h(
    "table",
    {},
    h("thead", {}, h("tr", {}, h("th", { text: "when" }), h("th", { text: "what" }), h("th", { class: "num", text: "at env steps" }), h("th", { text: "description" }))),
    h(
      "tbody",
      {},
      [...events].reverse().map((e) =>
        h(
          "tr",
          {},
          h("td", { class: "small", text: new Date(e.timestamp).toLocaleString() }),
          h("td", { text: e.parent ? `${ACTION_LABELS[e.action]} (${e.parent.run} @ ${formatSteps(e.parent.env_steps)})` : ACTION_LABELS[e.action] ?? e.action }),
          h("td", { class: "num", text: formatNumber(e.env_steps_at_action) }),
          h("td", { class: "pre-wrap", text: e.description ?? "–" }),
        ),
      ),
    ),
  );
}

function renderConfig(config) {
  return Object.entries(config).map(([section, table]) =>
    h(
      "div",
      { style: "margin-top:10px" },
      h("div", { class: "label", text: `[${section}]` }),
      h("table", {}, h("tbody", {}, Object.entries(table).map(([k, v]) => h("tr", {}, h("td", { class: "mono", text: k }), h("td", { class: "mono", text: v === null ? "None" : String(v) }))))),
    ),
  );
}

function renderDetail(d) {
  const state = d.status.state;
  els.title.textContent = d.name;
  replace(els.badge, badge(state));
  if (state === "failed") els.badge.append(h("span", { class: "muted small", text: ` exit code ${d.status.exit_code}` }));
  els.stop.hidden = state !== "running";
  els.resume.disabled = state === "running";
  els.fork.disabled = d.checkpoints.length === 0;
  if (!els.keepLoaded) {
    els.keep.checked = d.config.train.keep_checkpoints !== false;
    els.keepLoaded = true;
    updateKeepNote();
  }
  replace(els.lineage, renderLineage(d.parent));
  els.lineage.hidden = !d.parent;
  replace(els.history, renderHistory(d.history));
  const total = d.total_env_steps || 0;
  const pct = total ? Math.min(100, (100 * d.env_steps) / total) : 0;
  els.progress.style.width = `${pct}%`;
  els.progressText.textContent = `${formatNumber(d.env_steps)} / ${formatNumber(total)} env steps (${pct.toFixed(1)}%) · last checkpoint at ${formatNumber(d.checkpoint_steps)}`;

  const records = monotonic(d.metrics);
  for (const s of series) s.points = points(records, s.key);
  chart.setSeries(series);

  const latest = d.latest_metrics;
  replace(
    els.stats,
    latest
      ? STAT_KEYS.filter((k) => k in latest).map((k) => h("div", { class: "stat" }, h("div", { class: "k", text: k }), h("div", { class: "v", text: formatNumber(latest[k]) })))
      : h("div", { class: "muted", text: "No metrics yet." }),
  );

  const previous = els.checkpoint.value;
  const checkpoints = [...d.checkpoints].reverse();
  replace(
    els.checkpoint,
    checkpoints.length === 0 ? h("option", { value: "", text: "No checkpoints yet" }) : null,
    checkpoints.map((c, i) => h("option", { value: c.name, text: `${formatSteps(c.steps)} steps${i === 0 ? " (latest)" : ""}` })),
  );
  if (checkpoints.some((c) => c.name === previous)) els.checkpoint.value = previous;

  const atBottom = els.log.scrollTop + els.log.clientHeight >= els.log.scrollHeight - 8;
  els.log.textContent = d.log || "(no log yet)";
  if (atBottom) els.log.scrollTop = els.log.scrollHeight;
  replace(els.config, renderConfig(d.config));
}

async function refresh() {
  try {
    renderList(await get("/api/runs"));
    if (selected) {
      detail = await get(`/api/runs/${encodeURIComponent(selected)}`);
      if (!els.title) buildDetail();
      renderDetail(detail);
    }
  } catch (error) {
    if (error.status === 404) selected = null; // the run directory was removed
    detail = null;
    replace(detailEl, h("div", { class: "message error", text: error.message }));
    els = {};
  }
  schedule();
}

function schedule() {
  clearTimeout(timer);
  if (!visible) return;
  const running = detail && detail.status.state === "running";
  timer = setTimeout(refresh, running ? FAST_POLL_MS : SLOW_POLL_MS);
}

function select(name) {
  if (name !== selected) {
    selected = name;
    detail = null;
    replace(detailEl, h("div", { class: "empty", text: "Loading…" }));
    els = {};
  }
  refresh();
}

async function stop() {
  if (!confirm(`Stop training ${selected}? Progress since the last checkpoint is lost.`)) return;
  els.stop.disabled = true;
  message(els.actionMessage, "info", "Stopping…");
  try {
    await post(`/api/runs/${encodeURIComponent(selected)}/stop`);
    message(els.actionMessage, "info", "Stopped.");
  } catch (error) {
    message(els.actionMessage, "error", error.message);
  } finally {
    els.stop.disabled = false;
    refresh();
  }
}

async function resume() {
  const raw = els.resumeSteps.value.trim();
  const steps = raw === "" ? null : Number(raw);
  if (steps !== null && (!Number.isInteger(steps) || steps < 1)) {
    message(els.actionMessage, "error", "Total env steps must be a positive whole number (or empty to keep the current target).");
    return;
  }
  if (steps !== null && detail && steps <= detail.checkpoint_steps) {
    message(els.actionMessage, "error", `The run already has ${formatNumber(detail.checkpoint_steps)} steps; choose a larger total.`);
    return;
  }
  try {
    const body = { total_env_steps: steps, keep_checkpoints: els.keep.checked, description: els.resumeDescription.get() };
    await post(`/api/runs/${encodeURIComponent(selected)}/resume`, body);
    message(els.actionMessage, "info", "Resumed.");
    els.resumeSteps.value = "";
    els.resumeDescription.set("");
  } catch (error) {
    message(els.actionMessage, "error", error.message);
  }
  refresh();
}

async function fork() {
  if (!detail) return;
  const name = els.checkpoint.value;
  const checkpoint = detail.checkpoints.find((c) => c.name === name) ?? detail.checkpoints[detail.checkpoints.length - 1];
  if (!checkpoint) {
    message(els.actionMessage, "error", "This run has no checkpoint to continue from yet.");
    return;
  }
  try {
    await openForkForm(detail, checkpoint, (newRun) => select(newRun));
  } catch (error) {
    message(els.actionMessage, "error", `Could not open the form: ${error.message}`);
  }
}

export function init(element) {
  root = element;
  listEl = h("ul", { class: "run-list" });
  detailEl = h("div", {}, h("div", { class: "empty", text: "Select a run." }));
  replace(
    root,
    h(
      "div",
      { class: "columns narrow-left" },
      h("div", { class: "panel sticky" }, h("div", { class: "row spread" }, h("h3", { text: "Runs", style: "margin:0" }), h("button", { type: "button", class: "clear", text: "Refresh", onclick: refresh })), listEl),
      detailEl,
    ),
  );
}

export async function show(params = {}) {
  visible = true;
  if (params.run) {
    select(params.run);
    return;
  }
  if (!selected) {
    try {
      const runs = await get("/api/runs");
      if (runs.length > 0) {
        select(runs[0].name);
        return;
      }
    } catch {
      /* refresh() shows the error */
    }
  }
  refresh();
}

export function hide() {
  visible = false;
  clearTimeout(timer);
}
