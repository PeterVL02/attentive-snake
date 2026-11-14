// Tab 2: the selected game's runs, their progress, metrics, evaluations and history; stop, continue training
// (resume or a new run from a checkpoint), evaluate a checkpoint, and jump to watching one.

import { get, post } from "../api.js";
import { navigate } from "../bus.js";
import { badge, formatNumber, formatSteps, h, message, replace } from "../dom.js";
import { continueTrainingPanel } from "../continue_training.js";
import { evaluationPanel } from "../evaluation_panel.js";
import { currentGame, gameUrl } from "../game.js";
import { LineChart, renderLegend, seriesColor } from "../chart.js";
import { monotonic, points } from "../metric_series.js";
import { setCheckpointOptions } from "../run_options.js";
import { rulesSummary } from "../rules_summary.js";
import { loadSchema } from "../schema.js";

// Logged by every algorithm; each algorithm adds its own (schema.algorithms[name].metrics).
const METRICS = [
  { key: "mean_score", label: "mean score", hidden: false },
  { key: "mean_length", label: "mean episode length", hidden: true },
  { key: "mean_return", label: "mean return", hidden: true },
];
const SHARED_STATS = ["mean_score", "mean_length", "mean_return", "episodes"];
const LOOP_STATS = ["learning_rate", "steps_per_second"];
const FAST_POLL_MS = 2000;
const SLOW_POLL_MS = 8000;

let listEl;
let detailEl;
let selected = null; // run name, within the selected game
let visible = false;
let timer = null;
let detail = null;
let chart = null;
let series = [];
let seriesAlgorithm; // the algorithm whose metrics `series` holds
let algorithmInfos = {}; // schema.algorithms of the selected game
let els = {};

/** The run's algorithm facts from the schema (label, metrics), or null for an unreadable config. */
function algorithmInfo(name) {
  return algorithmInfos[name] ?? null;
}

/** Chart series: the shared metrics, then the algorithm's own (hidden until switched on in the legend). */
function chartSeries(info) {
  const own = (info?.metrics ?? []).map((m) => ({ key: m.key, label: m.label, hidden: true }));
  return [...METRICS, ...own].map((m, i) => ({ ...m, color: seriesColor(i), index: i, points: [] }));
}

function runMeta(run) {
  if (run.config_error) return "config.toml cannot be read by this version";
  return `${run.kind ?? "?"} · ${algorithmInfo(run.algorithm)?.label ?? run.algorithm ?? "?"} · ${rulesSummary(run.rules)}`;
}

function runUrl(name, suffix = "") {
  return gameUrl(`/api/runs/${encodeURIComponent(name)}${suffix}`);
}

function renderList(runs) {
  if (runs.length === 0) {
    replace(listEl, h("li", { class: "empty", text: `No ${currentGame().title} runs yet. Start one in New training.` }));
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
            h("span", { class: "run-meta", text: runMeta(run) }),
            run.description ? h("span", { class: "run-description", text: run.description, title: run.description }) : null,
          ),
          badge(run.status.state),
        ),
      ),
    ),
  );
}

function buildDetail(name) {
  els = {
    title: h("h2", {}),
    badge: h("span"),
    meta: h("div", { class: "muted small" }),
    stop: h("button", { type: "button", class: "danger", text: "Stop", onclick: stop }),
    progress: h("div", {}),
    progressText: h("div", { class: "muted small" }),
    continuePanel: continueTrainingPanel(currentGame().name, name, { onChanged: refresh, onForked: (newRun) => select(newRun) }),
    evaluationPanel: evaluationPanel(currentGame().name, name, { onSaved: refresh }),
    lineage: h("div", { class: "lineage" }),
    history: h("div", {}),
    actionMessage: h("div", { class: "message", role: "alert" }),
    legend: h("div", { class: "legend" }),
    chartWrap: h("div", { class: "chart-wrap" }),
    stats: h("div", { class: "stats" }),
    configError: h("div", { class: "message error", role: "alert", hidden: true }),
    checkpoint: h("select", { id: "run-checkpoint", "aria-label": "checkpoint" }),
    log: h("pre", { class: "log" }),
    config: h("div", {}),
  };
  replace(
    detailEl,
    h("div", { class: "row spread" }, h("div", { class: "row" }, els.title, els.badge), els.stop),
    els.meta,
    els.configError,
    els.lineage,
    h(
      "div",
      { class: "panel" },
      h("h3", { text: "Progress" }),
      h("div", { class: "progress" }, els.progress),
      h("div", { style: "margin-top:6px" }, els.progressText),
      h("div", { style: "margin-top:10px" }, els.actionMessage),
    ),
    els.continuePanel.el,
    h("div", { class: "panel" }, h("h3", { text: "Training curve (with exploration)" }), els.chartWrap, els.legend),
    els.evaluationPanel.el,
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
  seriesAlgorithm = undefined;
}

/** Rebuild the series and legend when the shown run's algorithm (and so its metrics) differs. */
function useAlgorithmSeries(algorithm) {
  if (algorithm === seriesAlgorithm) return;
  seriesAlgorithm = algorithm;
  series = chartSeries(algorithmInfo(algorithm));
  renderLegend(els.legend, series, () => chart.draw());
}

/** [key, label] of the latest-metrics tiles: shared statistics, the algorithm's own, then the loop's. */
function statKeys(info) {
  const own = (info?.metrics ?? []).map((m) => [m.key, m.label]);
  return [...SHARED_STATS.map((k) => [k, k]), ...own, ...LOOP_STATS.map((k) => [k, k])];
}

function renderLineage(parent) {
  if (!parent) return null;
  return [
    "Started from ",
    h("button", { type: "button", class: "link", text: parent.run, onclick: () => select(parent.run) }),
    ` at ${formatNumber(parent.env_steps)} steps (${parent.checkpoint}).`,
  ];
}

const ACTION_LABELS = { start: "started", resume: "resumed", fork: "started from another run's checkpoint" };

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
  if (!config) return h("p", { class: "muted small", text: "Not available: config.toml cannot be read (see above)." });
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
  els.meta.textContent = `${currentGame().title} · ${runMeta(d)}`;
  els.configError.hidden = !d.config_error;
  els.configError.textContent = d.config_error
    ? `This run's config.toml cannot be read by this version of the code, so it cannot be resumed, forked, watched or evaluated here: ${d.config_error}`
    : "";
  els.stop.hidden = state !== "running";
  els.continuePanel.update(d);
  els.evaluationPanel.update(d);
  replace(els.lineage, renderLineage(d.parent));
  els.lineage.hidden = !d.parent;
  replace(els.history, renderHistory(d.history));
  const total = d.total_env_steps || 0;
  const pct = total ? Math.min(100, (100 * d.env_steps) / total) : 0;
  els.progress.style.width = `${pct}%`;
  els.progressText.textContent = `${formatNumber(d.env_steps)} / ${formatNumber(total)} env steps (${pct.toFixed(1)}%) · last checkpoint at ${formatNumber(d.checkpoint_steps)}`;

  const info = algorithmInfo(d.algorithm);
  useAlgorithmSeries(d.algorithm);
  const records = monotonic(d.metrics);
  for (const s of series) s.points = points(records, s.key);
  chart.setSeries(series);

  const latest = d.latest_metrics;
  replace(
    els.stats,
    latest
      ? statKeys(info).filter(([k]) => k in latest).map(([k, label]) => h("div", { class: "stat", title: k }, h("div", { class: "k", text: label }), h("div", { class: "v", text: formatNumber(latest[k]) })))
      : h("div", { class: "muted", text: "No metrics yet." }),
  );
  setCheckpointOptions(els.checkpoint, d.checkpoints);

  const atBottom = els.log.scrollTop + els.log.clientHeight >= els.log.scrollHeight - 8;
  els.log.textContent = d.log || "(no log yet)";
  if (atBottom) els.log.scrollTop = els.log.scrollHeight;
  replace(els.config, renderConfig(d.config));
}

/** Reload the list and the selected run; results that arrive after the game or run changed are dropped. */
async function refresh() {
  const game = currentGame().name;
  const run = selected;
  const stale = () => game !== currentGame().name || run !== selected;
  try {
    const [runs, { schema }] = await Promise.all([get(gameUrl("/api/runs", game)), loadSchema(game)]);
    if (stale()) return;
    algorithmInfos = schema.algorithms;
    renderList(runs);
    if (run) {
      const loaded = await get(gameUrl(`/api/runs/${encodeURIComponent(run)}`, game));
      if (stale()) return;
      detail = loaded;
      if (!els.title) buildDetail(run);
      renderDetail(detail);
    }
  } catch (error) {
    if (stale()) return;
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
    await post(runUrl(selected, "/stop"));
    message(els.actionMessage, "info", "Stopped.");
  } catch (error) {
    message(els.actionMessage, "error", error.message);
  } finally {
    els.stop.disabled = false;
    refresh();
  }
}

export function init(element) {
  listEl = h("ul", { class: "run-list" });
  detailEl = h("div", {}, h("div", { class: "empty", text: "Select a run." }));
  replace(
    element,
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
    const game = currentGame().name;
    try {
      const runs = await get(gameUrl("/api/runs", game));
      if (game === currentGame().name && !selected && runs.length > 0) {
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

export function reset() {
  clearTimeout(timer);
  selected = null;
  detail = null;
  els = {};
  replace(listEl, h("li", { class: "empty", text: "Loading…" }));
  replace(detailEl, h("div", { class: "empty", text: "Select a run." }));
}
