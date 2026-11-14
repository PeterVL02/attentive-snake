// Tab: several runs of the selected game on one chart, with the config fields that differ between
// them. Runs of different games are never mixed: the list is the selected game's runs only.

import { get, post } from "../api.js";
import { badge, debounce, formatDuration, formatNumber, formatSteps, h, message, replace } from "../dom.js";
import { currentGame, gameUrl, isCurrent } from "../game.js";
import { LineChart, movingAverage, renderLegend, seriesColor, seriesDash } from "../chart.js";
import { estimatedSeconds, median, monotonic, points } from "../metric_series.js";
import { rulesSummary } from "../rules_summary.js";

const METRIC_LABELS = { mean_score: "mean score", mean_length: "mean episode length", mean_return: "mean return" };
const POLL_MS = 10000;

let els = {};
let chart;
let checked = new Set();
let data = null; // {runs, differences}
let series = []; // persistent per run name so legend toggles survive redraws
let visible = false;
let timer = null;

function runRow(run) {
  const id = `cmp-${run.name}`;
  const box = h("input", { type: "checkbox", id, checked: checked.has(run.name) });
  box.addEventListener("change", () => {
    if (box.checked) checked.add(run.name);
    else checked.delete(run.name);
    scheduleCompare();
  });
  return h(
    "li",
    { class: "compare-item" },
    box,
    h(
      "label",
      { for: id },
      h("span", { class: "name", text: run.name }),
      h("span", { class: "muted small", text: `${run.kind ?? "?"} · ${rulesSummary(run.rules)} · ${formatSteps(run.env_steps)} / ${formatSteps(run.total_env_steps)}` }),
      run.description ? h("span", { class: "run-description", text: run.description, title: run.description }) : null,
    ),
    badge(run.status.state),
  );
}

async function refreshRuns() {
  const game = currentGame().name;
  try {
    const runs = await get(gameUrl("/api/runs", game));
    if (!isCurrent(game)) return;
    const names = new Set(runs.map((r) => r.name));
    checked = new Set([...checked].filter((n) => names.has(n)));
    replace(els.list, runs.length === 0 ? h("li", { class: "empty", text: `No ${currentGame().title} runs yet.` }) : runs.map(runRow));
  } catch (error) {
    if (isCurrent(game)) message(els.status, "error", error.message);
  }
}

function seriesFor(name, index) {
  let s = series.find((x) => x.label === name);
  if (!s) {
    s = { label: name, hidden: false, points: [] };
    series.push(s);
  }
  s.color = seriesColor(index);
  s.dash = seriesDash(index);
  return s;
}

function drawChart() {
  if (!data) return;
  const metric = els.metric.value;
  const byTime = els.xAxis.value === "time";
  const window = Number(els.smooth.value);
  els.smoothLabel.textContent = window <= 1 ? "off" : `${window} points`;
  const shown = data.runs.map((run, i) => {
    const records = monotonic(run.metrics);
    const xs = byTime ? estimatedSeconds(records) : null;
    const s = seriesFor(run.name, i);
    s.points = byTime && !xs ? [] : movingAverage(points(records, metric, xs), window);
    return s;
  });
  series = shown;
  chart.options.xFormat = byTime ? formatDuration : formatSteps;
  chart.options.empty = checked.size ? "No metrics for the selected runs yet." : "Select runs to compare.";
  chart.setSeries(shown);
  renderLegend(els.legend, shown, () => chart.draw());
  const missingTime = byTime && data.runs.some((run) => !estimatedSeconds(monotonic(run.metrics)));
  els.timeNote.hidden = !byTime;
  els.timeNote.textContent = missingTime
    ? "Some runs have no steps_per_second in their metrics and are left out of the time axis."
    : "Training time is estimated from the logged steps_per_second (time spent outside training updates, and pauses between resumes, are not included).";
}

function summaryTable() {
  const head = ["", "run", "kind", "parameters", "rules", "final mean score", "best mean score", "steps / s", "description"];
  return h(
    "table",
    {},
    h("thead", {}, h("tr", {}, head.map((t, i) => h("th", { class: i >= 3 && i !== 4 && i < 8 ? "num" : null, text: t })))),
    h(
      "tbody",
      {},
      data.runs.map((run, i) => {
        const records = monotonic(run.metrics);
        const scores = records.map((r) => r.mean_score).filter((v) => typeof v === "number");
        const speeds = records.map((r) => r.steps_per_second).filter((v) => typeof v === "number");
        return h(
          "tr",
          {},
          h("td", {}, h("span", { class: "swatch", style: `background:${seriesColor(i)}` })),
          h("td", { class: "mono", text: run.name }),
          h("td", { text: run.config.model.kind }),
          h("td", { class: "num", text: formatNumber(run.parameters) }),
          h("td", { text: rulesSummary(run.config.game) }),
          h("td", { class: "num", text: scores.length ? formatNumber(scores[scores.length - 1], 2) : "–" }),
          h("td", { class: "num", text: scores.length ? formatNumber(Math.max(...scores), 2) : "–" }),
          h("td", { class: "num", text: formatNumber(median(speeds), 0) }),
          h("td", { class: "small pre-wrap", text: run.description ?? "–" }),
        );
      }),
    ),
  );
}

function differencesTable() {
  if (data.runs.length < 2) return h("p", { class: "muted small", text: "Select at least two runs to see how their configs differ." });
  if (data.differences.length === 0) return h("p", { class: "muted small", text: "The selected runs have identical configs." });
  const show = (v) => (v === null || v === undefined ? "None" : String(v));
  return h(
    "div",
    { style: "overflow-x:auto" },
    h(
      "table",
      {},
      h("thead", {}, h("tr", {}, h("th", { text: "field" }), data.runs.map((r) => h("th", { class: "mono", text: r.name })))),
      h(
        "tbody",
        {},
        data.differences.map((d) =>
          h("tr", {}, h("td", { class: "mono", text: `${d.section}.${d.field}` }), data.runs.map((r) => h("td", { class: "mono", text: show(d.values[r.name]) }))),
        ),
      ),
    ),
  );
}

async function compare() {
  clearTimeout(timer);
  if (checked.size === 0) {
    data = { runs: [], differences: [] };
    drawChart();
    replace(els.tables);
    message(els.status, "", "");
    return;
  }
  const game = currentGame().name;
  let result;
  try {
    result = await post("/api/compare", { game, runs: [...checked] });
  } catch (error) {
    if (isCurrent(game)) message(els.status, "error", error.message);
    return;
  }
  if (!isCurrent(game)) return; // the game changed while waiting: these runs are not shown any more
  data = result;
  message(els.status, "", "");
  drawChart();
  replace(
    els.tables,
    h("div", { class: "panel" }, h("h3", { text: "Runs" }), h("div", { style: "overflow-x:auto" }, summaryTable())),
    h("div", { class: "panel" }, h("h3", { text: "Config fields that differ" }), differencesTable()),
  );
  if (visible && data.runs.some((r) => r.status.state === "running")) timer = setTimeout(compare, POLL_MS);
}

const scheduleCompare = debounce(compare, 200);

export function init(root) {
  els = {
    list: h("ul", { class: "run-list compare-list" }),
    status: h("div", { class: "message", role: "status" }),
    metric: h("select", { id: "cmp-metric" }, Object.entries(METRIC_LABELS).map(([k, label]) => h("option", { value: k, text: label }))),
    xAxis: h("select", { id: "cmp-x" }, h("option", { value: "steps", text: "env steps" }), h("option", { value: "time", text: "training time (estimated)" })),
    smooth: h("input", { type: "range", id: "cmp-smooth", min: "1", max: "50", value: "1" }),
    smoothLabel: h("span", { class: "muted small mono", text: "off" }),
    chartWrap: h("div", { class: "chart-wrap", style: "height:340px" }),
    legend: h("div", { class: "legend" }),
    timeNote: h("p", { class: "muted small", hidden: true }),
    tables: h("div", {}),
  };
  for (const control of [els.metric, els.xAxis]) control.addEventListener("change", drawChart);
  els.smooth.addEventListener("input", drawChart);
  const label = (forId, text) => h("label", { for: forId, class: "label", text });
  replace(
    root,
    h("h2", { text: "Compare runs" }),
    h("p", { class: "lead", text: "Tick runs to draw them on one chart. Runs of different lengths simply end at different points; hover for values, click a legend entry to hide a line." }),
    h(
      "div",
      { class: "columns narrow-left" },
      h("div", { class: "panel sticky" }, h("div", { class: "row spread" }, h("h3", { text: "Runs", style: "margin:0" }), h("button", { type: "button", class: "clear", text: "Refresh", onclick: () => refreshRuns().then(compare) })), els.list),
      h(
        "div",
        {},
        h(
          "div",
          { class: "panel" },
          h("div", { class: "row", style: "margin-bottom:12px" }, label("cmp-metric", "metric"), els.metric, label("cmp-x", "x axis"), els.xAxis, label("cmp-smooth", "smoothing"), els.smooth, els.smoothLabel),
          els.chartWrap,
          els.legend,
          els.timeNote,
          els.status,
        ),
        els.tables,
      ),
    ),
  );
  chart = new LineChart(els.chartWrap, { xFormat: formatSteps, empty: "Select runs to compare." });
}

export async function show() {
  visible = true;
  await refreshRuns();
  compare();
}

export function hide() {
  visible = false;
  clearTimeout(timer);
}

export function reset() {
  clearTimeout(timer);
  checked = new Set();
  series = [];
  data = { runs: [], differences: [] };
  drawChart();
  replace(els.list, h("li", { class: "empty", text: "Loading…" }));
  replace(els.tables);
  message(els.status, "", "");
}
