// The Runs tab's "Evaluation" panel for one run: the latest recorded evaluation (greedy games, no exploration),
// a chart of one evaluation metric over env steps, every recorded evaluation, and "Run evaluation now".

import { LineChart, seriesColor } from "./chart.js";
import { formatNumber, formatSteps, h, replace } from "./dom.js";
import { chartableDefinitions, evaluationPoints, formatMetric, headlineDefinition, metricDefinitions, trainingEvaluations } from "./evaluation_metrics.js";
import { evaluationRunner } from "./evaluation_runner.js";
import { evaluationDetails } from "./evaluation_view.js";

const SOURCES = { training: "while training", manual: "on request" };

function emptyState(detail) {
  const interval = detail.config.evaluation?.interval ?? 0;
  if (interval === 0) {
    return "This run was not evaluated while it trained (evaluation.interval = 0). Evaluate any checkpoint below, or resume the run with an evaluation interval to record evaluations as it trains.";
  }
  return `No evaluation recorded yet: the run evaluates every ${formatSteps(interval)} env steps and at the end of training.`;
}

function checkpointCell(evaluation) {
  if (!evaluation.checkpoint) return h("td", { class: "muted", text: "–" });
  if (evaluation.checkpoint_available) return h("td", { class: "mono small", text: evaluation.checkpoint });
  return h("td", { class: "mono small muted", title: "Deleted (train.keep_checkpoints = false): this evaluation can no longer be reproduced." }, `${evaluation.checkpoint} (deleted)`);
}

/** Every recorded evaluation, newest first, with the headline metric and the mean score. */
function historyTable(evaluations) {
  const shown = [headlineDefinition(), metricDefinitions().find((d) => d.key === "mean_score")].filter((d, i, all) => d && all.indexOf(d) === i);
  return h(
    "table",
    {},
    h(
      "thead",
      {},
      h("tr", {}, h("th", { class: "num", text: "env steps" }), h("th", { text: "made" }), h("th", { text: "games · seed" }), h("th", { text: "checkpoint" }), shown.map((d) => h("th", { class: "num", text: d.label }))),
    ),
    h(
      "tbody",
      {},
      [...evaluations].reverse().map((e) =>
        h(
          "tr",
          {},
          h("td", { class: "num", text: formatNumber(e.env_steps) }),
          h("td", { class: "small", text: `${SOURCES[e.source] ?? e.source}, ${new Date(e.timestamp).toLocaleString()}` }),
          h("td", { class: "small", text: `${formatNumber(e.settings.games)} · ${e.settings.seed}` }),
          checkpointCell(e),
          shown.map((d) => h("td", { class: "num", text: formatMetric(d, e.statistics[d.key]) })),
        ),
      ),
    ),
  );
}

/** The panel for run `name` of `game`; `update(detail)` refreshes it, `onSaved()` runs after an evaluation was saved. */
export function evaluationPanel(game, name, { onSaved }) {
  const definitions = chartableDefinitions();
  let evaluations = [];
  const els = {
    latest: h("div", {}),
    metric: h("select", { id: "eval-metric" }, definitions.map((d) => h("option", { value: d.key, text: d.label }))),
    metricHelp: h("p", { class: "muted small" }),
    chartWrap: h("div", { class: "chart-wrap" }),
    history: h("details", {}),
    runner: evaluationRunner(game, name, { onSaved }),
  };
  const chart = new LineChart(els.chartWrap, { xFormat: formatSteps, empty: "No evaluations recorded while training." });

  function drawChart() {
    const definition = definitions.find((d) => d.key === els.metric.value) ?? definitions[0];
    if (!definition) return;
    els.metricHelp.textContent = definition.description;
    chart.options.yFormat = (value) => formatMetric({ ...definition, unit: "" }, value);
    chart.setSeries([{ label: definition.label, color: seriesColor(0), points: evaluationPoints(trainingEvaluations(evaluations), definition.key) }]);
  }

  els.metric.addEventListener("change", drawChart);

  const el = h(
    "div",
    { class: "panel" },
    h("h3", { text: "Evaluation (greedy, no exploration)" }),
    h("p", { class: "muted small", text: "Complete games played with the most likely action at every step, unlike the training curve above, which comes from the exploring policy." }),
    els.latest,
    h("div", { class: "row", style: "margin-top:14px" }, h("label", { for: "eval-metric", class: "label", text: "chart" }), els.metric),
    els.metricHelp,
    els.chartWrap,
    h("p", { class: "muted small", text: "Evaluations made while training, at their env steps. Missing values (N/A) are left out of the line." }),
    els.history,
    h("div", { style: "margin-top:14px" }, els.runner.el),
  );

  function update(detail) {
    evaluations = detail.evaluations;
    const latest = evaluations.at(-1);
    if (latest) {
      replace(
        els.latest,
        h("h4", { class: "evaluation-heading", text: "Latest evaluation" }),
        evaluationDetails(metricDefinitions(), latest),
        latest.checkpoint_available || !latest.checkpoint
          ? null
          : h("div", { class: "message warn", text: "Its checkpoint has been deleted (train.keep_checkpoints = false), so this evaluation can no longer be reproduced." }),
      );
    } else {
      replace(els.latest, h("div", { class: "message info", text: emptyState(detail) }));
    }
    replace(els.history, h("summary", { text: `All recorded evaluations (${evaluations.length})` }), evaluations.length ? historyTable(evaluations) : h("p", { class: "muted small", text: "None yet." }));
    els.history.hidden = evaluations.length === 0;
    drawChart();
    els.runner.update(detail);
  }

  return { el, update };
}
