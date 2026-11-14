// Drawing one evaluation: what was evaluated and how, a table of its statistics with descriptions, and outcome counts.

import { formatNumber, formatSteps, h } from "./dom.js";
import { formatMetric } from "./evaluation_metrics.js";

const SOURCES = { training: "while training", manual: "on request" };

/** "at 1.2M env steps · checkpoint_….pt · 100 games, seed 1000 · while training · <date>". */
export function evaluationCaption(evaluation) {
  const s = evaluation.settings;
  return [
    `at ${formatSteps(evaluation.env_steps)} env steps`,
    evaluation.checkpoint ?? "no checkpoint",
    `${formatNumber(s.games)} games, seed ${s.seed}, step cap ${formatNumber(s.max_episode_steps)}, ${s.device}`,
    SOURCES[evaluation.source] ?? evaluation.source,
    new Date(evaluation.timestamp).toLocaleString(),
  ].join(" · ");
}

/** A table: metric | value | what it means. Values that could not be computed show as N/A. */
export function statisticsTable(definitions, statistics) {
  return h(
    "table",
    { class: "evaluation-table" },
    h("thead", {}, h("tr", {}, h("th", { text: "metric" }), h("th", { class: "num", text: "value" }), h("th", { text: "what it means" }))),
    h(
      "tbody",
      {},
      definitions
        .filter((d) => d.key in statistics)
        .map((d) =>
          h(
            "tr",
            {},
            h("td", { text: d.label }),
            h("td", { class: `num${statistics[d.key] === null ? " muted" : ""}`, text: formatMetric(d, statistics[d.key]) }),
            h("td", { class: "muted small", text: d.description }),
          ),
        ),
    ),
  );
}

/** "how the games ended: hit_self 61 · hit_wall 37 · won 2". */
export function outcomesLine(outcomes) {
  const parts = Object.entries(outcomes).map(([outcome, count]) => `${outcome} ${formatNumber(count)}`);
  return h("p", { class: "muted small", text: `How the games ended: ${parts.join(" · ") || "–"}` });
}

/** Caption, statistics table and outcomes of one evaluation. */
export function evaluationDetails(definitions, evaluation) {
  return h(
    "div",
    { class: "evaluation" },
    h("p", { class: "muted small", text: evaluationCaption(evaluation) }),
    statisticsTable(definitions, evaluation.statistics),
    outcomesLine(evaluation.outcomes),
  );
}
