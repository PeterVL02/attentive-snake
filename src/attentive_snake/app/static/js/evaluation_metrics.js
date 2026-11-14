// Evaluation statistics without the DOM: the selected game's metric definitions (from /api/games),
// formatting values (N/A for missing ones), chart points and the best of several values.

import { formatNumber } from "./dom.js";
import { currentGame } from "./game.js";
import { monotonic } from "./metric_series.js";

export const NOT_AVAILABLE = "N/A";

/** [{key, label, description, higher_is_better, format, unit}]: the game's own metrics first, then the generic ones. */
export function metricDefinitions(game = currentGame()) {
  return game?.evaluation_metrics ?? [];
}

/** The definitions worth charting (everything but the number of games). */
export function chartableDefinitions(game = currentGame()) {
  return metricDefinitions(game).filter((d) => d.key !== "games");
}

/** The metric to show when there is room for one: the game's first own metric, else the mean score. */
export function headlineDefinition(game = currentGame()) {
  return chartableDefinitions(game)[0] ?? null; // own metrics come first; the first generic one is the mean score
}

export function formatMetric(definition, value) {
  if (value === null || value === undefined) return NOT_AVAILABLE;
  let text;
  if (definition.format === "rate") text = `${formatNumber(100 * value, 1)} %`;
  else if (definition.format === "integer") text = formatNumber(Math.round(value));
  else text = formatNumber(value, 2);
  return definition.unit ? `${text} ${definition.unit}` : text;
}

/** The evaluations made while training, in increasing env steps (a resumed run's superseded ones dropped). */
export function trainingEvaluations(evaluations) {
  return monotonic(evaluations.filter((e) => e.source === "training"));
}

/** [[env steps, value]] of `key` over `evaluations`; missing values (N/A) are left out, never drawn as 0. */
export function evaluationPoints(evaluations, key) {
  return evaluations.filter((e) => typeof e.statistics[key] === "number").map((e) => [e.env_steps, e.statistics[key]]);
}

/** The best of `values` by the definition's direction; null if there is none or neither direction is better. */
export function bestValue(definition, values) {
  const numbers = values.filter((v) => typeof v === "number");
  if (numbers.length === 0 || definition.higher_is_better === null) return null;
  return definition.higher_is_better ? Math.max(...numbers) : Math.min(...numbers);
}
