// A compact text for a game's rules ({field: value} of the [game] section), for run lists and tables.

const MAX_FIELDS = 3;

function isSize(value) {
  return Number.isInteger(value) && value > 0;
}

/** [width, height] if the rules have both as positive integers (board games), else null. */
export function boardSize(rules) {
  return rules && isSize(rules.width) && isSize(rules.height) ? [rules.width, rules.height] : null;
}

/** "8×8" for rules with width and height, otherwise the first few values ("gravity -10 · wind off"), plus any later switch that is on ("hide_velocities on"). */
export function rulesSummary(rules) {
  if (!rules) return "?";
  const size = boardSize(rules);
  if (size) return `${size[0]}×${size[1]}`;
  const entries = Object.entries(rules)
    .filter(([key]) => key !== "name");
  const laterSwitches = entries.slice(MAX_FIELDS).filter(([, value]) => value === true);
  const shown = [...entries.slice(0, MAX_FIELDS), ...laterSwitches]
    .map(([key, value]) => `${key} ${value === null ? "–" : value === true ? "on" : value === false ? "off" : value}`);
  return shown.length ? shown.join(" · ") : "default rules";
}
