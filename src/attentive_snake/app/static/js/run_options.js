// Filling <select> elements with a game's runs and a run's checkpoints (shared by several tabs).

import { get } from "./api.js";
import { formatSteps, h, replace } from "./dom.js";
import { gameUrl } from "./game.js";

export async function fillRunSelect(select, game, selected) {
  const runs = await get(gameUrl("/api/runs", game));
  replace(
    select,
    runs.length === 0 ? h("option", { value: "", text: "No runs yet" }) : null,
    runs.map((r) => h("option", { value: r.name, text: `${r.name}  (${r.kind ?? "?"}, ${r.status.state})` })),
  );
  if (selected && runs.some((r) => r.name === selected)) select.value = selected;
  return runs;
}

/** Fill `select` with `checkpoints` ([{name, steps}], ascending), latest first; keeps `selected` if present. */
export function setCheckpointOptions(select, checkpoints, selected = select.value) {
  const latestFirst = [...checkpoints].reverse();
  replace(
    select,
    latestFirst.length === 0 ? h("option", { value: "", text: "No checkpoints yet" }) : null,
    latestFirst.map((c, i) => h("option", { value: c.name, text: `${formatSteps(c.steps)} steps${i === 0 ? " (latest)" : ""}` })),
  );
  if (latestFirst.some((c) => c.name === selected)) select.value = selected;
  return latestFirst;
}

/** Fills with the checkpoints of `run` of `game`, latest first; returns them. */
export async function fillCheckpointSelect(select, game, run, selected) {
  if (!run) {
    replace(select, h("option", { value: "", text: "—" }));
    return [];
  }
  const detail = await get(gameUrl(`/api/runs/${encodeURIComponent(run)}?log_lines=1`, game));
  return setCheckpointOptions(select, detail.checkpoints, selected);
}
