// Filling <select> elements with runs and checkpoints (shared by several tabs).

import { get } from "./api.js";
import { formatSteps, h, replace } from "./dom.js";

export async function fillRunSelect(select, selected) {
  const runs = await get("/api/runs");
  replace(
    select,
    runs.length === 0 ? h("option", { value: "", text: "No runs yet" }) : null,
    runs.map((r) => h("option", { value: r.name, text: `${r.name}  (${r.kind ?? "?"}, ${r.status.state})` })),
  );
  if (selected && runs.some((r) => r.name === selected)) select.value = selected;
  return runs;
}

/** Fills with the run's checkpoints, latest first; returns them. */
export async function fillCheckpointSelect(select, run, selected) {
  if (!run) {
    replace(select, h("option", { value: "", text: "—" }));
    return [];
  }
  const detail = await get(`/api/runs/${encodeURIComponent(run)}?log_lines=1`);
  const checkpoints = [...detail.checkpoints].reverse();
  replace(
    select,
    checkpoints.length === 0 ? h("option", { value: "", text: "No checkpoints yet" }) : null,
    checkpoints.map((c, i) => h("option", { value: c.name, text: `${formatSteps(c.steps)} steps${i === 0 ? " (latest)" : ""}` })),
  );
  if (selected && checkpoints.some((c) => c.name === selected)) select.value = selected;
  return checkpoints;
}
