// The Compare tab's run list: one collapsible group per model kind with a select-all checkbox, and
// one checkbox row per run.

import { badge, formatSteps, h } from "./dom.js";
import { rulesSummary } from "./rules_summary.js";

function runRow(run, checked, onChange) {
  const id = `cmp-${run.name}`;
  const box = h("input", { type: "checkbox", id, checked: checked.has(run.name) });
  box.addEventListener("change", () => onChange(box.checked));
  const row = h(
    "li",
    { class: "compare-item" },
    box,
    h(
      "label",
      { for: id },
      h("span", { class: "name", text: run.name }),
      h("span", { class: "muted small", text: `${run.algorithm ?? "?"} · ${rulesSummary(run.rules)} · ${formatSteps(run.env_steps)} / ${formatSteps(run.total_env_steps)}` }),
      run.description ? h("span", { class: "run-description", text: run.description, title: run.description }) : null,
    ),
    badge(run.status.state),
  );
  return { row, box };
}

/**
 * A <details> for `group` ({kind, runs}). `checked`: the Set of selected run names (read when drawn);
 * `collapsed`: the Set of collapsed kinds (updated when the viewer toggles a group);
 * `onToggle(names, on)`: select or deselect those run names.
 */
export function groupSection(group, checked, collapsed, onToggle) {
  const all = h("input", { type: "checkbox", "aria-label": `select all ${group.kind} runs` });
  const sync = () => {
    const count = group.runs.filter((run) => checked.has(run.name)).length;
    all.checked = count === group.runs.length;
    all.indeterminate = count > 0 && count < group.runs.length;
  };
  const rows = group.runs.map((run) =>
    runRow(run, checked, (on) => {
      onToggle([run.name], on);
      sync();
    }),
  );
  all.addEventListener("click", (event) => event.stopPropagation()); // ticking must not fold the group
  all.addEventListener("change", () => {
    onToggle(group.runs.map((run) => run.name), all.checked);
    for (const { box } of rows) box.checked = all.checked;
    sync();
  });
  sync();
  const count = group.runs.length;
  const section = h(
    "details",
    { class: "compare-group", open: !collapsed.has(group.kind) },
    h("summary", {}, all, h("span", { class: "compare-group-name", text: group.kind }), h("span", { class: "muted small", text: `${count} ${count === 1 ? "run" : "runs"}` })),
    h("ul", { class: "compare-runs" }, rows.map(({ row }) => row)),
  );
  section.addEventListener("toggle", () => {
    if (section.open) collapsed.delete(group.kind);
    else collapsed.add(group.kind);
  });
  return section;
}
