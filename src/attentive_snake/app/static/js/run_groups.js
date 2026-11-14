// Grouping runs by model kind. Pure: no DOM.

export const UNKNOWN_KIND = "unknown";

/**
 * [{kind, runs}] for `runs`: one group per kind that has runs, in the order of `kindOrder` (the schema's
 * kind names); runs whose kind is missing or not in `kindOrder` form one last "unknown" group. Runs keep
 * their order within a group.
 */
export function groupRuns(runs, kindOrder) {
  const known = new Set(kindOrder);
  const kindOf = (run) => (known.has(run.kind) ? run.kind : UNKNOWN_KIND);
  return [...kindOrder, UNKNOWN_KIND]
    .map((kind) => ({ kind, runs: runs.filter((run) => kindOf(run) === kind) }))
    .filter((group) => group.runs.length > 0);
}
