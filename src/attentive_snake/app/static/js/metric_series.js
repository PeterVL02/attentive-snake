// Turning a run's metrics records into chart points.

/**
 * Records in increasing env_steps. A resumed run restarts from its last checkpoint, so records
 * logged after that checkpoint by the interrupted job are superseded by the newer ones.
 */
export function monotonic(records) {
  const out = [];
  for (const record of records) {
    if (typeof record.env_steps !== "number") continue;
    while (out.length && out[out.length - 1].env_steps >= record.env_steps) out.pop();
    out.push(record);
  }
  return out;
}

/**
 * Estimated training seconds at each record: the env steps since the previous record divided by
 * that record's steps_per_second. Null if a record lacks steps_per_second.
 */
export function estimatedSeconds(records) {
  let total = 0;
  let previous = 0;
  const out = [];
  for (const record of records) {
    if (!(record.steps_per_second > 0)) return null;
    total += Math.max(0, record.env_steps - previous) / record.steps_per_second;
    previous = record.env_steps;
    out.push(total);
  }
  return out;
}

/** [[x, y]] for `key`, with x = env steps or (xs given) the matching entry of xs. */
export function points(records, key, xs = null) {
  const out = [];
  records.forEach((record, i) => {
    if (typeof record[key] === "number") out.push([xs ? xs[i] : record.env_steps, record[key]]);
  });
  return out;
}

export function median(values) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * The estimated training seconds at env step `steps`, read off the training records and their `seconds`
 * (from estimatedSeconds): linear between the records around it; null outside them.
 */
export function secondsAtSteps(records, seconds, steps) {
  for (let i = 0; i < records.length; i += 1) {
    if (records[i].env_steps < steps) continue;
    if (records[i].env_steps === steps || i === 0) return records[i].env_steps === steps ? seconds[i] : null;
    const [s0, s1] = [records[i - 1].env_steps, records[i].env_steps];
    return seconds[i - 1] + ((steps - s0) / (s1 - s0)) * (seconds[i] - seconds[i - 1]);
  }
  return null;
}
