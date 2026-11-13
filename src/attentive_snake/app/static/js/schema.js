// The experiment schema and default config, fetched once and shared.

import { get } from "./api.js";

let cached = null;

export function loadSchema() {
  cached ??= Promise.all([get("/api/schema"), get("/api/defaults")]).then(([schema, defaults]) => ({
    schema,
    defaults,
  }));
  cached.catch(() => (cached = null)); // allow a retry after an error
  return cached;
}

export function fieldSpec(schema, section, name) {
  return schema.sections[section].find((f) => f.name === name);
}
