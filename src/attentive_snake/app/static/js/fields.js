// One labelled input per schema field: type-appropriate control, description, client-side check, error slot.

import { h } from "./dom.js";

let nextId = 0;

function controlFor(spec, id) {
  if (spec.type === "bool") return h("input", { type: "checkbox", id });
  if (spec.choices) {
    return h("select", { id }, spec.choices.map((c) => h("option", { value: c, text: c })));
  }
  if (spec.type === "int" || spec.type === "float") {
    return h("input", { type: "number", id, step: spec.type === "int" ? "1" : "any", inputmode: "decimal" });
  }
  return h("input", { type: "text", id, spellcheck: "false" });
}

function parse(spec, input) {
  if (spec.type === "bool") return { value: input.checked };
  const raw = input.value.trim();
  if (raw === "") {
    return spec.nullable ? { value: null } : { error: "Required." };
  }
  if (spec.type === "int") {
    const n = Number(raw);
    return Number.isInteger(n) ? { value: n } : { error: "Must be a whole number." };
  }
  if (spec.type === "float") {
    const n = Number(raw);
    return Number.isFinite(n) ? { value: n } : { error: "Must be a number." };
  }
  return { value: raw };
}

/** A field widget: {el, spec, get() -> {value} | {error}, set(value), setError(text), onChange(fn)}. */
export function fieldControl(spec, value = spec.default) {
  const id = `f${nextId++}-${spec.name}`;
  const input = controlFor(spec, id);
  const error = h("div", { class: "error", "aria-live": "polite" });
  const descriptionId = `${id}-desc`;
  input.setAttribute("aria-describedby", descriptionId);
  const control = h("div", { class: "control" }, input);
  if (spec.nullable && spec.type !== "bool") {
    input.placeholder = "None (empty)";
    control.append(
      h("button", {
        type: "button",
        class: "clear",
        title: "Clear (None)",
        "aria-label": `Clear ${spec.name}`,
        text: "×",
        onclick: () => {
          input.value = "";
          input.dispatchEvent(new Event("input", { bubbles: true }));
        },
      }),
    );
  }
  const el = h(
    "div",
    { class: `field${spec.type === "bool" ? " checkbox" : ""}` },
    h("label", { for: id, text: spec.name }),
    control,
    h("div", { class: "description", id: descriptionId, text: spec.description }),
    error,
  );
  const widget = {
    el,
    spec,
    input,
    get() {
      return parse(spec, input);
    },
    set(v) {
      if (spec.type === "bool") input.checked = Boolean(v);
      else input.value = v === null || v === undefined ? "" : String(v);
    },
    setError(text) {
      error.textContent = text || "";
      el.classList.toggle("invalid", Boolean(text));
    },
    onChange(fn) {
      input.addEventListener(spec.type === "bool" || input.tagName === "SELECT" ? "change" : "input", fn);
    },
  };
  widget.set(value);
  widget.onChange(() => widget.setError(""));
  return widget;
}

/** Read every widget into {name: value}; returns {values, errors} with per-field errors shown. */
export function readFields(widgets) {
  const values = {};
  let errors = 0;
  for (const widget of widgets) {
    const result = widget.get();
    widget.setError(result.error);
    if (result.error) errors += 1;
    else values[widget.spec.name] = result.value;
  }
  return { values, errors };
}

/** Show a server error next to the field it names (if any); returns true if one was found. */
export function attachServerError(widgets, text) {
  const match = widgets.find((w) => new RegExp(`\\b${w.spec.name}\\b`).test(text));
  if (match) match.setError(text);
  return Boolean(match);
}

/** An optional free-text note saved with the run (run.json): {el, get() -> string | null, set(text)}. */
export function descriptionField(help = "Optional. Saved with the run's history (run.json) and shown in the Runs and Compare tabs.") {
  const id = `f${nextId++}-description`;
  const input = h("textarea", { id, rows: "2", maxlength: "2000", placeholder: "What is this run trying out?", "aria-describedby": `${id}-desc` });
  const el = h(
    "div",
    { class: "field description-field" },
    h("label", { for: id, text: "description" }),
    input,
    h("div", { class: "description", id: `${id}-desc`, text: help }),
  );
  return {
    el,
    get() {
      return input.value.trim() || null;
    },
    set(text) {
      input.value = text || "";
    },
  };
}
