// The model-kind dropdown of a game's schema: every kind the schema lists, recommended ones first,
// the others marked "not recommended for this game" (still selectable), with the kind's description.

import { h } from "./dom.js";

/** Kind names in schema order, those with `recommended === false` last. */
export function orderedKinds(modelKinds) {
  const kinds = Object.keys(modelKinds);
  return [...kinds.filter((k) => modelKinds[k].recommended !== false), ...kinds.filter((k) => modelKinds[k].recommended === false)];
}

/**
 * {el, select, value(), set(kind), onChange(fn)} for `modelKinds` (schema.model_kinds).
 * `el` holds the labelled select and the selected kind's description.
 */
export function modelPicker(modelKinds, id = "kind-select") {
  const select = h(
    "select",
    { id },
    orderedKinds(modelKinds).map((kind) =>
      h("option", { value: kind, text: modelKinds[kind].recommended === false ? `${kind} (not recommended for this game)` : kind }),
    ),
  );
  const description = h("p", { class: "kind-description" });
  const warning = h("div", { class: "message warn", role: "status" });
  const describe = () => {
    const info = modelKinds[select.value];
    description.textContent = info?.description ?? "";
    warning.textContent =
      info?.recommended === false
        ? "Not recommended for this game: it was not built for this kind of observation. You can still train it."
        : "";
  };
  select.addEventListener("change", describe);
  const el = h("div", {}, h("div", { class: "field" }, h("label", { for: id, text: "kind" }), select), description, warning);
  return {
    el,
    select,
    value: () => select.value,
    set(kind) {
      if (kind in modelKinds) select.value = kind;
      describe();
    },
    onChange(fn) {
      select.addEventListener("change", fn);
    },
  };
}
