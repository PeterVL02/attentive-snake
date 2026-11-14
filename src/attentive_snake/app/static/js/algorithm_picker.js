// The training-algorithm dropdown of a game's schema (schema.algorithms): label per algorithm, with the
// selected one's description below.

import { h } from "./dom.js";

/** {el, select, value(), set(name), onChange(fn)} for `algorithms` (schema.algorithms). */
export function algorithmPicker(algorithms, id = "algorithm-select") {
  const select = h(
    "select",
    { id },
    Object.entries(algorithms).map(([name, info]) => h("option", { value: name, text: `${info.label} (${name})` })),
  );
  const description = h("p", { class: "kind-description" });
  const describe = () => (description.textContent = algorithms[select.value]?.description ?? "");
  select.addEventListener("change", describe);
  const el = h("div", {}, h("div", { class: "field" }, h("label", { for: id, text: "algorithm" }), select), description);
  return {
    el,
    select,
    value: () => select.value,
    set(name) {
      if (name in algorithms) select.value = name;
      describe();
    },
    onChange(fn) {
      select.addEventListener("change", fn);
    },
  };
}
