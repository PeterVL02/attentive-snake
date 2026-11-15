// Before a run starts: the game's warnings about the config, in a dialog the owner can confirm or go back from.

import { post } from "./api.js";
import { h } from "./dom.js";
import { gameUrl } from "./game.js";

/** The game's warnings (strings) about the config `tables`; an empty list if there are none or they cannot be fetched. */
async function fetchWarnings(tables, game) {
  try {
    const result = await post(gameUrl("/api/training-warnings", game), { config: tables });
    return result.warnings;
  } catch {
    return []; // a failed check must never keep a run from starting
  }
}

/** Show `warnings` in a modal dialog; resolves true for "Start anyway", false for "Go back" (or Escape). */
function askToProceed(warnings) {
  return new Promise((resolve) => {
    let proceed = false;
    const startAnyway = h("button", { type: "button", class: "primary", text: "Start anyway", onclick: () => { proceed = true; dialog.close(); } });
    const goBack = h("button", { type: "button", text: "Go back", onclick: () => dialog.close() });
    const dialog = h(
      "dialog",
      { class: "warning-dialog", "aria-labelledby": "warning-title" },
      h(
        "div",
        { class: "dialog-body" },
        h("h2", { id: "warning-title", text: "Before you start" }),
        warnings.map((text) => h("p", { text })),
        h("div", { class: "row dialog-actions" }, startAnyway, goBack),
      ),
    );
    dialog.addEventListener("close", () => {
      dialog.remove();
      resolve(proceed);
    });
    document.body.append(dialog);
    dialog.showModal();
  });
}

/** True if the run may start: the game has no warnings about `tables`, or the owner chose "Start anyway". */
export async function confirmWarnings(tables, game) {
  const warnings = await fetchWarnings(tables, game);
  return warnings.length === 0 || (await askToProceed(warnings));
}
