// "Parameters: 251,777": how many trainable parameters the policy of the form's current config would have.

import { post } from "./api.js";
import { debounce, formatNumber, h } from "./dom.js";
import { gameUrl, isCurrent } from "./game.js";

const DELAY_MS = 300; // after the last edit

/**
 * A line showing the parameter count of the config `readConfig()` returns ({tables, errors}, or null
 * while the form is not ready). Returns {el, update()}; call update() whenever the form may have changed.
 */
export function parameterCount(readConfig) {
  const value = h("strong", { text: "–" });
  const note = h("span", { class: "muted small" });
  const el = h("p", { class: "parameter-count", "aria-live": "polite" }, "Parameters: ", value, " ", note);
  let token = 0; // only the newest request may show its answer

  function show(text, noteText = "") {
    value.textContent = text;
    note.textContent = noteText;
  }

  async function refresh() {
    const mine = ++token;
    const config = readConfig();
    if (!config || config.errors) {
      show("–", config ? "fix the marked fields" : "");
      return;
    }
    const game = config.tables.game.name;
    try {
      const result = await post(gameUrl("/api/parameters", game), { config: config.tables });
      if (mine === token && isCurrent(game)) show(formatNumber(result.parameters, 0));
    } catch (error) {
      if (mine === token && isCurrent(game)) show("–", `not buildable: ${error.message}`);
    }
  }

  const update = debounce(refresh, DELAY_MS);
  return {
    el,
    update() {
      token += 1; // an answer for the previous form state is stale from now on
      show("–");
      update();
    },
  };
}
