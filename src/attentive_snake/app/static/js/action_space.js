// Whether the form's game rules give continuous actions, asked from the server (the game decides; nothing here knows one).

import { post } from "./api.js";
import { debounce } from "./dom.js";
import { gameUrl, isCurrent } from "./game.js";

const DELAY_MS = 300; // after the last edit of a rule

/**
 * Asks /api/action-space about the rules `readRules()` returns and calls `onAnswer(continuous)` with the newest
 * answer for `gameName`. A failed request (e.g. a half-typed rule) keeps the previous answer: starting the run still
 * reports any problem. Returns {update()}; call update() whenever the rules may have changed.
 */
export function actionSpaceWatcher(gameName, readRules, onAnswer) {
  let token = 0; // only the newest request may answer

  async function ask() {
    const mine = ++token;
    try {
      const { continuous } = await post(gameUrl("/api/action-space", gameName), { rules: readRules() });
      if (mine === token && isCurrent(gameName)) onAnswer(continuous);
    } catch {
      /* keep the previous answer */
    }
  }

  return { update: debounce(ask, DELAY_MS) };
}
