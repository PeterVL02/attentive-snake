// The experiment schema and default config of a game, fetched once per game and shared; the game's
// defaults for another algorithm (laid over default.toml by the server) are fetched once per pair.

import { get } from "./api.js";
import { currentGame, gameUrl } from "./game.js";

const cache = new Map(); // game -> Promise<{schema, defaults}>
const algorithmDefaults = new Map(); // "game/algorithm" -> Promise<defaults>

function cached(map, key, load) {
  if (!map.has(key)) {
    const loading = load();
    loading.catch(() => map.delete(key)); // allow a retry after an error
    map.set(key, loading);
  }
  return map.get(key);
}

export function loadSchema(game = currentGame().name) {
  return cached(cache, game, () =>
    Promise.all([get(gameUrl("/api/schema", game)), get(gameUrl("/api/defaults", game))]).then(([schema, defaults]) => ({ schema, defaults })),
  );
}

/** The game's default experiment for `algorithm` (its own section and the shared values that suit it). */
export function loadAlgorithmDefaults(game, algorithm) {
  return cached(algorithmDefaults, `${game}/${algorithm}`, () => get(`${gameUrl("/api/defaults", game)}&algorithm=${encodeURIComponent(algorithm)}`));
}
