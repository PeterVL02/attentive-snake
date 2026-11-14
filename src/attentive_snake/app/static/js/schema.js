// The experiment schema and default config of a game, fetched once per game and shared.

import { get } from "./api.js";
import { currentGame, gameUrl } from "./game.js";

const cache = new Map(); // game -> Promise<{schema, defaults}>

export function loadSchema(game = currentGame().name) {
  if (!cache.has(game)) {
    const loading = Promise.all([get(gameUrl("/api/schema", game)), get(gameUrl("/api/defaults", game))]).then(
      ([schema, defaults]) => ({ schema, defaults }),
    );
    loading.catch(() => cache.delete(game)); // allow a retry after an error
    cache.set(game, loading);
  }
  return cache.get(game);
}
