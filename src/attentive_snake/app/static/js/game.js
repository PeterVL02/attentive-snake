// The selected game: the list from /api/games, which one is selected (remembered in localStorage),
// who to tell when it changes, and per-game API URLs. Every tab works within the selected game.

import { get } from "./api.js";

const STORAGE_KEY = "attentive-snake.game";
const FALLBACK = "snake";

let games = [];
let current = null;
const listeners = [];

function remembered() {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null; // storage blocked: start with the fallback
  }
}

function remember(name) {
  try {
    localStorage.setItem(STORAGE_KEY, name);
  } catch {
    /* not remembered, still selected */
  }
}

/** Fetch the games and select the remembered one (or snake, or the first). */
export async function loadGames() {
  games = await get("/api/games");
  if (games.length === 0) throw new Error("The server knows no games.");
  const byName = (name) => games.find((g) => g.name === name);
  current = byName(remembered()) ?? byName(FALLBACK) ?? games[0];
  return games;
}

export function listGames() {
  return games;
}

/** The selected game's info: {name, title, description, observation_kind, view_kind, has_reward_table}. */
export function currentGame() {
  return current;
}

/** True while `name` is still the selected game (an async result for another game is stale). */
export function isCurrent(name) {
  return current?.name === name;
}

export function selectGame(name) {
  const game = games.find((g) => g.name === name);
  if (!game || game === current) return;
  current = game;
  remember(name);
  for (const listener of listeners) listener(game);
}

export function onGameChange(listener) {
  listeners.push(listener);
}

/** `url` with `?game=` (or `&game=`) for `game` (default: the selected one). */
export function gameUrl(url, game = current.name) {
  return `${url}${url.includes("?") ? "&" : "?"}game=${encodeURIComponent(game)}`;
}
