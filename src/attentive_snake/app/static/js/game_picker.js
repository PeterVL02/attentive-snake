// The game selector in the header: a dropdown of the games and the selected game's description.

import { h, replace } from "./dom.js";
import { currentGame, listGames, onGameChange, selectGame } from "./game.js";

/** Fill `container` with the selector; it follows (and drives) the selected game. */
export function renderGamePicker(container) {
  const select = h(
    "select",
    { id: "game-select", "aria-describedby": "game-description" },
    listGames().map((g) => h("option", { value: g.name, text: g.title })),
  );
  const description = h("span", { id: "game-description", class: "game-description" });
  const show = (game) => {
    select.value = game.name;
    description.textContent = game.description;
    description.title = game.description;
  };
  select.addEventListener("change", () => selectGame(select.value));
  onGameChange(show);
  show(currentGame());
  replace(container, h("label", { for: "game-select", class: "label", text: "game" }), select, description);
}
