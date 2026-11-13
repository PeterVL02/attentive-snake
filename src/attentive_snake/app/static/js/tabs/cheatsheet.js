// Tab: the CLI cheat sheet (docs/CLI.md) rendered as Markdown.

import { get } from "../api.js";
import { h, replace } from "../dom.js";
import { renderMarkdown } from "../markdown.js";

let root;
let loaded = false;

export function init(element) {
  root = element;
}

export async function show() {
  if (loaded) return;
  replace(root, h("div", { class: "empty", text: "Loading…" }));
  try {
    const text = await get("/api/cheatsheet");
    const article = h("article", { class: "markdown panel" });
    article.innerHTML = renderMarkdown(text); // renderMarkdown escapes all text
    replace(root, article);
    loaded = true;
  } catch (error) {
    replace(root, h("div", { class: "message error", text: `Could not load docs/CLI.md: ${error.message}` }));
  }
}
