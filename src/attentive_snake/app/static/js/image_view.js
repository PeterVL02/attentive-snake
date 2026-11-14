// The "image" view kind: a watched frame's screen picture ({width, height, png: base64}, see
// views/image.py), scaled to fit with its aspect ratio kept and without smoothing when enlarged.

import { h } from "./dom.js";

const MAX_PX = 560;

/** CSS width for a picture `width` pixels wide: whole-number enlargement of small pictures, shrink big ones. */
function displayWidth(width) {
  if (width >= MAX_PX) return MAX_PX;
  return width * Math.max(1, Math.floor(MAX_PX / width));
}

/** The image view for the Watch tab: {el, draw(frame)}. */
export function imageView() {
  const img = h("img", { class: "game-image", alt: "Game screen" });
  let size = "";
  return {
    el: img,
    draw(frame) {
      if (!frame.png) return; // a frame without a picture (e.g. cut down by the server): keep the last one
      const key = `${frame.width}x${frame.height}`;
      if (key !== size) {
        size = key;
        const width = displayWidth(frame.width);
        img.style.width = `${width}px`;
        img.style.imageRendering = width > frame.width ? "pixelated" : "auto";
      }
      img.src = `data:image/png;base64,${frame.png}`;
    },
  };
}
