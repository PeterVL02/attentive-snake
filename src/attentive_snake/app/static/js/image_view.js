// The "image" view kind: a watched frame's screen picture ({width, height, png: base64}, see
// views/image.py), scaled to fit with its aspect ratio kept and without smoothing when enlarged.
// A frame's attention heatmap (rows of weights in 0..1, any number of rows and columns: one cell
// per token of a model that reads the picture) is stretched over the whole picture.

import { cssVar, h } from "./dom.js";

const MAX_PX = 560;

/** CSS width for a picture `width` pixels wide: whole-number enlargement of small pictures, shrink big ones. */
function displayWidth(width) {
  if (width >= MAX_PX) return MAX_PX;
  return width * Math.max(1, Math.floor(MAX_PX / width));
}

/** Paint `heatmap` into `canvas`, one canvas pixel per cell (CSS stretches it over the picture); none: clear it. */
function drawHeat(canvas, heatmap) {
  const rows = heatmap ? heatmap.length : 0;
  const columns = rows ? heatmap[0].length : 0;
  canvas.hidden = !rows || !columns;
  if (canvas.hidden) return;
  canvas.width = columns;
  canvas.height = rows;
  const ctx = canvas.getContext("2d");
  const heat = cssVar("--heat");
  heatmap.forEach((row, y) =>
    row.forEach((value, x) => {
      if (value <= 0) return;
      ctx.fillStyle = `rgba(${heat}, ${(0.7 * value).toFixed(3)})`;
      ctx.fillRect(x, y, 1, 1);
    }),
  );
}

/** The image view for the Watch tab: {el, draw(frame)}. */
export function imageView() {
  const img = h("img", { class: "game-image", alt: "Game screen" });
  const heat = h("canvas", { class: "game-image-heat" });
  heat.hidden = true;
  const el = h("div", { class: "game-image-wrap" }, img, heat);
  let size = "";
  return {
    el,
    draw(frame) {
      drawHeat(heat, frame.heatmap);
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
