// The "board" view kind: a watched frame drawn on a canvas as a grid with the attention heatmap,
// the player body (distinct head) and the target cells (payload keys: see views/board.py), and
// optional named layers of cells in their own colours (views/board_layers.py) with a legend.

import { cssVar, h, replace } from "./dom.js";

const MAX_BOARD_PX = 520;

function roundedRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
  ctx.fill();
}

/** Draw `frame` ({width, height, snake, foods, heatmap}) into `canvas`. */
export function drawBoard(canvas, frame) {
  const cell = Math.max(12, Math.floor(MAX_BOARD_PX / Math.max(frame.width, frame.height)));
  const width = cell * frame.width;
  const height = cell * frame.height;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = cssVar("--board-bg");
  ctx.fillRect(0, 0, width, height);

  if (frame.heatmap) {
    const heat = cssVar("--heat");
    frame.heatmap.forEach((row, y) =>
      row.forEach((value, x) => {
        if (value <= 0) return;
        ctx.fillStyle = `rgba(${heat}, ${(0.08 + 0.82 * value).toFixed(3)})`;
        ctx.fillRect(x * cell, y * cell, cell, cell);
      }),
    );
  }

  ctx.strokeStyle = cssVar("--board-grid");
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 1; x < frame.width; x += 1) {
    ctx.moveTo(x * cell + 0.5, 0);
    ctx.lineTo(x * cell + 0.5, height);
  }
  for (let y = 1; y < frame.height; y += 1) {
    ctx.moveTo(0, y * cell + 0.5);
    ctx.lineTo(width, y * cell + 0.5);
  }
  ctx.stroke();

  // Segments are inset so the heatmap stays visible around them.
  const inset = Math.round(cell * 0.14);
  frame.snake.forEach(([x, y], i) => {
    ctx.fillStyle = i === 0 ? cssVar("--snake-head") : cssVar("--snake");
    roundedRect(ctx, x * cell + inset, y * cell + inset, cell - 2 * inset, cell - 2 * inset, cell * 0.18);
  });
  if (frame.snake.length > 0) {
    const [hx, hy] = frame.snake[0];
    ctx.fillStyle = cssVar("--board-bg");
    ctx.beginPath();
    ctx.arc(hx * cell + cell / 2, hy * cell + cell / 2, cell * 0.11, 0, 2 * Math.PI);
    ctx.fill();
  }
  ctx.fillStyle = cssVar("--food");
  frame.foods.forEach(([fx, fy]) => {
    ctx.beginPath();
    ctx.arc(fx * cell + cell / 2, fy * cell + cell / 2, cell * 0.3, 0, 2 * Math.PI);
    ctx.fill();
  });
  // Layers with fewer cells go on top (a ball over the bricks), each later one a little smaller,
  // so a cell holding several kinds of objects shows them all.
  const layers = [...(frame.layers || [])].sort((a, b) => b.cells.length - a.cells.length);
  layers.forEach((layer, i) => {
    const pad = inset + Math.min(i, 4) * Math.round(cell * 0.05);
    ctx.fillStyle = layer.color;
    layer.cells.forEach(([x, y]) => roundedRect(ctx, x * cell + pad, y * cell + pad, cell - 2 * pad, cell - 2 * pad, cell * 0.12));
  });
}

/** The legend of a frame's layers (none: nothing), in payload order. */
function legendItems(frame) {
  return (frame.layers || []).map((layer) =>
    h("label", {}, h("span", { class: "swatch", style: `background:${layer.color};height:12px` }), layer.name),
  );
}

/** The board view for the Watch tab: {el, draw(frame)}. */
export function boardView() {
  const canvas = h("canvas", { role: "img", "aria-label": "Game board with attention heatmap" });
  const legend = h("div", { class: "legend" });
  let names = "";
  return {
    el: h("div", {}, canvas, legend),
    draw(frame) {
      drawBoard(canvas, frame);
      const shown = (frame.layers || []).map((layer) => layer.name).join(",");
      if (shown !== names) {
        names = shown;
        replace(legend, legendItems(frame));
      }
    },
  };
}
