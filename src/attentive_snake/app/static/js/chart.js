// Line charts drawn on a canvas, with a clickable legend and a hover readout.
// Used by the Runs tab (metrics of one run) and the Compare tab (one metric of many runs).

import { cssVar, formatNumber, h, replace } from "./dom.js";

const PALETTE_SIZE = 8;

/** Theme colour for series i (cycles; the dash pattern tells repeats apart). */
export function seriesColor(i) {
  return cssVar(`--series-${(i % PALETTE_SIZE) + 1}`);
}

export function seriesDash(i) {
  return i < PALETTE_SIZE ? [] : [6, 4];
}

/** Trailing moving average over the last `window` points (window <= 1: unchanged). */
export function movingAverage(points, window) {
  if (window <= 1) return points;
  const out = [];
  let sum = 0;
  for (let i = 0; i < points.length; i += 1) {
    sum += points[i][1];
    if (i >= window) sum -= points[i - window][1];
    out.push([points[i][0], sum / Math.min(i + 1, window)]);
  }
  return out;
}

function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const residual = raw / magnitude;
  const nice = residual > 5 ? 10 : residual > 2 ? 5 : residual > 1 ? 2 : 1;
  return nice * magnitude;
}

function ticks(min, max, count) {
  const step = niceStep(max - min || 1, count);
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

function nearestIndex(points, x) {
  let lo = 0;
  let hi = points.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (points[mid][0] < x) lo = mid;
    else hi = mid;
  }
  return Math.abs(points[lo][0] - x) <= Math.abs(points[hi][0] - x) ? lo : hi;
}

/**
 * A chart in `container` (a .chart-wrap element). Series: [{label, color, dash, points: [[x, y]], hidden}].
 * Options: {xFormat(x), yFormat(y), empty: text shown without data}.
 */
export class LineChart {
  constructor(container, options = {}) {
    this.options = { xFormat: formatNumber, yFormat: formatNumber, empty: "No data yet.", ...options };
    this.series = [];
    this.canvas = h("canvas", { role: "img", "aria-label": "line chart" });
    this.tip = h("div", { class: "chart-tip", hidden: true });
    replace(container, this.canvas, this.tip);
    this.hoverX = null;
    this.canvas.addEventListener("mousemove", (e) => this.hover(e));
    this.canvas.addEventListener("mouseleave", () => {
      this.hoverX = null;
      this.tip.hidden = true;
      this.draw();
    });
    new ResizeObserver(() => this.draw()).observe(container);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.draw());
  }

  setSeries(series) {
    this.series = series;
    this.draw();
  }

  visible() {
    return this.series.filter((s) => !s.hidden && s.points.length > 0);
  }

  frame() {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    const pad = { left: 58, right: 16, top: 12, bottom: 30 };
    const all = this.visible().flatMap((s) => s.points);
    let [x0, x1] = [Math.min(...all.map((p) => p[0])), Math.max(...all.map((p) => p[0]))];
    let [y0, y1] = [Math.min(...all.map((p) => p[1])), Math.max(...all.map((p) => p[1]))];
    if (x0 === x1) [x0, x1] = [x0 - 1, x1 + 1];
    if (y0 === y1) [y0, y1] = [y0 - 1, y1 + 1];
    const ypad = (y1 - y0) * 0.06;
    [y0, y1] = [y0 - ypad, y1 + ypad];
    const sx = (x) => pad.left + ((x - x0) / (x1 - x0)) * (width - pad.left - pad.right);
    const sy = (y) => height - pad.bottom - ((y - y0) / (y1 - y0)) * (height - pad.top - pad.bottom);
    const invx = (px) => x0 + ((px - pad.left) / (width - pad.left - pad.right)) * (x1 - x0);
    return { width, height, pad, x0, x1, y0, y1, sx, sy, invx };
  }

  draw() {
    const dpr = window.devicePixelRatio || 1;
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width || !height) return; // hidden tab; redrawn by the ResizeObserver when shown
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.font = `12px ${cssVar("--font")}`;
    const muted = cssVar("--muted");
    const grid = cssVar("--border");
    const series = this.visible();
    if (series.length === 0) {
      ctx.fillStyle = muted;
      ctx.textAlign = "center";
      ctx.fillText(this.options.empty, width / 2, height / 2);
      return;
    }
    const f = this.frame();
    ctx.lineWidth = 1;
    ctx.strokeStyle = grid;
    ctx.fillStyle = muted;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const y of ticks(f.y0, f.y1, 5)) {
      const py = Math.round(f.sy(y)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(f.pad.left, py);
      ctx.lineTo(width - f.pad.right, py);
      ctx.stroke();
      ctx.fillText(this.options.yFormat(y), f.pad.left - 8, py);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const x of ticks(f.x0, f.x1, Math.max(2, Math.floor(width / 110)))) {
      ctx.fillText(this.options.xFormat(x), f.sx(x), height - f.pad.bottom + 8);
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(f.pad.left, f.pad.top, width - f.pad.left - f.pad.right, height - f.pad.top - f.pad.bottom);
    ctx.clip();
    for (const s of series) {
      ctx.strokeStyle = s.color;
      ctx.setLineDash(s.dash || []);
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.beginPath();
      s.points.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(f.sx(x), f.sy(y)) : ctx.lineTo(f.sx(x), f.sy(y))));
      ctx.stroke();
      if (s.points.length === 1) {
        ctx.fillStyle = s.color;
        ctx.beginPath();
        ctx.arc(f.sx(s.points[0][0]), f.sy(s.points[0][1]), 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
    ctx.restore();
    ctx.setLineDash([]);
    if (this.hoverX !== null) this.drawHover(ctx, f, series);
  }

  hover(event) {
    if (this.visible().length === 0) return;
    const rect = this.canvas.getBoundingClientRect();
    this.hoverX = this.frame().invx(event.clientX - rect.left);
    this.draw();
  }

  drawHover(ctx, f, series) {
    const rows = [];
    for (const s of series) {
      const [x, y] = s.points[nearestIndex(s.points, this.hoverX)];
      // Only runs that reach this far: a short run shows nothing past its end.
      if (Math.abs(f.sx(x) - f.sx(this.hoverX)) > 40) continue;
      rows.push({ s, x, y });
      ctx.fillStyle = s.color;
      ctx.beginPath();
      ctx.arc(f.sx(x), f.sy(y), 3.5, 0, 2 * Math.PI);
      ctx.fill();
    }
    const px = f.sx(this.hoverX);
    ctx.strokeStyle = cssVar("--muted");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px + 0.5, f.pad.top);
    ctx.lineTo(px + 0.5, f.height - f.pad.bottom);
    ctx.stroke();
    if (rows.length === 0) {
      this.tip.hidden = true;
      return;
    }
    replace(
      this.tip,
      h("div", { class: "muted", text: this.options.xFormat(rows[0].x) }),
      rows.map(({ s, y }) =>
        h("div", { class: "tip-row" }, h("span", { class: "swatch", style: `background:${s.color}` }), `${s.label}: ${this.options.yFormat(y)}`),
      ),
    );
    this.tip.hidden = false;
    const tipWidth = this.tip.offsetWidth;
    const left = px + 12 + tipWidth > f.width ? px - 12 - tipWidth : px + 12;
    this.tip.style.left = `${Math.max(0, left)}px`;
    this.tip.style.top = `${f.pad.top}px`;
  }
}

/** Legend buttons that toggle series visibility; calls onToggle() after a change. */
export function renderLegend(container, series, onToggle) {
  replace(
    container,
    series.map((s) =>
      h(
        "button",
        {
          type: "button",
          "aria-pressed": String(!s.hidden),
          onclick: (e) => {
            s.hidden = !s.hidden;
            e.currentTarget.setAttribute("aria-pressed", String(!s.hidden));
            onToggle();
          },
        },
        h("span", { class: "swatch", style: `background:${s.color}` }),
        s.label,
      ),
    ),
  );
}
