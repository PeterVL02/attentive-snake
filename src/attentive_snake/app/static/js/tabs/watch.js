// Tab 4: watch a checkpoint play, with the attention heatmap over the board.

import { del, post } from "../api.js";
import { drawBoard } from "../board.js";
import { formatNumber, formatSteps, h, message, replace } from "../dom.js";
import { fillCheckpointSelect, fillRunSelect } from "../run_options.js";

const EPISODE_END_PAUSE_MS = 700;

let els = {};
let session = null; // {id, info}
let playing = false;
let timer = null;
let busy = false;
let scores = [];
let lastEpisode = 0;

function option(value, text) {
  return h("option", { value: String(value), text });
}

function configureControls(info) {
  const view = info.attention_view;
  els.attention.disabled = view === null;
  els.layer.disabled = view === null;
  replace(els.layer, option(-1, "last"), Array.from({ length: info.num_layers }, (_, i) => option(i, `${i + 1} of ${info.num_layers}`)));
  els.frameField.hidden = view !== "frames";
  replace(
    els.frame,
    option(-1, "newest"),
    Array.from({ length: info.history_length }, (_, i) => option(i, i === 0 ? "1 (oldest)" : String(i + 1))),
  );
  els.note.hidden = view !== null;
  els.note.textContent = view === null ? `This model (${info.kind}) has no attention to show; only the game is drawn.` : "";
  els.weightsPanel.hidden = view !== "weights";
  els.sessionInfo.textContent = `${info.kind} · ${info.width}×${info.height} · checkpoint at ${formatSteps(info.env_steps)} env steps`;
}

function renderWeights(weights) {
  if (!weights) {
    replace(els.weights, h("div", { class: "muted small", text: "No weights for this frame." }));
    return;
  }
  const peak = Math.max(...weights, 1e-9);
  replace(
    els.weights,
    weights.map((w, i) =>
      h(
        "div",
        { class: "bar", title: `frame ${i + 1}: ${w.toFixed(3)}` },
        h("span", { text: w.toFixed(2) }),
        h("div", { style: `height:${Math.max(1, (70 * w) / peak)}px` }),
        h("span", { text: i === weights.length - 1 ? "now" : `t-${weights.length - 1 - i}` }),
      ),
    ),
  );
}

function render(frame) {
  drawBoard(els.canvas, frame);
  if (frame.done && frame.episode > lastEpisode) {
    scores.push(frame.score);
    lastEpisode = frame.episode;
  }
  const mean = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null;
  els.score.textContent = String(frame.score);
  els.steps.textContent = String(frame.steps);
  els.episode.textContent = String(frame.episode + (frame.done ? 0 : 1));
  els.mean.textContent = mean === null ? "–" : `${formatNumber(mean, 2)} (${scores.length})`;
  els.caption.textContent = frame.done ? `Episode over: ${frame.event}, score ${frame.score}.` : frame.caption || "";
  if (session.info.attention_view === "weights") renderWeights(frame.frame_weights);
}

async function call(fn) {
  if (busy || !session) return null;
  busy = true;
  try {
    return await fn();
  } catch (error) {
    message(els.status, "error", error.message);
    setPlaying(false);
    if (error.status === 404) session = null;
    return null;
  } finally {
    busy = false;
  }
}

async function stepOnce() {
  const result = await call(() => post(`/api/watch/${session.id}/step`, { count: 1 }));
  if (result) render(result.frames[result.frames.length - 1]);
  return result?.frames.at(-1);
}

function fps() {
  return Number(els.speed.value);
}

async function tick() {
  if (!playing) return;
  const started = performance.now();
  const frame = await stepOnce();
  if (!playing) return;
  const wait = frame?.done ? EPISODE_END_PAUSE_MS : 1000 / fps();
  timer = setTimeout(tick, Math.max(0, wait - (performance.now() - started)));
}

function setPlaying(on) {
  playing = on && session !== null;
  els.play.textContent = playing ? "Pause" : "Play";
  els.play.setAttribute("aria-pressed", String(playing));
  clearTimeout(timer);
  if (playing) tick();
}

async function setOptions(changes) {
  const result = await call(() => post(`/api/watch/${session.id}/options`, changes));
  if (result) {
    message(els.status, "", "");
    render(result.frame);
  }
}

async function closeSession() {
  setPlaying(false);
  if (session) {
    const id = session.id;
    session = null;
    del(`/api/watch/${id}`).catch(() => {}); // it may already have been evicted
  }
}

async function load() {
  await closeSession();
  if (!els.run.value) {
    message(els.status, "error", "Choose a run.");
    return;
  }
  message(els.status, "info", "Loading the checkpoint…");
  els.controls.hidden = true;
  let created;
  try {
    created = await post("/api/watch", {
      run: els.run.value,
      checkpoint: els.checkpoint.value || null,
      sample: els.sample.checked,
      attention: true,
      layer: -1,
      frame: -1,
    });
    session = { id: created.id, info: created.info };
  } catch (error) {
    message(els.status, "error", error.message);
    return;
  }
  message(els.status, "", "");
  scores = [];
  lastEpisode = 0;
  els.attention.checked = session.info.attention_view !== null;
  configureControls(session.info);
  els.controls.hidden = false;
  render(created.frame);
}

async function refreshCheckpoints(selected) {
  try {
    await fillCheckpointSelect(els.checkpoint, els.run.value, selected);
  } catch (error) {
    message(els.status, "error", error.message);
  }
}

function tile(label, valueEl) {
  return h("div", { class: "stat" }, h("div", { class: "k", text: label }), valueEl);
}

export function init(root) {
  const v = () => h("div", { class: "v", text: "–" });
  els = {
    run: h("select", { id: "watch-run" }),
    checkpoint: h("select", { id: "watch-checkpoint" }),
    status: h("div", { class: "message", role: "status" }),
    canvas: h("canvas", { role: "img", "aria-label": "Snake board with attention heatmap" }),
    play: h("button", { type: "button", class: "primary", text: "Play", "aria-pressed": "false", onclick: () => setPlaying(!playing) }),
    step: h("button", { type: "button", text: "Step", onclick: () => { setPlaying(false); stepOnce(); } }),
    reset: h("button", { type: "button", text: "New episode", onclick: () => call(() => post(`/api/watch/${session.id}/reset`)).then((r) => r && render(r.frame)) }),
    speed: h("input", { type: "range", id: "watch-speed", min: "1", max: "30", value: "8" }),
    speedLabel: h("span", { class: "muted small mono", text: "8 fps" }),
    sample: h("input", { type: "checkbox", id: "watch-sample" }),
    attention: h("input", { type: "checkbox", id: "watch-attention", checked: true }),
    layer: h("select", { id: "watch-layer" }),
    frame: h("select", { id: "watch-frame" }),
    note: h("div", { class: "message info", hidden: true }),
    weights: h("div", { class: "weights" }),
    caption: h("div", { class: "muted small", "aria-live": "polite" }),
    sessionInfo: h("div", { class: "muted small" }),
    score: v(),
    steps: v(),
    episode: v(),
    mean: v(),
  };
  els.frameField = h("label", { class: "row", style: "gap:6px" }, "frame", els.frame);
  els.weightsPanel = h("div", { hidden: true }, h("div", { class: "label", text: "attention per frame (oldest → newest)" }), els.weights);
  els.speed.addEventListener("input", () => (els.speedLabel.textContent = `${fps()} fps`));
  els.run.addEventListener("change", () => refreshCheckpoints());
  els.sample.addEventListener("change", () => session && setOptions({ sample: els.sample.checked }));
  els.attention.addEventListener("change", () => setOptions({ attention: els.attention.checked }));
  els.layer.addEventListener("change", () => setOptions({ layer: Number(els.layer.value) }));
  els.frame.addEventListener("change", () => setOptions({ frame: Number(els.frame.value) }));

  els.controls = h(
    "div",
    { class: "columns", hidden: true },
    h("div", { class: "panel" }, h("div", { class: "board-wrap" }, els.canvas), h("div", { style: "margin-top:8px" }, els.caption)),
    h(
      "div",
      {},
      h(
        "div",
        { class: "panel stack" },
        els.sessionInfo,
        h("div", { class: "row" }, els.play, els.step, els.reset),
        h("div", { class: "row" }, h("label", { for: "watch-speed", class: "label", text: "speed" }), els.speed, els.speedLabel),
        h("label", { class: "row", style: "gap:6px" }, els.sample, "sample actions (instead of the most likely one)"),
        h("label", { class: "row", style: "gap:6px" }, els.attention, "show attention"),
        h("div", { class: "row" }, h("label", { class: "row", style: "gap:6px" }, "layer", els.layer), els.frameField),
        els.note,
        els.weightsPanel,
      ),
      h(
        "div",
        { class: "panel" },
        h("div", { class: "readout", style: "margin-top:0" }, tile("score", els.score), tile("steps", els.steps), tile("episode", els.episode), tile("mean score (n)", els.mean)),
      ),
    ),
  );
  replace(
    root,
    h("h2", { text: "Watch & attention" }),
    h("p", { class: "lead", text: "Play a checkpoint greedily or by sampling. Orange shading shows how much the class token attends to each cell when choosing the next move (scaled so the brightest cell is 1)." }),
    h(
      "div",
      { class: "panel" },
      h(
        "div",
        { class: "row" },
        h("label", { for: "watch-run", class: "label", text: "run" }),
        els.run,
        h("label", { for: "watch-checkpoint", class: "label", text: "checkpoint" }),
        els.checkpoint,
        h("button", { type: "button", class: "primary", text: "Load", onclick: load }),
      ),
      h("div", { style: "margin-top:10px" }, els.status),
    ),
    els.controls,
  );
  window.addEventListener("pagehide", () => {
    if (session) fetch(`/api/watch/${session.id}`, { method: "DELETE", keepalive: true });
  });
}

export async function show(params = {}) {
  try {
    await fillRunSelect(els.run, params.run || els.run.value);
    await refreshCheckpoints(params.checkpoint || els.checkpoint.value);
  } catch (error) {
    message(els.status, "error", error.message);
    return;
  }
  if (params.run) load();
}

export function hide() {
  setPlaying(false);
}
