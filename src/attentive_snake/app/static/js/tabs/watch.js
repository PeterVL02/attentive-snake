// Tab 4: watch a checkpoint of the selected game play, with the attention it pays: a heatmap over a
// board, weights per history frame, or weights per observation element.

import { del, get, post } from "../api.js";
import { elementWeightBars, frameWeightBars } from "../attention_bars.js";
import { boardView } from "../board.js";
import { formatNumber, formatSteps, h, message, replace } from "../dom.js";
import { formatMetric, headlineDefinition, metricDefinitions } from "../evaluation_metrics.js";
import { currentGame, gameUrl, isCurrent } from "../game.js";
import { imageView } from "../image_view.js";
import { fillCheckpointSelect, fillRunSelect } from "../run_options.js";

const EPISODE_END_PAUSE_MS = 700;
const VIEWS = { board: boardView, image: imageView };
const STEPS_PER_TICK = [1, 2, 5, 10, 25];

let els = {};
let session = null; // {id, info, game}
let view = null; // {el, draw(frame)} for the session's view kind
let playing = false;
let timer = null;
let busy = false;
let loadToken = 0; // bumped by every load and reset; a load that finds it changed is dropped
let scores = [];
let lastEpisode = 0;
let showFrames = false;
let showElements = false;

function option(value, text) {
  return h("option", { value: String(value), text });
}

/** Element weights are expected for vector observations of a model with attention (the panel stays put while they come and go). */
function expectsElements(info) {
  return info.attention_view !== null && (Array.isArray(info.element_names) || currentGame().observation_kind === "vector");
}

function attentionNote(info) {
  if (info.attention_view === null) return `This model (${info.kind}) has no attention to show; only the game is drawn.`;
  const cellHeatmap = info.attention_view === "grid" || info.attention_view === "frames";
  if (cellHeatmap && info.view_kind !== "board" && !expectsElements(info)) return "This model's attention is a heatmap over the observation's cells, which cannot be laid over this game's picture.";
  return "";
}

function configureControls(info) {
  const attention = info.attention_view;
  els.attention.disabled = attention === null;
  els.layer.disabled = attention === null;
  replace(els.layer, option(-1, "last"), Array.from({ length: info.num_layers }, (_, i) => option(i, `${i + 1} of ${info.num_layers}`)));
  els.frameField.hidden = attention !== "frames";
  replace(
    els.frame,
    option(-1, "newest"),
    Array.from({ length: info.history_length }, (_, i) => option(i, i === 0 ? "1 (oldest)" : String(i + 1))),
  );
  const note = attentionNote(info);
  els.note.hidden = !note;
  els.note.textContent = note;
  showFrames = attention === "weights";
  showElements = expectsElements(info);
  els.sessionInfo.textContent = `${info.kind} · observation ${info.observation_shape.join("×")} · checkpoint at ${formatSteps(info.env_steps)} env steps`;
}

function renderAttention(frame) {
  showFrames ||= Array.isArray(frame.frame_weights);
  showElements ||= Array.isArray(frame.element_weights);
  els.framesPanel.hidden = !showFrames;
  els.elementsPanel.hidden = !showElements;
  if (showFrames) replace(els.frameWeights, frameWeightBars(frame.frame_weights));
  if (showElements) replace(els.elementWeights, elementWeightBars(frame.element_weights, session.info.element_names));
}

/** Count finished episodes in `frames` (all of a step call) and draw the last one. */
function render(frames) {
  for (const f of frames) {
    if (f.done && f.episode > lastEpisode) {
      scores.push(f.score);
      lastEpisode = f.episode;
    }
  }
  const frame = frames[frames.length - 1];
  view.draw(frame);
  const mean = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null;
  els.score.textContent = formatNumber(frame.score, 2);
  els.steps.textContent = String(frame.steps);
  els.episode.textContent = String(frame.episode + (frame.done ? 0 : 1));
  els.mean.textContent = mean === null ? "–" : `${formatNumber(mean, 2)} (${scores.length})`;
  els.caption.textContent = frame.done ? `Episode over: ${frame.event}, score ${formatNumber(frame.score, 2)}.` : frame.caption || "";
  renderAttention(frame);
}

/** Run `fn` against the open session; the result is dropped if the session changed meanwhile. */
async function call(fn) {
  if (busy || !session) return null;
  const used = session;
  busy = true;
  try {
    const result = await fn(used);
    return session === used ? result : null;
  } catch (error) {
    if (session !== used) return null;
    message(els.status, "error", error.message);
    setPlaying(false);
    if (error.status === 404) session = null;
    return null;
  } finally {
    busy = false;
  }
}

async function step(count) {
  const result = await call((s) => post(`/api/watch/${s.id}/step`, { count, frames: "last" }));
  if (result) render(result.frames);
  return result?.frames.at(-1);
}

function fps() {
  return Number(els.speed.value);
}

async function tick() {
  if (!playing) return;
  const started = performance.now();
  const frame = await step(Number(els.perTick.value));
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
  const result = await call((s) => post(`/api/watch/${s.id}/options`, changes));
  if (result) {
    message(els.status, "", "");
    render([result.frame]);
  }
}

function closeSession() {
  setPlaying(false);
  if (session) {
    const id = session.id;
    session = null;
    del(`/api/watch/${id}`).catch(() => {}); // it may already have been evicted
  }
}

async function load() {
  const token = ++loadToken;
  const game = currentGame().name;
  closeSession();
  if (!els.run.value) {
    message(els.status, "error", "Choose a run.");
    return;
  }
  message(els.status, "info", "Loading the checkpoint…");
  els.controls.hidden = true;
  let created;
  try {
    created = await post("/api/watch", {
      game,
      run: els.run.value,
      checkpoint: els.checkpoint.value || null,
      sample: els.sample.checked,
      attention: true,
      layer: -1,
      frame: -1,
    });
  } catch (error) {
    if (token === loadToken) message(els.status, "error", error.message);
    return;
  }
  if (token !== loadToken || !isCurrent(game)) {
    del(`/api/watch/${created.id}`).catch(() => {}); // superseded by a newer load or another game
    return;
  }
  session = { id: created.id, info: created.info, game };
  const makeView = VIEWS[created.info.view_kind];
  if (!makeView) {
    message(els.status, "error", `This page cannot draw the view kind “${created.info.view_kind}”.`);
    closeSession();
    return;
  }
  view = makeView();
  replace(els.viewWrap, view.el);
  message(els.status, "", "");
  scores = [];
  lastEpisode = 0;
  els.attention.checked = created.info.attention_view !== null;
  configureControls(created.info);
  els.controls.hidden = false;
  render([created.frame]);
  showEvaluation(game, els.run.value, created.info.checkpoint, token);
}

/** "Recorded evaluation of this checkpoint (100 greedy games): win rate 12.0 % · mean score 21.4", or a hint. */
function evaluationText(evaluations, checkpoint) {
  const evaluation = evaluations.filter((e) => e.checkpoint === checkpoint).at(-1);
  if (!evaluation) return "No recorded evaluation of this checkpoint (the Runs tab can run one).";
  const shown = [headlineDefinition(), metricDefinitions().find((d) => d.key === "mean_score")].filter((d, i, all) => d && all.indexOf(d) === i);
  const values = shown.map((d) => `${d.label} ${formatMetric(d, evaluation.statistics[d.key])}`).join(" · ");
  return `Recorded evaluation of this checkpoint (${formatNumber(evaluation.settings.games)} greedy games): ${values}`;
}

/** Fill in the recorded evaluation of the session's checkpoint, unless another load started meanwhile. */
async function showEvaluation(game, run, checkpointPath, token) {
  els.evaluation.textContent = "";
  try {
    const detail = await get(gameUrl(`/api/runs/${encodeURIComponent(run)}?log_lines=1`, game));
    if (token === loadToken) els.evaluation.textContent = evaluationText(detail.evaluations, checkpointPath.split(/[\\/]/).pop());
  } catch {
    /* only a hint: leave it empty */
  }
}

async function refreshCheckpoints(selected) {
  const game = currentGame().name;
  try {
    await fillCheckpointSelect(els.checkpoint, game, els.run.value, selected);
  } catch (error) {
    if (isCurrent(game)) message(els.status, "error", error.message);
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
    viewWrap: h("div", { class: "board-wrap" }),
    play: h("button", { type: "button", class: "primary", text: "Play", "aria-pressed": "false", onclick: () => setPlaying(!playing) }),
    step: h("button", { type: "button", text: "Step", onclick: () => { setPlaying(false); step(1); } }),
    reset: h("button", { type: "button", text: "New episode", onclick: () => call((s) => post(`/api/watch/${s.id}/reset`)).then((r) => r && render([r.frame])) }),
    speed: h("input", { type: "range", id: "watch-speed", min: "1", max: "30", value: "8" }),
    speedLabel: h("span", { class: "muted small mono", text: "8 fps" }),
    perTick: h("select", { id: "watch-per-tick" }, STEPS_PER_TICK.map((n) => option(n, n === 1 ? "1 (every step)" : `${n} (skip ahead)`))),
    sample: h("input", { type: "checkbox", id: "watch-sample" }),
    attention: h("input", { type: "checkbox", id: "watch-attention", checked: true }),
    layer: h("select", { id: "watch-layer" }),
    frame: h("select", { id: "watch-frame" }),
    note: h("div", { class: "message info", hidden: true }),
    frameWeights: h("div", {}),
    elementWeights: h("div", {}),
    caption: h("div", { class: "muted small", "aria-live": "polite" }),
    sessionInfo: h("div", { class: "muted small" }),
    evaluation: h("div", { class: "muted small" }),
    score: v(),
    steps: v(),
    episode: v(),
    mean: v(),
  };
  els.frameField = h("label", { class: "row", style: "gap:6px" }, "frame", els.frame);
  els.framesPanel = h("div", { hidden: true }, h("div", { class: "label", text: "attention per frame (oldest → newest)" }), els.frameWeights);
  els.elementsPanel = h(
    "div",
    { class: "panel", hidden: true },
    h("h3", { text: "Attention per observation element" }),
    h("p", { class: "muted small", text: "How much the model attends to each number of the observation when choosing the next action (scaled so the largest is a full bar)." }),
    els.elementWeights,
  );
  els.speed.addEventListener("input", () => (els.speedLabel.textContent = `${fps()} fps`));
  els.run.addEventListener("change", () => refreshCheckpoints());
  els.sample.addEventListener("change", () => session && setOptions({ sample: els.sample.checked }));
  els.attention.addEventListener("change", () => setOptions({ attention: els.attention.checked }));
  els.layer.addEventListener("change", () => setOptions({ layer: Number(els.layer.value) }));
  els.frame.addEventListener("change", () => setOptions({ frame: Number(els.frame.value) }));

  els.controls = h(
    "div",
    { class: "columns", hidden: true },
    h(
      "div",
      {},
      h("div", { class: "panel" }, els.viewWrap, h("div", { style: "margin-top:8px" }, els.caption)),
      els.elementsPanel,
    ),
    h(
      "div",
      {},
      h(
        "div",
        { class: "panel stack" },
        els.sessionInfo,
        els.evaluation,
        h("div", { class: "row" }, els.play, els.step, els.reset),
        h("div", { class: "row" }, h("label", { for: "watch-speed", class: "label", text: "speed" }), els.speed, els.speedLabel),
        h("div", { class: "row" }, h("label", { for: "watch-per-tick", class: "label", text: "steps per update" }), els.perTick),
        h("label", { class: "row", style: "gap:6px" }, els.sample, "sample actions (instead of the most likely one)"),
        h("label", { class: "row", style: "gap:6px" }, els.attention, "show attention"),
        h("div", { class: "row" }, h("label", { class: "row", style: "gap:6px" }, "layer", els.layer), els.frameField),
        els.note,
        els.framesPanel,
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
    h("p", {
      class: "lead",
      text: "Play a checkpoint greedily or by sampling. Attention shows what the class token looks at when choosing the next action: orange shading over a board's cells, or bars per history frame or per observation number (scaled so the largest is 1).",
    }),
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
  const game = currentGame().name;
  try {
    await fillRunSelect(els.run, game, params.run || els.run.value);
    if (!isCurrent(game)) return;
    await refreshCheckpoints(params.checkpoint || els.checkpoint.value);
  } catch (error) {
    if (isCurrent(game)) message(els.status, "error", error.message);
    return;
  }
  if (params.run && isCurrent(game)) load();
}

export function hide() {
  setPlaying(false);
}

export function reset() {
  loadToken += 1;
  closeSession();
  view = null;
  els.controls.hidden = true;
  replace(els.viewWrap);
  replace(els.run);
  replace(els.checkpoint);
  message(els.status, "", "");
}
