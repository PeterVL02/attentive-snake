// Tiny DOM helpers. Text is always set via textContent, never parsed as HTML.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (key in el && typeof value !== "string") el[key] = value;
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export function replace(el, ...children) {
  el.replaceChildren();
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export function message(el, kind, text) {
  el.className = `message ${kind}`;
  el.textContent = text || "";
}

export function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  if (typeof value !== "number") return String(value);
  if (Number.isInteger(value)) return value.toLocaleString("en-US");
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return value.toExponential(2);
  return value.toLocaleString("en-US", { maximumFractionDigits: digits });
}

export function formatSteps(value) {
  if (value === null || value === undefined) return "–";
  if (value >= 1e6) return `${(value / 1e6).toLocaleString("en-US", { maximumFractionDigits: 2 })}M`;
  if (value >= 1e3) return `${(value / 1e3).toLocaleString("en-US", { maximumFractionDigits: 1 })}k`;
  return String(value);
}

export function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "–";
  if (seconds < 90) return `${Math.round(seconds)}s`;
  if (seconds < 5400) return `${(seconds / 60).toFixed(1)}m`;
  return `${(seconds / 3600).toFixed(2)}h`;
}

export function badge(state) {
  return h("span", { class: `badge ${state}`, text: state });
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}
