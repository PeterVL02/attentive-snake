// Cross-tab navigation and shared providers, so tabs do not import each other.

const handlers = { navigate: null };
const providers = {};

export function onNavigate(handler) {
  handlers.navigate = handler;
}

export function navigate(tab, params = {}) {
  handlers.navigate?.(tab, params);
}

export function provide(name, fn) {
  providers[name] = fn;
}

export function use(name) {
  return providers[name]?.();
}
