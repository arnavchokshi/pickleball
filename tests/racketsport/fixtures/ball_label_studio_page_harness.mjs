// Headless harness for the ball-label studio page.
//
// Chrome/Playwright are not usable in this environment and installing browser
// tooling is out of scope, so the page's script is executed in Node against a
// minimal DOM + canvas stub and a REAL running studio server. That is enough to
// catch what a smoke test is for: a boot that throws, a missing element id, a
// bad shape assumption about the /api/state payload, a draw path that blows up
// on real data, or a broken save round trip.
//
// Usage: node ball_label_studio_page_harness.mjs <base-url>
// Prints one JSON object on stdout; exits non-zero if anything threw.
//
// The script is taken from the page the server actually serves, so the harness
// exercises the real per-session save token rather than the placeholder.

import vm from "node:vm";

const baseUrl = process.argv[2];
if (!baseUrl) {
  console.error("usage: harness <base-url>");
  process.exit(2);
}

const errors = [];
const consoleErrors = [];

// ---------------------------------------------------------------- DOM stubs
const CANVAS_CALLS = { fillRect: 0, stroke: 0, fill: 0, drawImage: 0, fillText: 0, arc: 0 };

function makeContext() {
  const noop = () => {};
  const ctx = new Proxy(
    {
      canvas: null,
      setTransform: noop,
      measureText: () => ({ width: 10 }),
    },
    {
      get(target, prop) {
        if (prop in target) return target[prop];
        if (typeof prop !== "string") return undefined;
        return (...args) => {
          if (prop in CANVAS_CALLS) CANVAS_CALLS[prop] += 1;
          void args;
        };
      },
      set(target, prop, value) {
        target[prop] = value;
        return true;
      },
    },
  );
  return ctx;
}

function makeElement(id) {
  const listeners = {};
  const el = {
    id,
    tagName: id === "notes" ? "INPUT" : "DIV",
    width: 900,
    height: 600,
    value: "",
    textContent: "",
    innerHTML: "",
    style: {},
    dataset: {},
    className: "",
    classList: {
      _set: new Set(),
      add(c) { this._set.add(c); },
      remove(c) { this._set.delete(c); },
      toggle(c, force) { force === undefined ? (this._set.has(c) ? this._set.delete(c) : this._set.add(c)) : (force ? this._set.add(c) : this._set.delete(c)); },
      contains(c) { return this._set.has(c); },
    },
    getContext: () => makeContext(),
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 900, height: 600 }),
    addEventListener: (name, fn) => { (listeners[name] = listeners[name] || []).push(fn); },
    removeEventListener: () => {},
    appendChild: () => {},
    querySelector: () => makeElement(id + "-child"),
    querySelectorAll: () => [],
    blur: () => {},
    focus: () => {},
    dispatch: (name, event) => (listeners[name] || []).forEach((fn) => fn(event)),
    _listeners: listeners,
  };
  return el;
}

const elements = new Map();
function byId(id) {
  if (!elements.has(id)) elements.set(id, makeElement(id));
  return elements.get(id);
}

const kindElements = ["bounce", "near_player", "free_flight"].map((k) => {
  const el = makeElement("kind-" + k);
  el.dataset = { k };
  return el;
});

const document = {
  getElementById: byId,
  createElement: (tag) => makeElement("created-" + tag),
  querySelectorAll: (selector) => (selector === ".kind" ? kindElements : []),
  addEventListener: () => {},
  body: makeElement("body"),
};

const windowListeners = {};
const globalWindow = {
  devicePixelRatio: 1,
  addEventListener: (name, fn) => { (windowListeners[name] = windowListeners[name] || []).push(fn); },
  prompt: () => "42",
  requestAnimationFrame: () => 0,
};

// Images resolve instantly and report a real size so every draw path is taken.
class FakeImage {
  constructor() {
    this.complete = true;
    this.naturalWidth = 1920;
    this.naturalHeight = 1080;
    this._src = "";
  }
  set src(value) { this._src = value; }
  get src() { return this._src; }
}

const sandbox = {
  document,
  window: globalWindow,
  Image: FakeImage,
  fetch: (path, opts) => fetch(baseUrl + path, opts),
  requestAnimationFrame: () => 0,
  setInterval: () => 0,
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  console: {
    log: (...a) => void a,
    warn: (...a) => void a,
    error: (...a) => consoleErrors.push(a.map(String).join(" ")),
  },
  Math, JSON, Object, Array, Map, Set, Number, String, Boolean, Error, Promise,
  isFinite, parseInt, parseFloat, Date,
};
sandbox.globalThis = sandbox;
sandbox.self = sandbox;
vm.createContext(sandbox);

// The page adds a window "keydown" listener; expose it so the harness can type.
globalWindow.addEventListener = (name, fn) => {
  (windowListeners[name] = windowListeners[name] || []).push(fn);
};

async function loadPageScript() {
  const response = await fetch(baseUrl + "/");
  if (!response.ok) throw new Error(`GET / returned ${response.status}`);
  const html = await response.text();
  if (html.includes("__STUDIO_SAVE_TOKEN__")) {
    throw new Error("the served page still contains the save-token placeholder");
  }
  const match = html.match(/<script>\n([\s\S]*?)\n<\/script>/);
  if (!match) throw new Error("no <script> block in the served page");
  return match[1];
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Top-level `let` bindings in a script live in the context's lexical scope, not
// on the sandbox object, so state is read back by evaluating in the context.
const evalIn = (expr) => vm.runInContext(expr, sandbox);
const callIn = (expr) => vm.runInContext(expr, sandbox);

function press(key, shift) {
  const event = { key, shiftKey: !!shift, target: { tagName: "BODY" }, preventDefault: () => {} };
  for (const fn of windowListeners.keydown || []) fn(event);
}

async function main() {
  try {
    vm.runInContext(await loadPageScript(), sandbox, { filename: "ball_label_studio_page.js" });
  } catch (err) {
    errors.push("script evaluation: " + (err.stack || err.message));
    return report();
  }

  // boot() is kicked off at the end of the page script; wait for it to land.
  for (let i = 0; i < 120 && !evalIn("typeof S !== 'undefined' && S !== null"); i++) await sleep(50);
  if (!evalIn("typeof S !== 'undefined' && S !== null")) {
    errors.push("boot() never populated S; boot text: " + byId("boot").textContent);
    return report();
  }

  // A frame with real skeletons, a detection and a bounce candidate.
  callIn("setFrame(S.bounce_candidate_frames[0])");
  await sleep(400);

  // Every draw path must survive real data.
  callIn("redraw()");

  // Click the detector's own guess: exercises /api/ray plus the depth UI.
  await evalIn(
    "(async () => { const d = S.detections[String(ui.frame)];" +
    " await solveRay(d ? d.pixel_xy : [S.image_size[0]/2, S.image_size[1]*0.75]); })()",
  );
  if (!evalIn("ui.pending !== null")) {
    errors.push("solveRay produced no ray");
    return report();
  }

  // The one piece of geometry the browser owns: origin + t * direction, in metres.
  const depths = evalIn(
    "(() => { const r = ui.pending.ray;" +
    " const dist = (t) => { const p = pointOnRay(r, t);" +
    " return Math.hypot(p[0]-r.origin_m[0], p[1]-r.origin_m[1], p[2]-r.origin_m[2]); };" +
    " return [dist(10), dist(20)]; })()",
  );
  if (Math.abs(depths[0] - 10) > 1e-6 || Math.abs(depths[1] - 20) > 1e-6) {
    errors.push(`page ray parameterisation is not metres from the camera: ${depths}`);
  }

  // Keyboard: frame stepping, kind selection, depth nudging, panels.
  const before = evalIn("ui.frame");
  press("ArrowRight");
  if (evalIn("ui.frame") !== before + 1) errors.push("ArrowRight did not step a frame");
  press("ArrowLeft");
  await sleep(300);
  await evalIn(
    "(async () => { const d = S.detections[String(ui.frame)];" +
    " await solveRay(d ? d.pixel_xy : [S.image_size[0]/2, S.image_size[1]*0.75]); })()",
  );
  press("3");
  if (evalIn("ui.kind") !== "free_flight") errors.push("key 3 did not select free_flight");
  const depthBefore = evalIn("ui.depth");
  press("ArrowUp");
  if (!(evalIn("ui.depth") > depthBefore)) errors.push("ArrowUp did not increase depth");
  press("k");
  press("m");
  press("?");
  press("c");
  callIn("redraw()");

  // Save then delete: the whole autosave round trip through the real server.
  const frame = evalIn("ui.frame");
  await evalIn("saveLabel()");
  await sleep(300);
  if (!evalIn(`labels.has(${frame})`)) {
    errors.push("saveLabel did not record a label; status: " + byId("saveState").textContent);
  } else {
    const saved = evalIn(`labels.get(${frame})`);
    if (saved.is_ground_truth_candidate && saved.kind !== "bounce") {
      errors.push("a non-bounce label claimed to be a ground truth candidate");
    }
    if (saved.kind !== evalIn("ui.kind")) errors.push("saved kind does not match the UI kind");
  }
  await evalIn("deleteLabel()");
  await sleep(300);
  if (evalIn(`labels.has(${frame})`)) errors.push("deleteLabel did not remove the label");

  return report();
}

function report() {
  const booted = evalIn("typeof S !== 'undefined' && S !== null");
  const result = {
    ok: errors.length === 0 && consoleErrors.length === 0,
    errors,
    console_errors: consoleErrors,
    clip_id: booted ? evalIn("S.clip_id") : null,
    frame_count: booted ? evalIn("S.frame_count") : null,
    skeleton_frames: booted ? evalIn("Object.keys(S.skeletons).length") : 0,
    bounce_candidates: booted ? evalIn("S.bounce_candidate_frames.length") : 0,
    canvas_calls: CANVAS_CALLS,
  };
  console.log(JSON.stringify(result, null, 2));
  return result.ok ? 0 : 1;
}

main()
  .then((code) => process.exit(code))
  .catch((err) => {
    console.log(JSON.stringify({ ok: false, errors: [String(err.stack || err)] }, null, 2));
    process.exit(1);
  });
