"""The single-page UI for the ball-labelling studio.

Kept in its own module so ``ball_label_studio.py`` stays readable. The page
does no camera math: it asks the server for the ray a click implies, then the
only thing it computes is ``origin + depth * direction``. Everything that
could carry a sign or convention error lives in tested Python.
"""

from __future__ import annotations

SAVE_TOKEN_PLACEHOLDER = "__STUDIO_SAVE_TOKEN__"

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ball Label Studio</title>
<style>
  :root {
    --bg: #0e1116; --panel: #161b22; --line: #262d38; --text: #e6edf3;
    --muted: #8b949e; --accent: #58a6ff; --bounce: #3fb950; --near: #d29922;
    --free: #f85149; --prefill: #a371f7; --detect: #f0d078;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    background: var(--bg); color: var(--text); font: 13px/1.45 -apple-system,
    BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; overflow: hidden;
  }
  #app { display: flex; flex-direction: column; height: 100vh; }
  header {
    display: flex; align-items: center; gap: 14px; padding: 7px 12px;
    background: var(--panel); border-bottom: 1px solid var(--line); flex: none;
  }
  header h1 { font-size: 13px; margin: 0; font-weight: 600; letter-spacing: .2px; }
  .chip {
    padding: 2px 8px; border-radius: 10px; background: #1f2630; color: var(--muted);
    font-size: 11px; white-space: nowrap; border: 1px solid var(--line);
  }
  .chip b { color: var(--text); font-weight: 600; }
  .chip.bounce b { color: var(--bounce); }
  .chip.near b { color: var(--near); }
  .chip.free b { color: var(--free); }
  .grow { flex: 1; }
  #saveState { font-size: 11px; color: var(--muted); }
  #saveState.ok { color: var(--bounce); }
  #saveState.err { color: var(--free); }
  main { flex: 1; display: flex; min-height: 0; }
  #left { flex: 1.35; display: flex; flex-direction: column; min-width: 0; border-right: 1px solid var(--line); }
  #right { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  .paneHead {
    display: flex; align-items: center; gap: 8px; padding: 4px 10px; flex: none;
    background: var(--panel); border-bottom: 1px solid var(--line);
    font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .6px;
  }
  .canvasWrap { position: relative; flex: 1; min-height: 0; background: #05070a; }
  canvas { display: block; width: 100%; height: 100%; }
  #magnifier {
    position: absolute; right: 10px; top: 10px; width: 190px; height: 190px;
    border: 1px solid var(--line); border-radius: 4px; background: #05070a;
    pointer-events: none; box-shadow: 0 4px 18px rgba(0,0,0,.55);
  }
  #magnifier.hidden { display: none; }
  #view3dWrap { flex: 1.25; min-height: 0; }
  #mapWrap { flex: 1; min-height: 0; border-top: 1px solid var(--line); }
  #controls {
    flex: none; background: var(--panel); border-top: 1px solid var(--line);
    padding: 8px 12px; display: flex; flex-direction: column; gap: 7px;
  }
  .row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .kinds { display: flex; gap: 6px; }
  .kind {
    padding: 4px 10px; border-radius: 4px; border: 1px solid var(--line);
    background: #1a212b; cursor: pointer; font-size: 12px; color: var(--muted);
    user-select: none;
  }
  .kind[data-k="bounce"].on { background: rgba(63,185,80,.18); border-color: var(--bounce); color: var(--bounce); }
  .kind[data-k="near_player"].on { background: rgba(210,153,34,.18); border-color: var(--near); color: var(--near); }
  .kind[data-k="free_flight"].on { background: rgba(248,81,73,.18); border-color: var(--free); color: var(--free); }
  .kind.disabled { opacity: .35; cursor: not-allowed; }
  #depth { flex: 1; min-width: 160px; accent-color: var(--accent); }
  #depth:disabled { opacity: .4; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; }
  button {
    padding: 4px 11px; border-radius: 4px; border: 1px solid var(--line);
    background: #1a212b; color: var(--text); cursor: pointer; font-size: 12px;
  }
  button:hover { border-color: var(--accent); }
  button.primary { background: rgba(88,166,255,.16); border-color: var(--accent); color: var(--accent); }
  button.danger:hover { border-color: var(--free); color: var(--free); }
  #kindHelp { font-size: 11.5px; color: var(--muted); min-height: 16px; }
  #warn { font-size: 11.5px; color: var(--near); min-height: 0; }
  #warn:empty { display: none; }
  #timeline { flex: none; height: 46px; background: #0b0e13; border-top: 1px solid var(--line); }
  #legend {
    position: absolute; inset: auto 0 0 0; max-height: 62vh; overflow: auto;
    background: rgba(13,17,23,.97); border-top: 1px solid var(--line); padding: 14px 20px;
    columns: 2; column-gap: 34px; z-index: 20;
  }
  #legend.hidden { display: none; }
  #legend div { break-inside: avoid; display: flex; gap: 10px; padding: 1.5px 0; }
  #legend kbd {
    font-family: ui-monospace, monospace; font-size: 11px; background: #1f2630;
    border: 1px solid var(--line); border-bottom-width: 2px; border-radius: 3px;
    padding: 1px 5px; min-width: 122px; text-align: center; color: var(--text);
  }
  #legend span { color: var(--muted); font-size: 11.5px; }
  #legend h3 { margin: 0 0 8px; font-size: 12px; column-span: all; }
  #boot { padding: 40px; color: var(--muted); }
  select, input[type=text] {
    background: #1a212b; color: var(--text); border: 1px solid var(--line);
    border-radius: 4px; padding: 3px 6px; font-size: 12px;
  }
  .tierNote { font-size: 10.5px; color: var(--muted); }
  .swatch { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
</style>
</head>
<body>
<div id="boot">Loading clip…</div>
<div id="app" style="display:none">
  <header>
    <h1>Ball Label Studio</h1>
    <span class="chip" id="clipChip"></span>
    <span class="chip" id="frameChip"></span>
    <span class="chip bounce" id="bounceChip"></span>
    <span class="chip near" id="nearChip"></span>
    <span class="chip free" id="freeChip"></span>
    <span class="chip" id="progressChip"></span>
    <span class="grow"></span>
    <span class="chip" style="color:var(--free);border-color:#3d2222">REVIEW-ONLY · VERIFIED=0</span>
    <span id="saveState">ready</span>
    <button id="legendBtn">Keys (?)</button>
  </header>
  <main>
    <div id="left">
      <div class="paneHead">
        <span>1 · Video — click the ball to set the ray</span>
        <span class="grow"></span>
        <span class="mono" id="videoInfo"></span>
      </div>
      <div class="canvasWrap">
        <canvas id="video"></canvas>
        <canvas id="magnifier" width="190" height="190"></canvas>
      </div>
    </div>
    <div id="right">
      <div class="paneHead">
        <span>2 · 3D court — set depth along the ray</span>
        <span class="grow"></span>
        <span class="mono" id="view3dInfo">drag to orbit</span>
      </div>
      <div class="canvasWrap" id="view3dWrap"><canvas id="view3d"></canvas></div>
      <div class="paneHead"><span>Top-down</span><span class="grow"></span>
        <span class="mono" id="mapInfo"></span></div>
      <div class="canvasWrap" id="mapWrap"><canvas id="map"></canvas></div>
    </div>
  </main>
  <div id="controls">
    <div class="row">
      <div class="kinds">
        <div class="kind" data-k="bounce">1 · Bounce</div>
        <div class="kind" data-k="near_player">2 · Near player</div>
        <div class="kind" data-k="free_flight">3 · Free flight</div>
      </div>
      <span class="mono" id="depthLabel">depth —</span>
      <input type="range" id="depth" min="0.5" max="40" step="0.005" value="10" disabled>
      <span class="mono" id="xyzLabel">xyz —</span>
      <span class="mono" id="sigmaLabel"></span>
    </div>
    <div class="row">
      <label>confidence
        <select id="confidence">
          <option value="high">high</option>
          <option value="medium" selected>medium</option>
          <option value="low">low</option>
        </select>
      </label>
      <input type="text" id="notes" placeholder="note (optional)" style="width:180px">
      <button class="primary" id="saveBtn">Save label (Enter)</button>
      <button id="prefillBtn">Load prefill (P)</button>
      <button id="interpBtn">Interpolate (I)</button>
      <button class="danger" id="deleteBtn">Delete (Del)</button>
      <span class="grow"></span>
      <span class="tierNote">
        <span class="swatch" style="background:var(--bounce)"></span>solved
        <span class="swatch" style="background:var(--near)"></span>player-referenced
        <span class="swatch" style="background:var(--free)"></span>estimate
        <span class="swatch" style="background:var(--prefill)"></span>prefill (not a label)
      </span>
    </div>
    <div id="kindHelp"></div>
    <div id="warn"></div>
  </div>
  <canvas id="timeline"></canvas>
  <div id="legend" class="hidden"><h3>Keyboard</h3></div>
</div>
<script>
"use strict";
const TOKEN = "__STUDIO_SAVE_TOKEN__";
const KIND_COLOR = { bounce: "#3fb950", near_player: "#d29922", free_flight: "#f85149" };
const KIND_ORDER = ["bounce", "near_player", "free_flight"];
const CONF_ORDER = ["low", "medium", "high"];

let S = null;                  // static state from /api/state
let labels = new Map();        // frame -> label record
let ui = {
  frame: 0, kind: "bounce", depth: 10, confidence: "medium",
  playing: false, zoom: 1, focus: [960, 540], magnify: true,
  pending: null,               // ray solution for the current click
  pixel: null,                 // clicked pixel for the current frame
  hover: null, proposal: null, orbit: { az: -90, el: 22, dist: 26 },
};
const imgCache = new Map();

// ---------------------------------------------------------------- utilities
const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmt = (v, n = 2) => (v === null || v === undefined || !isFinite(v)) ? "—" : v.toFixed(n);

async function api(path, body) {
  const opts = body === undefined
    ? {}
    : { method: "POST", headers: { "Content-Type": "application/json", "X-Studio-Token": TOKEN },
        body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  const payload = await res.json();
  if (!res.ok) throw new Error(payload.error || res.statusText);
  return payload;
}

function setSave(text, cls) {
  const el = $("saveState");
  el.textContent = text;
  el.className = cls || "";
}

function frameImage(index) {
  if (index < 0 || index >= S.frame_count) return null;
  let img = imgCache.get(index);
  if (!img) {
    img = new Image();
    img.src = "/frame/" + String(index).padStart(6, "0") + ".jpg";
    imgCache.set(index, img);
    if (imgCache.size > 240) imgCache.delete(imgCache.keys().next().value);
  }
  return img;
}
function prefetch(center) {
  for (let d = -4; d <= 12; d++) frameImage(center + d);
}

// -------------------------------------------------------------- vector math
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const norm = (a) => { const n = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0]/n, a[1]/n, a[2]/n]; };

// The ONLY geometry the browser does: a point at depth t on the server's ray.
function pointOnRay(ray, depth) {
  return add(ray.origin_m, mul(ray.direction_unit, depth));
}

// --------------------------------------------------------- canvas plumbing
function fitCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.max(1, Math.round(rect.width * dpr));
  const h = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return { ctx, w, h, dpr };
}

// ------------------------------------------------------------- video pane
function videoTransform(w, h) {
  const [iw, ih] = S.image_size;
  const base = Math.min(w / iw, h / ih);
  const scale = base * ui.zoom;
  return {
    scale,
    toCanvas: (p) => [(p[0] - ui.focus[0]) * scale + w / 2, (p[1] - ui.focus[1]) * scale + h / 2],
    toImage: (p) => [(p[0] - w / 2) / scale + ui.focus[0], (p[1] - h / 2) / scale + ui.focus[1]],
  };
}

function drawVideo() {
  const { ctx, w, h } = fitCanvas($("video"));
  ctx.fillStyle = "#05070a";
  ctx.fillRect(0, 0, w, h);
  const T = videoTransform(w, h);
  const img = frameImage(ui.frame);
  if (img && img.complete && img.naturalWidth) {
    const tl = T.toCanvas([0, 0]);
    ctx.imageSmoothingEnabled = ui.zoom < 2;
    ctx.drawImage(img, tl[0], tl[1], S.image_size[0] * T.scale, S.image_size[1] * T.scale);
  } else {
    ctx.fillStyle = "#30363d";
    ctx.font = "14px sans-serif";
    ctx.fillText("decoding frame " + ui.frame + "…", 16, 26);
  }

  // Player skeletons: the depth reference that makes a near-player label work.
  const people = S.skeletons[String(ui.frame)] || [];
  for (const person of people) {
    ctx.lineWidth = Math.max(1, 2 * (T.scale / 0.5));
    ctx.strokeStyle = person.implausible ? "rgba(248,81,73,.55)" : "rgba(88,166,255,.85)";
    ctx.beginPath();
    for (const [a, b] of S.bone_pairs) {
      const pa = person.pixels[a], pb = person.pixels[b];
      if (!pa || !pb) continue;
      const ca = T.toCanvas(pa), cb = T.toCanvas(pb);
      ctx.moveTo(ca[0], ca[1]); ctx.lineTo(cb[0], cb[1]);
    }
    ctx.stroke();
    ctx.fillStyle = "rgba(230,237,243,.9)";
    for (const p of person.pixels) {
      if (!p) continue;
      const c = T.toCanvas(p);
      ctx.beginPath(); ctx.arc(c[0], c[1], 2.2, 0, 7); ctx.fill();
    }
    const head = person.pixels[0];
    if (head) {
      const c = T.toCanvas(head);
      ctx.fillStyle = "rgba(88,166,255,.95)";
      ctx.font = "11px ui-monospace, monospace";
      ctx.fillText("P" + person.player_id, c[0] + 7, c[1] - 7);
    }
  }

  // Detector guess — a GUESS, drawn hollow and dashed so it never reads as truth.
  const det = S.detections[String(ui.frame)];
  if (det) {
    const c = T.toCanvas(det.pixel_xy);
    ctx.strokeStyle = "rgba(240,208,120,.95)";
    ctx.setLineDash([3, 3]); ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.arc(c[0], c[1], 11, 0, 7); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(240,208,120,.95)";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText("detector guess " + fmt(det.conf, 2), c[0] + 14, c[1] - 5);
  }
  for (const cand of (S.candidates[String(ui.frame)] || [])) {
    const c = T.toCanvas(cand.pixel_xy);
    ctx.strokeStyle = "rgba(240,208,120,.32)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(c[0], c[1], 5, 0, 7); ctx.stroke();
  }

  // Pipeline prefill — also not a label until confirmed.
  const pre = S.prefill[String(ui.frame)];
  if (pre) {
    const c = T.toCanvas(pre.pixel_xy);
    ctx.strokeStyle = "rgba(163,113,247,.9)"; ctx.setLineDash([2, 4]); ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.arc(c[0], c[1], 8, 0, 7); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(163,113,247,.9)"; ctx.font = "10px ui-monospace, monospace";
    ctx.fillText("prefill", c[0] + 11, c[1] + 12);
  }

  // Interpolation proposals.
  if (ui.proposal && ui.proposal.available) {
    ctx.strokeStyle = "rgba(88,166,255,.5)"; ctx.lineWidth = 1.4;
    ctx.beginPath();
    let started = false;
    for (const s of ui.proposal.samples) {
      if (!s.pixel_xy) { started = false; continue; }
      const c = T.toCanvas(s.pixel_xy);
      if (!started) { ctx.moveTo(c[0], c[1]); started = true; } else ctx.lineTo(c[0], c[1]);
    }
    ctx.stroke();
    for (const s of ui.proposal.samples) {
      if (!s.pixel_xy) continue;
      const c = T.toCanvas(s.pixel_xy);
      ctx.fillStyle = s.frame === ui.frame ? "#58a6ff" : "rgba(88,166,255,.45)";
      ctx.beginPath(); ctx.arc(c[0], c[1], s.frame === ui.frame ? 4 : 2.2, 0, 7); ctx.fill();
    }
  }

  // The saved label at this frame.
  const saved = labels.get(ui.frame);
  if (saved) drawBallMarker(ctx, T.toCanvas(saved.pixel_xy), KIND_COLOR[saved.kind], "saved · " + saved.kind);

  // The click in progress.
  if (ui.pixel) {
    const c = T.toCanvas(ui.pixel);
    const color = KIND_COLOR[ui.kind];
    ctx.strokeStyle = color; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(c[0] - 14, c[1]); ctx.lineTo(c[0] + 14, c[1]);
    ctx.moveTo(c[0], c[1] - 14); ctx.lineTo(c[0], c[1] + 14); ctx.stroke();
    ctx.beginPath(); ctx.arc(c[0], c[1], 7, 0, 7); ctx.stroke();
  }

  $("videoInfo").textContent =
    `zoom ${ui.zoom.toFixed(1)}x · ${ui.pixel ? "px " + ui.pixel.map(v => v.toFixed(1)).join(", ") : "no click"}`;
  drawMagnifier();
}

function drawBallMarker(ctx, c, color, text) {
  ctx.strokeStyle = color; ctx.lineWidth = 2.2;
  ctx.beginPath(); ctx.arc(c[0], c[1], 9, 0, 7); ctx.stroke();
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(c[0], c[1], 2.4, 0, 7); ctx.fill();
  if (text) {
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText(text, c[0] + 12, c[1] + 4);
  }
}

// 8x magnifier — a pickleball is a handful of pixels and it is moving.
function drawMagnifier() {
  const canvas = $("magnifier");
  canvas.classList.toggle("hidden", !ui.magnify);
  if (!ui.magnify) return;
  const ctx = canvas.getContext("2d");
  const size = canvas.width, zoom = 8;
  ctx.fillStyle = "#05070a"; ctx.fillRect(0, 0, size, size);
  const at = ui.hover || ui.pixel;
  const img = frameImage(ui.frame);
  if (!at || !img || !img.complete || !img.naturalWidth) return;
  const span = size / zoom;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(img, at[0] - span / 2, at[1] - span / 2, span, span, 0, 0, size, size);
  ctx.strokeStyle = "rgba(88,166,255,.85)"; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(size / 2, 0); ctx.lineTo(size / 2, size);
  ctx.moveTo(0, size / 2); ctx.lineTo(size, size / 2);
  ctx.stroke();
  ctx.strokeStyle = "rgba(88,166,255,.35)";
  ctx.strokeRect(size / 2 - zoom, size / 2 - zoom, zoom * 2, zoom * 2);
  const det = S.detections[String(ui.frame)];
  if (det) {
    const dx = (det.pixel_xy[0] - at[0]) * zoom + size / 2;
    const dy = (det.pixel_xy[1] - at[1]) * zoom + size / 2;
    if (dx > 0 && dx < size && dy > 0 && dy < size) {
      ctx.strokeStyle = "rgba(240,208,120,.9)";
      ctx.beginPath(); ctx.arc(dx, dy, 10, 0, 7); ctx.stroke();
    }
  }
  ctx.fillStyle = "rgba(139,148,158,.9)"; ctx.font = "9px ui-monospace, monospace";
  ctx.fillText(zoom + "x", 5, size - 5);
}

// --------------------------------------------------------------- 3D pane
function make3dCamera(w, h) {
  const o = ui.orbit;
  const az = o.az * Math.PI / 180, el = o.el * Math.PI / 180;
  const target = [0, 0, 1.0];
  const eye = add(target, [o.dist * Math.cos(el) * Math.cos(az),
                           o.dist * Math.cos(el) * Math.sin(az),
                           o.dist * Math.sin(el)]);
  const forward = norm(sub(target, eye));
  const right = norm(cross(forward, [0, 0, 1]));
  const up = cross(right, forward);
  const f = 0.92 * Math.min(w, h);
  return (p) => {
    const rel = sub(p, eye);
    const z = dot(rel, forward);
    if (z <= 0.08) return null;
    return [w / 2 + f * dot(rel, right) / z, h / 2 - f * dot(rel, up) / z, z];
  };
}

function poly(ctx, project, points, close) {
  ctx.beginPath();
  let started = false;
  for (const p of points) {
    const q = project(p);
    if (!q) { started = false; continue; }
    if (!started) { ctx.moveTo(q[0], q[1]); started = true; } else ctx.lineTo(q[0], q[1]);
  }
  if (close) ctx.closePath();
  ctx.stroke();
}

function draw3d() {
  const { ctx, w, h } = fitCanvas($("view3d"));
  ctx.fillStyle = "#080b10"; ctx.fillRect(0, 0, w, h);
  const P = make3dCamera(w, h);
  const court = S.court;

  // Court surface + regulation lines.
  ctx.fillStyle = "rgba(40,70,110,.30)";
  ctx.beginPath();
  let started = false;
  for (const c of court.corners_m) {
    const q = P(c);
    if (!q) { started = false; continue; }
    if (!started) { ctx.moveTo(q[0], q[1]); started = true; } else ctx.lineTo(q[0], q[1]);
  }
  ctx.closePath(); ctx.fill();
  ctx.strokeStyle = "rgba(200,220,245,.6)"; ctx.lineWidth = 1.4;
  for (const seg of Object.values(court.line_segments_m)) poly(ctx, P, seg, false);

  // Net, drawn to regulation heights.
  const hw = court.net_width_m / 2;
  ctx.strokeStyle = "rgba(230,237,243,.85)"; ctx.lineWidth = 1.6;
  poly(ctx, P, [[-hw, 0, 0], [-hw, 0, court.net_post_height_m],
                [0, 0, court.net_center_height_m], [hw, 0, court.net_post_height_m],
                [hw, 0, 0]], false);
  ctx.strokeStyle = "rgba(230,237,243,.22)"; ctx.lineWidth = 0.7;
  for (let i = 1; i < 10; i++) {
    const x = -hw + (2 * hw * i) / 10;
    const top = court.net_center_height_m +
      (court.net_post_height_m - court.net_center_height_m) * Math.abs(x) / hw;
    poly(ctx, P, [[x, 0, 0], [x, 0, top]], false);
  }

  // Camera position: where every ray starts.
  const cam = S.camera_origin_m;
  const camQ = P(cam);
  if (camQ) {
    ctx.fillStyle = "rgba(240,208,120,.95)";
    ctx.beginPath(); ctx.arc(camQ[0], camQ[1], 4, 0, 7); ctx.fill();
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText("camera", camQ[0] + 7, camQ[1] - 6);
  }

  // Player skeletons at their tracked 3D positions.
  for (const person of (S.skeletons[String(ui.frame)] || [])) {
    ctx.strokeStyle = person.implausible ? "rgba(248,81,73,.6)" : "rgba(88,166,255,.9)";
    ctx.lineWidth = 2;
    for (const [a, b] of S.bone_pairs) poly(ctx, P, [person.world[a], person.world[b]], false);
    ctx.fillStyle = "rgba(230,237,243,.85)";
    for (const p of person.world) {
      const q = P(p);
      if (q) { ctx.beginPath(); ctx.arc(q[0], q[1], 2, 0, 7); ctx.fill(); }
    }
    const head = P(person.world[0]);
    if (head) {
      ctx.fillStyle = "rgba(88,166,255,.95)"; ctx.font = "11px ui-monospace, monospace";
      ctx.fillText("P" + person.player_id, head[0] + 7, head[1] - 8);
    }
  }

  // Saved labels, as a faint trail.
  for (const label of labels.values()) {
    const q = P(label.world_xyz_m);
    if (!q) continue;
    const near = Math.abs(label.frame - ui.frame) <= 30;
    ctx.fillStyle = KIND_COLOR[label.kind];
    ctx.globalAlpha = near ? 0.9 : 0.28;
    ctx.beginPath(); ctx.arc(q[0], q[1], label.frame === ui.frame ? 6 : 3, 0, 7); ctx.fill();
    ctx.globalAlpha = 1;
  }

  // The prefill.
  const pre = S.prefill[String(ui.frame)];
  if (pre) {
    const q = P(pre.world_xyz_m);
    if (q) {
      ctx.strokeStyle = "rgba(163,113,247,.9)"; ctx.lineWidth = 1.5;
      ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.arc(q[0], q[1], 6, 0, 7); ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // Interpolation proposal arc.
  if (ui.proposal && ui.proposal.available) {
    ctx.strokeStyle = "rgba(88,166,255,.55)"; ctx.lineWidth = 1.4;
    poly(ctx, P, ui.proposal.samples.map(s => s.world_xyz_m), false);
  }

  // THE RAY. Depth is the only unknown, so the ray is the workspace.
  if (ui.pending) {
    const ray = ui.pending.ray;
    ctx.strokeStyle = "rgba(240,208,120,.45)"; ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 4]);
    poly(ctx, P, [pointOnRay(ray, 0.4), pointOnRay(ray, S.depth_range_m[1])], false);
    ctx.setLineDash([]);

    // Depth ticks so the slider reads as metres, not as an abstract number.
    ctx.strokeStyle = "rgba(240,208,120,.35)";
    ctx.fillStyle = "rgba(240,208,120,.6)";
    ctx.font = "9px ui-monospace, monospace";
    for (let d = 4; d <= S.depth_range_m[1]; d += 4) {
      const q = P(pointOnRay(ray, d));
      if (!q) continue;
      ctx.beginPath(); ctx.arc(q[0], q[1], 1.8, 0, 7); ctx.stroke();
      ctx.fillText(d + "m", q[0] + 4, q[1] + 3);
    }

    // The marker sliding along the ray: the one degree of freedom.
    const world = pointOnRay(ray, ui.depth);
    const q = P(world);
    if (q) {
      const color = KIND_COLOR[ui.kind];
      // uncertainty bar along the ray
      const sigma = currentSigma();
      if (sigma > 0) {
        const lo = P(pointOnRay(ray, Math.max(0.4, ui.depth - sigma)));
        const hi = P(pointOnRay(ray, ui.depth + sigma));
        if (lo && hi) {
          ctx.strokeStyle = color + "88"; ctx.lineWidth = 5;
          ctx.beginPath(); ctx.moveTo(lo[0], lo[1]); ctx.lineTo(hi[0], hi[1]); ctx.stroke();
        }
      }
      ctx.strokeStyle = color; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(q[0], q[1], 8, 0, 7); ctx.stroke();
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(q[0], q[1], 2.6, 0, 7); ctx.fill();
      // drop line to the court so height is readable
      const foot = P([world[0], world[1], 0]);
      if (foot) {
        ctx.strokeStyle = color + "66"; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
        ctx.beginPath(); ctx.moveTo(q[0], q[1]); ctx.lineTo(foot[0], foot[1]); ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.font = "11px ui-monospace, monospace";
      ctx.fillText(`z=${world[2].toFixed(2)}m`, q[0] + 12, q[1] - 8);
    }

    // The reference joint that justifies a near-player label.
    const ref = ui.pending.near_player;
    if (ref && ui.kind === "near_player") {
      const rq = P(ref.joint_world_m);
      if (rq && q) {
        ctx.strokeStyle = "rgba(210,153,34,.85)"; ctx.lineWidth = 1.3;
        ctx.beginPath(); ctx.moveTo(q[0], q[1]); ctx.lineTo(rq[0], rq[1]); ctx.stroke();
        ctx.fillStyle = "rgba(210,153,34,.95)"; ctx.font = "10px ui-monospace, monospace";
        ctx.fillText(`P${ref.player_id} ${ref.joint_name}`, rq[0] + 7, rq[1] + 12);
      }
    }
  }
  $("view3dInfo").textContent =
    `az ${ui.orbit.az.toFixed(0)}° el ${ui.orbit.el.toFixed(0)}° · ${ui.orbit.dist.toFixed(0)}m · drag / wheel`;
}

// --------------------------------------------------------------- minimap
function drawMap() {
  const { ctx, w, h } = fitCanvas($("map"));
  ctx.fillStyle = "#080b10"; ctx.fillRect(0, 0, w, h);
  const court = S.court;
  const halfW = court.width_m / 2 + 2.2, halfL = court.length_m / 2 + 3.0;
  const scale = Math.min(w / (2 * halfW), h / (2 * halfL));
  // x right, y up the screen (far court at the top)
  const M = (p) => [w / 2 + p[0] * scale, h / 2 - p[1] * scale];

  ctx.fillStyle = "rgba(40,70,110,.3)";
  ctx.beginPath();
  court.corners_m.forEach((c, i) => { const q = M(c); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
  ctx.closePath(); ctx.fill();
  ctx.strokeStyle = "rgba(200,220,245,.6)"; ctx.lineWidth = 1.2;
  for (const seg of Object.values(court.line_segments_m)) {
    const a = M(seg[0]), b = M(seg[1]);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }

  const cam = M(S.camera_origin_m);
  ctx.fillStyle = "rgba(240,208,120,.95)";
  ctx.beginPath(); ctx.arc(cam[0], cam[1], 3.5, 0, 7); ctx.fill();

  for (const person of (S.skeletons[String(ui.frame)] || [])) {
    const feet = person.world[13] || person.world[0];
    const q = M(feet);
    ctx.fillStyle = "rgba(88,166,255,.9)";
    ctx.beginPath(); ctx.arc(q[0], q[1], 5, 0, 7); ctx.fill();
    ctx.fillStyle = "#0e1116"; ctx.font = "9px ui-monospace, monospace";
    ctx.fillText(String(person.player_id), q[0] - 2.5, q[1] + 3);
  }

  for (const label of labels.values()) {
    const q = M(label.world_xyz_m);
    ctx.fillStyle = KIND_COLOR[label.kind];
    ctx.globalAlpha = label.frame === ui.frame ? 1 : 0.35;
    ctx.beginPath(); ctx.arc(q[0], q[1], label.frame === ui.frame ? 5 : 2.5, 0, 7); ctx.fill();
    ctx.globalAlpha = 1;
  }

  if (ui.pending) {
    const ray = ui.pending.ray;
    const a = M(pointOnRay(ray, 0.4)), b = M(pointOnRay(ray, S.depth_range_m[1]));
    ctx.strokeStyle = "rgba(240,208,120,.5)"; ctx.setLineDash([4, 4]); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    ctx.setLineDash([]);
    const world = pointOnRay(ray, ui.depth);
    const q = M(world);
    ctx.strokeStyle = KIND_COLOR[ui.kind]; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(q[0], q[1], 6, 0, 7); ctx.stroke();
    const inCourt = Math.abs(world[0]) <= court.width_m / 2 && Math.abs(world[1]) <= court.length_m / 2;
    $("mapInfo").textContent =
      `x ${world[0].toFixed(2)}  y ${world[1].toFixed(2)}  z ${world[2].toFixed(2)} m · ${inCourt ? "in bounds" : "outside court"}`;
  } else {
    $("mapInfo").textContent = "click the ball in the video";
  }
}

// -------------------------------------------------------------- timeline
function drawTimeline() {
  const { ctx, w, h } = fitCanvas($("timeline"));
  ctx.fillStyle = "#0b0e13"; ctx.fillRect(0, 0, w, h);
  const n = S.frame_count;
  const X = (f) => (f / Math.max(1, n - 1)) * (w - 8) + 4;

  ctx.fillStyle = "rgba(139,148,158,.30)";
  for (const key of Object.keys(S.detections)) ctx.fillRect(X(+key), h - 9, 1.5, 6);

  ctx.fillStyle = "rgba(240,208,120,.85)";
  for (const f of S.bounce_candidate_frames) {
    ctx.fillRect(X(f) - 1, 4, 2.5, 11);
  }

  for (const label of labels.values()) {
    ctx.fillStyle = KIND_COLOR[label.kind];
    ctx.fillRect(X(label.frame) - 1, 18, 2.5, 14);
  }

  ctx.strokeStyle = "#58a6ff"; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(X(ui.frame), 0); ctx.lineTo(X(ui.frame), h); ctx.stroke();

  ctx.fillStyle = "rgba(139,148,158,.75)"; ctx.font = "9px ui-monospace, monospace";
  ctx.fillText("bounce candidates", 6, 12);
  ctx.fillText("labels", 6, 30);
  ctx.fillText("detections", 6, h - 2);
}

// ------------------------------------------------------------------ state
function currentSigma() {
  if (!ui.pending) return 0;
  if (ui.kind === "bounce") {
    return ui.pending.bounce.available ? ui.pending.bounce.sigma_along_ray_m : 0;
  }
  if (ui.kind === "near_player" && ui.pending.near_player) {
    const off = ui.pending.near_player.offset_from_ray_m;
    return Math.min(2.0, 0.5 * (1 + off / S.near_player_max_offset_m));
  }
  return 2.0;
}

function refreshControls() {
  for (const el of document.querySelectorAll(".kind")) {
    el.classList.toggle("on", el.dataset.k === ui.kind);
    let disabled = !ui.pending;
    if (ui.pending) {
      if (el.dataset.k === "bounce") disabled = !ui.pending.bounce.available;
      if (el.dataset.k === "near_player") disabled = !ui.pending.near_player_usable;
    }
    el.classList.toggle("disabled", disabled);
  }
  const locked = ui.kind === "bounce";
  $("depth").disabled = !ui.pending || locked;
  $("depth").value = ui.depth;
  $("confidence").value = ui.confidence;

  if (ui.pending) {
    const world = pointOnRay(ui.pending.ray, ui.depth);
    $("depthLabel").textContent = `depth ${ui.depth.toFixed(2)} m` + (locked ? " (solved)" : "");
    $("xyzLabel").textContent =
      `xyz ${world[0].toFixed(2)}, ${world[1].toFixed(2)}, ${world[2].toFixed(2)} m`;
    const sigma = currentSigma();
    $("sigmaLabel").textContent = `±${sigma.toFixed(2)} m along ray`;
    $("sigmaLabel").style.color = KIND_COLOR[ui.kind];
  } else {
    $("depthLabel").textContent = "depth —";
    $("xyzLabel").textContent = "xyz —";
    $("sigmaLabel").textContent = "";
  }

  $("kindHelp").textContent = S.kind_help[ui.kind] || "";
  let warn = "";
  if (ui.pending && ui.kind === "bounce" && !ui.pending.bounce.available) {
    warn = "This pixel's ray never meets the court in front of the camera — it cannot be a bounce. "
         + (ui.pending.bounce.reason || "");
  } else if (ui.pending && ui.kind === "near_player" && !ui.pending.near_player_usable) {
    warn = `No tracked player joint within ${S.near_player_max_offset_m} m of this ray, so there is `
         + "no depth reference. Use free flight and accept the larger uncertainty.";
  } else if (ui.pending && ui.kind === "near_player" && ui.pending.near_player) {
    const r = ui.pending.near_player;
    warn = `Reference: player ${r.player_id} ${r.joint_name}, ${r.offset_from_ray_m.toFixed(2)} m off the ray, `
         + `at depth ${r.depth_along_ray_m.toFixed(2)} m.`;
  }
  if (ui.proposal && ui.proposal.available) {
    const p = ui.proposal;
    warn += (warn ? "  " : "") +
      `Proposal: frames ${p.start_frame + 1}–${p.end_frame - 1} over ${p.span_s.toFixed(2)}s, ` +
      `+${p.extra_sigma_along_ray_m.toFixed(2)} m sigma for neglected drag` +
      (p.detector_residual_px.median !== null
        ? `, detector residual median ${p.detector_residual_px.median}px` : "") +
      (p.physically_implausible ? " — WARNING: arc passes below the court." : "") +
      "  Shift+I accepts as free-flight labels.";
  }
  $("warn").textContent = warn;

  const counts = {};
  for (const k of KIND_ORDER) counts[k] = 0;
  for (const l of labels.values()) counts[l.kind]++;
  $("clipChip").innerHTML = "clip <b>" + S.clip_id + "</b>";
  $("frameChip").innerHTML = `frame <b>${ui.frame}</b> / ${S.frame_count - 1} · t=${(S.frame_times_s[ui.frame] || 0).toFixed(2)}s`;
  $("bounceChip").innerHTML = `bounce <b>${counts.bounce}</b>`;
  $("nearChip").innerHTML = `near-player <b>${counts.near_player}</b>`;
  $("freeChip").innerHTML = `free-flight <b>${counts.free_flight}</b>`;
  $("progressChip").innerHTML =
    `labelled <b>${labels.size}</b> / ${S.frame_count} · candidates left <b>${
      S.bounce_candidate_frames.filter(f => !labels.has(f)).length}</b>`;
}

function redraw() {
  drawVideo(); draw3d(); drawMap(); drawTimeline(); refreshControls();
}

function setFrame(next, keepClick) {
  const target = clamp(Math.round(next), 0, S.frame_count - 1);
  if (target === ui.frame && keepClick) { redraw(); return; }
  ui.frame = target;
  if (!keepClick) { ui.pending = null; ui.pixel = null; }
  prefetch(target);
  const saved = labels.get(target);
  if (saved) {
    ui.kind = saved.kind;
    ui.depth = saved.depth_along_ray_m;
    ui.confidence = saved.human_confidence;
    ui.pixel = saved.pixel_xy.slice();
    solveRay(saved.pixel_xy, saved.depth_along_ray_m, saved.kind);
  } else {
    redraw();
  }
  api("/api/session", { last_frame: ui.frame, last_kind: ui.kind }).catch(() => {});
}

async function solveRay(pixel, forceDepth, forceKind) {
  ui.pixel = pixel.slice();
  try {
    const solution = await api("/api/ray", { frame: ui.frame, pixel_xy: pixel });
    ui.pending = solution;
    if (forceKind && isKindAllowed(forceKind)) ui.kind = forceKind;
    else if (!isKindAllowed(ui.kind)) ui.kind = solution.suggested_kind;
    ui.depth = forceDepth !== undefined && forceDepth !== null
      ? forceDepth
      : depthForKind(ui.kind);
    ui.focus = pixel.slice();
    redraw();
  } catch (err) {
    setSave("ray failed: " + err.message, "err");
  }
}

function isKindAllowed(kind) {
  if (!ui.pending) return true;
  if (kind === "bounce") return !!ui.pending.bounce.available;
  if (kind === "near_player") return !!ui.pending.near_player_usable;
  return true;
}

function depthForKind(kind) {
  if (!ui.pending) return ui.depth;
  if (kind === "bounce" && ui.pending.bounce.available) return ui.pending.bounce.depth_along_ray_m;
  if (kind === "near_player" && ui.pending.near_player) return ui.pending.near_player.depth_along_ray_m;
  if (ui.pending.prefill_depth_along_ray_m) return ui.pending.prefill_depth_along_ray_m;
  return ui.pending.suggested_depth_along_ray_m;
}

function setKind(kind) {
  if (!isKindAllowed(kind)) {
    setSave(kind === "bounce"
      ? "not a bounce: the ray misses the court"
      : "no player near this ray to judge depth against", "err");
    refreshControls();
    return;
  }
  ui.kind = kind;
  ui.depth = depthForKind(kind);
  redraw();
}

function setDepth(value) {
  if (ui.kind === "bounce") return;   // solved, not draggable
  ui.depth = clamp(value, S.depth_range_m[0], S.depth_range_m[1]);
  redraw();
}

// ------------------------------------------------------------- operations
async function saveLabel(origin) {
  if (!ui.pending || !ui.pixel) { setSave("click the ball first", "err"); return; }
  setSave("saving…");
  try {
    const result = await api("/api/label", {
      frame: ui.frame,
      pixel_xy: ui.pixel,
      kind: ui.kind,
      depth_along_ray_m: ui.kind === "bounce" ? null : ui.depth,
      human_confidence: ui.confidence,
      origin: origin || currentOrigin(),
      notes: $("notes").value,
    });
    labels.set(result.label.frame, result.label);
    setSave(`saved f${result.label.frame} · ${result.summary.label_count} labels`, "ok");
    $("notes").value = "";
    redraw();
  } catch (err) {
    setSave("save failed: " + err.message, "err");
  }
}

function currentOrigin() {
  const pre = S.prefill[String(ui.frame)];
  if (!pre) return "fresh";
  const world = pointOnRay(ui.pending.ray, ui.depth);
  const d = Math.hypot(world[0] - pre.world_xyz_m[0], world[1] - pre.world_xyz_m[1],
                       world[2] - pre.world_xyz_m[2]);
  const px = Math.hypot(ui.pixel[0] - pre.pixel_xy[0], ui.pixel[1] - pre.pixel_xy[1]);
  return (d < 0.02 && px < 1.0) ? "prefill_confirmed" : "prefill_corrected";
}

// P seeds the click from pipeline output so the owner CORRECTS instead of
// creating. It deliberately does not save: a prefill is never promoted to a
// label without the owner pressing Enter on it.
async function seedFromPipeline() {
  const pre = S.prefill[String(ui.frame)];
  const det = S.detections[String(ui.frame)];
  const source = pre || det;
  if (!source) { setSave("no pipeline prefill or detection at this frame", "err"); return; }
  await solveRay(source.pixel_xy, null, null);
  if (!ui.pending) return;
  if (pre && ui.pending.prefill_depth_along_ray_m) {
    ui.depth = ui.pending.prefill_depth_along_ray_m;
    if (!isKindAllowed(ui.kind)) ui.kind = ui.pending.suggested_kind;
  } else {
    ui.kind = ui.pending.suggested_kind;
    ui.depth = depthForKind(ui.kind);
  }
  redraw();
  setSave(pre
    ? "prefill loaded (3D + pixel) — correct it, then Enter to confirm as a label"
    : "detector pixel loaded (no 3D prefill) — set depth, then Enter to save", "ok");
}

async function deleteLabel() {
  if (!labels.has(ui.frame)) { setSave("no label at this frame", "err"); return; }
  try {
    const result = await api("/api/label/delete", { frame: ui.frame });
    labels.delete(ui.frame);
    setSave(`deleted f${ui.frame} · ${result.summary.label_count} labels`, "ok");
    ui.pending = null; ui.pixel = null;
    redraw();
  } catch (err) { setSave("delete failed: " + err.message, "err"); }
}

async function proposeInterpolation() {
  try {
    ui.proposal = await api("/api/interpolate?frame=" + ui.frame);
    if (!ui.proposal.available) setSave(ui.proposal.reason, "err");
    else setSave(`proposal: ${ui.proposal.samples.length} frames`, "ok");
    redraw();
  } catch (err) { setSave("interpolate failed: " + err.message, "err"); }
}

async function acceptInterpolation() {
  if (!ui.proposal || !ui.proposal.available) { setSave("propose an arc first (I)", "err"); return; }
  const frames = ui.proposal.samples.filter(s => !s.already_labelled && s.in_front_of_camera)
                                    .map(s => s.frame);
  if (!frames.length) { setSave("nothing to accept", "err"); return; }
  try {
    const result = await api("/api/interpolate/accept", { frames, human_confidence: "low" });
    setSave(`accepted ${result.accepted} interpolated free-flight labels`, "ok");
    await reloadLabels();
    ui.proposal = null;
    redraw();
  } catch (err) { setSave("accept failed: " + err.message, "err"); }
}

async function reloadLabels() {
  const state = await api("/api/state");
  labels = new Map(state.labels.map(l => [l.frame, l]));
  S.summary = state.summary;
}

// ------------------------------------------------------------------ input
function bindVideo() {
  const canvas = $("video");
  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const dpr = canvas.width / rect.width;
    const T = videoTransform(canvas.width, canvas.height);
    ui.hover = T.toImage([(e.clientX - rect.left) * dpr, (e.clientY - rect.top) * dpr]);
    drawMagnifier();
  });
  canvas.addEventListener("mouseleave", () => { ui.hover = null; drawMagnifier(); });
  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const dpr = canvas.width / rect.width;
    const T = videoTransform(canvas.width, canvas.height);
    const p = T.toImage([(e.clientX - rect.left) * dpr, (e.clientY - rect.top) * dpr]);
    if (p[0] < 0 || p[1] < 0 || p[0] > S.image_size[0] || p[1] > S.image_size[1]) return;
    solveRay([Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10]);
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const before = ui.zoom;
    ui.zoom = clamp(ui.zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15), 1, 16);
    if (ui.zoom !== before && ui.hover) ui.focus = ui.hover.slice();
    redraw();
  }, { passive: false });
}

function bind3d() {
  const canvas = $("view3d");
  let dragging = false, last = null;
  canvas.addEventListener("mousedown", (e) => { dragging = true; last = [e.clientX, e.clientY]; });
  window.addEventListener("mouseup", () => { dragging = false; });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    ui.orbit.az += (e.clientX - last[0]) * 0.4;
    ui.orbit.el = clamp(ui.orbit.el - (e.clientY - last[1]) * 0.3, -5, 85);
    last = [e.clientX, e.clientY];
    draw3d();
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    ui.orbit.dist = clamp(ui.orbit.dist * (e.deltaY < 0 ? 0.92 : 1.08), 5, 70);
    draw3d();
  }, { passive: false });
}

function bindTimeline() {
  $("timeline").addEventListener("click", (e) => {
    const rect = e.target.getBoundingClientRect();
    const frac = (e.clientX - rect.left - 4) / Math.max(1, rect.width - 8);
    setFrame(Math.round(frac * (S.frame_count - 1)));
  });
}

function nextFrom(list, from, backwards) {
  const sorted = [...list].sort((a, b) => a - b);
  if (backwards) { for (let i = sorted.length - 1; i >= 0; i--) if (sorted[i] < from) return sorted[i]; return from; }
  for (const value of sorted) if (value > from) return value;
  return from;
}

function bindKeys() {
  window.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "select" || tag === "textarea") {
      if (e.key === "Enter") e.target.blur();
      return;
    }
    const step = e.shiftKey ? 10 : 1;
    switch (e.key) {
      case "ArrowLeft": setFrame(ui.frame - step); break;
      case "ArrowRight": setFrame(ui.frame + step); break;
      case "ArrowUp": setDepth(ui.depth + (e.shiftKey ? 0.01 : 0.10)); break;
      case "ArrowDown": setDepth(ui.depth - (e.shiftKey ? 0.01 : 0.10)); break;
      case " ": ui.playing = !ui.playing; setSave(ui.playing ? "playing" : "paused"); break;
      case "1": setKind("bounce"); break;
      case "2": setKind("near_player"); break;
      case "3": setKind("free_flight"); break;
      case "k": case "K":
        setKind(KIND_ORDER[(KIND_ORDER.indexOf(ui.kind) + 1) % KIND_ORDER.length]); break;
      case "b": case "B":
        setFrame(nextFrom(S.bounce_candidate_frames, ui.frame, e.shiftKey)); break;
      case "n": case "N": {
        const pool = Object.keys(S.detections).map(Number).filter(f => !labels.has(f));
        setFrame(nextFrom(pool, ui.frame, e.shiftKey)); break;
      }
      case "l": case "L":
        setFrame(nextFrom([...labels.keys()], ui.frame, e.shiftKey)); break;
      case "c": case "C":
        ui.confidence = CONF_ORDER[(CONF_ORDER.indexOf(ui.confidence) + 1) % CONF_ORDER.length];
        refreshControls(); break;
      case "Enter": saveLabel(); break;
      case "p": case "P": seedFromPipeline(); break;
      case "i": e.shiftKey ? acceptInterpolation() : proposeInterpolation(); break;
      case "I": acceptInterpolation(); break;
      case "Backspace": case "Delete": deleteLabel(); break;
      case "z": ui.zoom = clamp(ui.zoom * 1.4, 1, 16); redraw(); break;
      case "Z": ui.zoom = clamp(ui.zoom / 1.4, 1, 16); redraw(); break;
      case "m": case "M": ui.magnify = !ui.magnify; drawMagnifier(); break;
      case "g": case "G": {
        const answer = window.prompt("Go to frame", String(ui.frame));
        if (answer !== null && answer.trim() !== "") setFrame(parseInt(answer, 10) || 0);
        break;
      }
      case "?": case "/": $("legend").classList.toggle("hidden"); break;
      default: return;
    }
    e.preventDefault();
  });
}

function playLoop() {
  let last = 0;
  const tick = (now) => {
    if (ui.playing && now - last > 1000 / (S.fps || 30)) {
      last = now;
      if (ui.frame >= S.frame_count - 1) ui.playing = false;
      else setFrame(ui.frame + 1);
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// ------------------------------------------------------------------- boot
async function boot() {
  S = await api("/api/state");
  labels = new Map(S.labels.map(l => [l.frame, l]));
  ui.focus = [S.image_size[0] / 2, S.image_size[1] / 2];
  ui.orbit.dist = Math.max(18, S.court.length_m * 1.9);

  const legend = $("legend");
  for (const [keys, what] of S.keyboard_map) {
    const row = document.createElement("div");
    row.innerHTML = "<kbd></kbd><span></span>";
    row.querySelector("kbd").textContent = keys;
    row.querySelector("span").textContent = what;
    legend.appendChild(row);
  }
  const evidence = S.calibration_evidence || {};
  const plane = evidence.plane_residual_check || {};
  if (plane.available) {
    const row = document.createElement("div");
    row.innerHTML = "<kbd>accuracy floor</kbd><span></span>";
    row.querySelector("span").textContent =
      `this clip's calibration puts bounce labels ${plane.median_m.toFixed(3)} m (median) / ` +
      `${plane.max_m.toFixed(3)} m (worst) off on the court plane, before any click error`;
    legend.appendChild(row);
  }
  if (S.missing_artifacts && S.missing_artifacts.length) {
    const row = document.createElement("div");
    row.innerHTML = "<kbd>missing</kbd><span></span>";
    row.querySelector("span").textContent = S.missing_artifacts.join(", ");
    legend.appendChild(row);
  }

  $("legendBtn").onclick = () => $("legend").classList.toggle("hidden");
  $("saveBtn").onclick = () => saveLabel();
  $("prefillBtn").onclick = () => seedFromPipeline();
  $("interpBtn").onclick = () => proposeInterpolation();
  $("deleteBtn").onclick = () => deleteLabel();
  $("depth").oninput = (e) => setDepth(parseFloat(e.target.value));
  $("depth").min = S.depth_range_m[0];
  $("depth").max = S.depth_range_m[1];
  $("confidence").onchange = (e) => { ui.confidence = e.target.value; };
  for (const el of document.querySelectorAll(".kind")) {
    el.onclick = () => setKind(el.dataset.k);
  }

  bindVideo(); bind3d(); bindTimeline(); bindKeys(); playLoop();
  window.addEventListener("resize", redraw);
  // Repaint when a decoded frame arrives so stepping never shows a stale image.
  setInterval(() => { const img = frameImage(ui.frame); if (img && img.complete) drawVideo(); }, 120);

  $("boot").style.display = "none";
  $("app").style.display = "flex";
  const resume = (S.session && S.session.last_frame) || 0;
  setFrame(resume);
  if (resume) setSave("resumed at frame " + resume, "ok");
  prefetch(resume);
}

boot().catch((err) => {
  document.getElementById("boot").textContent = "Failed to start: " + (err.stack || err.message);
});
</script>
</body>
</html>
"""

__all__ = ["HTML", "SAVE_TOKEN_PLACEHOLDER"]
