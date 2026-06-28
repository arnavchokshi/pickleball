"""Generate browser-reviewable Three.js pages for VirtualWorld artifacts."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Mapping

from .schemas import VirtualWorld, validate_artifact_file


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_virtual_world_review"
DEFAULT_THREE_MODULE_URL = "../../../../web/replay/node_modules/three/build/three.module.js"


def build_virtual_world_review_html(
    virtual_world: VirtualWorld | Mapping[str, Any],
    *,
    title: str = "Pickleball World Review",
    three_module_url: str = DEFAULT_THREE_MODULE_URL,
) -> str:
    world = virtual_world.model_dump(mode="json") if isinstance(virtual_world, VirtualWorld) else dict(virtual_world)
    payload_json = json.dumps(world, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")
    title_html = escape(title)
    three_url = escape(three_module_url, quote=True)
    warnings = ", ".join(str(warning) for warning in world.get("summary", {}).get("warnings", [])) or "none"
    summary = world.get("summary", {})
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_html}</title>
<style>
:root {{
  color-scheme: dark;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #11130f;
  color: #f2f1e8;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; overflow: hidden; }}
#app {{ min-height: 100vh; position: relative; }}
#viewport {{ display: block; height: 100vh; width: 100vw; }}
.hud {{
  background: rgba(17, 19, 15, 0.86);
  border: 1px solid rgba(222, 218, 199, 0.22);
  border-radius: 8px;
  left: 16px;
  max-width: min(420px, calc(100vw - 32px));
  padding: 14px;
  position: fixed;
  top: 16px;
}}
h1 {{ font-size: 17px; line-height: 1.2; margin: 0 0 10px; }}
dl {{ display: grid; gap: 8px; grid-template-columns: repeat(2, minmax(0, 1fr)); margin: 0; }}
dt {{ color: #aaa48f; font-size: 11px; text-transform: uppercase; }}
dd {{ font-size: 14px; margin: 2px 0 0; overflow-wrap: anywhere; }}
.warnings {{ color: #ffc66d; grid-column: 1 / -1; }}
.timebar {{
  align-items: center;
  background: rgba(17, 19, 15, 0.9);
  border: 1px solid rgba(222, 218, 199, 0.2);
  border-radius: 8px;
  bottom: 16px;
  display: grid;
  gap: 10px;
  grid-template-columns: auto 1fr auto;
  left: 16px;
  padding: 10px;
  position: fixed;
  right: 16px;
}}
button {{
  background: #e8e1c7;
  border: 0;
  border-radius: 6px;
  color: #17140e;
  cursor: pointer;
  font-weight: 700;
  min-width: 64px;
  padding: 8px 10px;
}}
input[type="range"] {{ accent-color: #76d1b2; width: 100%; }}
.badge {{ color: #aaa48f; font-variant-numeric: tabular-nums; min-width: 92px; text-align: right; }}
.error {{ color: #ff8a8a; padding: 16px; }}
@media (max-width: 680px) {{
  .hud {{ left: 10px; right: 10px; top: 10px; }}
  .timebar {{ bottom: 10px; left: 10px; right: 10px; }}
  dl {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
</style>
</head>
<body>
<div id="app">
  <canvas id="viewport" aria-label="3D virtual world review"></canvas>
  <aside class="hud">
    <h1>{title_html}</h1>
    <dl>
      <div><dt>Players</dt><dd>{escape(str(summary.get("player_count", 0)))}</dd></div>
      <div><dt>Mesh players</dt><dd>{escape(str(summary.get("mesh_player_count", 0)))}</dd></div>
      <div><dt>Mesh frames</dt><dd>{escape(str(summary.get("mesh_player_frame_count", 0)))}</dd></div>
      <div><dt>Ball frames</dt><dd>{escape(str(summary.get("ball_frame_count", 0)))}</dd></div>
      <div><dt>Approx ball frames</dt><dd>{escape(str(summary.get("approx_ball_frame_count", 0)))}</dd></div>
      <div><dt>Paddle players</dt><dd>{escape(str(summary.get("paddle_player_count", 0)))}</dd></div>
      <div><dt>Paddle frames</dt><dd>{escape(str(summary.get("paddle_frame_count", 0)))}</dd></div>
      <div><dt>Ambiguous paddle frames</dt><dd>{escape(str(summary.get("ambiguous_paddle_frame_count", 0)))}</dd></div>
      <div class="warnings"><dt>Warnings</dt><dd>{escape(warnings)}</dd></div>
    </dl>
  </aside>
  <div class="timebar">
    <button id="play" type="button">Play</button>
    <input id="time" type="range" min="0" max="0" value="0" step="1" aria-label="Frame">
    <span id="timeLabel" class="badge">t=0.000</span>
  </div>
</div>
<script id="virtual-world-data" type="application/json">{payload_json}</script>
<script type="module">
import * as THREE from "{three_url}";

const world = JSON.parse(document.getElementById("virtual-world-data").textContent);
const canvas = document.getElementById("viewport");
const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x11130f, 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
camera.position.set(0, 10.5, 16);
camera.lookAt(0, 0, 0);
scene.add(new THREE.HemisphereLight(0xf4ead5, 0x253027, 1.45));
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(-4, 10, 6);
scene.add(key);

const courtGroup = new THREE.Group();
scene.add(courtGroup);
const actorsGroup = new THREE.Group();
scene.add(actorsGroup);

const matCourt = new THREE.LineBasicMaterial({{ color: 0xded7bd }});
const matNet = new THREE.LineBasicMaterial({{ color: 0xf0a650 }});
const matBall = new THREE.MeshStandardMaterial({{ color: 0xf4d35e, emissive: 0x3a2700, roughness: 0.42 }});
const matTrack = new THREE.MeshStandardMaterial({{ color: 0x76d1b2, roughness: 0.55 }});
const matMesh = new THREE.PointsMaterial({{ color: 0x8fb7ff, size: 0.035, transparent: true, opacity: 0.78 }});
const matJoints = new THREE.PointsMaterial({{ color: 0xffffff, size: 0.06 }});
const matPaddle = new THREE.MeshStandardMaterial({{ color: 0x68e1fd, side: THREE.DoubleSide, roughness: 0.35, metalness: 0.05 }});
const matPaddleAmbiguous = new THREE.MeshStandardMaterial({{ color: 0xffb14a, side: THREE.DoubleSide, roughness: 0.35 }});

function mapPoint(point) {{
  return new THREE.Vector3(Number(point[0] || 0), Number(point[2] || 0), -Number(point[1] || 0));
}}

function lineObject(points, material) {{
  const geometry = new THREE.BufferGeometry().setFromPoints(points.map(mapPoint));
  return new THREE.Line(geometry, material);
}}

for (const [id, segment] of Object.entries(world.court.line_segments || {{}})) {{
  courtGroup.add(lineObject(segment, id === "net" ? matNet : matCourt));
}}
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(Number(world.court.width_m || 6.096), Number(world.court.length_m || 13.4112)),
  new THREE.MeshStandardMaterial({{ color: 0x24392e, roughness: 0.92, metalness: 0.0 }})
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.012;
scene.add(floor);

const times = Array.from(new Set([
  ...world.players.flatMap((player) => player.frames.map((frame) => frame.t)),
  ...world.ball.frames.map((frame) => frame.t),
  ...world.paddles.flatMap((paddle) => paddle.frames.map((frame) => frame.t)),
])).sort((a, b) => a - b);
if (!times.length) times.push(0);

const playerObjects = world.players.map(() => ({{
  track: new THREE.Mesh(new THREE.SphereGeometry(0.11, 16, 12), matTrack),
  mesh: new THREE.Points(new THREE.BufferGeometry(), matMesh),
  joints: new THREE.Points(new THREE.BufferGeometry(), matJoints),
}}));
for (const objectSet of playerObjects) {{
  actorsGroup.add(objectSet.track, objectSet.mesh, objectSet.joints);
}}
const ballObject = new THREE.Mesh(new THREE.SphereGeometry(0.07, 18, 14), matBall);
actorsGroup.add(ballObject);
const ballTrail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({{ color: 0xf4d35e, transparent: true, opacity: 0.62 }}));
actorsGroup.add(ballTrail);
const paddleObjects = world.paddles.map(() => new THREE.Mesh(new THREE.BufferGeometry(), matPaddle));
for (const paddle of paddleObjects) actorsGroup.add(paddle);

function nearestFrame(frames, t) {{
  if (!frames || !frames.length) return null;
  let best = frames[0];
  let bestDistance = Math.abs(best.t - t);
  for (const frame of frames) {{
    const distance = Math.abs(frame.t - t);
    if (distance < bestDistance) {{
      best = frame;
      bestDistance = distance;
    }}
  }}
  return best;
}}

function sampledPoints(vertices, maxPoints) {{
  if (!vertices || !vertices.length) return [];
  const step = Math.max(1, Math.ceil(vertices.length / maxPoints));
  const sampled = [];
  for (let index = 0; index < vertices.length; index += step) sampled.push(mapPoint(vertices[index]));
  return sampled;
}}

function setPoints(pointsObject, points) {{
  pointsObject.geometry.dispose();
  pointsObject.geometry = new THREE.BufferGeometry().setFromPoints(points);
  pointsObject.visible = points.length > 0;
}}

function update(index) {{
  const t = times[Math.max(0, Math.min(index, times.length - 1))];
  document.getElementById("timeLabel").textContent = `t=${{Number(t).toFixed(3)}}`;
  world.players.forEach((player, playerIndex) => {{
    const frame = nearestFrame(player.frames, t);
    const objectSet = playerObjects[playerIndex];
    if (!frame) {{
      objectSet.track.visible = false;
      setPoints(objectSet.mesh, []);
      setPoints(objectSet.joints, []);
      return;
    }}
    const track = frame.track_world_xy || frame.transl_world;
    objectSet.track.visible = Boolean(track);
    if (track) objectSet.track.position.copy(mapPoint([track[0], track[1], 0.08]));
    setPoints(objectSet.mesh, sampledPoints(frame.mesh_vertices_world, 1200));
    setPoints(objectSet.joints, (frame.joints_world || []).map(mapPoint));
  }});

  const ballFrame = nearestFrame(world.ball.frames.filter((frame) => frame.world_xyz), t);
  ballObject.visible = Boolean(ballFrame);
  if (ballFrame) ballObject.position.copy(mapPoint(ballFrame.world_xyz));
  const trailFrames = world.ball.frames.filter((frame) => frame.world_xyz && frame.t <= t);
  ballTrail.geometry.dispose();
  ballTrail.geometry = new THREE.BufferGeometry().setFromPoints(trailFrames.map((frame) => mapPoint(frame.world_xyz)));
  ballTrail.visible = trailFrames.length > 1;

  world.paddles.forEach((paddle, paddleIndex) => {{
    const frame = nearestFrame(paddle.frames, t);
    const mesh = paddleObjects[paddleIndex];
    if (!frame || !frame.mesh_vertices_world.length) {{
      mesh.visible = false;
      return;
    }}
    const vertices = frame.mesh_vertices_world.map(mapPoint);
    const positions = [];
    for (const face of frame.mesh_faces) {{
      for (const vertexIndex of face) positions.push(...vertices[vertexIndex].toArray());
    }}
    mesh.geometry.dispose();
    mesh.geometry = new THREE.BufferGeometry();
    mesh.geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    mesh.geometry.computeVertexNormals();
    mesh.material = frame.ambiguous ? matPaddleAmbiguous : matPaddle;
    mesh.visible = true;
  }});
}}

function resize() {{
  const width = window.innerWidth;
  const height = window.innerHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}}

let playing = false;
let frameIndex = 0;
const slider = document.getElementById("time");
slider.max = String(Math.max(0, times.length - 1));
slider.addEventListener("input", () => {{
  frameIndex = Number(slider.value);
  update(frameIndex);
}});
document.getElementById("play").addEventListener("click", () => {{
  playing = !playing;
  document.getElementById("play").textContent = playing ? "Pause" : "Play";
}});
window.addEventListener("resize", resize);

resize();
update(0);
function animate() {{
  requestAnimationFrame(animate);
  if (playing && times.length > 1) {{
    frameIndex = (frameIndex + 1) % times.length;
    slider.value = String(frameIndex);
    update(frameIndex);
  }}
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>
"""


def build_virtual_world_review_index(
    *,
    virtual_world_path: str | Path,
    review_html_path: str | Path,
    virtual_world: VirtualWorld | Mapping[str, Any],
    clip: str | None = None,
) -> dict[str, Any]:
    world = virtual_world.model_dump(mode="json") if isinstance(virtual_world, VirtualWorld) else dict(virtual_world)
    summary = world.get("summary", {})
    warnings = summary.get("warnings", [])
    details = [
        f"Players: {summary.get('player_count', 0)}",
        f"Mesh players: {summary.get('mesh_player_count', 0)}",
        f"Ball frames: {summary.get('ball_frame_count', 0)}",
    ]
    _append_count_detail(details, "Mesh player frames", summary.get("mesh_player_frame_count"))
    _append_count_detail(details, "Approx ball frames", summary.get("approx_ball_frame_count"))
    _append_count_detail(details, "Paddle players", summary.get("paddle_player_count"))
    details.append(f"Paddle frames: {summary.get('paddle_frame_count', 0)}")
    _append_count_detail(details, "Ambiguous paddle frames", summary.get("ambiguous_paddle_frame_count"))
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip": clip or Path(virtual_world_path).parent.name,
        "status": "rendered",
        "source_world_path": str(virtual_world_path),
        "review_html": str(review_html_path),
        "details": details,
        "warnings": [str(warning) for warning in warnings] if isinstance(warnings, list) else [],
    }


def _append_count_detail(details: list[str], label: str, value: Any) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int | float) and value > 0:
        details.append(f"{label}: {int(value)}")


def build_virtual_world_review_from_file(
    *,
    virtual_world_path: str | Path,
    out_html_path: str | Path,
    index_out_path: str | Path | None = None,
    title: str | None = None,
    three_module_url: str = DEFAULT_THREE_MODULE_URL,
    clip: str | None = None,
) -> dict[str, Any]:
    world_path = Path(virtual_world_path)
    parsed = validate_artifact_file("virtual_world", world_path)
    if not isinstance(parsed, VirtualWorld):
        raise ValueError("virtual world artifact did not parse as VirtualWorld")
    html_path = Path(out_html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html = build_virtual_world_review_html(
        parsed,
        title=title or f"{world_path.parent.name} World Review",
        three_module_url=three_module_url,
    )
    html_path.write_text(html, encoding="utf-8")
    index = build_virtual_world_review_index(
        virtual_world_path=world_path,
        review_html_path=html_path,
        virtual_world=parsed,
        clip=clip,
    )
    if index_out_path is not None:
        index_path = Path(index_out_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


__all__ = [
    "ARTIFACT_TYPE",
    "DEFAULT_THREE_MODULE_URL",
    "build_virtual_world_review_from_file",
    "build_virtual_world_review_html",
    "build_virtual_world_review_index",
]
