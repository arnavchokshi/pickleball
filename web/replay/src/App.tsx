import { Canvas, useFrame, useThree } from "@react-three/fiber";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { BufferAttribute, BufferGeometry, Color } from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import {
  activeBallContactPlayerIds,
  ballFrameForTime,
  contactEventCount,
  frameForTime,
  parseContactWindows,
  parsePhysicsRefinement,
  parseViewerManifest,
  parseVirtualWorld,
  startTimeFromSearch,
  type ContactWindows,
  type PhysicsRefinement,
  type Vec3,
  type ViewerManifest,
  type VirtualWorld,
  type VirtualWorldFrame,
  type VirtualWorldPlayer,
  worldStats,
} from "./viewerData";

const sampleWorld = {
  schema_version: 1,
  artifact_type: "racketsport_virtual_world",
  world_frame: "court_Z0",
  fps: 30,
  court: {
    sport: "pickleball",
    coordinate_frame: "origin_net_center_x_width_y_length_z_up_m",
    length_m: 13.41,
    width_m: 6.1,
    line_segments: {
      near_baseline: [
        [-3.05, 0, 0],
        [3.05, 0, 0],
      ],
      far_baseline: [
        [-3.05, 13.41, 0],
        [3.05, 13.41, 0],
      ],
      left_sideline: [
        [-3.05, 0, 0],
        [-3.05, 13.41, 0],
      ],
      right_sideline: [
        [3.05, 0, 0],
        [3.05, 13.41, 0],
      ],
    },
    net: {
      endpoints: [
        [-3.05, 6.705, 0.91],
        [3.05, 6.705, 0.91],
      ],
      center_height_m: 0.86,
      post_height_m: 0.91,
    },
  },
  players: [],
  ball: { source: null, frames: [] },
  paddles: [],
  summary: {
    player_count: 0,
    mesh_player_count: 0,
    mesh_player_frame_count: 0,
    joint_player_frame_count: 0,
    track_only_player_frame_count: 0,
    floor_placed_player_frame_count: 0,
    floor_contact_player_frame_count: 0,
    max_floor_penetration_m: 0,
    max_abs_floor_offset_m: 0,
    physics_modes: [],
    ball_frame_count: 0,
    approx_ball_frame_count: 0,
    paddle_player_count: 0,
    paddle_frame_count: 0,
    ambiguous_paddle_frame_count: 0,
    warnings: ["load_a_manifest_query_param"],
  },
};

type LabelItem = {
  frame?: string | number;
  bbox?: number[];
  bbox_xyxy?: number[];
  id?: string;
  status?: string;
};

export default function App() {
  const initialTime = useMemo(() => startTimeFromSearch(window.location.search), []);
  const [manifest, setManifest] = useState<ViewerManifest | null>(null);
  const [world, setWorld] = useState<VirtualWorld>(() => parseVirtualWorld(sampleWorld));
  const [labels, setLabels] = useState<LabelItem[]>([]);
  const [physics, setPhysics] = useState<PhysicsRefinement | null>(null);
  const [contactWindows, setContactWindows] = useState<ContactWindows | null>(null);
  const [currentTime, setCurrentTime] = useState(initialTime);
  const [loadError, setLoadError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const currentTimeRef = useRef(0);
  const initialSeekAppliedRef = useRef(false);

  useEffect(() => {
    const manifestUrlParam = new URLSearchParams(window.location.search).get("manifest");
    if (!manifestUrlParam) return;
    const manifestUrl = manifestUrlParam;
    let cancelled = false;
    async function load() {
      try {
        const manifestPayload = parseViewerManifest(await fetchJson(manifestUrl));
        const worldPayload = parseVirtualWorld(await fetchJson(manifestPayload.virtual_world_url));
        const firstOverlay = manifestPayload.label_overlays.find((overlay) => overlay.kind === "player_boxes");
        const labelPayload = firstOverlay ? await fetchJson(firstOverlay.url) : null;
        const physicsPayload = manifestPayload.physics_refinement_url
          ? parsePhysicsRefinement(await fetchJson(manifestPayload.physics_refinement_url))
          : null;
        const contactPayload = manifestPayload.contact_windows_url
          ? parseContactWindows(await fetchJson(manifestPayload.contact_windows_url))
          : null;
        if (cancelled) return;
        setManifest(manifestPayload);
        setWorld(worldPayload);
        setLabels(readLabelItems(labelPayload));
        setPhysics(physicsPayload);
        setContactWindows(contactPayload);
        setLoadError(null);
      } catch (error) {
        if (!cancelled) setLoadError(error instanceof Error ? error.message : String(error));
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    currentTimeRef.current = currentTime;
  }, [currentTime]);

  useEffect(() => {
    let animationFrame = 0;
    let lastSampleMs = 0;
    const minIntervalMs = 1000 / Math.min(60, Math.max(24, world.fps || 30));
    const tick = (now: number) => {
      const video = videoRef.current;
      if (video && !video.paused && now - lastSampleMs >= minIntervalMs) {
        lastSampleMs = now;
        if (Math.abs(video.currentTime - currentTimeRef.current) > 0.004) {
          setCurrentTime(video.currentTime);
        }
      }
      animationFrame = window.requestAnimationFrame(tick);
    };
    animationFrame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(animationFrame);
  }, [manifest?.video_url, world.fps]);

  const stats = useMemo(() => worldStats(world), [world]);
  const activeLabels = useMemo(() => labelsForTime(labels, currentTime, world.fps), [labels, currentTime, world.fps]);
  const playerBoxOverlay = useMemo(
    () => manifest?.label_overlays.find((overlay) => overlay.kind === "player_boxes") ?? null,
    [manifest],
  );
  const activeContactPlayerIds = useMemo(
    () => activeBallContactPlayerIds(world, contactWindows, currentTime),
    [world, contactWindows, currentTime],
  );
  const viewBox = useMemo(() => labelViewBox(labels), [labels]);
  const contactReadout = activeContactPlayerIds.size
    ? `3D contact: ${Array.from(activeContactPlayerIds).map((id) => `p${id}`).join(", ")}`
    : "3D contact: none";

  const syncVideoTime = (video: HTMLVideoElement) => {
    if (Math.abs(video.currentTime - currentTimeRef.current) > 0.004) {
      setCurrentTime(video.currentTime);
    }
  };

  const syncLoadedVideoTime = (video: HTMLVideoElement) => {
    if (!initialSeekAppliedRef.current && initialTime > 0) {
      const duration = Number.isFinite(video.duration) ? video.duration : initialTime;
      video.currentTime = Math.min(initialTime, duration);
      initialSeekAppliedRef.current = true;
    }
    syncVideoTime(video);
  };

  return (
    <main className="viewer-shell" aria-label="Replay viewer">
      <header className="viewer-header">
        <div>
          <p className="eyebrow">Court Z0 Review</p>
          <h1>{manifest?.clip ?? "Replay Viewer"}</h1>
        </div>
        <div className="status-grid">
          <Metric label="Players" value={stats.players} />
          <Metric label="Mesh Frames" value={stats.meshFrames} />
          <Metric label="Floor Frames" value={stats.floorPlacedFrames} />
          <Metric label="Ball Contacts" value={contactEventCount(contactWindows)} />
        </div>
      </header>

      {loadError ? <p className="load-error">{loadError}</p> : null}

      <section className="review-layout">
        <div className="video-panel">
          <div className="video-frame">
            {manifest ? (
              <video
                ref={videoRef}
                src={manifest.video_url}
                controls
                playsInline
                onLoadedMetadata={(event) => syncLoadedVideoTime(event.currentTarget)}
                onSeeked={(event) => syncVideoTime(event.currentTarget)}
                onSeeking={(event) => syncVideoTime(event.currentTarget)}
                onTimeUpdate={(event) => syncVideoTime(event.currentTarget)}
              />
            ) : (
              <div className="empty-video">Add ?manifest=/@fs/absolute/path/replay_viewer_manifest.json</div>
            )}
            <svg className="box-overlay" viewBox={viewBox} preserveAspectRatio="xMidYMid meet" aria-hidden="true">
              {activeLabels.map((item, index) => {
                const box = item.bbox_xyxy ?? xywhToXyxy(item.bbox);
                if (!box) return null;
                const [x1, y1, x2, y2] = box;
                return (
                  <g key={`${item.id ?? "box"}-${index}`}>
                    <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} className={item.status === "uncertain" ? "box uncertain" : "box"} />
                    <text x={x1 + 4} y={Math.max(12, y1 - 4)}>{item.id ?? "player"}</text>
                  </g>
                );
              })}
            </svg>
          </div>
          <div className="timeline-readout">
            <span>{currentTime.toFixed(2)}s</span>
            <span>{activeLabels.length} boxes</span>
            <span>{contactReadout}</span>
            <span>{labelTrustText(playerBoxOverlay)}</span>
          </div>
        </div>

        <div className="world-panel">
          <Canvas camera={{ position: [0, -18, 8.5], fov: 50, near: 0.05, far: 100 }}>
            <color attach="background" args={["#111315"]} />
            <ambientLight intensity={1.8} />
            <directionalLight position={[0, -4, 8]} intensity={2.2} />
            <OrbitRig world={world} />
            <CourtSurface world={world} />
            <CourtLines world={world} />
            <NetAssembly world={world} />
            <Players world={world} currentTime={currentTime} activeContactPlayerIds={activeContactPlayerIds} />
            <Ball world={world} currentTime={currentTime} />
          </Canvas>
          <div className="scene-legend">
            <span><i className="swatch floor" /> floor</span>
            <span><i className="swatch mesh" /> contact BODY</span>
            <span><i className="swatch joints" /> joints</span>
            <span><i className="swatch ball" /> ball</span>
          </div>
        </div>
      </section>

      <section className="details-band">
        <p>Physics modes: {stats.physicsModes.length ? stats.physicsModes.join(", ") : "none"}</p>
        <p>{physics ? `Physics artifact: ${physics.physics}; FOOT-2 done: ${String(physics.foot2_done)}` : "Physics artifact: none"}</p>
        <p>Max floor penetration: {stats.maxFloorPenetrationM.toFixed(4)} m</p>
        <p>{manifest?.notes[0] ?? "Review-only viewer. Artifact gates stay separate from visual inspection."}</p>
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function OrbitRig({ world }: { world: VirtualWorld }) {
  const { camera, gl } = useThree();
  const controlsRef = useRef<OrbitControls | null>(null);
  const pose = useMemo(() => defaultCameraPose(world), [world]);
  useEffect(() => {
    const controls = new OrbitControls(camera, gl.domElement);
    camera.up.set(0, 0, 1);
    camera.position.set(...pose.position);
    controls.target.set(...pose.target);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 3;
    controls.maxDistance = 32;
    controls.maxPolarAngle = Math.PI * 0.48;
    controls.update();
    controlsRef.current = controls;
    return () => {
      controlsRef.current = null;
      controls.dispose();
    };
  }, [camera, gl, pose.position, pose.target]);
  useFrame(() => controlsRef.current?.update());
  return null;
}

function CourtSurface({ world }: { world: VirtualWorld }) {
  const bounds = courtBounds(world);
  return (
    <mesh position={[bounds.centerX, bounds.centerY, -0.012]}>
      <planeGeometry args={[bounds.width, bounds.length]} />
      <meshStandardMaterial color="#1d7250" roughness={0.82} metalness={0.02} />
    </mesh>
  );
}

function CourtLines({ world }: { world: VirtualWorld }) {
  const courtPoints = Object.values(world.court.line_segments).flat();
  const netPoints = world.court.net.endpoints;
  return (
    <>
      <LineSegments points={courtPoints} color="#e9f4e8" />
      <LineSegments points={netPoints} color="#ffcf5a" />
    </>
  );
}

function NetAssembly({ world }: { world: VirtualWorld }) {
  const [left, right] = world.court.net.endpoints;
  const width = Math.hypot(right[0] - left[0], right[1] - left[1]);
  const center: Vec3 = [(left[0] + right[0]) / 2, (left[1] + right[1]) / 2, world.court.net.post_height_m / 2];
  const topLeft: Vec3 = [left[0], left[1], world.court.net.post_height_m];
  const topRight: Vec3 = [right[0], right[1], world.court.net.post_height_m];
  const centerTop: Vec3 = [center[0], center[1], world.court.net.center_height_m];
  return (
    <>
      <mesh position={center}>
        <boxGeometry args={[width, 0.045, world.court.net.post_height_m]} />
        <meshStandardMaterial color="#9fd3d6" transparent opacity={0.24} roughness={0.7} />
      </mesh>
      <mesh position={[left[0], left[1], world.court.net.post_height_m / 2]}>
        <boxGeometry args={[0.075, 0.075, world.court.net.post_height_m]} />
        <meshStandardMaterial color="#f2f2e8" />
      </mesh>
      <mesh position={[right[0], right[1], world.court.net.post_height_m / 2]}>
        <boxGeometry args={[0.075, 0.075, world.court.net.post_height_m]} />
        <meshStandardMaterial color="#f2f2e8" />
      </mesh>
      <LineSegments points={[topLeft, centerTop, centerTop, topRight]} color="#ffcf5a" />
    </>
  );
}

function Players({
  world,
  currentTime,
  activeContactPlayerIds,
}: {
  world: VirtualWorld;
  currentTime: number;
  activeContactPlayerIds: Set<number>;
}) {
  return (
    <>
      {world.players.map((player) => {
        const frame = frameForTime(player, currentTime);
        return <Player key={player.id} player={player} frame={frame} isBallContactActive={activeContactPlayerIds.has(player.id)} />;
      })}
    </>
  );
}

function Player({
  player,
  frame,
  isBallContactActive,
}: {
  player: VirtualWorldPlayer;
  frame?: VirtualWorldFrame;
  isBallContactActive: boolean;
}) {
  const floor = frame?.floor_world_xyz ?? (frame?.track_world_xy ? [frame.track_world_xy[0], frame.track_world_xy[1], 0] as Vec3 : null);
  const track = player.frames.map((entry) => entry.floor_world_xyz).filter(isVec3);
  const meshPoints = isBallContactActive ? sampleMeshPoints(frame?.mesh_vertices_world ?? [], 320) : [];
  const bodyJoints = isBallContactActive ? frame?.joints_world ?? [] : [];
  const proxySkeleton = skeletonForFrame(frame);
  return (
    <>
      {track.length >= 2 ? <LineStrip points={track} color="#6cb2ff" /> : null}
      {floor ? (
        <mesh position={floor}>
          <cylinderGeometry args={[0.16, 0.16, 0.025, 28]} />
          <meshStandardMaterial color={isBallContactActive ? "#e8ff34" : "#6cb2ff"} />
        </mesh>
      ) : null}
      {proxySkeleton ? <SkeletonGraph skeleton={proxySkeleton} active={isBallContactActive} /> : null}
      {bodyJoints.length ? <PointCloud points={bodyJoints} color="#ffb45d" size={0.055} /> : null}
      {meshPoints.length ? <PointCloud points={meshPoints} color="#b4f2bf" size={0.032} /> : null}
    </>
  );
}

function Ball({ world, currentTime }: { world: VirtualWorld; currentTime: number }) {
  const frame = ballFrameForTime(world, currentTime);
  if (!frame?.world_xyz) return null;
  return (
    <mesh position={frame.world_xyz}>
      <sphereGeometry args={[0.055, 16, 16]} />
      <meshStandardMaterial color="#e8ff34" emissive="#526000" />
    </mesh>
  );
}

function SkeletonGraph({ skeleton, active }: { skeleton: { joints: Vec3[]; bones: Vec3[] }; active: boolean }) {
  return (
    <>
      <LineSegments points={skeleton.bones} color={active ? "#7cff87" : "#8bc9ff"} />
      <PointCloud points={skeleton.joints} color={active ? "#7cff87" : "#a8d8ff"} size={active ? 0.052 : 0.042} />
    </>
  );
}

function LineSegments({ points, color }: { points: Vec3[]; color: string }) {
  const geometry = useMemo(() => geometryFromPoints(points), [points]);
  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color={color} />
    </lineSegments>
  );
}

function labelTrustText(overlay: ViewerManifest["label_overlays"][number] | null): string {
  if (!overlay) return "labels: none";
  if (overlay.not_ground_truth) return "labels: review only";
  return overlay.trusted_for_metrics ? "labels: trusted" : "labels: not trusted";
}

function LineStrip({ points, color }: { points: Vec3[]; color: string }) {
  const geometry = useMemo(() => geometryFromPolylineSegments(points), [points]);
  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color={color} />
    </lineSegments>
  );
}

function PointCloud({ points, color, size }: { points: Vec3[]; color: string; size: number }) {
  const geometry = useMemo(() => geometryFromPoints(points), [points]);
  return (
    <points geometry={geometry}>
      <pointsMaterial color={new Color(color)} size={size} sizeAttenuation />
    </points>
  );
}

function geometryFromPoints(points: Vec3[]) {
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(new Float32Array(points.flat()), 3));
  return geometry;
}

function geometryFromPolylineSegments(points: Vec3[]) {
  const segments: Vec3[] = [];
  for (let index = 1; index < points.length; index += 1) {
    segments.push(points[index - 1], points[index]);
  }
  return geometryFromPoints(segments);
}

function courtBounds(world: VirtualWorld) {
  const points = Object.values(world.court.line_segments).flat();
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const minX = Math.min(...xs, -world.court.width_m / 2);
  const maxX = Math.max(...xs, world.court.width_m / 2);
  const minY = Math.min(...ys, 0);
  const maxY = Math.max(...ys, world.court.length_m);
  return {
    centerX: (minX + maxX) / 2,
    centerY: (minY + maxY) / 2,
    minY,
    maxY,
    width: Math.max(1, maxX - minX),
    length: Math.max(1, maxY - minY),
  };
}

function defaultCameraPose(world: VirtualWorld): { position: Vec3; target: Vec3 } {
  const bounds = courtBounds(world);
  return {
    position: [
      bounds.centerX,
      bounds.minY - bounds.length * 0.86,
      Math.max(6.5, bounds.length * 0.64),
    ],
    target: [bounds.centerX, bounds.centerY, 0.35],
  };
}

function skeletonForFrame(frame: VirtualWorldFrame | undefined): { joints: Vec3[]; bones: Vec3[] } | null {
  const floor = frame?.floor_world_xyz ?? (frame?.track_world_xy ? ([frame.track_world_xy[0], frame.track_world_xy[1], 0] as Vec3) : null);
  if (!floor) return null;
  const [x, y] = floor;
  const joints: Vec3[] = [
    [x, y, 0.92],
    [x, y, 1.28],
    [x, y, 1.66],
    [x - 0.24, y, 1.34],
    [x + 0.24, y, 1.34],
    [x - 0.38, y, 1.02],
    [x + 0.38, y, 1.02],
    [x - 0.17, y, 0.86],
    [x + 0.17, y, 0.86],
    [x - 0.18, y - 0.04, 0.42],
    [x + 0.18, y - 0.04, 0.42],
    [x - 0.2, y - 0.08, 0.06],
    [x + 0.2, y - 0.08, 0.06],
  ];
  const bonePairs = [
    [0, 1],
    [1, 2],
    [1, 3],
    [1, 4],
    [3, 5],
    [4, 6],
    [0, 7],
    [0, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [10, 12],
  ];
  return { joints, bones: bonePairs.flatMap(([left, right]) => [joints[left], joints[right]]) };
}

function sampleMeshPoints(points: Vec3[], maxPoints: number): Vec3[] {
  if (points.length <= maxPoints) return points;
  const stride = Math.ceil(points.length / maxPoints);
  return points.filter((_, index) => index % stride === 0);
}

async function fetchJson(url: string) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`failed to fetch ${url}: ${response.status}`);
  return response.json() as Promise<unknown>;
}

function readLabelItems(input: unknown): LabelItem[] {
  if (!input || typeof input !== "object" || Array.isArray(input)) return [];
  const annotation = "annotation" in input ? (input as { annotation?: unknown }).annotation : null;
  if (!annotation || typeof annotation !== "object" || Array.isArray(annotation)) return [];
  const items = (annotation as { items?: unknown }).items;
  return Array.isArray(items) ? items.filter((item): item is LabelItem => typeof item === "object" && item !== null) : [];
}

function labelsForTime(labels: LabelItem[], currentTime: number, fps: number) {
  const frameIndex = Math.max(0, Math.round(currentTime * fps));
  return labels.filter((item) => labelFrameIndex(item.frame) === frameIndex).slice(0, 8);
}

function labelFrameIndex(value: LabelItem["frame"]): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return Math.max(0, Math.round(value));
  if (typeof value !== "string") return null;
  const match = value.match(/(\d+)/);
  return match ? Math.max(0, Number.parseInt(match[1], 10) - 1) : null;
}

function labelViewBox(labels: LabelItem[]) {
  let maxX = 1920;
  let maxY = 1080;
  for (const item of labels) {
    const box = item.bbox_xyxy ?? xywhToXyxy(item.bbox);
    if (!box) continue;
    maxX = Math.max(maxX, box[2]);
    maxY = Math.max(maxY, box[3]);
  }
  return `0 0 ${Math.ceil(maxX)} ${Math.ceil(maxY)}`;
}

function xywhToXyxy(value?: number[]): [number, number, number, number] | null {
  if (!Array.isArray(value) || value.length < 4) return null;
  return [value[0], value[1], value[0] + value[2], value[1] + value[3]];
}

function isVec3(value: Vec3 | null | undefined): value is Vec3 {
  return Array.isArray(value) && value.length === 3;
}
