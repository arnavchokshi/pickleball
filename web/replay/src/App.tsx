import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { BufferAttribute, BufferGeometry, Color, DoubleSide } from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

import { activeReplayPointForTime, parseReplayScene, resolveReplaySceneAssetUrl, type ReplayScene } from "./replayScene";
import { UploadPanel } from "./UploadPanel";
import {
  activeBallContactPlayerIds,
  ballRenderInfoForTime,
  contactEventCount,
  frameForTime,
  labelOverlayForTime,
  labelViewBox,
  parseBodyMesh,
  parseContactWindows,
  parseLabelOverlayPayload,
  parsePhysicsRefinement,
  parseViewerManifest,
  parseVirtualWorld,
  playerCoverageStats,
  solidBodyMeshFramesForTime,
  startTimeFromSearch,
  type ActiveBodyMeshFrame,
  type BodyMesh,
  type ContactWindows,
  type LabelOverlayPayload,
  type PhysicsRefinement,
  type Vec3,
  type ViewerManifest,
  type VirtualWorld,
  type VirtualWorldFrame,
  type VirtualWorldPlayer,
  worldStats,
} from "./viewerData";

const DEFAULT_REPLAY_MANIFEST_URL = import.meta.env.VITE_REPLAY_MANIFEST_URL?.trim() || null;

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

export function manifestUrlFromSearch(search: string): string | null {
  const url = new URLSearchParams(search).get("manifest");
  return url && url.trim() ? url : DEFAULT_REPLAY_MANIFEST_URL;
}

export function bodyMeshOpacityFromBlendWeight(frame: Pick<ActiveBodyMeshFrame["frame"], "blend_weight">): number {
  return Math.max(0, Math.min(1, frame.blend_weight)) * 0.68;
}

export function contactPlayerIdsForViewer(
  activeContactPlayerIds: Set<number>,
  activeBodyMeshes: ActiveBodyMeshFrame[],
): Set<number> {
  const playerIds = new Set(activeContactPlayerIds);
  for (const mesh of activeBodyMeshes) {
    playerIds.add(mesh.playerId);
  }
  return playerIds;
}

export function contactReadoutText(
  activeContactPlayerIds: Set<number>,
  activeBodyMeshes: ActiveBodyMeshFrame[],
): string {
  const playerIds = contactPlayerIdsForViewer(activeContactPlayerIds, activeBodyMeshes);
  return playerIds.size
    ? `3D contact: ${Array.from(playerIds).map((id) => `p${id}`).join(", ")}`
    : "3D contact: none";
}

export default function App() {
  const initialTime = useMemo(() => startTimeFromSearch(window.location.search), []);
  const [manifest, setManifest] = useState<ViewerManifest | null>(null);
  const [world, setWorld] = useState<VirtualWorld>(() => parseVirtualWorld(sampleWorld));
  const [labelOverlay, setLabelOverlay] = useState<LabelOverlayPayload>(() => parseLabelOverlayPayload(null));
  const [physics, setPhysics] = useState<PhysicsRefinement | null>(null);
  const [contactWindows, setContactWindows] = useState<ContactWindows | null>(null);
  const [bodyMesh, setBodyMesh] = useState<BodyMesh | null>(null);
  const [replayScene, setReplayScene] = useState<ReplayScene | null>(null);
  const [currentTime, setCurrentTime] = useState(initialTime);
  const [videoDuration, setVideoDuration] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const currentTimeRef = useRef(0);
  const initialSeekAppliedRef = useRef(false);

  useEffect(() => {
    const manifestUrl = manifestUrlFromSearch(window.location.search);
    if (manifestUrl === null) {
      setManifest(null);
      setBodyMesh(null);
      setLoadError(null);
      return;
    }
    const resolvedManifestUrl = manifestUrl;
    let cancelled = false;
    async function load() {
      try {
        const manifestPayload = parseViewerManifest(await fetchJson(resolvedManifestUrl));
        const worldPayload = parseVirtualWorld(await fetchJson(manifestPayload.virtual_world_url));
        const firstOverlay = manifestPayload.label_overlays.find((overlay) => overlay.kind === "player_boxes");
        const labelPayload = firstOverlay ? await fetchJson(firstOverlay.url) : null;
        const physicsPayload = manifestPayload.physics_refinement_url
          ? parsePhysicsRefinement(await fetchJson(manifestPayload.physics_refinement_url))
          : null;
        const contactPayload = manifestPayload.contact_windows_url
          ? parseContactWindows(await fetchJson(manifestPayload.contact_windows_url))
          : null;
        const bodyMeshPayload = manifestPayload.body_mesh_url
          ? parseBodyMesh(await fetchJson(manifestPayload.body_mesh_url))
          : null;
        const replayScenePayload = manifestPayload.replay_scene_url
          ? parseReplayScene(await fetchJson(manifestPayload.replay_scene_url))
          : null;
        if (cancelled) return;
        setManifest(manifestPayload);
        setWorld(worldPayload);
        setLabelOverlay(parseLabelOverlayPayload(labelPayload));
        setPhysics(physicsPayload);
        setContactWindows(contactPayload);
        setBodyMesh(bodyMeshPayload);
        setReplayScene(replayScenePayload);
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
  const coverage = useMemo(() => playerCoverageStats(world), [world]);
  const activeLabels = useMemo(() => labelOverlayForTime(labelOverlay, currentTime), [labelOverlay, currentTime]);
  const ballRenderInfo = useMemo(() => ballRenderInfoForTime(world, currentTime), [world, currentTime]);
  const playerBoxOverlay = useMemo(
    () => manifest?.label_overlays.find((overlay) => overlay.kind === "player_boxes") ?? null,
    [manifest],
  );
  const trustedAnnotationSources = useMemo(
    () => manifest?.annotation_sources.filter((source) => source.trusted_for_metrics) ?? [],
    [manifest],
  );
  const activeContactPlayerIds = useMemo(
    () => activeBallContactPlayerIds(world, contactWindows, currentTime),
    [world, contactWindows, currentTime],
  );
  const activeBodyMeshes = useMemo(
    () => solidBodyMeshFramesForTime(bodyMesh, contactWindows, currentTime),
    [bodyMesh, contactWindows, currentTime],
  );
  const viewerContactPlayerIds = useMemo(
    () => contactPlayerIdsForViewer(activeContactPlayerIds, activeBodyMeshes),
    [activeContactPlayerIds, activeBodyMeshes],
  );
  const viewBox = useMemo(() => labelViewBox(labelOverlay), [labelOverlay]);
  const activeReplayPoint = useMemo(() => (replayScene ? activeReplayPointForTime(replayScene, currentTime) : undefined), [replayScene, currentTime]);
  const coverageGapActive = coverage.lastTime !== null && currentTime > coverage.lastTime + Math.max(0.12, 1 / (world.fps || 30));
  const contactReadout = contactReadoutText(activeContactPlayerIds, activeBodyMeshes);
  const ballReadout = ballRenderText(ballRenderInfo.mode);

  const syncVideoTime = (video: HTMLVideoElement) => {
    if (Math.abs(video.currentTime - currentTimeRef.current) > 0.004) {
      setCurrentTime(video.currentTime);
    }
  };

  const syncLoadedVideoTime = (video: HTMLVideoElement) => {
    if (Number.isFinite(video.duration)) {
      setVideoDuration(video.duration);
    }
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
        <div className="viewer-brand">
          <div className="brand-mark" aria-hidden="true">
            <span />
          </div>
          <div>
            <p className="eyebrow">Court intelligence</p>
            <h1>{manifest?.clip ?? "Replay Review"}</h1>
          </div>
        </div>
        <div className="status-grid">
          <Metric label="Players" value={stats.players} />
          <Metric label="Mesh Frames" value={stats.meshFrames} />
          <Metric label="Floor Frames" value={stats.floorPlacedFrames} />
          <Metric label="Ball Contacts" value={contactEventCount(contactWindows)} />
          <Metric label="Replay Points" value={replayScene?.points.length ?? 0} />
          <Metric label="Player Span" value={coverage.lastTime === null ? "0.0s" : `${coverage.lastTime.toFixed(1)}s`} />
        </div>
      </header>

      {loadError ? <p className="load-error">{loadError}</p> : null}

      <UploadPanel />

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
              <div className="empty-video">Load a replay manifest with ?manifest=...</div>
            )}
            <svg className="box-overlay" viewBox={viewBox} preserveAspectRatio="xMidYMid meet" aria-hidden="true">
              {activeLabels.map((item, index) => {
                const box = item.bbox_xyxy ?? xywhToXyxy(item.bbox);
                if (!box) return null;
                const [x1, y1, x2, y2] = box;
                const className = [
                  "box",
                  item.status === "uncertain" ? "uncertain" : "",
                  labelOverlay.notGroundTruth ? "draft" : "",
                ]
                  .filter(Boolean)
                  .join(" ");
                return (
                  <g key={`${item.id ?? "box"}-${index}`}>
                    <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} className={className} />
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
            <span>{ballReadout}</span>
            <span>{labelTrustText(playerBoxOverlay)}</span>
          </div>
        </div>

        <div className="world-panel">
          <Canvas dpr={[1, 1.5]} camera={{ position: [0, -18, 8.5], fov: 50, near: 0.05, far: 100 }}>
            <color attach="background" args={["#111315"]} />
            <ambientLight intensity={1.8} />
            <directionalLight position={[0, -4, 8]} intensity={2.2} />
            <OrbitRig world={world} />
            <CourtSurface world={world} />
            <CourtLines world={world} />
            <NetAssembly world={world} />
            <ReplayGlbLayer replayScene={replayScene} replaySceneUrl={manifest?.replay_scene_url ?? null} currentTime={currentTime} />
            <Players world={world} currentTime={currentTime} activeContactPlayerIds={viewerContactPlayerIds} />
            <SolidBodyMeshes meshes={activeBodyMeshes} />
            <Ball world={world} currentTime={currentTime} />
          </Canvas>
          {coverageGapActive ? <div className="world-warning">No player artifact coverage after {coverage.lastTime?.toFixed(2)}s</div> : null}
          <div className="scene-legend">
            <span><i className="swatch floor" /> floor</span>
            <span><i className="swatch mesh" /> BODY mesh</span>
            <span><i className="swatch joints" /> BODY joints</span>
            <span><i className="swatch ball" /> ball</span>
          </div>
        </div>
      </section>

      <section className="details-band">
        <p>Physics modes: {stats.physicsModes.length ? stats.physicsModes.join(", ") : "none"}</p>
        <p>{physics ? `Physics artifact: ${physics.physics}; FOOT-2 done: ${String(physics.foot2_done)}` : "Physics artifact: none"}</p>
        <p>{replayScene ? replaySceneReadout(replayScene, activeReplayPoint?.id ?? null) : "Replay scene: none"}</p>
        <p>{annotationSourceReadout(trustedAnnotationSources)}</p>
        <p>{coverageReadout(coverage, videoDuration)}</p>
        <p>Max floor penetration: {stats.maxFloorPenetrationM.toFixed(4)} m</p>
        <p>{manifest?.notes[0] ?? "Review-only viewer. Artifact gates stay separate from visual inspection."}</p>
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: number | string }) {
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

function ReplayGlbLayer({
  replayScene,
  replaySceneUrl,
  currentTime,
}: {
  replayScene: ReplayScene | null;
  replaySceneUrl: string | null;
  currentTime: number;
}) {
  const activePoint = useMemo(
    () => (replayScene ? activeReplayPointForTime(replayScene, currentTime) : undefined),
    [replayScene, currentTime],
  );
  const urls = useMemo(() => {
    if (!replayScene || !replaySceneUrl) return [];
    const activeGlb = activePoint ? resolveReplaySceneAssetUrl(replaySceneUrl, activePoint.glb_url) : null;
    return [
      resolveReplaySceneAssetUrl(replaySceneUrl, replayScene.court_glb),
      activeGlb,
    ].filter((url): url is string => Boolean(url));
  }, [activePoint, replayScene, replaySceneUrl]);

  return (
    <>
      {urls.map((url) => (
        <ReplayGlb key={url} url={url} />
      ))}
    </>
  );
}

function ReplayGlb({ url }: { url: string }) {
  const gltf = useLoader(GLTFLoader, url);
  const scene = useMemo(() => gltf.scene.clone(true), [gltf.scene]);
  useEffect(() => {
    scene.traverse((child) => {
      child.frustumCulled = false;
      if (child.name.toLowerCase().includes("ball")) {
        child.visible = false;
      }
    });
  }, [scene]);
  return <primitive object={scene} />;
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
  const floor = floorWorldForFrame(frame);
  const track = player.frames.map(floorWorldForFrame).filter(isVec3);
  const bodyJoints = frame?.joints_world ?? [];
  const meshPoints = sampleMeshPoints(frame?.mesh_vertices_world ?? [], isBallContactActive ? 1800 : 850);
  const proxySkeleton = bodyJoints.length ? null : skeletonForFrame(frame);
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
      {bodyJoints.length ? <PointCloud points={bodyJoints} color="#ffb45d" size={isBallContactActive ? 0.07 : 0.055} /> : null}
      {meshPoints.length ? (
        <PointCloud points={meshPoints} color={isBallContactActive ? "#b4f2bf" : "#86c7ff"} size={isBallContactActive ? 0.024 : 0.018} />
      ) : null}
    </>
  );
}

function SolidBodyMeshes({ meshes }: { meshes: ActiveBodyMeshFrame[] }) {
  return (
    <>
      {meshes.map(({ playerId, frame }) => (
        <SolidBodyMesh key={`${playerId}-${frame.frame_idx}`} frame={frame} />
      ))}
    </>
  );
}

function SolidBodyMesh({ frame }: { frame: ActiveBodyMeshFrame["frame"] }) {
  const geometry = useMemo(
    () => geometryFromIndexedMesh(frame.mesh_vertices_world, frame.mesh_faces),
    [frame.mesh_faces, frame.mesh_vertices_world],
  );
  const opacity = bodyMeshOpacityFromBlendWeight(frame);
  return (
    <mesh geometry={geometry} renderOrder={20}>
      <meshStandardMaterial color="#b4f2bf" emissive="#102d18" roughness={0.58} metalness={0.02} transparent opacity={opacity} side={DoubleSide} depthWrite={false} />
    </mesh>
  );
}

function Ball({ world, currentTime }: { world: VirtualWorld; currentTime: number }) {
  const info = ballRenderInfoForTime(world, currentTime);
  if (!info.frame?.world_xyz || !info.render3d) return null;
  const isApprox = info.mode === "court_plane_projection";
  return (
    <mesh position={info.frame.world_xyz}>
      <sphereGeometry args={[0.055, 16, 16]} />
      <meshStandardMaterial color={isApprox ? "#ffcf5a" : "#e8ff34"} emissive={isApprox ? "#4f3f00" : "#526000"} />
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

function replaySceneReadout(scene: ReplayScene, activePointId: number | null): string {
  const pointCount = scene.points.length;
  const totalMb = scene.points.reduce((total, point) => total + point.size_mb, 0);
  const active = activePointId === null ? "none" : `point ${activePointId}`;
  return `Replay scene: ${pointCount} static review point${pointCount === 1 ? "" : "s"}, ${totalMb.toFixed(3)} MB GLB refs, active ${active}`;
}

function annotationSourceReadout(sources: ViewerManifest["annotation_sources"]): string {
  if (!sources.length) return "Trusted annotation sources: none";
  return `Trusted annotation sources: ${sources.length} (${sources.map((source) => source.clip_id).join(", ")})`;
}

function ballRenderText(mode: ReturnType<typeof ballRenderInfoForTime>["mode"]): string {
  if (mode === "calibrated_3d") return "ball: calibrated 3D";
  if (mode === "court_plane_projection") return "ball: court-plane approx";
  if (mode === "off_court_projection") return "ball: off-court hidden";
  return "ball: missing";
}

function coverageReadout(coverage: ReturnType<typeof playerCoverageStats>, videoDuration: number): string {
  if (coverage.firstTime === null || coverage.lastTime === null) return "Player artifact coverage: none";
  const suffix = videoDuration > 0 ? ` of ${videoDuration.toFixed(2)}s video` : "";
  return `Player artifact coverage: ${coverage.firstTime.toFixed(2)}-${coverage.lastTime.toFixed(2)}s${suffix}`;
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

function geometryFromIndexedMesh(points: Vec3[], faces: Array<[number, number, number]>) {
  const geometry = geometryFromPoints(points);
  geometry.setIndex(faces.flat());
  geometry.computeVertexNormals();
  return geometry;
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

function floorWorldForFrame(frame: VirtualWorldFrame | undefined): Vec3 | null {
  return frame?.floor_world_xyz ?? (frame?.track_world_xy ? ([frame.track_world_xy[0], frame.track_world_xy[1], 0] as Vec3) : null);
}

function skeletonForFrame(frame: VirtualWorldFrame | undefined): { joints: Vec3[]; bones: Vec3[] } | null {
  const floor = floorWorldForFrame(frame);
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

function xywhToXyxy(value?: number[]): [number, number, number, number] | null {
  if (!Array.isArray(value) || value.length < 4) return null;
  return [value[0], value[1], value[0] + value[2], value[1] + value[3]];
}

function isVec3(value: Vec3 | null | undefined): value is Vec3 {
  return Array.isArray(value) && value.length === 3;
}
