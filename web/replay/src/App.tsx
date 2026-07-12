import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { BufferAttribute, BufferGeometry, Color, DoubleSide, Quaternion, Spherical, Vector3 as ThreeVector3, type Camera } from "three";
import { MapControls } from "three/examples/jsm/controls/MapControls.js";
import { MeshoptDecoder } from "three/examples/jsm/libs/meshopt_decoder.module.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

import { activeReplayPointForTime, parseReplayScene, resolveReplaySceneAssetUrl, type ReplayScene } from "./replayScene";
import { CourtMapPanel } from "./CourtMapPanel";
import {
  parseBallArcRender,
  replayViewFromSearch,
  sampleBallArcRenderAtTime,
  type BallArcRender,
  type ReplayViewMode,
} from "./ballArcRender";
import { BallHonestyHud } from "./components/modules/BallHonestyHud";
import { BallTrailLayer } from "./components/modules/BallTrailLayer";
import { ImpactMarkers } from "./components/modules/ImpactMarkers";
import {
  ballHudStateForTime,
  parseAutoBounceCandidates,
  parseBallTrailArtifact,
  samplesFromVirtualWorld,
  type BallTrailArtifact,
  type BounceCandidate,
} from "./components/modules/ballTrail";
import { UploadPanel } from "./UploadPanel";
import {
  buildShotTrailGroups,
  filterShots,
  parseBallArcSolved,
  parseShots,
  qualityBandForShot,
  shotOutcomeColor,
  shotTypeLabel,
  type BallArcSolved,
  type RacketsportShots,
  type ShotRecord,
  type ShotTrailFilters,
  type ShotTrailGroup,
} from "./shotTrails";
import {
  activeBallContactPlayerIds,
  ballCoverageKpiReadout,
  ballTrailSegmentsForTime,
  ballRenderInfoForTime,
  bodyMeshDebugSnapshot,
  bodyMeshIndexWindowForTime,
  bodyMeshInterpolationReadout,
  bodyMeshStatusTileValue,
  contactEventCount,
  displayFpsReadout,
  displayFpsReplayData,
  effectiveTrustBadge,
  fetchBodyMeshChunk,
  frameForTime,
  labelOverlayForTime,
  labelViewBox,
  parseBallArcEventsSelected,
  parseBallInflections,
  parseBodyMesh,
  parseBodyMeshFaces,
  parseBodyMeshIndex,
  parseCoachingCardFacts,
  parseContactWindows,
  parseLabelOverlayPayload,
  parsePhysicsRefinement,
  parseRallySpans,
  parseReviewedBounces,
  parseViewerManifest,
  parseVirtualWorld,
  resolveCanonicalPlaybackTime,
  resolveViewerManifestUrls,
  activePaddleFramesForTime,
  playerPresenceForTime,
  playerTrailPointsForTime,
  playerCoverageStats,
  resolveBodyMeshAssetUrl,
  solidBodyMeshFramesForTime,
  solidMeshRenderedPlayerCount,
  startTimeFromSearch,
  timelineChaptersFromMarkers,
  timelineChaptersFromRallySpans,
  timelineEventJump,
  timelineMarkersFromArtifacts,
  trustBadgeColor,
  trustBandChipText,
  videoBallOverlayForTime,
  worldWarningsReadout,
  type ActiveBodyMeshFrame,
  type ActivePaddleFrame,
  type BallArcEventsSelected,
  type BallInflections,
  type BodyMesh,
  type BodyMeshDebugSnapshot,
  type BodyMeshFaces,
  type BodyMeshIndex,
  type BodyMeshLoadStatus,
  type CoachingCardFacts,
  type ContactWindows,
  type DisplayFpsReplayData,
  type LabelOverlayPayload,
  type PhysicsRefinement,
  type ReviewedBounces,
  type RallySpans,
  type TimelineChapter,
  type TimelineMarker,
  type TrustBand,
  type TrustBadge,
  type Vec3,
  type VideoBallOverlay,
  type ViewerManifest,
  type VirtualWorld,
  type VirtualWorldFrame,
  type VirtualWorldPaddleFrame,
  type VirtualWorldPlayer,
  worldStats,
} from "./viewerData";
import {
  VIEW_LAYER_DEFINITIONS,
  applyViewPreset,
  clearFocus,
  entityFocusStyle,
  eventMarkersForTime,
  loadPersistedViewState,
  persistViewState,
  sceneLayerSnapshotForTime,
  toggleViewLayer,
  viewStateToSearch,
  type EntityFocusStyle,
  type ViewLayerDefinition,
  type ViewLayerKey,
  type ViewState,
  type WorldEventMarker,
} from "./viewState";

export { CoachingCardPanel, coachingCardForTimeline } from "./CoachingCard";

export type ProductCameraPreset = "court" | "follow_player" | "free_orbit";
export type CameraPreset = ProductCameraPreset | "broadcast" | "behind_baseline" | "top_down" | "paddle_review" | "shot_trails";
type CameraDragKind = "pan" | "orbit";
type CameraDragCommand = { kind: CameraDragKind; dx: number; dy: number; seq: number };

export function hasExplicitReviewStartTime(search: string): boolean {
  const params = new URLSearchParams(search);
  return params.has("t") || params.has("time");
}

export function defaultReviewStartTime(world: VirtualWorld): number {
  const playerTimesByPlayer = world.players.map((player) =>
    player.frames.filter(frameHasReviewGeometry).map((frame) => frame.t).filter((time) => Number.isFinite(time) && time >= 0),
  );
  const playerTimes = playerTimesByPlayer.flat();
  if (playerTimes.length) {
    const tolerance = 0.55 / Math.max(1, world.fps || 30);
    const candidates = Array.from(new Set(playerTimes)).sort((a, b) => a - b);
    let best = candidates[0] ?? 0;
    let bestVisiblePlayers = 0;
    for (const candidate of candidates) {
      const visiblePlayers = playerTimesByPlayer.filter((times) => times.some((time) => Math.abs(time - candidate) <= tolerance)).length;
      if (visiblePlayers > bestVisiblePlayers) {
        best = candidate;
        bestVisiblePlayers = visiblePlayers;
      }
    }
    return best;
  }

  const ballTimes = world.ball.frames.map((frame) => frame.t).filter((time) => Number.isFinite(time) && time >= 0);
  return ballTimes.length ? Math.min(...ballTimes) : 0;
}

export function shouldRenderReplayScenePointClouds(viewState: Pick<ViewState, "layers">, replayScene: ReplayScene | null, currentTime: number): boolean {
  return viewState.layers.debugPointClouds && replayScene !== null && activeReplayPointForTime(replayScene, currentTime) !== undefined;
}

function frameHasReviewGeometry(frame: VirtualWorldFrame): boolean {
  return Boolean(
    frame.floor_world_xyz ||
      frame.track_world_xy ||
      frame.mesh_ref ||
      frame.joints_world.length > 0 ||
      frame.mesh_vertices_world.length > 0,
  );
}

export type FpsSample = { framesSinceReport: number; windowStartMs: number; fps: number };

/** Roll a `performance.now()`-driven frame counter into a fps reading, refreshed every ~500ms. */
export function updateFpsSample(sample: FpsSample, nowMs: number): FpsSample {
  const framesSinceReport = sample.framesSinceReport + 1;
  const elapsedMs = nowMs - sample.windowStartMs;
  if (elapsedMs >= 500) {
    return { framesSinceReport: 0, windowStartMs: nowMs, fps: (framesSinceReport / elapsedMs) * 1000 };
  }
  return { framesSinceReport, windowStartMs: sample.windowStartMs, fps: sample.fps };
}

const DEFAULT_REPLAY_MANIFEST_URL = import.meta.env.VITE_REPLAY_MANIFEST_URL?.trim() || null;
const LOCAL_REPO_ROOT = import.meta.env.VITE_PICKLEBALL_REPO_ROOT?.trim() || "/Users/arnavchokshi/Desktop/pickleball";

export type RecentReplayRun = {
  id: string;
  label: string;
  clip: string;
  runLabel: string;
  updatedLabel: string;
  manifestUrl: string;
};

function localManifestUrl(relativePath: string): string {
  return `/@fs/${LOCAL_REPO_ROOT.replace(/\/+$/, "")}/${relativePath}`;
}

export const RECENT_REPLAY_RUNS: readonly RecentReplayRun[] = [
  {
    id: "burlington",
    label: "Burlington",
    clip: "burlington_gold_0300_low_steep_corner",
    runLabel: "ball_f1_three_clip_runs",
    updatedLabel: "Jul 5 2:09 PM",
    manifestUrl: localManifestUrl(
      "runs/lanes/ball_f1_three_clip_runs_20260705/burlington_gold_0300_low_steep_corner/replay_viewer_manifest.json",
    ),
  },
  {
    id: "wolverine",
    label: "Wolverine",
    clip: "wolverine_mixed_0200_mid_steep_corner",
    runLabel: "visual1_wolverine",
    updatedLabel: "Jul 5 3:14 PM",
    manifestUrl: localManifestUrl(
      "runs/visual1_wolverine_20260705T220517Z/wolverine_mixed_0200_mid_steep_corner/replay_viewer_manifest.json",
    ),
  },
  {
    id: "outdoor",
    label: "Outdoor",
    clip: "outdoor_webcam_iynbd_1500_long_high_baseline",
    runLabel: "ball_f1_three_clip_runs",
    updatedLabel: "Jul 5 2:09 PM",
    manifestUrl: localManifestUrl(
      "runs/lanes/ball_f1_three_clip_runs_20260705/outdoor_webcam_iynbd_1500_long_high_baseline/replay_viewer_manifest.json",
    ),
  },
];

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

export function recentRunHref(run: Pick<RecentReplayRun, "manifestUrl">, replayViewMode: ReplayViewMode, pathname = "/"): string {
  const params = new URLSearchParams();
  params.set("manifest", run.manifestUrl);
  if (replayViewMode === "courtmap") params.set("view", "courtmap");
  return `${pathname}?${params.toString()}`;
}

function sameManifestUrl(left: string | null, right: string): boolean {
  return (left ?? "").trim() === right.trim();
}

export function coachingFactsUrlFromSearch(
  search: string,
  manifest: Pick<ViewerManifest, "coaching_card_facts_url"> | null,
): string | null {
  const explicit = new URLSearchParams(search).get("coaching");
  if (explicit && explicit.trim()) return explicit.trim();
  return manifest?.coaching_card_facts_url?.trim() || null;
}

export function bodyMeshOpacityFromBlendWeight(
  frame: Pick<ActiveBodyMeshFrame["frame"], "blend_weight">,
  presenceOpacity = 1,
): number {
  return (0.26 + Math.max(0, Math.min(1, frame.blend_weight)) * 0.42) * Math.max(0, Math.min(1, presenceOpacity));
}

export function adjacentBodyMeshWindow(index: BodyMeshIndex, activeWindowId: number, direction: 1 | -1) {
  const windows = [...index.windows].sort((left, right) => left.t0 - right.t0);
  const activeIndex = windows.findIndex((window) => window.source_window_index === activeWindowId);
  return activeIndex < 0 ? null : windows[activeIndex + direction] ?? null;
}

export type BodyMeshTrustMaterial = {
  fillColor: string;
  emissiveColor: string;
  opacityScale: number;
  label: "solid" | "estimated";
};

export function bodyMeshMaterialForTrustBadge(badge: TrustBadge | undefined): BodyMeshTrustMaterial {
  if (badge === undefined || badge === "preview") {
    return { fillColor: "#ffb454", emissiveColor: "#5a3500", opacityScale: 0.62, label: "estimated" };
  }
  if (badge === "low_confidence") {
    return { fillColor: "#8a8f98", emissiveColor: "#15171b", opacityScale: 0.42, label: "estimated" };
  }
  return { fillColor: "#b4f2bf", emissiveColor: "#102d18", opacityScale: 1, label: "solid" };
}

export function configureGltfLoader(loader: GLTFLoader): GLTFLoader {
  loader.setMeshoptDecoder(MeshoptDecoder);
  return loader;
}

export type SolidBodyMeshGeometryCache = {
  geometries: Map<string, BufferGeometry>;
  normalComputeCount: number;
  readonly geometryCount: number;
  dispose: () => void;
};

export function createSolidBodyMeshGeometryCache(bodyMesh: BodyMesh | null): SolidBodyMeshGeometryCache {
  const cache: SolidBodyMeshGeometryCache = {
    geometries: new Map<string, BufferGeometry>(),
    normalComputeCount: 0,
    get geometryCount() {
      return this.geometries.size;
    },
    dispose() {
      for (const geometry of this.geometries.values()) {
        geometry.dispose();
      }
      this.geometries.clear();
    },
  };
  for (const player of bodyMesh?.players ?? []) {
    for (const frame of player.frames) {
      if (frame.mesh_vertices_world.length === 0 || frame.mesh_faces.length === 0) continue;
      geometryForSolidBodyMeshFrame(cache, player.id, frame);
    }
  }
  return cache;
}

export function geometryForSolidBodyMeshFrame(
  cache: SolidBodyMeshGeometryCache,
  playerId: number,
  frame: ActiveBodyMeshFrame["frame"],
): BufferGeometry {
  const key = solidBodyMeshGeometryKey(playerId, frame);
  const cached = cache.geometries.get(key);
  if (cached) return cached;
  const geometry = geometryFromIndexedMesh(frame.mesh_vertices_world, frame.mesh_faces);
  cache.normalComputeCount += 1;
  cache.geometries.set(key, geometry);
  return geometry;
}

function solidBodyMeshGeometryKey(playerId: number, frame: ActiveBodyMeshFrame["frame"]): string {
  if (frame.mesh_interpolated && frame.interpolation) {
    return `${playerId}:interp:${frame.interpolation.from_frame_idx}:${frame.interpolation.to_frame_idx}:${frame.interpolation.alpha}`;
  }
  return `${playerId}:${frame.frame_idx}`;
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

export type EntityLayerCounts = {
  playerMeshCount: number;
  playerSkeletonCount: number;
  ballTrailCount: number;
  paddleCount: number;
  contactSurfaceCount: number;
  targetZoneCount: number;
  ghostPositionCount: number;
  contactSurfaceDeclared?: boolean;
  targetZoneDeclared?: boolean;
  ghostPositionDeclared?: boolean;
};

export function entityLayerEmptyStates(layers: ViewState["layers"], counts: EntityLayerCounts): string[] {
  const messages: string[] = [];
  if (layers.playerSolidMeshes && counts.playerMeshCount === 0) messages.push("Player meshes: no data at this time");
  if (layers.playerSkeletons && counts.playerSkeletonCount === 0) messages.push("Player skeletons: no data at this time");
  if (layers.ballTrail && counts.ballTrailCount === 0) messages.push("Ball trail: no data at this time");
  if (layers.paddles && counts.paddleCount === 0) messages.push("Paddles: no data at this time");
  if (layers.contactSurfaces && counts.contactSurfaceCount === 0) messages.push(counts.contactSurfaceDeclared ? "Contact surfaces: referenced, renderer not ready" : "Contact surfaces: artifact not supplied");
  if (layers.targetZones && counts.targetZoneCount === 0) messages.push(counts.targetZoneDeclared ? "Target zones: referenced, renderer not ready" : "Target zones: artifact not supplied");
  if (layers.ghostPositioning && counts.ghostPositionCount === 0) messages.push(counts.ghostPositionDeclared ? "Ghost positioning: referenced, renderer not ready" : "Ghost positioning: artifact not supplied");
  return messages;
}

const CAMERA_PRESET_LABELS: Record<CameraPreset, string> = {
  court: "Court",
  follow_player: "Follow player",
  free_orbit: "Free orbit",
  broadcast: "Broadcast",
  behind_baseline: "Behind",
  top_down: "Top",
  paddle_review: "Paddles",
  shot_trails: "Trails",
};

const PRODUCT_CAMERA_PRESETS: readonly ProductCameraPreset[] = ["court", "follow_player", "free_orbit"];
const CAMERA_PREFERENCE_STORAGE_KEY = "pickleball.replay.camera.v2";

export type CameraPreference = { preset: ProductCameraPreset; playerId: number | null };
type ViewerStorage = Pick<Storage, "getItem" | "setItem">;

export function loadCameraPreference(storage?: ViewerStorage | null): CameraPreference {
  const fallback: CameraPreference = { preset: "court", playerId: null };
  if (!storage) return fallback;
  try {
    const raw = storage.getItem(CAMERA_PREFERENCE_STORAGE_KEY);
    if (!raw) return fallback;
    const value = JSON.parse(raw) as Partial<CameraPreference>;
    const preset = PRODUCT_CAMERA_PRESETS.includes(value.preset as ProductCameraPreset) ? value.preset as ProductCameraPreset : "court";
    const playerId = typeof value.playerId === "number" && Number.isInteger(value.playerId) ? value.playerId : null;
    return { preset, playerId };
  } catch {
    return fallback;
  }
}

export function persistCameraPreference(preference: CameraPreference, storage?: ViewerStorage | null): void {
  if (!storage) return;
  try {
    storage.setItem(CAMERA_PREFERENCE_STORAGE_KEY, JSON.stringify(preference));
  } catch {
    // Storage can be denied; replay remains fully usable for this session.
  }
}

export function cameraTransitionDurationMs(prefersReducedMotion: boolean): number {
  return prefersReducedMotion ? 0 : 280;
}

function viewerStorage(): ViewerStorage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

const DEFAULT_SHOT_TRAIL_FILTERS: ShotTrailFilters = {
  playerId: null,
  shotType: "all",
  outcome: "all",
  quality: "all",
};

const INITIAL_BODY_MESH_STATUS: BodyMeshLoadStatus = { state: "idle", label: "mesh: idle" };

function meshLoadStatus(
  state: BodyMeshLoadStatus["state"],
  options: Omit<BodyMeshLoadStatus, "state" | "label"> = {},
): BodyMeshLoadStatus {
  return { state, label: `mesh: ${state}`, ...options };
}

function meshFailureStatus(
  stage: NonNullable<BodyMeshLoadStatus["stage"]>,
  url: string,
  error: unknown,
  windowId?: number | null,
): BodyMeshLoadStatus {
  const message = errorMessage(error);
  const state: BodyMeshLoadStatus["state"] =
    stage === "chunk" && message.startsWith("failed to fetch body mesh chunk") ? "fetch_failed" : stage === "chunk" ? "decode_failed" : "parse_failed";
  return meshLoadStatus(state, { stage, url, windowId: windowId ?? null, message });
}

function warnBodyMeshFailure({
  stage,
  url,
  error,
  windowId,
}: {
  stage: NonNullable<BodyMeshLoadStatus["stage"]>;
  url: string;
  error: unknown;
  windowId?: number | null;
}) {
  console.warn("[body-mesh]", {
    stage,
    url,
    windowId: windowId ?? null,
    error,
  });
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function scheduleViewerIdleWork(callback: () => void): () => void {
  const idleWindow = window as Window & {
    requestIdleCallback?: (handler: () => void, options?: { timeout?: number }) => number;
    cancelIdleCallback?: (handle: number) => void;
  };
  if (idleWindow.requestIdleCallback && idleWindow.cancelIdleCallback) {
    const handle = idleWindow.requestIdleCallback(callback, { timeout: 120 });
    return () => idleWindow.cancelIdleCallback?.(handle);
  }
  const handle = window.setTimeout(callback, 0);
  return () => window.clearTimeout(handle);
}

export default function App() {
  const initialTime = useMemo(() => startTimeFromSearch(window.location.search), []);
  const hasExplicitInitialTime = useMemo(() => hasExplicitReviewStartTime(window.location.search), []);
  const initialCameraPreference = useMemo(() => loadCameraPreference(viewerStorage()), []);
  const [manifest, setManifest] = useState<ViewerManifest | null>(null);
  const [world, setWorld] = useState<VirtualWorld>(() => parseVirtualWorld(sampleWorld));
  const [labelOverlay, setLabelOverlay] = useState<LabelOverlayPayload>(() => parseLabelOverlayPayload(null));
  const [physics, setPhysics] = useState<PhysicsRefinement | null>(null);
  const [contactWindows, setContactWindows] = useState<ContactWindows | null>(null);
  const [ballInflections, setBallInflections] = useState<BallInflections | null>(null);
  const [ballArcEventsSelected, setBallArcEventsSelected] = useState<BallArcEventsSelected | null>(null);
  const [reviewedBounces, setReviewedBounces] = useState<ReviewedBounces | null>(null);
  const [rallySpans, setRallySpans] = useState<RallySpans | null>(null);
  const [coachingFacts, setCoachingFacts] = useState<CoachingCardFacts | null>(null);
  const [shots, setShots] = useState<RacketsportShots | null>(null);
  const [ballArcSolved, setBallArcSolved] = useState<BallArcSolved | null>(null);
  const [ballArcRender, setBallArcRender] = useState<BallArcRender | null>(null);
  const [ballTrailArtifact, setBallTrailArtifact] = useState<BallTrailArtifact | null>(null);
  const [autoBounceCandidates, setAutoBounceCandidates] = useState<BounceCandidate[]>([]);
  const [shotTrailFilters, setShotTrailFilters] = useState<ShotTrailFilters>(DEFAULT_SHOT_TRAIL_FILTERS);
  const [selectedShotId, setSelectedShotId] = useState<string | null>(null);
  const [correctionStatus, setCorrectionStatus] = useState<string | null>(null);
  const [bodyMesh, setBodyMesh] = useState<BodyMesh | null>(null);
  const [bodyMeshIndex, setBodyMeshIndex] = useState<BodyMeshIndex | null>(null);
  const [bodyMeshFaces, setBodyMeshFaces] = useState<BodyMeshFaces | null>(null);
  const [bodyMeshLoadStatus, setBodyMeshLoadStatus] = useState<BodyMeshLoadStatus>(INITIAL_BODY_MESH_STATUS);
  const [displayFpsEnabled, setDisplayFpsEnabled] = useState(false);
  const [displayFpsProcessing, setDisplayFpsProcessing] = useState(false);
  const [displayFpsData, setDisplayFpsData] = useState<DisplayFpsReplayData | null>(null);
  const [replayScene, setReplayScene] = useState<ReplayScene | null>(null);
  const [currentTime, setCurrentTime] = useState(initialTime);
  const [videoDuration, setVideoDuration] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [capabilityErrors, setCapabilityErrors] = useState<Record<string, string>>({});
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>(initialCameraPreference.preset);
  const [cameraFollowPlayerId, setCameraFollowPlayerId] = useState<number | null>(initialCameraPreference.playerId);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(() =>
    typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  const [replayViewMode, setReplayViewMode] = useState<ReplayViewMode>(() => replayViewFromSearch(window.location.search));
  const [cameraResetToken, setCameraResetToken] = useState(0);
  const [cameraDragCommand, setCameraDragCommand] = useState<CameraDragCommand | null>(null);
  const [showShotsPanel, setShowShotsPanel] = useState(false);
  const [viewState, setViewState] = useState<ViewState>(() => loadPersistedViewState(window.location.search, viewerStorage()));
  const [fps, setFps] = useState(0);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const bodyMeshChunkCacheRef = useRef<Map<number, BodyMesh>>(new Map());
  const bodyMeshChunkInflightRef = useRef<Map<number, Promise<BodyMesh>>>(new Map());
  const currentTimeRef = useRef(0);
  const playbackDirectionRef = useRef<1 | -1>(1);
  const preferredInitialTimeRef = useRef(initialTime);
  const cameraDragSeqRef = useRef(0);
  const initialSeekAppliedRef = useRef(false);

  useEffect(() => {
    const manifestUrl = manifestUrlFromSearch(window.location.search);
    if (manifestUrl === null) {
      setManifest(null);
      setBodyMesh(null);
      setBodyMeshLoadStatus(INITIAL_BODY_MESH_STATUS);
      setBallInflections(null);
      setBallArcEventsSelected(null);
      setReviewedBounces(null);
      setRallySpans(null);
      setCoachingFacts(null);
      setShots(null);
      setBallArcSolved(null);
      setBallArcRender(null);
      setBallTrailArtifact(null);
      setAutoBounceCandidates([]);
      setSelectedShotId(null);
      setCorrectionStatus(null);
      setBodyMeshIndex(null);
      setBodyMeshFaces(null);
      setDisplayFpsData(null);
      setDisplayFpsProcessing(false);
      setLoadError(null);
      setCapabilityErrors({});
      return;
    }
    const resolvedManifestUrl = manifestUrl;
    let cancelled = false;
    async function load() {
      try {
        const manifestPayload = resolveViewerManifestUrls(
          parseViewerManifest(await fetchJson(resolvedManifestUrl)),
          resolvedManifestUrl,
        );
        const worldPayload = parseVirtualWorld(await fetchJson(manifestPayload.virtual_world_url));
        const optionalErrors: Record<string, string> = {};
        const recordOptionalError = (capability: string, error: unknown) => {
          optionalErrors[capability] = error instanceof Error ? error.message : String(error);
        };
        const reviewStartTime = hasExplicitInitialTime ? initialTime : defaultReviewStartTime(worldPayload);
        const firstOverlay = manifestPayload.label_overlays.find((overlay) => overlay.kind === "player_boxes");
        const labelPayload = firstOverlay
          ? await loadOptionalArtifact("player labels", () => fetchJson(firstOverlay.url), recordOptionalError)
          : null;
        const physicsPayload = manifestPayload.physics_refinement_url
          ? await loadOptionalArtifact("physics", async () => parsePhysicsRefinement(await fetchJson(manifestPayload.physics_refinement_url!)), recordOptionalError)
          : null;
        const contactPayload = manifestPayload.contact_windows_url
          ? await loadOptionalArtifact("contacts", async () => parseContactWindows(await fetchJson(manifestPayload.contact_windows_url!)), recordOptionalError)
          : null;
        const ballInflectionsPayload = manifestPayload.ball_inflections_url
          ? await loadOptionalArtifact("ball inflections", async () => parseBallInflections(await fetchJson(manifestPayload.ball_inflections_url!)), recordOptionalError)
          : null;
        const ballArcEventsSelectedPayload = manifestPayload.events_selected_url
          ? await loadOptionalArtifact("selected events", async () => parseBallArcEventsSelected(await fetchJson(manifestPayload.events_selected_url!)), recordOptionalError)
          : null;
        const reviewedBouncesPayload = manifestPayload.reviewed_bounces_url
          ? await loadOptionalArtifact("reviewed bounces", async () => parseReviewedBounces(await fetchJson(manifestPayload.reviewed_bounces_url!)), recordOptionalError)
          : null;
        const rallySpansPayload = manifestPayload.rally_spans_url
          ? await loadOptionalArtifact("rally spans", async () => parseRallySpans(await fetchJson(manifestPayload.rally_spans_url!)), recordOptionalError)
          : null;
        const replayScenePayload = manifestPayload.replay_scene_url
          ? await loadOptionalArtifact("replay scene", async () => parseReplayScene(await fetchJson(manifestPayload.replay_scene_url!)), recordOptionalError)
          : null;
        let bodyMeshIndexPayload: BodyMeshIndex | null = null;
        let bodyMeshFacesPayload: BodyMeshFaces | null = null;
        if (manifestPayload.body_mesh_index_url) {
          const indexUrl = manifestPayload.body_mesh_index_url;
          let indexJson: unknown;
          try {
            indexJson = await fetchJson(indexUrl);
            bodyMeshIndexPayload = parseBodyMeshIndex(indexJson);
          } catch (error) {
            warnBodyMeshFailure({ stage: "index", url: indexUrl, error });
            if (!cancelled) setBodyMeshLoadStatus(meshFailureStatus("index", indexUrl, error));
            recordOptionalError("body mesh index", error);
          }
          if (bodyMeshIndexPayload) {
            const facesUrl = resolveBodyMeshAssetUrl(indexUrl, bodyMeshIndexPayload.faces_url);
            try {
              bodyMeshFacesPayload = parseBodyMeshFaces(await fetchJson(facesUrl));
            } catch (error) {
              warnBodyMeshFailure({ stage: "faces", url: facesUrl, error });
              if (!cancelled) setBodyMeshLoadStatus(meshFailureStatus("faces", facesUrl, error));
              recordOptionalError("body mesh faces", error);
            }
          }
        }
        const coachingFactsUrl = coachingFactsUrlFromSearch(window.location.search, manifestPayload);
        const coachingFactsPayload = coachingFactsUrl
          ? await loadOptionalArtifact("coaching facts", async () => parseCoachingCardFacts(await fetchJson(coachingFactsUrl)), recordOptionalError)
          : null;
        const shotsPayload = manifestPayload.shots_url
          ? await loadOptionalArtifact("shots", async () => parseShots(await fetchJson(manifestPayload.shots_url!)), recordOptionalError)
          : null;
        const ballArcPayload = manifestPayload.ball_arc_solved_url
          ? await loadOptionalArtifact("ball arc solve", async () => {
              const json = await fetchJson(manifestPayload.ball_arc_solved_url!);
              return { solved: parseBallArcSolved(json), trail: parseBallTrailArtifact(json) };
            }, recordOptionalError)
          : null;
        const ballArcSolvedPayload = ballArcPayload?.solved ?? null;
        const ballTrailArtifactPayload = ballArcPayload?.trail ?? null;
        const ballArcRenderPayload = manifestPayload.ball_arc_render_url
          ? await loadOptionalArtifact("ball arc render", async () => parseBallArcRender(await fetchJson(manifestPayload.ball_arc_render_url!)), recordOptionalError)
          : null;
        const autoBounceCandidatesUrl = manifestPayload.auto_bounce_candidates_url ?? manifestPayload.ball_bounce_candidates_url ?? null;
        const autoBounceCandidatesPayload = autoBounceCandidatesUrl
          ? (await loadOptionalArtifact("bounce candidates", async () => parseAutoBounceCandidates(await fetchJson(autoBounceCandidatesUrl)), recordOptionalError)) ?? []
          : [];
        if (cancelled) return;
        setManifest(manifestPayload);
        setWorld(worldPayload);
        preferredInitialTimeRef.current = reviewStartTime;
        setCurrentTime(reviewStartTime);
        setLabelOverlay(parseLabelOverlayPayload(labelPayload));
        setPhysics(physicsPayload);
        setContactWindows(contactPayload);
        setBallInflections(ballInflectionsPayload);
        setBallArcEventsSelected(ballArcEventsSelectedPayload);
        setReviewedBounces(reviewedBouncesPayload);
        setRallySpans(rallySpansPayload);
        setCoachingFacts(coachingFactsPayload);
        setShots(shotsPayload);
        setBallArcSolved(ballArcSolvedPayload);
        setBallArcRender(ballArcRenderPayload);
        setBallTrailArtifact(ballTrailArtifactPayload);
        setAutoBounceCandidates(autoBounceCandidatesPayload);
        setSelectedShotId(null);
        setCorrectionStatus(null);
        setBodyMeshIndex(bodyMeshIndexPayload);
        setBodyMeshFaces(bodyMeshFacesPayload);
        setBodyMesh(null);
        setBodyMeshLoadStatus(bodyMeshIndexPayload ? meshLoadStatus("index_ready", { stage: "index" }) : INITIAL_BODY_MESH_STATUS);
        setReplayScene(replayScenePayload);
        setLoadError(null);
        setCapabilityErrors(optionalErrors);
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
    const bodyMeshUrl = manifest?.body_mesh_url;
    if (manifest?.body_mesh_index_url) {
      return;
    }
    if (!bodyMeshUrl) {
      setBodyMesh(null);
      setBodyMeshLoadStatus(INITIAL_BODY_MESH_STATUS);
      return;
    }
    const resolvedBodyMeshUrl = bodyMeshUrl;
    let cancelled = false;
    async function loadBodyMesh() {
      try {
        const payload = parseBodyMesh(await fetchJson(resolvedBodyMeshUrl));
        if (!cancelled) {
          setBodyMesh(payload);
          setBodyMeshLoadStatus(meshLoadStatus("loaded", { stage: "legacy", url: resolvedBodyMeshUrl }));
        }
      } catch (error) {
        warnBodyMeshFailure({ stage: "legacy", url: resolvedBodyMeshUrl, error });
        if (!cancelled) {
          setBodyMesh(null);
          setBodyMeshLoadStatus(meshFailureStatus("legacy", resolvedBodyMeshUrl, error));
        }
      }
    }
    const timeoutId = window.setTimeout(() => {
      void loadBodyMesh();
    }, 2000);
    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [manifest?.body_mesh_index_url, manifest?.body_mesh_url]);

  useEffect(() => {
    bodyMeshChunkCacheRef.current.clear();
    bodyMeshChunkInflightRef.current.clear();
    if (manifest?.body_mesh_index_url) {
      setBodyMesh(null);
      setBodyMeshLoadStatus(meshLoadStatus("index_ready", { stage: "index" }));
    }
  }, [manifest?.body_mesh_index_url]);

  useEffect(() => {
    const indexUrl = manifest?.body_mesh_index_url;
    if (!indexUrl || !bodyMeshIndex || !bodyMeshFaces) {
      return;
    }
    const activeWindow = bodyMeshIndexWindowForTime(bodyMeshIndex, currentTime);
    if (!activeWindow) {
      setBodyMesh(null);
      setBodyMeshLoadStatus(meshLoadStatus("no_window", { stage: "chunk", windowId: null }));
      return;
    }
    const cacheKey = activeWindow.source_window_index;
    const chunkUrl = resolveBodyMeshAssetUrl(indexUrl, activeWindow.url);
    const cached = bodyMeshChunkCacheRef.current.get(cacheKey);
    if (cached) {
      bodyMeshChunkCacheRef.current.delete(cacheKey);
      bodyMeshChunkCacheRef.current.set(cacheKey, cached);
      setBodyMesh(cached);
      setBodyMeshLoadStatus(meshLoadStatus("loaded", { stage: "chunk", url: chunkUrl, windowId: cacheKey }));
      return;
    }
    let cancelled = false;
    const inflight =
      bodyMeshChunkInflightRef.current.get(cacheKey) ??
      fetchBodyMeshChunk(indexUrl, bodyMeshIndex, activeWindow, bodyMeshFaces);
    bodyMeshChunkInflightRef.current.set(cacheKey, inflight);
    setBodyMeshLoadStatus(meshLoadStatus("loading", { stage: "chunk", url: chunkUrl, windowId: cacheKey }));
    inflight
      .then((chunk) => {
        bodyMeshChunkInflightRef.current.delete(cacheKey);
        bodyMeshChunkCacheRef.current.set(cacheKey, chunk);
        while (bodyMeshChunkCacheRef.current.size > 6) {
          const oldest = bodyMeshChunkCacheRef.current.keys().next().value;
          if (oldest === undefined) break;
          bodyMeshChunkCacheRef.current.delete(oldest);
        }
        if (!cancelled) {
          setBodyMesh(chunk);
          setBodyMeshLoadStatus(meshLoadStatus("loaded", { stage: "chunk", url: chunkUrl, windowId: cacheKey }));
        }
      })
      .catch((error) => {
        bodyMeshChunkInflightRef.current.delete(cacheKey);
        warnBodyMeshFailure({ stage: "chunk", url: chunkUrl, windowId: cacheKey, error });
        if (!cancelled) {
          setBodyMesh(null);
          setBodyMeshLoadStatus(meshFailureStatus("chunk", chunkUrl, error, cacheKey));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [bodyMeshFaces, bodyMeshIndex, currentTime, manifest?.body_mesh_index_url]);

  useEffect(() => {
    const indexUrl = manifest?.body_mesh_index_url;
    if (!indexUrl || !bodyMeshIndex || !bodyMeshFaces) return;
    if (currentTime !== currentTimeRef.current) playbackDirectionRef.current = currentTime >= currentTimeRef.current ? 1 : -1;
    const activeWindow = bodyMeshIndexWindowForTime(bodyMeshIndex, currentTime);
    const adjacent = activeWindow
      ? adjacentBodyMeshWindow(bodyMeshIndex, activeWindow.source_window_index, playbackDirectionRef.current)
      : null;
    if (!adjacent) return;
    const key = adjacent.source_window_index;
    if (bodyMeshChunkCacheRef.current.has(key) || bodyMeshChunkInflightRef.current.has(key)) return;
    const inflight = fetchBodyMeshChunk(indexUrl, bodyMeshIndex, adjacent, bodyMeshFaces);
    bodyMeshChunkInflightRef.current.set(key, inflight);
    void inflight.then((chunk) => {
      bodyMeshChunkInflightRef.current.delete(key);
      bodyMeshChunkCacheRef.current.set(key, chunk);
      while (bodyMeshChunkCacheRef.current.size > 6) {
        const oldest = bodyMeshChunkCacheRef.current.keys().next().value;
        if (oldest === undefined) break;
        bodyMeshChunkCacheRef.current.delete(oldest);
      }
    }).catch((error) => {
      bodyMeshChunkInflightRef.current.delete(key);
      warnBodyMeshFailure({ stage: "chunk", url: resolveBodyMeshAssetUrl(indexUrl, adjacent.url), windowId: key, error });
    });
  }, [bodyMeshFaces, bodyMeshIndex, currentTime, manifest?.body_mesh_index_url]);

  useEffect(() => {
    if (!displayFpsEnabled) {
      setDisplayFpsData(null);
      setDisplayFpsProcessing(false);
      return;
    }
    let cancelled = false;
    setDisplayFpsProcessing(true);
    const cancelIdle = scheduleViewerIdleWork(() => {
      const doubled = displayFpsReplayData(world, bodyMesh, true);
      if (!cancelled) {
        setDisplayFpsData(doubled);
        setDisplayFpsProcessing(false);
      }
    });
    return () => {
      cancelled = true;
      cancelIdle();
    };
  }, [bodyMesh, displayFpsEnabled, world]);

  useEffect(() => {
    currentTimeRef.current = currentTime;
  }, [currentTime]);

  useEffect(() => {
    const nextSearch = viewStateToSearch(window.location.search, viewState);
    if (nextSearch !== window.location.search) {
      window.history.replaceState(null, "", `${window.location.pathname}${nextSearch}${window.location.hash}`);
    }
    persistViewState(viewState, viewerStorage());
  }, [viewState]);

  useEffect(() => {
    if (!PRODUCT_CAMERA_PRESETS.includes(cameraPreset as ProductCameraPreset)) return;
    persistCameraPreference({ preset: cameraPreset as ProductCameraPreset, playerId: cameraFollowPlayerId }, viewerStorage());
  }, [cameraFollowPlayerId, cameraPreset]);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setPrefersReducedMotion(media.matches);
    update();
    media.addEventListener?.("change", update);
    return () => media.removeEventListener?.("change", update);
  }, []);

  useEffect(() => {
    let animationFrame = 0;
    let lastSampleMs = 0;
    const minIntervalMs = 1000 / Math.min(60, Math.max(24, world.fps || 30));
    const tick = (now: number) => {
      const video = videoRef.current;
      if (video && !video.paused && now - lastSampleMs >= minIntervalMs) {
        lastSampleMs = now;
        const canonicalTime = resolveCanonicalPlaybackTime(video.currentTime, video.duration, currentTimeRef.current);
        if (Math.abs(canonicalTime - currentTimeRef.current) > 0.004) {
          currentTimeRef.current = canonicalTime;
          setCurrentTime(canonicalTime);
        }
      }
      animationFrame = window.requestAnimationFrame(tick);
    };
    animationFrame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(animationFrame);
  }, [manifest?.video_url, world.fps]);

  const displayFpsStats = displayFpsData?.stats ?? displayFpsReplayData(world, bodyMesh, false).stats;
  const displayFpsControlReadout = displayFpsEnabled && displayFpsProcessing ? `${displayFpsStats.displayFps}fps display: processing` : displayFpsReadout(displayFpsStats);
  const renderWorld = displayFpsEnabled && displayFpsData ? displayFpsData.world : world;
  const renderBodyMesh = displayFpsEnabled && displayFpsData ? displayFpsData.bodyMesh : bodyMesh;
  const stats = useMemo(() => worldStats(world), [world]);
  const coverage = useMemo(() => playerCoverageStats(world), [world]);
  const activeLabels = useMemo(() => labelOverlayForTime(labelOverlay, currentTime), [labelOverlay, currentTime]);
  const ballRenderInfo = useMemo(() => ballRenderInfoForTime(world, currentTime), [world, currentTime]);
  const videoBallOverlay = useMemo(() => videoBallOverlayForTime(world, currentTime), [world, currentTime]);
  const playerBoxOverlay = useMemo(
    () => manifest?.label_overlays.find((overlay) => overlay.kind === "player_boxes") ?? null,
    [manifest],
  );
  const activeContactPlayerIds = useMemo(
    () => activeBallContactPlayerIds(world, contactWindows, currentTime),
    [world, contactWindows, currentTime],
  );
  const activeBodyMeshes = useMemo(
    () => solidBodyMeshFramesForTime(renderBodyMesh, contactWindows, currentTime, renderWorld),
    [contactWindows, currentTime, renderBodyMesh, renderWorld],
  );
  const activePaddles = useMemo(() => activePaddleFramesForTime(world, currentTime), [currentTime, world]);
  const ballArcRenderSamples = useMemo(() => {
    const samples = ballArcRender?.samples ?? [];
    const current = sampleBallArcRenderAtTime(samples, currentTime);
    if (!current) return samples;
    const withoutDuplicate = samples.filter((sample) => Math.abs(sample.t - current.t) > 1e-6);
    return [...withoutDuplicate, current].sort((left, right) => left.t - right.t);
  }, [ballArcRender, currentTime]);
  const resolvedBallSamples = useMemo(
    () => ballArcRenderSamples.length
      ? ballArcRenderSamples
      : ballTrailArtifact?.samples.length
        ? ballTrailArtifact.samples
        : samplesFromVirtualWorld(world),
    [ballArcRenderSamples, ballTrailArtifact, world],
  );
  const playerPresenceGaps = useMemo(
    () => world.players.filter((player) => playerPresenceForTime(player, currentTime).missingEvidence),
    [currentTime, world.players],
  );
  const solidGeometryCache = useMemo(() => createSolidBodyMeshGeometryCache(renderBodyMesh), [renderBodyMesh]);
  useEffect(() => () => solidGeometryCache.dispose(), [solidGeometryCache]);
  const renderedSolidMeshPlayers = useMemo(() => solidMeshRenderedPlayerCount(activeBodyMeshes), [activeBodyMeshes]);
  const solidMeshTileValue = useMemo(
    () => bodyMeshStatusTileValue(renderedSolidMeshPlayers, bodyMeshLoadStatus),
    [bodyMeshLoadStatus, renderedSolidMeshPlayers],
  );
  const meshInterpolationReadout = useMemo(() => bodyMeshInterpolationReadout(renderBodyMesh), [renderBodyMesh]);
  const meshDebugSnapshot = useMemo(
    () =>
      bodyMeshDebugSnapshot({
        bodyMeshIndex,
        bodyMesh: renderBodyMesh,
        world: renderWorld,
        currentTime,
        loadStatus: bodyMeshLoadStatus,
        activeBodyMeshes,
      }),
    [activeBodyMeshes, bodyMeshIndex, bodyMeshLoadStatus, currentTime, renderBodyMesh, renderWorld],
  );
  const entityStaleReadout = useMemo(
    () => entityStaleAgeReadout(renderWorld, activeBodyMeshes, activePaddles, resolvedBallSamples, currentTime),
    [activeBodyMeshes, activePaddles, currentTime, renderWorld, resolvedBallSamples],
  );
  const sceneLayers = useMemo(
    () =>
      sceneLayerSnapshotForTime({
        world: renderWorld,
        bodyMesh: renderBodyMesh,
        contactWindows,
        ballArcEventsSelected,
        ballSamples: resolvedBallSamples,
        currentTime,
        viewState,
      }),
    [ballArcEventsSelected, contactWindows, currentTime, renderBodyMesh, renderWorld, resolvedBallSamples, viewState],
  );
  const worldEventMarkers = useMemo(
    () => eventMarkersForTime(world, contactWindows, currentTime),
    [contactWindows, currentTime, world],
  );
  const entityEmptyStates = useMemo(
    () => entityLayerEmptyStates(viewState.layers, {
      playerMeshCount: sceneLayers.playerSolidMeshes.objectCount,
      playerSkeletonCount: sceneLayers.playerSkeletons.objectCount,
      ballTrailCount: sceneLayers.ballTrail.objectCount,
      paddleCount: sceneLayers.paddles.objectCount,
      contactSurfaceCount: 0,
      targetZoneCount: 0,
      ghostPositionCount: 0,
      contactSurfaceDeclared: Boolean(manifest?.contact_surfaces_url),
      targetZoneDeclared: Boolean(manifest?.target_zones_url),
      ghostPositionDeclared: Boolean(manifest?.ghost_positions_url),
    }),
    [manifest?.contact_surfaces_url, manifest?.ghost_positions_url, manifest?.target_zones_url, sceneLayers, viewState.layers],
  );
  const eventMarkerEmpty =
    viewState.layers.eventMarkers &&
    worldEventMarkers.length === 0 &&
    autoBounceCandidates.length === 0 &&
    (contactWindows?.events.length ?? 0) === 0 &&
    (ballTrailArtifact?.segments.length ?? 0) === 0;
  const viewerContactPlayerIds = useMemo(
    () => contactPlayerIdsForViewer(activeContactPlayerIds, activeBodyMeshes),
    [activeContactPlayerIds, activeBodyMeshes],
  );
  const viewBox = useMemo(() => labelViewBox(labelOverlay), [labelOverlay]);
  const coverageGapActive = coverage.lastTime !== null && currentTime > coverage.lastTime + Math.max(0.12, 1 / (world.fps || 30));
  const contactReadout = contactReadoutText(activeContactPlayerIds, activeBodyMeshes);
  const ballReadout = ballRenderText(ballRenderInfo.mode, videoBallOverlay);
  const warningsReadout = useMemo(() => worldWarningsReadout(world), [world]);
  const ballCoverage = useMemo(
    () => ballHudStateForTime(resolvedBallSamples, currentTime).label.replace("ball: ", ""),
    [currentTime, resolvedBallSamples],
  );
  const timelineMarkers = useMemo(
    () => timelineMarkersFromArtifacts(contactWindows, ballInflections, reviewedBounces, shots),
    [contactWindows, ballInflections, reviewedBounces, shots],
  );
  const timelineDuration = videoDuration > 0 ? videoDuration : coverage.lastTime ?? 0;
  const timelineChapters = useMemo(
    () => {
      const authoritativeChapters = timelineChaptersFromRallySpans(rallySpans);
      return authoritativeChapters.length ? authoritativeChapters : timelineChaptersFromMarkers(timelineMarkers, timelineDuration);
    },
    [rallySpans, timelineMarkers, timelineDuration],
  );
  const playerIds = useMemo(() => world.players.map((player) => player.id), [world.players]);
  const shotTrailsMode = cameraPreset === "shot_trails";
  const filteredShots = useMemo(
    () => filterShots(shots?.shots ?? [], shotTrailFilters),
    [shotTrailFilters, shots],
  );
  const shotTrailGroups = useMemo(
    () => buildShotTrailGroups(filteredShots, ballArcSolved, world),
    [ballArcSolved, filteredShots, world],
  );
  const selectedShot = useMemo(
    () => (selectedShotId ? (shots?.shots ?? []).find((shot) => shot.shot_id === selectedShotId) ?? null : null),
    [selectedShotId, shots],
  );
  const shotFilterOptions = useMemo(() => shotTrailFilterOptions(shots?.shots ?? []), [shots]);
  const drawableShotCount = useMemo(
    () => shotTrailGroups.filter((group) => group.segments.length > 0).length,
    [shotTrailGroups],
  );
  const seekTo = (seconds: number) => {
    const video = videoRef.current;
    const duration = video && Number.isFinite(video.duration) ? video.duration : timelineDuration;
    const canonicalTime = resolveCanonicalPlaybackTime(seconds, duration, currentTimeRef.current);
    if (video) video.currentTime = canonicalTime;
    currentTimeRef.current = canonicalTime;
    setCurrentTime(canonicalTime);
  };

  const jumpToEvent = (direction: "previous" | "next") => {
    const eventTime = timelineEventJump(timelineMarkers, currentTime, direction);
    if (eventTime !== null) seekTo(eventTime);
  };

  const syncVideoTime = (video: HTMLVideoElement) => {
    const canonicalTime = resolveCanonicalPlaybackTime(video.currentTime, video.duration, currentTimeRef.current);
    if (Math.abs(canonicalTime - currentTimeRef.current) > 0.004) {
      currentTimeRef.current = canonicalTime;
      setCurrentTime(canonicalTime);
    }
  };

  const syncLoadedVideoTime = (video: HTMLVideoElement) => {
    if (Number.isFinite(video.duration)) {
      setVideoDuration(video.duration);
    }
    const preferredInitialTime = preferredInitialTimeRef.current;
    if (!initialSeekAppliedRef.current && preferredInitialTime > 0) {
      const duration = Number.isFinite(video.duration) ? video.duration : preferredInitialTime;
      video.currentTime = Math.min(preferredInitialTime, duration);
      initialSeekAppliedRef.current = true;
    }
    syncVideoTime(video);
  };

  const toggleLayer = (layer: ViewLayerKey) => {
    setViewState((state) => toggleViewLayer(state, layer));
  };

  const focusBall = () => {
    setViewState((state) => applyViewPreset(state, "ballFocus"));
  };

  const focusPlayer = (playerId: number) => {
    setViewState((state) => applyViewPreset(state, "playerFocus", { playerId }));
  };

  const updateShotFilter = (key: keyof ShotTrailFilters, value: string) => {
    setShotTrailFilters((filters) => ({
      ...filters,
      [key]: key === "playerId" ? (value === "all" ? null : Number(value)) : value,
    }));
    setSelectedShotId(null);
    setCorrectionStatus(null);
  };

  const selectShot = (shot: ShotRecord) => {
    setSelectedShotId(shot.shot_id);
    seekTo(shot.t);
  };

  const writeCorrection = (shot: ShotRecord) => {
    writeShotCorrectionPayload(manifest?.clip ?? shots?.clip_id ?? "unknown_clip", shot);
    setCorrectionStatus(`corrections.json updated for ${shot.shot_id}`);
  };

  const resetFocus = () => {
    setViewState((state) => clearFocus(state));
  };

  const resetView = () => {
    setViewState((state) => applyViewPreset(state, "default"));
  };

  const selectCameraPreset = (preset: CameraPreset) => {
    setCameraPreset(preset);
    setCameraResetToken((token) => token + 1);
  };

  const applyCameraDrag = (kind: CameraDragKind, dx: number, dy: number) => {
    if (dx === 0 && dy === 0) return;
    cameraDragSeqRef.current += 1;
    setCameraDragCommand({ kind, dx, dy, seq: cameraDragSeqRef.current });
  };

  const toggleShotsPanel = () => {
    if (showShotsPanel) {
      setShowShotsPanel(false);
      if (cameraPreset === "shot_trails") selectCameraPreset("court");
      return;
    }
    setShowShotsPanel(true);
    selectCameraPreset("shot_trails");
  };
  const selectReplayViewMode = (mode: ReplayViewMode) => {
    setReplayViewMode(mode);
    const params = new URLSearchParams(window.location.search);
    if (mode === "courtmap") {
      params.set("view", "courtmap");
    } else {
      params.delete("view");
    }
    const nextSearch = params.toString();
    window.history.replaceState(null, "", `${window.location.pathname}${nextSearch ? `?${nextSearch}` : ""}${window.location.hash}`);
  };
  const replayLoaded = manifest !== null;
  const currentManifestUrl = manifestUrlFromSearch(window.location.search);

  return (
    <main className="viewer-shell" aria-label="Replay viewer">
      <header className="viewer-header">
        <div className="viewer-brand">
          <div className="brand-mark" aria-hidden="true">
            <span />
          </div>
          <div>
            <p className="eyebrow">{replayLoaded ? "Replay review" : "Video intake"}</p>
            <h1 title={manifest?.clip ?? "Pickleball video"}>{manifest?.clip ?? "Pickleball video"}</h1>
          </div>
        </div>
        {replayLoaded ? (
          <div className="status-grid">
            <Metric label="Players" value={stats.players} />
            <Metric label="Coverage gaps" value={`${playerPresenceGaps.length}/${world.players.length}`} />
            <Metric label="Contacts" value={contactEventCount(contactWindows)} />
            <Metric label="Ball" value={ballCoverage} />
            <Metric label="Warnings" value={warningsReadout} />
            <Metric label="3D FPS" value={fps > 0 ? fps.toFixed(1) : "--"} />
          </div>
        ) : null}
      </header>

      {replayLoaded ? (
        <div className="entity-trust-strip" aria-label="Entity trust badges">
          <PlayerTrustBandPanels players={world.players} />
          <TrustBandPanel label="Ball" trustBand={world.ball.trust_band} />
          {world.paddles.map((paddle) => (
            <TrustBandPanel key={`paddle-${paddle.player_id}`} label={`Paddle ${paddle.player_id} (preview)`} trustBand={paddle.trust_band} />
          ))}
        </div>
      ) : null}

      {loadError ? <p className="load-error">{loadError}</p> : null}
      {Object.keys(capabilityErrors).length ? (
        <div className="capability-error-strip" aria-label="Missing optional capabilities">
          {Object.entries(capabilityErrors).map(([capability, reason]) => (
            <span key={capability} className="trust-badge-chip preview" title={reason}>{capability}: missing</span>
          ))}
        </div>
      ) : null}

      <UploadPanel />

      <RecentRunSwitcher currentManifestUrl={currentManifestUrl} replayViewMode={replayViewMode} />

      {replayLoaded ? (
        <section className="review-control-strip" aria-label="Replay review controls">
          <ViewLayerPanel
            viewState={viewState}
            playerIds={playerIds}
            displayFps={{
              enabled: displayFpsEnabled,
              processing: displayFpsProcessing,
              readout: displayFpsControlReadout,
              onToggle: () => setDisplayFpsEnabled((enabled) => !enabled),
            }}
            onToggleLayer={toggleLayer}
            onBallFocus={focusBall}
            onPlayerFocus={focusPlayer}
            onClearFocus={resetFocus}
            onResetView={resetView}
          />
          <button
            type="button"
            className={showShotsPanel ? "shots-toggle active" : "shots-toggle"}
            aria-expanded={showShotsPanel}
            aria-controls="shots-panel"
            onClick={toggleShotsPanel}
          >
            {showShotsPanel ? "Hide shots" : "Shots"}
          </button>
          <div className="replay-view-toggle" role="group" aria-label="Replay view">
            <button
              type="button"
              className={replayViewMode === "world" ? "view-mode-button active" : "view-mode-button"}
              aria-pressed={replayViewMode === "world"}
              onClick={() => selectReplayViewMode("world")}
            >
              3D
            </button>
            <button
              type="button"
              className={replayViewMode === "courtmap" ? "view-mode-button active" : "view-mode-button"}
              aria-pressed={replayViewMode === "courtmap"}
              onClick={() => selectReplayViewMode("courtmap")}
            >
              Court map
            </button>
          </div>
        </section>
      ) : null}

      {replayLoaded && showShotsPanel ? (
        <section id="shots-panel" className="shots-panel" aria-label="Shots panel">
          <div className="shot-workspace">
            <ShotTrailsControls
              filters={shotTrailFilters}
              options={shotFilterOptions}
              totalCount={shots?.shots.length ?? 0}
              visibleCount={filteredShots.length}
              drawableCount={drawableShotCount}
              hasArcSource={Boolean(ballArcSolved?.trusted)}
              onFilterChange={updateShotFilter}
            />
            <ShotDetailPanel
              shot={selectedShot}
              groups={shotTrailGroups}
              correctionStatus={correctionStatus}
              onWrong={writeCorrection}
            />
          </div>
        </section>
      ) : null}

      {replayLoaded ? (
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
                onPause={(event) => syncVideoTime(event.currentTarget)}
                onPlay={(event) => syncVideoTime(event.currentTarget)}
                onRateChange={(event) => syncVideoTime(event.currentTarget)}
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
              {videoBallOverlay ? <VideoBallOverlayMark overlay={videoBallOverlay} /> : null}
            </svg>
          </div>
          <div className="timeline-readout">
            <span>{currentTime.toFixed(2)}s</span>
            <span>{activeLabels.length} boxes</span>
            <span>{contactReadout}</span>
            <span>{ballReadout}</span>
            <span>{labelTrustText(playerBoxOverlay)}</span>
          </div>
          <TimelineStrip
            durationSeconds={timelineDuration}
            currentTime={currentTime}
            markers={timelineMarkers}
            chapters={timelineChapters}
            onSeek={seekTo}
            onPreviousEvent={() => jumpToEvent("previous")}
            onNextEvent={() => jumpToEvent("next")}
          />
        </div>

        <div className="world-panel">
          {replayViewMode === "courtmap" ? (
            <CourtMapPanel world={renderWorld} arcRender={ballArcRender} currentTime={currentTime} />
          ) : (
          <>
          <div className="camera-preset-bar" role="group" aria-label="3D camera">
            {PRODUCT_CAMERA_PRESETS.map((preset) => (
              <button
                key={preset}
                type="button"
                className={preset === cameraPreset ? "camera-preset active" : "camera-preset"}
                data-camera-preset={preset}
                onClick={() => selectCameraPreset(preset)}
                title={CAMERA_PRESET_LABELS[preset]}
              >
                {CAMERA_PRESET_LABELS[preset]}
              </button>
            ))}
            <label className="follow-player-select">
              <span>Follow</span>
              <select
                aria-label="Follow player"
                value={cameraFollowPlayerId ?? playerIds[0] ?? ""}
                onChange={(event) => {
                  const playerId = Number(event.currentTarget.value);
                  setCameraFollowPlayerId(Number.isInteger(playerId) ? playerId : null);
                  selectCameraPreset("follow_player");
                }}
              >
                {playerIds.map((playerId) => <option key={playerId} value={playerId}>P{playerId}</option>)}
              </select>
            </label>
          </div>
          <Canvas
            dpr={[1.5, 2]}
            gl={{ powerPreference: "high-performance", antialias: true }}
            camera={{ position: [0, -18, 8.5], fov: 50, near: 0.05, far: 100 }}
            onContextMenu={(event) => event.preventDefault()}
          >
            <color attach="background" args={["#f4f1e8"]} />
            <ambientLight intensity={2.25} />
            <directionalLight position={[0, -4, 8]} intensity={1.75} />
            <FpsProbe onSample={setFps} />
            <OrbitRig
              world={renderWorld}
              preset={cameraPreset}
              activePaddles={activePaddles}
              selectedPlayerId={viewState.selectedPlayerId}
              followPlayerId={cameraFollowPlayerId ?? viewState.selectedPlayerId ?? playerIds[0] ?? null}
              currentTime={currentTime}
              prefersReducedMotion={prefersReducedMotion}
              resetToken={cameraResetToken}
              dragCommand={cameraDragCommand}
            />
            <CourtSurface world={renderWorld} />
            <CourtLines world={renderWorld} />
            <NetAssembly world={renderWorld} />
            {shotTrailsMode ? (
              <ShotTrailsLayer groups={shotTrailGroups} selectedShotId={selectedShotId} onSelectShot={selectShot} />
            ) : (
              <>
                {shouldRenderReplayScenePointClouds(viewState, replayScene, currentTime) ? (
                  <ReplayGlbLayer replayScene={replayScene} replaySceneUrl={manifest?.replay_scene_url ?? null} currentTime={currentTime} />
                ) : null}
                {sceneLayers.playerTrails.visible ? <PlayerMotionTrails world={renderWorld} currentTime={currentTime} viewState={viewState} /> : null}
                <Players
                  world={renderWorld}
                  currentTime={currentTime}
                  activeContactPlayerIds={viewerContactPlayerIds}
                  showSkeletons={sceneLayers.playerSkeletons.visible}
                  showImplausibleSkeletons={viewState.layers.implausibleSkeletons}
                  showHandJointPoints={sceneLayers.handJointPoints.visible}
                  showFloorMarkers={sceneLayers.floorContactMarkers.visible}
                  showVertexClouds={sceneLayers.debugPointClouds.visible}
                  viewState={viewState}
                  onSelectPlayer={focusPlayer}
                />
                {sceneLayers.paddles.visible ? (
                  <Paddles
                    paddles={activePaddles}
                    showNormals={viewState.layers.paddleNormals}
                    viewState={viewState}
                    onSelectPlayer={focusPlayer}
                  />
                ) : null}
                {viewState.layers.playerSolidMeshes ? (
                  <SolidBodyMeshes
                    meshes={activeBodyMeshes}
                    world={renderWorld}
                    currentTime={currentTime}
                    geometryCache={solidGeometryCache}
                    viewState={viewState}
                    onSelectPlayer={focusPlayer}
                  />
                ) : null}
                {viewState.layers.eventMarkers ? (
                  <ImpactMarkers
                    world={world}
                    arcSolved={ballTrailArtifact}
                    currentTime={currentTime}
                    contactWindows={contactWindows}
                    bounceCandidates={autoBounceCandidates}
                  />
                ) : null}
                {sceneLayers.ballTrail.visible || sceneLayers.ballDot.visible ? (
                  <BallTrailLayer
                    samples={resolvedBallSamples}
                    currentTime={currentTime}
                    showTrail={sceneLayers.ballTrail.visible}
                    showBall={sceneLayers.ballDot.visible}
                    focusStyle={entityFocusStyle(viewState, { kind: "ball" })}
                  />
                ) : null}
              </>
            )}
          </Canvas>
          <div style={{ position: "absolute", left: 12, bottom: 12, zIndex: 2, pointerEvents: "none" }}>
            <BallHonestyHud
              samples={resolvedBallSamples}
              arcSolved={ballArcRenderSamples.length ? null : ballTrailArtifact}
              currentTime={currentTime}
            />
          </div>
          {coverageGapActive ? <div className="world-warning">No player artifact coverage after {coverage.lastTime?.toFixed(2)}s</div> : null}
          {playerPresenceGaps.length ? (
            <div className="player-gap-strip" aria-label="Player coverage gaps">
              {playerPresenceGaps.map((player) => <span key={player.id}>Player {player.id}: no detection</span>)}
            </div>
          ) : null}
          {eventMarkerEmpty ? <div className="event-empty-reason">Events: no marker evidence at this time</div> : null}
          {entityEmptyStates.length ? (
            <div className="layer-empty-strip" aria-label="Unavailable entity layers">
              {entityEmptyStates.map((message) => <span key={message}>{message}</span>)}
            </div>
          ) : null}
          {sceneLayers.debugPointClouds.visible ? <MeshDebugReadout snapshot={meshDebugSnapshot} /> : null}
          <div className="world-mini-readout">
            <span>{CAMERA_PRESET_LABELS[cameraPreset]}</span>
            <span>{solidMeshTileValue}</span>
            <span>{meshInterpolationReadout}</span>
            {displayFpsEnabled ? <span>{displayFpsControlReadout}</span> : null}
            {sceneLayers.debugPointClouds.visible ? <span data-entity-stale-ages>{entityStaleReadout}</span> : null}
          </div>
          <CameraDragPads onDrag={applyCameraDrag} onReset={() => selectCameraPreset(cameraPreset)} />
          </>
          )}
        </div>
      </section>
      ) : null}

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

export function entityStaleAgeReadout(
  world: VirtualWorld,
  meshes: ActiveBodyMeshFrame[],
  paddles: ActivePaddleFrame[],
  ballSamples: Array<{ t: number }>,
  currentTime: number,
): string {
  const players = world.players.map((player) => {
    const age = playerPresenceForTime(player, currentTime).staleAgeSeconds;
    return `P${player.id}:${age === null ? "missing" : `${age.toFixed(3)}s`}`;
  });
  const meshAges = meshes.map((mesh) => `M${mesh.playerId}:${Math.abs(currentTime - mesh.frame.t).toFixed(3)}s`);
  const paddleAges = paddles.map((paddle) => `R${paddle.playerId}:${paddle.staleAgeSeconds.toFixed(3)}s`);
  const ballAge = ballSamples.length
    ? Math.min(...ballSamples.map((sample) => Math.abs(currentTime - sample.t))).toFixed(3) + "s"
    : "missing";
  return [...players, ...meshAges, ...paddleAges, `B:${ballAge}`].join(" ");
}

export function RecentRunSwitcher({
  currentManifestUrl,
  replayViewMode,
}: {
  currentManifestUrl: string | null;
  replayViewMode: ReplayViewMode;
}) {
  const pathname = typeof window === "undefined" ? "/" : window.location.pathname;
  return (
    <nav className="recent-run-switcher" aria-label="Latest video runs">
      <div className="recent-run-heading">
        <span>Latest video runs</span>
        <small>Most recent local manifests</small>
      </div>
      <div className="recent-run-list" role="list">
        {RECENT_REPLAY_RUNS.map((run) => {
          const active = sameManifestUrl(currentManifestUrl, run.manifestUrl);
          return (
            <a
              key={run.id}
              className={active ? "recent-run-chip active" : "recent-run-chip"}
              href={recentRunHref(run, replayViewMode, pathname)}
              aria-current={active ? "page" : undefined}
              title={run.clip}
              role="listitem"
            >
              <span className="recent-run-name">{run.label}</span>
              <span className="recent-run-meta">{run.runLabel}</span>
              <span className="recent-run-time">{run.updatedLabel}</span>
            </a>
          );
        })}
      </div>
    </nav>
  );
}

export function MeshDebugReadout({ snapshot }: { snapshot: BodyMeshDebugSnapshot }) {
  const players = snapshot.players
    .map((player) => {
      const frame = player.mesh_frame_present ? `frame ${player.mesh_frame_idx}` : player.mesh_player_present ? "no frame" : "no mesh";
      return `P${player.world_player_id}->M${player.normalized_mesh_player_id}:${frame}`;
    })
    .join(" | ");
  const readout = [
    `window=${snapshot.active_window_id ?? "none"}`,
    `load=${snapshot.load_state}`,
    `rendered=${snapshot.rendered_player_count}`,
    `floorGuards=${snapshot.alignment_floor_guard_count}`,
    players,
  ]
    .filter(Boolean)
    .join(" | ");
  return (
    <pre className="mesh-debug-readout" data-mesh-debug={JSON.stringify(snapshot)}>
      {readout}
    </pre>
  );
}

export function ViewLayerPanel({
  viewState,
  playerIds,
  displayFps,
  onToggleLayer,
  onBallFocus,
  onPlayerFocus,
  onClearFocus,
  onResetView,
}: {
  viewState: ViewState;
  playerIds: number[];
  displayFps?: {
    enabled: boolean;
    processing: boolean;
    readout: string;
    onToggle: () => void;
  };
  onToggleLayer: (layer: ViewLayerKey) => void;
  onBallFocus: () => void;
  onPlayerFocus: (playerId: number) => void;
  onClearFocus: () => void;
  onResetView: () => void;
}) {
  const groups = groupLayerDefinitions(VIEW_LAYER_DEFINITIONS);
  const debugDefinitions = groups.find(([group]) => group === "Debug")?.[1] ?? [];
  return (
    <div className="layer-panel" aria-label="Layer controls">
      <div className="layer-panel-header">
        <span>Layers</span>
        <button type="button" className="layer-reset" onClick={onResetView}>
          Reset
        </button>
      </div>
      {groups.filter(([group]) => group !== "Debug").map(([group, definitions]) => (
        <div key={group} className="layer-group">
          <span className="layer-group-title">{group}</span>
          <div className="layer-buttons">
            {definitions.map((definition) => (
              <LayerToggleButton
                key={definition.key}
                definition={definition}
                pressed={viewState.layers[definition.key]}
                onToggle={() => onToggleLayer(definition.key)}
              />
            ))}
          </div>
        </div>
      ))}
      {displayFps ? (
        <div className="layer-group layer-fps-group">
          <span className="layer-group-title">Playback</span>
          <DisplayFpsControl {...displayFps} />
        </div>
      ) : null}
      <details className="layer-group debug-layer-group">
        <summary className="layer-group-title">Debug</summary>
        <div className="layer-buttons">
          {debugDefinitions.map((definition) => (
            <LayerToggleButton
              key={definition.key}
              definition={definition}
              pressed={viewState.layers[definition.key]}
              onToggle={() => onToggleLayer(definition.key)}
            />
          ))}
        </div>
      </details>
      <div className="focus-group">
        <span className="layer-group-title">Isolate</span>
        <div className="focus-buttons">
          <button
            type="button"
            className={viewState.focus?.kind === "ball" ? "focus-button active" : "focus-button"}
            aria-pressed={viewState.focus?.kind === "ball"}
            onClick={onBallFocus}
          >
            Ball focus
          </button>
          <button type="button" className="focus-button" aria-pressed={viewState.focus === null} onClick={onClearFocus}>
            Clear
          </button>
        </div>
        <div className="player-chip-row" role="group" aria-label="Player focus">
          {playerIds.map((playerId) => (
            <button
              key={playerId}
              type="button"
              className={viewState.selectedPlayerId === playerId ? "player-chip active" : "player-chip"}
              aria-pressed={viewState.selectedPlayerId === playerId}
              onClick={() => onPlayerFocus(playerId)}
            >
              Player {playerId}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export function DisplayFpsControl({
  enabled,
  processing,
  readout,
  onToggle,
}: {
  enabled: boolean;
  processing: boolean;
  readout: string;
  onToggle: () => void;
}) {
  const badge = displayFpsBadgeText({ enabled, processing, readout });
  return (
    <div className="layer-fps-control" role="group" aria-label="Display FPS">
      <button
        type="button"
        className={enabled ? "layer-toggle active" : "layer-toggle"}
        aria-pressed={enabled}
        onClick={onToggle}
        title={readout}
      >
        2x FPS (interpolated)
      </button>
      <span className="display-fps-badge" aria-live="polite" title={readout}>
        {badge}
      </span>
    </div>
  );
}

function displayFpsBadgeText({ enabled, processing, readout }: { enabled: boolean; processing: boolean; readout: string }): string {
  if (processing) return "processing";
  if (!enabled) return "original";
  const leading = readout.match(/^\d+fps display/)?.[0];
  return leading ?? readout;
}

type ShotTrailFilterOptions = {
  players: number[];
  shotTypes: string[];
  outcomes: string[];
  qualities: Array<"high" | "mid" | "low">;
};

function ShotTrailsControls({
  filters,
  options,
  totalCount,
  visibleCount,
  drawableCount,
  hasArcSource,
  onFilterChange,
}: {
  filters: ShotTrailFilters;
  options: ShotTrailFilterOptions;
  totalCount: number;
  visibleCount: number;
  drawableCount: number;
  hasArcSource: boolean;
  onFilterChange: (key: keyof ShotTrailFilters, value: string) => void;
}) {
  return (
    <div className="shot-trails-controls" aria-label="Shot trail filters">
      <div className="shot-trails-counts">
        <strong>{visibleCount}/{totalCount} shots</strong>
        <span>{drawableCount} arc trails</span>
        <span>{hasArcSource ? "arc source loaded" : "arc source missing"}</span>
      </div>
      <label>
        <span>Player</span>
        <select value={filters.playerId ?? "all"} onChange={(event) => onFilterChange("playerId", event.currentTarget.value)}>
          <option value="all">All</option>
          {options.players.map((playerId) => (
            <option key={playerId} value={playerId}>P{playerId}</option>
          ))}
        </select>
      </label>
      <label>
        <span>Type</span>
        <select value={filters.shotType ?? "all"} onChange={(event) => onFilterChange("shotType", event.currentTarget.value)}>
          <option value="all">All</option>
          {options.shotTypes.map((shotType) => (
            <option key={shotType} value={shotType}>{shotType}</option>
          ))}
        </select>
      </label>
      <label>
        <span>Outcome</span>
        <select value={filters.outcome ?? "all"} onChange={(event) => onFilterChange("outcome", event.currentTarget.value)}>
          <option value="all">All</option>
          {options.outcomes.map((outcome) => (
            <option key={outcome} value={outcome}>{outcome}</option>
          ))}
        </select>
      </label>
      <label>
        <span>Quality</span>
        <select value={filters.quality ?? "all"} onChange={(event) => onFilterChange("quality", event.currentTarget.value)}>
          <option value="all">All</option>
          {options.qualities.map((quality) => (
            <option key={quality} value={quality}>{quality}</option>
          ))}
        </select>
      </label>
    </div>
  );
}

function ShotDetailPanel({
  shot,
  groups,
  correctionStatus,
  onWrong,
}: {
  shot: ShotRecord | null;
  groups: ShotTrailGroup[];
  correctionStatus: string | null;
  onWrong: (shot: ShotRecord) => void;
}) {
  const group = shot ? groups.find((candidate) => candidate.shot.shot_id === shot.shot_id) ?? null : null;
  if (!shot) {
    return (
      <aside className="shot-detail-panel">
        <div className="shot-detail-header">
          <span>Shot Trails</span>
          <strong>{groups.length}</strong>
        </div>
        <p className="shot-detail-muted">No shot selected</p>
        {correctionStatus ? <p className="shot-correction-status">{correctionStatus}</p> : null}
      </aside>
    );
  }
  const outcome = shot.outcome.call;
  const quality = qualityBandForShot(shot);
  return (
    <aside className="shot-detail-panel">
      <div className="shot-detail-header">
        <span>Shot</span>
        <strong>{shot.frame === null ? shot.t.toFixed(2) : `F${shot.frame}`}</strong>
      </div>
      <div className="shot-badge-row">
        <span className="shot-badge">P{shot.player_id ?? "?"}</span>
        <span className="shot-badge">{shotTypeLabel(shot)}</span>
        <span className={`shot-badge outcome ${outcome}`}>{outcome}</span>
        {shot.outcome.let_candidate ? <span className="shot-badge let">let_candidate</span> : null}
      </div>
      <dl className="shot-detail-grid">
        <div>
          <dt>Speed</dt>
          <dd>{shot.speed_mph === null ? "n/a" : `${shot.speed_mph.toFixed(1)} mph`}</dd>
        </div>
        <div>
          <dt>Confidence</dt>
          <dd>{shot.confidence.toFixed(2)}</dd>
        </div>
        <div>
          <dt>Quality</dt>
          <dd>{quality}</dd>
        </div>
        <div>
          <dt>Arc Samples</dt>
          <dd>{group?.points.length ?? 0}</dd>
        </div>
      </dl>
      <button type="button" className="shot-wrong-button" onClick={() => onWrong(shot)}>
        wrong?
      </button>
      {correctionStatus ? <p className="shot-correction-status">{correctionStatus}</p> : null}
    </aside>
  );
}

function groupLayerDefinitions(definitions: ViewLayerDefinition[]): Array<[ViewLayerDefinition["group"], ViewLayerDefinition[]]> {
  const groups: ViewLayerDefinition["group"][] = ["Ball", "Players", "Events", "Debug"];
  return groups.map((group) => [group, definitions.filter((definition) => definition.group === group)]);
}

function LayerToggleButton({
  definition,
  pressed,
  onToggle,
}: {
  definition: ViewLayerDefinition;
  pressed: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      className={pressed ? "layer-toggle active" : "layer-toggle"}
      data-layer-key={definition.key}
      aria-pressed={pressed}
      title={definition.description}
      onClick={onToggle}
    >
      {definition.label}
    </button>
  );
}

/**
 * Honest FPS measurement: samples the Three.js/R3F render loop directly via
 * `useFrame` (the browser's own requestAnimationFrame callback, gated on an
 * actual rendered frame) and reports a rolling ~500ms-window reading via
 * `updateFpsSample`. This is the "renders end-to-end at >=30fps" number for
 * the W3-SCRUBBER-V0 gate -- see the run summary for how it was captured.
 */
function FpsProbe({ onSample }: { onSample: (fps: number) => void }) {
  const sampleRef = useRef<FpsSample>({ framesSinceReport: 0, windowStartMs: 0, fps: 0 });
  useFrame(() => {
    const now = performance.now();
    const current = sampleRef.current;
    const seeded = current.windowStartMs === 0 ? { ...current, windowStartMs: now } : current;
    const next = updateFpsSample(seeded, now);
    sampleRef.current = next;
    if (next.framesSinceReport === 0 && next.fps !== current.fps) {
      onSample(next.fps);
    }
  });
  return null;
}

export function TimelineStrip({
  durationSeconds,
  currentTime,
  markers,
  chapters,
  onSeek,
  onPreviousEvent,
  onNextEvent,
}: {
  durationSeconds: number;
  currentTime: number;
  markers: TimelineMarker[];
  chapters: TimelineChapter[];
  onSeek: (seconds: number) => void;
  onPreviousEvent: () => void;
  onNextEvent: () => void;
}) {
  const duration = durationSeconds > 0 ? durationSeconds : 1;
  const playheadPercent = Math.min(100, Math.max(0, (currentTime / duration) * 100));
  const hasPreviousEvent = timelineEventJump(markers, currentTime, "previous") !== null;
  const hasNextEvent = timelineEventJump(markers, currentTime, "next") !== null;
  return (
    <div className="timeline-strip" aria-label="Event timeline">
      <div className="timeline-controls" role="group" aria-label="Event navigation">
        <button type="button" className="event-nav-button" onClick={onPreviousEvent} disabled={!hasPreviousEvent}>
          Prev Event
        </button>
        <button type="button" className="event-nav-button" onClick={onNextEvent} disabled={!hasNextEvent}>
          Next Event
        </button>
        <div className="timeline-provenance-legend" aria-label="Marker provenance">
          <span className="measured">Measured</span>
          <span className="model_estimated">Model estimated</span>
          <span className="physics_predicted">Physics predicted</span>
        </div>
      </div>
      <div
        className="timeline-track"
        role="slider"
        aria-valuemin={0}
        aria-valuemax={duration}
        aria-valuenow={currentTime}
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") {
            event.preventDefault();
            onPreviousEvent();
          }
          if (event.key === "ArrowRight") {
            event.preventDefault();
            onNextEvent();
          }
        }}
        onClick={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const fraction = rect.width > 0 ? (event.clientX - rect.left) / rect.width : 0;
          onSeek(Math.min(duration, Math.max(0, fraction * duration)));
        }}
      >
        {chapters.map((chapter) => {
          const left = Math.min(100, Math.max(0, (chapter.t0 / duration) * 100));
          const right = Math.min(100, Math.max(left, (chapter.t1 / duration) * 100));
          return (
            <div
              key={chapter.index}
              className={`timeline-chapter ${chapter.badge}`}
              style={{ left: `${left}%`, width: `${Math.max(1.4, right - left)}%` }}
              title={`${chapter.label} ${chapter.t0.toFixed(2)}s-${chapter.t1.toFixed(2)}s`}
            >
              <span>{chapter.label}</span>
            </div>
          );
        })}
        {markers.map((marker, index) => (
          <button
            key={`${marker.kind}-${marker.t}-${index}`}
            type="button"
            className={`timeline-marker ${marker.kind} ${marker.provenance} ${marker.badge} ${marker.humanReviewed ? "human-reviewed" : ""}`}
            style={{ left: `${Math.min(100, Math.max(0, (marker.t / duration) * 100))}%` }}
            title={`${marker.label} (${marker.provenance.replace("_", " ")}; ${marker.humanReviewed ? "human reviewed, " : ""}${marker.badge.replace("_", " ")}, confidence ${marker.confidence.toFixed(2)})`}
            aria-label={`Jump to ${marker.label}`}
            onClick={(event) => {
              event.stopPropagation();
              onSeek(marker.t);
            }}
          >
            <span className="timeline-marker-label">{marker.label}</span>
          </button>
        ))}
        <div className="timeline-playhead" style={{ left: `${playheadPercent}%` }} />
      </div>
    </div>
  );
}

function TrustBandPanel({ label, trustBand }: { label: string; trustBand?: TrustBand | null }) {
  return (
    <div className="trust-band-card compact" title={trustBand?.reason ?? "No trust-band provenance on this artifact."}>
      <div className="trust-band-header">
        <span>{label}</span>
        <span className={`trust-badge-chip ${trustBand?.badge ?? "none"}`}>{trustBandChipText(trustBand)}</span>
      </div>
    </div>
  );
}

function PlayerTrustBandPanels({ players }: { players: VirtualWorldPlayer[] }) {
  return (
    <>
      {players.map((player) => (
        <TrustBandPanel key={player.id} label={`Player ${player.id} (${player.representation})`} trustBand={player.trust_band} />
      ))}
    </>
  );
}

function CameraDragPads({
  onDrag,
  onReset,
}: {
  onDrag: (kind: CameraDragKind, dx: number, dy: number) => void;
  onReset: () => void;
}) {
  return (
    <div className="camera-drag-pads" aria-label="Camera drag controls">
      <CameraDragPad kind="pan" label="Move" onDrag={onDrag} />
      <CameraDragPad kind="orbit" label="Orbit" onDrag={onDrag} />
      <button type="button" className="camera-reset-pad" onClick={onReset}>
        Reset
      </button>
    </div>
  );
}

function CameraDragPad({
  kind,
  label,
  onDrag,
}: {
  kind: CameraDragKind;
  label: string;
  onDrag: (kind: CameraDragKind, dx: number, dy: number) => void;
}) {
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);
  const startDrag = (event: React.PointerEvent<HTMLButtonElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    lastPointRef.current = { x: event.clientX, y: event.clientY };
  };
  const moveDrag = (event: React.PointerEvent<HTMLButtonElement>) => {
    const lastPoint = lastPointRef.current;
    if (!lastPoint) return;
    const dx = event.clientX - lastPoint.x;
    const dy = event.clientY - lastPoint.y;
    lastPointRef.current = { x: event.clientX, y: event.clientY };
    onDrag(kind, dx, dy);
  };
  const endDrag = (event: React.PointerEvent<HTMLButtonElement>) => {
    lastPointRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };
  return (
    <button
      type="button"
      className={`camera-drag-pad ${kind}`}
      onPointerDown={startDrag}
      onPointerMove={moveDrag}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
    >
      <span>{label}</span>
      <small>drag</small>
    </button>
  );
}

function OrbitRig({
  world,
  preset,
  activePaddles,
  selectedPlayerId,
  followPlayerId,
  currentTime,
  prefersReducedMotion,
  resetToken,
  dragCommand,
}: {
  world: VirtualWorld;
  preset: CameraPreset;
  activePaddles: ActivePaddleFrame[];
  selectedPlayerId: number | null;
  followPlayerId: number | null;
  currentTime: number;
  prefersReducedMotion: boolean;
  resetToken: number;
  dragCommand: CameraDragCommand | null;
}) {
  const { camera, gl } = useThree();
  const controlsRef = useRef<MapControls | null>(null);
  const transitionRef = useRef<{
    startedAtMs: number;
    durationMs: number;
    fromPosition: ThreeVector3;
    fromTarget: ThreeVector3;
    toPosition: ThreeVector3;
    toTarget: ThreeVector3;
  } | null>(null);
  const pose = useMemo(
    () => cameraPresetPose(world, preset, activePaddles, followPlayerId ?? selectedPlayerId, currentTime),
    [activePaddles, currentTime, followPlayerId, selectedPlayerId, world, preset],
  );
  const poseKey = `${preset}:${pose.position.join(",")}:${pose.target.join(",")}:${resetToken}`;
  useEffect(() => {
    const controls = new MapControls(camera, gl.domElement);
    camera.up.set(0, 0, 1);
    camera.position.set(...pose.position);
    controls.target.set(...pose.target);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.48;
    controls.panSpeed = 0.78;
    controls.zoomSpeed = 0.72;
    controls.screenSpacePanning = false;
    controls.minDistance = preset === "paddle_review" ? 0.35 : 2.2;
    controls.maxDistance = Math.max(24, courtBounds(world).length * 2.1);
    controls.maxPolarAngle = Math.PI * 0.48;
    controls.update();
    controlsRef.current = controls;
    return () => {
      controlsRef.current = null;
      controls.dispose();
    };
  }, [camera, gl, world]);
  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls || preset === "free_orbit") {
      transitionRef.current = null;
      return;
    }
    const durationMs = cameraTransitionDurationMs(prefersReducedMotion);
    if (durationMs === 0) {
      camera.position.set(...pose.position);
      controls.target.set(...pose.target);
      controls.update();
      transitionRef.current = null;
      return;
    }
    transitionRef.current = {
      startedAtMs: performance.now(),
      durationMs,
      fromPosition: camera.position.clone(),
      fromTarget: controls.target.clone(),
      toPosition: new ThreeVector3(...pose.position),
      toTarget: new ThreeVector3(...pose.target),
    };
  }, [camera, poseKey, prefersReducedMotion, preset]);
  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls || !dragCommand) return;
    if (dragCommand.kind === "pan") {
      applyCameraPan(camera, controls, dragCommand.dx, dragCommand.dy);
    } else {
      applyCameraOrbit(camera, controls, dragCommand.dx, dragCommand.dy);
    }
    controls.update();
  }, [camera, dragCommand]);
  useFrame(() => {
    const controls = controlsRef.current;
    if (!controls) return;
    const transition = transitionRef.current;
    if (transition) {
      const rawProgress = Math.min(1, Math.max(0, (performance.now() - transition.startedAtMs) / transition.durationMs));
      const progress = 1 - Math.pow(1 - rawProgress, 3);
      camera.position.lerpVectors(transition.fromPosition, transition.toPosition, progress);
      controls.target.lerpVectors(transition.fromTarget, transition.toTarget, progress);
      if (rawProgress >= 1) transitionRef.current = null;
    }
    controls.update();
  });
  return null;
}

function applyCameraPan(camera: Camera, controls: MapControls, dx: number, dy: number) {
  const distance = camera.position.distanceTo(controls.target);
  const scale = Math.max(0.006, distance * 0.0014);
  const elements = camera.matrix.elements;
  const right = new ThreeVector3(elements[0], elements[1], elements[2]);
  const up = new ThreeVector3(elements[4], elements[5], elements[6]);
  const delta = right.multiplyScalar(-dx * scale).add(up.multiplyScalar(dy * scale));
  camera.position.add(delta);
  controls.target.add(delta);
}

function applyCameraOrbit(camera: Camera, controls: MapControls, dx: number, dy: number) {
  const offset = new ThreeVector3().subVectors(camera.position, controls.target);
  const spherical = new Spherical().setFromVector3(offset);
  spherical.theta -= dx * 0.006;
  spherical.phi = Math.min(Math.PI * 0.48, Math.max(0.1, spherical.phi - dy * 0.006));
  offset.setFromSpherical(spherical);
  camera.position.copy(controls.target).add(offset);
  camera.lookAt(controls.target);
}

const COURT_RENDER_COLORS = {
  surface: "#f7f5ee",
  boundary: "#656f68",
  netTape: "#4d5752",
  netMesh: "#7f8a83",
  post: "#58625d",
};

function CourtSurface({ world }: { world: VirtualWorld }) {
  const bounds = courtBounds(world);
  return (
    <mesh position={[bounds.centerX, bounds.centerY, -0.012]}>
      <planeGeometry args={[bounds.width, bounds.length]} />
      <meshStandardMaterial color={COURT_RENDER_COLORS.surface} roughness={0.78} metalness={0.01} />
    </mesh>
  );
}

function CourtLines({ world }: { world: VirtualWorld }) {
  const courtPoints = Object.values(world.court.line_segments).flat();
  const netPoints = world.court.net.endpoints;
  return (
    <>
      <WorldLineSegments points={courtPoints} color={COURT_RENDER_COLORS.boundary} radius={0.015} zOffset={0.018} />
      <WorldLineSegments points={netPoints} color={COURT_RENDER_COLORS.netTape} radius={0.012} />
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
        <meshStandardMaterial color={COURT_RENDER_COLORS.netMesh} transparent opacity={0.16} roughness={0.7} />
      </mesh>
      <mesh position={[left[0], left[1], world.court.net.post_height_m / 2]}>
        <boxGeometry args={[0.075, 0.075, world.court.net.post_height_m]} />
        <meshStandardMaterial color={COURT_RENDER_COLORS.post} />
      </mesh>
      <mesh position={[right[0], right[1], world.court.net.post_height_m / 2]}>
        <boxGeometry args={[0.075, 0.075, world.court.net.post_height_m]} />
        <meshStandardMaterial color={COURT_RENDER_COLORS.post} />
      </mesh>
      <WorldLineSegments points={[topLeft, centerTop, centerTop, topRight]} color={COURT_RENDER_COLORS.netTape} radius={0.012} />
    </>
  );
}

function WorldLineSegments({
  points,
  color,
  radius,
  zOffset = 0,
  opacity = 1,
}: {
  points: Vec3[];
  color: string;
  radius: number;
  zOffset?: number;
  opacity?: number;
}) {
  const raisedPoints = useMemo(
    () => points.map((point) => [point[0], point[1], point[2] + zOffset] as Vec3),
    [points, zOffset],
  );
  return <BoneSegments points={raisedPoints} color={color} opacity={opacity} radius={radius} />;
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
  const gltf = useLoader(GLTFLoader, url, configureGltfLoader);
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
  showSkeletons,
  showImplausibleSkeletons,
  showHandJointPoints,
  showFloorMarkers,
  showVertexClouds,
  viewState,
  onSelectPlayer,
}: {
  world: VirtualWorld;
  currentTime: number;
  activeContactPlayerIds: Set<number>;
  showSkeletons: boolean;
  showImplausibleSkeletons: boolean;
  showHandJointPoints: boolean;
  showFloorMarkers: boolean;
  showVertexClouds: boolean;
  viewState: ViewState;
  onSelectPlayer: (playerId: number) => void;
}) {
  return (
    <>
      {world.players.map((player) => {
        const presence = playerPresenceForTime(player, currentTime);
        const frame = presence.frame;
        return (
          <Player
            key={player.id}
            player={player}
            frame={frame}
            presence={presence}
            isBallContactActive={activeContactPlayerIds.has(player.id)}
            showSkeletons={showSkeletons}
            showImplausibleSkeletons={showImplausibleSkeletons}
            showHandJointPoints={showHandJointPoints}
            showFloorMarkers={showFloorMarkers}
            showVertexClouds={showVertexClouds}
            jointNames={world.joint_names}
            focusStyle={entityFocusStyle(viewState, { kind: "player", playerId: player.id })}
            onSelectPlayer={onSelectPlayer}
          />
        );
      })}
    </>
  );
}

function Player({
  player,
  frame,
  presence,
  isBallContactActive,
  showSkeletons,
  showImplausibleSkeletons,
  showHandJointPoints,
  showFloorMarkers,
  showVertexClouds,
  jointNames,
  focusStyle,
  onSelectPlayer,
}: {
  player: VirtualWorldPlayer;
  frame?: VirtualWorldFrame;
  presence: ReturnType<typeof playerPresenceForTime>;
  isBallContactActive: boolean;
  showSkeletons: boolean;
  showImplausibleSkeletons: boolean;
  showHandJointPoints: boolean;
  showFloorMarkers: boolean;
  showVertexClouds: boolean;
  jointNames?: string[];
  focusStyle: EntityFocusStyle;
  onSelectPlayer: (playerId: number) => void;
}) {
  const floor = floorWorldForFrame(frame);
  const meshPoints = vertexDebugPointsForFrame(frame, showVertexClouds, isBallContactActive ? 1800 : 850);
  const bodySkeleton = bodyJointSkeletonForFrame(frame, jointNames, { includeImplausible: showImplausibleSkeletons });
  const handPoints = handJointPointsForFrame(frame, jointNames);
  const implausibleSuppressed = frame?.skeleton_implausible === true && !showImplausibleSkeletons;
  const proxySkeleton = bodySkeleton || implausibleSuppressed ? null : skeletonForFrame(frame);
  const placeholderAnchor = floor ?? presence.lastKnownFloor;
  const showMissingPlaceholder = presence.missingEvidence || implausibleSuppressed;
  const badge = effectiveTrustBadge(player.trust_band);
  const baseColor = trustBadgeColor(badge);
  const opacityScale = focusStyle.dimmed ? 0.22 : 1;
  const highlightScale = focusStyle.highlighted ? 1.25 : 1;
  const dotOpacity = (badge === "low_confidence" ? 0.55 : 1) * opacityScale;
  const skeletonColor = focusStyle.highlighted ? "#7cff87" : isBallContactActive ? "#ffb45d" : baseColor;
  return (
    <group
      onClick={(event) => {
        event.stopPropagation();
        onSelectPlayer(player.id);
      }}
    >
      {showMissingPlaceholder && placeholderAnchor ? (
        <NoDetectionPlaceholder
          position={placeholderAnchor}
          label={implausibleSuppressed ? `Player ${player.id}: implausible pose hidden` : `Player ${player.id}: no detection`}
          color={baseColor}
          opacity={focusStyle.dimmed ? 0.1 : 0.24}
        />
      ) : null}
      {showFloorMarkers && floor ? (
        <mesh position={floor} scale={[highlightScale, highlightScale, 1]}>
          <cylinderGeometry args={[0.16, 0.16, 0.025, 28]} />
          <meshStandardMaterial color={isBallContactActive ? "#e8ff34" : baseColor} transparent opacity={dotOpacity} />
        </mesh>
      ) : null}
      {showSkeletons && proxySkeleton ? (
        <SkeletonGraph
          skeleton={proxySkeleton}
          active={isBallContactActive || focusStyle.highlighted}
          baseColor={skeletonColor}
          opacity={opacityScale}
          scale={highlightScale}
        />
      ) : null}
      {showSkeletons && bodySkeleton ? (
        <SkeletonGraph
          skeleton={bodySkeleton}
          active={isBallContactActive || focusStyle.highlighted}
          baseColor={skeletonColor}
          opacity={Math.max(0.35, opacityScale)}
          scale={highlightScale}
          boneRadius={(isBallContactActive || focusStyle.highlighted ? 0.024 : 0.018) * highlightScale}
          jointSize={(isBallContactActive || focusStyle.highlighted ? 0.075 : 0.058) * highlightScale}
        />
      ) : null}
      {meshPoints.length ? (
        <PointCloud
          points={meshPoints}
          color={isBallContactActive || focusStyle.highlighted ? "#b4f2bf" : baseColor}
          size={(isBallContactActive || focusStyle.highlighted ? 0.024 : 0.018) * highlightScale}
          opacity={Math.max(0.18, opacityScale)}
        />
      ) : null}
      {showHandJointPoints && handPoints.length ? (
        <PointCloud
          points={handPoints}
          color={focusStyle.highlighted ? "#dfff3d" : "#e7edf5"}
          size={(focusStyle.highlighted ? 0.044 : 0.032) * highlightScale}
          opacity={Math.max(0.22, opacityScale * 0.68)}
        />
      ) : null}
    </group>
  );
}

function NoDetectionPlaceholder({
  position,
  label,
  color,
  opacity,
}: {
  position: Vec3;
  label: string;
  color: string;
  opacity: number;
}) {
  return (
    <group position={[position[0], position[1], 0.018]} userData={{ badge: "missing", label }}>
      {Array.from({ length: 12 }, (_, index) => (
        <mesh key={index} rotation={[0, 0, (index * Math.PI * 2) / 12]}>
          <torusGeometry args={[0.22, 0.009, 5, 8, Math.PI / 12]} />
          <meshBasicMaterial color={color} transparent opacity={opacity} depthWrite={false} />
        </mesh>
      ))}
      <mesh position={[0, 0, 0.025]}>
        <circleGeometry args={[0.075, 20]} />
        <meshBasicMaterial color={color} transparent opacity={opacity * 0.28} depthWrite={false} />
      </mesh>
    </group>
  );
}

function VideoBallOverlayMark({ overlay }: { overlay: VideoBallOverlay }) {
  const [cx, cy] = overlay.point;
  const classes = ["video-ball", overlay.confidenceClass, overlay.interpolated ? "interpolated" : ""].filter(Boolean).join(" ");
  return (
    <g className={classes} opacity={overlay.opacity}>
      <circle className="video-ball-halo" cx={cx} cy={cy} r={overlay.radius * 1.9} />
      <circle className="video-ball-dot" cx={cx} cy={cy} r={overlay.radius} />
      <text className="video-ball-label" x={cx + overlay.radius + 8} y={Math.max(14, cy - overlay.radius - 5)}>
        ball
      </text>
    </g>
  );
}

function SolidBodyMeshes({
  meshes,
  world,
  currentTime,
  geometryCache,
  viewState,
  onSelectPlayer,
}: {
  meshes: ActiveBodyMeshFrame[];
  world: VirtualWorld;
  currentTime: number;
  geometryCache: SolidBodyMeshGeometryCache;
  viewState: ViewState;
  onSelectPlayer: (playerId: number) => void;
}) {
  const activePlayerIds = new Set(meshes.map((mesh) => mesh.playerId));
  const gaps = world.players
    .filter((player) => !activePlayerIds.has(player.id))
    .map((player) => ({ player, presence: playerPresenceForTime(player, currentTime) }))
    .filter(({ presence }) => presence.lastKnownFloor !== null);
  return (
    <>
      {meshes.map(({ playerId, meshPlayerId, frame, presenceOpacity, renderTranslation }) => (
        <SolidBodyMesh
          key={solidBodyMeshRenderKey(playerId, frame)}
          playerId={playerId}
          meshPlayerId={meshPlayerId}
          frame={frame}
          presenceOpacity={presenceOpacity}
          renderTranslation={renderTranslation}
          geometryCache={geometryCache}
          focusStyle={entityFocusStyle(viewState, { kind: "player", playerId })}
          onSelectPlayer={onSelectPlayer}
        />
      ))}
      {gaps.map(({ player, presence }) => (
        <NoDetectionPlaceholder
          key={`mesh-gap-${player.id}`}
          position={presence.lastKnownFloor!}
          label={`Player ${player.id}: mesh no coverage`}
          color="#ffb454"
          opacity={0.16}
        />
      ))}
    </>
  );
}

function Paddles({
  paddles,
  showNormals,
  viewState,
  onSelectPlayer,
}: {
  paddles: ActivePaddleFrame[];
  showNormals: boolean;
  viewState: ViewState;
  onSelectPlayer: (playerId: number) => void;
}) {
  return (
    <>
      {paddles.map((paddle) => (
        <PaddleProxy
          key={`${paddle.playerId}-${paddle.frame.t}`}
          paddle={paddle}
          showNormal={showNormals}
          focusStyle={entityFocusStyle(viewState, { kind: "player", playerId: paddle.playerId })}
          onSelectPlayer={onSelectPlayer}
        />
      ))}
    </>
  );
}

function PaddleProxy({
  paddle,
  showNormal,
  focusStyle,
  onSelectPlayer,
}: {
  paddle: ActivePaddleFrame;
  showNormal: boolean;
  focusStyle: EntityFocusStyle;
  onSelectPlayer: (playerId: number) => void;
}) {
  const renderGeometry = useMemo(
    () => paddleRenderGeometryForFrame(paddle.frame, paddle.paddle.paddle_dims_in),
    [paddle.frame, paddle.paddle.paddle_dims_in],
  );
  const geometry = useMemo(
    () => geometryFromIndexedMesh(renderGeometry.vertices, renderGeometry.faces),
    [renderGeometry.faces, renderGeometry.vertices],
  );
  const opacity = (focusStyle.dimmed ? 0.22 : focusStyle.highlighted ? 0.92 : renderGeometry.material.fillOpacity) * paddle.opacity;
  const edgeOpacity = (focusStyle.dimmed ? 0.28 : focusStyle.highlighted ? 1 : renderGeometry.material.edgeOpacity) * paddle.opacity;
  const fillColor = focusStyle.highlighted ? "#dfff3d" : renderGeometry.material.fillColor;
  const emissiveColor = focusStyle.highlighted ? "#425800" : renderGeometry.material.emissiveColor;
  return (
    <group
      onClick={(event) => {
        event.stopPropagation();
        onSelectPlayer(paddle.playerId);
      }}
    >
      <mesh geometry={geometry} renderOrder={22}>
        <meshStandardMaterial
          color={fillColor}
          emissive={emissiveColor}
          roughness={0.44}
          metalness={0.02}
          transparent
          opacity={opacity}
          side={DoubleSide}
          depthWrite
        />
      </mesh>
      <BoneSegments
        points={renderGeometry.edgeSegments}
        color={renderGeometry.material.edgeColor}
        opacity={edgeOpacity}
        radius={renderGeometry.material.edgeRadiusM}
      />
      {showNormal ? (
        <BoneSegments
          points={renderGeometry.normalSegment}
          color={renderGeometry.material.normalColor}
          opacity={edgeOpacity}
          radius={renderGeometry.material.normalRadiusM}
          renderOrder={26}
          depthTest={!renderGeometry.material.normalOverlay}
        />
      ) : null}
      {showNormal && renderGeometry.normalTip ? (
        <mesh position={renderGeometry.normalTip} renderOrder={27}>
          <sphereGeometry args={[renderGeometry.material.normalTipRadiusM, 18, 18]} />
          <meshStandardMaterial
            color={renderGeometry.material.normalColor}
            emissive={renderGeometry.material.normalColor}
            roughness={0.35}
            metalness={0.01}
            transparent
            opacity={edgeOpacity}
            depthWrite={false}
            depthTest={!renderGeometry.material.normalOverlay}
          />
        </mesh>
      ) : null}
    </group>
  );
}

function SolidBodyMesh({
  playerId,
  meshPlayerId,
  frame,
  presenceOpacity,
  renderTranslation,
  geometryCache,
  focusStyle,
  onSelectPlayer,
}: {
  playerId: number;
  meshPlayerId: number;
  frame: ActiveBodyMeshFrame["frame"];
  presenceOpacity: number;
  renderTranslation: Vec3;
  geometryCache: SolidBodyMeshGeometryCache;
  focusStyle: EntityFocusStyle;
  onSelectPlayer: (playerId: number) => void;
}) {
  const geometry = useMemo(
    () => geometryForSolidBodyMeshFrame(geometryCache, meshPlayerId, frame),
    [frame, geometryCache, meshPlayerId],
  );
  const material = bodyMeshMaterialForTrustBadge(frame.trust_badge);
  const baseOpacity = bodyMeshOpacityFromBlendWeight(frame, presenceOpacity) * material.opacityScale;
  const opacity = focusStyle.dimmed
    ? Math.min(baseOpacity, 0.14 * material.opacityScale)
    : focusStyle.highlighted
      ? Math.min(material.label === "estimated" ? 0.56 : 0.9, baseOpacity + (material.label === "estimated" ? 0.08 : 0.16))
      : baseOpacity;
  const fillColor = focusStyle.highlighted && material.label === "solid" ? "#dfff3d" : material.fillColor;
  const emissiveColor = focusStyle.highlighted && material.label === "solid" ? "#425800" : material.emissiveColor;
  return (
    <mesh
      geometry={geometry}
      position={renderTranslation}
      renderOrder={20}
      onClick={(event) => {
        event.stopPropagation();
        onSelectPlayer(playerId);
      }}
    >
      <meshStandardMaterial
        color={fillColor}
        emissive={emissiveColor}
        roughness={0.58}
        metalness={0.02}
        transparent
        opacity={opacity}
        side={DoubleSide}
        depthWrite={false}
      />
    </mesh>
  );
}

function solidBodyMeshRenderKey(playerId: number, frame: ActiveBodyMeshFrame["frame"]): string {
  if (frame.mesh_interpolated && frame.interpolation) {
    return `${playerId}-interp-${frame.interpolation.from_frame_idx}-${frame.interpolation.to_frame_idx}`;
  }
  return `${playerId}-${frame.frame_idx}`;
}

function BallTrail({
  world,
  currentTime,
  ballArcEventsSelected,
  focusStyle,
}: {
  world: VirtualWorld;
  currentTime: number;
  ballArcEventsSelected: BallArcEventsSelected | null;
  focusStyle: EntityFocusStyle;
}) {
  const segments = useMemo(
    () => ballTrailSegmentsForTime(world, currentTime, 0.75, ballArcEventsSelected),
    [ballArcEventsSelected, world, currentTime],
  );
  if (!segments.length) return null;
  const badge = effectiveTrustBadge(world.ball.trust_band);
  const color = badge === "low_confidence" ? "#adb3bb" : "#e8ff34";
  const focusOpacityScale = focusStyle.dimmed ? 0.32 : focusStyle.highlighted ? 1.75 : 1;
  const focusWidthScale = focusStyle.highlighted ? 2.1 : focusStyle.dimmed ? 0.75 : 1;
  const maxOpacity = Math.min(0.86, (badge === "low_confidence" ? 0.28 : 0.48) * focusOpacityScale);
  return (
    <>
      {segments.map((segment, index) =>
        segment.courtStyle === "outside_court" ? (
          <DashedFadingLineSegment
            key={`ball-trail-out-${index}`}
            from={segment.from}
            to={segment.to}
            color="#ffb454"
            opacity={maxOpacity * segment.opacityScale}
            lineWidth={Math.max(0.55, segment.thicknessScale * focusWidthScale)}
          />
        ) : (
          <FadingLineSegment
            key={`ball-trail-in-${index}`}
            points={[segment.from, segment.to]}
            color={color}
            opacity={maxOpacity * segment.opacityScale}
            lineWidth={Math.max(0.65, segment.thicknessScale * focusWidthScale)}
          />
        ),
      )}
    </>
  );
}

function ShotTrailsLayer({
  groups,
  selectedShotId,
  onSelectShot,
}: {
  groups: ShotTrailGroup[];
  selectedShotId: string | null;
  onSelectShot: (shot: ShotRecord) => void;
}) {
  return (
    <>
      {groups.map((group) => (
        <ShotTrailGroupMesh
          key={group.shot.shot_id}
          group={group}
          selected={group.shot.shot_id === selectedShotId}
          dimmed={Boolean(selectedShotId && group.shot.shot_id !== selectedShotId)}
          onSelectShot={onSelectShot}
        />
      ))}
    </>
  );
}

function ShotTrailGroupMesh({
  group,
  selected,
  dimmed,
  onSelectShot,
}: {
  group: ShotTrailGroup;
  selected: boolean;
  dimmed: boolean;
  onSelectShot: (shot: ShotRecord) => void;
}) {
  const color = shotOutcomeColor(group.shot);
  const opacity = dimmed ? 0.16 : selected ? 0.92 : group.quality === "low" ? 0.38 : 0.64;
  const lineWidth = selected ? 2.2 : group.quality === "low" ? 0.8 : 1.25;
  return (
    <group
      onClick={(event) => {
        event.stopPropagation();
        onSelectShot(group.shot);
      }}
    >
      {group.segments.map((segment, index) => (
        <FadingLineSegment
          key={`${group.shot.shot_id}-${index}`}
          points={[segment.from, segment.to]}
          color={color}
          opacity={opacity}
          lineWidth={lineWidth}
        />
      ))}
      <ShotLandingMarker shot={group.shot} color={color} selected={selected} dimmed={dimmed} />
    </group>
  );
}

function ShotLandingMarker({
  shot,
  color,
  selected,
  dimmed,
}: {
  shot: ShotRecord;
  color: string;
  selected: boolean;
  dimmed: boolean;
}) {
  const landing = shot.landing;
  const ellipse = landing?.uncertainty_ellipse ?? null;
  const dotPosition = landing?.world_xyz ?? (ellipse ? ([ellipse.center_xy[0], ellipse.center_xy[1], 0.045] as Vec3) : null);
  if (!dotPosition && !ellipse) return null;
  const opacity = dimmed ? 0.18 : selected ? 0.9 : 0.62;
  return (
    <>
      {ellipse ? (
        <mesh
          position={[ellipse.center_xy[0], ellipse.center_xy[1], 0.052]}
          rotation={[0, 0, (ellipse.angle_deg * Math.PI) / 180]}
          scale={[Math.max(0.08, ellipse.semi_major_m), Math.max(0.08, ellipse.semi_minor_m), 1]}
          renderOrder={24}
        >
          <ringGeometry args={[0.86, 1, 72]} />
          <meshBasicMaterial color={color} transparent opacity={opacity * 0.38} depthWrite={false} side={DoubleSide} />
        </mesh>
      ) : null}
      {dotPosition ? (
        <mesh position={[dotPosition[0], dotPosition[1], Math.max(0.07, dotPosition[2])]} renderOrder={25}>
          <sphereGeometry args={[selected ? 0.12 : 0.078, 18, 18]} />
          <meshStandardMaterial color={color} emissive={selected ? color : "#000000"} transparent opacity={opacity} depthWrite={false} />
        </mesh>
      ) : null}
    </>
  );
}

function PlayerMotionTrails({ world, currentTime, viewState }: { world: VirtualWorld; currentTime: number; viewState: ViewState }) {
  return (
    <>
      {world.players.map((player) => {
        const points = playerTrailPointsForTime(player, currentTime, 1.2);
        if (points.length < 2) return null;
        const badge = effectiveTrustBadge(player.trust_band);
        const focusStyle = entityFocusStyle(viewState, { kind: "player", playerId: player.id });
        return (
          <FadingLineStrip
            key={player.id}
            points={points}
            color={focusStyle.highlighted ? "#dfff3d" : trustBadgeColor(badge)}
            maxOpacity={focusStyle.dimmed ? 0.08 : focusStyle.highlighted ? 0.62 : badge === "low_confidence" ? 0.22 : 0.36}
          />
        );
      })}
    </>
  );
}

function WorldEventMarkers({ markers, viewState }: { markers: WorldEventMarker[]; viewState: ViewState }) {
  return (
    <>
      {markers.map((marker, index) => {
        const target =
          marker.playerId === null
            ? ({ kind: "ball" } as const)
            : ({ kind: "player", playerId: marker.playerId } as const);
        const focusStyle = entityFocusStyle(viewState, target);
        const color = marker.kind === "bounce" ? "#dfff3d" : marker.kind === "net_cross" ? "#63d9ff" : "#ffb454";
        const opacity = focusStyle.dimmed ? 0.18 : focusStyle.highlighted ? 0.96 : 0.74;
        const scale = focusStyle.highlighted ? 1.35 : 1;
        return (
          <group key={`${marker.kind}-${marker.position.join(",")}-${index}`} position={marker.position} scale={[scale, scale, scale]}>
            <mesh rotation={[Math.PI / 2, 0, 0]}>
              <torusGeometry args={[0.18, 0.018, 8, 28]} />
              <meshStandardMaterial color={color} emissive={focusStyle.highlighted ? color : "#000000"} transparent opacity={opacity} depthWrite={false} />
            </mesh>
            <mesh position={[0, 0, 0.16]}>
              <coneGeometry args={[0.07, 0.2, 16]} />
              <meshStandardMaterial color={color} transparent opacity={opacity} />
            </mesh>
          </group>
        );
      })}
    </>
  );
}

function SkeletonGraph({
  skeleton,
  active,
  baseColor,
  opacity,
  scale,
  boneRadius = 0.012,
  jointSize,
}: {
  skeleton: BodyJointSkeleton;
  active: boolean;
  baseColor: string;
  opacity: number;
  scale: number;
  boneRadius?: number;
  jointSize?: number;
}) {
  const color = active ? "#7cff87" : baseColor;
  return (
    <>
      <BoneSegments points={skeleton.bones} color={color} opacity={opacity} radius={boneRadius} />
      <PointCloud points={skeleton.joints} color={color} size={jointSize ?? (active ? 0.052 : 0.042) * scale} opacity={opacity} />
    </>
  );
}

function BoneSegments({
  points,
  color,
  opacity,
  radius,
  renderOrder = 19,
  depthTest = true,
}: {
  points: Vec3[];
  color: string;
  opacity: number;
  radius: number;
  renderOrder?: number;
  depthTest?: boolean;
}) {
  const segments = useMemo(() => {
    const pairs: Array<[Vec3, Vec3]> = [];
    for (let index = 1; index < points.length; index += 2) {
      pairs.push([points[index - 1], points[index]]);
    }
    return pairs;
  }, [points]);
  return (
    <>
      {segments.map(([from, to], index) => (
        <BoneSegment
          key={`${index}-${from.join(",")}-${to.join(",")}`}
          from={from}
          to={to}
          color={color}
          opacity={opacity}
          radius={radius}
          renderOrder={renderOrder}
          depthTest={depthTest}
        />
      ))}
    </>
  );
}

function BoneSegment({
  from,
  to,
  color,
  opacity,
  radius,
  renderOrder = 19,
  depthTest = true,
}: {
  from: Vec3;
  to: Vec3;
  color: string;
  opacity: number;
  radius: number;
  renderOrder?: number;
  depthTest?: boolean;
}) {
  const transform = useMemo(() => {
    const start = new ThreeVector3(...from);
    const end = new ThreeVector3(...to);
    const direction = end.clone().sub(start);
    const length = direction.length();
    const midpoint = start.clone().add(end).multiplyScalar(0.5);
    const quaternion =
      length > 0
        ? new Quaternion().setFromUnitVectors(new ThreeVector3(0, 1, 0), direction.clone().normalize())
        : new Quaternion();
    return {
      length,
      position: [midpoint.x, midpoint.y, midpoint.z] as Vec3,
      quaternion,
    };
  }, [from, to]);
  if (transform.length <= 1e-6) return null;
  return (
    <mesh position={transform.position} quaternion={transform.quaternion} renderOrder={renderOrder}>
      <cylinderGeometry args={[radius, radius, transform.length, 8]} />
      <meshStandardMaterial color={color} transparent opacity={opacity} roughness={0.5} metalness={0.02} depthWrite={false} depthTest={depthTest} />
    </mesh>
  );
}

function LineSegments({ points, color, opacity = 1 }: { points: Vec3[]; color: string; opacity?: number }) {
  const geometry = useMemo(() => geometryFromPoints(points), [points]);
  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color={color} transparent={opacity < 1} opacity={opacity} />
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

function ballRenderText(mode: ReturnType<typeof ballRenderInfoForTime>["mode"], videoOverlay: VideoBallOverlay | null): string {
  if (mode === "calibrated_3d") return "ball: calibrated 3D";
  if (mode === "court_plane_projection") return "ball: court-plane approx";
  if (mode === "off_court_projection") return "ball: off-court hidden";
  if (mode === "2d_only_no_3d_solve") return "ball: 2D-only, no 3D solve";
  if (videoOverlay) {
    const suffix = videoOverlay.interpolated ? " interpolated" : "";
    return `ball: video ${videoOverlay.confidenceClass}${suffix}`;
  }
  return "ball: missing";
}

function shotTrailFilterOptions(shots: ShotRecord[]): ShotTrailFilterOptions {
  const players = Array.from(new Set(shots.map((shot) => shot.player_id).filter((playerId): playerId is number => playerId !== null))).sort(
    (left, right) => left - right,
  );
  const shotTypes = Array.from(new Set(shots.map((shot) => shotTypeLabel(shot)))).sort();
  const outcomes = Array.from(new Set(shots.map((shot) => shot.outcome.call))).sort();
  const qualities = (["high", "mid", "low"] as const).filter((quality) => shots.some((shot) => qualityBandForShot(shot) === quality));
  return {
    players,
    shotTypes,
    outcomes,
    qualities: qualities.length ? [...qualities] : ["high", "mid", "low"],
  };
}

function writeShotCorrectionPayload(clipId: string, shot: ShotRecord) {
  const generatedAt = new Date().toISOString();
  const correction = {
    shot_id: shot.shot_id,
    clip_id: clipId,
    status: "pending",
    source: "web_replay_shot_trails",
    reason: "wrong_affordance",
    t: shot.t,
    frame: shot.frame,
    player_id: shot.player_id,
    current_shot_type: shot.shot_type,
    current_outcome_call: shot.outcome.call,
    created_at: generatedAt,
  };
  const storageKey = "pickleball.replay.shot_trails.corrections";
  const previous = readStoredCorrections(storageKey);
  const corrections = [
    ...previous.filter((entry) => entry.shot_id !== shot.shot_id),
    correction,
  ];
  const payload = {
    artifact_type: "racketsport_shot_corrections",
    schema_version: 1,
    generated_at: generatedAt,
    clip_id: clipId,
    corrections,
  };
  window.localStorage.setItem(storageKey, JSON.stringify(payload, null, 2));
  downloadJson("corrections.json", payload);
}

function readStoredCorrections(storageKey: string): Array<{ shot_id?: string }> {
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as { corrections?: unknown };
    if (!Array.isArray(parsed.corrections)) return [];
    return parsed.corrections.filter((entry): entry is { shot_id?: string } => Boolean(entry && typeof entry === "object"));
  } catch {
    return [];
  }
}

function downloadJson(filename: string, payload: unknown) {
  if (typeof document === "undefined" || typeof Blob === "undefined" || typeof URL === "undefined") return;
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
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

function FadingLineStrip({ points, color, maxOpacity }: { points: Vec3[]; color: string; maxOpacity: number }) {
  const segments = useMemo(
    () =>
      points.slice(1).map((point, index) => ({
        points: [points[index], point] as Vec3[],
        opacity: maxOpacity * ((index + 1) / Math.max(1, points.length - 1)),
      })),
    [points, maxOpacity],
  );
  return (
    <>
      {segments.map((segment, index) => (
        <FadingLineSegment key={`${index}-${segment.opacity.toFixed(3)}`} points={segment.points} color={color} opacity={segment.opacity} />
      ))}
    </>
  );
}

function FadingLineSegment({
  points,
  color,
  opacity,
  lineWidth = 1,
}: {
  points: Vec3[];
  color: string;
  opacity: number;
  lineWidth?: number;
}) {
  const geometry = useMemo(() => geometryFromPoints(points), [points]);
  return (
    <lineSegments geometry={geometry} renderOrder={18}>
      <lineBasicMaterial color={color} transparent opacity={opacity} linewidth={lineWidth} depthWrite={false} />
    </lineSegments>
  );
}

function DashedFadingLineSegment({
  from,
  to,
  color,
  opacity,
  lineWidth,
}: {
  from: Vec3;
  to: Vec3;
  color: string;
  opacity: number;
  lineWidth: number;
}) {
  const points = useMemo(() => dashedSegmentPoints(from, to, 7), [from, to]);
  return <FadingLineSegment points={points} color={color} opacity={opacity} lineWidth={lineWidth} />;
}

function dashedSegmentPoints(from: Vec3, to: Vec3, dashCount: number): Vec3[] {
  const points: Vec3[] = [];
  for (let index = 0; index < dashCount; index += 1) {
    const start = index / dashCount;
    const end = Math.min(1, start + 0.52 / dashCount);
    points.push(interpolateVec3(from, to, start), interpolateVec3(from, to, end));
  }
  return points;
}

function interpolateVec3(from: Vec3, to: Vec3, t: number): Vec3 {
  return [
    from[0] + (to[0] - from[0]) * t,
    from[1] + (to[1] - from[1]) * t,
    from[2] + (to[2] - from[2]) * t,
  ];
}

function PointCloud({ points, color, size, opacity = 1 }: { points: Vec3[]; color: string; size: number; opacity?: number }) {
  const geometry = useMemo(() => geometryFromPoints(points), [points]);
  return (
    <points geometry={geometry}>
      <pointsMaterial color={new Color(color)} size={size} sizeAttenuation transparent={opacity < 1} opacity={opacity} />
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

export function courtBounds(world: VirtualWorld) {
  const playableSegments = Object.entries(world.court.line_segments)
    .filter(([key]) => key !== "net")
    .map(([, segment]) => segment);
  const points = (playableSegments.length ? playableSegments : Object.values(world.court.line_segments)).flat();
  const xs = points.map((point) => point[0]).filter(Number.isFinite);
  const ys = points.map((point) => point[1]).filter(Number.isFinite);
  const rawMinX = xs.length ? Math.min(...xs) : -world.court.width_m / 2;
  const rawMaxX = xs.length ? Math.max(...xs) : world.court.width_m / 2;
  const rawMinY = ys.length ? Math.min(...ys) : -world.court.length_m / 2;
  const rawMaxY = ys.length ? Math.max(...ys) : world.court.length_m / 2;
  const rawCenterX = (rawMinX + rawMaxX) / 2;
  const rawCenterY = (rawMinY + rawMaxY) / 2;
  const rawWidth = Math.max(0, rawMaxX - rawMinX);
  const rawLength = Math.max(0, rawMaxY - rawMinY);
  const width = rawWidth >= world.court.width_m * 0.8 ? rawWidth : world.court.width_m;
  const length = rawLength >= world.court.length_m * 0.8 ? rawLength : world.court.length_m;
  const minX = rawCenterX - width / 2;
  const maxX = rawCenterX + width / 2;
  const minY = rawCenterY - length / 2;
  const maxY = rawCenterY + length / 2;
  return {
    centerX: (minX + maxX) / 2,
    centerY: (minY + maxY) / 2,
    minY,
    maxY,
    width: Math.max(1, width),
    length: Math.max(1, length),
  };
}

/**
 * Free-viewpoint orbit already exists (OrbitControls); these are fixed
 * starting poses a coach can jump to before continuing to orbit freely.
 */
export function cameraPresetPose(
  world: VirtualWorld,
  preset: CameraPreset,
  activePaddles: ActivePaddleFrame[] = [],
  selectedPlayerId: number | null = null,
  currentTime = 0,
): { position: Vec3; target: Vec3 } {
  const bounds = courtBounds(world);
  const groundTarget: Vec3 = [bounds.centerX, bounds.centerY, 0.35];
  if (preset === "court") return topDownCameraPose(bounds, false);
  if (preset === "follow_player") {
    const selectedPlayer = selectedPlayerId === null ? undefined : world.players.find((player) => player.id === selectedPlayerId);
    const floor = selectedPlayer ? floorWorldForFrame(frameForTime(selectedPlayer, currentTime)) : null;
    if (floor) {
      const target: Vec3 = [floor[0], floor[1], Math.max(0.8, floor[2] + 1.05)];
      return {
        position: [target[0] + 2.4, target[1] - 3.2, target[2] + 2.1],
        target,
      };
    }
    return topDownCameraPose(bounds, false);
  }
  if (preset === "free_orbit") {
    return {
      position: [bounds.centerX + bounds.width * 0.7, bounds.minY - bounds.length * 0.44, Math.max(4.8, bounds.length * 0.42)],
      target: groundTarget,
    };
  }
  if (preset === "paddle_review") {
    const paddlePose = paddleReviewCameraPose(activePaddles, selectedPlayerId);
    if (paddlePose) return paddlePose;
    return topDownCameraPose(bounds, false);
  }
  if (preset === "top_down" || preset === "shot_trails") {
    return topDownCameraPose(bounds, preset === "shot_trails");
  }
  if (preset === "behind_baseline") {
    return {
      position: [bounds.centerX, bounds.minY - bounds.length * 0.32, 2.1],
      target: groundTarget,
    };
  }
  return {
    position: [bounds.centerX, bounds.minY - bounds.length * 0.62, Math.max(5.4, bounds.length * 0.5)],
    target: groundTarget,
  };
}

function topDownCameraPose(bounds: ReturnType<typeof courtBounds>, wide: boolean): { position: Vec3; target: Vec3 } {
  return {
    position: [bounds.centerX, bounds.centerY, Math.max(wide ? 14 : 10, bounds.length * (wide ? 1.35 : 1.1))],
    target: [bounds.centerX, bounds.centerY, 0],
  };
}

function paddleReviewCameraPose(activePaddles: ActivePaddleFrame[], selectedPlayerId: number | null): { position: Vec3; target: Vec3 } | null {
  const selectedPaddles =
    selectedPlayerId === null ? activePaddles : activePaddles.filter((paddle) => paddle.playerId === selectedPlayerId);
  const paddles = selectedPaddles.length ? selectedPaddles : activePaddles;
  const points = paddles.flatMap((paddle) => paddleReviewPoints(paddle.frame));
  if (!points.length) return null;
  const center: Vec3 = [
    points.reduce((sum, point) => sum + point[0], 0) / points.length,
    points.reduce((sum, point) => sum + point[1], 0) / points.length,
    points.reduce((sum, point) => sum + point[2], 0) / points.length,
  ];
  const radius = Math.max(
    0.25,
    ...points.map((point) => Math.hypot(point[0] - center[0], point[1] - center[1], point[2] - center[2])),
  );
  const distance = Math.max(1.15, radius * 2.8);
  return {
    position: [center[0], center[1] - distance * 0.75, center[2] + distance * 1.35],
    target: center,
  };
}

function paddleReviewPoints(frame: VirtualWorldPaddleFrame): Vec3[] {
  if (frame.mesh_vertices_world.length) return frame.mesh_vertices_world;
  return [frame.pose_se3.t];
}

function floorWorldForFrame(frame: VirtualWorldFrame | undefined): Vec3 | null {
  return frame?.floor_world_xyz ?? (frame?.track_world_xy ? ([frame.track_world_xy[0], frame.track_world_xy[1], 0] as Vec3) : null);
}

function skeletonForFrame(frame: VirtualWorldFrame | undefined): BodyJointSkeleton | null {
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
  return { joints, bones: bonePairs.flatMap(([left, right]) => [joints[left], joints[right]]), boneNames: [] };
}

export type BodyJointSkeleton = { joints: Vec3[]; bones: Vec3[]; boneNames: Array<[string, string]> };

export const cocoWholeBodyCoreBoneNames: Array<[string, string]> = [
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
  ["left_shoulder", "right_shoulder"],
  ["left_hip", "right_hip"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["neck", "nose"],
];

const COCO_WHOLEBODY_133_JOINT_NAMES = [
  "nose",
  "left_eye",
  "right_eye",
  "left_ear",
  "right_ear",
  "left_shoulder",
  "right_shoulder",
  "left_elbow",
  "right_elbow",
  "left_wrist",
  "right_wrist",
  "left_hip",
  "right_hip",
  "left_knee",
  "right_knee",
  "left_ankle",
  "right_ankle",
  "left_big_toe",
  "left_small_toe",
  "left_heel",
  "right_big_toe",
  "right_small_toe",
  "right_heel",
  ...Array.from({ length: 68 }, (_, index) => `face-${index}`),
  "left_hand_root",
  ...Array.from({ length: 20 }, (_, index) => `left_hand_${index}`),
  "right_hand_root",
  ...Array.from({ length: 20 }, (_, index) => `right_hand_${index}`),
];

export function bodyJointSkeletonForFrame(
  frame: VirtualWorldFrame | undefined,
  jointNames?: readonly string[],
  options: { includeImplausible?: boolean } = {},
): BodyJointSkeleton | null {
  const joints = frame?.joints_world ?? [];
  if (!frame || joints.length < 2) return null;
  if (frame.skeleton_implausible === true && options.includeImplausible !== true) return null;
  const names = normalizedJointNames(jointNames, joints.length);
  const pointByName = jointPointLookup(joints, frame.joint_conf ?? [], names);
  const skeletonJoints: Vec3[] = [];
  const seenJointNames = new Set<string>();
  const bones: Vec3[] = [];
  const boneNames: Array<[string, string]> = [];
  for (const [leftName, rightName] of cocoWholeBodyCoreBoneNames) {
    const left = pointForJointName(leftName, pointByName);
    const right = pointForJointName(rightName, pointByName);
    if (!left || !right) continue;
    bones.push(left, right);
    boneNames.push([leftName, rightName]);
    for (const name of [leftName, rightName]) {
      if (name === "neck" || seenJointNames.has(name)) continue;
      const point = pointForJointName(name, pointByName);
      if (point) {
        skeletonJoints.push(point);
        seenJointNames.add(name);
      }
    }
  }
  return bones.length ? { joints: skeletonJoints, bones, boneNames } : null;
}

export function handJointPointsForFrame(frame: VirtualWorldFrame | undefined, jointNames?: readonly string[]): Vec3[] {
  const joints = frame?.joints_world ?? [];
  if (!frame || frame.skeleton_implausible === true || !joints.length) return [];
  const names = normalizedJointNames(jointNames, joints.length);
  return names
    .map((name, index) => ({ name, point: joints[index], conf: frame.joint_conf?.[index] }))
    .filter(({ name, point, conf }) => isHandJointName(name) && point && confidencePasses(conf))
    .map(({ point }) => point);
}

function normalizedJointNames(jointNames: readonly string[] | undefined, jointCount: number): string[] {
  if (jointNames?.length) return Array.from(jointNames);
  if (jointCount >= COCO_WHOLEBODY_133_JOINT_NAMES.length) return COCO_WHOLEBODY_133_JOINT_NAMES;
  return Array.from({ length: jointCount }, (_, index) => `joint_${index}`);
}

function jointPointLookup(joints: Vec3[], conf: number[], jointNames: readonly string[]): Map<string, Vec3> {
  const points = new Map<string, Vec3>();
  jointNames.forEach((name, index) => {
    const point = joints[index];
    if (point && confidencePasses(conf[index])) points.set(name, point);
  });
  return points;
}

function pointForJointName(name: string, pointByName: Map<string, Vec3>): Vec3 | null {
  if (name !== "neck") return pointByName.get(name) ?? null;
  const left = pointByName.get("left_shoulder");
  const right = pointByName.get("right_shoulder");
  if (!left || !right) return null;
  return [(left[0] + right[0]) / 2, (left[1] + right[1]) / 2, (left[2] + right[2]) / 2];
}

function confidencePasses(confidence: number | undefined): boolean {
  return confidence === undefined || !Number.isFinite(confidence) || confidence >= 0.05;
}

function isHandJointName(name: string): boolean {
  return (
    name.includes("hand") ||
    name.includes("thumb") ||
    name.includes("index") ||
    name.includes("middle") ||
    name.includes("ring") ||
    name.includes("pinky")
  );
}

export function paddleRenderGeometryForFrame(
  frame: VirtualWorldPaddleFrame,
  dimsIn: Record<string, number>,
): {
  vertices: Vec3[];
  faces: Array<[number, number, number]>;
  edgeSegments: Vec3[];
  normalSegment: Vec3[];
  normalTip: Vec3 | null;
  estimated: boolean;
  material: {
    fillColor: string;
    emissiveColor: string;
    edgeColor: string;
    normalColor: string;
    fillOpacity: number;
    edgeOpacity: number;
    edgeRadiusM: number;
    normalRadiusM: number;
    normalLengthM: number;
    normalTipRadiusM: number;
    normalOverlay: boolean;
  };
} {
  const estimated = frame.source.includes("wrist_proxy") || frame.render_only === true || frame.not_for_detection_metrics === true;
  const material = paddleRenderMaterial(estimated);
  const normalSegment = paddleFaceNormalSegment(frame);
  if (!estimated && frame.mesh_vertices_world.length > 4 && frame.mesh_faces.length > 2) {
    return {
      vertices: frame.mesh_vertices_world,
      faces: frame.mesh_faces,
      edgeSegments: paddleEdgeSegmentsForIndexedMesh(frame.mesh_vertices_world, frame.mesh_faces),
      normalSegment,
      normalTip: normalSegment.length === 2 ? normalSegment[1] : null,
      estimated,
      material,
    };
  }
  const widthM = (dimsIn.width ?? dimsIn.w ?? 7.5) * 0.0254;
  const lengthM = (dimsIn.length ?? dimsIn.h ?? 15.5) * 0.0254;
  const radius = Math.min(widthM, lengthM) * 0.18;
  const local = roundedRectanglePoints(widthM, lengthM, radius, 4);
  const center = frame.pose_se3.t;
  const thicknessM = Math.max(0.008, Math.min(0.022, (dimsIn.thickness ?? 0.55) * 0.0254));
  const frontCenter = transformPaddleLocalPoint3(frame.pose_se3.R, center, 0, 0, thicknessM / 2);
  const backCenter = transformPaddleLocalPoint3(frame.pose_se3.R, center, 0, 0, -thicknessM / 2);
  const front = local.map(([x, y]) => transformPaddleLocalPoint3(frame.pose_se3.R, center, x * 0.96, y * 0.96, thicknessM / 2));
  const middle = local.map(([x, y]) => transformPaddleLocalPoint3(frame.pose_se3.R, center, x, y, 0));
  const back = local.map(([x, y]) => transformPaddleLocalPoint3(frame.pose_se3.R, center, x * 0.96, y * 0.96, -thicknessM / 2));
  const vertices: Vec3[] = [frontCenter, backCenter, ...front, ...middle, ...back];
  const faces: Array<[number, number, number]> = [];
  const count = local.length;
  const frontStart = 2;
  const middleStart = frontStart + count;
  const backStart = middleStart + count;
  for (let index = 0; index < count; index += 1) {
    const next = (index + 1) % count;
    faces.push([0, frontStart + index, frontStart + next]);
    faces.push([1, backStart + next, backStart + index]);
    faces.push([frontStart + index, middleStart + index, middleStart + next], [frontStart + index, middleStart + next, frontStart + next]);
    faces.push([middleStart + index, backStart + index, backStart + next], [middleStart + index, backStart + next, middleStart + next]);
  }
  return {
    vertices,
    faces,
    edgeSegments: front.flatMap((point, index) => [point, front[(index + 1) % front.length]]),
    normalSegment,
    normalTip: normalSegment.length === 2 ? normalSegment[1] : null,
    estimated,
    material,
  };
}

function paddleRenderMaterial(estimated: boolean) {
  return estimated
    ? {
        fillColor: "#ffb454",
        emissiveColor: "#5a3500",
        edgeColor: "#2a1700",
        normalColor: "#23c9ff",
        fillOpacity: 0.58,
        edgeOpacity: 0.95,
        edgeRadiusM: 0.009,
        normalRadiusM: 0.026,
        normalLengthM: 0.52,
        normalTipRadiusM: 0.052,
        normalOverlay: true,
      }
    : {
        fillColor: "#e8ff34",
        emissiveColor: "#526000",
        edgeColor: "#2b3000",
        normalColor: "#1fa9ff",
        fillOpacity: 0.88,
        edgeOpacity: 0.96,
        edgeRadiusM: 0.012,
        normalRadiusM: 0.022,
        normalLengthM: 0.48,
        normalTipRadiusM: 0.046,
        normalOverlay: true,
      };
}

function paddleFaceNormalSegment(frame: VirtualWorldPaddleFrame): Vec3[] {
  const normal = normalizeVec3([
    frame.pose_se3.R[0]?.[2] ?? 0,
    frame.pose_se3.R[1]?.[2] ?? 0,
    frame.pose_se3.R[2]?.[2] ?? 0,
  ]);
  if (!normal) return [];
  const center = frame.pose_se3.t;
  const lengthM = paddleRenderMaterial(frame.source.includes("wrist_proxy") || frame.render_only === true || frame.not_for_detection_metrics === true)
    .normalLengthM;
  return [center, [center[0] + normal[0] * lengthM, center[1] + normal[1] * lengthM, center[2] + normal[2] * lengthM]];
}

function normalizeVec3(vector: Vec3): Vec3 | null {
  const length = Math.hypot(vector[0], vector[1], vector[2]);
  if (length <= 1e-9) return null;
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

export function paddleEdgeSegmentsForIndexedMesh(vertices: Vec3[], faces: Array<[number, number, number]>): Vec3[] {
  const edgeCounts = new Map<string, { count: number; edge: [number, number] }>();
  for (const face of faces) {
    for (const [left, right] of [
      [face[0], face[1]],
      [face[1], face[2]],
      [face[2], face[0]],
    ] as Array<[number, number]>) {
      if (!vertices[left] || !vertices[right]) continue;
      const key = left < right ? `${left}:${right}` : `${right}:${left}`;
      const existing = edgeCounts.get(key);
      if (existing) {
        existing.count += 1;
      } else {
        edgeCounts.set(key, { count: 1, edge: [left, right] });
      }
    }
  }
  const segments: Vec3[] = [];
  for (const { count, edge } of edgeCounts.values()) {
    if (count !== 1) continue;
    segments.push(vertices[edge[0]], vertices[edge[1]]);
  }
  return segments;
}

function roundedRectanglePoints(widthM: number, lengthM: number, radiusM: number, cornerSteps: number): Array<[number, number]> {
  const halfW = widthM / 2;
  const halfL = lengthM / 2;
  const corners = [
    { cx: halfW - radiusM, cy: halfL - radiusM, start: 0 },
    { cx: -halfW + radiusM, cy: halfL - radiusM, start: Math.PI / 2 },
    { cx: -halfW + radiusM, cy: -halfL + radiusM, start: Math.PI },
    { cx: halfW - radiusM, cy: -halfL + radiusM, start: (3 * Math.PI) / 2 },
  ];
  const points: Array<[number, number]> = [];
  for (const corner of corners) {
    for (let step = 0; step <= cornerSteps; step += 1) {
      const angle = corner.start + (step / cornerSteps) * (Math.PI / 2);
      points.push([corner.cx + Math.cos(angle) * radiusM, corner.cy + Math.sin(angle) * radiusM]);
    }
  }
  return points;
}

function transformPaddleLocalPoint(R: VirtualWorldPaddleFrame["pose_se3"]["R"], t: Vec3, x: number, y: number): Vec3 {
  return transformPaddleLocalPoint3(R, t, x, y, 0);
}

function transformPaddleLocalPoint3(R: VirtualWorldPaddleFrame["pose_se3"]["R"], t: Vec3, x: number, y: number, z: number): Vec3 {
  const axisX: Vec3 = [R[0][0], R[1][0], R[2][0]];
  const axisY: Vec3 = [R[0][1], R[1][1], R[2][1]];
  const axisZ: Vec3 = [R[0][2], R[1][2], R[2][2]];
  return [
    t[0] + axisX[0] * x + axisY[0] * y + axisZ[0] * z,
    t[1] + axisX[1] * x + axisY[1] * y + axisZ[1] * z,
    t[2] + axisX[2] * x + axisY[2] * y + axisZ[2] * z,
  ];
}

function sampleMeshPoints(points: Vec3[], maxPoints: number): Vec3[] {
  if (points.length <= maxPoints) return points;
  const stride = Math.ceil(points.length / maxPoints);
  return points.filter((_, index) => index % stride === 0);
}

export function vertexDebugPointsForFrame(frame: VirtualWorldFrame | undefined, enabled: boolean, maxPoints: number): Vec3[] {
  if (!enabled) return [];
  return sampleMeshPoints(frame?.mesh_vertices_world ?? [], maxPoints);
}

async function fetchJson(url: string) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`failed to fetch ${url}: ${response.status}`);
  return response.json() as Promise<unknown>;
}

export async function loadOptionalArtifact<T>(
  capability: string,
  loader: () => Promise<T>,
  onFailure: (capability: string, error: unknown) => void,
): Promise<T | null> {
  try {
    return await loader();
  } catch (error) {
    onFailure(capability, error);
    return null;
  }
}

function xywhToXyxy(value?: number[]): [number, number, number, number] | null {
  if (!Array.isArray(value) || value.length < 4) return null;
  return [value[0], value[1], value[0] + value[2], value[1] + value[3]];
}

function isVec3(value: Vec3 | null | undefined): value is Vec3 {
  return Array.isArray(value) && value.length === 3;
}
