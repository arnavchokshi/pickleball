import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { BufferAttribute, BufferGeometry, Color, DoubleSide, Quaternion, Vector3 as ThreeVector3 } from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { MeshoptDecoder } from "three/examples/jsm/libs/meshopt_decoder.module.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

import { CoachingCardPanel, coachingCardForTimeline } from "./CoachingCard";
import { activeReplayPointForTime, parseReplayScene, resolveReplaySceneAssetUrl, type ReplayScene } from "./replayScene";
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
import { UploadPanel } from "./UploadPanel";
import {
  activeBallContactPlayerIds,
  ballTrailSegmentsForTime,
  ballRenderInfoForTime,
  bodyMeshDebugSnapshot,
  bodyMeshIndexWindowForTime,
  bodyMeshStatusTileValue,
  contactEventCount,
  entityCoverageReadout,
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
  activePaddleFramesForTime,
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
  type BallGhostMarker,
  type BallInflections,
  type BodyMesh,
  type BodyMeshDebugSnapshot,
  type BodyMeshFaces,
  type BodyMeshIndex,
  type BodyMeshLoadStatus,
  type CoachingCardFacts,
  type ContactWindows,
  type LabelOverlayPayload,
  type PhysicsRefinement,
  type ReviewedBounces,
  type RallySpans,
  type TimelineChapter,
  type TimelineMarker,
  type TrustBand,
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
  parseViewStateFromSearch,
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

export type CameraPreset = "broadcast" | "behind_baseline" | "top_down" | "shot_trails";

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

const CAMERA_PRESET_LABELS: Record<CameraPreset, string> = {
  broadcast: "Broadcast",
  behind_baseline: "Behind Baseline",
  top_down: "Top Down",
  shot_trails: "Shot Trails",
};

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

export default function App() {
  const initialTime = useMemo(() => startTimeFromSearch(window.location.search), []);
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
  const [shotTrailFilters, setShotTrailFilters] = useState<ShotTrailFilters>(DEFAULT_SHOT_TRAIL_FILTERS);
  const [selectedShotId, setSelectedShotId] = useState<string | null>(null);
  const [correctionStatus, setCorrectionStatus] = useState<string | null>(null);
  const [bodyMesh, setBodyMesh] = useState<BodyMesh | null>(null);
  const [bodyMeshIndex, setBodyMeshIndex] = useState<BodyMeshIndex | null>(null);
  const [bodyMeshFaces, setBodyMeshFaces] = useState<BodyMeshFaces | null>(null);
  const [bodyMeshLoadStatus, setBodyMeshLoadStatus] = useState<BodyMeshLoadStatus>(INITIAL_BODY_MESH_STATUS);
  const [replayScene, setReplayScene] = useState<ReplayScene | null>(null);
  const [currentTime, setCurrentTime] = useState(initialTime);
  const [videoDuration, setVideoDuration] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>("broadcast");
  const [viewState, setViewState] = useState<ViewState>(() => parseViewStateFromSearch(window.location.search));
  const [fps, setFps] = useState(0);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const bodyMeshChunkCacheRef = useRef<Map<number, BodyMesh>>(new Map());
  const bodyMeshChunkInflightRef = useRef<Map<number, Promise<BodyMesh>>>(new Map());
  const currentTimeRef = useRef(0);
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
      setSelectedShotId(null);
      setCorrectionStatus(null);
      setBodyMeshIndex(null);
      setBodyMeshFaces(null);
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
        const ballInflectionsPayload = manifestPayload.ball_inflections_url
          ? parseBallInflections(await fetchJson(manifestPayload.ball_inflections_url))
          : null;
        const ballArcEventsSelectedPayload = manifestPayload.events_selected_url
          ? parseBallArcEventsSelected(await fetchJson(manifestPayload.events_selected_url))
          : null;
        const reviewedBouncesPayload = manifestPayload.reviewed_bounces_url
          ? parseReviewedBounces(await fetchJson(manifestPayload.reviewed_bounces_url))
          : null;
        const rallySpansPayload = manifestPayload.rally_spans_url
          ? parseRallySpans(await fetchJson(manifestPayload.rally_spans_url))
          : null;
        const replayScenePayload = manifestPayload.replay_scene_url
          ? parseReplayScene(await fetchJson(manifestPayload.replay_scene_url))
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
            throw error;
          }
          const facesUrl = resolveBodyMeshAssetUrl(indexUrl, bodyMeshIndexPayload.faces_url);
          try {
            bodyMeshFacesPayload = parseBodyMeshFaces(await fetchJson(facesUrl));
          } catch (error) {
            warnBodyMeshFailure({ stage: "faces", url: facesUrl, error });
            if (!cancelled) setBodyMeshLoadStatus(meshFailureStatus("faces", facesUrl, error));
            throw error;
          }
        }
        const coachingFactsUrl = coachingFactsUrlFromSearch(window.location.search, manifestPayload);
        const coachingFactsPayload = coachingFactsUrl ? parseCoachingCardFacts(await fetchJson(coachingFactsUrl)) : null;
        const shotsPayload = manifestPayload.shots_url ? parseShots(await fetchJson(manifestPayload.shots_url)) : null;
        const ballArcSolvedPayload = manifestPayload.ball_arc_solved_url
          ? parseBallArcSolved(await fetchJson(manifestPayload.ball_arc_solved_url))
          : null;
        if (cancelled) return;
        setManifest(manifestPayload);
        setWorld(worldPayload);
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
        setSelectedShotId(null);
        setCorrectionStatus(null);
        setBodyMeshIndex(bodyMeshIndexPayload);
        setBodyMeshFaces(bodyMeshFacesPayload);
        setBodyMesh(null);
        setBodyMeshLoadStatus(bodyMeshIndexPayload ? meshLoadStatus("index_ready", { stage: "index" }) : INITIAL_BODY_MESH_STATUS);
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
        while (bodyMeshChunkCacheRef.current.size > 2) {
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
    currentTimeRef.current = currentTime;
  }, [currentTime]);

  useEffect(() => {
    const nextSearch = viewStateToSearch(window.location.search, viewState);
    if (nextSearch !== window.location.search) {
      window.history.replaceState(null, "", `${window.location.pathname}${nextSearch}${window.location.hash}`);
    }
  }, [viewState]);

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
  const videoBallOverlay = useMemo(() => videoBallOverlayForTime(world, currentTime), [world, currentTime]);
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
    () => solidBodyMeshFramesForTime(bodyMesh, contactWindows, currentTime, world),
    [bodyMesh, contactWindows, currentTime, world],
  );
  const activePaddles = useMemo(() => activePaddleFramesForTime(world, currentTime), [currentTime, world]);
  const solidGeometryCache = useMemo(() => createSolidBodyMeshGeometryCache(bodyMesh), [bodyMesh]);
  useEffect(() => () => solidGeometryCache.dispose(), [solidGeometryCache]);
  const renderedSolidMeshPlayers = useMemo(() => solidMeshRenderedPlayerCount(activeBodyMeshes), [activeBodyMeshes]);
  const solidMeshTileValue = useMemo(
    () => bodyMeshStatusTileValue(renderedSolidMeshPlayers, bodyMeshLoadStatus),
    [bodyMeshLoadStatus, renderedSolidMeshPlayers],
  );
  const meshDebugSnapshot = useMemo(
    () =>
      bodyMeshDebugSnapshot({
        bodyMeshIndex,
        bodyMesh,
        world,
        currentTime,
        loadStatus: bodyMeshLoadStatus,
        activeBodyMeshes,
      }),
    [activeBodyMeshes, bodyMesh, bodyMeshIndex, bodyMeshLoadStatus, currentTime, world],
  );
  const sceneLayers = useMemo(
    () =>
      sceneLayerSnapshotForTime({
        world,
        bodyMesh,
        contactWindows,
        ballArcEventsSelected,
        currentTime,
        viewState,
      }),
    [ballArcEventsSelected, bodyMesh, contactWindows, currentTime, viewState, world],
  );
  const worldEventMarkers = useMemo(
    () => eventMarkersForTime(world, contactWindows, currentTime),
    [contactWindows, currentTime, world],
  );
  const viewerContactPlayerIds = useMemo(
    () => contactPlayerIdsForViewer(activeContactPlayerIds, activeBodyMeshes),
    [activeContactPlayerIds, activeBodyMeshes],
  );
  const viewBox = useMemo(() => labelViewBox(labelOverlay), [labelOverlay]);
  const activeReplayPoint = useMemo(() => (replayScene ? activeReplayPointForTime(replayScene, currentTime) : undefined), [replayScene, currentTime]);
  const coverageGapActive = coverage.lastTime !== null && currentTime > coverage.lastTime + Math.max(0.12, 1 / (world.fps || 30));
  const contactReadout = contactReadoutText(activeContactPlayerIds, activeBodyMeshes);
  const ballReadout = ballRenderText(ballRenderInfo.mode, videoBallOverlay);
  const warningsReadout = useMemo(() => worldWarningsReadout(world), [world]);
  const ballCoverage = useMemo(() => entityCoverageReadout("Ball", world.ball).replace(/^Ball\s+/, ""), [world.ball]);
  const timelineMarkers = useMemo(
    () => timelineMarkersFromArtifacts(contactWindows, ballInflections, reviewedBounces),
    [contactWindows, ballInflections, reviewedBounces],
  );
  const timelineDuration = videoDuration > 0 ? videoDuration : coverage.lastTime ?? 0;
  const timelineChapters = useMemo(
    () => {
      const authoritativeChapters = timelineChaptersFromRallySpans(rallySpans);
      return authoritativeChapters.length ? authoritativeChapters : timelineChaptersFromMarkers(timelineMarkers, timelineDuration);
    },
    [rallySpans, timelineMarkers, timelineDuration],
  );
  const coachingPlayerIds = useMemo(() => world.players.map((player) => String(player.id)), [world.players]);
  const coachingCard = useMemo(
    () => coachingCardForTimeline(coachingFacts, timelineChapters, currentTime, coachingPlayerIds),
    [coachingFacts, timelineChapters, currentTime, coachingPlayerIds],
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
    if (video) video.currentTime = Math.max(0, seconds);
    setCurrentTime(Math.max(0, seconds));
  };

  const jumpToEvent = (direction: "previous" | "next") => {
    const eventTime = timelineEventJump(timelineMarkers, currentTime, direction);
    if (eventTime !== null) seekTo(eventTime);
  };

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
    setCurrentTime(shot.t);
    if (videoRef.current) videoRef.current.currentTime = shot.t;
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
          <Metric label="Solid Mesh" value={solidMeshTileValue} />
          <Metric label="Floor Frames" value={stats.floorPlacedFrames} />
          <Metric label="Ball Contacts" value={contactEventCount(contactWindows)} />
          <Metric label="Replay Points" value={replayScene?.points.length ?? 0} />
          <Metric label="Player Span" value={coverage.lastTime === null ? "0.0s" : `${coverage.lastTime.toFixed(1)}s`} />
          <Metric label="Warnings" value={warningsReadout} />
          <Metric label="Ball Coverage" value={ballCoverage} />
          <Metric label="3D FPS" value={fps > 0 ? fps.toFixed(1) : "measuring..."} />
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
          <div className="camera-preset-bar" role="group" aria-label="Camera presets">
            {(Object.keys(CAMERA_PRESET_LABELS) as CameraPreset[]).map((preset) => (
              <button
                key={preset}
                type="button"
                className={preset === cameraPreset ? "camera-preset active" : "camera-preset"}
                onClick={() => setCameraPreset(preset)}
              >
                {CAMERA_PRESET_LABELS[preset]}
              </button>
            ))}
          </div>
          {shotTrailsMode ? (
            <ShotTrailsControls
              filters={shotTrailFilters}
              options={shotFilterOptions}
              totalCount={shots?.shots.length ?? 0}
              visibleCount={filteredShots.length}
              drawableCount={drawableShotCount}
              hasArcSource={Boolean(ballArcSolved)}
              onFilterChange={updateShotFilter}
            />
          ) : (
            <ViewLayerPanel
              viewState={viewState}
              playerIds={playerIds}
              onToggleLayer={toggleLayer}
              onBallFocus={focusBall}
              onPlayerFocus={focusPlayer}
              onClearFocus={resetFocus}
              onResetView={resetView}
            />
          )}
          <Canvas dpr={[1, 1.5]} camera={{ position: [0, -18, 8.5], fov: 50, near: 0.05, far: 100 }}>
            <color attach="background" args={[shotTrailsMode ? "#fbfaf4" : "#111315"]} />
            <ambientLight intensity={shotTrailsMode ? 2.25 : 1.8} />
            <directionalLight position={[0, -4, 8]} intensity={shotTrailsMode ? 1.4 : 2.2} />
            <FpsProbe onSample={setFps} />
            <OrbitRig world={world} preset={cameraPreset} />
            <CourtSurface world={world} muted={shotTrailsMode} />
            <CourtLines world={world} muted={shotTrailsMode} />
            <NetAssembly world={world} muted={shotTrailsMode} />
            {shotTrailsMode ? (
              <ShotTrailsLayer groups={shotTrailGroups} selectedShotId={selectedShotId} onSelectShot={selectShot} />
            ) : (
              <>
                <ReplayGlbLayer replayScene={replayScene} replaySceneUrl={manifest?.replay_scene_url ?? null} currentTime={currentTime} />
                {sceneLayers.playerTrails.visible ? <PlayerMotionTrails world={world} currentTime={currentTime} viewState={viewState} /> : null}
                <Players
                  world={world}
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
                {sceneLayers.paddles.visible ? <Paddles paddles={activePaddles} viewState={viewState} onSelectPlayer={focusPlayer} /> : null}
                {sceneLayers.playerSolidMeshes.visible ? (
                  <SolidBodyMeshes meshes={activeBodyMeshes} geometryCache={solidGeometryCache} viewState={viewState} onSelectPlayer={focusPlayer} />
                ) : null}
                {sceneLayers.eventMarkers.visible ? <WorldEventMarkers markers={worldEventMarkers} viewState={viewState} /> : null}
                {sceneLayers.ballTrail.visible ? (
                  <BallTrail
                    world={world}
                    currentTime={currentTime}
                    ballArcEventsSelected={ballArcEventsSelected}
                    focusStyle={entityFocusStyle(viewState, { kind: "ball" })}
                  />
                ) : null}
                {sceneLayers.ballDot.visible ? (
                  <Ball world={world} currentTime={currentTime} focusStyle={entityFocusStyle(viewState, { kind: "ball" })} />
                ) : null}
              </>
            )}
          </Canvas>
          {coverageGapActive ? <div className="world-warning">No player artifact coverage after {coverage.lastTime?.toFixed(2)}s</div> : null}
          {sceneLayers.debugPointClouds.visible ? <MeshDebugReadout snapshot={meshDebugSnapshot} /> : null}
          {shotTrailsMode ? (
            <ShotDetailPanel
              shot={selectedShot}
              groups={shotTrailGroups}
              correctionStatus={correctionStatus}
              onWrong={writeCorrection}
            />
          ) : null}
          <div className="scene-legend">
            <span><i className="swatch floor" /> floor</span>
            <span><i className="swatch mesh" /> BODY mesh</span>
            <span><i className="swatch joints" /> BODY joints</span>
            <span><i className="swatch ball" /> ball</span>
            {shotTrailsMode ? <span><i className="swatch shot" /> shot trail</span> : null}
            <span><i className="swatch ball-ghost" /> 2D-only ball</span>
            <span><i className="swatch badge-preview" /> preview</span>
            <span><i className="swatch badge-low" /> low confidence</span>
          </div>
        </div>
      </section>

      <section className="provenance-band">
        <TrustBandPanel label="Court (CAL)" trustBand={world.court.trust_band} />
        <TrustBandPanel label="Ball (BALL)" trustBand={world.ball.trust_band} />
        <PlayerTrustBandPanels players={world.players} />
      </section>

      <CoachingCardPanel card={coachingCard} />

      <section className="details-band">
        <p>Physics modes: {stats.physicsModes.length ? stats.physicsModes.join(", ") : "none"}</p>
        <p>{physics ? `Physics artifact: ${physics.physics}; FOOT-2 done: ${String(physics.foot2_done)}` : "Physics artifact: none"}</p>
        <p>{replayScene ? replaySceneReadout(replayScene, activeReplayPoint?.id ?? null) : "Replay scene: none"}</p>
        <p>{annotationSourceReadout(trustedAnnotationSources)}</p>
        <p>{coverageReadout(coverage, videoDuration)}</p>
        <p>{entityCoverageReadout("Ball", world.ball)}</p>
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
  onToggleLayer,
  onBallFocus,
  onPlayerFocus,
  onClearFocus,
  onResetView,
}: {
  viewState: ViewState;
  playerIds: number[];
  onToggleLayer: (layer: ViewLayerKey) => void;
  onBallFocus: () => void;
  onPlayerFocus: (playerId: number) => void;
  onClearFocus: () => void;
  onResetView: () => void;
}) {
  const groups = groupLayerDefinitions(VIEW_LAYER_DEFINITIONS);
  return (
    <div className="layer-panel" aria-label="Layer controls">
      <div className="layer-panel-header">
        <span>Layers</span>
        <button type="button" className="layer-reset" onClick={onResetView}>
          Reset
        </button>
      </div>
      {groups.map(([group, definitions]) => (
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

function TimelineStrip({
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
            <button
              key={chapter.index}
              type="button"
              className={`timeline-chapter ${chapter.badge}`}
              style={{ left: `${left}%`, width: `${Math.max(1.4, right - left)}%` }}
              title={`${chapter.label} ${chapter.t0.toFixed(2)}s-${chapter.t1.toFixed(2)}s`}
              aria-label={`Jump to ${chapter.label}`}
              onClick={(event) => {
                event.stopPropagation();
                onSeek(chapter.t0);
              }}
            >
              <span>{chapter.label}</span>
            </button>
          );
        })}
        {markers.map((marker, index) => (
          <button
            key={`${marker.kind}-${marker.t}-${index}`}
            type="button"
            className={`timeline-marker ${marker.kind} ${marker.badge} ${marker.humanReviewed ? "human-reviewed" : ""}`}
            style={{ left: `${Math.min(100, Math.max(0, (marker.t / duration) * 100))}%` }}
            title={`${marker.label} (${marker.humanReviewed ? "human reviewed, " : ""}${marker.badge.replace("_", " ")}, confidence ${marker.confidence.toFixed(2)})`}
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
    <div className="trust-band-card">
      <div className="trust-band-header">
        <span>{label}</span>
        <span className={`trust-badge-chip ${trustBand?.badge ?? "none"}`}>{trustBandChipText(trustBand)}</span>
      </div>
      <p>{trustBand?.reason ?? "No trust-band provenance on this artifact."}</p>
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

function OrbitRig({ world, preset }: { world: VirtualWorld; preset: CameraPreset }) {
  const { camera, gl } = useThree();
  const controlsRef = useRef<OrbitControls | null>(null);
  const pose = useMemo(() => cameraPresetPose(world, preset), [world, preset]);
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

function CourtSurface({ world, muted = false }: { world: VirtualWorld; muted?: boolean }) {
  const bounds = courtBounds(world);
  return (
    <mesh position={[bounds.centerX, bounds.centerY, -0.012]}>
      <planeGeometry args={[bounds.width, bounds.length]} />
      <meshStandardMaterial color={muted ? "#ece8df" : "#1d7250"} roughness={0.82} metalness={0.02} />
    </mesh>
  );
}

function CourtLines({ world, muted = false }: { world: VirtualWorld; muted?: boolean }) {
  const courtPoints = Object.values(world.court.line_segments).flat();
  const netPoints = world.court.net.endpoints;
  return (
    <>
      <LineSegments points={courtPoints} color={muted ? "#9ca39f" : "#e9f4e8"} opacity={muted ? 0.7 : 1} />
      <LineSegments points={netPoints} color={muted ? "#727a76" : "#ffcf5a"} opacity={muted ? 0.7 : 1} />
    </>
  );
}

function NetAssembly({ world, muted = false }: { world: VirtualWorld; muted?: boolean }) {
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
        <meshStandardMaterial color={muted ? "#767e7a" : "#9fd3d6"} transparent opacity={muted ? 0.14 : 0.24} roughness={0.7} />
      </mesh>
      <mesh position={[left[0], left[1], world.court.net.post_height_m / 2]}>
        <boxGeometry args={[0.075, 0.075, world.court.net.post_height_m]} />
        <meshStandardMaterial color={muted ? "#767e7a" : "#f2f2e8"} />
      </mesh>
      <mesh position={[right[0], right[1], world.court.net.post_height_m / 2]}>
        <boxGeometry args={[0.075, 0.075, world.court.net.post_height_m]} />
        <meshStandardMaterial color={muted ? "#767e7a" : "#f2f2e8"} />
      </mesh>
      <LineSegments points={[topLeft, centerTop, centerTop, topRight]} color={muted ? "#606966" : "#ffcf5a"} opacity={muted ? 0.7 : 1} />
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
        const frame = frameForTime(player, currentTime);
        return (
          <Player
            key={player.id}
            player={player}
            frame={frame}
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
  const proxySkeleton = bodySkeleton || frame?.skeleton_implausible ? null : skeletonForFrame(frame);
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
  geometryCache,
  viewState,
  onSelectPlayer,
}: {
  meshes: ActiveBodyMeshFrame[];
  geometryCache: SolidBodyMeshGeometryCache;
  viewState: ViewState;
  onSelectPlayer: (playerId: number) => void;
}) {
  return (
    <>
      {meshes.map(({ playerId, meshPlayerId, frame, presenceOpacity }) => (
        <SolidBodyMesh
          key={`${playerId}-${frame.frame_idx}`}
          playerId={playerId}
          meshPlayerId={meshPlayerId}
          frame={frame}
          presenceOpacity={presenceOpacity}
          geometryCache={geometryCache}
          focusStyle={entityFocusStyle(viewState, { kind: "player", playerId })}
          onSelectPlayer={onSelectPlayer}
        />
      ))}
    </>
  );
}

function Paddles({
  paddles,
  viewState,
  onSelectPlayer,
}: {
  paddles: ActivePaddleFrame[];
  viewState: ViewState;
  onSelectPlayer: (playerId: number) => void;
}) {
  return (
    <>
      {paddles.map((paddle) => (
        <PaddleProxy
          key={`${paddle.playerId}-${paddle.frame.t}`}
          paddle={paddle}
          focusStyle={entityFocusStyle(viewState, { kind: "player", playerId: paddle.playerId })}
          onSelectPlayer={onSelectPlayer}
        />
      ))}
    </>
  );
}

function PaddleProxy({
  paddle,
  focusStyle,
  onSelectPlayer,
}: {
  paddle: ActivePaddleFrame;
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
  const opacity = focusStyle.dimmed ? 0.18 : focusStyle.highlighted ? 0.72 : paddle.estimated ? 0.46 : 0.64;
  return (
    <mesh
      geometry={geometry}
      renderOrder={22}
      onClick={(event) => {
        event.stopPropagation();
        onSelectPlayer(paddle.playerId);
      }}
    >
      <meshStandardMaterial
        color={focusStyle.highlighted ? "#dfff3d" : paddle.estimated ? "#ffb454" : "#e8ff34"}
        emissive={focusStyle.highlighted ? "#425800" : paddle.estimated ? "#3c2600" : "#526000"}
        roughness={0.5}
        metalness={0.02}
        transparent
        opacity={opacity}
        side={DoubleSide}
        depthWrite={false}
      />
    </mesh>
  );
}

function SolidBodyMesh({
  playerId,
  meshPlayerId,
  frame,
  presenceOpacity,
  geometryCache,
  focusStyle,
  onSelectPlayer,
}: {
  playerId: number;
  meshPlayerId: number;
  frame: ActiveBodyMeshFrame["frame"];
  presenceOpacity: number;
  geometryCache: SolidBodyMeshGeometryCache;
  focusStyle: EntityFocusStyle;
  onSelectPlayer: (playerId: number) => void;
}) {
  const geometry = useMemo(
    () => geometryForSolidBodyMeshFrame(geometryCache, meshPlayerId, frame),
    [frame, geometryCache, meshPlayerId],
  );
  const baseOpacity = bodyMeshOpacityFromBlendWeight(frame, presenceOpacity);
  const opacity = focusStyle.dimmed ? Math.min(baseOpacity, 0.14) : focusStyle.highlighted ? Math.min(0.9, baseOpacity + 0.16) : baseOpacity;
  return (
    <mesh
      geometry={geometry}
      renderOrder={20}
      onClick={(event) => {
        event.stopPropagation();
        onSelectPlayer(playerId);
      }}
    >
      <meshStandardMaterial
        color={focusStyle.highlighted ? "#dfff3d" : "#b4f2bf"}
        emissive={focusStyle.highlighted ? "#425800" : "#102d18"}
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

function Ball({ world, currentTime, focusStyle }: { world: VirtualWorld; currentTime: number; focusStyle: EntityFocusStyle }) {
  const info = ballRenderInfoForTime(world, currentTime);
  if (info.ghost) return <BallGhostMarkerRing ghost={info.ghost} focusStyle={focusStyle} />;
  if (!info.frame?.world_xyz || !info.render3d) return null;
  const isApprox = info.mode === "court_plane_projection";
  const badge = effectiveTrustBadge(world.ball.trust_band);
  const isLowConfidence = badge === "low_confidence";
  const color = isLowConfidence ? "#9aa0a8" : isApprox ? "#ffcf5a" : "#e8ff34";
  const emissive = isLowConfidence ? "#2a2c30" : isApprox ? "#4f3f00" : "#526000";
  const opacity = focusStyle.dimmed ? 0.24 : isLowConfidence ? 0.6 : 1;
  const radius = focusStyle.highlighted ? 0.082 : 0.055;
  return (
    <group position={info.frame.world_xyz}>
      {focusStyle.highlighted ? (
        <mesh>
          <sphereGeometry args={[0.15, 24, 24]} />
          <meshStandardMaterial color="#dfff3d" emissive="#526000" transparent opacity={0.22} depthWrite={false} />
        </mesh>
      ) : null}
      <mesh>
        <sphereGeometry args={[radius, 16, 16]} />
        <meshStandardMaterial color={color} emissive={focusStyle.highlighted ? "#9bb000" : emissive} transparent opacity={opacity} />
      </mesh>
    </group>
  );
}

function BallGhostMarkerRing({ ghost, focusStyle }: { ghost: BallGhostMarker; focusStyle: EntityFocusStyle }) {
  const opacity = focusStyle.dimmed ? 0.18 : focusStyle.highlighted ? 0.72 : 0.42;
  const scale = focusStyle.highlighted ? 1.35 : 1;
  return (
    <group position={ghost.position} scale={[scale, scale, scale]} userData={{ label: ghost.label }}>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.18, 0.012, 8, 36]} />
        <meshStandardMaterial color="#63d9ff" emissive="#10384a" transparent opacity={opacity} depthWrite={false} />
      </mesh>
      <mesh position={[0, 0, 0.012]} rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.1, 0.18, 36]} />
        <meshStandardMaterial color="#63d9ff" transparent opacity={opacity * 0.22} depthWrite={false} side={DoubleSide} />
      </mesh>
    </group>
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

function BoneSegments({ points, color, opacity, radius }: { points: Vec3[]; color: string; opacity: number; radius: number }) {
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
        <BoneSegment key={`${index}-${from.join(",")}-${to.join(",")}`} from={from} to={to} color={color} opacity={opacity} radius={radius} />
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
}: {
  from: Vec3;
  to: Vec3;
  color: string;
  opacity: number;
  radius: number;
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
    <mesh position={transform.position} quaternion={transform.quaternion} renderOrder={19}>
      <cylinderGeometry args={[radius, radius, transform.length, 8]} />
      <meshStandardMaterial color={color} transparent opacity={opacity} roughness={0.5} metalness={0.02} depthWrite={false} />
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

/**
 * Free-viewpoint orbit already exists (OrbitControls); these are fixed
 * starting poses a coach can jump to before continuing to orbit freely.
 */
export function cameraPresetPose(world: VirtualWorld, preset: CameraPreset): { position: Vec3; target: Vec3 } {
  const bounds = courtBounds(world);
  const groundTarget: Vec3 = [bounds.centerX, bounds.centerY, 0.35];
  if (preset === "top_down" || preset === "shot_trails") {
    return {
      position: [bounds.centerX, bounds.centerY, Math.max(preset === "shot_trails" ? 14 : 10, bounds.length * (preset === "shot_trails" ? 1.35 : 1.1))],
      target: [bounds.centerX, bounds.centerY, 0],
    };
  }
  if (preset === "behind_baseline") {
    return {
      position: [bounds.centerX, bounds.minY - bounds.length * 0.32, 2.1],
      target: groundTarget,
    };
  }
  return {
    position: [bounds.centerX, bounds.minY - bounds.length * 0.86, Math.max(6.5, bounds.length * 0.64)],
    target: groundTarget,
  };
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
): { vertices: Vec3[]; faces: Array<[number, number, number]>; estimated: boolean } {
  const widthM = (dimsIn.width ?? dimsIn.w ?? 7.5) * 0.0254;
  const lengthM = (dimsIn.length ?? dimsIn.h ?? 15.5) * 0.0254;
  const radius = Math.min(widthM, lengthM) * 0.18;
  const local = roundedRectanglePoints(widthM, lengthM, radius, 4);
  const center = frame.pose_se3.t;
  const vertices: Vec3[] = [center, ...local.map(([x, y]) => transformPaddleLocalPoint(frame.pose_se3.R, center, x, y))];
  const faces: Array<[number, number, number]> = [];
  for (let index = 1; index < vertices.length; index += 1) {
    const next = index === vertices.length - 1 ? 1 : index + 1;
    faces.push([0, index, next]);
  }
  return { vertices, faces, estimated: frame.source.includes("wrist_proxy") || frame.render_only === true || frame.not_for_detection_metrics === true };
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
  const axisX: Vec3 = [R[0][0], R[1][0], R[2][0]];
  const axisY: Vec3 = [R[0][1], R[1][1], R[2][1]];
  return [t[0] + axisX[0] * x + axisY[0] * y, t[1] + axisX[1] * x + axisY[1] * y, t[2] + axisX[2] * x + axisY[2] * y];
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

function xywhToXyxy(value?: number[]): [number, number, number, number] | null {
  if (!Array.isArray(value) || value.length < 4) return null;
  return [value[0], value[1], value[0] + value[2], value[1] + value[3]];
}

function isVec3(value: Vec3 | null | undefined): value is Vec3 {
  return Array.isArray(value) && value.length === 3;
}
