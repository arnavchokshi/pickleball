import {
  activeBallContactPlayerIds,
  activePaddleFramesForTime,
  ballRenderInfoForTime,
  ballTrailSegmentsForTime,
  frameForTime,
  playerTrailPointsForTime,
  solidBodyMeshFramesForTime,
  type BallArcEventsSelected,
  type BodyMesh,
  type ContactWindowEvent,
  type ContactWindows,
  type Vec3,
  type VirtualWorld,
  type VirtualWorldFrame,
  type VirtualWorldPlayer,
} from "./viewerData";
import { ballHudStateForTime, buildBallTrail, samplesFromVirtualWorld, type BallTrailSample } from "./components/modules/ballTrail";

export type ViewLayerKey =
  | "ballDot"
  | "ballTrail"
  | "paddles"
  | "paddleNormals"
  | "playerSkeletons"
  | "playerSolidMeshes"
  | "playerTrails"
  | "floorContactMarkers"
  | "eventMarkers"
  | "handJointPoints"
  | "implausibleSkeletons"
  | "debugPointClouds";

export type ViewLayerDefinition = {
  key: ViewLayerKey;
  label: string;
  description: string;
  group: "Ball" | "Players" | "Events" | "Debug";
  queryToken: string;
};

export const VIEW_LAYER_DEFINITIONS: ViewLayerDefinition[] = [
  { key: "ballDot", label: "Ball", description: "3D ball dot", group: "Ball", queryToken: "ball" },
  { key: "ballTrail", label: "Ball trail", description: "Recent 3D ball path", group: "Ball", queryToken: "trail" },
  { key: "paddles", label: "Paddles", description: "Estimated paddle proxy faces", group: "Players", queryToken: "paddles" },
  { key: "paddleNormals", label: "Paddle normals", description: "Review-only paddle direction arrows", group: "Debug", queryToken: "paddle-normals" },
  { key: "playerSkeletons", label: "Skeletons", description: "Player joint skeletons", group: "Players", queryToken: "players" },
  { key: "playerSolidMeshes", label: "Solid meshes", description: "BODY solid mesh frames", group: "Players", queryToken: "mesh" },
  { key: "playerTrails", label: "Player trails", description: "Recent player floor paths", group: "Players", queryToken: "ptrails" },
  { key: "floorContactMarkers", label: "Floor/contact", description: "Player floor and contact discs", group: "Events", queryToken: "floor" },
  { key: "eventMarkers", label: "Events", description: "Contact and bounce markers", group: "Events", queryToken: "events" },
  { key: "handJointPoints", label: "Hand points", description: "Optional whole-body hand joint dots", group: "Debug", queryToken: "hands" },
  {
    key: "implausibleSkeletons",
    label: "Implausible skeletons",
    description: "Show skeleton frames flagged low-confidence by the plausibility gate",
    group: "Debug",
    queryToken: "implausible",
  },
  { key: "debugPointClouds", label: "Point clouds", description: "Mesh vertex debug points", group: "Debug", queryToken: "debug" },
];

const LAYER_BY_TOKEN = new Map<string, ViewLayerKey>(
  VIEW_LAYER_DEFINITIONS.flatMap((definition) => [
    [definition.queryToken, definition.key],
    [definition.key, definition.key],
  ]),
);

export type ViewLayers = Record<ViewLayerKey, boolean>;

export type FocusState = { kind: "ball" } | { kind: "player"; playerId: number } | null;

export type ViewState = {
  layers: ViewLayers;
  focus: FocusState;
  selectedPlayerId: number | null;
};

export const DEFAULT_VIEW_LAYERS: ViewLayers = {
  ballDot: true,
  ballTrail: true,
  paddles: true,
  paddleNormals: false,
  playerSkeletons: true,
  playerSolidMeshes: true,
  playerTrails: false,
  floorContactMarkers: false,
  eventMarkers: true,
  handJointPoints: false,
  implausibleSkeletons: false,
  debugPointClouds: false,
};

export const DEFAULT_VIEW_STATE: ViewState = {
  layers: { ...DEFAULT_VIEW_LAYERS },
  focus: null,
  selectedPlayerId: null,
};

export type ViewPreset = "default" | "ballFocus" | "playerFocus";

export function toggleViewLayer(state: ViewState, layer: ViewLayerKey): ViewState {
  return {
    ...state,
    layers: {
      ...state.layers,
      [layer]: !state.layers[layer],
    },
  };
}

export function setViewLayer(state: ViewState, layer: ViewLayerKey, enabled: boolean): ViewState {
  return {
    ...state,
    layers: {
      ...state.layers,
      [layer]: enabled,
    },
  };
}

export function clearFocus(state: ViewState): ViewState {
  return { ...state, focus: null, selectedPlayerId: null };
}

export function applyViewPreset(
  state: ViewState,
  preset: ViewPreset,
  options: { playerId?: number } = {},
): ViewState {
  if (preset === "default") {
    return {
      layers: { ...DEFAULT_VIEW_LAYERS },
      focus: null,
      selectedPlayerId: null,
    };
  }
  if (preset === "ballFocus") {
    return {
      ...state,
      layers: {
        ...state.layers,
        ballDot: true,
        ballTrail: true,
        paddles: true,
        playerSkeletons: true,
        playerSolidMeshes: true,
        handJointPoints: false,
        implausibleSkeletons: false,
        debugPointClouds: false,
      },
      focus: { kind: "ball" },
      selectedPlayerId: null,
    };
  }

  const playerId = options.playerId;
  if (typeof playerId !== "number" || !Number.isFinite(playerId)) return state;
  return {
    ...state,
    layers: {
      ...state.layers,
      playerSkeletons: true,
      playerSolidMeshes: true,
      handJointPoints: false,
      implausibleSkeletons: false,
      debugPointClouds: false,
    },
    focus: { kind: "player", playerId },
    selectedPlayerId: playerId,
  };
}

export function parseViewStateFromSearch(search: string): ViewState {
  const params = new URLSearchParams(search.startsWith("?") ? search.slice(1) : search);
  const layers = params.has("layers") ? layersFromQuery(params.get("layers") ?? "") : { ...DEFAULT_VIEW_LAYERS };
  const focus = focusFromQuery(params.get("focus"), params.get("player"));
  return {
    layers,
    focus,
    selectedPlayerId: focus?.kind === "player" ? focus.playerId : null,
  };
}

export function viewStateToSearch(search: string, state: ViewState): string {
  const params = new URLSearchParams(search.startsWith("?") ? search.slice(1) : search);
  const layerTokens = VIEW_LAYER_DEFINITIONS.filter((definition) => state.layers[definition.key]).map(
    (definition) => definition.queryToken,
  );
  if (layersEqual(state.layers, DEFAULT_VIEW_LAYERS)) {
    params.delete("layers");
  } else {
    params.set("layers", layerTokens.join(","));
  }

  if (state.focus?.kind === "ball") {
    params.set("focus", "ball");
    params.delete("player");
  } else if (state.focus?.kind === "player") {
    params.set("focus", `player:${state.focus.playerId}`);
    params.set("player", String(state.focus.playerId));
  } else {
    params.delete("focus");
    params.delete("player");
  }

  const value = params.toString();
  return value ? `?${value}` : "";
}

function layersFromQuery(value: string): ViewLayers {
  const layers = Object.fromEntries(
    VIEW_LAYER_DEFINITIONS.map((definition) => [definition.key, false]),
  ) as ViewLayers;
  for (const rawToken of value.split(",")) {
    const token = rawToken.trim();
    const layer = LAYER_BY_TOKEN.get(token);
    if (layer) layers[layer] = true;
  }
  return layers;
}

function focusFromQuery(focus: string | null, player: string | null): FocusState {
  if (!focus) return null;
  if (focus === "ball") return { kind: "ball" };
  if (focus === "player") {
    const playerId = readPlayerId(player);
    return playerId === null ? null : { kind: "player", playerId };
  }
  if (focus.startsWith("player:")) {
    const playerId = readPlayerId(focus.slice("player:".length));
    return playerId === null ? null : { kind: "player", playerId };
  }
  return null;
}

function readPlayerId(value: string | null): number | null {
  if (!value) return null;
  const parsed = Number(value);
  return Number.isInteger(parsed) ? parsed : null;
}

function layersEqual(left: ViewLayers, right: ViewLayers): boolean {
  return VIEW_LAYER_DEFINITIONS.every((definition) => left[definition.key] === right[definition.key]);
}

export type EntityFocusTarget = { kind: "ball" } | { kind: "player"; playerId: number };

export type EntityFocusStyle = {
  dimmed: boolean;
  highlighted: boolean;
};

export function entityFocusStyle(state: ViewState, target: EntityFocusTarget): EntityFocusStyle {
  if (!state.focus) return { dimmed: false, highlighted: false };
  if (state.focus.kind === "ball") {
    return target.kind === "ball" ? { dimmed: false, highlighted: true } : { dimmed: true, highlighted: false };
  }
  if (target.kind === "player" && target.playerId === state.focus.playerId) {
    return { dimmed: false, highlighted: true };
  }
  return { dimmed: true, highlighted: false };
}

export type SceneLayerStatus = {
  visible: boolean;
  objectCount: number;
};

export type SceneLayerSnapshot = Record<ViewLayerKey | "courtNet", SceneLayerStatus>;

export function sceneLayerSnapshotForTime({
  world,
  bodyMesh,
  contactWindows,
  ballArcEventsSelected = null,
  ballSamples,
  currentTime,
  viewState,
}: {
  world: VirtualWorld;
  bodyMesh: BodyMesh | null;
  contactWindows: ContactWindows | null;
  ballArcEventsSelected?: BallArcEventsSelected | null;
  ballSamples?: BallTrailSample[] | null;
  currentTime: number;
  viewState: ViewState;
}): SceneLayerSnapshot {
  const resolvedBallSamples = ballSamples ?? samplesFromVirtualWorld(world);
  const ballDotCount = ballHudStateForTime(resolvedBallSamples, currentTime).sample ? 1 : 0;
  const ballTrailCount = buildBallTrail(resolvedBallSamples, currentTime, { windowSeconds: 0.75 }).segments.length;
  const paddleCount = activePaddleFramesForTime(world, currentTime).length;
  const skeletonCount = world.players.filter((player) => playerHasSkeletonFrame(player, currentTime, false)).length;
  const implausibleSkeletonCount = world.players.filter((player) => playerHasSkeletonFrame(player, currentTime, true, true)).length;
  const handPointCount = world.players.filter((player) => playerHasHandJointFrame(player, currentTime, world.joint_names)).length;
  const solidMeshCount = solidBodyMeshFramesForTime(bodyMesh, contactWindows, currentTime, world).length;
  const playerTrailCount = world.players.filter((player) => playerTrailPointsForTime(player, currentTime, 1.2).length >= 2).length;
  const floorMarkerCount = world.players.filter((player) => floorWorldForFrame(frameForTime(player, currentTime)) !== null).length;
  const eventMarkerCount = eventMarkersForTime(world, contactWindows, currentTime).length;
  const pointCloudCount = world.players.filter((player) => (frameForTime(player, currentTime)?.mesh_vertices_world.length ?? 0) > 0).length;

  return {
    courtNet: { visible: true, objectCount: 1 },
    ballDot: layerStatus(viewState.layers.ballDot, ballDotCount),
    ballTrail: layerStatus(viewState.layers.ballTrail, ballTrailCount),
    paddles: layerStatus(viewState.layers.paddles, paddleCount),
    paddleNormals: layerStatus(viewState.layers.paddleNormals, paddleCount),
    playerSkeletons: layerStatus(
      viewState.layers.playerSkeletons,
      skeletonCount + (viewState.layers.implausibleSkeletons ? implausibleSkeletonCount : 0),
    ),
    playerSolidMeshes: layerStatus(viewState.layers.playerSolidMeshes, solidMeshCount),
    playerTrails: layerStatus(viewState.layers.playerTrails, playerTrailCount),
    floorContactMarkers: layerStatus(viewState.layers.floorContactMarkers, floorMarkerCount),
    eventMarkers: layerStatus(viewState.layers.eventMarkers, eventMarkerCount),
    handJointPoints: layerStatus(viewState.layers.handJointPoints, handPointCount),
    implausibleSkeletons: layerStatus(viewState.layers.implausibleSkeletons, implausibleSkeletonCount),
    debugPointClouds: layerStatus(viewState.layers.debugPointClouds, pointCloudCount),
  };
}

function layerStatus(enabled: boolean, count: number): SceneLayerStatus {
  return { visible: enabled && count > 0, objectCount: enabled ? count : 0 };
}

function playerHasSkeletonFrame(player: VirtualWorldPlayer, currentTime: number, includeImplausible: boolean, onlyImplausible = false): boolean {
  const frame = frameForTime(player, currentTime);
  if (!frame) return false;
  if (onlyImplausible && frame.skeleton_implausible !== true) return false;
  if (!includeImplausible && frame.skeleton_implausible === true) return false;
  return frame.joints_world.length > 0 || floorWorldForFrame(frame) !== null;
}

function playerHasHandJointFrame(player: VirtualWorldPlayer, currentTime: number, jointNames?: string[]): boolean {
  const frame = frameForTime(player, currentTime);
  if (!frame || !jointNames?.length) return false;
  return jointNames.some((name, index) => isHandJointName(name) && frame.joints_world[index]);
}

function isHandJointName(name: string): boolean {
  return name.includes("hand") || name.includes("thumb") || name.includes("index") || name.includes("middle") || name.includes("ring") || name.includes("pinky");
}

export type WorldEventMarker = {
  kind: ContactWindowEvent["type"];
  playerId: number | null;
  position: Vec3;
  confidence: number;
};

export function eventMarkersForTime(
  world: VirtualWorld,
  contactWindows: ContactWindows | null,
  currentTime: number,
): WorldEventMarker[] {
  if (!contactWindows) return [];
  return contactWindows.events
    .filter((event) => event.window.t0 <= currentTime && currentTime <= event.window.t1)
    .map((event) => markerFromContactEvent(world, event, currentTime))
    .filter((marker): marker is WorldEventMarker => marker !== null);
}

function markerFromContactEvent(
  world: VirtualWorld,
  event: ContactWindowEvent,
  currentTime: number,
): WorldEventMarker | null {
  const eventPlayer = event.player_id === null ? undefined : world.players.find((player) => player.id === event.player_id);
  const playerFloor = eventPlayer ? floorWorldForFrame(frameForTime(eventPlayer, currentTime)) : null;
  const ballPoint = ballRenderInfoForTime(world, event.t).frame?.world_xyz ?? null;
  const position = playerFloor ?? ballPoint;
  if (!position) return null;
  return {
    kind: event.type,
    playerId: event.player_id,
    position,
    confidence: event.confidence,
  };
}

function floorWorldForFrame(frame: VirtualWorldFrame | undefined): Vec3 | null {
  return frame?.floor_world_xyz ?? (frame?.track_world_xy ? ([frame.track_world_xy[0], frame.track_world_xy[1], 0] as Vec3) : null);
}
