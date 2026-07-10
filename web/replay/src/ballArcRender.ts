import { TRUSTED_BALL_ARC_SOLVER_STATUSES, type BallBand, type BallTrailSample, type Vec3 } from "./components/modules/ballTrail";
import { ENTITY_HOLD_TOLERANCES_SECONDS, resolveTimeSample } from "./viewerData";
import type { Vec2 } from "./viewerData";

export type ReplayViewMode = "world" | "courtmap";

export type BallArcRenderPoint = {
  world_xyz: Vec3;
  court_xy: Vec2;
};

export type BallArcRenderShot = {
  start: BallArcRenderPoint;
  peak: BallArcRenderPoint;
  end: BallArcRenderPoint;
  speed_mps: number;
  speed_mph: number;
  height_over_net_m: number | null;
  distance_m: number;
  path_distance_m: number;
};

export type BallArcRenderSegment = {
  segment_id: number | string;
  t0: number;
  t1: number;
  frame_start: number;
  frame_end: number;
  anchor_types: string[];
  anchor_frames: number[];
  confidence: number;
  flight_sanity_verdict: string;
  bridge: boolean;
  shot: BallArcRenderShot;
};

export type BallArcRenderBridge = {
  bridge_id: string;
  t0: number;
  t1: number;
  confidence: number;
  reason: string;
};

export type BallArcRender = {
  schema_version: 1;
  artifact_type: "racketsport_ball_arc_render";
  clip_id: string;
  solver_status: string;
  trusted: boolean;
  render_only: boolean;
  not_for_detection_metrics: boolean;
  segments: BallArcRenderSegment[];
  bridges: BallArcRenderBridge[];
  samples: BallTrailSample[];
  summary: Record<string, unknown>;
};

export type CourtMapShot = {
  segmentId: number | string;
  t0: number;
  t1: number;
  start: Vec2;
  peak: Vec2;
  end: Vec2;
  confidence: number;
  active: boolean;
  speedMph: number;
  heightOverNetM: number | null;
};

export function parseBallArcRender(input: unknown): BallArcRender {
  const value = parseMaybeJson(input);
  assertRecord(value, "ball_arc_render");
  if (value.schema_version !== 1) throw new Error("ball_arc_render.schema_version must be 1");
  if (value.artifact_type !== "racketsport_ball_arc_render") {
    throw new Error("ball_arc_render.artifact_type must be racketsport_ball_arc_render");
  }
  const solverStatus = readString(value.solver_status, "ball_arc_render.solver_status");
  const trusted = TRUSTED_BALL_ARC_SOLVER_STATUSES.has(solverStatus);
  return {
    schema_version: 1,
    artifact_type: "racketsport_ball_arc_render",
    clip_id: readString(value.clip_id, "ball_arc_render.clip_id"),
    solver_status: solverStatus,
    trusted,
    render_only: readBoolean(value.render_only, "ball_arc_render.render_only"),
    not_for_detection_metrics: readBoolean(value.not_for_detection_metrics, "ball_arc_render.not_for_detection_metrics"),
    segments: trusted ? readArray(value.segments, "ball_arc_render.segments").map(readRenderSegment) : [],
    bridges: trusted ? readArray(value.bridges, "ball_arc_render.bridges").map(readRenderBridge) : [],
    samples: trusted ? readArray(value.samples, "ball_arc_render.samples").map(readRenderSample) : [],
    summary: readOptionalRecord(value.summary, "ball_arc_render.summary") ?? {},
  };
}

export function sampleBallArcRenderAtTime(samples: BallTrailSample[], currentTime: number, maxGapSeconds = 0.25): BallTrailSample | null {
  const renderable = samples
    .filter((sample): sample is BallTrailSample & { world_xyz: Vec3 } => Array.isArray(sample.world_xyz))
    .slice()
    .sort((left, right) => left.t - right.t);
  if (!renderable.length) return null;
  const exact = renderable.find((sample) => Math.abs(sample.t - currentTime) <= 1e-9);
  if (exact) return exact;
  let left: (BallTrailSample & { world_xyz: Vec3 }) | null = null;
  let right: (BallTrailSample & { world_xyz: Vec3 }) | null = null;
  for (const sample of renderable) {
    if (sample.t <= currentTime) left = sample;
    if (sample.t >= currentTime) {
      right = sample;
      break;
    }
  }
  if (!left || !right || right.t <= left.t || right.t - left.t > maxGapSeconds) {
    return resolveTimeSample(renderable, currentTime, ENTITY_HOLD_TOLERANCES_SECONDS.ball).sample ?? null;
  }
  const segmentChanged = left.segmentId != null && right.segmentId != null && left.segmentId !== right.segmentId;
  if (segmentChanged && left.bridge !== true && right.bridge !== true) {
    return resolveTimeSample(renderable, currentTime, ENTITY_HOLD_TOLERANCES_SECONDS.ball).sample ?? null;
  }
  const alpha = (currentTime - left.t) / (right.t - left.t);
  return {
    t: round6(currentTime),
    frame: null,
    band: leastCertainBand(left.band, right.band),
    conf: round6(Math.min(left.conf, right.conf)),
    visible: true,
    world_xyz: interpolateVec3(left.world_xyz, right.world_xyz, alpha),
    renderOnly: true,
    segmentId: left.segmentId === right.segmentId ? left.segmentId : null,
    source: "ball_arc_render_interpolated",
  };
}

export function buildCourtMapShots(render: BallArcRender | null | undefined, currentTime: number): CourtMapShot[] {
  return (render?.segments ?? []).map((segment) => ({
    segmentId: segment.segment_id,
    t0: segment.t0,
    t1: segment.t1,
    start: segment.shot.start.court_xy,
    peak: segment.shot.peak.court_xy,
    end: segment.shot.end.court_xy,
    confidence: segment.confidence,
    active: segment.t0 <= currentTime && currentTime <= segment.t1,
    speedMph: segment.shot.speed_mph,
    heightOverNetM: segment.shot.height_over_net_m,
  }));
}

export function svgCourtProjector({
  widthM,
  lengthM,
  paddingPx,
  widthPx,
  heightPx,
  xMin,
  xMax,
  yMin,
  yMax,
}: {
  widthM: number;
  lengthM: number;
  paddingPx: number;
  widthPx: number;
  heightPx: number;
  xMin?: number;
  xMax?: number;
  yMin?: number;
  yMax?: number;
}): (point: Vec2) => Vec2 {
  const innerWidth = widthPx - paddingPx * 2;
  const innerHeight = heightPx - paddingPx * 2;
  const minX = Number.isFinite(xMin) ? Number(xMin) : -widthM / 2;
  const maxX = Number.isFinite(xMax) ? Number(xMax) : widthM / 2;
  const minY = Number.isFinite(yMin) ? Number(yMin) : 0;
  const maxY = Number.isFinite(yMax) ? Number(yMax) : lengthM;
  const xSpan = maxX > minX ? maxX - minX : widthM;
  const ySpan = maxY > minY ? maxY - minY : lengthM;
  return ([x, y]) => [
    clampSvgCoordinate(Math.round(paddingPx + ((x - minX) / xSpan) * innerWidth), paddingPx, widthPx - paddingPx),
    clampSvgCoordinate(Math.round(heightPx - paddingPx - ((y - minY) / ySpan) * innerHeight), paddingPx, heightPx - paddingPx),
  ];
}

function clampSvgCoordinate(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

export function replayViewFromSearch(search: string): ReplayViewMode {
  const value = new URLSearchParams(search).get("view")?.trim().toLowerCase();
  return value === "courtmap" ? "courtmap" : "world";
}

function readRenderSegment(input: unknown, index: number): BallArcRenderSegment {
  const path = `ball_arc_render.segments[${index}]`;
  assertRecord(input, path);
  return {
    segment_id: readSegmentId(input.segment_id, `${path}.segment_id`),
    t0: readNumber(input.t0, `${path}.t0`),
    t1: readNumber(input.t1, `${path}.t1`),
    frame_start: readInteger(input.frame_start, `${path}.frame_start`),
    frame_end: readInteger(input.frame_end, `${path}.frame_end`),
    anchor_types: readArray(input.anchor_types, `${path}.anchor_types`).map((item, itemIndex) => readString(item, `${path}.anchor_types[${itemIndex}]`)),
    anchor_frames: readArray(input.anchor_frames, `${path}.anchor_frames`).map((item, itemIndex) => readInteger(item, `${path}.anchor_frames[${itemIndex}]`)),
    confidence: readNumber(input.confidence, `${path}.confidence`),
    flight_sanity_verdict: readString(input.flight_sanity_verdict, `${path}.flight_sanity_verdict`),
    bridge: readBoolean(input.bridge, `${path}.bridge`),
    shot: readShot(input.shot, `${path}.shot`),
  };
}

function readRenderBridge(input: unknown, index: number): BallArcRenderBridge {
  const path = `ball_arc_render.bridges[${index}]`;
  assertRecord(input, path);
  return {
    bridge_id: readString(input.bridge_id, `${path}.bridge_id`),
    t0: readNumber(input.t0, `${path}.t0`),
    t1: readNumber(input.t1, `${path}.t1`),
    confidence: readNumber(input.confidence, `${path}.confidence`),
    reason: readString(input.reason, `${path}.reason`),
  };
}

function readRenderSample(input: unknown, index: number): BallTrailSample {
  const path = `ball_arc_render.samples[${index}]`;
  assertRecord(input, path);
  return {
    t: readNumber(input.t, `${path}.t`),
    frame: input.frame === null || input.frame === undefined ? null : readInteger(input.frame, `${path}.frame`),
    band: readBand(input.band, `${path}.band`),
    conf: readNumber(input.confidence, `${path}.confidence`),
    visible: true,
    world_xyz: readVec3(input.world_xyz, `${path}.world_xyz`),
    renderOnly: readBoolean(input.render_only, `${path}.render_only`),
    segmentId: readSegmentId(input.segment_id, `${path}.segment_id`),
    bridge: readBoolean(input.bridge, `${path}.bridge`),
    source: readBoolean(input.bridge, `${path}.bridge`) ? "ball_arc_render_bridge" : "ball_arc_render",
  };
}

function readShot(input: unknown, path: string): BallArcRenderShot {
  assertRecord(input, path);
  return {
    start: readPoint(input.start, `${path}.start`),
    peak: readPoint(input.peak, `${path}.peak`),
    end: readPoint(input.end, `${path}.end`),
    speed_mps: readNumber(input.speed_mps, `${path}.speed_mps`),
    speed_mph: readNumber(input.speed_mph, `${path}.speed_mph`),
    height_over_net_m:
      input.height_over_net_m === null || input.height_over_net_m === undefined
        ? null
        : readNumber(input.height_over_net_m, `${path}.height_over_net_m`),
    distance_m: readNumber(input.distance_m, `${path}.distance_m`),
    path_distance_m: readNumber(input.path_distance_m, `${path}.path_distance_m`),
  };
}

function readPoint(input: unknown, path: string): BallArcRenderPoint {
  assertRecord(input, path);
  return {
    world_xyz: readVec3(input.world_xyz, `${path}.world_xyz`),
    court_xy: readVec2(input.court_xy, `${path}.court_xy`),
  };
}

function leastCertainBand(a: BallBand, b: BallBand): BallBand {
  const rank: Record<BallBand, number> = {
    anchored_measured: 0,
    arc_interpolated: 1,
    physics_predicted: 1.5,
    physics_predicted_low: 2.5,
    arc_extrapolated: 2,
    arc_weak: 3,
    unknown: 4,
    hidden: 5,
  };
  return rank[a] >= rank[b] ? a : b;
}

function interpolateVec3(left: Vec3, right: Vec3, alpha: number): Vec3 {
  return [
    round6(left[0] + (right[0] - left[0]) * alpha),
    round6(left[1] + (right[1] - left[1]) * alpha),
    round6(left[2] + (right[2] - left[2]) * alpha),
  ];
}

function round6(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function parseMaybeJson(input: unknown): unknown {
  return typeof input === "string" ? JSON.parse(input) : input;
}

function assertRecord(input: unknown, path: string): asserts input is Record<string, unknown> {
  if (!input || typeof input !== "object" || Array.isArray(input)) throw new Error(`${path} must be an object`);
}

function readOptionalRecord(input: unknown, path: string): Record<string, unknown> | null {
  if (input === null || input === undefined) return null;
  assertRecord(input, path);
  return input;
}

function readArray(input: unknown, path: string): unknown[] {
  if (!Array.isArray(input)) throw new Error(`${path} must be an array`);
  return input;
}

function readString(input: unknown, path: string): string {
  if (typeof input !== "string") throw new Error(`${path} must be a string`);
  return input;
}

function readBoolean(input: unknown, path: string): boolean {
  if (typeof input !== "boolean") throw new Error(`${path} must be a boolean`);
  return input;
}

function readNumber(input: unknown, path: string): number {
  if (typeof input !== "number" || !Number.isFinite(input)) throw new Error(`${path} must be a finite number`);
  return input;
}

function readInteger(input: unknown, path: string): number {
  const value = readNumber(input, path);
  if (!Number.isInteger(value)) throw new Error(`${path} must be an integer`);
  return value;
}

function readSegmentId(input: unknown, path: string): number | string {
  if (typeof input === "string" || (typeof input === "number" && Number.isFinite(input))) return input;
  throw new Error(`${path} must be a string or number`);
}

function readBand(input: unknown, path: string): BallBand {
  const value = readString(input, path);
  if (value === "anchored_measured" || value === "physics_predicted" || value === "physics_predicted_low" || value === "arc_interpolated" || value === "arc_extrapolated" || value === "arc_weak" || value === "hidden") {
    return value;
  }
  return "unknown";
}

function readVec2(input: unknown, path: string): Vec2 {
  const values = readArray(input, path);
  if (values.length !== 2) throw new Error(`${path} must have 2 numbers`);
  return [readNumber(values[0], `${path}[0]`), readNumber(values[1], `${path}[1]`)];
}

function readVec3(input: unknown, path: string): Vec3 {
  const values = readArray(input, path);
  if (values.length !== 3) throw new Error(`${path} must have 3 numbers`);
  return [readNumber(values[0], `${path}[0]`), readNumber(values[1], `${path}[1]`), readNumber(values[2], `${path}[2]`)];
}
