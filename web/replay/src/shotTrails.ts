import type { Vec2, Vec3, VirtualWorld } from "./viewerData";
import { readArcSolverStatus, readKillReasons, TRUSTED_BALL_ARC_SOLVER_STATUSES } from "./components/modules/ballTrail";

export type ShotQualityBand = "high" | "mid" | "low";

export type ShotTrailFilters = {
  playerId?: number | null;
  shotType?: string | "all" | null;
  outcome?: string | "all" | null;
  quality?: ShotQualityBand | "all" | null;
};

export type ShotOutcome = {
  call: string;
  faults: string[];
  let_candidate: boolean;
  out?: {
    direction?: string | null;
    side?: string | null;
    landed?: boolean | null;
  } | null;
};

export type ShotLandingEllipse = {
  angle_deg: number;
  center_xy: Vec2;
  semi_major_m: number;
  semi_minor_m: number;
  source: string | null;
};

export type ShotLanding = {
  source: string | null;
  world_xyz: Vec3 | null;
  zone: string | null;
  uncertainty_ellipse: ShotLandingEllipse | null;
  line_call: {
    call: string;
    direction: string | null;
    side: string | null;
    boundary_margin_m: number | null;
    uncertainty_radius_m: number | null;
    within_uncertainty: boolean | null;
  } | null;
};

export type ShotRecord = {
  shot_id: string;
  event_anchor_id: string | null;
  segment_id: number | string;
  player_id: number | null;
  shot_type: string | null;
  shot_type_abstained: boolean;
  outcome: ShotOutcome;
  confidence: number;
  speed_mph: number | null;
  t: number;
  frame: number | null;
  peak_height_m: number | null;
  landing: ShotLanding | null;
};

export type RacketsportShots = {
  schema_version: 1;
  artifact_type: "racketsport_shots";
  clip_id: string;
  policy: {
    internal_val_only: boolean;
    not_for_detection_metrics: boolean;
    not_ground_truth: boolean;
  };
  shots: ShotRecord[];
};

export type BallArcSolvedFrame = {
  t: number;
  visible: boolean;
  conf: number;
  sigma_m: number | null;
  world_xyz: Vec3 | null;
  segment_id: number | string | null;
};

export type BallArcSolved = {
  schema_version: 1 | 2;
  artifact_type: "racketsport_ball_track_arc_solved";
  clip_id: string | null;
  status: string;
  trusted: boolean;
  killReasons: string[];
  frames: BallArcSolvedFrame[];
};

export type ShotTrailSegment = {
  from: Vec3;
  to: Vec3;
};

export type ShotTrailGroup = {
  shot: ShotRecord;
  segment_id: number | string;
  points: Vec3[];
  samples: BallArcSolvedFrame[];
  segments: ShotTrailSegment[];
  color: string;
  quality: ShotQualityBand;
};

export function parseShots(input: unknown): RacketsportShots {
  const value = parseMaybeJson(input);
  assertRecord(value, "shots");
  if (value.schema_version !== 1) throw new Error("shots.schema_version must be 1");
  if (value.artifact_type !== "racketsport_shots") throw new Error("shots.artifact_type must be racketsport_shots");
  const policy = readPolicy(value.policy, "shots.policy");
  return {
    schema_version: 1,
    artifact_type: "racketsport_shots",
    clip_id: readString(value.clip_id, "shots.clip_id"),
    policy,
    shots: readArray(value.shots, "shots.shots").map(readShot),
  };
}

export function parseBallArcSolved(input: unknown): BallArcSolved {
  const value = parseMaybeJson(input);
  assertRecord(value, "ball_track_arc_solved");
  if (value.schema_version !== 1 && value.schema_version !== 2) throw new Error("ball_track_arc_solved.schema_version must be 1 or 2");
  if (value.artifact_type !== "racketsport_ball_track_arc_solved") {
    throw new Error("ball_track_arc_solved.artifact_type must be racketsport_ball_track_arc_solved");
  }
  const status = readArcSolverStatus(value.status, "ball_track_arc_solved.status");
  const trusted = TRUSTED_BALL_ARC_SOLVER_STATUSES.has(status);
  return {
    schema_version: value.schema_version,
    artifact_type: "racketsport_ball_track_arc_solved",
    clip_id:
      value.clip_id === null || value.clip_id === undefined
        ? null
        : readString(value.clip_id, "ball_track_arc_solved.clip_id"),
    status,
    trusted,
    killReasons: readKillReasons(value.kill_reasons, "ball_track_arc_solved.kill_reasons"),
    // A self-killed or unrecognized-status solve must not let any frame be
    // eligible for measured/anchored trail styling downstream (fail-closed
    // trusted-status allowlist; see ballTrail.ts TRUSTED_BALL_ARC_SOLVER_STATUSES).
    frames: trusted ? readArray(value.frames, "ball_track_arc_solved.frames").map(readBallArcSolvedFrame) : [],
  };
}

export function qualityBandForShot(shot: Pick<ShotRecord, "confidence">): ShotQualityBand {
  if (shot.confidence >= 0.7) return "high";
  if (shot.confidence >= 0.45) return "mid";
  return "low";
}

export function shotOutcomeColor(shot: Pick<ShotRecord, "confidence" | "outcome">): string {
  const call = shot.outcome.call.toLowerCase();
  if (["out", "net", "excess_bounce", "kitchen", "paddle_hit_net"].includes(call)) return "#df3f31";
  const quality = qualityBandForShot(shot);
  if (call === "in" && quality === "high") return "#16a05d";
  if (call === "in" && quality === "mid") return "#7fc96f";
  return "#d49b24";
}

export function filterShots(shots: ShotRecord[], filters: ShotTrailFilters): ShotRecord[] {
  return shots.filter((shot) => {
    if (filters.playerId !== null && filters.playerId !== undefined && shot.player_id !== filters.playerId) return false;
    if (filters.shotType && filters.shotType !== "all") {
      const shotType = shot.shot_type ?? "unknown";
      if (shotType !== filters.shotType) return false;
    }
    if (filters.outcome && filters.outcome !== "all" && shot.outcome.call !== filters.outcome) return false;
    if (filters.quality && filters.quality !== "all" && qualityBandForShot(shot) !== filters.quality) return false;
    return true;
  });
}

export function buildShotTrailGroups(
  shots: ShotRecord[],
  arcSolved: BallArcSolved | null | undefined,
  world?: VirtualWorld | null,
): ShotTrailGroup[] {
  const framesBySegment = arcSolved ? arcFramesBySegment(arcSolved.frames) : worldFramesBySegment(world);
  return shots.map((shot) => {
    const samples = framesBySegment.get(segmentKey(shot.segment_id)) ?? [];
    const points = samples
      .slice()
      .sort((left, right) => left.t - right.t)
      .map((sample) => sample.world_xyz)
      .filter((point): point is Vec3 => point !== null);
    return {
      shot,
      segment_id: shot.segment_id,
      points,
      samples,
      segments: points.slice(1).map((point, index) => ({ from: points[index], to: point })),
      color: shotOutcomeColor(shot),
      quality: qualityBandForShot(shot),
    };
  });
}

export function shotTypeLabel(shot: ShotRecord): string {
  return shot.shot_type ?? "unknown";
}

function arcFramesBySegment(frames: BallArcSolvedFrame[]): Map<string, BallArcSolvedFrame[]> {
  const grouped = new Map<string, BallArcSolvedFrame[]>();
  for (const frame of frames) {
    if (frame.segment_id === null || frame.world_xyz === null) continue;
    appendSegmentFrame(grouped, frame.segment_id, frame);
  }
  return grouped;
}

function worldFramesBySegment(world: VirtualWorld | null | undefined): Map<string, BallArcSolvedFrame[]> {
  const grouped = new Map<string, BallArcSolvedFrame[]>();
  if (!world) return grouped;
  for (const frame of world.ball.frames) {
    const segmentId = frame.arc_segment_id ?? null;
    if (segmentId === null || segmentId === undefined || !frame.world_xyz) continue;
    appendSegmentFrame(grouped, segmentId, {
      t: frame.t,
      visible: frame.visible,
      conf: frame.conf,
      sigma_m: frame.confidence_provenance?.predicted_sigma_m ?? frame.physics_fill?.uncertainty_m ?? null,
      world_xyz: frame.world_xyz,
      segment_id: segmentId,
    });
  }
  return grouped;
}

function appendSegmentFrame(grouped: Map<string, BallArcSolvedFrame[]>, segmentId: number | string, frame: BallArcSolvedFrame) {
  const key = segmentKey(segmentId);
  const frames = grouped.get(key);
  if (frames) {
    frames.push(frame);
  } else {
    grouped.set(key, [frame]);
  }
}

function readShot(input: unknown, index: number): ShotRecord {
  const path = `shots.shots[${index}]`;
  assertRecord(input, path);
  return {
    shot_id: readString(input.shot_id, `${path}.shot_id`),
    event_anchor_id:
      input.event_anchor_id === null || input.event_anchor_id === undefined
        ? null
        : readString(input.event_anchor_id, `${path}.event_anchor_id`),
    segment_id: readSegmentId(input.segment_id, `${path}.segment_id`),
    player_id: input.player_id === null || input.player_id === undefined ? null : readInteger(input.player_id, `${path}.player_id`),
    shot_type: input.shot_type === null || input.shot_type === undefined ? null : readString(input.shot_type, `${path}.shot_type`),
    shot_type_abstained:
      input.shot_type_abstained === null || input.shot_type_abstained === undefined
        ? false
        : readBoolean(input.shot_type_abstained, `${path}.shot_type_abstained`),
    outcome: readOutcome(input.outcome, `${path}.outcome`),
    confidence: readNumber(input.confidence, `${path}.confidence`),
    speed_mph: input.speed_mph === null || input.speed_mph === undefined ? null : readNumber(input.speed_mph, `${path}.speed_mph`),
    t: readNumber(input.t, `${path}.t`),
    frame: input.frame === null || input.frame === undefined ? null : readInteger(input.frame, `${path}.frame`),
    peak_height_m:
      input.peak_height_m === null || input.peak_height_m === undefined ? null : readNumber(input.peak_height_m, `${path}.peak_height_m`),
    landing: input.landing === null || input.landing === undefined ? null : readLanding(input.landing, `${path}.landing`),
  };
}

function readOutcome(input: unknown, path: string): ShotOutcome {
  assertRecord(input, path);
  return {
    call: readString(input.call, `${path}.call`),
    faults:
      input.faults === null || input.faults === undefined
        ? []
        : readArray(input.faults, `${path}.faults`).map((fault, index) => readString(fault, `${path}.faults[${index}]`)),
    let_candidate:
      input.let_candidate === null || input.let_candidate === undefined ? false : readBoolean(input.let_candidate, `${path}.let_candidate`),
    out: input.out === null || input.out === undefined ? null : readOut(input.out, `${path}.out`),
  };
}

function readOut(input: unknown, path: string): NonNullable<ShotOutcome["out"]> {
  assertRecord(input, path);
  return {
    direction: input.direction === null || input.direction === undefined ? null : readString(input.direction, `${path}.direction`),
    side: input.side === null || input.side === undefined ? null : readString(input.side, `${path}.side`),
    landed: input.landed === null || input.landed === undefined ? null : readBoolean(input.landed, `${path}.landed`),
  };
}

function readLanding(input: unknown, path: string): ShotLanding {
  assertRecord(input, path);
  return {
    source: input.source === null || input.source === undefined ? null : readString(input.source, `${path}.source`),
    world_xyz: input.world_xyz === null || input.world_xyz === undefined ? null : readVec3(input.world_xyz, `${path}.world_xyz`),
    zone: input.zone === null || input.zone === undefined ? null : readString(input.zone, `${path}.zone`),
    uncertainty_ellipse:
      input.uncertainty_ellipse === null || input.uncertainty_ellipse === undefined
        ? null
        : readLandingEllipse(input.uncertainty_ellipse, `${path}.uncertainty_ellipse`),
    line_call: input.line_call === null || input.line_call === undefined ? null : readLineCall(input.line_call, `${path}.line_call`),
  };
}

function readLandingEllipse(input: unknown, path: string): ShotLandingEllipse {
  assertRecord(input, path);
  return {
    angle_deg: readNumber(input.angle_deg, `${path}.angle_deg`),
    center_xy: readVec2(input.center_xy, `${path}.center_xy`),
    semi_major_m: readNumber(input.semi_major_m, `${path}.semi_major_m`),
    semi_minor_m: readNumber(input.semi_minor_m, `${path}.semi_minor_m`),
    source: input.source === null || input.source === undefined ? null : readString(input.source, `${path}.source`),
  };
}

function readLineCall(input: unknown, path: string): NonNullable<ShotLanding["line_call"]> {
  assertRecord(input, path);
  return {
    call: readString(input.call, `${path}.call`),
    direction: input.direction === null || input.direction === undefined ? null : readString(input.direction, `${path}.direction`),
    side: input.side === null || input.side === undefined ? null : readString(input.side, `${path}.side`),
    boundary_margin_m:
      input.boundary_margin_m === null || input.boundary_margin_m === undefined
        ? null
        : readNumber(input.boundary_margin_m, `${path}.boundary_margin_m`),
    uncertainty_radius_m:
      input.uncertainty_radius_m === null || input.uncertainty_radius_m === undefined
        ? null
        : readNumber(input.uncertainty_radius_m, `${path}.uncertainty_radius_m`),
    within_uncertainty:
      input.within_uncertainty === null || input.within_uncertainty === undefined
        ? null
        : readBoolean(input.within_uncertainty, `${path}.within_uncertainty`),
  };
}

function readBallArcSolvedFrame(input: unknown, index: number): BallArcSolvedFrame {
  const path = `ball_track_arc_solved.frames[${index}]`;
  assertRecord(input, path);
  return {
    t: readNumber(input.t, `${path}.t`),
    visible: input.visible === null || input.visible === undefined ? true : readBoolean(input.visible, `${path}.visible`),
    conf: input.conf === null || input.conf === undefined ? 0 : readNumber(input.conf, `${path}.conf`),
    sigma_m: input.sigma_m === null || input.sigma_m === undefined ? null : readNumber(input.sigma_m, `${path}.sigma_m`),
    world_xyz: input.world_xyz === null || input.world_xyz === undefined ? null : readVec3(input.world_xyz, `${path}.world_xyz`),
    segment_id: readOptionalArcSegmentId(input, path),
  };
}

function readOptionalArcSegmentId(input: Record<string, unknown>, path: string): number | string | null {
  const direct = input.arc_segment_id ?? input.segment_id;
  if (direct !== null && direct !== undefined) return readSegmentId(direct, `${path}.segment_id`);
  const arcSolver = input.arc_solver;
  if (arcSolver && typeof arcSolver === "object" && !Array.isArray(arcSolver)) {
    const value = (arcSolver as Record<string, unknown>).segment_id;
    if (value !== null && value !== undefined) return readSegmentId(value, `${path}.arc_solver.segment_id`);
  }
  return null;
}

function readPolicy(input: unknown, path: string): RacketsportShots["policy"] {
  assertRecord(input, path);
  return {
    internal_val_only: readBoolean(input.internal_val_only, `${path}.internal_val_only`),
    not_for_detection_metrics: readBoolean(input.not_for_detection_metrics, `${path}.not_for_detection_metrics`),
    not_ground_truth: readBoolean(input.not_ground_truth, `${path}.not_ground_truth`),
  };
}

function segmentKey(value: number | string): string {
  return String(value);
}

function parseMaybeJson(input: unknown): unknown {
  if (typeof input === "string") return JSON.parse(input) as unknown;
  return input;
}

function assertRecord(value: unknown, path: string): asserts value is Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${path} must be an object`);
}

function readArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${path} must be an array`);
  return value;
}

function readString(value: unknown, path: string): string {
  if (typeof value !== "string" || value.trim() === "") throw new Error(`${path} must be a non-empty string`);
  return value;
}

function readBoolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") throw new Error(`${path} must be boolean`);
  return value;
}

function readNumber(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${path} must be a finite number`);
  return value;
}

function readInteger(value: unknown, path: string): number {
  const number = readNumber(value, path);
  if (!Number.isInteger(number)) throw new Error(`${path} must be an integer`);
  return number;
}

function readSegmentId(value: unknown, path: string): number | string {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error(`${path} must be finite`);
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") return value;
  throw new Error(`${path} must be a number or non-empty string`);
}

function readVec2(value: unknown, path: string): Vec2 {
  const array = readArray(value, path);
  if (array.length !== 2) throw new Error(`${path} must contain 2 numbers`);
  return [readNumber(array[0], `${path}[0]`), readNumber(array[1], `${path}[1]`)];
}

function readVec3(value: unknown, path: string): Vec3 {
  const array = readArray(value, path);
  if (array.length !== 3) throw new Error(`${path} must contain 3 numbers`);
  return [readNumber(array[0], `${path}[0]`), readNumber(array[1], `${path}[1]`), readNumber(array[2], `${path}[2]`)];
}
