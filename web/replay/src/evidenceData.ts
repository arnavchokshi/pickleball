import type { Vec2, Vec3 } from "./viewerData";

export type CourtEvidencePoint = {
  semanticName: string;
  imageXY: Vec2;
  source: string;
};

export type CourtEvidence = {
  points: CourtEvidencePoint[];
  authorityState: string;
  measurementValid: boolean;
};

export type CourtCalibrationEvidence = {
  imageSize: [number, number];
  intrinsics: {
    fx: number;
    fy: number;
    cx: number;
    cy: number;
    dist: number[];
  };
  extrinsics: {
    R: [Vec3, Vec3, Vec3];
    t: Vec3;
  };
};

export type Sam3DKeypoint = {
  name: string;
  index: number;
  xyPx: Vec2;
  confidence: number;
};

export type Sam3DKeypointFrame = {
  frameIdx: number;
  t: number;
  keypoints: Sam3DKeypoint[];
};

export type Sam3DKeypointPlayer = {
  id: number;
  frames: Sam3DKeypointFrame[];
};

export type Sam3DKeypointEvidence = {
  source: string;
  players: Sam3DKeypointPlayer[];
};

const COURT_SEGMENT_NAMES: ReadonlyArray<readonly [string, string]> = [
  ["near_left_corner", "near_baseline_center"],
  ["near_baseline_center", "near_right_corner"],
  ["far_left_corner", "far_baseline_center"],
  ["far_baseline_center", "far_right_corner"],
  ["near_left_corner", "far_left_corner"],
  ["near_right_corner", "far_right_corner"],
  ["near_nvz_left", "near_nvz_center"],
  ["near_nvz_center", "near_nvz_right"],
  ["far_nvz_left", "far_nvz_center"],
  ["far_nvz_center", "far_nvz_right"],
  ["near_baseline_center", "near_nvz_center"],
  ["far_baseline_center", "far_nvz_center"],
];

export function parseCourtEvidence(input: unknown): CourtEvidence {
  const value = record(input, "court evidence");
  if (value.artifact_type !== "racketsport_court_lock_visualization_adapter") {
    throw new Error("court evidence artifact_type is not a court lock visualization adapter");
  }
  const trust = record(value.trust, "court evidence.trust");
  const points = array(value.named_floor_correspondences, "court evidence.named_floor_correspondences").map((entry, index) => {
    const point = record(entry, `court evidence point ${index}`);
    return {
      semanticName: stringValue(point.semantic_name, `court evidence point ${index}.semantic_name`),
      imageXY: vec2(point.raw_image_xy ?? point.image_xy, `court evidence point ${index}.image_xy`),
      source: stringValue(point.source, `court evidence point ${index}.source`),
    };
  });
  return {
    points,
    authorityState: typeof trust.authority_state === "string" ? trust.authority_state : "review_only",
    measurementValid: trust.measurement_valid === true,
  };
}

export function parseCourtCalibrationEvidence(input: unknown): CourtCalibrationEvidence {
  const value = record(input, "court calibration");
  const intrinsics = record(value.intrinsics, "court calibration.intrinsics");
  const extrinsics = record(value.extrinsics, "court calibration.extrinsics");
  const imageSize = vec2(value.image_size, "court calibration.image_size");
  const rows = array(extrinsics.R, "court calibration.extrinsics.R").map((row, index) =>
    vec3(row, `court calibration.extrinsics.R[${index}]`),
  );
  if (rows.length !== 3) throw new Error("court calibration.extrinsics.R must contain three rows");
  const dist = Array.isArray(intrinsics.dist)
    ? intrinsics.dist.map((item, index) => finiteNumber(item, `court calibration.intrinsics.dist[${index}]`))
    : [];
  return {
    imageSize: [imageSize[0], imageSize[1]],
    intrinsics: {
      fx: finiteNumber(intrinsics.fx, "court calibration.intrinsics.fx"),
      fy: finiteNumber(intrinsics.fy, "court calibration.intrinsics.fy"),
      cx: finiteNumber(intrinsics.cx, "court calibration.intrinsics.cx"),
      cy: finiteNumber(intrinsics.cy, "court calibration.intrinsics.cy"),
      dist,
    },
    extrinsics: {
      R: [rows[0], rows[1], rows[2]],
      t: vec3(extrinsics.t, "court calibration.extrinsics.t"),
    },
  };
}

export function parseSam3DKeypointEvidence(input: unknown): Sam3DKeypointEvidence {
  const value = record(input, "SAM-3D keypoint evidence");
  if (value.artifact_type !== "racketsport_sam3d_keypoints_2d") {
    throw new Error("SAM-3D keypoint evidence artifact_type is invalid");
  }
  return {
    source: stringValue(value.source, "SAM-3D keypoint evidence.source"),
    players: array(value.players, "SAM-3D keypoint evidence.players").map((entry, playerIndex) => {
      const player = record(entry, `SAM-3D player ${playerIndex}`);
      return {
        id: integer(player.id, `SAM-3D player ${playerIndex}.id`),
        frames: array(player.frames, `SAM-3D player ${playerIndex}.frames`).map((frameEntry, frameIndex) => {
          const frame = record(frameEntry, `SAM-3D player ${playerIndex} frame ${frameIndex}`);
          return {
            frameIdx: integer(frame.frame_idx, `SAM-3D player ${playerIndex} frame ${frameIndex}.frame_idx`),
            t: finiteNumber(frame.t, `SAM-3D player ${playerIndex} frame ${frameIndex}.t`),
            keypoints: array(frame.keypoints, `SAM-3D player ${playerIndex} frame ${frameIndex}.keypoints`).map((item, keypointIndex) => {
              const point = record(item, `SAM-3D keypoint ${keypointIndex}`);
              return {
                name: stringValue(point.name, `SAM-3D keypoint ${keypointIndex}.name`),
                index: integer(point.index, `SAM-3D keypoint ${keypointIndex}.index`),
                xyPx: vec2(point.xy_px, `SAM-3D keypoint ${keypointIndex}.xy_px`),
                confidence: finiteNumber(point.conf, `SAM-3D keypoint ${keypointIndex}.conf`),
              };
            }),
          };
        }).sort((left, right) => left.t - right.t),
      };
    }),
  };
}

export function courtEvidenceSegments(evidence: CourtEvidence): Array<[CourtEvidencePoint, CourtEvidencePoint]> {
  const byName = new Map(evidence.points.map((point) => [point.semanticName, point]));
  return COURT_SEGMENT_NAMES.flatMap(([leftName, rightName]) => {
    const left = byName.get(leftName);
    const right = byName.get(rightName);
    return left && right ? [[left, right] as [CourtEvidencePoint, CourtEvidencePoint]] : [];
  });
}

export function sam3DKeypointFramesForTime(
  evidence: Sam3DKeypointEvidence | null,
  timeSeconds: number,
  fps: number,
): Array<{ playerId: number; frame: Sam3DKeypointFrame }> {
  if (!evidence) return [];
  const tolerance = 0.55 / Math.max(1, fps);
  const active: Array<{ playerId: number; frame: Sam3DKeypointFrame }> = [];
  for (const player of evidence.players) {
    const frame = nearestFrame(player.frames, timeSeconds);
    if (frame && Math.abs(frame.t - timeSeconds) <= tolerance) active.push({ playerId: player.id, frame });
  }
  return active;
}

export function projectCourtWorldPoint(point: Vec3, calibration: CourtCalibrationEvidence): Vec2 | null {
  const { R, t } = calibration.extrinsics;
  const cameraX = R[0][0] * point[0] + R[0][1] * point[1] + R[0][2] * point[2] + t[0];
  const cameraY = R[1][0] * point[0] + R[1][1] * point[1] + R[1][2] * point[2] + t[1];
  const cameraZ = R[2][0] * point[0] + R[2][1] * point[1] + R[2][2] * point[2] + t[2];
  if (!Number.isFinite(cameraZ) || cameraZ <= 1e-6) return null;
  let normalizedX = cameraX / cameraZ;
  let normalizedY = cameraY / cameraZ;
  const [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0] = calibration.intrinsics.dist;
  if (k1 || k2 || p1 || p2 || k3) {
    const r2 = normalizedX * normalizedX + normalizedY * normalizedY;
    const radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    const distortedX = normalizedX * radial + 2 * p1 * normalizedX * normalizedY + p2 * (r2 + 2 * normalizedX * normalizedX);
    const distortedY = normalizedY * radial + p1 * (r2 + 2 * normalizedY * normalizedY) + 2 * p2 * normalizedX * normalizedY;
    normalizedX = distortedX;
    normalizedY = distortedY;
  }
  const imageX = calibration.intrinsics.fx * normalizedX + calibration.intrinsics.cx;
  const imageY = calibration.intrinsics.fy * normalizedY + calibration.intrinsics.cy;
  return Number.isFinite(imageX) && Number.isFinite(imageY) ? [imageX, imageY] : null;
}

function nearestFrame(frames: Sam3DKeypointFrame[], timeSeconds: number): Sam3DKeypointFrame | null {
  if (!frames.length) return null;
  let low = 0;
  let high = frames.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (frames[middle].t < timeSeconds) low = middle + 1;
    else high = middle;
  }
  const right = frames[low];
  const left = frames[Math.max(0, low - 1)];
  return Math.abs(left.t - timeSeconds) <= Math.abs(right.t - timeSeconds) ? left : right;
}

function record(value: unknown, name: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${name} must be an object`);
  return value as Record<string, unknown>;
}

function array(value: unknown, name: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${name} must be an array`);
  return value;
}

function finiteNumber(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${name} must be a finite number`);
  return value;
}

function integer(value: unknown, name: string): number {
  const parsed = finiteNumber(value, name);
  if (!Number.isInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
}

function stringValue(value: unknown, name: string): string {
  if (typeof value !== "string" || !value) throw new Error(`${name} must be a non-empty string`);
  return value;
}

function vec2(value: unknown, name: string): Vec2 {
  const values = array(value, name);
  if (values.length !== 2) throw new Error(`${name} must contain two numbers`);
  return [finiteNumber(values[0], `${name}[0]`), finiteNumber(values[1], `${name}[1]`)];
}

function vec3(value: unknown, name: string): Vec3 {
  const values = array(value, name);
  if (values.length !== 3) throw new Error(`${name} must contain three numbers`);
  return [
    finiteNumber(values[0], `${name}[0]`),
    finiteNumber(values[1], `${name}[1]`),
    finiteNumber(values[2], `${name}[2]`),
  ];
}
