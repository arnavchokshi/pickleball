export type ReplayPoint = {
  schema_version?: never;
  id: number;
  t0: number;
  t1: number;
  glb_url: string;
  size_mb: number;
};

export type ReplayScene = {
  schema_version: 1;
  world_frame: "court_Z0";
  fps: number;
  court_glb: string;
  players: number[];
  points: ReplayPoint[];
};

const replaySceneKeys = ["schema_version", "world_frame", "fps", "court_glb", "players", "points"];
const replayPointKeys = ["id", "t0", "t1", "glb_url", "size_mb"];

export function parseReplayScene(input: unknown): ReplayScene {
  const value = typeof input === "string" ? parseJson(input) : input;
  assertRecord(value, "scene");
  assertExactKeys(value, replaySceneKeys);

  if (value.schema_version !== 1) {
    throw new Error("schema_version must be 1");
  }
  if (value.world_frame !== "court_Z0") {
    throw new Error("world_frame must be court_Z0");
  }

  const fps = readNumber(value.fps, "fps");
  const courtGlb = readString(value.court_glb, "court_glb");
  const players = readNumberArray(value.players, "players", true);
  const points = readReplayPoints(value.points);

  return {
    schema_version: 1,
    world_frame: "court_Z0",
    fps,
    court_glb: courtGlb,
    players,
    points,
  };
}

export function activeReplayPointForTime(scene: ReplayScene, timeSeconds: number): ReplayPoint | undefined {
  return scene.points.find((point, index) => {
    const isLastPoint = index === scene.points.length - 1;
    return point.t0 <= timeSeconds && (timeSeconds < point.t1 || (isLastPoint && timeSeconds <= point.t1));
  });
}

export function resolveReplaySceneAssetUrl(replaySceneUrl: string, assetPath: string): string {
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(assetPath)) return assetPath;
  if (assetPath.startsWith("/")) return assetPath;
  const origin = typeof window === "undefined" ? "http://localhost" : window.location.origin;
  const resolved = new URL(assetPath, new URL(replaySceneUrl, origin));
  if (/^(https?|file):\/\//.test(replaySceneUrl)) {
    return resolved.href;
  }
  return resolved.pathname + resolved.search + resolved.hash;
}

function parseJson(input: string): unknown {
  try {
    return JSON.parse(input) as unknown;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`invalid JSON: ${message}`);
  }
}

function readReplayPoints(value: unknown): ReplayPoint[] {
  if (!Array.isArray(value)) {
    throw new Error("points must be an array");
  }

  let previousT1: number | null = null;
  return value.map((point, index) => {
    const path = `points[${index}]`;
    assertRecord(point, path);
    assertExactKeys(point, replayPointKeys, path);
    const t0 = readNumber(point.t0, `${path}.t0`);
    const t1 = readNumber(point.t1, `${path}.t1`);
    if (t1 <= t0) {
      throw new Error(`${path}.t1 must be greater than ${path}.t0`);
    }
    if (previousT1 !== null && t0 < previousT1) {
      throw new Error(`${path}.t0 must be greater than or equal to previous point t1`);
    }
    previousT1 = t1;

    return {
      id: readNumber(point.id, `${path}.id`, true),
      t0,
      t1,
      glb_url: readString(point.glb_url, `${path}.glb_url`),
      size_mb: readNumber(point.size_mb, `${path}.size_mb`),
    };
  });
}

function readNumberArray(value: unknown, path: string, integer = false): number[] {
  if (!Array.isArray(value)) {
    throw new Error(`${path} must be an array`);
  }

  return value.map((entry, index) => readNumber(entry, `${path}[${index}]`, integer));
}

function readNumber(value: unknown, path: string, integer = false): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${path} must be a number`);
  }
  if (integer && !Number.isInteger(value)) {
    throw new Error(`${path} must be an integer`);
  }

  return value;
}

function readString(value: unknown, path: string): string {
  if (typeof value !== "string") {
    throw new Error(`${path} must be a string`);
  }

  return value;
}

function assertRecord(value: unknown, path: string): asserts value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${path} must be an object`);
  }
}

function assertExactKeys(value: Record<string, unknown>, allowed: string[], path?: string): void {
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) {
      const prefix = path ? `${path}.` : "";
      throw new Error(`unexpected field: ${prefix}${key}`);
    }
  }
}
