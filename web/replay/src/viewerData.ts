export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

export type LabelOverlay = {
  kind: "player_boxes" | string;
  label: string;
  url: string;
  trusted_for_metrics: boolean;
  not_ground_truth: boolean;
};

export type AnnotationSource = {
  kind: "person_ground_truth" | "annotation" | string;
  clip_id: string;
  url: string;
  trusted_for_metrics: boolean;
};

export type ViewerManifest = {
  schema_version: 1;
  artifact_type: "racketsport_replay_viewer_manifest";
  clip: string;
  video_url: string;
  virtual_world_url: string;
  replay_scene_url: string | null;
  physics_refinement_url: string | null;
  contact_windows_url: string | null;
  label_overlays: LabelOverlay[];
  annotation_sources: AnnotationSource[];
  notes: string[];
};

export type ContactWindowEvent = {
  type: string;
  t: number;
  frame: number | null;
  player_id: number | null;
  confidence: number | null;
  window: {
    t0: number;
    t1: number;
    importance: number | null;
  };
};

export type ContactWindows = {
  schema_version: 1;
  events: ContactWindowEvent[];
};

export type PhysicsRefinement = {
  schema_version: 1;
  artifact_type: "racketsport_physics_refinement";
  physics: string;
  foot2_done: boolean;
  must_not_mark_done_verified: boolean;
  constraint_summary: {
    contact_frames: number;
    max_contact_slide_m: number;
    max_floor_penetration_m: number;
    inter_player_penetration_frames: number;
    max_inter_player_penetration_m: number;
  };
  execution_plan: {
    mode: string;
    will_run_mjx: boolean;
    reason: string;
  };
};

export type VirtualWorldFrame = {
  t: number;
  track_world_xy?: Vec2 | null;
  bbox?: [number, number, number, number] | null;
  transl_world?: Vec3 | null;
  joints_world: Vec3[];
  mesh_vertices_world: Vec3[];
  joint_count: number;
  mesh_vertex_count: number;
  floor_world_xyz?: Vec3 | null;
  floor_source?: string | null;
  floor_offset_m?: number | null;
  min_mesh_z_m?: number | null;
  floor_penetration_m?: number;
  foot_contact?: { left: boolean; right: boolean } | null;
  contact_locked?: boolean;
  physics?: string | null;
};

export type VirtualWorldPlayer = {
  id: number;
  side?: string | null;
  role?: string | null;
  representation: "track_only" | "joints" | "mesh" | string;
  frames: VirtualWorldFrame[];
};

export type VirtualWorld = {
  schema_version: 1;
  artifact_type: "racketsport_virtual_world";
  world_frame: "court_Z0";
  fps: number;
  court: {
    sport: "pickleball" | "tennis";
    coordinate_frame: string;
    length_m: number;
    width_m: number;
    line_segments: Record<string, [Vec3, Vec3]>;
    net: {
      endpoints: [Vec3, Vec3];
      center_height_m: number;
      post_height_m: number;
    };
  };
  players: VirtualWorldPlayer[];
  ball: { source: string | null; frames: Array<{ t: number; world_xyz?: Vec3 | null; visible?: boolean }> };
  paddles: unknown[];
  summary: {
    player_count: number;
    mesh_player_count: number;
    mesh_player_frame_count: number;
    joint_player_frame_count: number;
    track_only_player_frame_count: number;
    floor_placed_player_frame_count: number;
    floor_contact_player_frame_count: number;
    max_floor_penetration_m: number;
    max_abs_floor_offset_m: number;
    physics_modes: string[];
    ball_frame_count: number;
    approx_ball_frame_count: number;
    paddle_player_count: number;
    paddle_frame_count: number;
    ambiguous_paddle_frame_count: number;
    warnings: string[];
  };
};

export function parseViewerManifest(input: unknown): ViewerManifest {
  const value = parseMaybeJson(input);
  assertRecord(value, "manifest");
  if (value.schema_version !== 1) throw new Error("manifest.schema_version must be 1");
  if (value.artifact_type !== "racketsport_replay_viewer_manifest") {
    throw new Error("manifest.artifact_type must be racketsport_replay_viewer_manifest");
  }
  return {
    schema_version: 1,
    artifact_type: "racketsport_replay_viewer_manifest",
    clip: readString(value.clip, "manifest.clip"),
    video_url: readString(value.video_url, "manifest.video_url"),
    virtual_world_url: readString(value.virtual_world_url, "manifest.virtual_world_url"),
    replay_scene_url: value.replay_scene_url === null ? null : readString(value.replay_scene_url, "manifest.replay_scene_url"),
    physics_refinement_url:
      value.physics_refinement_url === null || value.physics_refinement_url === undefined
        ? null
        : readString(value.physics_refinement_url, "manifest.physics_refinement_url"),
    contact_windows_url:
      value.contact_windows_url === null || value.contact_windows_url === undefined
        ? null
        : readString(value.contact_windows_url, "manifest.contact_windows_url"),
    label_overlays: readArray(value.label_overlays, "manifest.label_overlays").map(readLabelOverlay),
    annotation_sources: readArray(value.annotation_sources, "manifest.annotation_sources").map(readAnnotationSource),
    notes: readArray(value.notes, "manifest.notes").map((entry, index) => readString(entry, `manifest.notes[${index}]`)),
  };
}

export function parseContactWindows(input: unknown): ContactWindows {
  const value = parseMaybeJson(input);
  assertRecord(value, "contact_windows");
  if (value.schema_version !== 1) throw new Error("contact_windows.schema_version must be 1");
  return {
    schema_version: 1,
    events: readArray(value.events, "contact_windows.events").map(readContactWindowEvent),
  };
}

export function parsePhysicsRefinement(input: unknown): PhysicsRefinement {
  const value = parseMaybeJson(input);
  assertRecord(value, "physics_refinement");
  if (value.schema_version !== 1) throw new Error("physics_refinement.schema_version must be 1");
  if (value.artifact_type !== "racketsport_physics_refinement") {
    throw new Error("physics_refinement.artifact_type must be racketsport_physics_refinement");
  }
  assertRecord(value.constraint_summary, "physics_refinement.constraint_summary");
  assertRecord(value.execution_plan, "physics_refinement.execution_plan");
  return {
    schema_version: 1,
    artifact_type: "racketsport_physics_refinement",
    physics: readString(value.physics, "physics_refinement.physics"),
    foot2_done: readBoolean(value.foot2_done, "physics_refinement.foot2_done"),
    must_not_mark_done_verified: readBoolean(
      value.must_not_mark_done_verified,
      "physics_refinement.must_not_mark_done_verified",
    ),
    constraint_summary: {
      contact_frames: readNumber(value.constraint_summary.contact_frames, "constraint_summary.contact_frames", true),
      max_contact_slide_m: readNumber(value.constraint_summary.max_contact_slide_m, "constraint_summary.max_contact_slide_m"),
      max_floor_penetration_m: readNumber(
        value.constraint_summary.max_floor_penetration_m,
        "constraint_summary.max_floor_penetration_m",
      ),
      inter_player_penetration_frames: readNumber(
        value.constraint_summary.inter_player_penetration_frames,
        "constraint_summary.inter_player_penetration_frames",
        true,
      ),
      max_inter_player_penetration_m: readNumber(
        value.constraint_summary.max_inter_player_penetration_m,
        "constraint_summary.max_inter_player_penetration_m",
      ),
    },
    execution_plan: {
      mode: readString(value.execution_plan.mode, "execution_plan.mode"),
      will_run_mjx: readBoolean(value.execution_plan.will_run_mjx, "execution_plan.will_run_mjx"),
      reason: readString(value.execution_plan.reason, "execution_plan.reason"),
    },
  };
}

export function parseVirtualWorld(input: unknown): VirtualWorld {
  const value = parseMaybeJson(input);
  assertRecord(value, "virtual_world");
  if (value.schema_version !== 1) throw new Error("virtual_world.schema_version must be 1");
  if (value.artifact_type !== "racketsport_virtual_world") {
    throw new Error("virtual_world.artifact_type must be racketsport_virtual_world");
  }
  if (value.world_frame !== "court_Z0") throw new Error("virtual_world.world_frame must be court_Z0");
  const court = readCourt(value.court);
  const players = readArray(value.players, "virtual_world.players").map(readPlayer);
  const summary = readSummary(value.summary);
  return {
    schema_version: 1,
    artifact_type: "racketsport_virtual_world",
    world_frame: "court_Z0",
    fps: readNumber(value.fps, "virtual_world.fps"),
    court,
    players,
    ball: readBall(value.ball),
    paddles: readArray(value.paddles, "virtual_world.paddles"),
    summary,
  };
}

export function frameForTime(player: VirtualWorldPlayer, timeSeconds: number): VirtualWorldFrame | undefined {
  if (!player.frames.length) return undefined;
  return player.frames.reduce((best, frame) =>
    Math.abs(frame.t - timeSeconds) < Math.abs(best.t - timeSeconds) ? frame : best,
  );
}

export function ballFrameForTime(world: VirtualWorld, timeSeconds: number): VirtualWorld["ball"]["frames"][number] | undefined {
  const frames = world.ball.frames.filter((frame) => frame.visible !== false && frame.world_xyz);
  if (!frames.length) return undefined;
  return frames.reduce((best, frame) => (Math.abs(frame.t - timeSeconds) < Math.abs(best.t - timeSeconds) ? frame : best));
}

export function contactEventsForTime(contactWindows: ContactWindows | null, timeSeconds: number): ContactWindowEvent[] {
  if (!contactWindows) return [];
  return contactWindows.events.filter((event) => event.type === "contact" && event.window.t0 <= timeSeconds && timeSeconds <= event.window.t1);
}

export function contactEventCount(contactWindows: ContactWindows | null): number {
  return contactWindows?.events.filter((event) => event.type === "contact").length ?? 0;
}

export function activeBallContactPlayerIds(
  world: VirtualWorld,
  contactWindows: ContactWindows | null,
  timeSeconds: number,
): Set<number> {
  const activeEvents = contactEventsForTime(contactWindows, timeSeconds);
  const playerIds = new Set<number>();
  for (const event of activeEvents) {
    if (event.player_id !== null) {
      playerIds.add(event.player_id);
    }
  }
  if (playerIds.size > 0 || activeEvents.length === 0) return playerIds;

  const ball = ballFrameForTime(world, timeSeconds)?.world_xyz;
  if (!ball) return playerIds;
  const candidates = world.players
    .map((player) => {
      const frame = frameForTime(player, timeSeconds);
      const floor = frame?.floor_world_xyz ?? (frame?.track_world_xy ? ([frame.track_world_xy[0], frame.track_world_xy[1], 0] as Vec3) : null);
      if (!floor) return null;
      return { playerId: player.id, distance: Math.hypot(floor[0] - ball[0], floor[1] - ball[1]) };
    })
    .filter((candidate): candidate is { playerId: number; distance: number } => candidate !== null)
    .sort((left, right) => left.distance - right.distance);
  if (candidates[0]) playerIds.add(candidates[0].playerId);
  return playerIds;
}

export function startTimeFromSearch(search: string): number {
  const params = new URLSearchParams(search);
  const raw = params.get("t") ?? params.get("time");
  if (raw === null) return 0;
  const seconds = Number(raw);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : 0;
}

export function worldStats(world: VirtualWorld) {
  return {
    players: world.players.length,
    meshFrames: world.summary.mesh_player_frame_count,
    floorPlacedFrames: world.summary.floor_placed_player_frame_count,
    contactFrames: world.summary.floor_contact_player_frame_count,
    maxFloorPenetrationM: world.summary.max_floor_penetration_m,
    physicsModes: world.summary.physics_modes,
  };
}

function readLabelOverlay(input: unknown, index: number): LabelOverlay {
  const path = `manifest.label_overlays[${index}]`;
  assertRecord(input, path);
  return {
    kind: readString(input.kind, `${path}.kind`),
    label: readString(input.label, `${path}.label`),
    url: readString(input.url, `${path}.url`),
    trusted_for_metrics: readBoolean(input.trusted_for_metrics, `${path}.trusted_for_metrics`),
    not_ground_truth: readBoolean(input.not_ground_truth, `${path}.not_ground_truth`),
  };
}

function readAnnotationSource(input: unknown, index: number): AnnotationSource {
  const path = `manifest.annotation_sources[${index}]`;
  assertRecord(input, path);
  return {
    kind: readString(input.kind, `${path}.kind`),
    clip_id: readString(input.clip_id, `${path}.clip_id`),
    url: readString(input.url, `${path}.url`),
    trusted_for_metrics: readBoolean(input.trusted_for_metrics, `${path}.trusted_for_metrics`),
  };
}

function readContactWindowEvent(input: unknown, index: number): ContactWindowEvent {
  const path = `contact_windows.events[${index}]`;
  assertRecord(input, path);
  assertRecord(input.window, `${path}.window`);
  return {
    type: readString(input.type, `${path}.type`),
    t: readNumber(input.t, `${path}.t`),
    frame: input.frame === null || input.frame === undefined ? null : readNumber(input.frame, `${path}.frame`, true),
    player_id: input.player_id === null || input.player_id === undefined ? null : readPlayerId(input.player_id, `${path}.player_id`),
    confidence: input.confidence === null || input.confidence === undefined ? null : readNumber(input.confidence, `${path}.confidence`),
    window: {
      t0: readNumber(input.window.t0, `${path}.window.t0`),
      t1: readNumber(input.window.t1, `${path}.window.t1`),
      importance:
        input.window.importance === null || input.window.importance === undefined
          ? null
          : readNumber(input.window.importance, `${path}.window.importance`),
    },
  };
}

function readCourt(input: unknown): VirtualWorld["court"] {
  assertRecord(input, "virtual_world.court");
  assertRecord(input.net, "virtual_world.court.net");
  const rawSegments = input.line_segments;
  assertRecord(rawSegments, "virtual_world.court.line_segments");
  const line_segments: Record<string, [Vec3, Vec3]> = {};
  for (const [key, value] of Object.entries(rawSegments)) {
    const segment = readArray(value, `virtual_world.court.line_segments.${key}`);
    line_segments[key] = [readVec3(segment[0], `${key}[0]`), readVec3(segment[1], `${key}[1]`)];
  }
  const endpoints = readArray(input.net.endpoints, "virtual_world.court.net.endpoints");
  return {
    sport: readString(input.sport, "virtual_world.court.sport") as "pickleball" | "tennis",
    coordinate_frame: readString(input.coordinate_frame, "virtual_world.court.coordinate_frame"),
    length_m: readNumber(input.length_m, "virtual_world.court.length_m"),
    width_m: readNumber(input.width_m, "virtual_world.court.width_m"),
    line_segments,
    net: {
      endpoints: [readVec3(endpoints[0], "net.endpoints[0]"), readVec3(endpoints[1], "net.endpoints[1]")],
      center_height_m: readNumber(input.net.center_height_m, "virtual_world.court.net.center_height_m"),
      post_height_m: readNumber(input.net.post_height_m, "virtual_world.court.net.post_height_m"),
    },
  };
}

function readPlayer(input: unknown, index: number): VirtualWorldPlayer {
  const path = `virtual_world.players[${index}]`;
  assertRecord(input, path);
  return {
    id: readNumber(input.id, `${path}.id`, true),
    side: input.side === null || input.side === undefined ? null : readString(input.side, `${path}.side`),
    role: input.role === null || input.role === undefined ? null : readString(input.role, `${path}.role`),
    representation: readString(input.representation, `${path}.representation`),
    frames: readArray(input.frames, `${path}.frames`).map((frame, frameIndex) => readFrame(frame, `${path}.frames[${frameIndex}]`)),
  };
}

function readFrame(input: unknown, path: string): VirtualWorldFrame {
  assertRecord(input, path);
  return {
    t: readNumber(input.t, `${path}.t`),
    track_world_xy: input.track_world_xy === null || input.track_world_xy === undefined ? null : readVec2(input.track_world_xy, `${path}.track_world_xy`),
    bbox: input.bbox === null || input.bbox === undefined ? null : readBBox(input.bbox, `${path}.bbox`),
    transl_world: input.transl_world === null || input.transl_world === undefined ? null : readVec3(input.transl_world, `${path}.transl_world`),
    joints_world: readArray(input.joints_world, `${path}.joints_world`).map((point, index) => readVec3(point, `${path}.joints_world[${index}]`)),
    mesh_vertices_world: readArray(input.mesh_vertices_world, `${path}.mesh_vertices_world`).map((point, index) => readVec3(point, `${path}.mesh_vertices_world[${index}]`)),
    joint_count: readNumber(input.joint_count, `${path}.joint_count`, true),
    mesh_vertex_count: readNumber(input.mesh_vertex_count, `${path}.mesh_vertex_count`, true),
    floor_world_xyz: input.floor_world_xyz === null || input.floor_world_xyz === undefined ? null : readVec3(input.floor_world_xyz, `${path}.floor_world_xyz`),
    floor_source: input.floor_source === null || input.floor_source === undefined ? null : readString(input.floor_source, `${path}.floor_source`),
    floor_offset_m: input.floor_offset_m === null || input.floor_offset_m === undefined ? null : readNumber(input.floor_offset_m, `${path}.floor_offset_m`),
    min_mesh_z_m: input.min_mesh_z_m === null || input.min_mesh_z_m === undefined ? null : readNumber(input.min_mesh_z_m, `${path}.min_mesh_z_m`),
    floor_penetration_m: input.floor_penetration_m === undefined ? 0 : readNumber(input.floor_penetration_m, `${path}.floor_penetration_m`),
    foot_contact: input.foot_contact === null || input.foot_contact === undefined ? null : readFootContact(input.foot_contact, `${path}.foot_contact`),
    contact_locked: input.contact_locked === undefined ? false : readBoolean(input.contact_locked, `${path}.contact_locked`),
    physics: input.physics === null || input.physics === undefined ? null : readString(input.physics, `${path}.physics`),
  };
}

function readBall(input: unknown): VirtualWorld["ball"] {
  assertRecord(input, "virtual_world.ball");
  return {
    source: input.source === null || input.source === undefined ? null : readString(input.source, "virtual_world.ball.source"),
    frames: readArray(input.frames, "virtual_world.ball.frames").map((frame, index) => {
      const path = `virtual_world.ball.frames[${index}]`;
      assertRecord(frame, path);
      return {
        t: readNumber(frame.t, `${path}.t`),
        world_xyz: frame.world_xyz === null || frame.world_xyz === undefined ? null : readVec3(frame.world_xyz, `${path}.world_xyz`),
        visible: frame.visible === undefined ? undefined : readBoolean(frame.visible, `${path}.visible`),
      };
    }),
  };
}

function readSummary(input: unknown): VirtualWorld["summary"] {
  assertRecord(input, "virtual_world.summary");
  return {
    player_count: readNumber(input.player_count, "summary.player_count", true),
    mesh_player_count: readNumber(input.mesh_player_count, "summary.mesh_player_count", true),
    mesh_player_frame_count: readNumber(input.mesh_player_frame_count, "summary.mesh_player_frame_count", true),
    joint_player_frame_count: readNumber(input.joint_player_frame_count, "summary.joint_player_frame_count", true),
    track_only_player_frame_count: readNumber(input.track_only_player_frame_count, "summary.track_only_player_frame_count", true),
    floor_placed_player_frame_count: readNumber(input.floor_placed_player_frame_count, "summary.floor_placed_player_frame_count", true),
    floor_contact_player_frame_count: readNumber(input.floor_contact_player_frame_count, "summary.floor_contact_player_frame_count", true),
    max_floor_penetration_m: readNumber(input.max_floor_penetration_m, "summary.max_floor_penetration_m"),
    max_abs_floor_offset_m: readNumber(input.max_abs_floor_offset_m, "summary.max_abs_floor_offset_m"),
    physics_modes: readArray(input.physics_modes, "summary.physics_modes").map((mode, index) => readString(mode, `summary.physics_modes[${index}]`)),
    ball_frame_count: readNumber(input.ball_frame_count, "summary.ball_frame_count", true),
    approx_ball_frame_count: readNumber(input.approx_ball_frame_count, "summary.approx_ball_frame_count", true),
    paddle_player_count: readNumber(input.paddle_player_count, "summary.paddle_player_count", true),
    paddle_frame_count: readNumber(input.paddle_frame_count, "summary.paddle_frame_count", true),
    ambiguous_paddle_frame_count: readNumber(input.ambiguous_paddle_frame_count, "summary.ambiguous_paddle_frame_count", true),
    warnings: readArray(input.warnings, "summary.warnings").map((warning, index) => readString(warning, `summary.warnings[${index}]`)),
  };
}

function readFootContact(input: unknown, path: string): { left: boolean; right: boolean } {
  assertRecord(input, path);
  return {
    left: readBoolean(input.left, `${path}.left`),
    right: readBoolean(input.right, `${path}.right`),
  };
}

function parseMaybeJson(input: unknown): unknown {
  if (typeof input !== "string") return input;
  try {
    return JSON.parse(input) as unknown;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`invalid JSON: ${message}`);
  }
}

function readArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${path} must be an array`);
  return value;
}

function readNumber(value: unknown, path: string, integer = false): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${path} must be a number`);
  if (integer && !Number.isInteger(value)) throw new Error(`${path} must be an integer`);
  return value;
}

function readPlayerId(value: unknown, path: string): number {
  if (typeof value === "number") return readNumber(value, path, true);
  if (typeof value === "string" && value.trim() !== "") {
    const number = Number(value);
    if (Number.isInteger(number)) return number;
  }
  throw new Error(`${path} must be an integer player id`);
}

function readString(value: unknown, path: string): string {
  if (typeof value !== "string") throw new Error(`${path} must be a string`);
  return value;
}

function readBoolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") throw new Error(`${path} must be a boolean`);
  return value;
}

function readVec2(value: unknown, path: string): Vec2 {
  const values = readArray(value, path);
  return [readNumber(values[0], `${path}[0]`), readNumber(values[1], `${path}[1]`)];
}

function readVec3(value: unknown, path: string): Vec3 {
  const values = readArray(value, path);
  return [readNumber(values[0], `${path}[0]`), readNumber(values[1], `${path}[1]`), readNumber(values[2], `${path}[2]`)];
}

function readBBox(value: unknown, path: string): [number, number, number, number] {
  const values = readArray(value, path);
  return [
    readNumber(values[0], `${path}[0]`),
    readNumber(values[1], `${path}[1]`),
    readNumber(values[2], `${path}[2]`),
    readNumber(values[3], `${path}[3]`),
  ];
}

function assertRecord(value: unknown, path: string): asserts value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error(`${path} must be an object`);
}
