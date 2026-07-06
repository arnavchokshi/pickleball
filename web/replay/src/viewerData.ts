export type Vec2 = [number, number];
export type Vec3 = [number, number, number];
export type Matrix3 = [Vec3, Vec3, Vec3];
export type MeshFace = [number, number, number];

export type TrustBadge = "verified" | "preview" | "low_confidence";
export type CoachingFactTrust = "ok" | "estimated" | "unverified_cue";

export type TrustBand = {
  stage: string;
  gate_id: string;
  gate_status: string;
  badge: TrustBadge;
  reason: string;
  evidence_path: string | null;
};

export type EntityCoverage = {
  coverage_fraction?: number | null;
  min_t?: number | null;
  max_t?: number | null;
};

export type ConfidenceProvenance = {
  band: string | null;
  display_band: string | null;
  horizon_frames: number | null;
  predicted_sigma_m: number | null;
  predictor: string | null;
};

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

export type LabelItem = {
  frame?: string | number;
  bbox?: number[];
  bbox_xyxy?: number[];
  id?: string;
  status?: string;
};

export type LabelOverlayPayload = {
  items: LabelItem[];
  notGroundTruth: boolean;
  status: string | null;
  sourceWidth: number;
  sourceHeight: number;
  secondsPerFrame: number;
};

export type ViewerManifest = {
  schema_version: 1;
  artifact_type: "racketsport_replay_viewer_manifest";
  clip: string;
  video_url: string;
  virtual_world_url: string;
  replay_scene_url: string | null;
  body_mesh_url: string | null;
  body_mesh_index_url: string | null;
  physics_refinement_url: string | null;
  contact_windows_url: string | null;
  reviewed_bounces_url: string | null;
  ball_inflections_url: string | null;
  events_selected_url: string | null;
  shots_url?: string;
  ball_arc_solved_url?: string;
  ball_arc_render_url?: string;
  auto_bounce_candidates_url?: string;
  ball_bounce_candidates_url?: string;
  ball_flight_sanity_url?: string;
  rally_spans_url?: string;
  rally_metrics_url?: string;
  coaching_card_facts_url?: string;
  label_overlays: LabelOverlay[];
  annotation_sources: AnnotationSource[];
  notes: string[];
};

export type CoachingCardFact = {
  rally_id: string;
  player_id: string;
  metric: string;
  value: number;
  unit: string;
  trust: CoachingFactTrust;
  coverage_fraction: number;
  frames_total: number;
  frames_used: number;
  rally_scope: string;
};

export type CoachingCardFacts = {
  artifact_type: "coaching_card_facts";
  rally_scope: string;
  priority_rule: string[];
  facts: CoachingCardFact[];
};

export type BodyMeshFrame = {
  frame_idx: number;
  t: number;
  source_window_index: number | null;
  blend_weight: number;
  joints_world: Vec3[];
  joint_conf: number[];
  mesh_vertices_world: Vec3[];
  mesh_faces: MeshFace[];
  smplx_params: Record<string, number[]>;
  reasons: string[];
  mesh_interpolated: boolean;
  interpolation: BodyMeshInterpolation | null;
  mesh_alignment?: BodyMeshAlignmentDebug | null;
};

export type BodyMeshInterpolation = {
  from_frame_idx: number;
  to_frame_idx: number;
  alpha: number;
  max_gap_s: number;
};

export type BodyMeshAlignmentDebug = {
  applied: boolean;
  reason: "skeleton_root" | "missing_world_frame" | "missing_skeleton_root" | "missing_mesh_root";
  delta: Vec3;
  mesh_root: Vec3 | null;
  skeleton_root: Vec3 | null;
  floor_guard_applied: boolean;
  floor_lift_m: number;
};

export type BodyMeshPlayer = {
  id: number;
  frames: BodyMeshFrame[];
};

export type BodyMeshFaces = {
  schema_version: 1;
  artifact_type: "racketsport_body_mesh_faces";
  faces_ref: string;
  mesh_faces: MeshFace[];
};

export type BodyMeshChunkEncoding = "gzip_int16_world_vertices_v1" | "raw_int16_world_vertices_v1";

export type BodyMeshIndexFrame = {
  frame_idx: number;
  t: number;
  source_window_index: number | null;
  blend_weight: number;
  vertex_count: number;
  joint_count: number;
  joint_conf: number[];
  reasons: string[];
};

export type BodyMeshIndexPlayer = {
  id: number;
  frames: BodyMeshIndexFrame[];
};

export type BodyMeshIndexWindow = {
  source_window_index: number;
  frame_start: number;
  frame_end: number;
  t0: number;
  t1: number;
  frame_count: number;
  player_frame_count: number;
  target_player_ids: number[];
  player_ids: number[];
  target_representation: string;
  fallback_representation: string;
  reason_counts: Record<string, number>;
  max_score: number;
  url: string;
  byte_size: number;
  encoding: BodyMeshChunkEncoding;
  quantization: {
    scale: number;
    unit: "m";
  };
  players: BodyMeshIndexPlayer[];
};

export type BodyMeshIndex = {
  schema_version: 1;
  artifact_type: "racketsport_body_mesh_index";
  clip: string;
  model: string;
  fps: number;
  world_frame: "court_Z0";
  faces_ref: string;
  faces_url: string;
  windows: BodyMeshIndexWindow[];
  summary: {
    window_count: number;
    mesh_frame_count: number;
    player_count: number;
    faces_count: number;
  };
};

export type BodyMesh = {
  schema_version: 1;
  artifact_type: "racketsport_body_mesh";
  clip: string;
  model: string;
  fps: number;
  world_frame: "court_Z0";
  faces_ref: string;
  mesh_faces: MeshFace[];
  joint_names: string[];
  players: BodyMeshPlayer[];
  summary: {
    mesh_frame_count: number;
    player_count: number;
    contact_window_count: number;
  };
};

export type ActiveBodyMeshFrame = {
  playerId: number;
  meshPlayerId: number;
  frame: BodyMeshFrame;
  presenceOpacity: number;
  renderTranslation: Vec3;
  alignmentDebug?: BodyMeshAlignmentDebug;
};

export type BodyMeshLoadState =
  | "idle"
  | "index_ready"
  | "loading"
  | "loaded"
  | "no_window"
  | "fetch_failed"
  | "parse_failed"
  | "decode_failed";

export type BodyMeshLoadStatus = {
  state: BodyMeshLoadState;
  label: string;
  stage?: "index" | "faces" | "chunk" | "legacy";
  url?: string;
  windowId?: number | null;
  message?: string;
};

export type BodyMeshDebugPlayer = {
  world_player_id: number;
  world_frame_t: number | null;
  mesh_ref_player_id: number | null;
  mesh_ref_frame_idx: number | null;
  normalized_mesh_player_id: number;
  mesh_player_present: boolean;
  mesh_frame_present: boolean;
  mesh_frame_idx: number | null;
};

export type BodyMeshDebugSnapshot = {
  current_time: number;
  active_window_id: number | null;
  active_window_url: string | null;
  load_state: BodyMeshLoadState;
  load_stage: BodyMeshLoadStatus["stage"] | null;
  load_url: string | null;
  load_message: string | null;
  rendered_player_count: number;
  alignment_floor_guard_count: number;
  players: BodyMeshDebugPlayer[];
};

export type BodyMeshInterpolationStats = {
  computedFrameCount: number;
  eligiblePairCount: number;
  heldPairCount: number;
  gapRefusedPairCount: number;
  boundaryRefusedPairCount: number;
  mismatchedVertexRefusedPairCount: number;
  displayMultiplier: 1 | 2;
};

export type DisplayFpsStats = {
  enabled: boolean;
  sourceFps: number;
  displayFps: number;
  worldComputedFrameCount: number;
  worldInterpolatedFrameCount: number;
  meshComputedFrameCount: number;
  meshInterpolatedFrameCount: number;
  meshMaxInterpolatedGapMs: number;
  meshRefusedPairCount: number;
};

export type DisplayFpsReplayData = {
  world: VirtualWorld;
  bodyMesh: BodyMesh | null;
  stats: DisplayFpsStats;
};

export type DisplayFpsOptions = {
  meshMaxGapSeconds?: number;
};

export type ContactWindowEvent = {
  type: "contact" | "bounce" | "net_cross";
  t: number;
  frame: number;
  player_id: number | null;
  confidence: number;
  sources: {
    audio?: number;
    wrist_vel?: number;
    ball_inflection?: number;
    human_review?: number | null;
  };
  window: {
    t0: number;
    t1: number;
    importance: number;
  };
};

export type ContactWindows = {
  schema_version: 1;
  events: ContactWindowEvent[];
};

export type BallInflectionCandidate = {
  time_s: number;
  frame: number | null;
  confidence: number;
};

export type BallInflections = {
  candidates: BallInflectionCandidate[];
};

export type BallArcSelectedEvent = {
  anchor_id: string | null;
  kind: string;
  frame: number | null;
  t: number;
};

export type BallArcEventsSelected = {
  artifact_type: string | null;
  selected: BallArcSelectedEvent[];
};

export type ReviewedBounce = {
  review_id: string;
  frame: number;
  t: number;
};

export type ReviewedBounces = {
  schema_version: 1;
  artifact_type: "racketsport_reviewed_ball_bounces";
  source: string;
  bounces: ReviewedBounce[];
};

export type TimelineMarker = {
  kind: "contact" | "bounce" | "net_cross" | "ball_inflection" | "reviewed_bounce";
  t: number;
  t0: number;
  t1: number;
  confidence: number;
  badge: TrustBadge;
  label: string;
  humanReviewed?: boolean;
};

export type TimelineChapter = {
  index: number;
  rallyId?: string;
  t0: number;
  t1: number;
  label: string;
  badge: TrustBadge;
};

export type RallySpan = {
  rallyId: string;
  t0: number;
  t1: number;
  sources: string[];
};

export type RallySpans = {
  schema_version: 1;
  artifact_type: "racketsport_rally_spans";
  clip_id: string | null;
  duration_s: number | null;
  not_ground_truth: boolean;
  spans: RallySpan[];
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
  mesh_ref?: {
    artifact: string;
    player_id: number;
    frame_idx: number;
    t: number;
  } | null;
  track_world_xy?: Vec2 | null;
  track_conf?: number | null;
  bbox?: [number, number, number, number] | null;
  transl_world?: Vec3 | null;
  joints_world: Vec3[];
  joint_conf: number[];
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
  grf?: Vec3[] | null;
  skeleton_implausible?: boolean;
  trust_band?: TrustBand | null;
};

export type VirtualWorldPlayer = EntityCoverage & {
  id: number;
  side?: string | null;
  role?: string | null;
  representation: "track_only" | "joints" | "mesh";
  frames: VirtualWorldFrame[];
  trust_band?: TrustBand | null;
};

export type VirtualWorldPaddleFrame = {
  t: number;
  pose_se3: {
    R: Matrix3;
    t: Vec3;
  };
  mesh_vertices_world: Vec3[];
  mesh_faces: Array<[number, number, number]>;
  conf: number;
  world_frame: "court_Z0";
  translation_unit: "m";
  source: string;
  reprojection_error_px?: number | null;
  ambiguous: boolean;
  confidence_provenance?: ConfidenceProvenance | null;
  render_only?: boolean;
  not_for_detection_metrics?: boolean;
  trust_band?: TrustBand | null;
};

export type VirtualWorldPaddle = EntityCoverage & {
  player_id: number;
  paddle_dims_in: Record<string, number>;
  frames: VirtualWorldPaddleFrame[];
  trust_band?: TrustBand | null;
};

export type ActivePaddleFrame = {
  playerId: number;
  paddle: VirtualWorldPaddle;
  frame: VirtualWorldPaddleFrame;
  estimated: boolean;
};

export type VirtualWorldBallFrame = {
  t: number;
  xy: Vec2;
  xy_interpolated?: boolean;
  conf: number;
  visible: boolean;
  world_xyz?: Vec3 | null;
  court_intersection_world_xyz?: Vec3 | null;
  arc_segment_id?: number | string | null;
  approx: boolean;
  confidence_provenance?: ConfidenceProvenance | null;
  render_only?: boolean;
  not_for_detection_metrics?: boolean;
  trust_band?: TrustBand | null;
  physics_fill?: {
    uncertainty_m?: number | null;
    render_only?: boolean;
    not_for_detection_metrics?: boolean;
  } | null;
};

export type BallTrailPoint = {
  point: Vec3;
  courtStyle: "inside_court" | "outside_court";
  uncertaintySigmaM: number | null;
  opacityScale: number;
  thicknessScale: number;
};

export type BallTrailSegment = {
  from: Vec3;
  to: Vec3;
  courtStyle: "inside_court" | "outside_court";
  opacityScale: number;
  thicknessScale: number;
};

export type VideoBallOverlay = {
  frame: VirtualWorldBallFrame;
  point: Vec2;
  confidenceClass: "high" | "medium" | "low";
  interpolated: boolean;
  opacity: number;
  radius: number;
};

export type BallGhostMarker = {
  frame: VirtualWorldBallFrame;
  position: Vec3;
  label: "2D-only, no 3D solve";
};

export type VirtualWorld = {
  schema_version: 1;
  artifact_type: "racketsport_virtual_world";
  world_frame: "court_Z0";
  fps: number;
  joint_names?: string[];
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
    trust_band?: TrustBand | null;
  };
  players: VirtualWorldPlayer[];
  ball: {
    // `source` is opaque provenance metadata naming which ball detector/fuser
    // produced this track (see `BallTrack.source` /
    // `VirtualWorldBall.source` in threed/racketsport/schemas/__init__.py).
    // It is display-only here -- nothing in the viewer branches on its
    // value, and BALL's actual trust/confidence comes from `trust_band`
    // below, which is validated independently. The Python-side enum has
    // already grown once (tracknet/tap/pbmat/totnet -> +wasb, fused,
    // vn_trajectories) and will keep growing, so we deliberately do not
    // mirror it as a closed TS union: a stricter type would just reject
    // valid future sources again. Any non-empty string is accepted.
    source: string | null;
    frames: VirtualWorldBallFrame[];
    trust_band?: TrustBand | null;
  } & EntityCoverage;
  paddles: VirtualWorldPaddle[];
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
  const manifest: ViewerManifest = {
    schema_version: 1,
    artifact_type: "racketsport_replay_viewer_manifest",
    clip: readString(value.clip, "manifest.clip"),
    video_url: readString(value.video_url, "manifest.video_url"),
    virtual_world_url: readString(value.virtual_world_url, "manifest.virtual_world_url"),
    replay_scene_url: value.replay_scene_url === null ? null : readString(value.replay_scene_url, "manifest.replay_scene_url"),
    body_mesh_url:
      value.body_mesh_url === null || value.body_mesh_url === undefined
        ? null
        : readString(value.body_mesh_url, "manifest.body_mesh_url"),
    body_mesh_index_url:
      value.body_mesh_index_url === null || value.body_mesh_index_url === undefined
        ? null
        : readString(value.body_mesh_index_url, "manifest.body_mesh_index_url"),
    physics_refinement_url:
      value.physics_refinement_url === null || value.physics_refinement_url === undefined
        ? null
        : readString(value.physics_refinement_url, "manifest.physics_refinement_url"),
    contact_windows_url:
      value.contact_windows_url === null || value.contact_windows_url === undefined
        ? null
        : readString(value.contact_windows_url, "manifest.contact_windows_url"),
    reviewed_bounces_url:
      value.reviewed_bounces_url === null || value.reviewed_bounces_url === undefined
        ? null
        : readString(value.reviewed_bounces_url, "manifest.reviewed_bounces_url"),
    ball_inflections_url:
      value.ball_inflections_url === null || value.ball_inflections_url === undefined
        ? null
        : readString(value.ball_inflections_url, "manifest.ball_inflections_url"),
    events_selected_url:
      value.events_selected_url === null || value.events_selected_url === undefined
        ? null
        : readString(value.events_selected_url, "manifest.events_selected_url"),
    label_overlays: readArray(value.label_overlays, "manifest.label_overlays").map(readLabelOverlay),
    annotation_sources: readArray(value.annotation_sources, "manifest.annotation_sources").map(readAnnotationSource),
    notes: readArray(value.notes, "manifest.notes").map((entry, index) => readString(entry, `manifest.notes[${index}]`)),
  };
  if (value.rally_metrics_url !== null && value.rally_metrics_url !== undefined) {
    manifest.rally_metrics_url = readString(value.rally_metrics_url, "manifest.rally_metrics_url");
  }
  if (value.coaching_card_facts_url !== null && value.coaching_card_facts_url !== undefined) {
    manifest.coaching_card_facts_url = readString(value.coaching_card_facts_url, "manifest.coaching_card_facts_url");
  }
  if (value.rally_spans_url !== null && value.rally_spans_url !== undefined) {
    manifest.rally_spans_url = readString(value.rally_spans_url, "manifest.rally_spans_url");
  }
  if (value.shots_url !== null && value.shots_url !== undefined) {
    manifest.shots_url = readString(value.shots_url, "manifest.shots_url");
  }
  if (value.ball_arc_solved_url !== null && value.ball_arc_solved_url !== undefined) {
    manifest.ball_arc_solved_url = readString(value.ball_arc_solved_url, "manifest.ball_arc_solved_url");
  }
  if (value.ball_arc_render_url !== null && value.ball_arc_render_url !== undefined) {
    manifest.ball_arc_render_url = readString(value.ball_arc_render_url, "manifest.ball_arc_render_url");
  }
  if (value.auto_bounce_candidates_url !== null && value.auto_bounce_candidates_url !== undefined) {
    manifest.auto_bounce_candidates_url = readString(value.auto_bounce_candidates_url, "manifest.auto_bounce_candidates_url");
  }
  if (value.ball_bounce_candidates_url !== null && value.ball_bounce_candidates_url !== undefined) {
    manifest.ball_bounce_candidates_url = readString(value.ball_bounce_candidates_url, "manifest.ball_bounce_candidates_url");
  }
  if (value.ball_flight_sanity_url !== null && value.ball_flight_sanity_url !== undefined) {
    manifest.ball_flight_sanity_url = readString(value.ball_flight_sanity_url, "manifest.ball_flight_sanity_url");
  }
  return manifest;
}

export function parseCoachingCardFacts(input: unknown): CoachingCardFacts {
  const value = parseMaybeJson(input);
  assertRecord(value, "coaching_card_facts");
  if (value.artifact_type !== "coaching_card_facts") {
    throw new Error("coaching_card_facts.artifact_type must be coaching_card_facts");
  }
  return {
    artifact_type: "coaching_card_facts",
    rally_scope: readString(value.rally_scope, "coaching_card_facts.rally_scope"),
    priority_rule: readArray(value.priority_rule, "coaching_card_facts.priority_rule").map((entry, index) =>
      readString(entry, `coaching_card_facts.priority_rule[${index}]`),
    ),
    facts: readArray(value.facts, "coaching_card_facts.facts").map(readCoachingCardFact),
  };
}

export function coachingTrustChipClass(trust: CoachingFactTrust): TrustBadge {
  if (trust === "ok") return "verified";
  if (trust === "estimated") return "preview";
  return "low_confidence";
}

export function parseBodyMesh(input: unknown): BodyMesh {
  const value = parseMaybeJson(input);
  assertRecord(value, "body_mesh");
  if (value.schema_version !== 1) throw new Error("body_mesh.schema_version must be 1");
  if (value.artifact_type !== "racketsport_body_mesh") {
    throw new Error("body_mesh.artifact_type must be racketsport_body_mesh");
  }
  const mesh_faces =
    value.mesh_faces === undefined
      ? []
      : readArray(value.mesh_faces, "body_mesh.mesh_faces").map((face, index) => readFace(face, `body_mesh.mesh_faces[${index}]`));
  const fps = readNumber(value.fps, "body_mesh.fps");
  const players = readArray(value.players, "body_mesh.players").map((player, index) => readBodyMeshPlayer(player, index, mesh_faces));
  assertRecord(value.summary, "body_mesh.summary");
  return {
    schema_version: 1,
    artifact_type: "racketsport_body_mesh",
    clip: readString(value.clip, "body_mesh.clip"),
    model: readString(value.model, "body_mesh.model"),
    fps,
    world_frame: readEnum(value.world_frame, "body_mesh.world_frame", ["court_Z0"] as const),
    faces_ref: readString(value.faces_ref, "body_mesh.faces_ref"),
    mesh_faces,
    joint_names: readArray(value.joint_names, "body_mesh.joint_names").map((name, index) => readString(name, `body_mesh.joint_names[${index}]`)),
    players,
    summary: {
      mesh_frame_count: readNumber(value.summary.mesh_frame_count, "body_mesh.summary.mesh_frame_count", true),
      player_count: readNumber(value.summary.player_count, "body_mesh.summary.player_count", true),
      contact_window_count: readNumber(value.summary.contact_window_count, "body_mesh.summary.contact_window_count", true),
    },
  };
}

export function parseBodyMeshFaces(input: unknown): BodyMeshFaces {
  const value = parseMaybeJson(input);
  assertRecord(value, "body_mesh_faces");
  if (value.schema_version !== 1) throw new Error("body_mesh_faces.schema_version must be 1");
  if (value.artifact_type !== "racketsport_body_mesh_faces") {
    throw new Error("body_mesh_faces.artifact_type must be racketsport_body_mesh_faces");
  }
  return {
    schema_version: 1,
    artifact_type: "racketsport_body_mesh_faces",
    faces_ref: readString(value.faces_ref, "body_mesh_faces.faces_ref"),
    mesh_faces: readArray(value.mesh_faces, "body_mesh_faces.mesh_faces").map((face, index) =>
      readFace(face, `body_mesh_faces.mesh_faces[${index}]`),
    ),
  };
}

export function parseBodyMeshIndex(input: unknown): BodyMeshIndex {
  const value = parseMaybeJson(input);
  assertRecord(value, "body_mesh_index");
  if (value.schema_version !== 1) throw new Error("body_mesh_index.schema_version must be 1");
  if (value.artifact_type !== "racketsport_body_mesh_index") {
    throw new Error("body_mesh_index.artifact_type must be racketsport_body_mesh_index");
  }
  assertRecord(value.summary, "body_mesh_index.summary");
  const windows = readArray(value.windows, "body_mesh_index.windows").map(readBodyMeshIndexWindow);
  return {
    schema_version: 1,
    artifact_type: "racketsport_body_mesh_index",
    clip: readString(value.clip, "body_mesh_index.clip"),
    model: readString(value.model, "body_mesh_index.model"),
    fps: readNumber(value.fps, "body_mesh_index.fps"),
    world_frame: readEnum(value.world_frame, "body_mesh_index.world_frame", ["court_Z0"] as const),
    faces_ref: readString(value.faces_ref, "body_mesh_index.faces_ref"),
    faces_url: readString(value.faces_url, "body_mesh_index.faces_url"),
    windows,
    summary: {
      window_count: readNonNegativeInteger(value.summary.window_count, "body_mesh_index.summary.window_count"),
      mesh_frame_count: readNonNegativeInteger(value.summary.mesh_frame_count, "body_mesh_index.summary.mesh_frame_count"),
      player_count: readNonNegativeInteger(value.summary.player_count, "body_mesh_index.summary.player_count"),
      faces_count: readNonNegativeInteger(value.summary.faces_count, "body_mesh_index.summary.faces_count"),
    },
  };
}

export function bodyMeshIndexWindowForTime(index: BodyMeshIndex | null, timeSeconds: number): BodyMeshIndexWindow | null {
  if (!index || !index.windows.length) return null;
  const tolerance = Math.max(1 / Math.max(index.fps || 30, 1) * 1.5, 0.04, BODY_MESH_FADE_SECONDS);
  const matching = index.windows.filter((window) => window.t0 - tolerance <= timeSeconds && timeSeconds <= window.t1 + tolerance);
  if (!matching.length) return null;
  return matching.reduce((best, window) => {
    const bestDistance = windowDistanceFromTime(best, timeSeconds);
    const windowDistance = windowDistanceFromTime(window, timeSeconds);
    return windowDistance < bestDistance ? window : best;
  });
}

export function decodeBodyMeshChunkBytes(
  index: BodyMeshIndex,
  window: BodyMeshIndexWindow,
  faces: BodyMeshFaces,
  bytes: ArrayBuffer,
): BodyMesh {
  const view = new DataView(bytes);
  const scale = window.quantization.scale;
  let offsetBytes = 0;
  const players: BodyMeshPlayer[] = [];
  for (const player of window.players) {
    const frames: BodyMeshFrame[] = [];
    for (const frameMeta of player.frames) {
      const mesh_vertices_world = readQuantizedVec3Array(view, offsetBytes, frameMeta.vertex_count, scale);
      offsetBytes += frameMeta.vertex_count * 3 * 2;
      const joints_world = readQuantizedVec3Array(view, offsetBytes, frameMeta.joint_count, scale);
      offsetBytes += frameMeta.joint_count * 3 * 2;
      frames.push({
        frame_idx: frameMeta.frame_idx,
        t: frameMeta.t,
        source_window_index: frameMeta.source_window_index,
        blend_weight: frameMeta.blend_weight,
        joints_world,
        joint_conf: frameMeta.joint_conf,
        mesh_vertices_world,
        mesh_faces: faces.mesh_faces,
        smplx_params: {},
        reasons: frameMeta.reasons,
        mesh_interpolated: false,
        interpolation: null,
      });
    }
    if (frames.length) players.push({ id: player.id, frames });
  }
  if (offsetBytes !== view.byteLength) {
    throw new Error(`body mesh chunk byte length mismatch: decoded ${offsetBytes}, got ${view.byteLength}`);
  }
  return {
    schema_version: 1,
    artifact_type: "racketsport_body_mesh",
    clip: index.clip,
    model: index.model,
    fps: index.fps,
    world_frame: index.world_frame,
    faces_ref: index.faces_ref,
    mesh_faces: faces.mesh_faces,
    joint_names: [],
    players,
    summary: {
      mesh_frame_count: players.reduce((total, player) => total + player.frames.length, 0),
      player_count: players.length,
      contact_window_count: index.windows.length,
    },
  };
}

const bodyMeshChunkFetchCache = new Map<string, Promise<BodyMesh>>();

export function clearBodyMeshChunkFetchCache(): void {
  bodyMeshChunkFetchCache.clear();
}

export async function decodeFetchedBodyMeshChunkBytes(
  index: BodyMeshIndex,
  window: BodyMeshIndexWindow,
  faces: BodyMeshFaces,
  bytes: ArrayBuffer,
): Promise<BodyMesh> {
  const decoded =
    window.encoding === "gzip_int16_world_vertices_v1" && bodyMeshBytesLookGzipped(bytes)
      ? await decompressGzipBytes(bytes)
      : bytes;
  return decodeBodyMeshChunkBytes(index, window, faces, decoded);
}

export async function fetchBodyMeshChunk(
  indexUrl: string,
  index: BodyMeshIndex,
  window: BodyMeshIndexWindow,
  faces: BodyMeshFaces,
): Promise<BodyMesh> {
  const resolvedUrl = resolveBodyMeshAssetUrl(indexUrl, window.url);
  const cached = bodyMeshChunkFetchCache.get(resolvedUrl);
  if (cached) return cached;
  const request = fetch(resolvedUrl)
    .then(async (response) => {
      if (!response.ok) throw new Error(`failed to fetch body mesh chunk ${resolvedUrl}: ${response.status}`);
      const encoded = await response.arrayBuffer();
      return decodeFetchedBodyMeshChunkBytes(index, window, faces, encoded);
    })
    .catch((error) => {
      bodyMeshChunkFetchCache.delete(resolvedUrl);
      throw error;
    });
  bodyMeshChunkFetchCache.set(resolvedUrl, request);
  return request;
}

export function resolveBodyMeshAssetUrl(indexUrl: string, assetUrl: string): string {
  if (/^(https?:|file:|\/)/.test(assetUrl)) return assetUrl;
  const origin = typeof window === "undefined" ? "http://localhost" : window.location.origin;
  return new URL(assetUrl, new URL(indexUrl, origin)).toString();
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

/**
 * `ball_inflections.json` is a review-only, non-schema-registered BALL cue
 * artifact (see `threed/racketsport/ball_inflections.py`). This parser is
 * intentionally lenient: it tolerates missing/extra fields and always
 * returns a (possibly empty) candidate list rather than throwing, since the
 * timeline strip treats these as low-confidence markers either way.
 */
export function parseBallInflections(input: unknown): BallInflections {
  const value = parseMaybeJson(input);
  if (!value || typeof value !== "object" || Array.isArray(value)) return { candidates: [] };
  const record = value as Record<string, unknown>;
  const rawCandidates = Array.isArray(record.candidates) ? record.candidates : [];
  const candidates: BallInflectionCandidate[] = [];
  for (const entry of rawCandidates) {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) continue;
    const candidate = entry as Record<string, unknown>;
    const timeSeconds = readOptionalPositiveOrZeroNumber(candidate.time_s);
    if (timeSeconds === null) continue;
    candidates.push({
      time_s: timeSeconds,
      frame: readOptionalPositiveOrZeroNumber(candidate.frame),
      confidence: readOptionalPositiveOrZeroNumber(candidate.confidence) ?? 0,
    });
  }
  return { candidates };
}

export function parseBallArcEventsSelected(input: unknown): BallArcEventsSelected {
  const value = parseMaybeJson(input);
  assertRecord(value, "events_selected");
  const artifactType =
    value.artifact_type === null || value.artifact_type === undefined
      ? null
      : readString(value.artifact_type, "events_selected.artifact_type");
  const selected = readArray(value.selected, "events_selected.selected")
    .map((event, index) => readBallArcSelectedEvent(event, index))
    .sort((left, right) => left.t - right.t);
  return { artifact_type: artifactType, selected };
}

export function parseReviewedBounces(input: unknown): ReviewedBounces {
  const value = parseMaybeJson(input);
  assertRecord(value, "reviewed_bounces");
  if (value.schema_version !== 1) throw new Error("reviewed_bounces.schema_version must be 1");
  if (value.artifact_type !== "racketsport_reviewed_ball_bounces") {
    throw new Error("reviewed_bounces.artifact_type must be racketsport_reviewed_ball_bounces");
  }
  return {
    schema_version: 1,
    artifact_type: "racketsport_reviewed_ball_bounces",
    source: readString(value.source, "reviewed_bounces.source"),
    bounces: readArray(value.bounces, "reviewed_bounces.bounces").map((bounce, index) => readReviewedBounce(bounce, index)),
  };
}

export function parseRallySpans(input: unknown): RallySpans {
  const value = parseMaybeJson(input);
  assertRecord(value, "rally_spans");
  if (value.schema_version !== 1) throw new Error("rally_spans.schema_version must be 1");
  if (value.artifact_type !== "racketsport_rally_spans") {
    throw new Error("rally_spans.artifact_type must be racketsport_rally_spans");
  }
  const spans = readArray(value.spans, "rally_spans.spans").map(readRallySpan);
  return {
    schema_version: 1,
    artifact_type: "racketsport_rally_spans",
    clip_id: value.clip_id === null || value.clip_id === undefined ? null : readString(value.clip_id, "rally_spans.clip_id"),
    duration_s:
      value.duration_s === null || value.duration_s === undefined ? null : readNonNegativeNumber(value.duration_s, "rally_spans.duration_s"),
    not_ground_truth: value.not_ground_truth === undefined ? true : readBoolean(value.not_ground_truth, "rally_spans.not_ground_truth"),
    spans,
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
  const parsed: VirtualWorld = {
    schema_version: 1,
    artifact_type: "racketsport_virtual_world",
    world_frame: "court_Z0",
    fps: readNumber(value.fps, "virtual_world.fps"),
    court,
    players,
    ball: readBall(value.ball),
    paddles: readArray(value.paddles, "virtual_world.paddles").map(readPaddle),
    summary,
  };
  if (value.joint_names !== undefined) {
    parsed.joint_names = readArray(value.joint_names, "virtual_world.joint_names").map((name, index) =>
      readString(name, `virtual_world.joint_names[${index}]`),
    );
  }
  return parsed;
}

export function frameForTime(player: VirtualWorldPlayer, timeSeconds: number): VirtualWorldFrame | undefined {
  if (!player.frames.length) return undefined;
  if (!isWithinFrameRange(player.frames, timeSeconds)) return undefined;
  return player.frames.reduce((best, frame) =>
    Math.abs(frame.t - timeSeconds) < Math.abs(best.t - timeSeconds) ? frame : best,
  );
}

export function ballFrameForTime(world: VirtualWorld, timeSeconds: number): VirtualWorld["ball"]["frames"][number] | undefined {
  const frames = world.ball.frames.filter((frame) => frame.visible !== false && frame.world_xyz);
  if (!frames.length) return undefined;
  if (!isWithinFrameRange(frames, timeSeconds)) return undefined;
  return frames.reduce((best, frame) => (Math.abs(frame.t - timeSeconds) < Math.abs(best.t - timeSeconds) ? frame : best));
}

export function videoBallOverlayForTime(world: VirtualWorld, timeSeconds: number): VideoBallOverlay | null {
  const frames = world.ball.frames.filter((frame) => frame.visible !== false);
  if (!frames.length) return null;
  if (!isWithinFrameRange(frames, timeSeconds)) return null;
  const frame = frames.reduce((best, candidate) =>
    Math.abs(candidate.t - timeSeconds) < Math.abs(best.t - timeSeconds) ? candidate : best,
  );
  const confidenceClass = frame.conf >= 0.8 ? "high" : frame.conf >= 0.5 ? "medium" : "low";
  const interpolated = frame.xy_interpolated === true;
  return {
    frame,
    point: frame.xy,
    confidenceClass,
    interpolated,
    opacity: interpolated ? 0.42 : confidenceClass === "low" ? 0.52 : confidenceClass === "medium" ? 0.72 : 0.92,
    radius: interpolated ? 8.5 : confidenceClass === "low" ? 6.5 : 7.25,
  };
}

export function ballRenderInfoForTime(
  world: VirtualWorld,
  timeSeconds: number,
): {
  frame?: VirtualWorld["ball"]["frames"][number];
  mode: "missing" | "calibrated_3d" | "court_plane_projection" | "off_court_projection" | "2d_only_no_3d_solve";
  render3d: boolean;
  ghost?: BallGhostMarker;
} {
  const frame = ballFrameForTime(world, timeSeconds);
  if (!frame?.world_xyz) {
    const videoFrame = videoBallOverlayForTime(world, timeSeconds)?.frame;
    const ghostPosition = videoFrame?.court_intersection_world_xyz ?? null;
    if (videoFrame && ghostPosition) {
      return {
        frame: videoFrame,
        mode: "2d_only_no_3d_solve",
        render3d: false,
        ghost: { frame: videoFrame, position: ghostPosition, label: "2D-only, no 3D solve" },
      };
    }
    return { mode: "missing", render3d: false };
  }
  if (!frame.approx) return { frame, mode: "calibrated_3d", render3d: true };
  if (!isWorldPointInsideCourt(world, frame.world_xyz, 0.35)) {
    return { frame, mode: "off_court_projection", render3d: false };
  }
  return { frame, mode: "court_plane_projection", render3d: true };
}

function isWithinFrameRange(frames: Array<{ t: number }>, timeSeconds: number): boolean {
  const times = frames.map((frame) => frame.t).sort((a, b) => a - b);
  const first = times[0];
  const last = times[times.length - 1];
  const positiveGaps = times.slice(1).map((time, index) => time - times[index]).filter((gap) => gap > 0);
  const tolerance = positiveGaps.length ? Math.min(...positiveGaps) * 1.5 : 1 / 30;
  return first - tolerance <= timeSeconds && timeSeconds <= last + tolerance;
}

function isWorldPointInsideCourt(world: VirtualWorld, point: Vec3, marginM: number): boolean {
  const courtPoints = Object.values(world.court.line_segments).flat();
  const xs = courtPoints.map((entry) => entry[0]);
  const ys = courtPoints.map((entry) => entry[1]);
  const minX = Math.min(...xs, -world.court.width_m / 2) - marginM;
  const maxX = Math.max(...xs, world.court.width_m / 2) + marginM;
  const minY = Math.min(...ys, -world.court.length_m / 2, 0) - marginM;
  const maxY = Math.max(...ys, world.court.length_m / 2, world.court.length_m) + marginM;
  return minX <= point[0] && point[0] <= maxX && minY <= point[1] && point[1] <= maxY;
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

export function solidBodyMeshFramesForTime(
  bodyMesh: BodyMesh | null,
  _contactWindows: ContactWindows | null,
  timeSeconds: number,
  world?: VirtualWorld,
): ActiveBodyMeshFrame[] {
  if (!bodyMesh) return [];
  const results: ActiveBodyMeshFrame[] = [];
  const meshPlayersById = new Map(bodyMesh.players.map((player) => [player.id, player]));
  if (world) {
    for (const worldPlayer of world.players) {
      const worldFrame = frameForTime(worldPlayer, timeSeconds);
      const meshPlayerId = normalizedMeshPlayerIdForWorldFrame(worldPlayer, worldFrame);
      const meshPlayer = meshPlayersById.get(meshPlayerId);
      if (!meshPlayer) continue;
      const frame = bodyMeshFrameForTime(meshPlayer, timeSeconds, bodyMesh.fps);
      if (!solidMeshFrameRenderable(frame)) continue;
      const aligned = alignBodyMeshFrameToWorldSkeleton(frame, worldFrame, bodyMesh.joint_names, world.joint_names);
      results.push({
        playerId: worldPlayer.id,
        meshPlayerId: meshPlayer.id,
        frame: aligned.frame,
        presenceOpacity: bodyMeshPresenceOpacityForTime(meshPlayer, timeSeconds),
        renderTranslation: aligned.renderTranslation,
        alignmentDebug: aligned.debug,
      });
    }
    return results;
  }
  for (const player of bodyMesh.players) {
    const frame = bodyMeshFrameForTime(player, timeSeconds, bodyMesh.fps);
    if (!solidMeshFrameRenderable(frame)) continue;
    results.push({
      playerId: player.id,
      meshPlayerId: player.id,
      frame,
      presenceOpacity: bodyMeshPresenceOpacityForTime(player, timeSeconds),
      renderTranslation: [0, 0, 0],
    });
  }
  return results;
}

const BODY_MESH_FADE_SECONDS = 0.12;

export function activePaddleFramesForTime(world: VirtualWorld, timeSeconds: number): ActivePaddleFrame[] {
  return world.paddles
    .map((paddle) => {
      const frame = paddleFrameForTime(paddle, timeSeconds, world.fps);
      if (!frame) return null;
      return {
        playerId: paddle.player_id,
        paddle,
        frame,
        estimated: isEstimatedPaddleFrame(paddle, frame),
      };
    })
    .filter((entry): entry is ActivePaddleFrame => entry !== null);
}

function isEstimatedPaddleFrame(paddle: VirtualWorldPaddle, frame: VirtualWorldPaddleFrame): boolean {
  return (
    frame.source.includes("wrist_proxy") ||
    frame.render_only === true ||
    frame.not_for_detection_metrics === true ||
    paddle.trust_band?.badge !== "verified" ||
    frame.trust_band?.badge !== "verified"
  );
}

export function solidMeshRenderedPlayerCount(activeBodyMeshes: ActiveBodyMeshFrame[]): number {
  return activeBodyMeshes.length;
}

export function bodyMeshStatusTileValue(renderedSolidMeshPlayers: number, status: BodyMeshLoadStatus): number | string {
  if (status.state === "fetch_failed" || status.state === "parse_failed" || status.state === "decode_failed") return status.label;
  if (status.state === "loading" || status.state === "no_window") return status.label;
  return renderedSolidMeshPlayers;
}

export function bodyMeshDebugSnapshot({
  bodyMeshIndex,
  bodyMesh,
  world,
  currentTime,
  loadStatus,
  activeBodyMeshes,
}: {
  bodyMeshIndex: BodyMeshIndex | null;
  bodyMesh: BodyMesh | null;
  world: VirtualWorld;
  currentTime: number;
  loadStatus: BodyMeshLoadStatus;
  activeBodyMeshes?: ActiveBodyMeshFrame[];
}): BodyMeshDebugSnapshot {
  const activeWindow = bodyMeshIndexWindowForTime(bodyMeshIndex, currentTime);
  const meshPlayersById = new Map((bodyMesh?.players ?? []).map((player) => [player.id, player]));
  const active = activeBodyMeshes ?? solidBodyMeshFramesForTime(bodyMesh, null, currentTime, world);
  const rendered = new Set(active.map((frame) => `${frame.playerId}:${frame.meshPlayerId}:${frame.frame.frame_idx}`));
  return {
    current_time: currentTime,
    active_window_id: activeWindow?.source_window_index ?? null,
    active_window_url: activeWindow?.url ?? null,
    load_state: loadStatus.state,
    load_stage: loadStatus.stage ?? null,
    load_url: loadStatus.url ?? null,
    load_message: loadStatus.message ?? null,
    rendered_player_count: active.length,
    alignment_floor_guard_count: active.filter((frame) => frame.alignmentDebug?.floor_guard_applied === true).length,
    players: world.players.map((worldPlayer) => {
      const worldFrame = frameForTime(worldPlayer, currentTime);
      const normalizedMeshPlayerId = normalizedMeshPlayerIdForWorldFrame(worldPlayer, worldFrame);
      const meshPlayer = meshPlayersById.get(normalizedMeshPlayerId);
      const meshFrame = meshPlayer ? bodyMeshFrameForTime(meshPlayer, currentTime, bodyMesh?.fps ?? world.fps) : undefined;
      const renderedKey = meshFrame ? `${worldPlayer.id}:${normalizedMeshPlayerId}:${meshFrame.frame_idx}` : null;
      return {
        world_player_id: worldPlayer.id,
        world_frame_t: worldFrame?.t ?? null,
        mesh_ref_player_id: worldFrame?.mesh_ref?.player_id ?? null,
        mesh_ref_frame_idx: worldFrame?.mesh_ref?.frame_idx ?? null,
        normalized_mesh_player_id: normalizedMeshPlayerId,
        mesh_player_present: Boolean(meshPlayer),
        mesh_frame_present: Boolean(renderedKey && rendered.has(renderedKey)),
        mesh_frame_idx: renderedKey && rendered.has(renderedKey) && meshFrame ? meshFrame.frame_idx : null,
      };
    }),
  };
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

// `worldStats().meshFrames` reads virtual_world.json's own embedded
// mesh_vertices_world point-cloud path (populated only when `--smpl-motion`
// is fed directly into build_scrubber_v0_world.py). The solid indexed-mesh
// path added for W3-REPLAY-NATIVE ships as a separate `body_mesh.json`
// artifact (see `body_mesh_url`/`SolidBodyMesh`), so it needs its own,
// honestly-separate frame count rather than silently reusing (or leaving at
// 0 next to) the unrelated point-cloud counter.
export function solidMeshFrameCount(bodyMesh: BodyMesh | null): number {
  return bodyMesh?.summary.mesh_frame_count ?? 0;
}

export function playerCoverageStats(world: VirtualWorld): {
  firstTime: number | null;
  lastTime: number | null;
  playerCount: number;
  coveredFrameCount: number;
} {
  const times = world.players.flatMap((player) => player.frames.map((frame) => frame.t));
  if (!times.length) {
    return { firstTime: null, lastTime: null, playerCount: world.players.length, coveredFrameCount: 0 };
  }
  return {
    firstTime: Math.min(...times),
    lastTime: Math.max(...times),
    playerCount: world.players.length,
    coveredFrameCount: times.length,
  };
}

export function worldWarningsReadout(world: Pick<VirtualWorld, "summary">): string {
  const warnings = world.summary.warnings;
  if (!warnings.length) return "0 notices";
  const readable = warnings.map(friendlyWorldWarning);
  const suffix = warnings.length > 2 ? `, +${warnings.length - 2} more` : "";
  return `${warnings.length} notice${warnings.length === 1 ? "" : "s"}: ${readable.slice(0, 2).join(", ")}${suffix}`;
}

function friendlyWorldWarning(warning: string): string {
  if (warning === "unprojected_visible_ball_frames") return "2D-only ball frames outside solved arc coverage";
  if (warning === "missing_paddle_pose") return "missing paddle pose";
  if (warning === "missing_mesh_vertices") return "missing mesh vertices";
  return warning.replaceAll("_", " ");
}

export function entityCoverageReadout(label: string, entity: EntityCoverage | null | undefined): string {
  const coverage = entity?.coverage_fraction;
  const minT = entity?.min_t;
  const maxT = entity?.max_t;
  if (typeof coverage !== "number" || !Number.isFinite(coverage)) return `${label} coverage n/a`;
  const coverageText = `${(coverage * 100).toFixed(1)}%`;
  if (typeof minT === "number" && Number.isFinite(minT) && typeof maxT === "number" && Number.isFinite(maxT)) {
    return `${label} ${coverageText} / ${minT.toFixed(2)}-${maxT.toFixed(2)}s`;
  }
  return `${label} ${coverageText}`;
}

export function ballCoverageKpiReadout(world: Pick<VirtualWorld, "ball">): string {
  const frames = world.ball.frames;
  if (!frames.length) return "coverage n/a";
  const counts = frames.reduce(
    (acc, frame) => {
      acc[ballFrameCoverageBucket(frame)] += 1;
      return acc;
    },
    { measured: 0, predicted: 0, hidden: 0 },
  );
  if (counts.measured === 0 && counts.predicted === 0 && counts.hidden === 0) return "coverage n/a";
  return `${counts.measured}/${frames.length} measured · ${counts.predicted} predicted · ${counts.hidden} hidden`;
}

function ballFrameCoverageBucket(frame: VirtualWorldBallFrame): "measured" | "predicted" | "hidden" {
  const rawBand = (frame.confidence_provenance?.display_band ?? frame.confidence_provenance?.band ?? "").toLowerCase();
  if (!frame.world_xyz || rawBand.startsWith("hidden") || rawBand.includes("no_prediction") || rawBand.includes("no_anchor")) {
    return "hidden";
  }
  if (rawBand === "measured" || rawBand === "anchored_measured") return "measured";
  if (rawBand.includes("predicted") || rawBand.startsWith("physics") || rawBand.startsWith("arc_")) return "predicted";
  return frame.approx ? "predicted" : "measured";
}

const TRUST_BADGE_COLORS: Record<TrustBadge, string> = {
  verified: "#6cb2ff",
  preview: "#ffb454",
  low_confidence: "#8a8f98",
};

/**
 * Effective trust badge for styling purposes. Missing/null trust-band provenance must
 * fail closed to "low_confidence", never "verified" -- an entity with no gate provenance
 * (e.g. a `virtual_world.json` built by a lane/builder that has not wired trust bands
 * through yet) has not been verified by anything, so it must render as the least-trusted
 * state, not the default (most-trusted) one.
 */
export function effectiveTrustBadge(trustBand: TrustBand | null | undefined): TrustBadge {
  return trustBand?.badge ?? "low_confidence";
}

/** Color for a trust badge, used to grey out (low_confidence) or amber-flag (preview) scrubber entities.
 * Missing/null badges fail closed to the low_confidence color -- never verified. */
export function trustBadgeColor(badge: TrustBadge | null | undefined): string {
  return TRUST_BADGE_COLORS[badge ?? "low_confidence"];
}

/** Short "STAGE: badge" chip text for a trust band, or a neutral placeholder when absent. */
export function trustBandChipText(trustBand: TrustBand | null | undefined): string {
  if (!trustBand) return "no trust band";
  return `${trustBand.stage}: ${trustBand.badge.replace("_", " ")}`;
}

/**
 * Classify one contact-window event into a trust badge for timeline coloring.
 * There is no dedicated CONTACT gate in the gate ladder, so this reads the
 * event's own provenance: a human-reviewed contact (sources.human_review set)
 * is a "preview"-grade marker; anything else falls back to its numeric
 * confidence, and low-confidence detector-only events are "low_confidence".
 */
export function contactEventBadge(event: ContactWindowEvent): TrustBadge {
  if (typeof event.sources.human_review === "number" && event.sources.human_review >= 0.5) return "preview";
  return event.confidence >= 0.75 ? "preview" : "low_confidence";
}

/**
 * BALL has 0/8 milestone gates passing today (MASTER_PLAN.md), so every ball
 * inflection/bounce candidate is low_confidence regardless of its own
 * reported confidence -- coloring markers by a fabricated per-event tier
 * would overstate BALL's actual gate state.
 */
export function ballInflectionBadge(): TrustBadge {
  return "low_confidence";
}

/** Build scrub-linked timeline markers from contact-window and ball-inflection cues. */
export function timelineMarkersFromArtifacts(
  contactWindows: ContactWindows | null,
  ballInflections: BallInflections | null,
  reviewedBounces: ReviewedBounces | null = null,
): TimelineMarker[] {
  const markers: TimelineMarker[] = [];
  for (const event of contactWindows?.events ?? []) {
    markers.push({
      kind: event.type,
      t: event.t,
      t0: event.window.t0,
      t1: event.window.t1,
      confidence: event.confidence,
      badge: contactEventBadge(event),
      label: `${event.type}${event.player_id !== null ? ` (p${event.player_id})` : ""} @ ${event.t.toFixed(2)}s`,
    });
  }
  for (const candidate of ballInflections?.candidates ?? []) {
    markers.push({
      kind: "ball_inflection",
      t: candidate.time_s,
      t0: candidate.time_s,
      t1: candidate.time_s,
      confidence: candidate.confidence,
      badge: ballInflectionBadge(),
      label: `ball inflection @ ${candidate.time_s.toFixed(2)}s`,
    });
  }
  for (const bounce of reviewedBounces?.bounces ?? []) {
    markers.push({
      kind: "reviewed_bounce",
      t: bounce.t,
      t0: bounce.t,
      t1: bounce.t,
      confidence: 1,
      badge: "preview",
      label: `reviewed bounce ${bounce.review_id} @ ${bounce.t.toFixed(2)}s`,
      humanReviewed: true,
    });
  }
  return markers.sort((left, right) => left.t - right.t);
}

export function timelineEventJump(
  markers: TimelineMarker[],
  currentTime: number,
  direction: "previous" | "next",
): number | null {
  const sorted = markers.map((marker) => marker.t).sort((left, right) => left - right);
  if (!sorted.length) return null;
  const epsilon = 1e-6;
  if (direction === "next") {
    return sorted.find((time) => time > currentTime + epsilon) ?? null;
  }
  const previous = sorted.filter((time) => time < currentTime - epsilon);
  return previous.length ? previous[previous.length - 1] : null;
}

export function timelineChaptersFromMarkers(
  markers: TimelineMarker[],
  durationSeconds: number,
  gapSeconds = 2,
): TimelineChapter[] {
  const sorted = [...markers].sort((left, right) => left.t - right.t);
  if (!sorted.length) return [];
  const duration = durationSeconds > 0 ? durationSeconds : sorted[sorted.length - 1].t1;
  const chapters: TimelineChapter[] = [];
  let group: TimelineMarker[] = [sorted[0]];
  for (const marker of sorted.slice(1)) {
    const previous = group[group.length - 1];
    if (marker.t - previous.t > gapSeconds) {
      chapters.push(chapterFromMarkers(chapters.length + 1, group, duration));
      group = [marker];
    } else {
      group.push(marker);
    }
  }
  chapters.push(chapterFromMarkers(chapters.length + 1, group, duration));
  return chapters;
}

export function timelineChaptersFromRallySpans(rallySpans: RallySpans | null): TimelineChapter[] {
  if (!rallySpans) return [];
  return [...rallySpans.spans]
    .sort((left, right) => left.t0 - right.t0 || left.t1 - right.t1)
    .map((span, index) => ({
      index: index + 1,
      rallyId: span.rallyId,
      t0: span.t0,
      t1: span.t1,
      label: labelForRallyId(span.rallyId, index + 1),
      badge: rallySpans.not_ground_truth ? "preview" : "verified",
    }));
}

function labelForRallyId(rallyId: string, fallbackIndex: number): string {
  const match = rallyId.match(/^rally_(\d+)$/);
  if (match) return `Rally ${match[1]}`;
  return `Rally ${fallbackIndex}`;
}

function chapterFromMarkers(index: number, markers: TimelineMarker[], durationSeconds: number): TimelineChapter {
  const t0 = Math.max(0, Math.min(...markers.map((marker) => marker.t0)));
  const t1 = Math.min(durationSeconds, Math.max(...markers.map((marker) => marker.t1)));
  const badge = markers.some((marker) => marker.badge === "low_confidence")
    ? "low_confidence"
    : markers.some((marker) => marker.badge === "preview")
      ? "preview"
      : "verified";
  return {
    index,
    t0,
    t1: Math.max(t0, t1),
    label: `Rally ${index}`,
    badge,
  };
}

/** A single derived rally span (t0..t1) bounding every contact-window event. */
export function rallySpanFromContactWindows(contactWindows: ContactWindows | null): { t0: number; t1: number } | null {
  if (!contactWindows || !contactWindows.events.length) return null;
  const times = contactWindows.events.map((event) => event.t);
  return { t0: Math.min(...times), t1: Math.max(...times) };
}

/**
 * Sum of consecutive floor-position displacement (meters) for one player
 * within [t0, t1]. This is a world-frame-meter metric, so callers must
 * surface it alongside the CAL/BODY trust band's world-scale caveat rather
 * than presenting it as a verified number.
 */
export function playerCoverageDistanceM(player: VirtualWorldPlayer, t0: number, t1: number): number {
  const points = player.frames
    .filter((frame) => frame.t >= t0 && frame.t <= t1)
    .map((frame) => frame.floor_world_xyz ?? (frame.track_world_xy ? ([frame.track_world_xy[0], frame.track_world_xy[1], 0] as Vec3) : null))
    .filter((point): point is Vec3 => point !== null);
  let total = 0;
  for (let index = 1; index < points.length; index += 1) {
    total += Math.hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1]);
  }
  return total;
}

export function ballTrailPointsForTime(world: VirtualWorld, currentTime: number, lookbackSeconds = 0.7): BallTrailPoint[] {
  const t0 = currentTime - Math.max(0, lookbackSeconds);
  return world.ball.frames
    .filter((frame) => frame.visible !== false && frame.world_xyz && t0 <= frame.t && frame.t <= currentTime)
    .map((frame) => {
      const point = frame.world_xyz as Vec3;
      const insideCourt = isWorldPointInsideCourt(world, point, 0.35);
      return ballTrailPointFromFrame(frame, point, insideCourt);
    });
}

export function ballTrailSegmentsForTime(
  world: VirtualWorld,
  currentTime: number,
  lookbackSeconds = 0.7,
  eventsSelected: BallArcEventsSelected | null = null,
): BallTrailSegment[] {
  const runs = ballTrailRunsForTime(world, currentTime, lookbackSeconds, eventsSelected);
  const segments: BallTrailSegment[] = [];
  for (const run of runs) {
    const points = smoothBallTrailRun(run);
    for (let index = 1; index < points.length; index += 1) {
      const from = points[index - 1];
      const to = points[index];
      const courtStyle = from.courtStyle === "outside_court" || to.courtStyle === "outside_court" ? "outside_court" : "inside_court";
      segments.push({
        from: from.point,
        to: to.point,
        courtStyle,
        opacityScale: Math.min(from.opacityScale, to.opacityScale, courtStyle === "outside_court" ? 0.6 : 1),
        thicknessScale: Math.min(from.thicknessScale, to.thicknessScale, courtStyle === "outside_court" ? 0.75 : 1),
      });
    }
  }
  return segments;
}

type BallTrailRunPoint = BallTrailPoint & {
  t: number;
  segmentKey: string | null;
};

function ballTrailRunsForTime(
  world: VirtualWorld,
  currentTime: number,
  lookbackSeconds: number,
  eventsSelected: BallArcEventsSelected | null,
): BallTrailRunPoint[][] {
  const t0 = currentTime - Math.max(0, lookbackSeconds);
  const frames = world.ball.frames
    .filter((frame) => t0 <= frame.t && frame.t <= currentTime)
    .slice()
    .sort((left, right) => left.t - right.t);
  const runs: BallTrailRunPoint[][] = [];
  let currentRun: BallTrailRunPoint[] = [];
  let previousFrame: VirtualWorldBallFrame | null = null;
  let previousPoint: BallTrailRunPoint | null = null;

  for (const frame of frames) {
    const point = ballTrailRunPointFromFrame(world, frame);
    if (!point) {
      if (currentRun.length) runs.push(currentRun);
      currentRun = [];
      previousFrame = frame;
      previousPoint = null;
      continue;
    }

    if (
      previousPoint &&
      previousFrame &&
      (hasSelectedArcBoundaryBetween(previousFrame.t, frame.t, eventsSelected) || previousPoint.segmentKey !== point.segmentKey)
    ) {
      if (currentRun.length) runs.push(currentRun);
      currentRun = [];
    }

    currentRun.push(point);
    previousFrame = frame;
    previousPoint = point;
  }

  if (currentRun.length) runs.push(currentRun);
  return runs.filter((run) => run.length >= 2);
}

function ballTrailRunPointFromFrame(world: VirtualWorld, frame: VirtualWorldBallFrame): BallTrailRunPoint | null {
  if (frame.visible === false || !frame.world_xyz) return null;
  const point = frame.world_xyz;
  const insideCourt = isWorldPointInsideCourt(world, point, 0.35);
  return {
    ...ballTrailPointFromFrame(frame, point, insideCourt),
    t: frame.t,
    segmentKey: ballFrameSegmentKey(frame),
  };
}

function ballFrameSegmentKey(frame: VirtualWorldBallFrame): string | null {
  if (frame.arc_segment_id !== null && frame.arc_segment_id !== undefined) return `arc:${String(frame.arc_segment_id)}`;
  const predictor = frame.confidence_provenance?.predictor ?? null;
  const band = frame.confidence_provenance?.band ?? frame.confidence_provenance?.display_band ?? null;
  if (predictor && predictor.includes("arc") && band) return `${predictor}:${band}`;
  return null;
}

function hasSelectedArcBoundaryBetween(
  previousTime: number,
  nextTime: number,
  eventsSelected: BallArcEventsSelected | null,
): boolean {
  if (!eventsSelected?.selected.length) return false;
  const low = Math.min(previousTime, nextTime);
  const high = Math.max(previousTime, nextTime);
  return eventsSelected.selected.some((event) => low < event.t && event.t <= high);
}

function smoothBallTrailRun(run: BallTrailRunPoint[]): BallTrailRunPoint[] {
  if (run.length < 4) return run;
  const points: BallTrailRunPoint[] = [];
  const stepsPerPair = 4;
  for (let index = 0; index < run.length - 1; index += 1) {
    const p0 = run[Math.max(0, index - 1)];
    const p1 = run[index];
    const p2 = run[index + 1];
    const p3 = run[Math.min(run.length - 1, index + 2)];
    if (index === 0) points.push(p1);
    for (let step = 1; step <= stepsPerPair; step += 1) {
      const t = step / stepsPerPair;
      points.push({
        ...interpolatedTrailStyle(p1, p2, t),
        point: catmullRomVec3(p0.point, p1.point, p2.point, p3.point, t),
        t: p1.t + (p2.t - p1.t) * t,
        segmentKey: p1.segmentKey === p2.segmentKey ? p1.segmentKey : null,
      });
    }
  }
  return points;
}

function interpolatedTrailStyle(from: BallTrailRunPoint, to: BallTrailRunPoint, t: number): Omit<BallTrailRunPoint, "point" | "t" | "segmentKey"> {
  const courtStyle = from.courtStyle === "outside_court" || to.courtStyle === "outside_court" ? "outside_court" : "inside_court";
  return {
    courtStyle,
    uncertaintySigmaM:
      from.uncertaintySigmaM === null && to.uncertaintySigmaM === null
        ? null
        : (from.uncertaintySigmaM ?? 0) + ((to.uncertaintySigmaM ?? 0) - (from.uncertaintySigmaM ?? 0)) * t,
    opacityScale: from.opacityScale + (to.opacityScale - from.opacityScale) * t,
    thicknessScale: from.thicknessScale + (to.thicknessScale - from.thicknessScale) * t,
  };
}

function catmullRomVec3(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: number): Vec3 {
  return [0, 1, 2].map((axis) => catmullRomScalar(p0[axis], p1[axis], p2[axis], p3[axis], t)) as Vec3;
}

function catmullRomScalar(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const t2 = t * t;
  const t3 = t2 * t;
  return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
}

function ballTrailPointFromFrame(frame: VirtualWorldBallFrame, point: Vec3, insideCourt: boolean): BallTrailPoint {
  const uncertaintySigmaM = ballFrameUncertaintySigmaM(frame);
  const uncertaintyStyle = ballTrailStyleForSigma(uncertaintySigmaM);
  return {
    point,
    courtStyle: insideCourt ? "inside_court" : "outside_court",
    uncertaintySigmaM,
    opacityScale: Math.min(uncertaintyStyle.opacityScale, insideCourt ? 1 : 0.6),
    thicknessScale: Math.min(uncertaintyStyle.thicknessScale, insideCourt ? 1 : 0.75),
  };
}

function ballFrameUncertaintySigmaM(frame: VirtualWorldBallFrame): number | null {
  const provenanceSigma = frame.confidence_provenance?.predicted_sigma_m;
  if (typeof provenanceSigma === "number" && Number.isFinite(provenanceSigma) && provenanceSigma >= 0) return provenanceSigma;
  const physicsSigma = frame.physics_fill?.uncertainty_m;
  if (typeof physicsSigma === "number" && Number.isFinite(physicsSigma) && physicsSigma >= 0) return physicsSigma;
  return null;
}

function ballTrailStyleForSigma(sigmaM: number | null): { opacityScale: number; thicknessScale: number } {
  if (sigmaM === null) return { opacityScale: 1, thicknessScale: 1 };
  if (sigmaM >= 0.4) return { opacityScale: 0.35, thicknessScale: 0.55 };
  if (sigmaM >= 0.2) return { opacityScale: 0.55, thicknessScale: 0.7 };
  if (sigmaM >= 0.1) return { opacityScale: 0.75, thicknessScale: 0.85 };
  return { opacityScale: 0.9, thicknessScale: 0.95 };
}

export function playerTrailPointsForTime(player: VirtualWorldPlayer, currentTime: number, lookbackSeconds = 1.2): Vec3[] {
  const t0 = currentTime - Math.max(0, lookbackSeconds);
  return player.frames
    .filter((frame) => t0 <= frame.t && frame.t <= currentTime)
    .map((frame) => frame.floor_world_xyz ?? (frame.track_world_xy ? ([frame.track_world_xy[0], frame.track_world_xy[1], 0] as Vec3) : null))
    .filter((point): point is Vec3 => point !== null);
}

export function parseLabelOverlayPayload(input: unknown): LabelOverlayPayload {
  const value = parseMaybeJson(input);
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return emptyLabelOverlayPayload();
  }
  const record = value as Record<string, unknown>;
  const annotation = record.annotation;
  const items =
    annotation && typeof annotation === "object" && !Array.isArray(annotation)
      ? (annotation as { items?: unknown }).items
      : null;
  const labelItems = Array.isArray(items)
    ? items.filter((item): item is LabelItem => typeof item === "object" && item !== null)
    : [];
  const annotationMeta =
    annotation && typeof annotation === "object" && !Array.isArray(annotation) ? (annotation as Record<string, unknown>) : {};
  const frameMeta = record.frames && typeof record.frames === "object" && !Array.isArray(record.frames)
    ? (record.frames as Record<string, unknown>)
    : {};
  const sourceFps = readOptionalPositiveNumber(frameMeta.source_fps) ?? readOptionalPositiveNumber(frameMeta.frame_rate_fps) ?? 30;
  const sampleEveryFrames = readOptionalPositiveNumber(frameMeta.sample_every_frames) ?? 1;
  const inferredSize = inferLabelSourceSize(record, frameMeta, annotationMeta);
  return {
    items: labelItems,
    notGroundTruth: record.not_ground_truth === true,
    status: typeof record.status === "string" ? record.status : null,
    sourceWidth: inferredSize[0],
    sourceHeight: inferredSize[1],
    secondsPerFrame: sampleEveryFrames / sourceFps,
  };
}

export function labelOverlayForTime(labelOverlay: LabelOverlayPayload, currentTime: number): LabelItem[] {
  if (!labelOverlay.items.length) return [];
  const secondsPerFrame = labelOverlay.secondsPerFrame > 0 ? labelOverlay.secondsPerFrame : 1 / 30;
  const frameIndex = Math.max(0, Math.round(currentTime / secondsPerFrame));
  return labelOverlay.items.filter((item) => labelFrameIndex(item.frame) === frameIndex).slice(0, 8);
}

export function labelViewBox(labelOverlay: LabelOverlayPayload): string {
  return `0 0 ${Math.ceil(labelOverlay.sourceWidth)} ${Math.ceil(labelOverlay.sourceHeight)}`;
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

function readCoachingCardFact(input: unknown, index: number): CoachingCardFact {
  const path = `coaching_card_facts.facts[${index}]`;
  assertRecord(input, path);
  return {
    coverage_fraction: readUnitNumber(input.coverage_fraction, `${path}.coverage_fraction`),
    frames_total: readNonNegativeInteger(input.frames_total, `${path}.frames_total`),
    frames_used: readNonNegativeInteger(input.frames_used, `${path}.frames_used`),
    metric: readString(input.metric, `${path}.metric`),
    player_id: readString(input.player_id, `${path}.player_id`),
    rally_id: readString(input.rally_id, `${path}.rally_id`),
    rally_scope: readString(input.rally_scope, `${path}.rally_scope`),
    trust: readEnum(input.trust, `${path}.trust`, ["ok", "estimated", "unverified_cue"] as const),
    unit: readString(input.unit, `${path}.unit`),
    value: readNumber(input.value, `${path}.value`),
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

function readBodyMeshPlayer(input: unknown, index: number, artifactFaces: MeshFace[]): BodyMeshPlayer {
  const path = `body_mesh.players[${index}]`;
  assertRecord(input, path);
  const rawId = input.id === undefined || input.id === null ? input.player_id : input.id;
  return {
    id: readPlayerId(rawId, `${path}.id`),
    frames: readArray(input.frames, `${path}.frames`).map((frame, frameIndex) =>
      readBodyMeshFrame(frame, `${path}.frames[${frameIndex}]`, artifactFaces),
    ),
  };
}

function readBodyMeshIndexWindow(input: unknown, index: number): BodyMeshIndexWindow {
  const path = `body_mesh_index.windows[${index}]`;
  assertRecord(input, path);
  assertRecord(input.quantization, `${path}.quantization`);
  const sourceWindowIndex = readNonNegativeInteger(input.source_window_index, `${path}.source_window_index`);
  return {
    source_window_index: sourceWindowIndex,
    frame_start: readNonNegativeInteger(input.frame_start, `${path}.frame_start`),
    frame_end: readNonNegativeInteger(input.frame_end, `${path}.frame_end`),
    t0: readNonNegativeNumber(input.t0, `${path}.t0`),
    t1: readNonNegativeNumber(input.t1, `${path}.t1`),
    frame_count: readNonNegativeInteger(input.frame_count, `${path}.frame_count`),
    player_frame_count:
      input.player_frame_count === undefined
        ? 0
        : readNonNegativeInteger(input.player_frame_count, `${path}.player_frame_count`),
    target_player_ids: readArray(input.target_player_ids, `${path}.target_player_ids`).map((playerId, playerIndex) =>
      readPlayerId(playerId, `${path}.target_player_ids[${playerIndex}]`),
    ),
    player_ids: readArray(input.player_ids, `${path}.player_ids`).map((playerId, playerIndex) =>
      readPlayerId(playerId, `${path}.player_ids[${playerIndex}]`),
    ),
    target_representation:
      input.target_representation === undefined
        ? "world_mesh"
        : readString(input.target_representation, `${path}.target_representation`),
    fallback_representation:
      input.fallback_representation === undefined
        ? "lane_a_skeleton"
        : readString(input.fallback_representation, `${path}.fallback_representation`),
    reason_counts: readNumberRecord(input.reason_counts, `${path}.reason_counts`),
    max_score: input.max_score === undefined ? 0 : readNumber(input.max_score, `${path}.max_score`),
    url: readString(input.url, `${path}.url`),
    byte_size: readNonNegativeInteger(input.byte_size, `${path}.byte_size`),
    encoding: readEnum(input.encoding, `${path}.encoding`, ["gzip_int16_world_vertices_v1", "raw_int16_world_vertices_v1"] as const),
    quantization: {
      scale: readPositiveNumber(input.quantization.scale, `${path}.quantization.scale`),
      unit: readEnum(input.quantization.unit, `${path}.quantization.unit`, ["m"] as const),
    },
    players: readArray(input.players, `${path}.players`).map((player, playerIndex) =>
      readBodyMeshIndexPlayer(player, `${path}.players[${playerIndex}]`, sourceWindowIndex),
    ),
  };
}

function readBodyMeshIndexPlayer(input: unknown, path: string, sourceWindowIndex: number): BodyMeshIndexPlayer {
  assertRecord(input, path);
  const rawId = input.id === undefined || input.id === null ? input.player_id : input.id;
  return {
    id: readPlayerId(rawId, `${path}.id`),
    frames: readArray(input.frames, `${path}.frames`).map((frame, frameIndex) =>
      readBodyMeshIndexFrame(frame, `${path}.frames[${frameIndex}]`, sourceWindowIndex),
    ),
  };
}

function readBodyMeshIndexFrame(input: unknown, path: string, defaultSourceWindowIndex: number): BodyMeshIndexFrame {
  assertRecord(input, path);
  return {
    frame_idx: readNonNegativeInteger(input.frame_idx, `${path}.frame_idx`),
    t: readNonNegativeNumber(input.t, `${path}.t`),
    source_window_index:
      input.source_window_index === null || input.source_window_index === undefined
        ? defaultSourceWindowIndex
        : readNonNegativeInteger(input.source_window_index, `${path}.source_window_index`),
    blend_weight: input.blend_weight === undefined ? 1 : readUnitNumber(input.blend_weight, `${path}.blend_weight`),
    vertex_count: readNonNegativeInteger(input.vertex_count, `${path}.vertex_count`),
    joint_count: input.joint_count === undefined ? 0 : readNonNegativeInteger(input.joint_count, `${path}.joint_count`),
    joint_conf:
      input.joint_conf === undefined
        ? []
        : readArray(input.joint_conf, `${path}.joint_conf`).map((confidence, index) => readNumber(confidence, `${path}.joint_conf[${index}]`)),
    reasons:
      input.reasons === undefined
        ? []
        : readArray(input.reasons, `${path}.reasons`).map((reason, index) => readString(reason, `${path}.reasons[${index}]`)),
  };
}

function readBodyMeshFrame(input: unknown, path: string, artifactFaces: MeshFace[]): BodyMeshFrame {
  assertRecord(input, path);
  const vertices = readArray(input.mesh_vertices_world, `${path}.mesh_vertices_world`).map((point, index) =>
    readVec3(point, `${path}.mesh_vertices_world[${index}]`),
  );
  const frameFaces =
    input.mesh_faces === undefined
      ? artifactFaces
      : readArray(input.mesh_faces, `${path}.mesh_faces`).map((face, index) => readFace(face, `${path}.mesh_faces[${index}]`));
  validateFacesForVertices(frameFaces, vertices.length, `${path}.mesh_faces`);
  return {
    frame_idx: readNonNegativeInteger(input.frame_idx, `${path}.frame_idx`),
    t: readNonNegativeNumber(input.t, `${path}.t`),
    source_window_index:
      input.source_window_index === null || input.source_window_index === undefined
        ? null
        : readNonNegativeInteger(input.source_window_index, `${path}.source_window_index`),
    blend_weight: input.blend_weight === undefined ? 1 : readUnitNumber(input.blend_weight, `${path}.blend_weight`),
    joints_world:
      input.joints_world === undefined
        ? []
        : readArray(input.joints_world, `${path}.joints_world`).map((point, index) => readVec3(point, `${path}.joints_world[${index}]`)),
    joint_conf:
      input.joint_conf === undefined
        ? []
        : readArray(input.joint_conf, `${path}.joint_conf`).map((confidence, index) => readNumber(confidence, `${path}.joint_conf[${index}]`)),
    mesh_vertices_world: vertices,
    mesh_faces: frameFaces,
    smplx_params: readSmplxParams(input.smplx_params, `${path}.smplx_params`),
    reasons:
      input.reasons === undefined
        ? []
        : readArray(input.reasons, `${path}.reasons`).map((reason, index) => readString(reason, `${path}.reasons[${index}]`)),
    mesh_interpolated: false,
    interpolation: null,
  };
}

function readSmplxParams(input: unknown, path: string): Record<string, number[]> {
  assertRecord(input, path);
  const params: Record<string, number[]> = {};
  for (const [key, value] of Object.entries(input)) {
    params[key] = readArray(value, `${path}.${key}`).map((entry, index) => readNumber(entry, `${path}.${key}[${index}]`));
  }
  return params;
}

const BODY_MESH_INTERPOLATION_MAX_GAP_SECONDS = 0.066;
const BODY_MESH_HOLD_MAX_GAP_SECONDS = 0.15;
const DISPLAY_FPS_MESH_MAX_GAP_SECONDS = 0.15;
const MESH_INTERPOLATION_REASON = "client_midpoint_interpolation";
const DISPLAY_FPS_INTERPOLATION_REASON = "user_enabled_2x_display";

type BodyMeshInterpolationPair = {
  from: BodyMeshFrame;
  to: BodyMeshFrame;
  key: string;
  gapSeconds: number;
  sameWindow: boolean;
  sameRun: boolean;
  holdEligible: boolean;
  matchingVertexCount: boolean;
  matchingJointCount: boolean;
  eligible: boolean;
};

type BodyMeshInterpolationState = {
  sortedFrames: BodyMeshFrame[];
  pairs: BodyMeshInterpolationPair[];
  reusableFrames: Map<string, BodyMeshFrame>;
};

const bodyMeshInterpolationStateCache = new WeakMap<BodyMeshPlayer, BodyMeshInterpolationState>();

function bodyMeshFrameForTime(player: BodyMeshPlayer, timeSeconds: number, fps: number): BodyMeshFrame | undefined {
  if (!player.frames.length) return undefined;
  const state = bodyMeshInterpolationStateForPlayer(player);
  const sortedFrames = state.sortedFrames;
  const tolerance = Math.max(1 / Math.max(fps || 30, 1) * 1.5, 0.04, BODY_MESH_FADE_SECONDS);
  const first = sortedFrames[0].t;
  const last = sortedFrames[sortedFrames.length - 1].t;
  if (timeSeconds < first - tolerance || timeSeconds > last + tolerance) return undefined;

  const exactFrame = sortedFrames.find((frame) => Math.abs(frame.t - timeSeconds) <= 1e-6);
  if (exactFrame) return exactFrame;
  if (timeSeconds <= first) return sortedFrames[0];
  for (const pair of state.pairs) {
    if (pair.from.t <= timeSeconds && timeSeconds < pair.to.t) {
      if (Math.abs(timeSeconds - pair.from.t) <= 1e-9) return pair.from;
      if (pair.holdEligible) {
        return pair.from;
      }
      if (timeSeconds - pair.from.t <= BODY_MESH_FADE_SECONDS) return pair.from;
      if (pair.to.t - timeSeconds <= BODY_MESH_FADE_SECONDS) return pair.to;
      return undefined;
    }
  }
  if (timeSeconds >= last) return sortedFrames[sortedFrames.length - 1];

  return sortedFrames.reduce((best, frame) =>
    Math.abs(frame.t - timeSeconds) < Math.abs(best.t - timeSeconds) ? frame : best,
  );
}

function bodyMeshInterpolationStateForPlayer(player: BodyMeshPlayer): BodyMeshInterpolationState {
  const cached = bodyMeshInterpolationStateCache.get(player);
  if (cached) return cached;
  const sortedFrames = [...player.frames].sort((left, right) => left.t - right.t);
  const pairs: BodyMeshInterpolationPair[] = [];
  for (let index = 0; index < sortedFrames.length - 1; index += 1) {
    const from = sortedFrames[index];
    const to = sortedFrames[index + 1];
    const gapSeconds = to.t - from.t;
    const sameWindow =
      from.source_window_index !== null &&
      to.source_window_index !== null &&
      from.source_window_index === to.source_window_index;
    const sameRun = from.source_window_index === to.source_window_index;
    const matchingVertexCount = from.mesh_vertices_world.length === to.mesh_vertices_world.length;
    const matchingJointCount = from.joints_world.length === to.joints_world.length;
    const holdEligible = gapSeconds > 0 && gapSeconds <= BODY_MESH_HOLD_MAX_GAP_SECONDS && sameRun;
    pairs.push({
      from,
      to,
      key: `${from.source_window_index ?? "none"}:${from.frame_idx}:${to.frame_idx}`,
      gapSeconds,
      sameWindow,
      sameRun,
      holdEligible,
      matchingVertexCount,
      matchingJointCount,
      eligible:
        gapSeconds > 0 &&
        gapSeconds <= BODY_MESH_INTERPOLATION_MAX_GAP_SECONDS &&
        sameWindow &&
        matchingVertexCount &&
        matchingJointCount,
    });
  }
  const state = { sortedFrames, pairs, reusableFrames: new Map<string, BodyMeshFrame>() };
  bodyMeshInterpolationStateCache.set(player, state);
  return state;
}

function interpolatedBodyMeshFrameForPair(
  state: BodyMeshInterpolationState,
  pair: BodyMeshInterpolationPair,
  timeSeconds: number,
): BodyMeshFrame {
  let frame = state.reusableFrames.get(pair.key);
  if (!frame) {
    frame = {
      frame_idx: pair.from.frame_idx,
      t: pair.from.t,
      source_window_index: pair.from.source_window_index,
      blend_weight: pair.from.blend_weight,
      joints_world: pair.from.joints_world.map(() => [0, 0, 0] as Vec3),
      joint_conf: pair.from.joint_conf.map(() => 0),
      mesh_vertices_world: pair.from.mesh_vertices_world.map(() => [0, 0, 0] as Vec3),
      mesh_faces: pair.from.mesh_faces,
      smplx_params: {},
      reasons: Array.from(new Set([...pair.from.reasons, ...pair.to.reasons, MESH_INTERPOLATION_REASON])),
      mesh_interpolated: true,
      interpolation: {
        from_frame_idx: pair.from.frame_idx,
        to_frame_idx: pair.to.frame_idx,
        alpha: 0,
        max_gap_s: BODY_MESH_INTERPOLATION_MAX_GAP_SECONDS,
      },
    };
    state.reusableFrames.set(pair.key, frame);
  }

  const alpha = clamp01((timeSeconds - pair.from.t) / pair.gapSeconds);
  frame.frame_idx = pair.from.frame_idx + (pair.to.frame_idx - pair.from.frame_idx) * alpha;
  frame.t = timeSeconds;
  frame.source_window_index = pair.from.source_window_index;
  frame.blend_weight = pair.from.blend_weight + (pair.to.blend_weight - pair.from.blend_weight) * alpha;
  frame.mesh_interpolated = true;
  if (frame.interpolation) {
    frame.interpolation.from_frame_idx = pair.from.frame_idx;
    frame.interpolation.to_frame_idx = pair.to.frame_idx;
    frame.interpolation.alpha = Number(alpha.toFixed(6));
    frame.interpolation.max_gap_s = BODY_MESH_INTERPOLATION_MAX_GAP_SECONDS;
  }
  interpolateVec3ArrayInPlace(frame.mesh_vertices_world, pair.from.mesh_vertices_world, pair.to.mesh_vertices_world, alpha);
  interpolateVec3ArrayInPlace(frame.joints_world, pair.from.joints_world, pair.to.joints_world, alpha);
  interpolateNumberArrayInPlace(frame.joint_conf, pair.from.joint_conf, pair.to.joint_conf, alpha);
  return frame;
}

function interpolateVec3ArrayInPlace(target: Vec3[], from: Vec3[], to: Vec3[], alpha: number): void {
  for (let index = 0; index < target.length; index += 1) {
    target[index][0] = cleanInterpolatedNumber(from[index][0] + (to[index][0] - from[index][0]) * alpha);
    target[index][1] = cleanInterpolatedNumber(from[index][1] + (to[index][1] - from[index][1]) * alpha);
    target[index][2] = cleanInterpolatedNumber(from[index][2] + (to[index][2] - from[index][2]) * alpha);
  }
}

function interpolateNumberArrayInPlace(target: number[], from: number[], to: number[], alpha: number): void {
  const length = Math.min(target.length, from.length, to.length);
  for (let index = 0; index < length; index += 1) {
    target[index] = cleanInterpolatedNumber(from[index] + (to[index] - from[index]) * alpha);
  }
}

function cleanInterpolatedNumber(value: number): number {
  const rounded = Number(value.toFixed(12));
  return Object.is(rounded, -0) ? 0 : rounded;
}

export function bodyMeshInterpolationStats(bodyMesh: BodyMesh | null): BodyMeshInterpolationStats {
  if (!bodyMesh) {
    return {
      computedFrameCount: 0,
      eligiblePairCount: 0,
      heldPairCount: 0,
      gapRefusedPairCount: 0,
      boundaryRefusedPairCount: 0,
      mismatchedVertexRefusedPairCount: 0,
      displayMultiplier: 1,
    };
  }
  let computedFrameCount = 0;
  let eligiblePairCount = 0;
  let heldPairCount = 0;
  let gapRefusedPairCount = 0;
  let boundaryRefusedPairCount = 0;
  let mismatchedVertexRefusedPairCount = 0;
  for (const player of bodyMesh.players) {
    const state = bodyMeshInterpolationStateForPlayer(player);
    computedFrameCount += state.sortedFrames.length;
    for (const pair of state.pairs) {
      if (pair.eligible) {
        eligiblePairCount += 1;
        continue;
      }
      if (pair.holdEligible) {
        heldPairCount += 1;
      }
      if (pair.gapSeconds <= 0 || pair.gapSeconds > BODY_MESH_INTERPOLATION_MAX_GAP_SECONDS) {
        gapRefusedPairCount += 1;
      }
      if (!pair.sameWindow) {
        boundaryRefusedPairCount += 1;
      }
      if (!pair.matchingVertexCount || !pair.matchingJointCount) {
        mismatchedVertexRefusedPairCount += 1;
      }
    }
  }
  return {
    computedFrameCount,
    eligiblePairCount,
    heldPairCount,
    gapRefusedPairCount,
    boundaryRefusedPairCount,
    mismatchedVertexRefusedPairCount,
    displayMultiplier: eligiblePairCount > 0 ? 2 : 1,
  };
}

export function bodyMeshInterpolationReadout(bodyMesh: BodyMesh | null): string {
  const stats = bodyMeshInterpolationStats(bodyMesh);
  if (!stats.computedFrameCount) return "mesh: no computed frames";
  if (stats.eligiblePairCount > 0) {
    const fps = Math.round(bodyMesh?.fps || 30);
    const pairWord = stats.eligiblePairCount === 1 ? "pair" : "pairs";
    return `mesh: computed ${fps}fps + interpolated x2 (${stats.eligiblePairCount} safe ${pairWord})`;
  }
  if (stats.heldPairCount > 0) {
    const gapWord = stats.heldPairCount === 1 ? "gap" : "gaps";
    return `mesh: computed sparse (${stats.computedFrameCount} frames, held ${stats.heldPairCount} ${gapWord})`;
  }
  return `mesh: computed sparse (${stats.computedFrameCount} frames, 0 safe pairs)`;
}

function alignBodyMeshFrameToWorldSkeleton(
  frame: BodyMeshFrame,
  worldFrame: VirtualWorldFrame | undefined,
  meshJointNames: readonly string[] | undefined,
  worldJointNames: readonly string[] | undefined,
): { frame: BodyMeshFrame; renderTranslation: Vec3; debug: BodyMeshAlignmentDebug } {
  const meshRoot = meshRootForFrame(frame, jointNamesForFrame(frame, meshJointNames, worldJointNames));
  const skeletonRoot = skeletonRootForWorldFrame(worldFrame, worldJointNames);
  if (!worldFrame) {
    return { frame, renderTranslation: [0, 0, 0], debug: bodyMeshAlignmentDebug(false, "missing_world_frame", meshRoot, null) };
  }
  if (!skeletonRoot) {
    return { frame, renderTranslation: [0, 0, 0], debug: bodyMeshAlignmentDebug(false, "missing_skeleton_root", meshRoot, null) };
  }
  if (!meshRoot) {
    return { frame, renderTranslation: [0, 0, 0], debug: bodyMeshAlignmentDebug(false, "missing_mesh_root", null, skeletonRoot) };
  }
  const rootDelta = subtractVec3(skeletonRoot, meshRoot);
  const floorLift = meshFloorLiftForTranslation(frame.mesh_vertices_world, rootDelta);
  const delta: Vec3 = floorLift > 0 ? [rootDelta[0], rootDelta[1], cleanInterpolatedNumber(rootDelta[2] + floorLift)] : rootDelta;
  const debug = bodyMeshAlignmentDebug(true, "skeleton_root", meshRoot, skeletonRoot, delta, floorLift);
  return { frame, renderTranslation: delta, debug };
}

function bodyMeshAlignmentDebug(
  applied: boolean,
  reason: BodyMeshAlignmentDebug["reason"],
  meshRoot: Vec3 | null,
  skeletonRoot: Vec3 | null,
  delta: Vec3 = [0, 0, 0],
  floorLiftM = 0,
): BodyMeshAlignmentDebug {
  return {
    applied,
    reason,
    delta,
    mesh_root: meshRoot,
    skeleton_root: skeletonRoot,
    floor_guard_applied: floorLiftM > 0,
    floor_lift_m: cleanInterpolatedNumber(floorLiftM),
  };
}

function skeletonRootForWorldFrame(frame: VirtualWorldFrame | undefined, jointNames: readonly string[] | undefined): Vec3 | null {
  if (!frame || !frame.joints_world.length) return null;
  return rootFromNamedJoints(frame.joints_world, frame.joint_conf, jointNames);
}

function meshRootForFrame(frame: BodyMeshFrame, jointNames: readonly string[] | undefined): Vec3 | null {
  return rootFromNamedJoints(frame.joints_world, frame.joint_conf, jointNames);
}

function jointNamesForFrame(
  frame: BodyMeshFrame,
  meshJointNames: readonly string[] | undefined,
  worldJointNames: readonly string[] | undefined,
): readonly string[] | undefined {
  if (meshJointNames?.length) return meshJointNames;
  if (worldJointNames?.length && worldJointNames.length === frame.joints_world.length) return worldJointNames;
  return undefined;
}

function rootFromNamedJoints(joints: Vec3[], conf: number[], jointNames: readonly string[] | undefined): Vec3 | null {
  if (!jointNames?.length) return null;
  const points = new Map<string, Vec3>();
  jointNames.forEach((name, index) => {
    const point = joints[index];
    const confidence = conf[index];
    if (point && (confidence === undefined || !Number.isFinite(confidence) || confidence >= 0.05)) {
      points.set(name.toLowerCase(), point);
    }
  });
  const leftHip = points.get("left_hip");
  const rightHip = points.get("right_hip");
  if (leftHip && rightHip) return midpointVec3(leftHip, rightHip);
  return points.get("pelvis") ?? points.get("root") ?? points.get("smpl_root") ?? points.get("hips") ?? null;
}

function midpointVec3(left: Vec3, right: Vec3): Vec3 {
  return [(left[0] + right[0]) / 2, (left[1] + right[1]) / 2, (left[2] + right[2]) / 2];
}

function subtractVec3(left: Vec3, right: Vec3): Vec3 {
  return [
    cleanInterpolatedNumber(left[0] - right[0]),
    cleanInterpolatedNumber(left[1] - right[1]),
    cleanInterpolatedNumber(left[2] - right[2]),
  ];
}

function meshFloorLiftForTranslation(vertices: Vec3[], delta: Vec3): number {
  if (!vertices.length) return 0;
  const translatedLowest = Math.min(...vertices.map((vertex) => vertex[2] + delta[2]));
  return translatedLowest < -0.08 ? cleanInterpolatedNumber(-translatedLowest) : 0;
}

export function displayFpsReplayData(
  world: VirtualWorld,
  bodyMesh: BodyMesh | null,
  enabled: boolean,
  options: DisplayFpsOptions = {},
): DisplayFpsReplayData {
  const sourceFps = Math.round(world.fps || bodyMesh?.fps || 30);
  const baseStats: DisplayFpsStats = {
    enabled,
    sourceFps,
    displayFps: enabled ? sourceFps * 2 : sourceFps,
    worldComputedFrameCount: countWorldPlayerFrames(world),
    worldInterpolatedFrameCount: 0,
    meshComputedFrameCount: countBodyMeshFrames(bodyMesh),
    meshInterpolatedFrameCount: 0,
    meshMaxInterpolatedGapMs: 0,
    meshRefusedPairCount: 0,
  };
  if (!enabled) return { world, bodyMesh, stats: baseStats };

  const doubledWorld = doubleFpsWorld(world);
  const doubledBodyMesh = doubleFpsBodyMesh(bodyMesh, options.meshMaxGapSeconds ?? DISPLAY_FPS_MESH_MAX_GAP_SECONDS);
  return {
    world: doubledWorld.world,
    bodyMesh: doubledBodyMesh.bodyMesh,
    stats: {
      ...baseStats,
      displayFps: sourceFps * 2,
      worldInterpolatedFrameCount: doubledWorld.interpolatedFrameCount,
      meshInterpolatedFrameCount: doubledBodyMesh.interpolatedFrameCount,
      meshMaxInterpolatedGapMs: doubledBodyMesh.maxInterpolatedGapMs,
      meshRefusedPairCount: doubledBodyMesh.refusedPairCount,
    },
  };
}

export function displayFpsReadout(stats: DisplayFpsStats): string {
  if (!stats.enabled) return `${stats.sourceFps}fps display: original`;
  const skeletonWord = stats.worldInterpolatedFrameCount === 1 ? "skeleton" : "skeletons";
  const meshText = `${stats.meshInterpolatedFrameCount} mesh`;
  const cadenceText =
    stats.meshInterpolatedFrameCount > 0 && stats.meshMaxInterpolatedGapMs > 0
      ? `; mesh interpolated across ${stats.meshMaxInterpolatedGapMs}ms gaps`
      : "";
  return `${stats.displayFps}fps display: computed ${stats.sourceFps} + interpolated ${stats.worldInterpolatedFrameCount} ${skeletonWord}, ${meshText}${cadenceText}`;
}

function doubleFpsWorld(world: VirtualWorld): { world: VirtualWorld; interpolatedFrameCount: number } {
  let interpolatedFrameCount = 0;
  const players = world.players.map((player) => {
    const frames: VirtualWorldFrame[] = [];
    const sorted = [...player.frames].sort((left, right) => left.t - right.t);
    for (let index = 0; index < sorted.length; index += 1) {
      const current = sorted[index];
      frames.push(current);
      const next = sorted[index + 1];
      if (!next) continue;
      const midpoint = interpolatedWorldFrame(current, next);
      if (midpoint) {
        frames.push(midpoint);
        interpolatedFrameCount += 1;
      }
    }
    return { ...player, frames };
  });
  return {
    world: {
      ...world,
      fps: (world.fps || 30) * 2,
      players,
      summary: {
        ...world.summary,
        joint_player_frame_count: players.reduce(
          (total, player) => total + player.frames.filter((frame) => frame.joints_world.length > 0).length,
          0,
        ),
        floor_placed_player_frame_count: players.reduce(
          (total, player) => total + player.frames.filter((frame) => frame.floor_world_xyz || frame.track_world_xy).length,
          0,
        ),
      },
    },
    interpolatedFrameCount,
  };
}

function interpolatedWorldFrame(from: VirtualWorldFrame, to: VirtualWorldFrame): VirtualWorldFrame | null {
  const gapSeconds = to.t - from.t;
  if (gapSeconds <= 0) return null;
  if (from.joints_world.length !== to.joints_world.length || from.joints_world.length === 0) return null;
  const alpha = 0.5;
  const joints_world = interpolateVec3Array(from.joints_world, to.joints_world, alpha);
  const mesh_vertices_world =
    from.mesh_vertices_world.length === to.mesh_vertices_world.length
      ? interpolateVec3Array(from.mesh_vertices_world, to.mesh_vertices_world, alpha)
      : [];
  return {
    ...from,
    t: cleanInterpolatedNumber(from.t + gapSeconds * alpha),
    mesh_ref:
      from.mesh_ref && to.mesh_ref && from.mesh_ref.player_id === to.mesh_ref.player_id
        ? {
            ...from.mesh_ref,
            t: cleanInterpolatedNumber(from.mesh_ref.t + ((to.mesh_ref.t ?? to.t) - from.mesh_ref.t) * alpha),
            frame_idx: Math.round(from.mesh_ref.frame_idx + (to.mesh_ref.frame_idx - from.mesh_ref.frame_idx) * alpha),
          }
        : from.mesh_ref ?? null,
    track_world_xy:
      from.track_world_xy && to.track_world_xy
        ? interpolateVec2(from.track_world_xy, to.track_world_xy, alpha)
        : from.track_world_xy ?? null,
    transl_world: from.transl_world && to.transl_world ? interpolateVec3(from.transl_world, to.transl_world, alpha) : from.transl_world ?? null,
    joints_world,
    joint_conf: interpolateNumberArray(from.joint_conf, to.joint_conf, alpha),
    mesh_vertices_world,
    joint_count: joints_world.length,
    mesh_vertex_count: mesh_vertices_world.length,
    floor_world_xyz:
      from.floor_world_xyz && to.floor_world_xyz
        ? interpolateVec3(from.floor_world_xyz, to.floor_world_xyz, alpha)
        : from.floor_world_xyz ?? null,
  };
}

function doubleFpsBodyMesh(
  bodyMesh: BodyMesh | null,
  meshMaxGapSeconds: number,
): { bodyMesh: BodyMesh | null; interpolatedFrameCount: number; maxInterpolatedGapMs: number; refusedPairCount: number } {
  if (!bodyMesh) return { bodyMesh: null, interpolatedFrameCount: 0, maxInterpolatedGapMs: 0, refusedPairCount: 0 };
  const maxGap = Math.max(BODY_MESH_INTERPOLATION_MAX_GAP_SECONDS, Math.min(meshMaxGapSeconds, DISPLAY_FPS_MESH_MAX_GAP_SECONDS));
  let interpolatedFrameCount = 0;
  let maxInterpolatedGapMs = 0;
  let refusedPairCount = 0;
  const players = bodyMesh.players.map((player) => {
    const sorted = [...player.frames].sort((left, right) => left.t - right.t);
    const frames: BodyMeshFrame[] = [];
    for (let index = 0; index < sorted.length; index += 1) {
      const current = sorted[index];
      frames.push(current);
      const next = sorted[index + 1];
      if (!next) continue;
      const pair = bodyMeshInterpolationPairForFrames(current, next);
      if (
        pair.gapSeconds > 0 &&
        pair.gapSeconds <= maxGap &&
        pair.sameWindow &&
        pair.matchingVertexCount &&
        pair.matchingJointCount
      ) {
        frames.push(interpolatedBodyMeshFrame(current, next, 0.5, maxGap, DISPLAY_FPS_INTERPOLATION_REASON));
        interpolatedFrameCount += 1;
        maxInterpolatedGapMs = Math.max(maxInterpolatedGapMs, Math.round(pair.gapSeconds * 1000));
      } else {
        refusedPairCount += 1;
      }
    }
    return { ...player, frames };
  });
  return {
    bodyMesh: {
      ...bodyMesh,
      fps: (bodyMesh.fps || 30) * 2,
      players,
      summary: {
        ...bodyMesh.summary,
        mesh_frame_count: players.reduce((total, player) => total + player.frames.length, 0),
      },
    },
    interpolatedFrameCount,
    maxInterpolatedGapMs,
    refusedPairCount,
  };
}

function bodyMeshInterpolationPairForFrames(from: BodyMeshFrame, to: BodyMeshFrame): Omit<BodyMeshInterpolationPair, "key" | "eligible"> {
  const gapSeconds = to.t - from.t;
  const sameWindow =
    from.source_window_index !== null &&
    to.source_window_index !== null &&
    from.source_window_index === to.source_window_index;
  const sameRun = from.source_window_index === to.source_window_index;
  const matchingVertexCount = from.mesh_vertices_world.length === to.mesh_vertices_world.length;
  const matchingJointCount = from.joints_world.length === to.joints_world.length;
  return {
    from,
    to,
    gapSeconds,
    sameWindow,
    sameRun,
    holdEligible: gapSeconds > 0 && gapSeconds <= BODY_MESH_HOLD_MAX_GAP_SECONDS && sameRun,
    matchingVertexCount,
    matchingJointCount,
  };
}

function interpolatedBodyMeshFrame(
  from: BodyMeshFrame,
  to: BodyMeshFrame,
  alpha: number,
  maxGapSeconds: number,
  reason: string,
): BodyMeshFrame {
  return {
    frame_idx: cleanInterpolatedNumber(from.frame_idx + (to.frame_idx - from.frame_idx) * alpha),
    t: cleanInterpolatedNumber(from.t + (to.t - from.t) * alpha),
    source_window_index: from.source_window_index,
    blend_weight: cleanInterpolatedNumber(from.blend_weight + (to.blend_weight - from.blend_weight) * alpha),
    joints_world: interpolateVec3Array(from.joints_world, to.joints_world, alpha),
    joint_conf: interpolateNumberArray(from.joint_conf, to.joint_conf, alpha),
    mesh_vertices_world: interpolateVec3Array(from.mesh_vertices_world, to.mesh_vertices_world, alpha),
    mesh_faces: from.mesh_faces,
    smplx_params: {},
    reasons: Array.from(new Set([...from.reasons, ...to.reasons, reason])),
    mesh_interpolated: true,
    interpolation: {
      from_frame_idx: from.frame_idx,
      to_frame_idx: to.frame_idx,
      alpha: cleanInterpolatedNumber(alpha),
      max_gap_s: maxGapSeconds,
    },
  };
}

function countWorldPlayerFrames(world: VirtualWorld): number {
  return world.players.reduce((total, player) => total + player.frames.length, 0);
}

function countBodyMeshFrames(bodyMesh: BodyMesh | null): number {
  return bodyMesh?.players.reduce((total, player) => total + player.frames.length, 0) ?? 0;
}

function interpolateVec2(from: Vec2, to: Vec2, alpha: number): Vec2 {
  return [cleanInterpolatedNumber(from[0] + (to[0] - from[0]) * alpha), cleanInterpolatedNumber(from[1] + (to[1] - from[1]) * alpha)];
}

function interpolateVec3(from: Vec3, to: Vec3, alpha: number): Vec3 {
  return [
    cleanInterpolatedNumber(from[0] + (to[0] - from[0]) * alpha),
    cleanInterpolatedNumber(from[1] + (to[1] - from[1]) * alpha),
    cleanInterpolatedNumber(from[2] + (to[2] - from[2]) * alpha),
  ];
}

function interpolateVec3Array(from: Vec3[], to: Vec3[], alpha: number): Vec3[] {
  return from.map((point, index) => interpolateVec3(point, to[index], alpha));
}

function interpolateNumberArray(from: number[], to: number[], alpha: number): number[] {
  const length = Math.min(from.length, to.length);
  return Array.from({ length }, (_, index) => cleanInterpolatedNumber(from[index] + (to[index] - from[index]) * alpha));
}

function bodyMeshPresenceOpacityForTime(player: BodyMeshPlayer, timeSeconds: number): number {
  if (!player.frames.length) return 0;
  const times = player.frames.map((frame) => frame.t).sort((left, right) => left - right);
  const first = times[0];
  const last = times[times.length - 1];
  if (timeSeconds < first) return clamp01((timeSeconds - (first - BODY_MESH_FADE_SECONDS)) / BODY_MESH_FADE_SECONDS);
  if (timeSeconds > last) return clamp01(((last + BODY_MESH_FADE_SECONDS) - timeSeconds) / BODY_MESH_FADE_SECONDS);
  return 1;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function paddleFrameForTime(paddle: VirtualWorldPaddle, timeSeconds: number, fps: number): VirtualWorldPaddleFrame | undefined {
  if (!paddle.frames.length) return undefined;
  const sortedFrames = [...paddle.frames].sort((left, right) => left.t - right.t);
  const tolerance = Math.max(1 / Math.max(fps || 30, 1) * 1.5, 0.04);
  const first = sortedFrames[0].t;
  const last = sortedFrames[sortedFrames.length - 1].t;
  if (timeSeconds < first - tolerance || timeSeconds > last + tolerance) return undefined;
  return sortedFrames.reduce((best, frame) => (Math.abs(frame.t - timeSeconds) < Math.abs(best.t - timeSeconds) ? frame : best));
}

function solidMeshFrameRenderable(frame: BodyMeshFrame | undefined): frame is BodyMeshFrame {
  return Boolean(frame && frame.mesh_vertices_world.length > 0 && frame.mesh_faces.length > 0);
}

function normalizedMeshPlayerIdForWorldFrame(player: VirtualWorldPlayer, frame: VirtualWorldFrame | undefined): number {
  return frame?.mesh_ref?.player_id ?? player.id;
}

function windowDistanceFromTime(window: BodyMeshIndexWindow, timeSeconds: number): number {
  if (window.t0 <= timeSeconds && timeSeconds <= window.t1) return 0;
  return Math.min(Math.abs(timeSeconds - window.t0), Math.abs(timeSeconds - window.t1));
}

function readQuantizedVec3Array(view: DataView, offsetBytes: number, count: number, scale: number): Vec3[] {
  const points: Vec3[] = [];
  let offset = offsetBytes;
  for (let index = 0; index < count; index += 1) {
    if (offset + 6 > view.byteLength) {
      throw new Error(`body mesh chunk ended while reading vec3 ${index}`);
    }
    points.push([
      view.getInt16(offset, true) / scale,
      view.getInt16(offset + 2, true) / scale,
      view.getInt16(offset + 4, true) / scale,
    ]);
    offset += 6;
  }
  return points;
}

function bodyMeshBytesLookGzipped(bytes: ArrayBuffer): boolean {
  if (bytes.byteLength < 2) return false;
  const view = new Uint8Array(bytes, 0, 2);
  return view[0] === 0x1f && view[1] === 0x8b;
}

async function decompressGzipBytes(bytes: ArrayBuffer): Promise<ArrayBuffer> {
  if (typeof DecompressionStream === "undefined") {
    throw new Error("gzip body mesh chunks require browser DecompressionStream support");
  }
  const stream = new Response(bytes).body;
  if (!stream) throw new Error("failed to create gzip decode stream");
  return new Response(stream.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}

function emptyLabelOverlayPayload(): LabelOverlayPayload {
  return {
    items: [],
    notGroundTruth: true,
    status: null,
    sourceWidth: 1920,
    sourceHeight: 1080,
    secondsPerFrame: 1 / 30,
  };
}

function readOptionalPositiveOrZeroNumber(value: unknown): number | null {
  if (typeof value === "number") return Number.isFinite(value) && value >= 0 ? value : null;
  if (typeof value === "string") {
    const parsed = Number(value.trim());
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
  }
  return null;
}

function readOptionalPositiveNumber(value: unknown): number | null {
  if (typeof value === "number") {
    return Number.isFinite(value) && value > 0 ? value : null;
  }
  if (typeof value === "string") {
    const parsed = Number(value.trim());
    return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
  }
  return null;
}

function inferLabelSourceSize(
  payload: Record<string, unknown>,
  frameMeta: Record<string, unknown>,
  annotationMeta: Record<string, unknown>,
): [number, number] {
  const explicitLabelResolution = readFirstResolution(
    frameMeta.label_resolution,
    frameMeta.render_resolution,
    frameMeta.coordinate_resolution,
    frameMeta.frame_pack_resolution,
    frameMeta.frame_resolution,
    frameMeta.image_resolution,
    annotationMeta.label_resolution,
    annotationMeta.render_resolution,
    annotationMeta.coordinate_resolution,
    annotationMeta.resolution,
    payload.label_resolution,
    payload.render_resolution,
    payload.coordinate_resolution,
  );
  if (explicitLabelResolution) return explicitLabelResolution;
  return readFirstResolution(frameMeta.source_resolution, frameMeta.resolution, payload.source_resolution, payload.resolution) ?? [1920, 1080];
}

function readFirstResolution(...values: unknown[]): [number, number] | null {
  for (const value of values) {
    const resolution = readOptionalResolution(value);
    if (resolution) return resolution;
  }
  return null;
}

function readOptionalResolution(value: unknown): [number, number] | null {
  if (Array.isArray(value) && value.length >= 2) {
    const width = readOptionalPositiveNumber(value[0]);
    const height = readOptionalPositiveNumber(value[1]);
    return width !== null && height !== null ? [width, height] : null;
  }
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const record = value as Record<string, unknown>;
    const width = readOptionalPositiveNumber(record.width) ?? readOptionalPositiveNumber(record.w);
    const height = readOptionalPositiveNumber(record.height) ?? readOptionalPositiveNumber(record.h);
    return width !== null && height !== null ? [width, height] : null;
  }
  return null;
}

function labelFrameIndex(value: LabelItem["frame"]): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return Math.max(0, Math.round(value));
  if (typeof value !== "string") return null;
  const match = value.match(/(\d+)/);
  return match ? Math.max(0, Number.parseInt(match[1], 10) - 1) : null;
}

function readContactWindowEvent(input: unknown, index: number): ContactWindowEvent {
  const path = `contact_windows.events[${index}]`;
  assertRecord(input, path);
  assertRecord(input.window, `${path}.window`);
  const windowStart = readNonNegativeNumber(input.window.t0, `${path}.window.t0`);
  const windowEnd = readNonNegativeNumber(input.window.t1, `${path}.window.t1`);
  if (windowEnd < windowStart) throw new Error(`${path}.window.t1 must be greater than or equal to ${path}.window.t0`);
  return {
    type: readEnum(input.type, `${path}.type`, ["contact", "bounce", "net_cross"] as const),
    t: readNonNegativeNumber(input.t, `${path}.t`),
    frame: readNonNegativeInteger(input.frame, `${path}.frame`),
    player_id: input.player_id === null || input.player_id === undefined ? null : readPlayerId(input.player_id, `${path}.player_id`),
    confidence: readUnitNumber(input.confidence, `${path}.confidence`),
    sources: readContactSources(input.sources, `${path}.sources`),
    window: {
      t0: windowStart,
      t1: windowEnd,
      importance: readUnitNumber(input.window.importance, `${path}.window.importance`),
    },
  };
}

function readContactSources(input: unknown, path: string): ContactWindowEvent["sources"] {
  assertRecord(input, path);
  const sources: ContactWindowEvent["sources"] = {};
  let presentSourceCount = 0;
  for (const key of ["audio", "wrist_vel", "ball_inflection"] as const) {
    if (input[key] === null || input[key] === undefined) continue;
    sources[key] = readUnitNumber(input[key], `${path}.${key}`);
    presentSourceCount += 1;
  }
  if (input.human_review !== undefined) {
    sources.human_review = input.human_review === null ? null : readUnitNumber(input.human_review, `${path}.human_review`);
    if (input.human_review !== null) presentSourceCount += 1;
  }
  if (presentSourceCount === 0) throw new Error(`${path} must include at least one source score`);
  return sources;
}

function readBallArcSelectedEvent(input: unknown, index: number): BallArcSelectedEvent {
  const path = `events_selected.selected[${index}]`;
  assertRecord(input, path);
  return {
    anchor_id: input.anchor_id === null || input.anchor_id === undefined ? null : readString(input.anchor_id, `${path}.anchor_id`),
    kind: readString(input.kind, `${path}.kind`),
    frame: input.frame === null || input.frame === undefined ? null : readNonNegativeInteger(input.frame, `${path}.frame`),
    t: readNonNegativeNumber(input.t, `${path}.t`),
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
  const endpoints = readFixedArray(input.net.endpoints, "virtual_world.court.net.endpoints", 2);
  return {
    sport: readEnum(input.sport, "virtual_world.court.sport", ["pickleball", "tennis"] as const),
    coordinate_frame: readString(input.coordinate_frame, "virtual_world.court.coordinate_frame"),
    length_m: readNumber(input.length_m, "virtual_world.court.length_m"),
    width_m: readNumber(input.width_m, "virtual_world.court.width_m"),
    line_segments,
    net: {
      endpoints: [readVec3(endpoints[0], "net.endpoints[0]"), readVec3(endpoints[1], "net.endpoints[1]")],
      center_height_m: readNumber(input.net.center_height_m, "virtual_world.court.net.center_height_m"),
      post_height_m: readNumber(input.net.post_height_m, "virtual_world.court.net.post_height_m"),
    },
    trust_band: readTrustBand(input.trust_band, "virtual_world.court.trust_band"),
  };
}

function readTrustBand(input: unknown, path: string): TrustBand | null {
  if (input === null || input === undefined) return null;
  assertRecord(input, path);
  return {
    stage: readString(input.stage, `${path}.stage`),
    gate_id: readString(input.gate_id, `${path}.gate_id`),
    gate_status: readString(input.gate_status, `${path}.gate_status`),
    badge: readEnum(input.badge, `${path}.badge`, ["verified", "preview", "low_confidence"] as const),
    reason: readString(input.reason, `${path}.reason`),
    evidence_path:
      input.evidence_path === null || input.evidence_path === undefined
        ? null
        : readString(input.evidence_path, `${path}.evidence_path`),
  };
}

function readCoverageFields(input: Record<string, unknown>, path: string): EntityCoverage {
  const coverage: EntityCoverage = {};
  if (input.coverage_fraction !== null && input.coverage_fraction !== undefined) {
    coverage.coverage_fraction = readUnitNumber(input.coverage_fraction, `${path}.coverage_fraction`);
  }
  if (input.min_t !== null && input.min_t !== undefined) {
    coverage.min_t = readNumber(input.min_t, `${path}.min_t`);
  }
  if (input.max_t !== null && input.max_t !== undefined) {
    coverage.max_t = readNumber(input.max_t, `${path}.max_t`);
  }
  return coverage;
}

function readConfidenceProvenance(input: unknown, path: string): ConfidenceProvenance | null {
  if (input === null || input === undefined) return null;
  assertRecord(input, path);
  return {
    band: input.band === null || input.band === undefined ? null : readString(input.band, `${path}.band`),
    display_band:
      input.display_band === null || input.display_band === undefined ? null : readString(input.display_band, `${path}.display_band`),
    horizon_frames:
      input.horizon_frames === null || input.horizon_frames === undefined
        ? null
        : readNonNegativeInteger(input.horizon_frames, `${path}.horizon_frames`),
    predicted_sigma_m:
      input.predicted_sigma_m === null || input.predicted_sigma_m === undefined
        ? null
        : readNonNegativeNumber(input.predicted_sigma_m, `${path}.predicted_sigma_m`),
    predictor: input.predictor === null || input.predictor === undefined ? null : readString(input.predictor, `${path}.predictor`),
  };
}

function readBallPhysicsFill(input: unknown, path: string): VirtualWorldBallFrame["physics_fill"] {
  if (input === null || input === undefined) return null;
  assertRecord(input, path);
  const fill: NonNullable<VirtualWorldBallFrame["physics_fill"]> = {};
  if (input.uncertainty_m !== null && input.uncertainty_m !== undefined) {
    fill.uncertainty_m = readNonNegativeNumber(input.uncertainty_m, `${path}.uncertainty_m`);
  }
  if (input.render_only !== null && input.render_only !== undefined) {
    fill.render_only = readBoolean(input.render_only, `${path}.render_only`);
  }
  if (input.not_for_detection_metrics !== null && input.not_for_detection_metrics !== undefined) {
    fill.not_for_detection_metrics = readBoolean(input.not_for_detection_metrics, `${path}.not_for_detection_metrics`);
  }
  return fill;
}

function readReviewedBounce(input: unknown, index: number): ReviewedBounce {
  const path = `reviewed_bounces.bounces[${index}]`;
  assertRecord(input, path);
  return {
    review_id: readString(input.review_id, `${path}.review_id`),
    frame: readNonNegativeInteger(input.frame, `${path}.frame`),
    t: readNonNegativeNumber(input.t, `${path}.t`),
  };
}

function readRallySpan(input: unknown, index: number): RallySpan {
  const path = `rally_spans.spans[${index}]`;
  assertRecord(input, path);
  const t0 = readNonNegativeNumber(input.t0, `${path}.t0`);
  const t1 = readNonNegativeNumber(input.t1, `${path}.t1`);
  if (t1 < t0) throw new Error(`${path}.t1 must be greater than or equal to ${path}.t0`);
  const explicitRallyId =
    input.rally_id === null || input.rally_id === undefined
      ? input.id === null || input.id === undefined
        ? null
        : readString(input.id, `${path}.id`)
      : readString(input.rally_id, `${path}.rally_id`);
  return {
    rallyId: explicitRallyId ?? `rally_${String(index).padStart(3, "0")}`,
    t0,
    t1,
    sources:
      input.sources === undefined
        ? []
        : readArray(input.sources, `${path}.sources`).map((source, sourceIndex) => readString(source, `${path}.sources[${sourceIndex}]`)),
  };
}

function readPlayer(input: unknown, index: number): VirtualWorldPlayer {
  const path = `virtual_world.players[${index}]`;
  assertRecord(input, path);
  return {
    ...readCoverageFields(input, path),
    id: readNumber(input.id, `${path}.id`, true),
    side: input.side === null || input.side === undefined ? null : readString(input.side, `${path}.side`),
    role: input.role === null || input.role === undefined ? null : readString(input.role, `${path}.role`),
    representation: readEnum(input.representation, `${path}.representation`, ["track_only", "joints", "mesh"] as const),
    frames: readArray(input.frames, `${path}.frames`).map((frame, frameIndex) => readFrame(frame, `${path}.frames[${frameIndex}]`)),
    trust_band: readTrustBand(input.trust_band, `${path}.trust_band`),
  };
}

function readFrame(input: unknown, path: string): VirtualWorldFrame {
  assertRecord(input, path);
  return {
    t: readNumber(input.t, `${path}.t`),
    mesh_ref: readMeshRef(input.mesh_ref, `${path}.mesh_ref`),
    track_world_xy: input.track_world_xy === null || input.track_world_xy === undefined ? null : readVec2(input.track_world_xy, `${path}.track_world_xy`),
    track_conf: input.track_conf === null || input.track_conf === undefined ? null : readNumber(input.track_conf, `${path}.track_conf`),
    bbox: input.bbox === null || input.bbox === undefined ? null : readBBox(input.bbox, `${path}.bbox`),
    transl_world: input.transl_world === null || input.transl_world === undefined ? null : readVec3(input.transl_world, `${path}.transl_world`),
    joints_world: readArray(input.joints_world, `${path}.joints_world`).map((point, index) => readVec3(point, `${path}.joints_world[${index}]`)),
    joint_conf:
      input.joint_conf === undefined
        ? []
        : readArray(input.joint_conf, `${path}.joint_conf`).map((confidence, index) => readNumber(confidence, `${path}.joint_conf[${index}]`)),
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
    grf:
      input.grf === null || input.grf === undefined
        ? null
        : readArray(input.grf, `${path}.grf`).map((point, index) => readVec3(point, `${path}.grf[${index}]`)),
    skeleton_implausible:
      input.skeleton_implausible === undefined ? false : readBoolean(input.skeleton_implausible, `${path}.skeleton_implausible`),
    trust_band: readTrustBand(input.trust_band, `${path}.trust_band`),
  };
}

function readMeshRef(input: unknown, path: string): VirtualWorldFrame["mesh_ref"] {
  if (input === null || input === undefined) return null;
  assertRecord(input, path);
  return {
    artifact: readString(input.artifact, `${path}.artifact`),
    player_id: readPlayerId(input.player_id, `${path}.player_id`),
    frame_idx: readNonNegativeInteger(input.frame_idx, `${path}.frame_idx`),
    t: readNonNegativeNumber(input.t, `${path}.t`),
  };
}

function readBall(input: unknown): VirtualWorld["ball"] {
  assertRecord(input, "virtual_world.ball");
  return {
    ...readCoverageFields(input, "virtual_world.ball"),
    // See the `VirtualWorld["ball"]["source"]` type comment above: this is
    // deliberately a free-form string read, not a closed-enum `readEnum`
    // check, so new Python-side ball sources don't need a matching TS
    // release before the viewer can load their runs again.
    source: input.source === null || input.source === undefined ? null : readString(input.source, "virtual_world.ball.source"),
    frames: readArray(input.frames, "virtual_world.ball.frames").map((frame, index) => {
      const path = `virtual_world.ball.frames[${index}]`;
      assertRecord(frame, path);
      return {
        t: readNumber(frame.t, `${path}.t`),
        xy: readVec2(frame.xy, `${path}.xy`),
        xy_interpolated: frame.xy_interpolated === undefined ? false : readBoolean(frame.xy_interpolated, `${path}.xy_interpolated`),
        conf: readNumber(frame.conf, `${path}.conf`),
        visible: readBoolean(frame.visible, `${path}.visible`),
        world_xyz: frame.world_xyz === null || frame.world_xyz === undefined ? null : readVec3(frame.world_xyz, `${path}.world_xyz`),
        court_intersection_world_xyz: readOptionalVec3FromKeys(frame, path, [
          "court_intersection_world_xyz",
          "ray_court_intersection_world_xyz",
          "floor_projected_world_xyz",
        ]),
        arc_segment_id: readOptionalSegmentId(frame, path),
        approx: frame.approx === undefined ? false : readBoolean(frame.approx, `${path}.approx`),
        confidence_provenance: readConfidenceProvenance(frame.confidence_provenance, `${path}.confidence_provenance`),
        render_only: frame.render_only === undefined ? false : readBoolean(frame.render_only, `${path}.render_only`),
        not_for_detection_metrics:
          frame.not_for_detection_metrics === undefined
            ? false
            : readBoolean(frame.not_for_detection_metrics, `${path}.not_for_detection_metrics`),
        trust_band: readTrustBand(frame.trust_band, `${path}.trust_band`),
        physics_fill: readBallPhysicsFill(frame.physics_fill, `${path}.physics_fill`),
      };
    }),
    trust_band: readTrustBand(input.trust_band, "virtual_world.ball.trust_band"),
  };
}

function readOptionalVec3FromKeys(input: Record<string, unknown>, path: string, keys: string[]): Vec3 | null {
  for (const key of keys) {
    if (input[key] !== null && input[key] !== undefined) return readVec3(input[key], `${path}.${key}`);
  }
  return null;
}

function readOptionalSegmentId(input: Record<string, unknown>, path: string): number | string | null {
  const direct = input.arc_segment_id ?? input.segment_id;
  if (direct !== null && direct !== undefined) return readSegmentIdValue(direct, `${path}.arc_segment_id`);
  const arcSolver = input.arc_solver;
  if (arcSolver && typeof arcSolver === "object" && !Array.isArray(arcSolver)) {
    const value = (arcSolver as Record<string, unknown>).segment_id;
    if (value !== null && value !== undefined) return readSegmentIdValue(value, `${path}.arc_solver.segment_id`);
  }
  return null;
}

function readSegmentIdValue(value: unknown, path: string): number | string {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error(`${path} must be finite`);
    return value;
  }
  if (typeof value === "string" && value.trim()) return value;
  throw new Error(`${path} must be a number or non-empty string`);
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

function readPaddle(input: unknown, index: number): VirtualWorldPaddle {
  const path = `virtual_world.paddles[${index}]`;
  assertRecord(input, path);
  assertRecord(input.paddle_dims_in, `${path}.paddle_dims_in`);
  const paddleDims = readPaddleDims(input.paddle_dims_in, `${path}.paddle_dims_in`);
  return {
    ...readCoverageFields(input, path),
    player_id: readNumber(input.player_id, `${path}.player_id`, true),
    paddle_dims_in: paddleDims,
    frames: readArray(input.frames, `${path}.frames`).map((frame, frameIndex) => readPaddleFrame(frame, `${path}.frames[${frameIndex}]`)),
    trust_band: readTrustBand(input.trust_band, `${path}.trust_band`),
  };
}

function readPaddleDims(input: Record<string, unknown>, path: string): Record<string, number> {
  const dims: Record<string, number> = {};
  for (const [key, value] of Object.entries(input)) {
    const number = readNumber(value, `${path}.${key}`);
    if (number <= 0) throw new Error(`${path}.${key} must be positive`);
    dims[key] = number;
  }
  const hasLengthWidth = typeof dims.length === "number" && typeof dims.width === "number";
  const hasHeightWidth = typeof dims.h === "number" && typeof dims.w === "number";
  if (!hasLengthWidth && !hasHeightWidth) throw new Error(`${path} must include length/width or h/w`);
  return dims;
}

function readPaddleFrame(input: unknown, path: string): VirtualWorldPaddleFrame {
  assertRecord(input, path);
  assertRecord(input.pose_se3, `${path}.pose_se3`);
  return {
    t: readNumber(input.t, `${path}.t`),
    pose_se3: {
      R: readRotationMatrix(input.pose_se3.R, `${path}.pose_se3.R`),
      t: readVec3(input.pose_se3.t, `${path}.pose_se3.t`),
    },
    mesh_vertices_world: readArray(input.mesh_vertices_world, `${path}.mesh_vertices_world`).map((point, index) =>
      readVec3(point, `${path}.mesh_vertices_world[${index}]`),
    ),
    mesh_faces: readArray(input.mesh_faces, `${path}.mesh_faces`).map((face, index) => readFace(face, `${path}.mesh_faces[${index}]`)),
    conf: readNumber(input.conf, `${path}.conf`),
    world_frame: readEnum(input.world_frame, `${path}.world_frame`, ["court_Z0"] as const),
    translation_unit: readEnum(input.translation_unit, `${path}.translation_unit`, ["m"] as const),
    source: readString(input.source, `${path}.source`),
    reprojection_error_px:
      input.reprojection_error_px === null || input.reprojection_error_px === undefined
        ? null
        : readNumber(input.reprojection_error_px, `${path}.reprojection_error_px`),
    ambiguous: input.ambiguous === undefined ? false : readBoolean(input.ambiguous, `${path}.ambiguous`),
    confidence_provenance: readConfidenceProvenance(input.confidence_provenance, `${path}.confidence_provenance`),
    render_only: input.render_only === undefined ? false : readBoolean(input.render_only, `${path}.render_only`),
    not_for_detection_metrics:
      input.not_for_detection_metrics === undefined
        ? false
        : readBoolean(input.not_for_detection_metrics, `${path}.not_for_detection_metrics`),
    trust_band: readTrustBand(input.trust_band, `${path}.trust_band`),
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

function readFixedArray(value: unknown, path: string, length: number): unknown[] {
  const values = readArray(value, path);
  if (values.length !== length) throw new Error(`${path} must have length ${length}`);
  return values;
}

function readNumber(value: unknown, path: string, integer = false): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${path} must be a number`);
  if (integer && !Number.isInteger(value)) throw new Error(`${path} must be an integer`);
  return value;
}

function readNonNegativeNumber(value: unknown, path: string): number {
  const number = readNumber(value, path);
  if (number < 0) throw new Error(`${path} must be non-negative`);
  return number;
}

function readNonNegativeInteger(value: unknown, path: string): number {
  const number = readNumber(value, path, true);
  if (number < 0) throw new Error(`${path} must be non-negative`);
  return number;
}

function readUnitNumber(value: unknown, path: string): number {
  const number = readNumber(value, path);
  if (number < 0 || number > 1) throw new Error(`${path} must be in [0, 1]`);
  return number;
}

function readPositiveNumber(value: unknown, path: string): number {
  const number = readNumber(value, path);
  if (number <= 0) throw new Error(`${path} must be positive`);
  return number;
}

function readNumberRecord(value: unknown, path: string): Record<string, number> {
  if (value === undefined) return {};
  assertRecord(value, path);
  const output: Record<string, number> = {};
  for (const [key, entry] of Object.entries(value)) {
    output[key] = readNumber(entry, `${path}.${key}`, true);
  }
  return output;
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

function readEnum<T extends string>(value: unknown, path: string, allowed: readonly T[]): T {
  const text = readString(value, path);
  if (!allowed.includes(text as T)) throw new Error(`${path} must be one of: ${allowed.join(", ")}`);
  return text as T;
}

function readBoolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") throw new Error(`${path} must be a boolean`);
  return value;
}

function readVec2(value: unknown, path: string): Vec2 {
  const values = readFixedArray(value, path, 2);
  return [readNumber(values[0], `${path}[0]`), readNumber(values[1], `${path}[1]`)];
}

function readVec3(value: unknown, path: string): Vec3 {
  const values = readFixedArray(value, path, 3);
  return [readNumber(values[0], `${path}[0]`), readNumber(values[1], `${path}[1]`), readNumber(values[2], `${path}[2]`)];
}

function readBBox(value: unknown, path: string): [number, number, number, number] {
  const values = readFixedArray(value, path, 4);
  return [
    readNumber(values[0], `${path}[0]`),
    readNumber(values[1], `${path}[1]`),
    readNumber(values[2], `${path}[2]`),
    readNumber(values[3], `${path}[3]`),
  ];
}

function readFace(value: unknown, path: string): [number, number, number] {
  const values = readFixedArray(value, path, 3);
  return [
    readNonNegativeInteger(values[0], `${path}[0]`),
    readNonNegativeInteger(values[1], `${path}[1]`),
    readNonNegativeInteger(values[2], `${path}[2]`),
  ];
}

function validateFacesForVertices(faces: MeshFace[], vertexCount: number, path: string): void {
  for (const [faceIndex, face] of faces.entries()) {
    for (const [componentIndex, vertexIndex] of face.entries()) {
      if (vertexIndex >= vertexCount) {
        throw new Error(`${path}[${faceIndex}][${componentIndex}] must reference an existing mesh vertex`);
      }
    }
  }
}

function readRotationMatrix(value: unknown, path: string): Matrix3 {
  const rows = readFixedArray(value, path, 3).map((row, index) => readVec3(row, `${path}[${index}]`)) as Matrix3;
  assertOrthonormalRotation(rows, path);
  return rows;
}

function assertOrthonormalRotation(rows: Matrix3, path: string) {
  const tolerance = 1e-3;
  for (const row of rows) {
    const norm = Math.sqrt(row.reduce((total, entry) => total + entry * entry, 0));
    if (Math.abs(norm - 1) > tolerance) throw new Error(`${path} must be orthonormal`);
  }
  for (let left = 0; left < 3; left += 1) {
    for (let right = left + 1; right < 3; right += 1) {
      const dot = rows[left].reduce((total, entry, index) => total + entry * rows[right][index], 0);
      if (Math.abs(dot) > tolerance) throw new Error(`${path} must be orthonormal`);
    }
  }
  const determinant =
    rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1]) -
    rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0]) +
    rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0]);
  if (Math.abs(determinant - 1) > tolerance) throw new Error(`${path} determinant must be 1`);
}

function assertRecord(value: unknown, path: string): asserts value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error(`${path} must be an object`);
}
