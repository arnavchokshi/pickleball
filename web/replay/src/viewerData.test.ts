import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { gunzipSync } from "node:zlib";

import { afterEach, describe, expect, it, vi } from "vitest";

import * as viewerData from "./viewerData";
import {
  activeBallContactPlayerIds,
  activePaddleFramesForTime,
  ballCoverageKpiReadout,
  ballFrameForTime,
  ballInflectionBadge,
  bodyMeshDebugSnapshot,
  ballTrailSegmentsForTime,
  ballTrailPointsForTime,
  ballRenderInfoForTime,
  bodyMeshIndexWindowForTime,
  bodyMeshInterpolationReadout,
  bodyMeshInterpolationStats,
  bodyMeshStatusTileValue,
  clearBodyMeshChunkFetchCache,
  contactEventBadge,
  contactEventCount,
  decodeBodyMeshChunkBytes,
  displayFpsReadout,
  displayFpsReplayData,
  entityCoverageReadout,
  effectiveTrustBadge,
  frameForTime,
  fetchBodyMeshChunk,
  labelOverlayForTime,
  labelViewBox,
  parseBallInflections,
  parseBallArcEventsSelected,
  parseBodyMesh,
  parseBodyMeshFaces,
  parseBodyMeshIndex,
  parseContactWindows,
  parseLabelOverlayPayload,
  parsePhysicsRefinement,
  parseReviewedBounces,
  parseRallySpans,
  parseViewerManifest,
  parseVirtualWorld,
  preparedTimeFramesForLookup,
  paddleOpacityForStaleAge,
  playerPresenceForTime,
  playerTrailPointsForTime,
  playerCoverageDistanceM,
  playerCoverageStats,
  rallySpanFromContactWindows,
  solidBodyMeshFramesForTime,
  solidMeshRenderedPlayerCount,
  solidMeshFrameCount,
  startTimeFromSearch,
  timelineChaptersFromMarkers,
  timelineChaptersFromRallySpans,
  timelineEventJump,
  timelineMarkersFromArtifacts,
  trustBadgeColor,
  trustBandChipText,
  videoBallOverlayForTime,
  worldWarningsReadout,
  worldStats,
  resolveTimeSample,
  resolveCanonicalPlaybackTime,
  resolveViewerManifestUrls,
} from "./viewerData";

describe("canonical entity time resolution", () => {
  it("memoizes one sorted frame index and resolves nearest PTS with binary lookup", () => {
    const frames = [{ t: 2, id: "late" }, { t: 0, id: "early" }, { t: 1, id: "middle" }];
    const prepared = preparedTimeFramesForLookup(frames);

    expect(prepared.map((frame) => frame.id)).toEqual(["early", "middle", "late"]);
    expect(preparedTimeFramesForLookup(frames)).toBe(prepared);
    expect(resolveTimeSample(frames, 1.49, 0.6).sample?.id).toBe("middle");
    expect(resolveTimeSample(frames, 1.51, 0.6).sample?.id).toBe("late");
  });

  it("refuses an internal nearest-sample hold beyond the declared tolerance", () => {
    const resolved = resolveTimeSample([{ t: 0 }, { t: 1 }], 0.5, 0.12);
    expect(resolved.insideCoverage).toBe(true);
    expect(resolved.sample).toBeUndefined();
    expect(resolved.staleAgeSeconds).toBe(0.5);
  });

  it("keeps scrub, pause, and rate-change clock updates on one clamped canonical PTS", () => {
    const scrub = resolveCanonicalPlaybackTime(3.4, 3.0);
    const paused = resolveCanonicalPlaybackTime(scrub, 3.0);
    const rateChanged = resolveCanonicalPlaybackTime(paused, 3.0);

    expect(scrub).toBe(3.0);
    expect(paused).toBe(scrub);
    expect(rateChanged).toBe(scrub);
    expect(resolveCanonicalPlaybackTime(Number.NaN, 3.0, 1.25)).toBe(1.25);
  });

  it("marks all-null poses missing and retains only a prior floor anchor", () => {
    const player = {
      frames: [
        { t: 0, joints_world: [], floor_world_xyz: [1, 2, 0], track_world_xy: [1, 2] },
        { t: 0.1, joints_world: [], floor_world_xyz: null, track_world_xy: null },
      ],
    } as any;
    expect(playerPresenceForTime(player, 0.1)).toMatchObject({
      missingEvidence: true,
      reason: "all_null_pose",
      lastKnownFloor: [1, 2, 0],
    });
  });

  it("keeps paddle opacity through two source frames, decays it, then drops it", () => {
    expect(paddleOpacityForStaleAge(0.083)).toBe(1);
    expect(paddleOpacityForStaleAge(0.16)).toBeGreaterThan(0);
    expect(paddleOpacityForStaleAge(0.16)).toBeLessThan(1);
    expect(paddleOpacityForStaleAge(0.25)).toBe(0);
  });
});

const realMeshFixtureDir = resolve(process.cwd(), "../../tests/racketsport/fixtures/solid_mesh_real_window_000");

function fixtureJson(name: string): unknown {
  return JSON.parse(readFileSync(resolve(realMeshFixtureDir, name), "utf8"));
}

function arrayBufferFrom(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

function vectorDelta(left: [number, number, number], right: [number, number, number]): [number, number, number] {
  return [left[0] - right[0], left[1] - right[1], left[2] - right[2]];
}

function mhr70JointNames(): string[] {
  const names = Array.from({ length: 70 }, (_, index) => `mhr70_joint_${index}`);
  names[0] = "nose";
  names[9] = "left_hip";
  names[10] = "right_hip";
  return names;
}

function mhr70Joints({
  nose = [0, 0, 1.3],
  leftHip = [-0.2, 0, 0.9],
  rightHip = [0.2, 0, 0.9],
}: {
  nose?: [number, number, number];
  leftHip?: [number, number, number];
  rightHip?: [number, number, number];
} = {}): [number, number, number][] {
  const joints = Array.from({ length: 70 }, () => [0, 0, 0] as [number, number, number]);
  joints[0] = nose;
  joints[9] = leftHip;
  joints[10] = rightHip;
  return joints;
}

const manifest = {
  schema_version: 1,
  artifact_type: "racketsport_replay_viewer_manifest",
  clip: "clip_a",
  video_url: "/@fs/tmp/clip_a/input.mp4",
  virtual_world_url: "/@fs/tmp/clip_a/virtual_world.json",
  replay_scene_url: null,
  body_mesh_url: "/@fs/tmp/clip_a/body_mesh.json",
  body_mesh_index_url: "/@fs/tmp/clip_a/body_mesh_index.json",
  physics_refinement_url: "/@fs/tmp/clip_a/physics_refinement.json",
  contact_windows_url: "/@fs/tmp/clip_a/contact_windows.json",
  reviewed_bounces_url: "/@fs/tmp/clip_a/reviewed_ball_bounces.json",
  ball_inflections_url: null,
  events_selected_url: null,
  rally_spans_url: "/@fs/tmp/clip_a/rally_spans.json",
  court_evidence_url: null,
  court_calibration_url: null,
  skeleton_evidence_url: null,
  label_overlays: [
    {
      kind: "player_boxes",
      label: "prototype player boxes",
      url: "/@fs/tmp/clip_a/labels/players.json",
      trusted_for_metrics: false,
      not_ground_truth: true,
    },
  ],
  annotation_sources: [
    {
      kind: "person_ground_truth",
      clip_id: "task_2376761",
      url: "/@fs/tmp/task_2376761/person_ground_truth.json",
      trusted_for_metrics: true,
    },
  ],
  notes: ["review-only"],
};

afterEach(() => {
  clearBodyMeshChunkFetchCache();
  vi.unstubAllGlobals();
});

const world = {
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
  players: [
    {
      id: 1,
      side: "near",
      role: "left",
      representation: "mesh",
      frames: [
        {
          t: 0,
          track_world_xy: [0, 1],
          floor_world_xyz: [0, 1, 0],
          floor_source: "track_footpoint+smpl_world",
          foot_contact: { left: true, right: false },
          contact_locked: true,
          physics: "worldhmr_grounded_not_footlocked",
          floor_penetration_m: 0,
          joints_world: [[0, 1, 1]],
          mesh_vertices_world: [[0, 1, 0]],
          joint_count: 1,
          mesh_vertex_count: 1,
        },
        {
          t: 1,
          track_world_xy: [0.5, 1.5],
          floor_world_xyz: [0.5, 1.5, 0],
          floor_source: "track_footpoint",
          contact_locked: false,
          floor_penetration_m: 0,
          joints_world: [],
          mesh_vertices_world: [],
          joint_count: 0,
          mesh_vertex_count: 0,
        },
      ],
    },
  ],
  ball: { source: "tracknet", frames: [] },
  paddles: [],
  summary: {
    player_count: 1,
    mesh_player_count: 1,
    mesh_player_frame_count: 1,
    joint_player_frame_count: 1,
    track_only_player_frame_count: 1,
    floor_placed_player_frame_count: 2,
    floor_contact_player_frame_count: 1,
    max_floor_penetration_m: 0,
    max_abs_floor_offset_m: 0,
    physics_modes: ["worldhmr_grounded_not_footlocked"],
    ball_frame_count: 0,
    approx_ball_frame_count: 0,
    paddle_player_count: 0,
    paddle_frame_count: 0,
    ambiguous_paddle_frame_count: 0,
    warnings: [],
  },
};

const physics = {
  schema_version: 1,
  artifact_type: "racketsport_physics_refinement",
  physics: "cpu_fallback_scaffold",
  foot2_done: false,
  must_not_mark_done_verified: true,
  constraint_summary: {
    contact_frames: 0,
    max_contact_slide_m: 0,
    max_floor_penetration_m: 0,
    inter_player_penetration_frames: 0,
    max_inter_player_penetration_m: 0,
  },
  execution_plan: {
    mode: "cpu_fallback",
    will_run_mjx: false,
    reason: "Using deterministic CPU fallback scaffold; no physics model or simulator was run.",
  },
};

const contactWindows = {
  schema_version: 1,
  events: [
    {
      type: "contact",
      t: 0.1,
      frame: 3,
      player_id: null,
      confidence: 1,
      sources: { audio: 0, wrist_vel: 0, ball_inflection: 0, human_review: 1 },
      window: { t0: 0, t1: 0.2, importance: 1 },
    },
    {
      type: "contact",
      t: 1.1,
      frame: 33,
      player_id: 2,
      confidence: 1,
      sources: { audio: 0, wrist_vel: 0, ball_inflection: 0, human_review: 1 },
      window: { t0: 1, t1: 1.2, importance: 1 },
    },
  ],
};

const bodyMesh = {
  schema_version: 1,
  artifact_type: "racketsport_body_mesh",
  clip: "clip_a",
  model: "sam3dbody_world_joints",
  fps: 30,
  world_frame: "court_Z0",
  faces_ref: "mhr_faces_static",
  mesh_faces: [
    [0, 1, 2],
    [0, 2, 3],
  ],
  joint_names: ["left_wrist"],
  players: [
    {
      id: 2,
      frames: [
        {
          frame_idx: 33,
          t: 1.1,
          source_window_index: 1,
          blend_weight: 0.42,
          joints_world: [[2, 1, 1.2]],
          joint_conf: [0.95],
          mesh_vertices_world: [
            [1.9, 0.9, 0.2],
            [2.1, 0.9, 0.2],
            [2.1, 1.1, 1.6],
            [1.9, 1.1, 1.6],
          ],
          smplx_params: {
            global_orient: [0, 0, 0],
            body_pose: [0.1],
            left_hand_pose: [0.2],
            right_hand_pose: [0.3],
            betas: [0.4],
            transl_world: [2, 1, 0],
          },
          reasons: ["contact_window"],
        },
      ],
    },
  ],
  summary: {
    mesh_frame_count: 1,
    player_count: 1,
    contact_window_count: 1,
  },
};

function twoComputedFrameBodyMesh({
  secondT = 1 + 1 / 30,
  secondFrameIdx = 31,
  secondWindow = 0,
  mismatchedVertexCount = false,
}: {
  secondT?: number;
  secondFrameIdx?: number;
  secondWindow?: number;
  mismatchedVertexCount?: boolean;
} = {}) {
  return {
    ...bodyMesh,
    mesh_faces: [[0, 1, 2]],
    players: [
      {
        id: 2,
        frames: [
          {
            frame_idx: 30,
            t: 1,
            source_window_index: 0,
            blend_weight: 1,
            joints_world: [[1, 1, 1]],
            joint_conf: [0.9],
            mesh_vertices_world: [
              [1, 1, 0],
              [2, 1, 0],
              [1, 2, 0],
            ],
            smplx_params: {},
            reasons: ["contact_dense_hitter_window"],
          },
          {
            frame_idx: secondFrameIdx,
            t: secondT,
            source_window_index: secondWindow,
            blend_weight: 1,
            joints_world: [[2, 2, 2]],
            joint_conf: [0.8],
            mesh_vertices_world: mismatchedVertexCount
              ? [
                  [2, 2, 1],
                  [3, 2, 1],
                  [2, 3, 1],
                  [3, 3, 1],
                ]
              : [
                  [2, 2, 1],
                  [3, 2, 1],
                  [2, 3, 1],
                ],
            smplx_params: {},
            reasons: ["contact_dense_hitter_window"],
          },
        ],
      },
    ],
    summary: {
      mesh_frame_count: 2,
      player_count: 1,
      contact_window_count: 1,
    },
  };
}

const contactWorld = {
  ...world,
  players: [
    {
      id: 1,
      representation: "track_only",
      frames: [
        {
          t: 0.1,
          track_world_xy: [0, 1],
          floor_world_xyz: [0, 1, 0],
          joints_world: [],
          mesh_vertices_world: [],
          joint_count: 0,
          mesh_vertex_count: 0,
        },
      ],
    },
    {
      id: 2,
      representation: "track_only",
      frames: [
        {
          t: 0.1,
          track_world_xy: [2, 1],
          floor_world_xyz: [2, 1, 0],
          joints_world: [],
          mesh_vertices_world: [],
          joint_count: 0,
          mesh_vertex_count: 0,
        },
      ],
    },
  ],
  ball: { source: "tracknet", frames: [{ t: 0.1, xy: [12, 18], conf: 0.92, world_xyz: [0.15, 1, 0], visible: true }] },
};

const sampledLabelOverlay = {
  schema_version: 1,
  not_ground_truth: true,
  status: "draft_prototype_unverified",
  frames: {
    source_fps: 30,
    sample_every_frames: 30,
    source_resolution: [1920, 1080],
  },
  annotation: {
    items: [
      { frame: "frame_000001.jpg", bbox_xyxy: [100, 50, 150, 180], id: "p1" },
      { frame: "frame_000003.jpg", bbox_xyxy: [700, 320, 955, 538], id: "p2" },
    ],
  },
};

const realCoachingCardFactsFixture = {
  artifact_type: "coaching_card_facts",
  rally_scope: "rally_spans",
  priority_rule: [
    "contact_count_when_present",
    "kitchen_proximity_when_positive",
    "distance_covered_when_positive",
    "p95_speed_when_positive",
    "dominant_zone_occupancy",
  ],
  facts: [
    {
      coverage_fraction: 0.9666666666666667,
      frames_total: 300,
      frames_used: 290,
      metric: "contact_count",
      player_id: "1",
      rally_id: "rally_000",
      rally_scope: "rally_spans",
      trust: "ok",
      unit: "count",
      value: 3,
    },
    {
      coverage_fraction: 1,
      frames_total: 300,
      frames_used: 300,
      metric: "contact_count",
      player_id: "2",
      rally_id: "rally_000",
      rally_scope: "rally_spans",
      trust: "ok",
      unit: "count",
      value: 7,
    },
    {
      coverage_fraction: 0.91,
      frames_total: 300,
      frames_used: 273,
      metric: "contact_count",
      player_id: "3",
      rally_id: "rally_000",
      rally_scope: "rally_spans",
      trust: "ok",
      unit: "count",
      value: 8,
    },
    {
      coverage_fraction: 0.8133333333333334,
      frames_total: 300,
      frames_used: 244,
      metric: "contact_count",
      player_id: "4",
      rally_id: "rally_000",
      rally_scope: "rally_spans",
      trust: "ok",
      unit: "count",
      value: 6,
    },
  ],
};

describe("viewer data contracts", () => {
  it("parses the local replay viewer manifest with overlay trust flags", () => {
    expect(parseViewerManifest(manifest)).toEqual(manifest);
  });

  it("resolves every relative child URL against an http manifest without checkout assumptions", () => {
    const resolved = resolveViewerManifestUrls(
      parseViewerManifest({
        ...manifest,
        video_url: "media/source.mp4",
        virtual_world_url: "world/virtual_world.json",
        replay_scene_url: "./scene/replay_scene.json",
        label_overlays: [{ ...manifest.label_overlays[0], url: "labels/players.json" }],
      }),
      "https://cdn.example.com/runs/clip/replay_viewer_manifest.json",
    );

    expect(resolved.video_url).toBe("https://cdn.example.com/runs/clip/media/source.mp4");
    expect(resolved.virtual_world_url).toBe("https://cdn.example.com/runs/clip/world/virtual_world.json");
    expect(resolved.replay_scene_url).toBe("https://cdn.example.com/runs/clip/scene/replay_scene.json");
    expect(resolved.label_overlays[0].url).toBe("https://cdn.example.com/runs/clip/labels/players.json");
  });

  it("preserves file-scheme parity for relative and root-absolute manifest children", () => {
    const resolved = resolveViewerManifestUrls(
      parseViewerManifest({
        ...manifest,
        video_url: "media/source.mp4",
        virtual_world_url: "/shared/virtual_world.json",
      }),
      "file:///Users/reviewer/run/replay_viewer_manifest.json",
    );

    expect(resolved.video_url).toBe("file:///Users/reviewer/run/media/source.mp4");
    expect(resolved.virtual_world_url).toBe("file:///shared/virtual_world.json");
  });

  it("resolves relative manifest bases while preserving already-absolute children", () => {
    const resolved = resolveViewerManifestUrls(
      parseViewerManifest({
        ...manifest,
        video_url: "media/source.mp4",
        virtual_world_url: "https://assets.example.com/world.json",
        contact_surfaces_url: "derived/contact_surfaces.json",
        target_zones_url: "derived/target_zones.json",
        ghost_positions_url: "derived/ghost_positions.json",
      }),
      "/runs/clip/replay_viewer_manifest.json",
    );

    expect(resolved.video_url).toBe("/runs/clip/media/source.mp4");
    expect(resolved.virtual_world_url).toBe("https://assets.example.com/world.json");
    expect(resolved.contact_surfaces_url).toBe("/runs/clip/derived/contact_surfaces.json");
    expect(resolved.target_zones_url).toBe("/runs/clip/derived/target_zones.json");
    expect(resolved.ghost_positions_url).toBe("/runs/clip/derived/ghost_positions.json");
  });

  it("accepts an optional coaching-card facts manifest pointer without requiring old manifests to have it", () => {
    expect(parseViewerManifest(manifest).coaching_card_facts_url).toBeUndefined();
    expect(
      parseViewerManifest({
        ...manifest,
        coaching_card_facts_url: "/@fs/tmp/clip_a/coaching_card_facts.json",
        rally_metrics_url: "/@fs/tmp/clip_a/rally_metrics.json",
        rally_spans_url: "/@fs/tmp/clip_a/authoritative_rally_spans.json",
      }),
    ).toMatchObject({
      coaching_card_facts_url: "/@fs/tmp/clip_a/coaching_card_facts.json",
      rally_metrics_url: "/@fs/tmp/clip_a/rally_metrics.json",
      rally_spans_url: "/@fs/tmp/clip_a/authoritative_rally_spans.json",
    });
  });

  it("accepts optional shot-trails manifest pointers without requiring old manifests to have them", () => {
    const parsedWithoutShots = parseViewerManifest(manifest);
    expect(parsedWithoutShots.shots_url).toBeUndefined();
    expect(parsedWithoutShots.ball_arc_solved_url).toBeUndefined();
    expect(parsedWithoutShots.auto_bounce_candidates_url).toBeUndefined();
    expect(parsedWithoutShots.ball_bounce_candidates_url).toBeUndefined();
    expect(parsedWithoutShots.ball_arc_render_url).toBeUndefined();
    expect(parsedWithoutShots.ball_flight_sanity_url).toBeUndefined();

    expect(
      parseViewerManifest({
        ...manifest,
        shots_url: "/@fs/tmp/clip_a/shots.json",
        ball_arc_solved_url: "/@fs/tmp/clip_a/ball_track_arc_solved.json",
        ball_arc_render_url: "/@fs/tmp/clip_a/ball_arc_render.json",
        auto_bounce_candidates_url: "/@fs/tmp/clip_a/ball_bounce_candidates.json",
        ball_bounce_candidates_url: "/@fs/tmp/clip_a/ball_bounce_candidates.json",
        ball_flight_sanity_url: "/@fs/tmp/clip_a/ball_flight_sanity.json",
      }),
    ).toMatchObject({
      shots_url: "/@fs/tmp/clip_a/shots.json",
      ball_arc_solved_url: "/@fs/tmp/clip_a/ball_track_arc_solved.json",
      ball_arc_render_url: "/@fs/tmp/clip_a/ball_arc_render.json",
      auto_bounce_candidates_url: "/@fs/tmp/clip_a/ball_bounce_candidates.json",
      ball_bounce_candidates_url: "/@fs/tmp/clip_a/ball_bounce_candidates.json",
      ball_flight_sanity_url: "/@fs/tmp/clip_a/ball_flight_sanity.json",
    });
  });

  it("strictly parses real Wolverine coaching-card facts without reshaping missing fields", () => {
    expect(typeof (viewerData as any).parseCoachingCardFacts).toBe("function");

    const parsed = (viewerData as any).parseCoachingCardFacts(realCoachingCardFactsFixture);

    expect(parsed.artifact_type).toBe("coaching_card_facts");
    expect(parsed.priority_rule).toEqual(realCoachingCardFactsFixture.priority_rule);
    expect(parsed.facts).toHaveLength(4);
    expect(parsed.facts[0]).toEqual(realCoachingCardFactsFixture.facts[0]);
  });

  it("rejects coaching-card facts that omit a trust badge", () => {
    expect(typeof (viewerData as any).parseCoachingCardFacts).toBe("function");
    const invalid = {
      ...realCoachingCardFactsFixture,
      facts: [{ ...realCoachingCardFactsFixture.facts[0], trust: undefined }],
    };

    expect(() => (viewerData as any).parseCoachingCardFacts(invalid)).toThrow("coaching_card_facts.facts[0].trust");
  });

  it("maps coaching fact trust states onto the existing trust-chip visual tiers", () => {
    expect(typeof (viewerData as any).coachingTrustChipClass).toBe("function");

    expect((viewerData as any).coachingTrustChipClass("ok")).toBe("verified");
    expect((viewerData as any).coachingTrustChipClass("estimated")).toBe("preview");
    expect((viewerData as any).coachingTrustChipClass("unverified_cue")).toBe("low_confidence");
  });

  it("parses virtual_world floor placement and summarizes scene stats", () => {
    const parsed = parseVirtualWorld(world);

    expect(parsed.players[0].frames[0].floor_source).toBe("track_footpoint+smpl_world");
    expect(parsed.players[0].frames[0].contact_locked).toBe(true);
    expect(worldStats(parsed)).toEqual({
      players: 1,
      meshFrames: 1,
      floorPlacedFrames: 2,
      contactFrames: 1,
      maxFloorPenetrationM: 0,
      physicsModes: ["worldhmr_grounded_not_footlocked"],
    });
  });

  it("maps true mesh absence and indexed-sidecar mesh warnings to distinct notice copy", () => {
    expect(
      worldWarningsReadout({
        summary: { ...world.summary, warnings: ["missing_mesh_vertices"] },
      }),
    ).toContain("missing mesh vertices");
    expect(
      worldWarningsReadout({
        summary: { ...world.summary, warnings: ["missing_embedded_mesh_vertices"] },
      }),
    ).toContain("mesh rendered from mesh index; embedded world is skeleton-only");
  });

  it("accepts every Python-schema ball source, plus future/unknown sources, as opaque metadata", () => {
    // Mirrors threed/racketsport/schemas/__init__.py's BallTrack.source /
    // VirtualWorldBall.source Literal set, plus one made-up value standing
    // in for whatever detector gets added next. `ball.source` only drives
    // display text in the viewer (BALL's trust badge comes from
    // `ball.trust_band`), so parseVirtualWorld must not fail-closed on any
    // non-empty string here -- that is exactly the "wasb" bug this test
    // guards against recurring for the next new source name.
    const knownSources = ["tracknet", "wasb", "fused", "tap", "pbmat", "totnet", "vn_trajectories"];
    for (const source of [...knownSources, "some_future_detector_v9"]) {
      const parsed = parseVirtualWorld({ ...world, ball: { source, frames: [] } });
      expect(parsed.ball.source).toBe(source);
    }
  });

  it("parses Python-shaped paddle frames instead of treating them as unknown UI data", () => {
    const withPaddle = {
      ...world,
      paddles: [
        {
          player_id: 1,
          paddle_dims_in: { length: 15.5, width: 7.5 },
          frames: [
            {
              t: 0.1,
              pose_se3: {
                R: [
                  [1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                ],
                t: [0.45, -1.9, 0.85],
              },
              mesh_vertices_world: [
                [0.35, -1.7, 0.85],
                [0.55, -1.7, 0.85],
                [0.55, -2.1, 0.85],
              ],
              mesh_faces: [[0, 1, 2]],
              conf: 0.72,
              world_frame: "court_Z0",
              translation_unit: "m",
              source: "pnp_ippe:court_Z0",
              reprojection_error_px: 2.1,
              ambiguous: false,
            },
          ],
        },
      ],
    };

    const parsed = parseVirtualWorld(withPaddle);

    expect(parsed.paddles[0].frames[0].pose_se3.t).toEqual([0.45, -1.9, 0.85]);
    expect(parsed.paddles[0].frames[0].mesh_faces).toEqual([[0, 1, 2]]);
  });

  it("selects a real wrist-proxy paddle frame for rendering at scrub time", () => {
    const stagedWorld = parseVirtualWorld(
      JSON.parse(
        readFileSync(
          resolve(process.cwd(), "../../runs/manager_rebuild_wolverine_20260702T23Z/virtual_world.json"),
          "utf8",
        ),
      ),
    );

    const active = activePaddleFramesForTime(stagedWorld, 2.0);

    expect(stagedWorld.summary.paddle_frame_count).toBe(1107);
    expect(active).toHaveLength(4);
    expect(active[0]).toMatchObject({
      playerId: 1,
      estimated: true,
      frame: { source: "wrist_proxy:court_Z0" },
    });
  });

  it("parses the physics-refinement scaffold status", () => {
    expect(parsePhysicsRefinement(physics).execution_plan.will_run_mjx).toBe(false);
  });

  it("uses contact windows to select the active 3D player", () => {
    const parsedWorld = parseVirtualWorld(contactWorld);
    const parsedContacts = parseContactWindows(contactWindows);

    expect(contactEventCount(parsedContacts)).toBe(2);
    expect(activeBallContactPlayerIds(parsedWorld, parsedContacts, 0.1)).toEqual(new Set([1]));
    expect(activeBallContactPlayerIds(parsedWorld, parsedContacts, 1.1)).toEqual(new Set([2]));
    expect(activeBallContactPlayerIds(parsedWorld, parsedContacts, 0.7)).toEqual(new Set());
  });

  it("parses body_mesh.json and selects current solid meshes even when contact windows are inactive", () => {
    const parsedBodyMesh = parseBodyMesh(bodyMesh);
    const parsedContacts = parseContactWindows(contactWindows);
    const inactiveContacts = parseContactWindows({
      schema_version: 1,
      events: [
        {
          type: "contact",
          t: 8,
          frame: 240,
          player_id: 2,
          confidence: 1,
          sources: { audio: 0, wrist_vel: 0, ball_inflection: 0, human_review: 1 },
          window: { t0: 7.95, t1: 8.05, importance: 1 },
        },
      ],
    });

    expect(parsedBodyMesh.mesh_faces).toEqual([
      [0, 1, 2],
      [0, 2, 3],
    ]);
    expect(parsedBodyMesh.players[0].frames[0].mesh_faces).toEqual(parsedBodyMesh.mesh_faces);
    expect(parsedBodyMesh.players[0].frames[0].blend_weight).toBe(0.42);
    expect(solidBodyMeshFramesForTime(parsedBodyMesh, parsedContacts, 1.1)).toMatchObject([
      {
        playerId: 2,
        frame: {
          frame_idx: 33,
          blend_weight: 0.42,
          mesh_faces: [
            [0, 1, 2],
            [0, 2, 3],
          ],
        },
      },
    ]);
    expect(solidBodyMeshFramesForTime(parsedBodyMesh, inactiveContacts, 1.1)).toMatchObject([
      {
        playerId: 2,
        frame: {
          frame_idx: 33,
          blend_weight: 0.42,
        },
        presenceOpacity: 1,
      },
    ]);
    expect(solidBodyMeshFramesForTime(parsedBodyMesh, parsedContacts, 0.7)).toEqual([]);
  });

  it("keeps a solid mesh frame visible with fade opacity just outside its time span", () => {
    const parsedBodyMesh = parseBodyMesh(bodyMesh);

    const fadeIn = solidBodyMeshFramesForTime(parsedBodyMesh, null, 0.99);
    const visible = solidBodyMeshFramesForTime(parsedBodyMesh, null, 1.1);
    const fadeOut = solidBodyMeshFramesForTime(parsedBodyMesh, null, 1.18);

    expect(fadeIn).toHaveLength(1);
    expect(fadeIn[0].presenceOpacity).toBeGreaterThan(0);
    expect(fadeIn[0].presenceOpacity).toBeLessThan(1);
    expect(visible[0].presenceOpacity).toBe(1);
    expect(fadeOut[0].presenceOpacity).toBeGreaterThan(0);
    expect(fadeOut[0].presenceOpacity).toBeLessThan(1);
    expect(solidBodyMeshFramesForTime(parsedBodyMesh, null, 1.3)).toEqual([]);
  });

  it("keeps base playback on computed mesh frames until 2x display data is explicitly built", () => {
    const parsedBodyMesh = parseBodyMesh(twoComputedFrameBodyMesh());
    const baseActive = solidBodyMeshFramesForTime(parsedBodyMesh, null, 1 + 1 / 60);
    const doubled = displayFpsReplayData(parseVirtualWorld(world), parsedBodyMesh, true);
    const doubledActive = solidBodyMeshFramesForTime(doubled.bodyMesh, null, 1 + 1 / 60);

    expect(baseActive).toHaveLength(1);
    expect(baseActive[0].frame.mesh_interpolated).toBe(false);
    expect(baseActive[0].frame.frame_idx).toBe(30);
    expect(doubledActive).toHaveLength(1);
    expect(doubledActive[0].frame.mesh_interpolated).toBe(true);
    expect(doubledActive[0].frame.interpolation).toMatchObject({ from_frame_idx: 30, to_frame_idx: 31, alpha: 0.5 });
    expect(doubledActive[0].frame.mesh_vertices_world).toEqual([
      [1.5, 1.5, 0.5],
      [2.5, 1.5, 0.5],
      [1.5, 2.5, 0.5],
    ]);
    expect(doubledActive[0].frame.joints_world).toEqual([[1.5, 1.5, 1.5]]);
  });

  it("refuses solid mesh interpolation across wide gaps, window boundaries, and mismatched vertex counts", () => {
    const wideGap = solidBodyMeshFramesForTime(parseBodyMesh(twoComputedFrameBodyMesh({ secondT: 1.2, secondFrameIdx: 36 })), null, 1.1);
    expect(wideGap[0].frame.mesh_interpolated).toBe(false);

    const boundary = solidBodyMeshFramesForTime(parseBodyMesh(twoComputedFrameBodyMesh({ secondWindow: 1 })), null, 1 + 1 / 60);
    expect(boundary[0].frame.mesh_interpolated).toBe(false);

    const mismatched = solidBodyMeshFramesForTime(parseBodyMesh(twoComputedFrameBodyMesh({ mismatchedVertexCount: true })), null, 1 + 1 / 60);
    expect(mismatched[0].frame.mesh_interpolated).toBe(false);
  });

  it("reuses the same held computed mesh frame buffers between arbitrary display times", () => {
    const parsedBodyMesh = parseBodyMesh(twoComputedFrameBodyMesh());
    const first = solidBodyMeshFramesForTime(parsedBodyMesh, null, 1 + 1 / 120)[0].frame;
    const verticesRef = first.mesh_vertices_world;
    const jointsRef = first.joints_world;
    const firstX = first.mesh_vertices_world[0][0];
    const second = solidBodyMeshFramesForTime(parsedBodyMesh, null, 1 + 3 / 120)[0].frame;

    expect(second).toBe(first);
    expect(second.mesh_vertices_world).toBe(verticesRef);
    expect(second.joints_world).toBe(jointsRef);
    expect(second.mesh_vertices_world[0][0]).toBe(firstX);
    expect(second.interpolation).toBeNull();
  });

  it("reports interpolation eligibility counts for the mesh honesty indicator", () => {
    const parsedBodyMesh = parseBodyMesh(twoComputedFrameBodyMesh());
    expect(bodyMeshInterpolationStats(parsedBodyMesh)).toMatchObject({
      computedFrameCount: 2,
      eligiblePairCount: 1,
      heldPairCount: 0,
      gapRefusedPairCount: 0,
      boundaryRefusedPairCount: 0,
      mismatchedVertexRefusedPairCount: 0,
      displayMultiplier: 2,
    });
    expect(bodyMeshInterpolationReadout(parsedBodyMesh)).toBe("mesh: computed 30fps + interpolated x2 (1 safe pair)");

    const refused = parseBodyMesh(twoComputedFrameBodyMesh({ secondT: 1.2, secondFrameIdx: 36, secondWindow: 1 }));
    expect(bodyMeshInterpolationStats(refused)).toMatchObject({
      eligiblePairCount: 0,
      heldPairCount: 0,
      gapRefusedPairCount: 1,
      boundaryRefusedPairCount: 1,
    });
    expect(bodyMeshInterpolationReadout(refused)).toBe("mesh: computed sparse (2 frames, 0 safe pairs)");
  });

  it("holds sparse same-window mesh presence on the previous computed frame without per-frame gaps", () => {
    const parsedBodyMesh = parseBodyMesh(twoComputedFrameBodyMesh({ secondT: 1.1, secondFrameIdx: 33 }));

    const activeFrames = [1.0, 1 + 1 / 30, 1 + 2 / 30, 1.099, 1.1].map((time) =>
      solidBodyMeshFramesForTime(parsedBodyMesh, null, time),
    );

    expect(activeFrames.every((frames) => frames.length === 1)).toBe(true);
    expect(activeFrames.slice(0, -1).map((frames) => frames[0].frame.frame_idx)).toEqual([30, 30, 30, 30]);
    expect(activeFrames.at(-1)?.[0].frame.frame_idx).toBe(33);
    expect(activeFrames.some((frames) => frames[0].frame.mesh_interpolated)).toBe(false);
    expect(bodyMeshInterpolationStats(parsedBodyMesh)).toMatchObject({
      computedFrameCount: 2,
      eligiblePairCount: 0,
      heldPairCount: 1,
    });
    expect(bodyMeshInterpolationReadout(parsedBodyMesh)).toBe("mesh: computed sparse (2 frames, held 1 gap)");
  });

  it("aligns held meshes with an object-level transform without deforming relative geometry", () => {
    const parsedBodyMesh = parseBodyMesh({
      ...bodyMesh,
      mesh_faces: [[0, 1, 2]],
      joint_names: ["left_hip", "right_hip"],
      players: [
        {
          id: 1,
          frames: [
            {
              frame_idx: 30,
              t: 1,
              source_window_index: 0,
              blend_weight: 1,
              joints_world: [
                [1, 1, 1],
                [3, 1, 1],
              ],
              joint_conf: [1, 1],
              mesh_vertices_world: [
                [1, 1, 0],
                [3, 1, 0],
                [1, 2, 0],
              ],
              smplx_params: {},
              reasons: ["contact_window"],
            },
          ],
        },
      ],
    });
    const parsedWorld = parseVirtualWorld({
      ...world,
      joint_names: ["left_hip", "right_hip"],
      players: [
        {
          ...world.players[0],
          id: 1,
          frames: [
            {
              ...world.players[0].frames[0],
              t: 1.05,
              joints_world: [
                [4, 6, 1],
                [6, 6, 1],
              ],
              joint_conf: [1, 1],
              mesh_ref: { artifact: "body_mesh.json", player_id: 1, frame_idx: 30, t: 1 },
            },
          ],
        },
      ],
    });

    const active = solidBodyMeshFramesForTime(parsedBodyMesh, null, 1.05, parsedWorld);

    expect(active).toHaveLength(1);
    expect(active[0].alignmentDebug).toMatchObject({ applied: true, delta: [3, 5, 0] });
    expect(active[0].renderTranslation).toEqual([3, 5, 0]);
    expect(active[0].frame.mesh_vertices_world).toEqual(parsedBodyMesh.players[0].frames[0].mesh_vertices_world);
    expect(vectorDelta(active[0].frame.mesh_vertices_world[0], active[0].frame.mesh_vertices_world[1])).toEqual([-2, 0, 0]);
    expect(vectorDelta(active[0].frame.mesh_vertices_world[0], active[0].frame.mesh_vertices_world[2])).toEqual([0, -1, 0]);
  });

  it("uses world MHR70 joint names for real-shaped chunked meshes that omit mesh joint_names", () => {
    const jointNames = mhr70JointNames();
    const parsedBodyMesh = parseBodyMesh({
      ...bodyMesh,
      mesh_faces: [[0, 1, 2]],
      joint_names: [],
      players: [
        {
          id: 1,
          frames: [
            {
              frame_idx: 30,
              t: 1,
              source_window_index: 0,
              blend_weight: 1,
              joints_world: mhr70Joints({
                nose: [0, 0, 1.34],
                leftHip: [-0.2, 0, 0.9],
                rightHip: [0.2, 0, 0.88],
              }),
              joint_conf: Array(70).fill(1),
              mesh_vertices_world: [
                [-0.25, 0, 0.05],
                [0.25, 0, 0.05],
                [0, 0.3, 1.4],
              ],
              smplx_params: { transl_world: [0, 0, 1.34] },
              reasons: ["real_shape_no_mesh_joint_names"],
            },
          ],
        },
      ],
    });
    const parsedWorld = parseVirtualWorld({
      ...world,
      joint_names: jointNames,
      players: [
        {
          ...world.players[0],
          id: 1,
          frames: [
            {
              ...world.players[0].frames[0],
              t: 1,
              joints_world: mhr70Joints({
                nose: [0, 0, 1.33],
                leftHip: [-0.2, 0, 0.92],
                rightHip: [0.2, 0, 0.9],
              }),
              joint_conf: Array(70).fill(1),
              mesh_ref: { artifact: "body_mesh.json", player_id: 1, frame_idx: 30, t: 1 },
            },
          ],
        },
      ],
    });

    const active = solidBodyMeshFramesForTime(parsedBodyMesh, null, 1, parsedWorld);

    expect(active).toHaveLength(1);
    expect(active[0].alignmentDebug).toMatchObject({
      applied: true,
      reason: "skeleton_root",
      delta: [0, 0, 0.02],
      floor_guard_applied: false,
    });
    expect(active[0].renderTranslation).toEqual([0, 0, 0.02]);
    const translatedLowest = Math.min(...active[0].frame.mesh_vertices_world.map((vertex) => vertex[2] + active[0].renderTranslation[2]));
    expect(translatedLowest).toBeCloseTo(0.07);
  });

  it("does not translate a mesh when no reliable named mesh root exists", () => {
    const parsedBodyMesh = parseBodyMesh({
      ...bodyMesh,
      mesh_faces: [[0, 1, 2]],
      joint_names: [],
      players: [
        {
          id: 1,
          frames: [
            {
              frame_idx: 30,
              t: 1,
              source_window_index: 0,
              blend_weight: 1,
              joints_world: [[0, 0, 1.4]],
              joint_conf: [1],
              mesh_vertices_world: [
                [-0.25, 0, 0.02],
                [0.25, 0, 0.02],
                [0, 0.3, 1.3],
              ],
              smplx_params: { transl_world: [0, 0, 1.4] },
              reasons: ["real_shape_no_mesh_joint_names"],
            },
          ],
        },
      ],
    });
    const parsedWorld = parseVirtualWorld({
      ...world,
      joint_names: mhr70JointNames(),
      players: [
        {
          ...world.players[0],
          id: 1,
          frames: [
            {
              ...world.players[0].frames[0],
              t: 1,
              joints_world: mhr70Joints(),
              joint_conf: Array(70).fill(1),
              mesh_ref: { artifact: "body_mesh.json", player_id: 1, frame_idx: 30, t: 1 },
            },
          ],
        },
      ],
    });

    const active = solidBodyMeshFramesForTime(parsedBodyMesh, null, 1, parsedWorld);

    expect(active).toHaveLength(1);
    expect(active[0].alignmentDebug).toMatchObject({ applied: false, reason: "missing_mesh_root" });
    expect(active[0].renderTranslation).toEqual([0, 0, 0]);
    expect(active[0].frame.mesh_vertices_world).toEqual(parsedBodyMesh.players[0].frames[0].mesh_vertices_world);
  });

  it("lifts the object transform when hip-aligned mesh vertices would sink below the floor plane", () => {
    const jointNames = mhr70JointNames();
    const parsedBodyMesh = parseBodyMesh({
      ...bodyMesh,
      mesh_faces: [[0, 1, 2]],
      joint_names: [],
      players: [
        {
          id: 1,
          frames: [
            {
              frame_idx: 30,
              t: 1,
              source_window_index: 0,
              blend_weight: 1,
              joints_world: mhr70Joints({
                nose: [0, 0, 1.45],
                leftHip: [-0.2, 0, 1.0],
                rightHip: [0.2, 0, 1.0],
              }),
              joint_conf: Array(70).fill(1),
              mesh_vertices_world: [
                [-0.25, 0, -0.1],
                [0.25, 0, -0.1],
                [0, 0.3, 1.4],
              ],
              smplx_params: {},
              reasons: ["floor_guard_probe"],
            },
          ],
        },
      ],
    });
    const parsedWorld = parseVirtualWorld({
      ...world,
      joint_names: jointNames,
      players: [
        {
          ...world.players[0],
          id: 1,
          frames: [
            {
              ...world.players[0].frames[0],
              t: 1,
              joints_world: mhr70Joints({
                nose: [0, 0, 1.35],
                leftHip: [-0.2, 0, 0.9],
                rightHip: [0.2, 0, 0.9],
              }),
              joint_conf: Array(70).fill(1),
              mesh_ref: { artifact: "body_mesh.json", player_id: 1, frame_idx: 30, t: 1 },
            },
          ],
        },
      ],
    });

    const active = solidBodyMeshFramesForTime(parsedBodyMesh, null, 1, parsedWorld);
    const translatedLowest = Math.min(...active[0].frame.mesh_vertices_world.map((vertex) => vertex[2] + active[0].renderTranslation[2]));
    const snapshot = bodyMeshDebugSnapshot({
      bodyMeshIndex: null,
      bodyMesh: parsedBodyMesh,
      world: parsedWorld,
      currentTime: 1,
      loadStatus: { state: "loaded", label: "loaded" },
      activeBodyMeshes: active,
    });

    expect(active[0].alignmentDebug).toMatchObject({
      applied: true,
      floor_guard_applied: true,
      floor_lift_m: 0.2,
      delta: [0, 0, 0.1],
    });
    expect(active[0].renderTranslation).toEqual([0, 0, 0.1]);
    expect(translatedLowest).toBeCloseTo(0);
    expect(snapshot.alignment_floor_guard_count).toBe(1);
  });

  it("falls back to the mesh position when the current skeleton root is missing", () => {
    const parsedBodyMesh = parseBodyMesh(twoComputedFrameBodyMesh());
    const parsedWorld = parseVirtualWorld({
      ...world,
      players: [
        {
          ...world.players[0],
          id: 2,
          frames: [
            {
              ...world.players[0].frames[0],
              t: 1,
              joints_world: [],
              joint_conf: [],
              mesh_ref: { artifact: "body_mesh.json", player_id: 2, frame_idx: 30, t: 1 },
            },
          ],
        },
      ],
    });

    const active = solidBodyMeshFramesForTime(parsedBodyMesh, null, 1, parsedWorld);

    expect(active).toHaveLength(1);
    expect(active[0].alignmentDebug).toMatchObject({ applied: false, reason: "missing_skeleton_root" });
    expect(active[0].renderTranslation).toEqual([0, 0, 0]);
    expect(active[0].frame.mesh_vertices_world).toEqual(parsedBodyMesh.players[0].frames[0].mesh_vertices_world);
  });

  it("leaves mesh-backed world samples computed while doubling the separate body-mesh display", () => {
    const parsedWorld = parseVirtualWorld({
      ...world,
      joint_names: ["left_hip", "right_hip"],
      players: [
        {
          ...world.players[0],
          frames: [
            {
              ...world.players[0].frames[0],
              t: 1,
              floor_world_xyz: [0, 0, 0],
              joints_world: [
                [0, 0, 1],
                [2, 0, 1],
              ],
              joint_conf: [0.8, 0.6],
            },
            {
              ...world.players[0].frames[0],
              t: 1 + 1 / 30,
              floor_world_xyz: [0, 2, 0],
              joints_world: [
                [0, 2, 1],
                [2, 2, 1],
              ],
              joint_conf: [0.6, 0.4],
            },
          ],
        },
      ],
    });
    const parsedBodyMesh = parseBodyMesh(twoComputedFrameBodyMesh());

    const off = displayFpsReplayData(parsedWorld, parsedBodyMesh, false);
    const on = displayFpsReplayData(parsedWorld, parsedBodyMesh, true);

    expect(off.world).toBe(parsedWorld);
    expect(off.bodyMesh).toBe(parsedBodyMesh);
    expect(on.world).toBe(parsedWorld);
    expect(on.world.fps).toBe(30);
    expect(on.world.players[0].frames.map((frame) => frame.t)).toHaveLength(2);
    expect(on.world.players[0].frames[0].t).toBeCloseTo(1);
    expect(on.world.players[0].frames[1].t).toBeCloseTo(1 + 1 / 30);
    expect(on.bodyMesh?.players[0].frames).toHaveLength(3);
    expect(on.bodyMesh?.players[0].frames[1].t).toBeCloseTo(1 + 1 / 60);
    expect(on.bodyMesh?.players[0].frames[1]).toMatchObject({
      mesh_interpolated: true,
      interpolation: { from_frame_idx: 30, to_frame_idx: 31, alpha: 0.5 },
    });
    expect(on.stats).toMatchObject({
      enabled: true,
      sourceFps: 30,
      displayFps: 60,
      worldComputedFrameCount: 2,
      worldInterpolatedFrameCount: 0,
      meshComputedFrameCount: 2,
      meshInterpolatedFrameCount: 1,
    });
  });

  it("adds only bounded runtime skeleton midpoints without changing measurement coverage", () => {
    const skeletonWorld = parseVirtualWorld({
      ...world,
      joint_names: ["left_hip", "right_hip"],
      players: [
        {
          ...world.players[0],
          representation: "joints",
          frames: [
            {
              ...world.players[0].frames[0],
              t: 1,
              joints_world: [
                [0, 0, 1],
                [2, 0, 1],
              ],
              joint_conf: [0.8, 0.6],
              mesh_vertices_world: [],
              joint_count: 2,
              mesh_vertex_count: 0,
            },
            {
              ...world.players[0].frames[0],
              t: 1.05,
              joints_world: [
                [0, 2, 1],
                [2, 2, 1],
              ],
              joint_conf: [0.6, 0.4],
              mesh_vertices_world: [],
              joint_count: 2,
              mesh_vertex_count: 0,
            },
          ],
        },
      ],
    });
    const originalSummary = skeletonWorld.summary;
    const originalCoverage = playerCoverageStats(skeletonWorld);

    const displayed = displayFpsReplayData(skeletonWorld, null, true);

    expect(skeletonWorld.players[0].frames).toHaveLength(2);
    expect(skeletonWorld.players[0].frames.every((frame) => frame.display_interpolated !== true)).toBe(true);
    expect(displayed.world.fps).toBe(skeletonWorld.fps);
    expect(displayed.world.summary).toBe(originalSummary);
    expect(displayed.world.players[0].frames).toHaveLength(3);
    const midpoint = displayed.world.players[0].frames[1];
    expect(midpoint).toMatchObject({
      t: 1.025,
      display_interpolated: true,
      display_interpolation: {
        kind: "skeleton_midpoint",
        player_id: 1,
        from_t: 1,
        to_t: 1.05,
        alpha: 0.5,
        max_gap_s: 0.05,
      },
      joints_world: [
        [0, 1, 1],
        [2, 1, 1],
      ],
      joint_conf: [0.7, 0.5],
      mesh_vertices_world: [],
      mesh_vertex_count: 0,
    });
    expect(midpoint.track_world_xy).toBeUndefined();
    expect(midpoint.floor_world_xyz).toBeUndefined();
    expect(midpoint.transl_world).toBeUndefined();
    expect(playerCoverageStats(displayed.world)).toEqual(originalCoverage);
    expect(displayed.stats).toMatchObject({
      worldComputedFrameCount: 2,
      worldInterpolatedFrameCount: 1,
    });

    const displayedAgain = displayFpsReplayData(displayed.world, null, true);
    expect(displayedAgain.world.players[0].frames).toHaveLength(3);
    expect(displayedAgain.stats.worldComputedFrameCount).toBe(2);
    expect(displayedAgain.stats.worldInterpolatedFrameCount).toBe(0);
  });

  it("preserves skeleton gaps across interruptions, wide spans, other players, and non-joint representations", () => {
    const computedFrame = (t: number, joints: number[][] = [[0, 0, 1], [1, 0, 1]]) => ({
      ...world.players[0].frames[0],
      t,
      joints_world: joints,
      joint_conf: joints.map(() => 0.8),
      mesh_vertices_world: [],
      joint_count: joints.length,
      mesh_vertex_count: 0,
    });
    const interrupted = parseVirtualWorld({
      ...world,
      players: [
        {
          ...world.players[0],
          representation: "joints",
          frames: [computedFrame(0), computedFrame(0.02, []), computedFrame(0.04)],
        },
      ],
    });
    const wide = parseVirtualWorld({
      ...world,
      players: [
        { ...world.players[0], representation: "joints", frames: [computedFrame(0), computedFrame(0.3)] },
      ],
    });
    const justOverLimit = parseVirtualWorld({
      ...world,
      players: [
        { ...world.players[0], representation: "joints", frames: [computedFrame(0), computedFrame(0.0501)] },
      ],
    });
    const separatePlayers = parseVirtualWorld({
      ...world,
      players: [
        { ...world.players[0], id: 1, representation: "joints", frames: [computedFrame(0)] },
        { ...world.players[0], id: 2, representation: "joints", frames: [computedFrame(0.03)] },
      ],
    });
    const nonJoint = parseVirtualWorld({
      ...world,
      players: [
        { ...world.players[0], representation: "mesh", frames: [computedFrame(0), computedFrame(0.03)] },
      ],
    });

    expect(displayFpsReplayData(interrupted, null, true).stats.worldInterpolatedFrameCount).toBe(0);
    expect(displayFpsReplayData(justOverLimit, null, true).stats.worldInterpolatedFrameCount).toBe(0);
    const displayedWide = displayFpsReplayData(wide, null, true);
    expect(displayedWide.stats.worldInterpolatedFrameCount).toBe(0);
    expect(frameForTime(wide.players[0], 0.15)).toBeUndefined();
    expect(frameForTime(displayedWide.world.players[0], 0.15)).toBeUndefined();
    expect(displayFpsReplayData(separatePlayers, null, true).stats.worldInterpolatedFrameCount).toBe(0);
    expect(displayFpsReplayData(nonJoint, null, true).stats.worldInterpolatedFrameCount).toBe(0);
  });

  it("drops persisted display interpolation claims and only marks viewer-generated frames", () => {
    const parsed = parseVirtualWorld({
      ...world,
      players: [
        {
          ...world.players[0],
          representation: "joints",
          frames: [
            { ...world.players[0].frames[0], display_interpolated: true, display_interpolation: { kind: "forged" } },
          ],
        },
      ],
    });

    expect(parsed.players[0].frames[0].display_interpolated).toBeUndefined();
    expect(parsed.players[0].frames[0].display_interpolation).toBeUndefined();
  });

  it("relaxes mesh midpoint interpolation to the user-enabled cadence ceiling and reports the real counts", () => {
    const parsedWorld = parseVirtualWorld(world);
    const parsedBodyMesh = parseBodyMesh(twoComputedFrameBodyMesh({ secondT: 1.1, secondFrameIdx: 33 }));

    const on = displayFpsReplayData(parsedWorld, parsedBodyMesh, true, { meshMaxGapSeconds: 0.15 });

    expect(on.bodyMesh?.players[0].frames.map((frame) => [Number(frame.t.toFixed(3)), frame.mesh_interpolated])).toEqual([
      [1, false],
      [1.05, true],
      [1.1, false],
    ]);
    expect(on.stats.meshInterpolatedFrameCount).toBe(1);
    expect(on.stats.meshMaxInterpolatedGapMs).toBe(100);
    expect(displayFpsReadout(on.stats)).toBe("60fps display: computed 30 + interpolated 0 skeletons, 1 mesh; mesh interpolated across 100ms gaps");
  });

  it("parses a chunked body-mesh index and selects the active mesh window by scrub time", () => {
    const index = parseBodyMeshIndex({
      schema_version: 1,
      artifact_type: "racketsport_body_mesh_index",
      clip: "clip_a",
      model: "sam3dbody_world_joints",
      fps: 30,
      world_frame: "court_Z0",
      faces_ref: "mhr_faces_static",
      faces_url: "body_mesh_faces.json",
      windows: [
        {
          source_window_index: 0,
          frame_start: 30,
          frame_end: 31,
          t0: 1.0,
          t1: 1.0333333333333334,
          frame_count: 2,
          target_player_ids: [2],
          player_ids: [2],
          url: "body_mesh_chunks/window_000.bin.gz",
          byte_size: 128,
          encoding: "gzip_int16_world_vertices_v1",
          quantization: { scale: 100, unit: "m" },
          players: [
            {
              player_id: 2,
              frames: [
                {
                  frame_idx: 30,
                  t: 1.0,
                  source_window_index: 0,
                  blend_weight: 0.5,
                  vertex_count: 3,
                  joint_count: 2,
                  joint_conf: [0.9, 0.8],
                  reasons: ["contact_window"],
                },
              ],
            },
          ],
        },
      ],
      summary: { window_count: 1, mesh_frame_count: 1, player_count: 1, faces_count: 1 },
    });

    expect(bodyMeshIndexWindowForTime(index, 1.01)?.source_window_index).toBe(0);
    expect(bodyMeshIndexWindowForTime(index, 0.8)).toBeNull();
  });

  it("decodes a quantized body-mesh chunk, joins through mesh_ref.player_id, and counts rendered players", () => {
    const faces = parseBodyMeshFaces({
      schema_version: 1,
      artifact_type: "racketsport_body_mesh_faces",
      faces_ref: "mhr_faces_static",
      mesh_faces: [[0, 1, 2]],
    });
    const index = parseBodyMeshIndex({
      schema_version: 1,
      artifact_type: "racketsport_body_mesh_index",
      clip: "clip_a",
      model: "sam3dbody_world_joints",
      fps: 30,
      world_frame: "court_Z0",
      faces_ref: "mhr_faces_static",
      faces_url: "body_mesh_faces.json",
      windows: [
        {
          source_window_index: 0,
          frame_start: 30,
          frame_end: 30,
          t0: 1.0,
          t1: 1.0333333333333334,
          frame_count: 1,
          target_player_ids: [2],
          player_ids: [2],
          url: "body_mesh_chunks/window_000.bin",
          byte_size: 30,
          encoding: "raw_int16_world_vertices_v1",
          quantization: { scale: 100, unit: "m" },
          players: [
            {
              player_id: 2,
              frames: [
                {
                  frame_idx: 30,
                  t: 1.0,
                  source_window_index: 0,
                  blend_weight: 0.5,
                  vertex_count: 3,
                  joint_count: 2,
                  joint_conf: [0.9, 0.8],
                  trust_badge: "preview",
                  reasons: ["contact_window"],
                },
              ],
            },
          ],
        },
      ],
      summary: { window_count: 1, mesh_frame_count: 1, player_count: 1, faces_count: 1 },
    });
    const window = index.windows[0];
    const values = new Int16Array([
      10, 100, 10,
      40, 100, 10,
      40, 140, 160,
      20, 120, 90,
      20, 120, 140,
    ]);
    const chunk = decodeBodyMeshChunkBytes(index, window, faces, values.buffer);
    const parsedWorld = parseVirtualWorld({
      ...world,
      players: [
        {
          ...world.players[0],
          id: 1,
          frames: [
            {
              ...world.players[0].frames[0],
              t: 1.0,
              mesh_ref: { artifact: "body_mesh.json", player_id: 2, frame_idx: 30, t: 1.0 },
            },
          ],
        },
      ],
    });

    expect(chunk.players[0].id).toBe(2);
    expect(chunk.players[0].frames[0].trust_badge).toBe("preview");
    expect(chunk.players[0].frames[0].mesh_vertices_world[2]).toEqual([0.4, 1.4, 1.6]);
    expect(chunk.players[0].frames[0].mesh_faces).toEqual([[0, 1, 2]]);

    const active = solidBodyMeshFramesForTime(chunk, null, 1.0, parsedWorld);
    expect(active).toMatchObject([{ playerId: 1, meshPlayerId: 2, frame: { frame_idx: 30 } }]);
    expect(solidMeshRenderedPlayerCount(active)).toBe(1);
  });

  it("reconstructs delta-encoded mesh and joint frames without changing quantized coordinates", () => {
    const faces = parseBodyMeshFaces({
      schema_version: 1,
      artifact_type: "racketsport_body_mesh_faces",
      faces_ref: "mhr_faces_static",
      mesh_faces: [[0, 0, 0]],
    });
    const index = parseBodyMeshIndex({
      schema_version: 1,
      artifact_type: "racketsport_body_mesh_index",
      clip: "clip_delta",
      model: "sam3dbody_world_joints",
      fps: 30,
      world_frame: "court_Z0",
      faces_ref: "mhr_faces_static",
      faces_url: "body_mesh_faces.json",
      windows: [
        {
          source_window_index: 0,
          frame_start: 0,
          frame_end: 1,
          t0: 0,
          t1: 1 / 15,
          frame_count: 2,
          player_frame_count: 2,
          target_player_ids: [1],
          player_ids: [1],
          url: "body_mesh_chunks/window_000.bin.gz",
          byte_size: 24,
          encoding: "gzip_int16_delta_world_vertices_v2",
          quantization: { scale: 100, unit: "m" },
          players: [
            {
              id: 1,
              frames: [
                { frame_idx: 0, t: 0, vertex_count: 1, joint_count: 1 },
                { frame_idx: 1, t: 1 / 30, vertex_count: 1, joint_count: 1, delta_from_previous: true },
              ],
            },
          ],
        },
      ],
      summary: { window_count: 1, mesh_frame_count: 2, player_count: 1, faces_count: 1 },
    });
    const bytes = new Int16Array([
      100, 200, 300,
      10, 20, 30,
      1, -2, 3,
      -1, 2, -3,
    ]).buffer;

    const decoded = decodeBodyMeshChunkBytes(index, index.windows[0], faces, bytes);

    expect(decoded.players[0].frames[0].mesh_vertices_world).toEqual([[1, 2, 3]]);
    expect(decoded.players[0].frames[1].mesh_vertices_world).toEqual([[1.01, 1.98, 3.03]]);
    expect(decoded.players[0].frames[1].joints_world).toEqual([[0.09, 0.22, 0.27]]);
  });

  it("rejects a delta frame without a previous-frame base", () => {
    const faces = parseBodyMeshFaces({
      schema_version: 1,
      artifact_type: "racketsport_body_mesh_faces",
      faces_ref: "mhr_faces_static",
      mesh_faces: [[0, 0, 0]],
    });
    const index = parseBodyMeshIndex({
      schema_version: 1,
      artifact_type: "racketsport_body_mesh_index",
      clip: "clip_bad_delta",
      model: "sam3dbody_world_joints",
      fps: 30,
      world_frame: "court_Z0",
      faces_ref: "mhr_faces_static",
      faces_url: "body_mesh_faces.json",
      windows: [{
        source_window_index: 0,
        frame_start: 0,
        frame_end: 0,
        t0: 0,
        t1: 1 / 30,
        frame_count: 1,
        player_frame_count: 1,
        target_player_ids: [1],
        player_ids: [1],
        url: "body_mesh_chunks/window_000.bin.gz",
        byte_size: 12,
        encoding: "gzip_int16_delta_world_vertices_v2",
        quantization: { scale: 100, unit: "m" },
        players: [{ id: 1, frames: [{
          frame_idx: 0,
          t: 0,
          vertex_count: 1,
          joint_count: 1,
          delta_from_previous: true,
        }] }],
      }],
      summary: { window_count: 1, mesh_frame_count: 1, player_count: 1, faces_count: 1 },
    });

    expect(() => decodeBodyMeshChunkBytes(index, index.windows[0], faces, new Int16Array(6).buffer))
      .toThrow("missing its previous-frame base");
  });

  it("proves midpoint interpolation and boundary refusal from a synthetic two-window index", () => {
    const faces = parseBodyMeshFaces({
      schema_version: 1,
      artifact_type: "racketsport_body_mesh_faces",
      faces_ref: "mhr_faces_static",
      mesh_faces: [[0, 1, 2]],
    });
    const index = parseBodyMeshIndex({
      schema_version: 1,
      artifact_type: "racketsport_body_mesh_index",
      clip: "clip_a",
      model: "sam3dbody_world_joints",
      fps: 30,
      world_frame: "court_Z0",
      faces_ref: "mhr_faces_static",
      faces_url: "body_mesh_faces.json",
      windows: [0, 1].map((windowIndex) => ({
        source_window_index: windowIndex,
        frame_start: 30 + windowIndex * 10,
        frame_end: 31 + windowIndex * 10,
        t0: 1 + windowIndex / 3,
        t1: 1 + windowIndex / 3 + 2 / 30,
        frame_count: 2,
        player_frame_count: 2,
        target_player_ids: [2],
        player_ids: [2],
        url: `body_mesh_chunks/window_00${windowIndex}.bin`,
        byte_size: 48,
        encoding: "raw_int16_world_vertices_v1",
        quantization: { scale: 10, unit: "m" },
        players: [
          {
            id: 2,
            frames: [0, 1].map((offset) => ({
              frame_idx: 30 + windowIndex * 10 + offset,
              t: 1 + windowIndex / 3 + offset / 30,
              source_window_index: windowIndex,
              blend_weight: 1,
              vertex_count: 3,
              joint_count: 1,
              joint_conf: [0.9],
              reasons: ["contact_dense_hitter_window"],
            })),
          },
        ],
      })),
      summary: { window_count: 2, mesh_frame_count: 4, player_count: 1, faces_count: 1 },
    });
    const window0 = decodeBodyMeshChunkBytes(
      index,
      index.windows[0],
      faces,
      new Int16Array([
        10, 10, 0,
        20, 10, 0,
        10, 20, 0,
        10, 10, 10,
        20, 20, 10,
        30, 20, 10,
        20, 30, 10,
        20, 20, 20,
      ]).buffer,
    );
    const doubledWindow0 = displayFpsReplayData(parseVirtualWorld(world), window0, true).bodyMesh!;
    const midpoint = solidBodyMeshFramesForTime(doubledWindow0, null, 1 + 1 / 60)[0].frame;
    expect(midpoint.mesh_interpolated).toBe(true);
    expect(midpoint.mesh_vertices_world[0]).toEqual([1.5, 1.5, 0.5]);

    const acrossWindows = {
      ...window0,
      players: [{ id: 2, frames: [window0.players[0].frames[1], { ...window0.players[0].frames[1], frame_idx: 40, t: 1 + 1 / 30, source_window_index: 1 }] }],
    };
    expect(solidBodyMeshFramesForTime(acrossWindows, null, 1 + 1 / 30 - 1 / 120)[0].frame.mesh_interpolated).toBe(false);
  });

  it("decodes the real staged window_000 when Vite has already decoded gzip content", async () => {
    const index = parseBodyMeshIndex(fixtureJson("body_mesh_index.json"));
    const faces = parseBodyMeshFaces(fixtureJson("body_mesh_faces.json"));
    const parsedWorld = parseVirtualWorld(fixtureJson("virtual_world_t0266_excerpt.json"));
    const activeWindow = bodyMeshIndexWindowForTime(index, 0.266);
    expect(activeWindow?.source_window_index).toBe(0);

    const compressedBytes = readFileSync(resolve(realMeshFixtureDir, "window_000.bin.gz"));
    const browserDecodedBytes = gunzipSync(compressedBytes);
    const fetchMock = vi.fn(
      async () => new Response(arrayBufferFrom(browserDecodedBytes), { status: 200, headers: { "content-encoding": "gzip" } }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const chunk = await fetchBodyMeshChunk("/fixtures/body_mesh_index.json", index, activeWindow!, faces);
    const active = solidBodyMeshFramesForTime(chunk, null, 0.266, parsedWorld);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(active.map((frame) => [frame.playerId, frame.meshPlayerId, Number(frame.frame.frame_idx.toFixed(2))])).toEqual([
      [1, 1, 7],
      [3, 3, 7],
    ]);
    expect(solidMeshRenderedPlayerCount(active)).toBe(2);
  });

  it("dedupes duplicate real chunk fetches for the same resolved mesh asset url", async () => {
    const index = parseBodyMeshIndex(fixtureJson("body_mesh_index.json"));
    const faces = parseBodyMeshFaces(fixtureJson("body_mesh_faces.json"));
    const activeWindow = bodyMeshIndexWindowForTime(index, 0.266);
    expect(activeWindow?.source_window_index).toBe(0);

    const compressedBytes = readFileSync(resolve(realMeshFixtureDir, "window_000.bin.gz"));
    const fetchMock = vi.fn(async () => new Response(arrayBufferFrom(compressedBytes), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const [first, second] = await Promise.all([
      fetchBodyMeshChunk("/fixtures/body_mesh_index.json", index, activeWindow!, faces),
      fetchBodyMeshChunk("/fixtures/body_mesh_index.json", index, activeWindow!, faces),
    ]);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(first.summary.mesh_frame_count).toBe(18);
    expect(second.summary.mesh_frame_count).toBe(18);
  });

  it("exposes the real staged window join state for browser diagnostics", () => {
    const index = parseBodyMeshIndex(fixtureJson("body_mesh_index.json"));
    const faces = parseBodyMeshFaces(fixtureJson("body_mesh_faces.json"));
    const parsedWorld = parseVirtualWorld(fixtureJson("virtual_world_t0266_excerpt.json"));
    const activeWindow = bodyMeshIndexWindowForTime(index, 0.266);
    const rawBytes = gunzipSync(readFileSync(resolve(realMeshFixtureDir, "window_000.bin.gz")));
    const chunk = decodeBodyMeshChunkBytes(index, activeWindow!, faces, arrayBufferFrom(rawBytes));
    const active = solidBodyMeshFramesForTime(chunk, null, 0.266, parsedWorld);

    const snapshot = bodyMeshDebugSnapshot({
      bodyMeshIndex: index,
      bodyMesh: chunk,
      world: parsedWorld,
      currentTime: 0.266,
      loadStatus: { state: "loaded", label: "mesh: loaded", windowId: 0 },
      activeBodyMeshes: active,
    });

    expect(snapshot).toMatchObject({
      active_window_id: 0,
      active_window_url: "body_mesh_chunks/window_000.bin.gz",
      load_state: "loaded",
      rendered_player_count: 2,
    });
    expect(snapshot.players.map((player) => ({
      world: player.world_player_id,
      normalized: player.normalized_mesh_player_id,
      present: player.mesh_player_present,
      frame: player.mesh_frame_idx === null ? null : Number(player.mesh_frame_idx.toFixed(2)),
    }))).toEqual([
      { world: 1, normalized: 1, present: true, frame: 7 },
      { world: 2, normalized: 2, present: false, frame: null },
      { world: 3, normalized: 3, present: true, frame: 7 },
      { world: 4, normalized: 4, present: false, frame: null },
    ]);
  });

  it("surfaces body mesh failure states in the Solid Mesh tile value", () => {
    expect(bodyMeshStatusTileValue(0, { state: "decode_failed", label: "mesh: decode_failed" })).toBe("mesh: decode_failed");
    expect(bodyMeshStatusTileValue(2, { state: "loaded", label: "mesh: loaded" })).toBe(2);
  });

  it("fails closed to no solid body mesh when body_mesh.json has vertices but no faces", () => {
    const noFaces = parseBodyMesh({
      ...bodyMesh,
      mesh_faces: [],
    });

    expect(solidBodyMeshFramesForTime(noFaces, null, 1.1)).toEqual([]);
  });

  it("reports the body_mesh.json summary frame count separately from the unrelated virtual_world point-cloud counter", () => {
    const parsedBodyMesh = parseBodyMesh(bodyMesh);

    expect(solidMeshFrameCount(parsedBodyMesh)).toBe(parsedBodyMesh.summary.mesh_frame_count);
    expect(solidMeshFrameCount(null)).toBe(0);
  });

  it("parses the real Wolverine wrist-only contact event without inventing absent audio", () => {
    const wristOnlyContactWindows = {
      schema_version: 1,
      events: [
        {
          confidence: 0.35,
          frame: 59,
          player_id: 20,
          sources: {
            ball_inflection: 0,
            wrist_vel: 0.71312,
          },
          t: 1.9666666666666666,
          trust_band_note: "wrist-cue-only, unverified",
          type: "contact",
          window: {
            importance: 0.35,
            t0: 1.8466666666666667,
            t1: 2.1466666666666665,
          },
        },
      ],
    };

    const parsed = parseContactWindows(wristOnlyContactWindows);

    expect(parsed.events[0].sources).toEqual({ ball_inflection: 0, wrist_vel: 0.71312 });
    expect(parsed.events[0].sources.audio).toBeUndefined();
  });

  it("rejects contact windows missing Python-required source scores", () => {
    const invalid = {
      schema_version: 1,
      events: [
        {
          type: "contact",
          t: 0.1,
          frame: 3,
          player_id: null,
          confidence: 1,
          window: { t0: 0, t1: 0.2, importance: 1 },
        },
      ],
    };

    expect(() => parseContactWindows(invalid)).toThrow("contact_windows.events[0].sources must be an object");
  });

  it("rejects contact windows whose sources object has no sensor or review signal", () => {
    const invalid = {
      schema_version: 1,
      events: [
        {
          type: "contact",
          t: 0.1,
          frame: 3,
          player_id: null,
          confidence: 1,
          sources: {},
          window: { t0: 0, t1: 0.2, importance: 1 },
        },
      ],
    };

    expect(() => parseContactWindows(invalid)).toThrow("contact_windows.events[0].sources must include at least one source score");
  });

  it("rejects contact windows with impossible Python-schema values", () => {
    expect(() =>
      parseContactWindows({
        schema_version: 1,
        events: [
          {
            type: "contact",
            t: -0.1,
            frame: -1,
            player_id: 1,
            confidence: 2,
            sources: { audio: 1, wrist_vel: 1, ball_inflection: 1 },
            window: { t0: 10, t1: 1, importance: -5 },
          },
        ],
      }),
    ).toThrow("contact_windows.events[0].window.t1 must be greater than or equal to contact_windows.events[0].window.t0");
  });

  it("rejects virtual-world ball frames missing Python-required xy/conf fields", () => {
    const invalid = {
      ...world,
      ball: { source: "tracknet", frames: [{ t: 0.1, world_xyz: [0.15, 1, 0], visible: true }] },
    };

    expect(() => parseVirtualWorld(invalid)).toThrow("virtual_world.ball.frames[0].xy must be an array");
  });

  it("tolerates confidence-gated world ball frame provenance without rejecting the artifact", () => {
    const realConfidenceGatedBallFrame = {
      approx: true,
      conf: 0.305206,
      confidence_provenance: {
        band: "physics_predicted",
        display_band: "hidden_no_prediction",
        horizon_frames: 2,
        predicted_sigma_m: 0.13530468171775023,
        predictor: "BallBallisticAdapter",
      },
      not_for_detection_metrics: true,
      render_only: true,
      t: 0.15,
      trust_band: {
        badge: "preview",
        evidence_path: "runs/phys_chain_20260702T174041Z/wolverine_v1_chain/ball_track_physics_filled.json",
        gate_id: "ball_track_physics_filled",
        gate_status: "interpolated",
        reason:
          "Ball sample consumed from ball_track_physics_filled.json as interpolated; virtual_world preserves the sample and applies no smoothing across bounce events.",
        stage: "PHYS-BALL",
      },
      visible: false,
      world_xyz: [-2.735352972773032, 2.8732401678170696, 0],
      xy: [0, 0],
    };
    const parsed = parseVirtualWorld({
      ...world,
      ball: { source: "physics_filled", frames: [realConfidenceGatedBallFrame] },
      summary: { ...world.summary, ball_frame_count: 1, approx_ball_frame_count: 1 },
    });

    expect(parsed.ball.frames[0].world_xyz).toEqual([-2.735352972773032, 2.8732401678170696, 0]);
    expect(parsed.ball.frames[0].visible).toBe(false);
  });

  it("selects video-space ball xy even when the current frame has no 3D world point", () => {
    const parsed = parseVirtualWorld({
      ...world,
      ball: {
        source: "wasb",
        frames: [
          {
            approx: false,
            conf: 0.90758002,
            t: 0.016666666666666666,
            visible: true,
            world_xyz: null,
            xy: [332.884094, 311.889221],
          },
          {
            approx: false,
            conf: 0.61,
            t: 0.03333333333333333,
            visible: true,
            world_xyz: null,
            xy: [312.817596, 317.564423],
            xy_interpolated: true,
          },
        ],
      },
      summary: { ...world.summary, ball_frame_count: 2 },
    });

    expect(videoBallOverlayForTime(parsed, 0.018)).toMatchObject({
      point: [332.884094, 311.889221],
      confidenceClass: "high",
      interpolated: false,
    });
    expect(videoBallOverlayForTime(parsed, 0.033)).toMatchObject({
      point: [312.817596, 317.564423],
      confidenceClass: "medium",
      interpolated: true,
    });
    expect(parsed.ball.frames[1].xy_interpolated).toBe(true);
  });

  it("rejects vectors, player representations, and paddle shapes outside the Python schema", () => {
    const extraVector = {
      ...world,
      ball: { source: "tracknet", frames: [{ t: 0.1, xy: [12, 18, 99], conf: 0.92, world_xyz: [0.15, 1, 0], visible: true }] },
    };
    const invalidRepresentation = {
      ...world,
      players: [{ id: 1, representation: "debug_proxy", frames: [] }],
    };
    const invalidPaddle = {
      ...world,
      paddles: [
        {
          player_id: 1,
          paddle_dims_in: { width: 7.5 },
          frames: [],
        },
      ],
    };

    expect(() => parseVirtualWorld(extraVector)).toThrow("virtual_world.ball.frames[0].xy must have length 2");
    expect(() => parseVirtualWorld(invalidRepresentation)).toThrow(
      "virtual_world.players[0].representation must be one of: track_only, joints, mesh",
    );
    expect(() => parseVirtualWorld(invalidPaddle)).toThrow("virtual_world.paddles[0].paddle_dims_in must include length/width or h/w");
  });

  it("does not hold the nearest player frame across a coverage gap", () => {
    const parsed = parseVirtualWorld(world);

    expect(frameForTime(parsed.players[0], 0.8)).toBeUndefined();
  });

  it("reports player artifact coverage so late-video gaps are visible", () => {
    const parsed = parseVirtualWorld(world);

    expect(playerCoverageStats(parsed)).toEqual({
      firstTime: 0,
      lastTime: 1,
      playerCount: 1,
      coveredFrameCount: 2,
    });
  });

  it("preserves declared source resolution when labels only occupy the upper-left half", () => {
    const parsed = parseLabelOverlayPayload(sampledLabelOverlay);

    expect(parsed.notGroundTruth).toBe(true);
    expect(parsed.secondsPerFrame).toBe(1);
    expect(labelOverlayForTime(parsed, 0.1).map((item) => item.id)).toEqual(["p1"]);
    expect(labelOverlayForTime(parsed, 2.4).map((item) => item.id)).toEqual(["p2"]);
    expect(labelViewBox(parsed)).toBe("0 0 1920 1080");
  });

  it("uses explicit label resolution metadata for half-resolution overlays", () => {
    const parsed = parseLabelOverlayPayload({
      ...sampledLabelOverlay,
      frames: {
        ...sampledLabelOverlay.frames,
        label_resolution: [960, 540],
      },
      annotation: {
        items: [{ frame: "frame_000001.jpg", id: "p1" }],
      },
    });

    expect(labelViewBox(parsed)).toBe("0 0 960 540");
  });

  it("classifies approximate off-court ball projections as hidden in 3D", () => {
    const parsed = parseVirtualWorld({
      ...world,
      ball: {
        source: "tracknet",
        frames: [
          { t: 0.1, xy: [12, 18], conf: 0.92, world_xyz: [100, 100, 0], visible: true, approx: true },
          { t: 0.2, xy: [12, 18], conf: 0.92, world_xyz: [0, 1, 0], visible: true, approx: true },
          { t: 0.3, xy: [12, 18], conf: 0.92, world_xyz: [0, 1, 0.6], visible: true, approx: false },
        ],
      },
    });

    expect(ballRenderInfoForTime(parsed, 0.1)).toMatchObject({ mode: "off_court_projection", render3d: false });
    expect(ballRenderInfoForTime(parsed, 0.2)).toMatchObject({ mode: "court_plane_projection", render3d: true });
    expect(ballRenderInfoForTime(parsed, 0.3)).toMatchObject({ mode: "calibrated_3d", render3d: true });
  });

  it("does not reuse stale player or ball frames far outside the artifact time range", () => {
    const parsed = parseVirtualWorld(world);

    expect(frameForTime(parsed.players[0], 999)).toBeUndefined();
    expect(ballFrameForTime(parsed, 999)).toBeUndefined();
  });

  it("reads a non-negative review start time from the query string", () => {
    expect(startTimeFromSearch("?manifest=/tmp/replay.json&t=0.58")).toBe(0.58);
    expect(startTimeFromSearch("?time=1.25")).toBe(1.25);
    expect(startTimeFromSearch("?t=-1")).toBe(0);
    expect(startTimeFromSearch("?t=bad")).toBe(0);
  });
});

const bodyTrustBand = {
  stage: "BODY",
  gate_id: "body_full_clip_gate+body_review_overlay_alignment",
  gate_status: "structural_pass_accuracy_unmeasured",
  badge: "preview",
  reason: "World-scale preview -- calibration upgrade pending.",
  evidence_path: "runs/x/body_gate_report.json",
};

const trackTrustBand = {
  stage: "TRK",
  gate_id: "trk_idf1_gate",
  gate_status: "do_not_promote",
  badge: "low_confidence",
  reason: "IDF1 below gate.",
  evidence_path: "runs/x/person_track_gt_score.json",
};

describe("trust-band provenance", () => {
  it("parses trust_band on court, players, ball, and paddles", () => {
    const worldWithTrustBands = {
      ...world,
      court: { ...world.court, trust_band: bodyTrustBand },
      players: [{ ...world.players[0], trust_band: bodyTrustBand }],
      ball: { source: "tracknet", frames: [], trust_band: trackTrustBand },
      paddles: [
        {
          player_id: 1,
          paddle_dims_in: { length: 15.5, width: 7.5 },
          frames: [],
          trust_band: trackTrustBand,
        },
      ],
    };

    const parsed = parseVirtualWorld(worldWithTrustBands);

    expect(parsed.court.trust_band).toEqual(bodyTrustBand);
    expect(parsed.players[0].trust_band).toEqual(bodyTrustBand);
    expect(parsed.ball.trust_band).toEqual(trackTrustBand);
    expect(parsed.paddles[0].trust_band).toEqual(trackTrustBand);
  });

  it("defaults trust_band to null when the artifact omits it", () => {
    const parsed = parseVirtualWorld(world);
    expect(parsed.court.trust_band).toBeNull();
    expect(parsed.players[0].trust_band).toBeNull();
    expect(parsed.ball.trust_band).toBeNull();
  });

  it("rejects an unknown badge value", () => {
    const invalid = { ...world, court: { ...world.court, trust_band: { ...bodyTrustBand, badge: "trusted" } } };
    expect(() => parseVirtualWorld(invalid)).toThrow("badge");
  });

  it("preserves the trust-contract too_close_to_call abstention badge", () => {
    const abstained = { ...bodyTrustBand, badge: "too_close_to_call" };
    const parsed = parseVirtualWorld({ ...world, court: { ...world.court, trust_band: abstained } });
    expect(parsed.court.trust_band?.badge).toBe("too_close_to_call");
    expect(trustBadgeColor("too_close_to_call")).not.toBe(trustBadgeColor("verified"));
  });

  it("colors verified/preview/low_confidence badges distinctly and fails closed on missing", () => {
    const verified = trustBadgeColor("verified");
    const preview = trustBadgeColor("preview");
    const lowConfidence = trustBadgeColor("low_confidence");
    const missingNull = trustBadgeColor(null);
    const missingUndefined = trustBadgeColor(undefined);
    expect(new Set([verified, preview, lowConfidence]).size).toBe(3);
    // A missing/null trust band must never render as "verified" -- it fails closed to
    // the low_confidence color so unverified entities are never mistaken for verified.
    expect(missingNull).toBe(lowConfidence);
    expect(missingUndefined).toBe(lowConfidence);
    expect(missingNull).not.toBe(verified);
  });

  it("effectiveTrustBadge fails closed to low_confidence for missing/null trust bands, never verified", () => {
    expect(effectiveTrustBadge(null)).toBe("low_confidence");
    expect(effectiveTrustBadge(undefined)).toBe("low_confidence");
    expect(effectiveTrustBadge(bodyTrustBand as never)).toBe(bodyTrustBand.badge);
    expect(effectiveTrustBadge({ ...bodyTrustBand, badge: "verified" } as never)).toBe("verified");
  });

  it("renders a short chip label from a trust band", () => {
    expect(trustBandChipText(bodyTrustBand as never)).toBe("BODY: preview");
    expect(trustBandChipText(null)).toBe("no trust band");
  });
});

describe("timeline markers and coaching-card metrics", () => {
  it("badges a human-reviewed contact event as preview and a low-confidence one as low_confidence", () => {
    const humanReviewed = {
      type: "contact" as const,
      t: 0.5,
      frame: 30,
      player_id: null,
      confidence: 1,
      sources: { audio: 0, wrist_vel: 0, ball_inflection: 0, human_review: 1 },
      window: { t0: 0.4, t1: 0.6, importance: 1 },
    };
    const detectorOnly = {
      ...humanReviewed,
      t: 1.5,
      confidence: 0.4,
      sources: { audio: 0.4, wrist_vel: 0, ball_inflection: 0, human_review: null },
    };
    expect(contactEventBadge(humanReviewed)).toBe("preview");
    expect(contactEventBadge(detectorOnly)).toBe("low_confidence");
  });

  it("always badges ball-inflection candidates low_confidence (BALL has 0/8 milestone gates passing)", () => {
    expect(ballInflectionBadge()).toBe("low_confidence");
  });

  it("parses a loosely-shaped ball_inflections.json into time-sorted candidates", () => {
    const parsed = parseBallInflections({
      artifact_type: "racketsport_ball_inflections",
      candidates: [
        { time_s: 1.5, frame: 90, confidence: 0.9, extra_field: "ignored" },
        { time_s: 0.5, frame: 30, confidence: 0.5 },
        { time_s: "not-a-number" },
      ],
    });
    expect(parsed.candidates).toEqual([
      { time_s: 1.5, frame: 90, confidence: 0.9 },
      { time_s: 0.5, frame: 30, confidence: 0.5 },
    ]);
  });

  it("returns an empty candidate list for a garbled ball_inflections payload", () => {
    expect(parseBallInflections(null)).toEqual({ candidates: [] });
    expect(parseBallInflections([1, 2, 3])).toEqual({ candidates: [] });
    expect(parseBallInflections({ candidates: "not-an-array" })).toEqual({ candidates: [] });
  });

  it("parses the real Wolverine event-subset boundaries that split arc trail segments", () => {
    const selectedEvents = parseBallArcEventsSelected(
      JSON.parse(
        readFileSync(
          resolve(
            process.cwd(),
            "../../runs/ball_arc_event_subset_20260703T02Z/wolverine_mixed_0200_mid_steep_corner/events_selected.json",
          ),
          "utf8",
        ),
      ),
    );

    expect(selectedEvents.artifact_type).toBe("racketsport_ball_arc_events_selected");
    expect(selectedEvents.selected.map((event) => event.t)).toContain(0.4);
    expect(selectedEvents.selected.map((event) => event.t)).toContain(7.233333333);
  });

  it("builds time-sorted markers from contact windows and ball inflections", () => {
    const markers = timelineMarkersFromArtifacts(
      {
        schema_version: 1,
        events: [
          {
            type: "contact",
            t: 2.0,
            frame: 120,
            player_id: 1,
            confidence: 1,
            sources: { audio: 0, wrist_vel: 0, ball_inflection: 0, human_review: 1 },
            window: { t0: 1.9, t1: 2.1, importance: 1 },
          },
        ],
      },
      { candidates: [{ time_s: 1.0, frame: 60, confidence: 0.5 }] },
    );

    expect(markers.map((marker) => marker.kind)).toEqual(["ball_inflection", "contact"]);
    expect(markers.map((marker) => marker.t)).toEqual([1.0, 2.0]);
    expect(markers[0].badge).toBe("low_confidence");
    expect(markers[1].badge).toBe("preview");
    expect(markers[0].provenance).toBe("model_estimated");
    expect(markers[1].provenance).toBe("measured");
  });

  it("adds directly seekable shot markers and keeps provenance separate from authority", () => {
    const markers = timelineMarkersFromArtifacts(
      {
        schema_version: 1,
        events: [{
          type: "contact",
          t: 1,
          frame: 30,
          player_id: 2,
          confidence: 1,
          sources: { human_review: 1 },
          window: { t0: 0.98, t1: 1.02, importance: 1 },
        }],
      },
      null,
      null,
      {
        schema_version: 1,
        artifact_type: "racketsport_shots",
        clip_id: "clip",
        policy: { internal_val_only: true, not_for_detection_metrics: true, not_ground_truth: true },
        shots: [{
          shot_id: "shot_1",
          event_anchor_id: "contact_1",
          segment_id: 1,
          player_id: 2,
          shot_type: "drive",
          shot_type_abstained: false,
          outcome: { call: "in", faults: [], let_candidate: false },
          confidence: 0.8,
          speed_mph: null,
          t: 1.01,
          frame: 30,
          peak_height_m: null,
          landing: null,
        }],
      },
    );

    expect(markers.map((marker) => [marker.kind, marker.t, marker.provenance])).toEqual([
      ["contact", 1, "measured"],
      ["shot", 1.01, "model_estimated"],
    ]);
    expect(markers[0].badge).toBe("preview");
  });

  it("parses reviewed bounce artifacts and promotes them into labeled timeline events", () => {
    const reviewedBounces = parseReviewedBounces({
      schema_version: 1,
      artifact_type: "racketsport_reviewed_ball_bounces",
      source: "human_review",
      bounces: [
        { review_id: "bounce_0003", frame: 267, t: 4.45445 },
        { review_id: "bounce_0005", frame: 500, t: 8.34167 },
      ],
    });
    const markers = timelineMarkersFromArtifacts(null, null, reviewedBounces);

    expect(markers.map((marker) => marker.kind)).toEqual(["reviewed_bounce", "reviewed_bounce"]);
    expect(markers[0].label).toBe("reviewed bounce bounce_0003 @ 4.45s");
    expect(markers[0].badge).toBe("preview");
    expect(markers[0].humanReviewed).toBe(true);
    expect(markers[0].provenance).toBe("measured");
  });

  it("parses real-shaped rally spans and makes them authoritative timeline chapters", () => {
    const rallySpans = parseRallySpans({
      artifact_type: "racketsport_rally_spans",
      clip_id: "wolverine_mixed_0200_mid_steep_corner",
      dead_time_fraction: 0,
      duration_s: 10,
      not_ground_truth: true,
      notes: [
        "Runtime optimization only, not an accuracy model.",
        "Signals fused with OR and biased toward over-inclusion; validate against reviewed contact/bounce timestamps before tightening thresholds.",
      ],
      pad_seconds: 0.5,
      schema_version: 1,
      signal_sources: {
        audio_onsets_path: null,
        ball_track_path:
          "/Users/arnavchokshi/Desktop/pickleball/runs/process_video_glue_20260702T_live_wolverine2/wolverine_mixed_0200_mid_steep_corner/ball_track.json",
        tracks_path:
          "/Users/arnavchokshi/Desktop/pickleball/runs/process_video_glue_20260702T_live_wolverine2/wolverine_mixed_0200_mid_steep_corner/tracks.json",
      },
      signals_used: ["ball_track", "player_motion"],
      span_count: 2,
      spans: [
        { rally_id: "rally_007", sources: ["ball"], t0: 3.2, t1: 4.8 },
        { rally_id: "rally_011", sources: ["player_motion"], t0: 7.0, t1: 8.5 },
      ],
    });

    expect(timelineChaptersFromRallySpans(rallySpans)).toEqual([
      { index: 1, rallyId: "rally_007", t0: 3.2, t1: 4.8, label: "Rally 007", badge: "preview" },
      { index: 2, rallyId: "rally_011", t0: 7.0, t1: 8.5, label: "Rally 011", badge: "preview" },
    ]);
  });

  it("derives rally chapter spans from event gaps and jumps to previous/next events", () => {
    const markers = timelineMarkersFromArtifacts(
      {
        schema_version: 1,
        events: [
          { type: "contact", t: 0.5, frame: 30, player_id: 1, confidence: 1, sources: { audio: 0, wrist_vel: 0, ball_inflection: 0, human_review: 1 }, window: { t0: 0.4, t1: 0.6, importance: 1 } },
          { type: "contact", t: 1.2, frame: 72, player_id: 2, confidence: 1, sources: { audio: 0, wrist_vel: 0, ball_inflection: 0, human_review: 1 }, window: { t0: 1.1, t1: 1.3, importance: 1 } },
          { type: "contact", t: 5.0, frame: 300, player_id: 1, confidence: 1, sources: { audio: 0, wrist_vel: 0, ball_inflection: 0, human_review: 1 }, window: { t0: 4.9, t1: 5.1, importance: 1 } },
        ],
      },
      null,
    );

    expect(timelineChaptersFromMarkers(markers, 8, 2)).toEqual([
      { index: 1, t0: 0.4, t1: 1.3, label: "Rally 1", badge: "preview" },
      { index: 2, t0: 4.9, t1: 5.1, label: "Rally 2", badge: "preview" },
    ]);
    expect(timelineEventJump(markers, 1.21, "next")).toBe(5.0);
    expect(timelineEventJump(markers, 1.21, "previous")).toBe(1.2);
    expect(timelineEventJump(markers, 0.5, "previous")).toBeNull();
  });

  it("returns no markers when both artifacts are absent", () => {
    expect(timelineMarkersFromArtifacts(null, null)).toEqual([]);
  });

  it("derives one rally span bounding every contact-window event", () => {
    const span = rallySpanFromContactWindows({
      schema_version: 1,
      events: [
        { type: "contact", t: 0.5, frame: 30, player_id: null, confidence: 1, sources: { audio: 0, wrist_vel: 0, ball_inflection: 0, human_review: 1 }, window: { t0: 0.4, t1: 0.6, importance: 1 } },
        { type: "contact", t: 3.2, frame: 190, player_id: null, confidence: 1, sources: { audio: 0, wrist_vel: 0, ball_inflection: 0, human_review: 1 }, window: { t0: 3.1, t1: 3.3, importance: 1 } },
      ],
    });
    expect(span).toEqual({ t0: 0.5, t1: 3.2 });
  });

  it("returns no rally span when there are no contact events", () => {
    expect(rallySpanFromContactWindows(null)).toBeNull();
    expect(rallySpanFromContactWindows({ schema_version: 1, events: [] })).toBeNull();
  });

  it("sums consecutive floor-position displacement within the rally span, ignoring frames outside it", () => {
    const player = {
      id: 1,
      representation: "track_only" as const,
      frames: [
        { t: 0.0, floor_world_xyz: [0, 0, 0] as [number, number, number], joints_world: [], joint_conf: [], mesh_vertices_world: [], joint_count: 0, mesh_vertex_count: 0 },
        { t: 1.0, floor_world_xyz: [3, 0, 0] as [number, number, number], joints_world: [], joint_conf: [], mesh_vertices_world: [], joint_count: 0, mesh_vertex_count: 0 },
        { t: 2.0, floor_world_xyz: [3, 4, 0] as [number, number, number], joints_world: [], joint_conf: [], mesh_vertices_world: [], joint_count: 0, mesh_vertex_count: 0 },
        { t: 9.0, floor_world_xyz: [100, 100, 0] as [number, number, number], joints_world: [], joint_conf: [], mesh_vertices_world: [], joint_count: 0, mesh_vertex_count: 0 },
      ],
    };
    expect(playerCoverageDistanceM(player, 0.0, 2.0)).toBeCloseTo(7.0);
  });

  it("falls back to track_world_xy when floor_world_xyz is absent", () => {
    const player = {
      id: 1,
      representation: "track_only" as const,
      frames: [
        { t: 0.0, track_world_xy: [0, 0] as [number, number], joints_world: [], joint_conf: [], mesh_vertices_world: [], joint_count: 0, mesh_vertex_count: 0 },
        { t: 1.0, track_world_xy: [0, 5] as [number, number], joints_world: [], joint_conf: [], mesh_vertices_world: [], joint_count: 0, mesh_vertex_count: 0 },
      ],
    };
    expect(playerCoverageDistanceM(player, 0.0, 1.0)).toBeCloseTo(5.0);
  });

  it("selects short ball and player trail windows behind the current frame without deleting out-of-court samples", () => {
    const parsed = parseVirtualWorld({
      ...world,
      ball: {
        source: "tracknet",
        frames: [
          { t: 0.0, xy: [0, 0], conf: 1, world_xyz: [0, 0, 0], visible: true, approx: true },
          { t: 0.5, xy: [1, 1], conf: 1, world_xyz: [1, 0, 0], visible: true, approx: true },
          {
            t: 0.9,
            xy: [2, 2],
            conf: 1,
            world_xyz: [9.5, 0, 0],
            visible: true,
            approx: true,
            confidence_provenance: {
              band: "physics_predicted",
              display_band: "physics_predicted_warn",
              horizon_frames: 2,
              predicted_sigma_m: 0.42,
              predictor: "BallBallisticAdapter",
            },
          },
        ],
      },
    });

    expect(ballTrailPointsForTime(parsed, 0.9, 0.55)).toEqual([
      { point: [1, 0, 0], courtStyle: "inside_court", uncertaintySigmaM: null, opacityScale: 1, thicknessScale: 1 },
      { point: [9.5, 0, 0], courtStyle: "outside_court", uncertaintySigmaM: 0.42, opacityScale: 0.35, thicknessScale: 0.55 },
    ]);
    expect(ballTrailSegmentsForTime(parsed, 0.9, 0.55)).toEqual([
      {
        from: [1, 0, 0],
        to: [9.5, 0, 0],
        courtStyle: "outside_court",
        opacityScale: 0.35,
        thicknessScale: 0.55,
      },
    ]);
    expect(playerTrailPointsForTime(parsed.players[0], 1.0, 0.6)).toEqual([
      [0.5, 1.5, 0],
    ]);
  });

  it("keeps visible measured and predicted gated ball trail samples while dropping hidden-no-prediction frames", () => {
    const parsed = parseVirtualWorld({
      ...world,
      ball: {
        source: "confidence_gated_world",
        frames: [
          {
            approx: false,
            conf: 0.89288396,
            confidence_provenance: {
              band: "hidden_no_prediction",
              display_band: "hidden_no_prediction",
              horizon_frames: 0,
              predicted_sigma_m: null,
              predictor: "none",
            },
            t: 0.15,
            visible: false,
            world_xyz: null,
            xy: [0, 0],
          },
          {
            approx: true,
            conf: 0.90876609,
            confidence_provenance: {
              band: "measured",
              display_band: "measured",
              horizon_frames: 0,
              predicted_sigma_m: null,
              predictor: "source_artifact",
            },
            t: 0.2,
            visible: true,
            world_xyz: [-2.600763233082503, 5.280123295160204, 0],
            xy: [151.942352, 391.208405],
          },
          {
            approx: true,
            conf: 0.62,
            confidence_provenance: {
              band: "physics_predicted",
              display_band: "physics_predicted_warn",
              horizon_frames: 2,
              predicted_sigma_m: 0.42,
              predictor: "BallBallisticAdapter",
            },
            render_only: true,
            not_for_detection_metrics: true,
            t: 0.23333333333333334,
            visible: true,
            world_xyz: [9.5, 5.6, 0],
            xy: [127.187706, 374.47641],
          },
        ],
      },
      summary: { ...world.summary, ball_frame_count: 3, approx_ball_frame_count: 2 },
    });

    expect(ballTrailPointsForTime(parsed, 0.24, 0.12)).toEqual([
      {
        point: [-2.600763233082503, 5.280123295160204, 0],
        courtStyle: "inside_court",
        uncertaintySigmaM: null,
        opacityScale: 1,
        thicknessScale: 1,
      },
      {
        point: [9.5, 5.6, 0],
        courtStyle: "outside_court",
        uncertaintySigmaM: 0.42,
        opacityScale: 0.35,
        thicknessScale: 0.55,
      },
    ]);
  });

  it("renders a kink-free ball trail from an arc-evaluated world stream and drops the honestly-hidden tail", () => {
    // Regression test for the owner's mid-air-direction-change report. This
    // shape is exactly what threed.racketsport.virtual_world.apply_ball_track_arc_solved_overlay
    // produces: a dense run of per-frame analytic (arc-evaluated) samples
    // with strictly monotonic horizontal progression, followed by frames the
    // arc solver could not bound between two confident events (world_xyz:
    // null) because no second confident endpoint existed. The viewer must
    // sample the covered run without ever reversing direction, and must
    // never bridge a straight line across the hidden tail.
    const covered = Array.from({ length: 6 }, (_, index) => ({
      t: index * 0.1,
      xy: [index, index] as [number, number],
      conf: 0.9,
      visible: true,
      world_xyz: [index * 0.5, index * 0.2, 0.9 - index * 0.05] as [number, number, number],
      approx: false,
    }));
    const hiddenTail = Array.from({ length: 3 }, (_, index) => ({
      t: 0.6 + (index + 1) * 0.1,
      xy: [0, 0] as [number, number],
      conf: 0.8,
      visible: true,
      world_xyz: null,
      approx: false,
    }));
    const parsed = parseVirtualWorld({
      ...world,
      ball: { source: "physics_filled", frames: [...covered, ...hiddenTail] },
    });

    const points = ballTrailPointsForTime(parsed, 0.9, 0.9);
    // Only the 6 arc-covered samples are drawable; the hidden tail contributes nothing.
    expect(points).toHaveLength(6);

    const segments = ballTrailSegmentsForTime(parsed, 0.9, 0.9);
    expect(segments.length).toBeGreaterThan(5);
    const xDeltas = segments.map((segment) => segment.to[0] - segment.from[0]);
    const yDeltas = segments.map((segment) => segment.to[1] - segment.from[1]);
    expect(xDeltas.every((delta) => delta > 0)).toBe(true);
    expect(yDeltas.every((delta) => delta > 0)).toBe(true);
  });

  it("does not draw ball-trail chords across selected arc events or hidden spans", () => {
    const parsed = parseVirtualWorld({
      ...world,
      ball: {
        source: "confidence_gated_world",
        frames: [
          { t: 0.0, xy: [0, 0], conf: 0.9, visible: true, world_xyz: [0, 0, 0.4], approx: false },
          { t: 0.1, xy: [1, 1], conf: 0.9, visible: true, world_xyz: [1, 0, 0.5], approx: false },
          { t: 0.2, xy: [2, 2], conf: 0.9, visible: true, world_xyz: [2, 0, 0.5], approx: false },
          { t: 0.3, xy: [3, 3], conf: 0.9, visible: true, world_xyz: [3, 0, 0.4], approx: false },
          {
            t: 0.4,
            xy: [4, 4],
            conf: 0.8,
            visible: true,
            world_xyz: null,
            approx: false,
            confidence_provenance: {
              band: "hidden_no_prediction",
              display_band: "hidden_no_prediction",
              horizon_frames: 0,
              predicted_sigma_m: null,
              predictor: "ball_arc_solver",
            },
          },
          { t: 0.5, xy: [5, 5], conf: 0.9, visible: true, world_xyz: [10, 0, 0.7], approx: false },
          { t: 0.6, xy: [6, 6], conf: 0.9, visible: true, world_xyz: [11, 0, 0.8], approx: false },
        ],
      },
    });
    const eventsSelected = {
      artifact_type: "racketsport_ball_arc_events_selected",
      selected: [{ t: 0.15, frame: 5, kind: "bounce", anchor_id: "bounce_0000" }],
    };

    const segments = ballTrailSegmentsForTime(parsed, 0.6, 0.7, eventsSelected);

    expect(segments).toEqual([
      expect.objectContaining({ from: [0, 0, 0.4], to: [1, 0, 0.5] }),
      expect.objectContaining({ from: [2, 0, 0.5], to: [3, 0, 0.4] }),
      expect.objectContaining({ from: [10, 0, 0.7], to: [11, 0, 0.8] }),
    ]);
    expect(segments).not.toContainEqual(expect.objectContaining({ from: [1, 0, 0.5], to: [2, 0, 0.5] }));
    expect(segments).not.toContainEqual(expect.objectContaining({ from: [3, 0, 0.4], to: [10, 0, 0.7] }));
  });

  it("reports a 2D-only floor ghost only when a visible video ball has an explicit court intersection", () => {
    const parsed = parseVirtualWorld({
      ...world,
      ball: {
        source: "confidence_gated_world",
        frames: [
          {
            t: 7.48,
            xy: [482, 391],
            conf: 0.86,
            visible: true,
            world_xyz: null,
            court_intersection_world_xyz: [0.35, 5.25, 0],
            approx: false,
            confidence_provenance: {
              band: "hidden_no_prediction",
              display_band: "hidden_no_prediction",
              horizon_frames: 0,
              predicted_sigma_m: null,
              predictor: "ball_arc_solver",
            },
          },
        ],
      },
    });

    expect(ballRenderInfoForTime(parsed, 7.48)).toMatchObject({
      mode: "2d_only_no_3d_solve",
      render3d: false,
      ghost: {
        position: [0.35, 5.25, 0],
        label: "2D-only, no 3D solve",
      },
    });
  });

  it("surfaces warning counts and optional entity coverage readouts without requiring new fields", () => {
    const parsed = parseVirtualWorld({
      ...world,
      summary: {
        ...world.summary,
        warnings: ["unprojected_visible_ball_frames", "missing_paddle_pose"],
      },
      ball: {
        ...world.ball,
        coverage_fraction: 0.42,
        min_t: 0.2,
        max_t: 4.4,
      },
      players: [
        {
          ...world.players[0],
          coverage_fraction: 1,
          min_t: 0,
          max_t: 10,
        },
      ],
    });

    expect(worldWarningsReadout(parsed)).toBe("2 notices: 2D-only ball frames outside solved arc coverage, missing paddle pose");
    expect(entityCoverageReadout("Ball", parsed.ball)).toBe("Ball 42.0% / 0.20-4.40s");
    expect(entityCoverageReadout("Player 1", parsed.players[0])).toBe("Player 1 100.0% / 0.00-10.00s");

    const absent = parseVirtualWorld(world);
    expect(worldWarningsReadout(absent)).toBe("0 notices");
    expect(entityCoverageReadout("Ball", absent.ball)).toBe("Ball coverage n/a");
  });

  it("builds the ball KPI from confidence-gated world bands instead of legacy coverage_fraction", () => {
    const parsed = parseVirtualWorld({
      ...world,
      ball: {
        ...world.ball,
        frames: [
          {
            t: 0,
            xy: [10, 20],
            world_xyz: [0, 0, 0.4],
            visible: true,
            conf: 0.91,
            approx: false,
            confidence_provenance: { band: "measured", display_band: "measured" },
          },
          {
            t: 0.033,
            xy: [11, 21],
            world_xyz: [0, 1, 0.5],
            visible: true,
            conf: 0.7,
            approx: true,
            confidence_provenance: { band: "physics_predicted", display_band: "physics_predicted" },
          },
          {
            t: 0.066,
            xy: [12, 22],
            world_xyz: null,
            visible: true,
            conf: 0.8,
            approx: false,
            confidence_provenance: { band: "hidden_no_prediction", display_band: "hidden_no_prediction" },
          },
        ],
      },
    });

    expect(ballCoverageKpiReadout(parsed)).toBe("1/3 measured · 1 predicted · 1 hidden");
    expect(ballCoverageKpiReadout(parseVirtualWorld(world))).toBe("coverage n/a");
  });
});
