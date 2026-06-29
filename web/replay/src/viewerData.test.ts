import { describe, expect, it } from "vitest";

import {
  activeBallContactPlayerIds,
  ballFrameForTime,
  ballRenderInfoForTime,
  contactEventCount,
  frameForTime,
  labelOverlayForTime,
  labelViewBox,
  parseContactWindows,
  parseLabelOverlayPayload,
  parsePhysicsRefinement,
  parseViewerManifest,
  parseVirtualWorld,
  playerCoverageStats,
  startTimeFromSearch,
  worldStats,
} from "./viewerData";

const manifest = {
  schema_version: 1,
  artifact_type: "racketsport_replay_viewer_manifest",
  clip: "clip_a",
  video_url: "/@fs/tmp/clip_a/input.mp4",
  virtual_world_url: "/@fs/tmp/clip_a/virtual_world.json",
  replay_scene_url: null,
  physics_refinement_url: "/@fs/tmp/clip_a/physics_refinement.json",
  contact_windows_url: "/@fs/tmp/clip_a/contact_windows.json",
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

describe("viewer data contracts", () => {
  it("parses the local replay viewer manifest with overlay trust flags", () => {
    expect(parseViewerManifest(manifest)).toEqual(manifest);
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

  it("selects the nearest player frame for the current video time", () => {
    const parsed = parseVirtualWorld(world);

    expect(frameForTime(parsed.players[0], 0.8)?.t).toBe(1);
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
