import { describe, expect, it } from "vitest";

import {
  DEFAULT_VIEW_STATE,
  applyViewPreset,
  entityFocusStyle,
  parseViewStateFromSearch,
  sceneLayerSnapshotForTime,
  toggleViewLayer,
  viewStateToSearch,
  type ViewLayerKey,
} from "./viewState";
import { parseBodyMesh, parseContactWindows, parseVirtualWorld } from "./viewerData";

const realShapedWorld = parseVirtualWorld({
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
          t: 0.7,
          track_world_xy: [0, 1],
          floor_world_xyz: [0, 1, 0],
          floor_source: "track_footpoint+smpl_world",
          joints_world: [
            [0, 1, 0.95],
            [0, 1, 1.45],
          ],
          joint_conf: [0.9, 0.9],
          mesh_vertices_world: [
            [-0.08, 0.96, 0.4],
            [0.08, 1.04, 0.4],
          ],
          joint_count: 2,
          mesh_vertex_count: 2,
        },
        {
          t: 1.0,
          track_world_xy: [0.3, 1.35],
          floor_world_xyz: [0.3, 1.35, 0],
          floor_source: "track_footpoint+smpl_world",
          joints_world: [
            [0.3, 1.35, 0.95],
            [0.3, 1.35, 1.45],
          ],
          joint_conf: [0.9, 0.9],
          mesh_vertices_world: [
            [0.22, 1.31, 0.4],
            [0.38, 1.39, 0.4],
          ],
          joint_count: 2,
          mesh_vertex_count: 2,
        },
      ],
    },
    {
      id: 2,
      side: "far",
      role: "right",
      representation: "joints",
      frames: [
        {
          t: 1.0,
          track_world_xy: [1.8, 2.4],
          floor_world_xyz: [1.8, 2.4, 0],
          floor_source: "track_footpoint",
          joints_world: [[1.8, 2.4, 1.2]],
          joint_conf: [0.8],
          mesh_vertices_world: [],
          joint_count: 1,
          mesh_vertex_count: 0,
        },
      ],
    },
  ],
  ball: {
    source: "physics_filled",
    frames: [
      { t: 0.6, xy: [300, 320], conf: 0.82, visible: true, world_xyz: [0.05, 1.05, 0.25] },
      { t: 0.8, xy: [310, 318], conf: 0.84, visible: true, world_xyz: [0.2, 1.2, 0.52] },
      { t: 1.0, xy: [322, 316], conf: 0.87, visible: true, world_xyz: [0.35, 1.45, 0.62] },
    ],
  },
  paddles: [],
  summary: {
    player_count: 2,
    mesh_player_count: 1,
    mesh_player_frame_count: 1,
    joint_player_frame_count: 3,
    track_only_player_frame_count: 0,
    floor_placed_player_frame_count: 3,
    floor_contact_player_frame_count: 0,
    max_floor_penetration_m: 0,
    max_abs_floor_offset_m: 0,
    physics_modes: [],
    ball_frame_count: 3,
    approx_ball_frame_count: 0,
    paddle_player_count: 0,
    paddle_frame_count: 0,
    ambiguous_paddle_frame_count: 0,
    warnings: [],
  },
});

const realShapedBodyMesh = parseBodyMesh({
  schema_version: 1,
  artifact_type: "racketsport_body_mesh",
  clip: "wolverine_mixed_0200_mid_steep_corner",
  model: "sam3dbody_world_joints",
  fps: 30,
  world_frame: "court_Z0",
  faces_ref: "mhr_faces_static",
  mesh_faces: [[0, 1, 2]],
  joint_names: ["sam3dbody_joint_000", "sam3dbody_joint_001"],
  players: [
    {
      id: 1,
      frames: [
        {
          frame_idx: 30,
          t: 1.0,
          source_window_index: 3,
          blend_weight: 0.73,
          joints_world: [
            [0.3, 1.35, 0.95],
            [0.3, 1.35, 1.45],
          ],
          joint_conf: [0.9, 0.9],
          mesh_vertices_world: [
            [0.15, 1.22, 0.1],
            [0.45, 1.22, 0.1],
            [0.3, 1.5, 1.7],
          ],
          smplx_params: { global_orient: [0, 0, 0], body_pose: [0], left_hand_pose: [0], right_hand_pose: [0], betas: [0] },
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
});

const realShapedContacts = parseContactWindows({
  schema_version: 1,
  events: [
    {
      type: "contact",
      t: 1.0,
      frame: 30,
      player_id: 1,
      confidence: 1,
      sources: { wrist_vel: 0.7, ball_inflection: 0.2 },
      window: { t0: 0.9, t1: 1.08, importance: 1 },
    },
  ],
});

describe("view-state URL persistence", () => {
  it("defaults to a clean scene and keeps debug point clouds off", () => {
    expect(DEFAULT_VIEW_STATE.layers).toMatchObject({
      ballDot: true,
      ballTrail: true,
      paddles: true,
      playerSkeletons: true,
      playerSolidMeshes: true,
      playerTrails: false,
      floorContactMarkers: false,
      eventMarkers: false,
      handJointPoints: false,
      implausibleSkeletons: false,
      debugPointClouds: false,
    });
    expect(DEFAULT_VIEW_STATE.focus).toBeNull();
  });

  it("round-trips layer state and focus through the URL query without dropping manifest", () => {
    const state = applyViewPreset(
      toggleViewLayer(toggleViewLayer(DEFAULT_VIEW_STATE, "playerTrails"), "eventMarkers"),
      "playerFocus",
      { playerId: 2 },
    );
    const search = viewStateToSearch("?manifest=/@fs/run/replay_viewer_manifest.json&t=1.2", state);
    const parsed = parseViewStateFromSearch(search);

    expect(search).toContain("manifest=%2F%40fs%2Frun%2Freplay_viewer_manifest.json");
    expect(parsed.layers.playerTrails).toBe(true);
    expect(parsed.layers.eventMarkers).toBe(true);
    expect(parsed.focus).toEqual({ kind: "player", playerId: 2 });
    expect(parsed.selectedPlayerId).toBe(2);
  });

  it("treats an explicit layers query as the shareable source of truth", () => {
    const parsed = parseViewStateFromSearch("?layers=ball,paddles,players,mesh,events,hands,implausible,debug");

    expect(parsed.layers.ballDot).toBe(true);
    expect(parsed.layers.paddles).toBe(true);
    expect(parsed.layers.playerSkeletons).toBe(true);
    expect(parsed.layers.playerSolidMeshes).toBe(true);
    expect(parsed.layers.eventMarkers).toBe(true);
    expect(parsed.layers.handJointPoints).toBe(true);
    expect(parsed.layers.implausibleSkeletons).toBe(true);
    expect(parsed.layers.ballTrail).toBe(false);
    expect(parsed.layers.debugPointClouds).toBe(true);
  });
});

describe("view-state focus presets", () => {
  it("uses ball focus as a layer bundle and highlights the ball while dimming players", () => {
    const state = applyViewPreset(DEFAULT_VIEW_STATE, "ballFocus");

    expect(state.layers.ballDot).toBe(true);
    expect(state.layers.ballTrail).toBe(true);
    expect(state.focus).toEqual({ kind: "ball" });
    expect(entityFocusStyle(state, { kind: "ball" })).toEqual({ dimmed: false, highlighted: true });
    expect(entityFocusStyle(state, { kind: "player", playerId: 1 })).toEqual({ dimmed: true, highlighted: false });
  });

  it("uses player focus as a layer bundle and highlights only the selected player", () => {
    const state = applyViewPreset(DEFAULT_VIEW_STATE, "playerFocus", { playerId: 1 });

    expect(state.layers.playerSkeletons).toBe(true);
    expect(state.layers.playerSolidMeshes).toBe(true);
    expect(state.focus).toEqual({ kind: "player", playerId: 1 });
    expect(entityFocusStyle(state, { kind: "player", playerId: 1 })).toEqual({ dimmed: false, highlighted: true });
    expect(entityFocusStyle(state, { kind: "player", playerId: 2 })).toEqual({ dimmed: true, highlighted: false });
    expect(entityFocusStyle(state, { kind: "ball" })).toEqual({ dimmed: true, highlighted: false });
  });
});

describe("scene-layer consequences", () => {
  function snapshotWithDisabled(layer: ViewLayerKey) {
    return sceneLayerSnapshotForTime({
      world: realShapedWorld,
      bodyMesh: realShapedBodyMesh,
      contactWindows: realShapedContacts,
      currentTime: 1.0,
      viewState: toggleViewLayer(DEFAULT_VIEW_STATE, layer),
    });
  }

  it("reports visible default scene layers from real-shaped world/body/contact fixtures", () => {
    const snapshot = sceneLayerSnapshotForTime({
      world: realShapedWorld,
      bodyMesh: realShapedBodyMesh,
      contactWindows: realShapedContacts,
      currentTime: 1.0,
      viewState: DEFAULT_VIEW_STATE,
    });

    expect(snapshot.courtNet.visible).toBe(true);
    expect(snapshot.ballDot).toMatchObject({ visible: true, objectCount: 1 });
    expect(snapshot.ballTrail.visible).toBe(true);
    expect(snapshot.ballTrail.objectCount).toBeGreaterThan(0);
    expect(snapshot.playerSkeletons).toMatchObject({ visible: true, objectCount: 2 });
    expect(snapshot.playerSolidMeshes).toMatchObject({ visible: true, objectCount: 1 });
    expect(snapshot.playerTrails).toMatchObject({ visible: false, objectCount: 0 });
    expect(snapshot.floorContactMarkers).toMatchObject({ visible: false, objectCount: 0 });
    expect(snapshot.eventMarkers).toMatchObject({ visible: false, objectCount: 0 });
    expect(snapshot.debugPointClouds).toMatchObject({ visible: false, objectCount: 0 });
  });

  it("gates every user-facing layer toggle to a scene-layer count", () => {
    expect(snapshotWithDisabled("ballDot").ballDot).toMatchObject({ visible: false, objectCount: 0 });
    expect(snapshotWithDisabled("ballTrail").ballTrail).toMatchObject({ visible: false, objectCount: 0 });
    expect(snapshotWithDisabled("playerSkeletons").playerSkeletons).toMatchObject({ visible: false, objectCount: 0 });
    expect(snapshotWithDisabled("playerSolidMeshes").playerSolidMeshes).toMatchObject({ visible: false, objectCount: 0 });

    const playerTrailsOn = sceneLayerSnapshotForTime({
      world: realShapedWorld,
      bodyMesh: realShapedBodyMesh,
      contactWindows: realShapedContacts,
      currentTime: 1.0,
      viewState: toggleViewLayer(DEFAULT_VIEW_STATE, "playerTrails"),
    });
    expect(playerTrailsOn.playerTrails).toMatchObject({ visible: true, objectCount: 1 });

    const floorOn = sceneLayerSnapshotForTime({
      world: realShapedWorld,
      bodyMesh: realShapedBodyMesh,
      contactWindows: realShapedContacts,
      currentTime: 1.0,
      viewState: toggleViewLayer(DEFAULT_VIEW_STATE, "floorContactMarkers"),
    });
    expect(floorOn.floorContactMarkers).toMatchObject({ visible: true, objectCount: 2 });

    const eventsOn = sceneLayerSnapshotForTime({
      world: realShapedWorld,
      bodyMesh: realShapedBodyMesh,
      contactWindows: realShapedContacts,
      currentTime: 1.0,
      viewState: toggleViewLayer(DEFAULT_VIEW_STATE, "eventMarkers"),
    });
    expect(eventsOn.eventMarkers).toMatchObject({ visible: true, objectCount: 1 });

    const debugOn = sceneLayerSnapshotForTime({
      world: realShapedWorld,
      bodyMesh: realShapedBodyMesh,
      contactWindows: realShapedContacts,
      currentTime: 1.0,
      viewState: toggleViewLayer(DEFAULT_VIEW_STATE, "debugPointClouds"),
    });
    expect(debugOn.debugPointClouds).toMatchObject({ visible: true, objectCount: 1 });
  });
});
