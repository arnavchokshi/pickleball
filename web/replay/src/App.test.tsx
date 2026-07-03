import { describe, expect, it } from "vitest";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import * as AppModule from "./App";
import {
  bodyJointSkeletonForFrame,
  bodyMeshOpacityFromBlendWeight,
  cameraPresetPose,
  cocoWholeBodyCoreBoneNames,
  createSolidBodyMeshGeometryCache,
  contactReadoutText,
  courtBounds,
  geometryForSolidBodyMeshFrame,
  handJointPointsForFrame,
  manifestUrlFromSearch,
  paddleRenderGeometryForFrame,
  updateFpsSample,
  vertexDebugPointsForFrame,
} from "./App";
import { parseBodyMesh } from "./viewerData";
import type { BodyMesh, TimelineChapter, VirtualWorld, VirtualWorldPlayer } from "./viewerData";
import { DEFAULT_VIEW_STATE, applyViewPreset } from "./viewState";

describe("manifestUrlFromSearch", () => {
  it("uses the manifest query parameter when present", () => {
    expect(manifestUrlFromSearch("?manifest=/@fs/tmp/replay_viewer_manifest.json")).toBe(
      "/@fs/tmp/replay_viewer_manifest.json",
    );
  });

  it("does not fall back to a checkout-specific absolute path", () => {
    expect(manifestUrlFromSearch("")).toBeNull();
  });
});

describe("coachingFactsUrlFromSearch", () => {
  it("uses the explicit coaching query parameter before the manifest pointer", () => {
    const coachingFactsUrlFromSearch = (AppModule as any).coachingFactsUrlFromSearch;
    expect(typeof coachingFactsUrlFromSearch).toBe("function");

    expect(
      coachingFactsUrlFromSearch("?coaching=/@fs/tmp/override/coaching_card_facts.json", {
        coaching_card_facts_url: "/@fs/tmp/manifest/coaching_card_facts.json",
      }),
    ).toBe("/@fs/tmp/override/coaching_card_facts.json");
    expect(
      coachingFactsUrlFromSearch("", {
        coaching_card_facts_url: "/@fs/tmp/manifest/coaching_card_facts.json",
      }),
    ).toBe("/@fs/tmp/manifest/coaching_card_facts.json");
    expect(coachingFactsUrlFromSearch("", null)).toBeNull();
  });
});

describe("bodyMeshOpacityFromBlendWeight", () => {
  it("keeps solid mesh endpoints visible while still scaling by the body_mesh blend weight", () => {
    expect(bodyMeshOpacityFromBlendWeight({ blend_weight: 0 })).toBeGreaterThan(0.2);
    expect(bodyMeshOpacityFromBlendWeight({ blend_weight: 0.5 })).toBeCloseTo(0.47);
    expect(bodyMeshOpacityFromBlendWeight({ blend_weight: 1 })).toBeCloseTo(0.68);
  });

  it("multiplies mesh presence fade at window edges instead of popping between visible and hidden", () => {
    expect(bodyMeshOpacityFromBlendWeight({ blend_weight: 1 }, 0)).toBe(0);
    expect(bodyMeshOpacityFromBlendWeight({ blend_weight: 1 }, 0.5)).toBeCloseTo(0.34);
    expect(bodyMeshOpacityFromBlendWeight({ blend_weight: 1 }, 1)).toBeCloseTo(0.68);
  });
});

describe("ViewLayerPanel", () => {
  it("renders all layer buttons with explicit pressed state", () => {
    const ViewLayerPanel = (AppModule as any).ViewLayerPanel;
    expect(typeof ViewLayerPanel).toBe("function");

    const html = renderToStaticMarkup(
      React.createElement(ViewLayerPanel, {
        viewState: DEFAULT_VIEW_STATE,
        playerIds: [1, 2],
        onToggleLayer: () => undefined,
        onBallFocus: () => undefined,
        onPlayerFocus: () => undefined,
        onClearFocus: () => undefined,
        onResetView: () => undefined,
      }),
    );

    expect(html).toContain("Layer controls");
    expect(html).toContain("aria-pressed=\"true\"");
    expect(html).toContain("Ball trail");
    expect(html).toContain("Paddles");
    expect(html).toContain("Skeletons");
    expect(html).toContain("Solid meshes");
    expect(html).toContain("Hand points");
    expect(html).toContain("Implausible skeletons");
    expect(html).toContain("aria-pressed=\"false\"");
    expect(html).toContain("Point clouds");
  });

  it("marks focus presets and player chips as active when selected", () => {
    const ViewLayerPanel = (AppModule as any).ViewLayerPanel;
    const focused = applyViewPreset(DEFAULT_VIEW_STATE, "playerFocus", { playerId: 2 });

    const html = renderToStaticMarkup(
      React.createElement(ViewLayerPanel, {
        viewState: focused,
        playerIds: [1, 2],
        onToggleLayer: () => undefined,
        onBallFocus: () => undefined,
        onPlayerFocus: () => undefined,
        onClearFocus: () => undefined,
        onResetView: () => undefined,
      }),
    );

    expect(html).toContain("Player focus");
    expect(html).toContain("Player 2");
    expect(html).toContain("player-chip active");
    expect(html).toContain("aria-pressed=\"true\"");
  });
});

describe("MeshDebugReadout", () => {
  it("renders an inspectable mesh-chain snapshot behind the debug layer", () => {
    const MeshDebugReadout = (AppModule as any).MeshDebugReadout;
    expect(typeof MeshDebugReadout).toBe("function");

    const html = renderToStaticMarkup(
      React.createElement(MeshDebugReadout, {
        snapshot: {
          current_time: 0.266,
          active_window_id: 0,
          active_window_url: "body_mesh_chunks/window_000.bin.gz",
          load_state: "loaded",
          load_stage: "chunk",
          load_url: "body_mesh_chunks/window_000.bin.gz",
          load_message: null,
          rendered_player_count: 2,
          players: [
            {
              world_player_id: 1,
              world_frame_t: 0.26666666666666666,
              mesh_ref_player_id: 1,
              mesh_ref_frame_idx: 8,
              normalized_mesh_player_id: 1,
              mesh_player_present: true,
              mesh_frame_present: true,
              mesh_frame_idx: 8,
            },
          ],
        },
      }),
    );

    expect(html).toContain("data-mesh-debug=");
    expect(html).toContain("window=0");
    expect(html).toContain("load=loaded");
    expect(html).toContain("P1-&gt;M1:frame 8");
  });
});

describe("solid mesh geometry cache", () => {
  const meshPayload = {
    schema_version: 1,
    artifact_type: "racketsport_body_mesh",
    clip: "clip_a",
    model: "sam3dbody_world_joints",
    fps: 30,
    world_frame: "court_Z0",
    faces_ref: "mhr_faces_static",
    mesh_faces: [[0, 1, 2]],
    joint_names: [],
    players: [
      {
        id: 4,
        frames: [
          {
            frame_idx: 42,
            t: 1.4,
            source_window_index: 0,
            blend_weight: 1,
            joints_world: [],
            joint_conf: [],
            mesh_vertices_world: [
              [0, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
            ],
            smplx_params: {},
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

  it("precomputes indexed geometry once per player frame and reuses it across render swaps", () => {
    const bodyMesh = parseBodyMesh(meshPayload) as BodyMesh;
    const cache = createSolidBodyMeshGeometryCache(bodyMesh);
    const frame = bodyMesh.players[0].frames[0];
    const first = geometryForSolidBodyMeshFrame(cache, 4, frame);
    const second = geometryForSolidBodyMeshFrame(cache, 4, frame);

    expect(first).toBe(second);
    expect(cache.geometryCount).toBe(1);
    expect(cache.normalComputeCount).toBe(1);
  });

  it("builds geometry for a truncated real Wolverine body_mesh frame and keeps zero-blend opacity visible", () => {
    const realShapedPayload = {
      schema_version: 1,
      artifact_type: "racketsport_body_mesh",
      clip: "wolverine_mixed_0200_mid_steep_corner",
      model: "sam3dbody_world_joints",
      fps: 30,
      world_frame: "court_Z0",
      faces_ref: "mhr_faces_static",
      // Remapped from body_mesh.json player 2 frame 41, source face [3997, 3895, 5229].
      mesh_faces: [[0, 1, 2]],
      joint_names: ["sam3dbody_joint_000", "sam3dbody_joint_001"],
      players: [
        {
          id: 2,
          frames: [
            {
              frame_idx: 41,
              t: 1.3666666666666667,
              source_window_index: 3,
              blend_weight: 0,
              joints_world: [
                [-0.4609871280688127, -2.86639920139313, 1.4727504227208539],
                [-0.44476821007905043, -2.89331748700701, 1.519577850798574],
              ],
              joint_conf: [0.9318317174911499, 0.9318317174911499],
              mesh_vertices_world: [
                [-0.42600738197685617, -2.859783359847331, 1.4446742849885692],
                [-0.42402143790016034, -2.8583956249317484, 1.444022971085586],
                [-0.4242655178571626, -2.858508900821761, 1.4441960016626272],
              ],
              smplx_params: {
                global_orient: [2.114689826965332, -1.3829503059387207, -2.1891682147979736],
                body_pose: [0.021708069369196892, 0.07678874582052231, 0.04448043555021286],
                left_hand_pose: [0.017201418057084084],
                right_hand_pose: [0.058121927082538605],
                betas: [-1.1666346788406372],
                transl_world: [-0.32252942875560225, -3.0421257901883143, 0],
              },
              reasons: ["contact_window", "low_track_confidence"],
            },
          ],
        },
      ],
      summary: {
        mesh_frame_count: 1,
        player_count: 1,
        contact_window_count: 8,
      },
    };
    const bodyMesh = parseBodyMesh(realShapedPayload) as BodyMesh;
    const cache = createSolidBodyMeshGeometryCache(bodyMesh);
    const frame = bodyMesh.players[0].frames[0];

    expect(cache.geometryCount).toBe(1);
    expect(geometryForSolidBodyMeshFrame(cache, 2, frame).attributes.position.count).toBe(3);
    expect(bodyMeshOpacityFromBlendWeight(frame)).toBeGreaterThan(0.2);
  });
});

describe("vertexDebugPointsForFrame", () => {
  it("defaults vertex debug clouds off and only samples the supplied current frame when enabled", () => {
    const frame = {
      t: 1,
      joints_world: [],
      joint_conf: [],
      mesh_vertices_world: [
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
      ],
      joint_count: 0,
      mesh_vertex_count: 3,
    } as VirtualWorldPlayer["frames"][number];
    const previousFrame = {
      ...frame,
      t: 0,
      mesh_vertices_world: [[99, 99, 99]],
      mesh_vertex_count: 1,
    } as VirtualWorldPlayer["frames"][number];

    expect(vertexDebugPointsForFrame(frame, false, 10)).toEqual([]);
    expect(vertexDebugPointsForFrame(frame, true, 2)).toEqual([
      [0, 0, 0],
      [2, 0, 0],
    ]);
    expect(vertexDebugPointsForFrame(frame, true, 10)).not.toContainEqual(previousFrame.mesh_vertices_world[0]);
  });
});

describe("bodyJointSkeletonForFrame", () => {
  it("uses the real staged COCO-WholeBody 133-joint names for body bones only", () => {
    const stagedSkeleton = JSON.parse(
      readFileSync(
        resolve(process.cwd(), "../../runs/manager_rebuild_wolverine_20260702T23Z/skeleton3d.json"),
        "utf8",
      ),
    );
    const jointNames = stagedSkeleton.joint_names as string[];
    const stagedFrame = stagedSkeleton.players[0].frames[0] as VirtualWorldPlayer["frames"][number];

    const skeleton = bodyJointSkeletonForFrame(stagedFrame, jointNames);

    expect(skeleton).not.toBeNull();
    if (!skeleton) throw new Error("expected staged COCO-WholeBody skeleton");
    expect(skeleton.boneNames).toEqual(cocoWholeBodyCoreBoneNames);
    expect(skeleton.bones).toHaveLength(cocoWholeBodyCoreBoneNames.length * 2);
    expect(skeleton.boneNames.some(([a, b]) => `${a} ${b}`.includes("face-") || `${a} ${b}`.includes("hand_"))).toBe(false);
    expect(skeleton.boneNames).not.toContainEqual(["left_hip", "right_shoulder"]);
    expect(skeleton.boneNames).not.toContainEqual(["right_hip", "left_shoulder"]);
  });

  it("hides implausible skeleton frames unless debug explicitly includes them", () => {
    const frame = {
      t: 1,
      skeleton_implausible: true,
      joints_world: [
        [0, 0, 0.9],
        [0, 0, 1.45],
      ],
      joint_conf: [0.9, 0.9],
      mesh_vertices_world: [],
      joint_count: 2,
      mesh_vertex_count: 0,
    } as VirtualWorldPlayer["frames"][number];

    expect(bodyJointSkeletonForFrame(frame, ["left_hip", "left_shoulder"])).toBeNull();
    expect(bodyJointSkeletonForFrame(frame, ["left_hip", "left_shoulder"], { includeImplausible: true })).not.toBeNull();
  });

  it("turns BODY joints_world into canonical bone line pairs instead of dots only", () => {
    const frame = {
      t: 1,
      joints_world: [
        [0, 0, 1.7],
        [-0.22, 0, 1.38],
        [0.22, 0, 1.38],
        [-0.48, 0, 1.12],
        [0.48, 0, 1.12],
      ],
      joint_conf: [0.9, 0.9, 0.9, 0.9, 0.9],
      mesh_vertices_world: [],
      joint_count: 5,
      mesh_vertex_count: 0,
    } as VirtualWorldPlayer["frames"][number];

    const skeleton = bodyJointSkeletonForFrame(frame, ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow"]);

    expect(skeleton).not.toBeNull();
    if (!skeleton) throw new Error("expected BODY joint skeleton");
    expect(skeleton.joints).toHaveLength(5);
    expect(skeleton.bones.length).toBeGreaterThan(0);
    expect(skeleton.bones.length % 2).toBe(0);
  });

  it("keeps whole-body hand joints as optional dots and never body bones", () => {
    const jointNames = [
      "nose",
      "left_shoulder",
      "right_shoulder",
      "left_elbow",
      "right_elbow",
      "left_wrist",
      "right_wrist",
      "left_hand_root",
      "left_thumb1",
      "right_hand_root",
    ];
    const frame = {
      t: 1,
      joints_world: jointNames.map((_, index) => [index * 0.01, 0, 1] as [number, number, number]),
      joint_conf: jointNames.map(() => 0.9),
      mesh_vertices_world: [],
      joint_count: jointNames.length,
      mesh_vertex_count: 0,
    } as VirtualWorldPlayer["frames"][number];

    const skeleton = bodyJointSkeletonForFrame(frame, jointNames);
    const handPoints = handJointPointsForFrame(frame, jointNames);

    expect(skeleton?.boneNames.some(([a, b]) => `${a} ${b}`.includes("hand") || `${a} ${b}`.includes("thumb"))).toBe(false);
    expect(handPoints).toHaveLength(3);
  });
});

describe("paddleRenderGeometryForFrame", () => {
  it("builds an estimated rounded face-plane paddle from a real wrist-proxy frame pose", () => {
    const stagedWorld = JSON.parse(
      readFileSync(
        resolve(process.cwd(), "../../runs/manager_rebuild_wolverine_20260702T23Z/virtual_world.json"),
        "utf8",
      ),
    );
    const paddle = stagedWorld.paddles[0];
    const frame = paddle.frames[0];

    const geometry = paddleRenderGeometryForFrame(frame, paddle.paddle_dims_in);

    expect(frame.source).toBe("wrist_proxy:court_Z0");
    expect(geometry.estimated).toBe(true);
    expect(geometry.vertices.length).toBeGreaterThan(4);
    expect(geometry.faces.length).toBeGreaterThan(2);
    expect(geometry.vertices[1]).not.toEqual(frame.pose_se3.t);
  });
});

function makeWorld(players: VirtualWorldPlayer[]): VirtualWorld {
  return {
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
      trust_band: null,
    },
    players,
    ball: { source: null, frames: [] },
    paddles: [],
    summary: {
      player_count: players.length,
      mesh_player_count: 0,
      mesh_player_frame_count: 0,
      joint_player_frame_count: 0,
      track_only_player_frame_count: 0,
      floor_placed_player_frame_count: 0,
      floor_contact_player_frame_count: 0,
      max_floor_penetration_m: 0,
      max_abs_floor_offset_m: 0,
      physics_modes: [],
      ball_frame_count: 0,
      approx_ball_frame_count: 0,
      paddle_player_count: 0,
      paddle_frame_count: 0,
      ambiguous_paddle_frame_count: 0,
      warnings: [],
    },
  };
}

describe("updateFpsSample", () => {
  it("accumulates frames without reporting inside a ~500ms window", () => {
    const first = updateFpsSample({ framesSinceReport: 0, windowStartMs: 0, fps: 0 }, 100);
    expect(first).toEqual({ framesSinceReport: 1, windowStartMs: 0, fps: 0 });
    const second = updateFpsSample(first, 200);
    expect(second).toEqual({ framesSinceReport: 2, windowStartMs: 0, fps: 0 });
  });

  it("reports an fps reading and resets the window once >=500ms elapse", () => {
    const midway = { framesSinceReport: 29, windowStartMs: 0, fps: 0 };
    const next = updateFpsSample(midway, 500);
    expect(next.framesSinceReport).toBe(0);
    expect(next.windowStartMs).toBe(500);
    expect(next.fps).toBeCloseTo(60, 0); // 30 frames / 500ms = 60fps
  });
});

describe("courtBounds and cameraPresetPose", () => {
  const world = makeWorld([]);

  it("computes court bounds from line segments and the template width/length", () => {
    const bounds = courtBounds(world);
    expect(bounds.width).toBeCloseTo(6.1);
    expect(bounds.length).toBeCloseTo(13.41);
    expect(bounds.centerX).toBeCloseTo(0);
  });

  it("produces three distinct camera poses for broadcast/behind_baseline/top_down", () => {
    const broadcast = cameraPresetPose(world, "broadcast");
    const behindBaseline = cameraPresetPose(world, "behind_baseline");
    const topDown = cameraPresetPose(world, "top_down");
    expect(broadcast.position).not.toEqual(behindBaseline.position);
    expect(broadcast.position).not.toEqual(topDown.position);
    // top_down looks straight down at court level.
    expect(topDown.target[2]).toBe(0);
    expect(topDown.position[2]).toBeGreaterThan(broadcast.position[2]);
  });
});

describe("coaching card facts UI", () => {
  const coachingFacts = {
    artifact_type: "coaching_card_facts",
    rally_scope: "rally_spans",
    priority_rule: ["contact_count_when_present"],
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
        coverage_fraction: 0.42,
        frames_total: 100,
        frames_used: 42,
        metric: "kitchen_time_s",
        player_id: "1",
        rally_id: "rally_001",
        rally_scope: "rally_spans",
        trust: "estimated",
        unit: "s",
        value: 1.25,
      },
    ],
  };

  const chapters: TimelineChapter[] = [
    { index: 1, t0: 0, t1: 1, label: "Rally 1", badge: "preview" },
    { index: 2, t0: 3, t1: 4, label: "Rally 2", badge: "preview" },
  ];

  it("renders missing player facts as not measured rather than zero", () => {
    const buildCard = (AppModule as any).coachingCardForTimeline;
    const CoachingCardPanel = (AppModule as any).CoachingCardPanel;
    expect(typeof buildCard).toBe("function");
    expect(typeof CoachingCardPanel).toBe("function");

    const card = buildCard(coachingFacts, chapters, 0.5, ["1", "2"]);
    const html = renderToStaticMarkup(React.createElement(CoachingCardPanel, { card }));

    expect(html).toContain("Per-rally coaching card");
    expect(html).toContain("contact count");
    expect(html).toContain("3 count");
    expect(html).toContain("ok");
    expect(html).toContain("96.7% coverage");
    expect(html).toContain("not measured");
    expect(html).not.toContain(">0<");
    expect(html).not.toContain("0 count");
  });

  it("switches the rendered facts when the active timeline chapter changes", () => {
    const buildCard = (AppModule as any).coachingCardForTimeline;
    expect(typeof buildCard).toBe("function");

    const first = buildCard(coachingFacts, chapters, 0.5, ["1"]);
    const second = buildCard(coachingFacts, chapters, 3.5, ["1"]);

    expect(first.rallyId).toBe("rally_000");
    expect(first.rows[0].fact.metric).toBe("contact_count");
    expect(first.rows[0].fact.value).toBe(3);
    expect(second.rallyId).toBe("rally_001");
    expect(second.rows[0].fact.metric).toBe("kitchen_time_s");
    expect(second.rows[0].fact.value).toBe(1.25);
  });

  it("joins coaching facts through the chapter rally id instead of the chapter index", () => {
    const buildCard = (AppModule as any).coachingCardForTimeline;
    expect(typeof buildCard).toBe("function");

    const chapters: TimelineChapter[] = [
      { index: 1, rallyId: "rally_042", t0: 0, t1: 2, label: "Rally 42", badge: "preview" },
    ];
    const facts = {
      ...coachingFacts,
      facts: [
        {
          coverage_fraction: 0.5,
          frames_total: 100,
          frames_used: 50,
          metric: "contact_count",
          player_id: "1",
          rally_id: "rally_000",
          rally_scope: "rally_spans",
          trust: "ok",
          unit: "count",
          value: 99,
        },
        {
          coverage_fraction: 0.75,
          frames_total: 100,
          frames_used: 75,
          metric: "distance_covered_m",
          player_id: "1",
          rally_id: "rally_042",
          rally_scope: "rally_spans",
          trust: "estimated",
          unit: "m",
          value: 12.5,
        },
      ],
    };

    const card = buildCard(facts, chapters, 1.0, ["1"]);

    expect(card.rallyId).toBe("rally_042");
    expect(card.rows[0].fact.metric).toBe("distance_covered_m");
    expect(card.rows[0].fact.value).toBe(12.5);
  });
});

describe("contactReadoutText", () => {
  it("uses active body mesh frames as contact evidence when contact_windows is absent", () => {
    expect(
      contactReadoutText(new Set(), [
        {
          playerId: 1,
          meshPlayerId: 1,
          presenceOpacity: 1,
          frame: {
            frame_idx: 150,
            t: 2.5,
            source_window_index: 0,
            blend_weight: 1,
            joints_world: [],
            joint_conf: [],
            mesh_vertices_world: [],
            mesh_faces: [],
            smplx_params: {},
            reasons: ["contact_window"],
          },
        },
      ]),
    ).toBe("3D contact: p1");
  });
});
