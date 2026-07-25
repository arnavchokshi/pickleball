import { describe, expect, it } from "vitest";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { CourtMapPanel } from "./CourtMapPanel";
import { parseBallArcRender } from "./ballArcRender";
import { parseVirtualWorld } from "./viewerData";

const world = parseVirtualWorld({
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
      near_baseline: [[-3.05, 0, 0], [3.05, 0, 0]],
      far_baseline: [[-3.05, 13.41, 0], [3.05, 13.41, 0]],
      left_sideline: [[-3.05, 0, 0], [-3.05, 13.41, 0]],
      right_sideline: [[3.05, 0, 0], [3.05, 13.41, 0]],
      net: [[-3.05, 6.705, 0], [3.05, 6.705, 0]],
    },
    net: {
      endpoints: [[-3.05, 6.705, 0.91], [3.05, 6.705, 0.91]],
      center_height_m: 0.86,
      post_height_m: 0.91,
    },
  },
  players: [
    {
      id: 1,
      side: "near",
      role: "left",
      representation: "track_only",
      frames: [
        {
          t: 0.2,
          track_world_xy: [1.1, 2.2],
          floor_world_xyz: [1.1, 2.2, 0],
          floor_source: "track_footpoint",
          contact_locked: false,
          floor_penetration_m: 0,
          joints_world: [],
          mesh_vertices_world: [],
          joint_count: 0,
          mesh_vertex_count: 0,
          placement_diagnostics: {
            covariance_m2: [[0.04, 0.01], [0.01, 0.09]],
            uncertainty_decomposition: {
              foot_localization_covariance_m2: [[0.03, 0], [0, 0.05]],
              court_calibration_covariance_m2: [[0.01, 0.01], [0.01, 0.04]],
              post_temporal_refinement_covariance_m2: [[0.04, 0.01], [0.01, 0.09]],
              dominant_input: "foot_localization",
            },
            selected_support_signal: {
              name: "native2d",
              pixel_xy: [512, 488],
              court_xy: [1.1, 2.2],
              confidence: 0.92,
            },
            foot_candidates: [
              {
                foot: "left",
                source: "native2d",
                pixel_xy: [512, 488],
                court_xy: [1.1, 2.2],
                confidence: 0.92,
                accepted: true,
                rejection_reason: null,
                keypoint_candidates: [
                  { semantic_name: "left_heel", pixel_xy: [510, 487], confidence: 0.94 },
                  { semantic_name: "left_toe", pixel_xy: [514, 489], confidence: 0.9 },
                ],
              },
            ],
            nearest_regulation_line: {
              line_name: "far_nvz",
              distance_m: 0.08,
              signed_distance_m: 0.08,
              signed_distance_ci95_m: [-0.02, 0.18],
              court_zone: "far_backcourt",
            },
            contact_state: { state: "planted", support_foot: "left" },
            support_state: {
              state: "left_planted",
              foot: "left",
              phase_id: "p1-phase-4",
              transition_reason: "continued_same_support",
            },
            sole_contact: {
              image_points: [[510, 487], [514, 489]],
              court_points: [[1.08, 2.18], [1.12, 2.22]],
              source: "native2d",
            },
            kitchen_decision: {
              spatial_state: "outside",
              court_contact_state: "confirmed_outside",
              gameplay_fault_state: "not_evaluated",
              signed_clearance_m: 0.12,
              signed_clearance_ci99_m: [0.05, 0.19],
              reasons: ["ci99_wholly_outside_nvz"],
            },
            measurement_provenance: "measured",
            metric_eligible: true,
          },
        },
      ],
    },
  ],
  ball: { source: null, frames: [] },
  paddles: [],
  summary: {
    player_count: 1,
    mesh_player_count: 0,
    mesh_player_frame_count: 0,
    joint_player_frame_count: 0,
    track_only_player_frame_count: 1,
    floor_placed_player_frame_count: 1,
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
});

const arcRender = parseBallArcRender({
  schema_version: 1,
  artifact_type: "racketsport_ball_arc_render",
  clip_id: "unit_clip",
  source_artifact: "ball_track_arc_solved.json",
  solver_status: "ran",
  render_only: true,
  not_for_detection_metrics: true,
  trusted_for_ball_detection_metrics: false,
  segments: [
    {
      segment_id: 0,
      t0: 0,
      t1: 0.4,
      frame_start: 0,
      frame_end: 12,
      anchor_types: ["contact", "bounce"],
      anchor_frames: [0, 12],
      confidence: 0.8,
      flight_sanity_verdict: "pass",
      bridge: false,
      shot: {
        start: { world_xyz: [0, 1, 0.5], court_xy: [0, 1] },
        peak: { world_xyz: [0.4, 3, 1.4], court_xy: [0.4, 3] },
        end: { world_xyz: [0.8, 5.7, 0.2], court_xy: [0.8, 5.7] },
        speed_mps: 10,
        speed_mph: 22.4,
        height_over_net_m: 0.3,
        distance_m: 4.8,
        path_distance_m: 5.1,
      },
    },
  ],
  bridges: [],
  samples: [
    {
      t: 0.2,
      frame_float: 6,
      segment_id: 0,
      world_xyz: [0.35, 2.6, 1.1],
      court_xy: [0.35, 2.6],
      confidence: 0.72,
      band: "arc_interpolated",
      bridge: false,
      render_only: true,
      not_for_detection_metrics: true,
    },
  ],
  summary: { segment_count: 1, sample_count: 1, bridge_sample_count: 0, rally_span_count: 0 },
});

describe("CourtMapPanel", () => {
  it("renders a dedicated SVG court map with active shots, persistent bounces, and player positions", () => {
    const html = renderToStaticMarkup(React.createElement(CourtMapPanel, { world, arcRender, currentTime: 0.2 }));

    expect(html).toContain("court-map-panel");
    expect(html).toContain("data-active=\"true\"");
    expect(html).toContain("court-map-shot-line");
    expect(html).toContain("court-map-bounce-dot");
    expect(html).toContain("court-map-current-ball");
    expect(html).toContain("court-map-player");
    expect(html).toContain("court-map-player-uncertainty");
    expect(html).toContain("court-map-placement-readout");
    expect(html).toContain("native2d");
    expect(html).toContain("left_heel+left_toe");
    expect(html).toContain("foot_localization-dominated");
    expect(html).toContain("metric eligible");
    expect(html).toContain("nvz-confirmed-outside");
    expect(html).toContain("p1-phase-4");
    expect(html).toContain("[0.05, 0.19]m");
    expect(html).toContain("aria-label=\"Top-down court map\"");
  });
});
