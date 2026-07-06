import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  buildCourtMapShots,
  parseBallArcRender,
  replayViewFromSearch,
  sampleBallArcRenderAtTime,
  svgCourtProjector,
  type BallArcRender,
} from "./ballArcRender";

const renderArtifact = {
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
      confidence: 0.82,
      flight_sanity_verdict: "pass",
      bridge: false,
      shot: {
        start: { world_xyz: [0, 1, 0.55], court_xy: [0, 1] },
        peak: { world_xyz: [0.42, 3.1, 1.45], court_xy: [0.42, 3.1] },
        end: { world_xyz: [0.8, 5.7, 0.2], court_xy: [0.8, 5.7] },
        speed_mps: 11.2,
        speed_mph: 25.05,
        height_over_net_m: 0.34,
        distance_m: 4.77,
        path_distance_m: 5.02,
      },
    },
  ],
  bridges: [
    { bridge_id: "bridge_0", t0: 0.4, t1: 0.6, reason: "rally_span_gap", confidence: 0.2, render_only: true, not_for_detection_metrics: true },
  ],
  samples: [
    { t: 0, frame_float: 0, segment_id: 0, world_xyz: [0, 1, 0.55], court_xy: [0, 1], confidence: 0.82, band: "arc_interpolated", bridge: false, render_only: true, not_for_detection_metrics: true },
    { t: 0.1, frame_float: 3, segment_id: 0, world_xyz: [0.2, 2, 1.1], court_xy: [0.2, 2], confidence: 0.82, band: "arc_interpolated", bridge: false, render_only: true, not_for_detection_metrics: true },
    { t: 0.2, frame_float: 6, segment_id: 0, world_xyz: [0.42, 3.1, 1.45], court_xy: [0.42, 3.1], confidence: 0.82, band: "arc_interpolated", bridge: false, render_only: true, not_for_detection_metrics: true },
    { t: 0.4, frame_float: 12, segment_id: 0, world_xyz: [0.8, 5.7, 0.2], court_xy: [0.8, 5.7], confidence: 0.82, band: "arc_interpolated", bridge: false, render_only: true, not_for_detection_metrics: true },
    { t: 0.5, frame_float: 15, segment_id: "bridge_0", world_xyz: [0.9, 6.1, 0.62], court_xy: [0.9, 6.1], confidence: 0.2, band: "arc_weak", bridge: true, render_only: true, not_for_detection_metrics: true },
  ],
  summary: { segment_count: 1, sample_count: 5, bridge_sample_count: 1, rally_span_count: 1 },
};

describe("ball arc render parser", () => {
  it("loads render-only dense samples and keeps bridge samples visible but low confidence", () => {
    const parsed = parseBallArcRender(renderArtifact);

    expect(parsed.samples).toHaveLength(5);
    expect(parsed.samples.find((sample) => sample.bridge)).toMatchObject({
      visible: true,
      band: "arc_weak",
      conf: 0.2,
      renderOnly: true,
    });
    expect(parsed.samples.every((sample) => sample.world_xyz !== null)).toBe(true);
  });

  it("interpolates the active ball along render samples instead of nearest-frame snapping", () => {
    const parsed = parseBallArcRender(renderArtifact);

    expect(sampleBallArcRenderAtTime(parsed.samples, 0.15)).toMatchObject({
      t: 0.15,
      world_xyz: [0.31, 2.55, 1.275],
      conf: 0.82,
      visible: true,
    });
  });

  it("suppresses render samples when the solver status is untrusted", () => {
    const parsed = parseBallArcRender({ ...renderArtifact, solver_status: "experimental_off" });

    expect(parsed.trusted).toBe(false);
    expect(parsed.samples).toHaveLength(0);
    expect(parsed.segments).toHaveLength(0);
  });
});

describe("court map geometry", () => {
  it("projects court coordinates into a stable SVG viewport with room for labels", () => {
    const projector = svgCourtProjector({ widthM: 6.1, lengthM: 13.41, paddingPx: 24, widthPx: 305, heightPx: 520 });

    expect(projector([-3.05, 0])).toEqual([24, 496]);
    expect(projector([3.05, 13.41])).toEqual([281, 24]);
  });

  it("builds per-shot map records and highlights the shot active at playback time", () => {
    const parsed = parseBallArcRender(renderArtifact) as BallArcRender;
    const shots = buildCourtMapShots(parsed, 0.22);

    expect(shots).toHaveLength(1);
    expect(shots[0]).toMatchObject({
      active: true,
      start: [0, 1],
      end: [0.8, 5.7],
      peak: [0.42, 3.1],
      confidence: 0.82,
    });
  });

  it("projects real final-verify court-map ball paths and current ball samples inside the SVG viewport", () => {
    const root = process.cwd();
    const world = JSON.parse(
      readFileSync(resolve(root, "../../runs/lanes/ball_final_verify_20260705/burlington/virtual_world.json"), "utf8"),
    );
    const render = parseBallArcRender(
      readFileSync(resolve(root, "../../runs/lanes/ball_final_verify_20260705/burlington/ball_arc_render.json"), "utf8"),
    );
    const firstShotTime = render.segments[0].t0 + 0.1;
    const shots = buildCourtMapShots(render, firstShotTime);
    const ball = sampleBallArcRenderAtTime(render.samples, firstShotTime);
    const courtPoints = Object.values(world.court.line_segments).flat() as [number, number, number][];
    const xs = courtPoints.map((point) => point[0]);
    const ys = courtPoints.map((point) => point[1]);
    const projector = svgCourtProjector({
      widthM: world.court.width_m,
      lengthM: world.court.length_m,
      paddingPx: 24,
      widthPx: 305,
      heightPx: 520,
      xMin: Math.min(...xs),
      xMax: Math.max(...xs),
      yMin: Math.min(...ys),
      yMax: Math.max(...ys),
    });

    expect(shots.length).toBeGreaterThan(0);
    expect(ball?.world_xyz).not.toBeNull();
    const projectedPoints = shots.flatMap((shot) => [shot.start, shot.end]);
    projectedPoints.push([ball!.world_xyz![0], ball!.world_xyz![1]]);
    for (const point of projectedPoints) {
      const [x, y] = projector(point);
      expect(x).toBeGreaterThanOrEqual(24);
      expect(x).toBeLessThanOrEqual(281);
      expect(y).toBeGreaterThanOrEqual(24);
      expect(y).toBeLessThanOrEqual(496);
    }
  });

  it("supports the ?view=courtmap verifier hook", () => {
    expect(replayViewFromSearch("?manifest=/@fs/tmp/replay.json&view=courtmap")).toBe("courtmap");
    expect(replayViewFromSearch("?view=world")).toBe("world");
    expect(replayViewFromSearch("")).toBe("world");
  });
});
