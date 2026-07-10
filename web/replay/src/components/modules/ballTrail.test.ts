import { describe, expect, it } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  buildBallTrail,
  ballHudStateForTime,
  extractImpactMarkers,
  parseAutoBounceCandidates,
  parseBallTrailArtifact,
  parseContactWindowsForImpacts,
  parseNetPlane,
  solverOffReadout,
  samplesFromVirtualWorld,
  styleForBand,
  TRUSTED_BALL_ARC_SOLVER_STATUSES,
  type BallTrailSample,
} from "./ballTrail";

const fixturePath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "__fixtures__",
  "ball_chain_excerpt.json",
);

const experimentalOffFixturePath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "__fixtures__",
  "ball_arc_solved_experimental_off_excerpt.json",
);

function loadFixture(): any {
  return JSON.parse(fs.readFileSync(fixturePath, "utf8"));
}

function loadExperimentalOffFixture(): any {
  return JSON.parse(fs.readFileSync(experimentalOffFixturePath, "utf8"));
}

describe("ball trail honesty styling", () => {
  it("preserves physics-predicted provenance with distinct styling and HUD text", () => {
    expect(styleForBand("physics_predicted", 0.8)).toMatchObject({
      band: "physics_predicted",
      label: "physics predicted",
      linePattern: "dashed",
    });
    const samples = samplesFromVirtualWorld({
      ball: { frames: [{ t: 1, conf: 0.8, visible: true, world_xyz: [0, 0, 1], confidence_provenance: { band: "physics_predicted" } }] },
    });
    expect(samples[0].band).toBe("physics_predicted");
    expect(ballHudStateForTime(samples, 1).label).toBe("ball: physics predicted");
    expect(styleForBand("physics_predicted_low", 0.9)).toMatchObject({
      band: "physics_predicted_low",
      label: "physics predicted low",
      lowConfidence: true,
    });
  });

  it("maps measured, predicted, weak, and hidden bands to distinct render vocabulary", () => {
    expect(styleForBand("anchored_measured", 0.91)).toMatchObject({
      hudState: "measured",
      linePattern: "solid",
      rendersTrail: true,
      rendersBall: true,
      pulsingBall: false,
      lowConfidence: false,
    });
    expect(styleForBand("arc_interpolated", 0.88)).toMatchObject({
      hudState: "predicted",
      linePattern: "dashed",
      rendersTrail: true,
      rendersBall: true,
      pulsingBall: true,
    });
    expect(styleForBand("arc_weak", 0.2)).toMatchObject({
      hudState: "predicted",
      linePattern: "dashed",
      lowConfidence: true,
    });
    expect(styleForBand("hidden", 0)).toMatchObject({
      hudState: "not_visible",
      rendersTrail: false,
      rendersBall: false,
    });
  });

  it("windows the trail and never connects through hidden frames", () => {
    const samples: BallTrailSample[] = [
      { t: 0, band: "anchored_measured", conf: 0.9, visible: true, world_xyz: [0, 0, 0.2] },
      { t: 0.4, band: "arc_interpolated", conf: 0.8, visible: true, world_xyz: [0, 1, 0.4] },
      { t: 0.7, band: "hidden", conf: 0, visible: false, world_xyz: [0, 2, 0.5] },
      { t: 0.9, band: "anchored_measured", conf: 0.9, visible: true, world_xyz: [0, 3, 0.3] },
      { t: 1.1, band: "anchored_measured", conf: 0.9, visible: true, world_xyz: [0, 4, 0.2] },
    ];

    const trail = buildBallTrail(samples, 1.1, { windowSeconds: 1.1 });

    expect(trail.segments).toHaveLength(2);
    expect(trail.segments.map((segment) => [segment.from.t, segment.to.t])).toEqual([
      [0, 0.4],
      [0.9, 1.1],
    ]);
    expect(trail.segments[0].style.linePattern).toBe("dashed");
    expect(trail.hiddenGapCount).toBe(1);
  });

  it("never draws across a segment change unless a trusted bridge sample is explicit", () => {
    const samples: BallTrailSample[] = [
      { t: 0, band: "anchored_measured", conf: 0.9, visible: true, world_xyz: [0, 0, 0], segmentId: 1 },
      { t: 0.1, band: "anchored_measured", conf: 0.9, visible: true, world_xyz: [1, 0, 0], segmentId: 2 },
    ];
    expect(buildBallTrail(samples, 0.1).segments).toHaveLength(0);
    expect(buildBallTrail([{ ...samples[0] }, { ...samples[1], bridge: true }], 0.1).segments).toHaveLength(1);
  });

  it("reports current HUD state with low-confidence predicted positions separated from missing ball", () => {
    const samples: BallTrailSample[] = [
      { t: 1, band: "arc_extrapolated", conf: 0.31, visible: true, world_xyz: [1, 2, 0.4] },
    ];

    expect(ballHudStateForTime(samples, 1.01, { confidenceThreshold: 0.5 })).toMatchObject({
      state: "predicted",
      lowConfidence: true,
      label: "ball: predicted",
    });
    expect(ballHudStateForTime([], 1.01)).toMatchObject({
      state: "not_visible",
      label: "ball: not visible",
    });
  });
});


describe("trusted-status allowlist (fail-closed gate)", () => {
  it("only trusts the solver's success status \"ran\"", () => {
    expect(TRUSTED_BALL_ARC_SOLVER_STATUSES.has("ran")).toBe(true);
    expect(TRUSTED_BALL_ARC_SOLVER_STATUSES.has("experimental_off")).toBe(false);
    expect(TRUSTED_BALL_ARC_SOLVER_STATUSES.has("degenerate_zero_segments")).toBe(false);
  });

  it("suppresses a real experimental_off self-kill artifact end to end (RED before the fix, GREEN after)", () => {
    // This fixture is a trimmed excerpt of the live-measured defect: the real
    // producer wrote status="experimental_off" with a physical-sanity kill
    // reason, yet still tagged frames band="anchored_measured". Before the
    // fail-closed gate, parseBallTrailArtifact would have returned those
    // frames as measured samples (the bug). After the fix, no frame survives.
    const fixture = loadExperimentalOffFixture();
    expect(fixture.status).toBe("experimental_off");
    expect(fixture.frames.some((frame: any) => frame.band === "anchored_measured")).toBe(true);

    const artifact = parseBallTrailArtifact(fixture);

    expect(artifact.trusted).toBe(false);
    expect(artifact.status).toBe("experimental_off");
    expect(artifact.killReasons).toEqual(["physical_sanity_violation_fraction 0.400000 exceeds 0.200000"]);
    expect(artifact.samples).toHaveLength(0);
    expect(artifact.segments).toHaveLength(0);
  });

  it("suppresses any unknown or future status value, not just the two known self-kill values", () => {
    const artifact = parseBallTrailArtifact({
      schema_version: 1,
      artifact_type: "racketsport_ball_track_arc_solved",
      clip_id: "unit_test_clip",
      status: "some_future_solver_mode_not_yet_allowlisted",
      kill_reasons: [],
      frames: [
        { t: 0, band: "anchored_measured", conf: 0.9, visible: true, world_xyz: [0, 1, 0.3] },
      ],
      segments: [{ segment_id: 0, t0: 0, t1: 0.1, net_clearance_m: null, net_clearance_ok: null }],
    });

    expect(artifact.trusted).toBe(false);
    expect(artifact.status).toBe("some_future_solver_mode_not_yet_allowlisted");
    expect(artifact.samples).toHaveLength(0);
    expect(artifact.segments).toHaveLength(0);
  });

  it("also suppresses the degenerate_zero_segments self-kill status", () => {
    const artifact = parseBallTrailArtifact({
      schema_version: 1,
      artifact_type: "racketsport_ball_track_arc_solved",
      clip_id: "unit_test_clip",
      status: "degenerate_zero_segments",
      kill_reasons: ["zero accepted segments with at least one rally anchor"],
      frames: [{ t: 0, band: "arc_weak", conf: 0.2, visible: true, world_xyz: [0, 1, 0.3] }],
    });

    expect(artifact.trusted).toBe(false);
    expect(artifact.samples).toHaveLength(0);
  });

  it("keeps the healthy status=\"ran\" path fully unaffected, including band styling", () => {
    const artifact = parseBallTrailArtifact({
      schema_version: 1,
      artifact_type: "racketsport_ball_track_arc_solved",
      clip_id: "unit_test_clip",
      status: "ran",
      kill_reasons: [],
      frames: [
        { t: 0, band: "anchored_measured", conf: 0.91, visible: true, world_xyz: [0, 1, 0.3] },
        { t: 0.1, band: "arc_interpolated", conf: 0.8, visible: true, world_xyz: [0, 1.1, 0.3] },
      ],
      segments: [{ segment_id: 0, t0: 0, t1: 0.1, net_clearance_m: null, net_clearance_ok: null }],
    });

    expect(artifact.trusted).toBe(true);
    expect(artifact.samples).toHaveLength(2);
    expect(artifact.segments).toHaveLength(1);
    expect(styleForBand(artifact.samples[0].band, artifact.samples[0].conf)).toMatchObject({
      hudState: "measured",
      linePattern: "solid",
    });
  });

  it("defaults a missing status key to trusted \"ran\" (legacy/synthetic fixtures only; real producer always writes status)", () => {
    const fixture = loadFixture();
    expect(fixture.ball_track_arc_solved.status).toBeUndefined();

    const artifact = parseBallTrailArtifact(fixture.ball_track_arc_solved);

    expect(artifact.trusted).toBe(true);
    expect(artifact.status).toBe("ran");
    expect(artifact.samples.length).toBeGreaterThan(0);
  });

  it("gives an honest fail-closed HUD readout naming the kill reason, not a silent not-visible", () => {
    const readout = solverOffReadout(["physical_sanity_violation_fraction 0.400000 exceeds 0.200000"]);

    expect(readout.state).toBe("solver_off");
    expect(readout.label).toContain("solver off");
    expect(readout.label).toContain("physical_sanity_violation_fraction 0.400000 exceeds 0.200000");
    expect(readout.sample).toBeNull();
  });

  it("still names something even when kill_reasons is empty", () => {
    const readout = solverOffReadout([]);
    expect(readout.state).toBe("solver_off");
    expect(readout.label.length).toBeGreaterThan(0);
  });
});

describe("ball impact extraction", () => {
  it("extracts bounce, contact, derived net hit, and persistent landing markers", () => {
    const samples: BallTrailSample[] = [
      { t: 0, band: "anchored_measured", conf: 1, visible: true, world_xyz: [0, 7.1, 0.3] },
      { t: 0.1, band: "anchored_measured", conf: 1, visible: true, world_xyz: [0, 6.5, 0.2] },
      { t: 0.2, band: "anchored_measured", conf: 1, visible: true, world_xyz: [0.4, 5.5, 0.15] },
    ];
    const markers = extractImpactMarkers({
      samples,
      bounceCandidates: [
        { t: 0.2, frame: 6, position: [0.4, 5.5, 0.0371], confidence: 0.8, source: "auto_bounce_candidate" },
      ],
      contactWindows: {
        events: [
          { type: "contact", t: 0.1, frame: 3, player_id: 2, confidence: 0.7, window: { t0: 0.08, t1: 0.12, importance: 1 } },
        ],
      },
      netPlane: { point: [0, 6.705, 0], normal: [0, 1, 0], topHeightM: 0.91 },
      landingLimit: 3,
    });

    expect(markers.map((marker) => marker.kind)).toEqual([
      "floor_bounce",
      "landing_spot",
      "paddle_contact",
      "net_hit",
    ]);
    expect(markers.find((marker) => marker.kind === "paddle_contact")).toMatchObject({
      playerId: 2,
      positionSource: "nearest_ball_sample",
    });
    expect(markers.find((marker) => marker.kind === "net_hit")).toMatchObject({
      derivation: "derived_crossing_below_net_top",
      derived: true,
    });
  });

  it("consumes a real chain-run excerpt without touching Outdoor/Indoor labels", () => {
    const fixture = loadFixture();
    const artifact = parseBallTrailArtifact(fixture.ball_track_arc_solved);
    const bounces = parseAutoBounceCandidates(fixture.auto_bounce_candidates);
    const contacts = parseContactWindowsForImpacts(fixture.contact_windows);
    const netPlane = parseNetPlane(fixture.net_plane);

    const trail = buildBallTrail(artifact.samples, 0.45, { windowSeconds: 1.5 });
    const markers = extractImpactMarkers({
      samples: artifact.samples,
      bounceCandidates: bounces,
      contactWindows: contacts,
      netPlane,
    });

    expect(fixture.chain_manifest.policy.outdoor_indoor_labels_read).toBe(false);
    expect(artifact.samples[0]).toMatchObject({ band: "anchored_measured", renderOnly: true });
    expect(artifact.samples.some((sample) => sample.band === "hidden")).toBe(true);
    expect(trail.hiddenGapCount).toBeGreaterThan(0);
    expect(trail.segments.some((segment) => segment.style.linePattern === "dashed")).toBe(true);
    expect(markers.some((marker) => marker.kind === "floor_bounce")).toBe(true);
    expect(markers.some((marker) => marker.kind === "paddle_contact")).toBe(true);
    expect(markers.some((marker) => marker.kind === "net_hit" && marker.derived)).toBe(true);
  });
});
