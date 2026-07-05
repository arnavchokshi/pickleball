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
  styleForBand,
  type BallTrailSample,
} from "./ballTrail";

const fixturePath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "__fixtures__",
  "ball_chain_excerpt.json",
);

function loadFixture(): any {
  return JSON.parse(fs.readFileSync(fixturePath, "utf8"));
}

describe("ball trail honesty styling", () => {
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
