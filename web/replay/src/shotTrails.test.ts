import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  buildShotTrailGroups,
  filterShots,
  parseBallArcSolved,
  parseShots,
  qualityBandForShot,
  shotOutcomeColor,
  type ShotTrailFilters,
} from "./shotTrails";

describe("shot trails data", () => {
  const wolverineShotsPath = resolve(
    process.cwd(),
    "../../runs/shot_taxonomy_20260703T01Z/wolverine_mixed_0200_mid_steep_corner/shots.json",
  );
  const wolverineArcPath = resolve(
    process.cwd(),
    "../../runs/ball_arc_event_subset_20260703T02Z/wolverine_mixed_0200_mid_steep_corner/ball_track_arc_solved.json",
  );
  const finalVerifyBurlingtonArcPath = resolve(
    process.cwd(),
    "../../runs/lanes/ball_final_verify_20260705/burlington/ball_track_arc_solved.json",
  );

  it("parses the real Wolverine shots.json fixture including landing uncertainty ellipses", () => {
    const shots = parseShots(readFileSync(wolverineShotsPath, "utf8"));

    expect(shots.artifact_type).toBe("racketsport_shots");
    expect(shots.policy.internal_val_only).toBe(true);
    expect(shots.shots).toHaveLength(6);
    expect(shots.shots[0]).toMatchObject({
      confidence: 0.743664,
      player_id: 3,
      segment_id: 2,
      shot_type: "drive",
      speed_mph: 31.714,
      outcome: { call: "out", let_candidate: false },
    });
    expect(shots.shots[0].landing?.uncertainty_ellipse).toEqual({
      angle_deg: -99.88,
      center_xy: [1.404037, -11.853341],
      semi_major_m: 0.758449,
      semi_minor_m: 0.35,
      source: "segment_sigma_endpoint_reprojection_v1",
    });
  });

  it("maps outcomes to the requested green/amber/red quality palette", () => {
    const base = parseShots({
      artifact_type: "racketsport_shots",
      schema_version: 1,
      clip_id: "clip",
      policy: { internal_val_only: true, not_for_detection_metrics: true, not_ground_truth: true },
      shots: [
        {
          shot_id: "high-in",
          segment_id: 1,
          player_id: 1,
          shot_type: "drive",
          shot_type_abstained: false,
          outcome: { call: "in", let_candidate: false, faults: [] },
          confidence: 0.82,
          speed_mph: 34,
          t: 1,
          frame: 30,
        },
        {
          shot_id: "mid-in",
          segment_id: 2,
          player_id: 1,
          shot_type: "dink",
          shot_type_abstained: false,
          outcome: { call: "in", let_candidate: true, faults: ["let_candidate"] },
          confidence: 0.56,
          speed_mph: 12,
          t: 2,
          frame: 60,
        },
        {
          shot_id: "low-in",
          segment_id: 3,
          player_id: 2,
          shot_type: null,
          shot_type_abstained: true,
          outcome: { call: "in", let_candidate: false, faults: [] },
          confidence: 0.32,
          speed_mph: 9,
          t: 3,
          frame: 90,
        },
        {
          shot_id: "fault",
          segment_id: 4,
          player_id: 2,
          shot_type: "lob",
          shot_type_abstained: false,
          outcome: { call: "net", let_candidate: false, faults: ["net"] },
          confidence: 0.77,
          speed_mph: 41,
          t: 4,
          frame: 120,
        },
      ],
    });

    expect(qualityBandForShot(base.shots[0])).toBe("high");
    expect(qualityBandForShot(base.shots[1])).toBe("mid");
    expect(qualityBandForShot(base.shots[2])).toBe("low");
    expect(shotOutcomeColor(base.shots[0])).toBe("#16a05d");
    expect(shotOutcomeColor(base.shots[1])).toBe("#7fc96f");
    expect(shotOutcomeColor(base.shots[2])).toBe("#d49b24");
    expect(shotOutcomeColor(base.shots[3])).toBe("#df3f31");
  });

  it("filters by player, shot type, outcome, and quality without mutating order", () => {
    const shots = parseShots(readFileSync(wolverineShotsPath, "utf8")).shots;

    const filters: ShotTrailFilters = {
      playerId: 2,
      shotType: "atp",
      outcome: "out",
      quality: "mid",
    };

    expect(filterShots(shots, filters).map((shot) => shot.shot_id)).toEqual([
      "wolverine_mixed_0200_mid_steep_corner:contact_018_p2_right",
    ]);
    expect(filterShots(shots, { playerId: null, shotType: "all", outcome: "all", quality: "all" })).toHaveLength(6);
  });

  it("groups shot trails from the arc-solved per-frame segment, never from endpoint chords", () => {
    const shots = parseShots(readFileSync(wolverineShotsPath, "utf8"));
    const arc = parseBallArcSolved(readFileSync(wolverineArcPath, "utf8"));

    const groups = buildShotTrailGroups(shots.shots, arc);
    const first = groups.find((group) => group.shot.shot_id === "wolverine_mixed_0200_mid_steep_corner:contact_004_p3_left");

    expect(first?.segment_id).toBe(2);
    expect(first?.points).toHaveLength(13);
    expect(first?.segments).toHaveLength(12);
    expect(first?.points[0]).toEqual([3.883701096, 2.384555358, 0.677551678]);
    expect(first?.points.at(-1)).toEqual([3.057036419, -2.362040684, 1.659497167]);
    expect(first?.segments).not.toContainEqual({
      from: [3.940276, 2.7094, 0.556422],
      to: [1.404037, -11.853341, 0.0371],
    });
  });

  it("marks the real Wolverine arc-solved fixture as trusted (status=ran)", () => {
    const arc = parseBallArcSolved(readFileSync(wolverineArcPath, "utf8"));

    expect(arc.status).toBe("ran");
    expect(arc.trusted).toBe(true);
    expect(arc.frames.length).toBeGreaterThan(0);
  });

  it("loads the final-verify schema v2 arc-solved fixture without rejecting the unchanged render fields", () => {
    const arc = parseBallArcSolved(readFileSync(finalVerifyBurlingtonArcPath, "utf8"));

    expect(arc.schema_version).toBe(2);
    expect(arc.status).toBe("ran");
    expect(arc.trusted).toBe(true);
    expect(arc.clip_id).toBe("burlington");
    expect(arc.killReasons).toEqual([]);
    expect(arc.frames.length).toBeGreaterThan(0);
    expect(arc.frames.some((frame) => frame.world_xyz !== null && frame.segment_id !== null)).toBe(true);
  });

  it("fail-closed gates a self-killed arc-solved artifact: no frames survive and shot trails draw nothing measured from it", () => {
    const shots = parseShots(readFileSync(wolverineShotsPath, "utf8"));
    const arc = parseBallArcSolved(readFileSync(wolverineArcPath, "utf8"));
    const killedArc = { ...arc, status: "experimental_off", trusted: false, killReasons: ["physical_sanity_violation_fraction 0.400000 exceeds 0.200000"], frames: [] };

    const groups = buildShotTrailGroups(shots.shots, killedArc);

    expect(groups.length).toBe(shots.shots.length);
    expect(groups.every((group) => group.points.length === 0)).toBe(true);
    expect(groups.every((group) => group.segments.length === 0)).toBe(true);
  });

  it("parseBallArcSolved itself suppresses frames when the artifact's own status is untrusted", () => {
    const arc = parseBallArcSolved({
      schema_version: 1,
      artifact_type: "racketsport_ball_track_arc_solved",
      clip_id: "unit_test_clip",
      status: "experimental_off",
      kill_reasons: ["physical_sanity_violation_fraction 0.400000 exceeds 0.200000"],
      frames: [{ t: 0, visible: true, conf: 0.9, sigma_m: null, world_xyz: [0, 1, 0.3], segment_id: 0 }],
    });

    expect(arc.trusted).toBe(false);
    expect(arc.frames).toHaveLength(0);
    expect(arc.killReasons).toEqual(["physical_sanity_violation_fraction 0.400000 exceeds 0.200000"]);
  });
});
