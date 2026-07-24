import { describe, expect, it } from "vitest";

import {
  courtEvidenceSegments,
  parseCourtCalibrationEvidence,
  parseCourtEvidence,
  parseSam3DKeypointEvidence,
  projectCourtWorldPoint,
  sam3DKeypointFramesForTime,
} from "./evidenceData";

describe("replay evidence parsing and projection", () => {
  it("parses named court points and builds only regulation evidence segments", () => {
    const evidence = parseCourtEvidence({
      artifact_type: "racketsport_court_lock_visualization_adapter",
      trust: { authority_state: "review_only", measurement_valid: false },
      named_floor_correspondences: [
        { semantic_name: "near_left_corner", raw_image_xy: [10, 90], source: "lock" },
        { semantic_name: "near_baseline_center", raw_image_xy: [50, 90], source: "lock" },
        { semantic_name: "near_right_corner", raw_image_xy: [90, 90], source: "lock" },
      ],
    });
    expect(evidence.measurementValid).toBe(false);
    expect(evidence.points).toHaveLength(3);
    expect(courtEvidenceSegments(evidence)).toHaveLength(2);
  });

  it("projects court-world joints with the declared world-to-camera convention", () => {
    const calibration = parseCourtCalibrationEvidence({
      image_size: [1920, 1080],
      intrinsics: { fx: 1000, fy: 1000, cx: 960, cy: 540, dist: [] },
      extrinsics: { R: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], t: [0, 0, 10] },
    });
    expect(projectCourtWorldPoint([1, 2, 0], calibration)).toEqual([1060, 740]);
    expect(projectCourtWorldPoint([0, 0, -11], calibration)).toBeNull();
  });

  it("returns only exact-time BODY foot evidence and never holds it through a gap", () => {
    const evidence = parseSam3DKeypointEvidence({
      artifact_type: "racketsport_sam3d_keypoints_2d",
      source: "orchestrator_body_stage",
      players: [{ id: 7, frames: [{ frame_idx: 3, t: 0.1, keypoints: [{ name: "left_heel", index: 17, xy_px: [5, 6], conf: 0.9 }] }] }],
    });
    expect(sam3DKeypointFramesForTime(evidence, 0.1, 30)).toHaveLength(1);
    expect(sam3DKeypointFramesForTime(evidence, 0.2, 30)).toEqual([]);
  });
});
