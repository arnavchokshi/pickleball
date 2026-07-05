import { describe, expect, it } from "vitest";

import {
  PICKLEBALL_COURT_REVIEW_POINTS,
  buildCourtAssistSeedPayload,
  buildCourtCornersPayload,
  buildReviewedCourtCorrection,
  importCourtProposals,
  validateCourtReviewPoints,
  type CourtAssistSeed,
  type CourtReviewPointMap,
} from "./courtReview";

function predictedPoints(): CourtReviewPointMap {
  const points: CourtReviewPointMap = {};
  for (const [index, name] of PICKLEBALL_COURT_REVIEW_POINTS.entries()) {
    points[name] = { xy: [180 + index * 8, 220 + index * 3], confidence: 0.72 };
  }
  points.near_left_corner = { xy: [180, 520], confidence: 0.72 };
  points.near_baseline_center = { xy: [500, 520], confidence: 0.72 };
  points.near_right_corner = { xy: [820, 520], confidence: 0.72 };
  points.far_right_corner = { xy: [780, 180], confidence: 0.72 };
  points.far_baseline_center = { xy: [500, 180], confidence: 0.72 };
  points.far_left_corner = { xy: [220, 180], confidence: 0.72 };
  return points;
}

describe("validateCourtReviewPoints", () => {
  it("flags missing points, low prediction confidence, and invalid corner geometry", () => {
    const predicted = predictedPoints();
    const adjusted = Object.fromEntries(Object.entries(predicted).map(([name, point]) => [name, { ...point }])) as CourtReviewPointMap;
    delete adjusted.net_center;
    predicted.near_left_corner!.confidence = 0.1;
    adjusted.far_left_corner = { xy: [900, 560], confidence: 0.72 };

    const report = validateCourtReviewPoints({ predicted, adjusted, imageSize: [1000, 600] });

    expect(report.status).toBe("warn");
    expect(report.warnings.map((warning) => warning.code)).toEqual(
      expect.arrayContaining(["missing_point", "low_prediction_confidence", "bad_geometry"]),
    );
  });
});

describe("buildCourtCornersPayload", () => {
  it("exports the four court corners in process_video court_corners.json format", () => {
    const payload = buildCourtCornersPayload({
      adjusted: predictedPoints(),
      imageSize: [1000, 600],
      frameIndex: 42,
      source: "court_detector_v2:selected_hypothesis=hypothesis_0001",
      reviewStatus: "auto_predicted_unreviewed",
    });

    const item = payload.annotation.items[0];
    expect(item.image_size).toEqual([1000, 600]);
    expect(item.frame).toBe("frame_000042.jpg");
    expect(item.status).toBe("auto_preview_unverified");
    expect(item.not_cal3_verified).toBe(true);
    expect(item.court_corners).toEqual({
      near_left: [180, 520],
      near_right: [820, 520],
      far_right: [780, 180],
      far_left: [220, 180],
    });
  });
});

describe("buildReviewedCourtCorrection", () => {
  it("records source metadata, original predictions, adjusted points, and moved flags", () => {
    const predicted = predictedPoints();
    const adjusted = Object.fromEntries(Object.entries(predicted).map(([name, point]) => [name, { ...point }])) as CourtReviewPointMap;
    adjusted.near_left_corner = { xy: [174, 526], confidence: 0.72 };

    const artifact = buildReviewedCourtCorrection({
      videoId: "match_01",
      videoPath: "match_01.mp4",
      videoSha256: "d".repeat(64),
      imageSize: [1000, 600],
      frameIndex: 42,
      frameTimeSeconds: 1.4,
      autoPredictionSource: "court_detector_v2:selected_hypothesis=hypothesis_0001",
      predicted,
      adjusted,
      createdAt: "2026-07-04T12:00:00Z",
    });

    expect(artifact.artifact_type).toBe("racketsport_reviewed_court_calibration");
    expect(artifact.review_status).toBe("human_reviewed");
    expect(artifact.auto_prediction.verified).toBe(false);
    expect(artifact.points.near_left_corner.manual_moved).toBe(true);
    expect(artifact.points.near_right_corner.manual_moved).toBe(false);
    expect(artifact.training.usable_for_court_detector_training).toBe(true);
  });

  it("marks one-click auto predictions as unreviewed and not detector-training-ready", () => {
    const predicted = predictedPoints();
    const adjusted = Object.fromEntries(Object.entries(predicted).map(([name, point]) => [name, { ...point }])) as CourtReviewPointMap;

    const artifact = buildReviewedCourtCorrection({
      videoId: "match_01",
      videoPath: "match_01.mp4",
      videoSha256: "d".repeat(64),
      imageSize: [1000, 600],
      frameIndex: 42,
      frameTimeSeconds: 1.4,
      autoPredictionSource: "court_detector_v2:selected_hypothesis=hypothesis_0001",
      predicted,
      adjusted,
      createdAt: "2026-07-04T12:00:00Z",
      reviewStatus: "auto_predicted_unreviewed",
    });

    expect(artifact.review_status).toBe("auto_predicted_unreviewed");
    expect(artifact.pipeline.trust).toBe("auto_predicted_unreviewed_court_layout");
    expect(artifact.training.usable_for_court_detector_training).toBe(false);
    expect(artifact.training.training_policy).toBe("auto_prediction_not_training_ready");
    expect(Object.values(artifact.points).every((point) => point.manual_moved === false)).toBe(true);
  });
});

describe("importCourtProposals", () => {
  it("imports an empty proposal artifact as review-needed and fail-closed", () => {
    const state = importCourtProposals({
      artifact_type: "racketsport_court_proposals",
      schema_version: 1,
      status: "ranked_not_verified",
      verified: false,
      not_cal3_verified: true,
      ranking: { selected_proposal_id: null, abstain: true, abstain_reasons: ["not_cal3_verified"] },
      assist: { mode: "none", tap_points: [], line_label: null },
      proposals: [],
    });

    expect(state.status).toBe("needs_review");
    expect(state.verified).toBe(false);
    expect(state.notCal3Verified).toBe(true);
    expect(state.proposals).toHaveLength(0);
  });

  it("imports the selected proposal points and one-tap assist constraint", () => {
    const state = importCourtProposals({
      artifact_type: "racketsport_court_proposals",
      schema_version: 1,
      status: "ranked_not_verified",
      verified: false,
      not_cal3_verified: true,
      ranking: { selected_proposal_id: "proposal_0001", abstain: true, abstain_reasons: ["not_cal3_verified"] },
      assist: { mode: "one_inside_tap", tap_points: [[500, 300]], line_label: null, trusted_calibration: false },
      proposals: [
        {
          proposal_id: "proposal_0001",
          source: "unit",
          verified: false,
          not_cal3_verified: true,
          court_keypoints: { near_left_corner: [180, 520], near_right_corner: [820, 520] },
          scores: { overall: 0.2 },
          gate: { auto_usable: false, review_usable: true, failed: ["not_verified"], warnings: [] },
          evidence: {},
        },
      ],
    });

    expect(state.selectedProposalId).toBe("proposal_0001");
    expect(state.proposals[0].points.near_left_corner?.xy).toEqual([180, 520]);
    expect(state.assist.mode).toBe("one_inside_tap");
    expect(state.assist.trustedCalibration).toBe(false);
  });
});

describe("buildCourtAssistSeedPayload", () => {
  it("serializes tap points and mode to the snake_case wire format, always untrusted", () => {
    const assist: CourtAssistSeed = {
      mode: "two_line_taps",
      tapPoints: [
        { x: 120, y: 340 },
        { x: 480, y: 340 },
      ],
      lineLabel: "near_baseline",
      trustedCalibration: false,
    };

    const payload = buildCourtAssistSeedPayload(assist);

    expect(payload).toEqual({
      mode: "two_line_taps",
      tap_points: [
        [120, 340],
        [480, 340],
      ],
      line_label: "near_baseline",
      trusted_calibration: false,
    });
  });

  it("round-trips through importCourtProposals's assist parser", () => {
    const assist: CourtAssistSeed = {
      mode: "one_inside_tap",
      tapPoints: [{ x: 500, y: 300 }],
      lineLabel: null,
      trustedCalibration: false,
    };

    const wirePayload = buildCourtAssistSeedPayload(assist);
    const state = importCourtProposals({
      artifact_type: "racketsport_court_proposals",
      schema_version: 1,
      status: "ranked_not_verified",
      verified: false,
      not_cal3_verified: true,
      ranking: { selected_proposal_id: null, abstain: true, abstain_reasons: ["not_cal3_verified"] },
      assist: wirePayload,
      proposals: [],
    });

    expect(state.assist).toEqual(assist);
  });
});
