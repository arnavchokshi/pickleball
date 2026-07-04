export const PICKLEBALL_COURT_REVIEW_POINTS = [
  "near_left_corner",
  "near_baseline_center",
  "near_right_corner",
  "far_right_corner",
  "far_baseline_center",
  "far_left_corner",
  "near_nvz_left",
  "near_nvz_center",
  "near_nvz_right",
  "net_left_sideline",
  "net_center",
  "net_right_sideline",
  "far_nvz_left",
  "far_nvz_center",
  "far_nvz_right",
] as const;

export type CourtReviewPointName = (typeof PICKLEBALL_COURT_REVIEW_POINTS)[number];
export type CourtReviewPoint = { xy: [number, number]; confidence: number };
export type CourtReviewPointMap = Partial<Record<CourtReviewPointName, CourtReviewPoint>>;
export type CourtReviewWarning = { code: string; severity: "warning"; point?: CourtReviewPointName; points?: CourtReviewPointName[]; message: string };
export type CourtReviewValidation = { status: "pass" | "warn"; warnings: CourtReviewWarning[]; point_count: number; required_point_count: number };
export type CourtReviewStatus = "human_reviewed" | "auto_predicted_unreviewed";
export type CourtCornersPayload = {
  annotation: {
    items: Array<{
      court_corners: {
        near_left: [number, number];
        near_right: [number, number];
        far_right: [number, number];
        far_left: [number, number];
      };
      frame: string;
      image_size: [number, number];
      source: string;
      status: "corrected_unverified" | "auto_preview_unverified";
      not_cal3_verified: boolean;
      review_status: CourtReviewStatus;
    }>;
  };
};

export const COURT_REVIEW_LINES: Array<{ id: string; points: [CourtReviewPointName, CourtReviewPointName] }> = [
  { id: "near_baseline", points: ["near_left_corner", "near_right_corner"] },
  { id: "far_baseline", points: ["far_left_corner", "far_right_corner"] },
  { id: "left_sideline", points: ["near_left_corner", "far_left_corner"] },
  { id: "right_sideline", points: ["near_right_corner", "far_right_corner"] },
  { id: "near_nvz", points: ["near_nvz_left", "near_nvz_right"] },
  { id: "far_nvz", points: ["far_nvz_left", "far_nvz_right"] },
  { id: "near_centerline", points: ["near_baseline_center", "near_nvz_center"] },
  { id: "far_centerline", points: ["far_baseline_center", "far_nvz_center"] },
  { id: "net", points: ["net_left_sideline", "net_right_sideline"] },
];

export type ReviewedCourtCorrection = {
  schema_version: 1;
  artifact_type: "racketsport_reviewed_court_calibration";
  review_status: CourtReviewStatus;
  source_video: { id: string; path: string; sha256: string };
  frame: { index: number; time_s: number; image_size: [number, number] };
  auto_prediction: {
    source: string;
    verified: false;
    not_cal3_verified: true;
    points: Record<CourtReviewPointName, CourtReviewPoint>;
  };
  points: Record<
    CourtReviewPointName,
    {
      predicted_xy: [number, number];
      adjusted_xy: [number, number];
      confidence: number;
      manual_moved: boolean;
    }
  >;
  validation: CourtReviewValidation;
  pipeline: {
    derived_artifact: "court_calibration.json";
    handoff: "--court-calibration";
    trust: "reviewed_manual_court_layout" | "auto_predicted_unreviewed_court_layout";
  };
  training: {
    usable_for_court_detector_training: boolean;
    training_policy: "human_reviewed_not_eval_promoted" | "auto_prediction_not_training_ready";
    protected_eval_clip: boolean;
  };
  created_at: string;
};

export function validateCourtReviewPoints({
  predicted,
  adjusted,
  imageSize,
}: {
  predicted: CourtReviewPointMap;
  adjusted: CourtReviewPointMap;
  imageSize: [number, number];
}): CourtReviewValidation {
  const warnings: CourtReviewWarning[] = [];
  for (const name of PICKLEBALL_COURT_REVIEW_POINTS) {
    const point = adjusted[name];
    if (!point) {
      warnings.push({ code: "missing_point", severity: "warning", point: name, message: `${name} is missing.` });
      continue;
    }
    if (point.xy[0] < 0 || point.xy[0] > imageSize[0] || point.xy[1] < 0 || point.xy[1] > imageSize[1]) {
      warnings.push({ code: "out_of_frame", severity: "warning", point: name, message: `${name} is outside the video frame.` });
    }
    const confidence = predicted[name]?.confidence;
    if (typeof confidence === "number" && Number.isFinite(confidence) && confidence < 0.25) {
      warnings.push({
        code: "low_prediction_confidence",
        severity: "warning",
        point: name,
        message: `${name} started from a low-confidence prediction.`,
      });
    }
  }
  const geometryWarning = courtGeometryWarning(adjusted, imageSize);
  if (geometryWarning) warnings.push(geometryWarning);
  return {
    status: warnings.length ? "warn" : "pass",
    warnings,
    point_count: Object.keys(adjusted).length,
    required_point_count: PICKLEBALL_COURT_REVIEW_POINTS.length,
  };
}

export function buildReviewedCourtCorrection({
  videoId,
  videoPath,
  videoSha256,
  imageSize,
  frameIndex,
  frameTimeSeconds,
  autoPredictionSource,
  predicted,
  adjusted,
  createdAt,
  reviewStatus = "human_reviewed",
}: {
  videoId: string;
  videoPath: string;
  videoSha256: string;
  imageSize: [number, number];
  frameIndex: number;
  frameTimeSeconds: number;
  autoPredictionSource: string;
  predicted: CourtReviewPointMap;
  adjusted: CourtReviewPointMap;
  createdAt: string;
  reviewStatus?: CourtReviewStatus;
}): ReviewedCourtCorrection {
  const normalizedPredicted = requireAllPoints(predicted, "predicted");
  const normalizedAdjusted = requireAllPoints(adjusted, "adjusted");
  const protectedEval = videoPath.includes("/eval_clips/") || videoPath.startsWith("eval_clips/");
  const isHumanReviewed = reviewStatus === "human_reviewed";
  const points = Object.fromEntries(
    PICKLEBALL_COURT_REVIEW_POINTS.map((name) => {
      const predictedPoint = normalizedPredicted[name];
      const adjustedPoint = normalizedAdjusted[name];
      return [
        name,
        {
          predicted_xy: predictedPoint.xy,
          adjusted_xy: adjustedPoint.xy,
          confidence: predictedPoint.confidence,
          manual_moved: distance(predictedPoint.xy, adjustedPoint.xy) > 0.75,
        },
      ];
    }),
  ) as ReviewedCourtCorrection["points"];
  return {
    schema_version: 1,
    artifact_type: "racketsport_reviewed_court_calibration",
    review_status: reviewStatus,
    source_video: { id: videoId, path: videoPath, sha256: videoSha256 },
    frame: { index: frameIndex, time_s: frameTimeSeconds, image_size: imageSize },
    auto_prediction: {
      source: autoPredictionSource,
      verified: false,
      not_cal3_verified: true,
      points: normalizedPredicted,
    },
    points,
    validation: validateCourtReviewPoints({ predicted: normalizedPredicted, adjusted: normalizedAdjusted, imageSize }),
    pipeline: {
      derived_artifact: "court_calibration.json",
      handoff: "--court-calibration",
      trust: isHumanReviewed ? "reviewed_manual_court_layout" : "auto_predicted_unreviewed_court_layout",
    },
    training: {
      usable_for_court_detector_training: isHumanReviewed && !protectedEval,
      training_policy: isHumanReviewed ? "human_reviewed_not_eval_promoted" : "auto_prediction_not_training_ready",
      protected_eval_clip: protectedEval,
    },
    created_at: createdAt,
  };
}

export function buildCourtCornersPayload({
  adjusted,
  imageSize,
  frameIndex,
  source,
  reviewStatus,
}: {
  adjusted: CourtReviewPointMap;
  imageSize: [number, number];
  frameIndex: number;
  source: string;
  reviewStatus: CourtReviewStatus;
}): CourtCornersPayload {
  const nearLeft = requirePoint(adjusted, "near_left_corner");
  const nearRight = requirePoint(adjusted, "near_right_corner");
  const farRight = requirePoint(adjusted, "far_right_corner");
  const farLeft = requirePoint(adjusted, "far_left_corner");
  return {
    annotation: {
      items: [
        {
          court_corners: {
            near_left: nearLeft.xy,
            near_right: nearRight.xy,
            far_right: farRight.xy,
            far_left: farLeft.xy,
          },
          frame: `frame_${Math.max(0, Math.trunc(frameIndex)).toString().padStart(6, "0")}.jpg`,
          image_size: imageSize,
          source,
          status: reviewStatus === "human_reviewed" ? "corrected_unverified" : "auto_preview_unverified",
          not_cal3_verified: true,
          review_status: reviewStatus,
        },
      ],
    },
  };
}

function requireAllPoints(points: CourtReviewPointMap, label: string): Record<CourtReviewPointName, CourtReviewPoint> {
  const missing = PICKLEBALL_COURT_REVIEW_POINTS.filter((name) => !points[name]);
  if (missing.length) throw new Error(`missing ${label} court point(s): ${missing.join(", ")}`);
  return Object.fromEntries(
    PICKLEBALL_COURT_REVIEW_POINTS.map((name) => {
      const point = points[name] as CourtReviewPoint;
      return [name, { xy: [Number(point.xy[0]), Number(point.xy[1])] as [number, number], confidence: Number(point.confidence) }];
    }),
  ) as Record<CourtReviewPointName, CourtReviewPoint>;
}

function requirePoint(points: CourtReviewPointMap, name: CourtReviewPointName): CourtReviewPoint {
  const point = points[name];
  if (!point) throw new Error(`missing court corner point: ${name}`);
  return { xy: [Number(point.xy[0]), Number(point.xy[1])], confidence: Number(point.confidence) };
}

function courtGeometryWarning(adjusted: CourtReviewPointMap, imageSize: [number, number]): CourtReviewWarning | null {
  const nearLeft = adjusted.near_left_corner?.xy;
  const nearRight = adjusted.near_right_corner?.xy;
  const farRight = adjusted.far_right_corner?.xy;
  const farLeft = adjusted.far_left_corner?.xy;
  if (!nearLeft || !nearRight || !farRight || !farLeft) return null;
  const area = Math.abs(polygonArea([nearLeft, nearRight, farRight, farLeft]));
  const nearMidY = (nearLeft[1] + nearRight[1]) / 2;
  const farMidY = (farLeft[1] + farRight[1]) / 2;
  const swapped = nearLeft[0] >= nearRight[0] || farLeft[0] >= farRight[0];
  const farBelowNear = farMidY >= nearMidY;
  if (area < imageSize[0] * imageSize[1] * 0.025 || swapped || farBelowNear) {
    return {
      code: "bad_geometry",
      severity: "warning",
      points: ["near_left_corner", "near_right_corner", "far_right_corner", "far_left_corner"],
      message: "Court corners do not form a plausible full-court quadrilateral.",
    };
  }
  return null;
}

function polygonArea(points: Array<[number, number]>): number {
  let total = 0;
  for (let index = 0; index < points.length; index += 1) {
    const point = points[index];
    const next = points[(index + 1) % points.length];
    total += point[0] * next[1] - next[0] * point[1];
  }
  return total / 2;
}

function distance(a: [number, number], b: [number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}
