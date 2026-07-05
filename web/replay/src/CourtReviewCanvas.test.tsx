import { describe, expect, it, vi } from "vitest";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

import {
  CourtReviewCanvas,
  applyPointDrag,
  clampPointToImage,
  courtReviewLinePoints,
  courtReviewPointStatus,
  imagePointFromDisplayOffset,
} from "./CourtReviewCanvas";
import { PICKLEBALL_COURT_REVIEW_POINTS, type CourtReviewPointMap } from "./courtReview";

function samplePoints(): CourtReviewPointMap {
  const points: CourtReviewPointMap = {};
  for (const [index, name] of PICKLEBALL_COURT_REVIEW_POINTS.entries()) {
    points[name] = { xy: [100 + index * 10, 200 + index * 5], confidence: 0.7 };
  }
  return points;
}

describe("clampPointToImage", () => {
  it("keeps in-bounds points unchanged", () => {
    expect(clampPointToImage([100, 200], [1000, 600])).toEqual([100, 200]);
  });

  it("clamps out-of-bounds drags to the frame edges", () => {
    expect(clampPointToImage([-40, 900], [1000, 600])).toEqual([0, 600]);
    expect(clampPointToImage([5000, -20], [1000, 600])).toEqual([1000, 0]);
  });
});

describe("applyPointDrag", () => {
  it("updates only the dragged point's xy, preserving every other point and its confidence", () => {
    const points = samplePoints();

    const next = applyPointDrag(points, "near_left_corner", [321, 456], [1000, 600]);

    expect(next.near_left_corner?.xy).toEqual([321, 456]);
    expect(next.near_left_corner?.confidence).toBe(points.near_left_corner?.confidence);
    for (const name of PICKLEBALL_COURT_REVIEW_POINTS) {
      if (name === "near_left_corner") continue;
      expect(next[name]).toEqual(points[name]);
    }
  });

  it("clamps the dragged point to the frame bounds", () => {
    const points = samplePoints();

    const next = applyPointDrag(points, "far_right_corner", [-100, 10000], [1000, 600]);

    expect(next.far_right_corner?.xy).toEqual([0, 600]);
  });
});

describe("courtReviewLinePoints", () => {
  it("produces an SVG points string per court line that updates live as points move", () => {
    const points = samplePoints();
    const before = courtReviewLinePoints(points);
    const nearBaseline = before.find((line) => line.id === "near_baseline");
    expect(nearBaseline?.svgPoints).toBe(
      `${points.near_left_corner!.xy[0]},${points.near_left_corner!.xy[1]} ${points.near_right_corner!.xy[0]},${points.near_right_corner!.xy[1]}`,
    );

    const moved = applyPointDrag(points, "near_left_corner", [999, 1], [1000, 600]);
    const after = courtReviewLinePoints(moved);
    const nearBaselineAfter = after.find((line) => line.id === "near_baseline");

    expect(nearBaselineAfter?.svgPoints).not.toBe(nearBaseline?.svgPoints);
    expect(nearBaselineAfter?.svgPoints).toBe(`999,1 ${points.near_right_corner!.xy[0]},${points.near_right_corner!.xy[1]}`);
  });

  it("omits lines with a missing endpoint instead of drawing a broken line", () => {
    const points = samplePoints();
    delete points.net_left_sideline;

    const lines = courtReviewLinePoints(points);

    expect(lines.find((line) => line.id === "net")).toBeUndefined();
  });
});

describe("imagePointFromDisplayOffset", () => {
  it("scales a CSS-space pointer offset into image-pixel space", () => {
    expect(imagePointFromDisplayOffset([320, 180], [640, 360], [1280, 720])).toEqual([640, 360]);
  });

  it("clamps the scaled result to the image bounds", () => {
    expect(imagePointFromDisplayOffset([-10, 400], [640, 360], [1280, 720])).toEqual([0, 720]);
  });
});

describe("courtReviewPointStatus", () => {
  it("flags points listed in needs_user_input and leaves the rest ok", () => {
    expect(courtReviewPointStatus("near_left_corner", ["near_left_corner", "net_center"])).toBe("needs_review");
    expect(courtReviewPointStatus("far_left_corner", ["near_left_corner", "net_center"])).toBe("ok");
    expect(courtReviewPointStatus("far_left_corner", undefined)).toBe("ok");
  });
});

describe("CourtReviewCanvas markup", () => {
  it("renders the preview frame, all 15 keypoints, court-line polylines, and the three review actions", () => {
    const markup = renderToStaticMarkup(
      <CourtReviewCanvas
        imageUrl="/api/court/predict/pred_1/frame"
        imageSize={[1000, 600]}
        points={samplePoints()}
        needsUserInput={["near_left_corner"]}
        onPointsChange={() => {}}
        onConfirm={() => {}}
        onRepredict={() => {}}
        onSkip={() => {}}
      />,
    );

    expect(markup).toContain('src="/api/court/predict/pred_1/frame"');
    expect(markup).toContain("Confirm court");
    expect(markup).toContain("Re-predict");
    expect(markup).toContain("Skip (no court)");
    for (const name of PICKLEBALL_COURT_REVIEW_POINTS) {
      expect(markup).toContain(`data-point-name="${name}"`);
    }
    expect(markup).toContain("court-review-point-needs_review");
    expect(markup).toContain("<polyline");
  });

  it("disables the review actions while a submission is in flight", () => {
    const onConfirm = vi.fn();
    const markup = renderToStaticMarkup(
      <CourtReviewCanvas
        imageUrl={null}
        imageSize={[1000, 600]}
        points={samplePoints()}
        onPointsChange={() => {}}
        onConfirm={onConfirm}
        onRepredict={() => {}}
        onSkip={() => {}}
        disabled
      />,
    );

    expect(markup).toContain('<button type="button" class="court-review-confirm" disabled=""');
  });
});
