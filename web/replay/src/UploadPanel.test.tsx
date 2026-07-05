import { describe, expect, it } from "vitest";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

import {
  ResourceUsagePanel,
  UploadPanel,
  assistSeedFromPrediction,
  jobStatusText,
  pipelineProgressLabel,
  submitCourtReview,
  uploadErrorText,
} from "./UploadPanel";
import { PICKLEBALL_COURT_REVIEW_POINTS, type CourtReviewPointMap } from "./courtReview";
import type { CourtPrediction } from "./uploadApi";

function fakePrediction(overrides: Partial<CourtPrediction> = {}): CourtPrediction {
  const points: CourtReviewPointMap = {};
  for (const [index, name] of PICKLEBALL_COURT_REVIEW_POINTS.entries()) {
    points[name] = { xy: [100 + index * 10, 200 + index * 5], confidence: 0.8 };
  }
  return {
    schema_version: 1,
    artifact_type: "racketsport_court_layout_prediction",
    clip: "drill_01",
    image_size: [1000, 600],
    frame_index: 4,
    frame_time_s: 0.13,
    prediction_source: "template_projection_seed:ffprobe_metadata",
    verified: false,
    not_cal3_verified: true,
    points,
    needs_user_input: [],
    assist: { mode: "none", tap_points: [], line_label: null },
    proposal_report: null,
    preview_frame_url: null,
    video: {
      id: "drill_01",
      filename: "drill.mp4",
      path: "/tmp/drill.mp4",
      sha256: "a".repeat(64),
      size_bytes: 42,
    },
    ...overrides,
  };
}

function fakeSequentialFetch(responses: { reviewStatus?: string } = {}) {
  const postedReviews: unknown[] = [];
  const postedJobForms: FormData[] = [];
  const fetchImpl: typeof fetch = async (url, init) => {
    const target = String(url);
    if (target.endsWith("/api/court/reviews")) {
      const body = JSON.parse(String(init?.body));
      postedReviews.push(body);
      return new Response(
        JSON.stringify({
          review: { artifact_type: "racketsport_reviewed_court_calibration", review_status: body.review_status },
          court_calibration: { schema_version: 1 },
          saved: { review_path: "/tmp/reviewed_court_calibration.json", court_calibration_path: "/tmp/court_calibration.json", index_path: "/tmp/index.json" },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      );
    }
    if (target.endsWith("/api/jobs")) {
      postedJobForms.push(init?.body as FormData);
      return new Response(JSON.stringify({ id: "job_1", status: "queued", links: { status: "/api/jobs/job_1" } }), {
        status: 202,
        headers: { "content-type": "application/json" },
      });
    }
    throw new Error(`unexpected fetch to ${target}`);
  };
  return { fetchImpl, postedReviews, postedJobForms };
}

describe("jobStatusText", () => {
  it("maps server job states to concise user-facing labels", () => {
    expect(jobStatusText(null)).toBe("No upload queued");
    expect(jobStatusText({ id: "job_1", status: "queued", links: { status: "/api/jobs/job_1" } })).toBe("Queued");
    expect(jobStatusText({ id: "job_1", status: "running", links: { status: "/api/jobs/job_1" } })).toBe("Processing on GPU");
    expect(jobStatusText({ id: "job_1", status: "complete", links: { status: "/api/jobs/job_1" } })).toBe("Replay ready");
    expect(jobStatusText({ id: "job_1", status: "failed", error: "boom", links: { status: "/api/jobs/job_1" } })).toBe("Failed");
  });
});

describe("pipelineProgressLabel", () => {
  it("uses server progress stage and ETA when present", () => {
    expect(
      pipelineProgressLabel({
        id: "job_1",
        status: "running",
        progress: {
          percent: 42,
          stage: "Running pipeline on GPU",
          message: "Tracking and body stages are active.",
          eta_seconds: 118,
        },
        links: { status: "/api/jobs/job_1" },
      }),
    ).toBe("Running pipeline on GPU · about 2 min left");
  });
});

describe("uploadErrorText", () => {
  it("does not render raw process_video JSON blobs for trusted-calibration failures", () => {
    const text = uploadErrorText(
      'RuntimeError: local process_video failed (1): {"stages":[{"stage":"calibration","notes":["intrinsics.source=estimated_from_reviewed_court_calibration is not a trusted external calibration source"]}]}',
    );

    expect(text).toBe(
      "Pipeline rejected an untrusted court calibration. The court prediction was saved as an unverified preview seed, not a trusted calibration.",
    );
  });
});

describe("UploadPanel", () => {
  it("renders the simple intake flow as video plus one primary Predict Court action", () => {
    const markup = renderToStaticMarkup(<UploadPanel />);

    expect(markup).toContain("Video");
    expect(markup).toContain("Predict Court");
    expect(markup).toContain("Pipeline progress");
    expect(markup).not.toContain("Upload and process");
    expect(markup).not.toContain("Capture sidecar");
    expect(markup).not.toContain("Court calibration");
  });
});

describe("ResourceUsagePanel", () => {
  it("summarizes GPU, VRAM, CPU, and stage timing in the completed job view", () => {
    const markup = renderToStaticMarkup(
      <ResourceUsagePanel
        resourceSummary={{
          gpu_utilization_avg_pct: 55.5,
          gpu_utilization_max_pct: 91,
          gpu_memory_used_max_mb: 12345,
          gpu_memory_total_mb: 24576,
          cpu_utilization_avg_pct: 38.4,
          sample_count: 18,
          duration_s: 96.2,
        }}
        stageSummary={[
          { stage: "ingest", wall_seconds: 1.25 },
          { stage: "body", wall_seconds: 9.5 },
        ]}
      />,
    );

    expect(markup).toContain("GPU utilization");
    expect(markup).toContain("VRAM peak");
    expect(markup).toContain("CPU avg");
    expect(markup).toContain("body");
    expect(markup).toContain("9.5s");
  });
});

describe("submitCourtReview", () => {
  it("confirmed: uploads court_corners (trusted channel) with review_status human_reviewed, and no assist seed", async () => {
    const { fetchImpl, postedReviews, postedJobForms } = fakeSequentialFetch();
    const video = new File(["video"], "drill.mp4", { type: "video/mp4" });
    const prediction = fakePrediction();

    const job = await submitCourtReview(
      { decision: "confirmed", video, prediction, adjustedPoints: prediction.points as CourtReviewPointMap },
      { fetchImpl },
    );

    expect(job.status).toBe("queued");
    expect(postedReviews).toHaveLength(1);
    expect((postedReviews[0] as { review_status: string }).review_status).toBe("human_reviewed");

    const form = postedJobForms[0];
    expect((form.get("court_corners") as File | null)?.name).toBe("court_corners.json");
    expect((form.get("court_review") as File | null)?.name).toBe("reviewed_court_calibration.json");
    expect(form.get("court_assist_seed")).toBeNull();
  });

  it("skipped: uploads ONLY court_review (auto_predicted_unreviewed) plus the assist seed - never court_corners", async () => {
    const { fetchImpl, postedReviews, postedJobForms } = fakeSequentialFetch();
    const video = new File(["video"], "drill.mp4", { type: "video/mp4" });
    const prediction = fakePrediction();

    const job = await submitCourtReview(
      { decision: "skipped", video, prediction, adjustedPoints: prediction.points as CourtReviewPointMap },
      { fetchImpl },
    );

    expect(job.status).toBe("queued");
    expect((postedReviews[0] as { review_status: string }).review_status).toBe("auto_predicted_unreviewed");

    const form = postedJobForms[0];
    expect(form.get("court_corners")).toBeNull();
    expect((form.get("court_review") as File | null)?.name).toBe("reviewed_court_calibration.json");
    expect((form.get("court_assist_seed") as File | null)?.name).toBe("court_assist_seed.json");
  });

  it("carries the report's real assist seed through to the uploaded court_assist_seed file when skipped", async () => {
    const { fetchImpl, postedJobForms } = fakeSequentialFetch();
    const video = new File(["video"], "drill.mp4", { type: "video/mp4" });
    const prediction = fakePrediction({
      proposal_report: {
        artifact_type: "racketsport_court_proposals",
        schema_version: 1,
        status: "ranked_not_verified",
        verified: false,
        not_cal3_verified: true,
        ranking: { selected_proposal_id: null, abstain: true, abstain_reasons: ["not_cal3_verified"] },
        assist: { mode: "one_inside_tap", tap_points: [[500, 300]], line_label: null },
        proposals: [],
      },
    });

    await submitCourtReview(
      { decision: "skipped", video, prediction, adjustedPoints: prediction.points as CourtReviewPointMap },
      { fetchImpl },
    );

    const assistFile = postedJobForms[0].get("court_assist_seed") as File;
    const assistPayload = JSON.parse(await assistFile.text());
    expect(assistPayload).toEqual({ mode: "one_inside_tap", tap_points: [[500, 300]], line_label: null, trusted_calibration: false });
  });
});

describe("assistSeedFromPrediction", () => {
  it("defaults to no-assist when the prediction carries no proposal report (template/detector fallback)", () => {
    expect(assistSeedFromPrediction(fakePrediction())).toEqual({
      mode: "none",
      tapPoints: [],
      lineLabel: null,
      trustedCalibration: false,
    });
  });

  it("falls back to no-assist rather than throwing when the proposal report fails validation", () => {
    const prediction = fakePrediction({
      proposal_report: { artifact_type: "not_a_court_proposal_report" },
    });

    expect(assistSeedFromPrediction(prediction)).toEqual({
      mode: "none",
      tapPoints: [],
      lineLabel: null,
      trustedCalibration: false,
    });
  });
});
