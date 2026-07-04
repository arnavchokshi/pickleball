import { describe, expect, it } from "vitest";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { UploadPanel, jobStatusText, pipelineProgressLabel, uploadErrorText } from "./UploadPanel";

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
