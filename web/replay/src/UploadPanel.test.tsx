import { describe, expect, it } from "vitest";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { UploadPanel, jobStatusText, pipelineProgressLabel } from "./UploadPanel";

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

describe("UploadPanel", () => {
  it("renders video, sidecar, calibration, and submit controls", () => {
    const markup = renderToStaticMarkup(<UploadPanel />);

    expect(markup).toContain("Video");
    expect(markup).toContain("Capture sidecar");
    expect(markup).toContain("Court calibration");
    expect(markup).toContain("Pipeline progress");
    expect(markup).toContain("Upload and process");
  });
});
