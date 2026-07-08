import { describe, expect, it } from "vitest";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

import {
  clipStatusLabel,
  combinedClipStatus,
  fetchClips,
  LibraryScreen,
  manifestUrlForJob,
  pollJobUntilSettled,
  uploadClipAndQueueJob,
  uploadPartsToS3,
  type ClipRecord,
} from "./LibraryScreen";
import { planParts } from "./uploadPlan";
import type { UploadJob } from "./uploadApi";

function jsonResponse(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

describe("LibraryScreen markup (idle state)", () => {
  it("renders the upload control and library heading before the clip list loads", () => {
    const markup = renderToStaticMarkup(
      <LibraryScreen onOpenViewer={() => {}} onLogout={() => {}} />,
    );

    expect(markup).toContain("Library");
    expect(markup).toContain("Upload a clip");
    expect(markup).toContain("Log out");
    // useEffect never runs under renderToStaticMarkup (no jsdom in this
    // repo's vitest setup), so the initial static snapshot is the
    // pre-fetch loading state -- matches the existing UploadPanel.test.tsx
    // convention of only asserting the idle/static form markup.
    expect(markup).toContain("Loading clips");
  });
});

describe("fetchClips", () => {
  it("GETs /api/clips with the bearer token and returns the clip array", async () => {
    let sawAuth: string | null = null;
    const fetchImpl: typeof fetch = async (url, init) => {
      sawAuth = new Headers(init?.headers).get("Authorization");
      expect(String(url)).toBe("/api/clips");
      return jsonResponse({
        clips: [{ id: "clip_1", filename: "drill.mp4", status: "uploaded", size_bytes: 100, key: "raw/u/clip_1/drill.mp4" }],
      });
    };

    const { setAccessToken } = await import("./authApi");
    setAccessToken("tok_abc");
    const clips = await fetchClips({ fetchImpl });
    setAccessToken(null);

    expect(sawAuth).toBe("Bearer tok_abc");
    expect(clips).toEqual([
      { id: "clip_1", filename: "drill.mp4", status: "uploaded", size_bytes: 100, key: "raw/u/clip_1/drill.mp4" },
    ]);
  });
});

describe("uploadPartsToS3", () => {
  it("PUTs each part to its presigned URL (plain fetch, not authedFetch) and captures ETag", async () => {
    const file = new File(["x".repeat(25)], "drill.mp4", { type: "video/mp4" });
    const plan = planParts(file.size, 10);
    const putCalls: Array<{ url: string; bodyLength: number }> = [];
    const fetchImpl: typeof fetch = async (url, init) => {
      const body = init?.body as Blob;
      putCalls.push({ url: String(url), bodyLength: body.size });
      return new Response(null, { status: 200, headers: { ETag: `"etag-${String(url).slice(-1)}"` } });
    };

    const parts = await uploadPartsToS3(
      file,
      plan,
      [
        { part_number: 1, url: "https://s3.example.com/bucket/key?part=1" },
        { part_number: 2, url: "https://s3.example.com/bucket/key?part=2" },
        { part_number: 3, url: "https://s3.example.com/bucket/key?part=3" },
      ],
      fetchImpl,
    );

    expect(putCalls.map((c) => c.bodyLength)).toEqual([10, 10, 5]);
    expect(parts).toEqual([
      { part_number: 1, etag: '"etag-1"' },
      { part_number: 2, etag: '"etag-2"' },
      { part_number: 3, etag: '"etag-3"' },
    ]);
  });

  it("throws a CORS-hint error when a part response is missing the ETag header", async () => {
    const file = new File(["short"], "drill.mp4", { type: "video/mp4" });
    const plan = planParts(file.size, 10);
    const fetchImpl: typeof fetch = async () => new Response(null, { status: 200 });

    await expect(
      uploadPartsToS3(file, plan, [{ part_number: 1, url: "https://s3.example.com/bucket/key" }], fetchImpl),
    ).rejects.toThrow("ExposeHeaders");
  });

  it("throws when a part upload response is not ok", async () => {
    const file = new File(["short"], "drill.mp4", { type: "video/mp4" });
    const plan = planParts(file.size, 10);
    const fetchImpl: typeof fetch = async () => new Response(null, { status: 403 });

    await expect(
      uploadPartsToS3(file, plan, [{ part_number: 1, url: "https://s3.example.com/bucket/key" }], fetchImpl),
    ).rejects.toThrow("status 403");
  });
});

describe("uploadClipAndQueueJob", () => {
  it("drives POST /api/clips -> PUT part(s) to S3 -> complete -> POST /api/jobs -> poll GET /api/jobs/{id}, in order", async () => {
    const calls: string[] = [];
    let pollCount = 0;
    const fetchImpl: typeof fetch = async (url, init) => {
      const method = (init?.method ?? "GET").toUpperCase();
      const urlStr = String(url);
      calls.push(`${method} ${urlStr}`);

      if (urlStr === "/api/clips" && method === "POST") {
        return jsonResponse(
          {
            id: "clip_1",
            filename: "drill.mp4",
            key: "raw/u/clip_1/drill.mp4",
            upload_id: "upload_1",
            part_count: 1,
            part_urls: [{ part_number: 1, url: "https://s3.example.com/bucket/key?part=1" }],
            sidecar_upload_url: "https://s3.example.com/bucket/sidecar",
          },
          201,
        );
      }
      if (urlStr === "https://s3.example.com/bucket/key?part=1" && method === "PUT") {
        return new Response(null, { status: 200, headers: { ETag: '"abc123"' } });
      }
      if (urlStr === "/api/clips/clip_1/complete" && method === "POST") {
        return jsonResponse({ id: "clip_1", status: "uploaded", key: "raw/u/clip_1/drill.mp4" });
      }
      if (urlStr === "/api/jobs" && method === "POST") {
        return jsonResponse(
          {
            id: "job_1",
            clip: "drill",
            status: "queued",
            links: { status: "/api/jobs/job_1", manifest: "/api/jobs/job_1/manifest" },
          },
          202,
        );
      }
      if (urlStr === "/api/jobs/job_1" && method === "GET") {
        pollCount += 1;
        if (pollCount === 1) {
          return jsonResponse({
            id: "job_1",
            status: "running",
            links: { status: "/api/jobs/job_1", manifest: "/api/jobs/job_1/manifest" },
          });
        }
        return jsonResponse({
          id: "job_1",
          status: "complete",
          links: { status: "/api/jobs/job_1", manifest: "/api/jobs/job_1/manifest" },
          result: { manifest_url: "/api/jobs/job_1/manifest" },
        });
      }
      throw new Error(`unexpected fetch: ${method} ${urlStr}`);
    };

    const file = new File(["x".repeat(20)], "drill.mp4", { type: "video/mp4" });
    const stages: string[] = [];
    const { clip, job } = await uploadClipAndQueueJob(file, { fetchImpl }, 10 * 1024 * 1024, 0, {
      onStage: (stage) => stages.push(stage),
    });

    expect(clip.id).toBe("clip_1");
    expect(job.status).toBe("complete");
    expect(calls).toEqual([
      "POST /api/clips",
      "PUT https://s3.example.com/bucket/key?part=1",
      "POST /api/clips/clip_1/complete",
      "POST /api/jobs",
      "GET /api/jobs/job_1",
      "GET /api/jobs/job_1",
    ]);
    expect(stages).toEqual(["creating_clip", "uploading_parts", "completing_clip", "creating_job", "polling_job"]);
  });
});

describe("pollJobUntilSettled", () => {
  it("stops as soon as a terminal status is returned", async () => {
    let calls = 0;
    const fetchImpl: typeof fetch = async () => {
      calls += 1;
      return jsonResponse({ id: "job_1", status: "failed", error: "boom", links: { status: "/api/jobs/job_1" } });
    };

    const job = await pollJobUntilSettled("/api/jobs/job_1", { fetchImpl }, 0);

    expect(job.status).toBe("failed");
    expect(calls).toBe(1);
  });

  it("gives up after maxAttempts instead of polling forever", async () => {
    const fetchImpl: typeof fetch = async () =>
      jsonResponse({ id: "job_1", status: "running", links: { status: "/api/jobs/job_1" } });

    await expect(pollJobUntilSettled("/api/jobs/job_1", { fetchImpl }, 0, undefined, 3)).rejects.toThrow(
      "did not settle after 3 polls",
    );
  });
});

describe("combinedClipStatus / clipStatusLabel", () => {
  it("prefers the live job status over the clip's own status", () => {
    const clip: ClipRecord = { id: "c1", filename: "a.mp4", status: "uploaded", size_bytes: 1, key: "k" };
    const job: UploadJob = { id: "j1", status: "running", links: { status: "/api/jobs/j1" } };

    expect(combinedClipStatus(clip, job)).toBe("running");
    expect(combinedClipStatus(clip, null)).toBe("uploaded");
    expect(combinedClipStatus({ ...clip, status: "uploading" })).toBe("uploading");
    expect(clipStatusLabel("running")).toBe("Processing on GPU");
    expect(clipStatusLabel("complete")).toBe("Ready");
  });
});

describe("manifestUrlForJob", () => {
  it("prefers result.manifest_url, falls back to links.manifest, and applies the API base", () => {
    const jobWithResult: UploadJob = {
      id: "j1",
      status: "complete",
      links: { status: "/api/jobs/j1", manifest: "/api/jobs/j1/manifest" },
      result: { manifest_url: "/api/jobs/j1/manifest_v2" },
    };
    expect(manifestUrlForJob(jobWithResult, "https://pb.example.com")).toBe(
      "https://pb.example.com/api/jobs/j1/manifest_v2",
    );

    const jobWithoutResult: UploadJob = {
      id: "j1",
      status: "complete",
      links: { status: "/api/jobs/j1", manifest: "/api/jobs/j1/manifest" },
    };
    expect(manifestUrlForJob(jobWithoutResult, "https://pb.example.com")).toBe(
      "https://pb.example.com/api/jobs/j1/manifest",
    );

    expect(manifestUrlForJob(undefined)).toBeNull();
  });
});
