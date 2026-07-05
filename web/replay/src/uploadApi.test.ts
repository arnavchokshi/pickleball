import { describe, expect, it } from "vitest";

import { apiUrl, fetchJobStatus, formatEta, jobProgressPercent, saveCourtReview, uploadVideoJob } from "./uploadApi";

describe("apiUrl", () => {
  it("uses same-origin API paths by default", () => {
    expect(apiUrl("/api/jobs", "")).toBe("/api/jobs");
  });

  it("joins configured API bases without double slashes", () => {
    expect(apiUrl("/api/jobs", "https://pb.example.com/")).toBe("https://pb.example.com/api/jobs");
  });
});

describe("uploadVideoJob", () => {
  it("posts video and optional sidecar fields to the jobs endpoint", async () => {
    let postedUrl = "";
    let postedBody: FormData | null = null;
    const fetchImpl: typeof fetch = async (url, init) => {
      postedUrl = String(url);
      postedBody = init?.body as FormData;
      return new Response(JSON.stringify({ id: "job_1", status: "queued", links: { status: "/api/jobs/job_1" } }), {
        status: 202,
        headers: { "content-type": "application/json" },
      });
    };

    const job = await uploadVideoJob(
      {
        video: new File(["video"], "drill.mp4", { type: "video/mp4" }),
        captureSidecar: new File(["{}"], "capture_sidecar.json", { type: "application/json" }),
        courtCorners: new File(["{}"], "court_corners.json", { type: "application/json" }),
        courtReview: new File(["{}"], "reviewed_court_calibration.json", { type: "application/json" }),
        courtCalibration: new File(["{}"], "court_calibration.json", { type: "application/json" }),
        courtAssistSeed: new File(["{}"], "court_assist_seed.json", { type: "application/json" }),
        clip: "drill_01",
        maxFrames: 8,
      },
      { baseUrl: "https://pb.example.com", fetchImpl },
    );

    expect(job.status).toBe("queued");
    expect(postedUrl).toBe("https://pb.example.com/api/jobs");
    expect(postedBody).not.toBeNull();
    const form = postedBody as unknown as FormData;
    expect(form.get("clip")).toBe("drill_01");
    expect(form.get("max_frames")).toBe("8");
    expect((form.get("video") as File).name).toBe("drill.mp4");
    expect((form.get("capture_sidecar") as File).name).toBe("capture_sidecar.json");
    expect((form.get("court_corners") as File).name).toBe("court_corners.json");
    expect((form.get("court_review") as File).name).toBe("reviewed_court_calibration.json");
    expect((form.get("court_calibration") as File).name).toBe("court_calibration.json");
    expect((form.get("court_assist_seed") as File).name).toBe("court_assist_seed.json");
  });

  it("omits the court_assist_seed field when none is provided", async () => {
    let postedBody: FormData | null = null;
    const fetchImpl: typeof fetch = async (_url, init) => {
      postedBody = init?.body as FormData;
      return new Response(JSON.stringify({ id: "job_1", status: "queued", links: { status: "/api/jobs/job_1" } }), {
        status: 202,
        headers: { "content-type": "application/json" },
      });
    };

    await uploadVideoJob(
      { video: new File(["video"], "drill.mp4", { type: "video/mp4" }) },
      { fetchImpl },
    );

    const form = postedBody as unknown as FormData;
    expect(form.get("court_assist_seed")).toBeNull();
  });

  it("raises the server message for failed uploads", async () => {
    const fetchImpl: typeof fetch = async () =>
      new Response(JSON.stringify({ detail: "unsafe slug" }), {
        status: 400,
        headers: { "content-type": "application/json" },
      });

    await expect(
      uploadVideoJob(
        { video: new File(["video"], "drill.mp4", { type: "video/mp4" }) },
        { fetchImpl },
      ),
    ).rejects.toThrow("unsafe slug");
  });

  it("explains local API 404s instead of surfacing a raw request failed message", async () => {
    const fetchImpl: typeof fetch = async () =>
      new Response("<html>not found</html>", {
        status: 404,
        headers: { "content-type": "text/html" },
      });

    await expect(
      uploadVideoJob(
        { video: new File(["video"], "drill.mp4", { type: "video/mp4" }) },
        { fetchImpl },
      ),
    ).rejects.toThrow("API server not found");
  });

  it("keeps backend JSON 404 details instead of treating them as missing local API servers", async () => {
    const fetchImpl: typeof fetch = async () =>
      new Response(JSON.stringify({ detail: "job not found" }), {
        status: 404,
        headers: { "content-type": "application/json" },
      });

    await expect(fetchJobStatus("/api/jobs/job_missing", { fetchImpl })).rejects.toThrow("job not found");
  });

  it("explains non-JSON proxy failures as an unreachable local API server", async () => {
    const fetchImpl: typeof fetch = async () =>
      new Response("proxy error", {
        status: 502,
        headers: { "content-type": "text/plain" },
      });

    await expect(
      uploadVideoJob(
        { video: new File(["video"], "drill.mp4", { type: "video/mp4" }) },
        { fetchImpl },
      ),
    ).rejects.toThrow("API server not reachable");
  });
});

describe("saveCourtReview", () => {
  it("posts a reviewed correction artifact and returns the derived calibration", async () => {
    let postedUrl = "";
    let postedJson: unknown = null;
    const fetchImpl: typeof fetch = async (url, init) => {
      postedUrl = String(url);
      postedJson = JSON.parse(String(init?.body));
      return new Response(
        JSON.stringify({
          review: { artifact_type: "racketsport_reviewed_court_calibration" },
          court_calibration: { schema_version: 1 },
          saved: { review_path: "/tmp/reviewed_court_calibration.json" },
        }),
        {
          status: 200,
          headers: { "content-type": "application/json" },
        },
      );
    };

    const response = await saveCourtReview(
      { artifact_type: "racketsport_reviewed_court_calibration", review_status: "human_reviewed" },
      { baseUrl: "https://pb.example.com", fetchImpl },
    );

    expect(postedUrl).toBe("https://pb.example.com/api/court/reviews");
    expect(postedJson).toMatchObject({ review_status: "human_reviewed" });
    expect(response.court_calibration.schema_version).toBe(1);
  });
});

describe("fetchJobStatus", () => {
  it("loads a status URL relative to the API base", async () => {
    const fetchImpl: typeof fetch = async (url) =>
      new Response(JSON.stringify({ id: "job_1", status: "complete", links: { status: "/api/jobs/job_1" } }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });

    const job = await fetchJobStatus("/api/jobs/job_1", { baseUrl: "https://pb.example.com/", fetchImpl });

    expect(job.status).toBe("complete");
  });
});

describe("job progress helpers", () => {
  it("normalizes server progress and formats ETA", () => {
    expect(
      jobProgressPercent({
        id: "job_1",
        status: "running",
        progress: { percent: 42, stage: "Running pipeline on GPU", eta_seconds: 118 },
        links: { status: "/api/jobs/job_1" },
      }),
    ).toBe(42);
    expect(
      jobProgressPercent({
        id: "job_1",
        status: "complete",
        links: { status: "/api/jobs/job_1" },
      }),
    ).toBe(100);
    expect(formatEta(118)).toBe("about 2 min");
    expect(formatEta(0)).toBe("less than 1 min");
  });
});
