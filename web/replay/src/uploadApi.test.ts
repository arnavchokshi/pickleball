import { describe, expect, it } from "vitest";

import { apiUrl, fetchJobStatus, uploadVideoJob } from "./uploadApi";

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
