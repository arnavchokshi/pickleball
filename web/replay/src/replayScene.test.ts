import { describe, expect, it } from "vitest";

import { activeReplayPointForTime, parseReplayScene, resolveReplaySceneAssetUrl } from "./replayScene";

const validScene = {
  schema_version: 1,
  world_frame: "court_Z0",
  fps: 60.0,
  court_glb: "court.glb",
  players: [1, 2],
  points: [
    { id: 1, t0: 0.0, t1: 3.5, glb_url: "points/point_1.glb", size_mb: 2.5 },
    { id: 2, t0: 3.5, t1: 7.0, glb_url: "points/point_2.glb", size_mb: 1.25 },
  ],
};

describe("parseReplayScene", () => {
  it("accepts the threed.racketsport.schemas ReplayScene shape", () => {
    expect(parseReplayScene(validScene)).toEqual(validScene);
  });

  it("rejects missing and unknown ReplayScene fields", () => {
    const missingCourtGlb = { ...validScene };
    delete (missingCourtGlb as Partial<typeof validScene>).court_glb;

    expect(() => parseReplayScene(missingCourtGlb)).toThrow("court_glb must be a string");
    expect(() => parseReplayScene({ ...validScene, unexpected: true })).toThrow(
      "unexpected field: unexpected",
    );
  });

  it("rejects malformed ReplayPoint entries", () => {
    expect(() =>
      parseReplayScene({
        ...validScene,
        points: [{ id: 1, t0: 0, t1: 1, glb_url: "points/point_1.glb" }],
      }),
    ).toThrow("points[0].size_mb must be a number");
  });

  it("rejects impossible or overlapping ReplayPoint time ranges", () => {
    expect(() =>
      parseReplayScene({
        ...validScene,
        points: [{ id: 1, t0: 2, t1: 1, glb_url: "points/point_1.glb", size_mb: 0.5 }],
      }),
    ).toThrow("points[0].t1 must be greater than points[0].t0");

    expect(() =>
      parseReplayScene({
        ...validScene,
        points: [
          { id: 1, t0: 0, t1: 2, glb_url: "points/point_1.glb", size_mb: 0.5 },
          { id: 2, t0: 1.5, t1: 3, glb_url: "points/point_2.glb", size_mb: 0.5 },
        ],
      }),
    ).toThrow("points[1].t0 must be greater than or equal to previous point t1");
  });

  it("selects active review GLB points by video time", () => {
    const scene = parseReplayScene(validScene);

    expect(activeReplayPointForTime(scene, 0)?.id).toBe(1);
    expect(activeReplayPointForTime(scene, 3.5)?.id).toBe(2);
    expect(activeReplayPointForTime(scene, 3.75)?.id).toBe(2);
    expect(activeReplayPointForTime(scene, 7)?.id).toBe(2);
    expect(activeReplayPointForTime(scene, 8)).toBeUndefined();
  });

  it("resolves replay scene assets relative to replay_scene.json", () => {
    const base = "/@fs//Users/arnavchokshi/Desktop/pickleball/runs/eval0/clip/replay_scene.json";

    expect(resolveReplaySceneAssetUrl(base, "replay_review/points/point_001_review.glb")).toBe(
      "/@fs//Users/arnavchokshi/Desktop/pickleball/runs/eval0/clip/replay_review/points/point_001_review.glb",
    );
    expect(resolveReplaySceneAssetUrl(base, "/@fs//tmp/court.glb")).toBe("/@fs//tmp/court.glb");
  });

  it("preserves remote replay scene asset origins", () => {
    const base = "https://cdn.example.com/runs/eval0/clip/replay_scene.json";

    expect(resolveReplaySceneAssetUrl(base, "replay_review/points/point_001_review.glb")).toBe(
      "https://cdn.example.com/runs/eval0/clip/replay_review/points/point_001_review.glb",
    );
  });

  it("preserves file replay scene asset origins for static headless verification", () => {
    const base = "file:///Users/arnavchokshi/Desktop/pickleball/runs/scrubber_v1_codex_20260702/burlington/replay_scene.json";

    expect(resolveReplaySceneAssetUrl(base, "body_mesh_animated_compressed.glb")).toBe(
      "file:///Users/arnavchokshi/Desktop/pickleball/runs/scrubber_v1_codex_20260702/burlington/body_mesh_animated_compressed.glb",
    );
    expect(resolveReplaySceneAssetUrl(base, "file:///tmp/court.glb")).toBe("file:///tmp/court.glb");
  });
});
