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
});
