import { describe, expect, it } from "vitest";

import { parseReplayScene } from "./replayScene";

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
});
