import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { alignedJoint, MeshComparePage, meshRoot } from "./MeshComparePage";
import type { ActiveBodyMeshFrame } from "../viewerData";

function activeFrame(): ActiveBodyMeshFrame {
  return {
    playerId: 1,
    meshPlayerId: 1,
    presenceOpacity: 1,
    renderTranslation: [4, 2, 0],
    frame: {
      frame_idx: 1,
      t: 1,
      source_window_index: 0,
      blend_weight: 1,
      joints_world: [
        [-0.2, 0, 1],
        [0.2, 0, 1],
        [0.7, 0, 1.25],
      ],
      joint_conf: [1, 1, 1],
      mesh_vertices_world: [
        [-0.4, 0, 0],
        [0.4, 0, 0],
        [0, 0, 1.8],
      ],
      mesh_faces: [[0, 1, 2]],
      smplx_params: {},
      reasons: [],
      mesh_interpolated: false,
      interpolation: null,
    },
  };
}

describe("MeshComparePage", () => {
  it("fails closed on an incomplete link without exposing the legacy replay interface", () => {
    const markup = renderToStaticMarkup(<MeshComparePage search="?comparison=/compare.json" hostname="127.0.0.1" />);

    expect(markup).toContain("REAL MESH REQUIRED");
    expect(markup).toContain("Comparison unavailable");
    expect(markup).not.toContain("Base video");
    expect(markup).not.toContain("Court evidence");
    expect(markup).not.toContain("Replay layers");
  });

  it("centers comparison geometry on the actual mesh hip root", () => {
    const frame = activeFrame();
    expect(meshRoot(frame, ["left_hip", "right_hip", "right_wrist"])).toEqual([0, 0, 1]);
  });

  it("aligns a cue joint in body-local space with scale, yaw, and mirroring", () => {
    const frame = activeFrame();
    const point = alignedJoint(
      frame,
      ["left_hip", "right_hip", "right_wrist"],
      "right_wrist",
      [0, 0, 1],
      90,
      2,
      false,
    );

    expect(point?.[0]).toBeCloseTo(0);
    expect(point?.[1]).toBeCloseTo(1.4);
    expect(point?.[2]).toBeCloseTo(0.5);
  });

  it("contains no skeleton or legacy-player renderer", () => {
    const source = readFileSync(resolve(__dirname, "MeshComparePage.tsx"), "utf8");

    expect(source).not.toContain("<SkeletonGraph");
    expect(source).not.toContain("<Players");
    expect(source).not.toContain("CoachComparisonWorkspace");
  });
});
