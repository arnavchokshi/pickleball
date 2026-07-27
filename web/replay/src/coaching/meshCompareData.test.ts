import { describe, expect, it } from "vitest";

import {
  isLoopbackHostname,
  meshCompareRouteFromSearch,
  referencePresentationForRights,
  validateDenseMeshCoverage,
} from "./meshCompareData";
import type { BodyMeshFaces, BodyMeshIndex } from "../viewerData";

function denseIndex({
  playerId = 1,
  t0 = 1,
  t1 = 3,
  vertexCount = 18_439,
  skipFrame = -1,
}: {
  playerId?: number;
  t0?: number;
  t1?: number;
  vertexCount?: number;
  skipFrame?: number;
} = {}): BodyMeshIndex {
  const startFrame = Math.round(t0 * 30);
  const endFrame = Math.round(t1 * 30);
  const frames = Array.from({ length: endFrame - startFrame + 1 }, (_, offset) => startFrame + offset)
    .filter((frame) => frame !== skipFrame)
    .map((frame) => ({
      frame_idx: frame,
      t: frame / 30,
      source_window_index: 0,
      blend_weight: 1,
      vertex_count: vertexCount,
      joint_count: 70,
      joint_conf: Array(70).fill(0.9),
      reasons: [],
      delta_from_previous: frame !== startFrame,
    }));
  return {
    schema_version: 1,
    artifact_type: "racketsport_body_mesh_index",
    clip: "clip",
    model: "sam3d_body_mhr70",
    fps: 30,
    world_frame: "court_Z0",
    faces_ref: "mhr70_faces",
    faces_url: "faces.json",
    windows: [
      {
        source_window_index: 0,
        frame_start: startFrame,
        frame_end: endFrame,
        t0,
        t1,
        frame_count: endFrame - startFrame + 1,
        player_frame_count: frames.length,
        target_player_ids: [playerId],
        player_ids: [playerId],
        target_representation: "world_mesh",
        fallback_representation: "none",
        reason_counts: {},
        max_score: 1,
        url: "chunk.bin.gz",
        byte_size: 1,
        encoding: "gzip_int16_delta_world_vertices_v2",
        quantization: { scale: 0.001, unit: "m" },
        players: [{ id: playerId, frames }],
      },
    ],
    summary: {
      window_count: 1,
      mesh_frame_count: frames.length,
      player_count: 1,
      faces_count: 36_874,
    },
  };
}

function denseFaces(count = 36_874): BodyMeshFaces {
  return {
    schema_version: 1,
    artifact_type: "racketsport_body_mesh_faces",
    faces_ref: "mhr70_faces",
    mesh_faces: Array.from({ length: count }, () => [0, 1, 2] as [number, number, number]),
  };
}

describe("mesh comparison routing", () => {
  it("requires an explicit mesh-comparison view plus both addressed artifacts", () => {
    expect(meshCompareRouteFromSearch("?view=mesh_compare&manifest=%2Fruns%2Fuser.json&comparison=%2Fruns%2Fcompare.json")).toEqual({
      manifestUrl: "/runs/user.json",
      comparisonUrl: "/runs/compare.json",
    });
    expect(meshCompareRouteFromSearch("?manifest=/runs/user.json&comparison=/runs/compare.json")).toBeNull();
    expect(meshCompareRouteFromSearch("?manifest=/runs/user.json")).toBeNull();
    expect(meshCompareRouteFromSearch("?comparison=/runs/compare.json")).toBeNull();
  });

  it("does not accept query-controlled native-mesh index overrides", () => {
    expect(meshCompareRouteFromSearch(
      "?view=mesh_compare&manifest=/user/replay.json&comparison=/compare.json&user_index=/wrong/user.json",
    )).toEqual({ manifestUrl: "/user/replay.json", comparisonUrl: "/compare.json" });
  });
});

describe("reference display rights", () => {
  it("uses PRO only after separate trusted clearance authorization", () => {
    expect(() => referencePresentationForRights("cleared", "example.com")).toThrow("server-issued");
    expect(referencePresentationForRights("cleared", "example.com", true)).toEqual({
      kind: "professional",
      badge: "PRO",
      displayName: null,
      publicDisplayReady: true,
    });
  });

  it("allows local-review-only motion only on loopback and never labels it PRO", () => {
    expect(isLoopbackHostname("[::1]")).toBe(true);
    expect(referencePresentationForRights("local_review_only", "127.0.0.1")).toEqual({
      kind: "local_reference_player",
      badge: "REFERENCE PLAYER",
      displayName: "Senior Pro reference — local review only",
      publicDisplayReady: false,
    });
    expect(() => referencePresentationForRights("local_review_only", "demo.example.com")).toThrow(
      "not cleared",
    );
  });
});

describe("native dense mesh gate", () => {
  it("accepts one stable dense surface with complete phase coverage", () => {
    const coverage = validateDenseMeshCoverage({
      sourceLabel: "user",
      index: denseIndex(),
      faces: denseFaces(),
      playerId: 1,
      interval: { t0: 1, t1: 3 },
      phaseTimes: [1, 1.4, 1.8, 2.2, 3],
    });

    expect(coverage.vertexCount).toBe(18_439);
    expect(coverage.faceCount).toBe(36_874);
    expect(coverage.maxGapSeconds).toBeCloseTo(1 / 30);
  });

  it("rejects low-resolution proxy geometry instead of falling back to a skeleton", () => {
    expect(() => validateDenseMeshCoverage({
      sourceLabel: "reference",
      index: denseIndex({ vertexCount: 70 }),
      faces: denseFaces(),
      playerId: 1,
      interval: { t0: 1, t1: 3 },
      phaseTimes: [1, 1.4, 1.8, 2.2, 3],
    })).toThrow("native dense mesh surface");
  });

  it("rejects an internal mesh gap across the compared motion", () => {
    const index = denseIndex({ skipFrame: 50 });
    index.windows[0].players[0].frames = index.windows[0].players[0].frames.filter(
      (frame) => frame.frame_idx < 49 || frame.frame_idx > 51,
    );
    expect(() => validateDenseMeshCoverage({
      sourceLabel: "user",
      index,
      faces: denseFaces(),
      playerId: 1,
      interval: { t0: 1, t1: 3 },
      phaseTimes: [1, 1.4, 1.8, 2.2, 3],
    })).toThrow("coverage gap");
  });

  it("rejects missing native mesh topology", () => {
    expect(() => validateDenseMeshCoverage({
      sourceLabel: "reference",
      index: denseIndex(),
      faces: denseFaces(3),
      playerId: 1,
      interval: { t0: 1, t1: 3 },
      phaseTimes: [1, 1.4, 1.8, 2.2, 3],
    })).toThrow("topology");
  });
});
