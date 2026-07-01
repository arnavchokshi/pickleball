import { describe, expect, it } from "vitest";

import { bodyMeshOpacityFromBlendWeight, contactReadoutText, manifestUrlFromSearch } from "./App";

describe("manifestUrlFromSearch", () => {
  it("uses the manifest query parameter when present", () => {
    expect(manifestUrlFromSearch("?manifest=/@fs/tmp/replay_viewer_manifest.json")).toBe(
      "/@fs/tmp/replay_viewer_manifest.json",
    );
  });

  it("does not fall back to a checkout-specific absolute path", () => {
    expect(manifestUrlFromSearch("")).toBeNull();
  });
});

describe("bodyMeshOpacityFromBlendWeight", () => {
  it("scales solid contact mesh opacity by the body_mesh blend weight", () => {
    expect(bodyMeshOpacityFromBlendWeight({ blend_weight: 0 })).toBe(0);
    expect(bodyMeshOpacityFromBlendWeight({ blend_weight: 0.5 })).toBeCloseTo(0.34);
    expect(bodyMeshOpacityFromBlendWeight({ blend_weight: 1 })).toBe(0.68);
  });
});

describe("contactReadoutText", () => {
  it("uses active body mesh frames as contact evidence when contact_windows is absent", () => {
    expect(
      contactReadoutText(new Set(), [
        {
          playerId: 1,
          frame: {
            frame_idx: 150,
            t: 2.5,
            source_window_index: 0,
            blend_weight: 1,
            joints_world: [],
            joint_conf: [],
            mesh_vertices_world: [],
            mesh_faces: [],
            smplx_params: {},
            reasons: ["contact_window"],
          },
        },
      ]),
    ).toBe("3D contact: p1");
  });
});
