import { describe, expect, it } from "vitest";

import { manifestUrlFromSearch } from "./App";

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
