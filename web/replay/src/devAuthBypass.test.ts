import { describe, expect, it } from "vitest";

import { isReplayVerifyDevBypassAllowed, shouldShowManifestDevHint } from "./devAuthBypass";

describe("isReplayVerifyDevBypassAllowed", () => {
  it("defaults off without the explicit verifier env flag", () => {
    expect(
      isReplayVerifyDevBypassAllowed({
        flag: undefined,
        hostname: "127.0.0.1",
        mode: "development",
        prod: false,
      }),
    ).toBe(false);
  });

  it("only allows localhost loopback origins", () => {
    expect(
      isReplayVerifyDevBypassAllowed({
        flag: "1",
        hostname: "127.0.0.1",
        mode: "development",
        prod: false,
      }),
    ).toBe(true);
    expect(
      isReplayVerifyDevBypassAllowed({
        flag: "1",
        hostname: "pickleball.example.com",
        mode: "development",
        prod: false,
      }),
    ).toBe(false);
  });

  it("refuses production mode even when the flag and localhost are present", () => {
    expect(
      isReplayVerifyDevBypassAllowed({
        flag: "1",
        hostname: "localhost",
        mode: "production",
        prod: true,
      }),
    ).toBe(false);
  });
});

describe("manifest development hint", () => {
  it("appears only for a loopback development manifest deep link without the bypass", () => {
    expect(shouldShowManifestDevHint({ hasManifestParam: true, hostname: "127.0.0.1", mode: "development", prod: false })).toBe(true);
    expect(shouldShowManifestDevHint({ hasManifestParam: false, hostname: "127.0.0.1", mode: "development", prod: false })).toBe(false);
    expect(shouldShowManifestDevHint({ hasManifestParam: true, hostname: "replay.example.com", mode: "development", prod: false })).toBe(false);
    expect(shouldShowManifestDevHint({ hasManifestParam: true, hostname: "127.0.0.1", mode: "production", prod: true })).toBe(false);
    expect(shouldShowManifestDevHint({ hasManifestParam: true, flag: "1", hostname: "127.0.0.1", mode: "development", prod: false })).toBe(false);
  });
});
