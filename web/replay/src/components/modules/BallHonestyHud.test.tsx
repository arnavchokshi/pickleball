import { describe, expect, it } from "vitest";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { BallHonestyHud } from "./BallHonestyHud";
import type { BallTrailArtifact } from "./ballTrail";

function trustedArtifact(overrides: Partial<BallTrailArtifact> = {}): BallTrailArtifact {
  return {
    samples: [{ t: 0, band: "anchored_measured", conf: 0.9, visible: true, world_xyz: [0, 1, 0.3] }],
    segments: [],
    status: "ran",
    trusted: true,
    killReasons: [],
    ...overrides,
  };
}

function untrustedArtifact(overrides: Partial<BallTrailArtifact> = {}): BallTrailArtifact {
  return {
    // Mirrors the measured live defect: frames still tagged anchored_measured
    // even though the solver self-killed. The HUD must not trust this.
    samples: [{ t: 0, band: "anchored_measured", conf: 0.86, visible: true, world_xyz: [-0.37, 7.06, 0.44] }],
    segments: [],
    status: "experimental_off",
    trusted: false,
    killReasons: ["physical_sanity_violation_fraction 0.400000 exceeds 0.200000"],
    ...overrides,
  };
}

describe("BallHonestyHud fail-closed rendering", () => {
  it("exposes the honesty aria-label and a measured data-ball-state on the healthy path (pixel-identical baseline)", () => {
    const markup = renderToStaticMarkup(<BallHonestyHud arcSolved={trustedArtifact()} currentTime={0} />);

    expect(markup).toContain('aria-label="Ball tracking honesty"');
    expect(markup).toContain('data-ball-state="measured"');
    expect(markup).toContain("ball: measured");
  });

  it("shows an explicit solver_off fail-closed state naming the kill reason instead of a silent/measured HUD", () => {
    const markup = renderToStaticMarkup(<BallHonestyHud arcSolved={untrustedArtifact()} currentTime={0} />);

    expect(markup).toContain('aria-label="Ball tracking honesty"');
    expect(markup).toContain('data-ball-state="solver_off"');
    expect(markup).not.toContain('data-ball-state="measured"');
    expect(markup).toContain("solver off");
    expect(markup).toContain("physical_sanity_violation_fraction 0.400000 exceeds 0.200000");
  });

  it("does not fall back to the confidence-gated world's readout when the arc-solved artifact itself is untrusted", () => {
    // Even if a (correctly gated, mostly-hidden) world were supplied alongside
    // the untrusted artifact, the explicit solver_off state must win so the
    // HUD never silently reads as plain "not visible" without naming why.
    const markup = renderToStaticMarkup(
      <BallHonestyHud
        arcSolved={untrustedArtifact()}
        world={{ ball: { frames: [{ t: 0, conf: 0, visible: false, world_xyz: null }] } }}
        currentTime={0}
      />,
    );

    expect(markup).toContain('data-ball-state="solver_off"');
    expect(markup).not.toContain('data-ball-state="not_visible"');
  });
});
