import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

import { CoachComparisonWorkspace, nearestUserPhase } from "./CoachComparisonWorkspace";
import type { CoachingComparison } from "./comparisonData";

const anchors = [
  { phase: "ready", user_t: 0.67, confidence: 0.9 },
  { phase: "load", user_t: 1.03, confidence: 0.8 },
  { phase: "forward", user_t: 1.43, confidence: 0.8 },
  { phase: "strike_window", user_t: 1.87, confidence: 0.65 },
  { phase: "finish", user_t: 2.23, confidence: 0.8 },
] as CoachingComparison["user_phase_anchors"];

const userMotion = {
  source_id: "user_motion",
  artifact_type: "racketsport_virtual_world",
  uri: "/world.json",
  sha256: "a".repeat(64),
  json_pointer: "/players/0/frames/56/joints_world",
  role: "user_motion",
  provenance_band: "model_estimated",
  authority_band: "preview",
  gate_id: "BODY",
  gate_status: "unpassed",
} as const;

const rubric = {
  source_id: "published_rubric",
  artifact_type: "racketsport_coaching_reference_sources",
  uri: "/reference_sources.json",
  sha256: "b".repeat(64),
  json_pointer: "/sources/1/rubric/0",
  role: "published_rubric",
  provenance_band: "measured",
  authority_band: "verified",
  gate_id: null,
  gate_status: "not_applicable",
} as const;

const comparison: CoachingComparison = {
  schema_version: 1,
  artifact_type: "racketsport_coaching_comparison",
  status: "kinematics_only",
  comparison_basis: "published_rubric",
  user: {
    clip_id: "clip_a",
    player_id: 1,
    replay_manifest_sha256: "c".repeat(64),
    interval: { t0: 0.65, t1: 2.25 },
  },
  reference: {
    athlete_name: "Ben Johns",
    reference_set_id: "ben_johns_joola_dink_lesson",
    exemplar_id: "official_dink_lesson",
    source_url: "https://example.com/official",
    display_rights: "official_link_only",
    expert_reviewed: true,
  },
  shot: { label: "Forehand dink motion", label_source: "human_reviewed", confidence: 0.8, context_tags: ["kitchen"] },
  user_phase_anchors: anchors,
  alignment: null,
  cues: [
    {
      id: "compact_dink",
      rank: 1,
      phase: "strike_window",
      headline: "Make the dink more compact",
      instruction: "Start the forward swing with your paddle-side wrist closer to your torso.",
      visual: { kind: "joint_arrow", joint_names: ["right_wrist"], preferred_view: "rear", reference_overlay: "none" },
      measurement: {
        domain: "kinematics",
        metric_id: "wrist_reach_torso_lengths",
        user_value: 1.22,
        reference_value: null,
        delta: null,
        unit: "torso lengths",
        provenance: "model_estimated",
        authority: "preview",
        confidence: 0.65,
        evidence_locators: [userMotion, rubric],
      },
      claim_boundary: {
        kinematics: "supported",
        contact: "not_verified",
        causality: "not_established",
        reason: "BODY supports the pose difference; ball contact is not verified.",
      },
    },
  ],
  abstention_reasons: [],
  source_artifacts: [userMotion, rubric],
};

describe("CoachComparisonWorkspace", () => {
  it("renders one simple action while preserving the contact and rights boundaries", () => {
    const markup = renderToStaticMarkup(
      <CoachComparisonWorkspace comparison={comparison} currentTime={1.87} onSeek={vi.fn()} />,
    );

    expect(markup).toContain("Make the dink more compact");
    expect(markup).toContain("1.22 torso lengths");
    expect(markup).toContain("Ball contact not verified");
    expect(markup).toContain("Official lesson only");
    expect(markup).toContain("No pro footage or ghost mesh is copied");
    expect(markup).not.toContain("verified contact");
    expect(markup).toContain('aria-current="step"');
  });

  it("does not leave a phase highlighted outside the compared motion window", () => {
    const markup = renderToStaticMarkup(
      <CoachComparisonWorkspace comparison={comparison} currentTime={9} onSeek={vi.fn()} />,
    );

    expect(markup).not.toContain('aria-current="step"');
  });

  it("renders an honest compact abstention state instead of a generic correction diagram", () => {
    const abstained: CoachingComparison = {
      ...comparison,
      status: "abstained",
      user_phase_anchors: [],
      cues: [],
      abstention_reasons: ["motion_not_reliable"],
    };
    const markup = renderToStaticMarkup(
      <CoachComparisonWorkspace comparison={abstained} currentTime={1.87} onSeek={vi.fn()} />,
    );

    expect(markup).toContain("the evidence did not support an actionable change");
    expect(markup).toContain("No reliable comparison to draw");
    expect(markup).not.toContain("less reach");
  });

  it("selects the closest reviewed user phase without inventing reference timing", () => {
    expect(nearestUserPhase(anchors, 1.8)?.phase).toBe("strike_window");
    expect(nearestUserPhase([], 1.8)).toBeNull();
  });
});
