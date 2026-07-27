import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import {
  bindCoachingComparisonToManifest,
  comparisonHasReferenceMotion,
  mapUserTimeToReferenceTime,
  parseCoachingComparison,
  sha256Utf8,
  type CoachingComparison,
  type ComparisonEvidenceReference,
} from "./comparisonData";

const HASH_A = "a".repeat(64);
const HASH_B = "b".repeat(64);

function evidence(
  role: ComparisonEvidenceReference["role"],
  overrides: Partial<ComparisonEvidenceReference> = {},
): ComparisonEvidenceReference {
  return {
    source_id: `${role}_source`,
    artifact_type: `${role}_artifact`,
    uri: `file:///tmp/${role}.json`,
    sha256: role === "reference_motion" ? HASH_B : HASH_A,
    json_pointer: "/frames/0",
    role,
    provenance_band: role === "published_rubric" ? "measured" : "model_estimated",
    authority_band: "preview",
    gate_id: role === "contact" ? "contact" : "BODY",
    gate_status: "unpassed",
    ...overrides,
  };
}

function motionComparison(): CoachingComparison {
  const userMotion = evidence("user_motion");
  const referenceMotion = evidence("reference_motion");
  return {
    schema_version: 1,
    artifact_type: "racketsport_coaching_comparison",
    status: "kinematics_only",
    comparison_basis: "reference_motion",
    user: {
      clip_id: "owner_serve_001",
      player_id: 1,
      replay_manifest_sha256: HASH_A,
      interval: { t0: 1, t1: 3 },
    },
    reference: {
      athlete_name: "Reviewed professional",
      reference_set_id: "serve_reference_v1",
      exemplar_id: "serve_001",
      source_url: "https://example.test/source",
      replay_manifest_url: "/@fs/tmp/pro/replay_viewer_manifest.json",
      replay_manifest_sha256: HASH_B,
      player_id: 4,
      interval: { t0: 10, t1: 12 },
      handedness: "right",
      display_rights: "cleared",
      expert_reviewed: true,
    },
    shot: {
      label: "serve",
      label_source: "human_reviewed",
      confidence: 1,
      context_tags: ["right_handed", "baseline"],
    },
    user_phase_anchors: [
      { phase: "ready", user_t: 1, confidence: 0.95 },
      { phase: "load", user_t: 1.4, confidence: 0.9 },
      { phase: "forward", user_t: 1.8, confidence: 0.88 },
      { phase: "strike_window", user_t: 2.2, confidence: 0.8 },
      { phase: "finish", user_t: 3, confidence: 0.92 },
    ],
    alignment: {
      method: "monotonic_phase_map_v1",
      phase_anchors: [
        { phase: "ready", user_t: 1, reference_t: 10, confidence: 0.95 },
        { phase: "load", user_t: 1.4, reference_t: 10.3, confidence: 0.9 },
        { phase: "forward", user_t: 1.8, reference_t: 10.9, confidence: 0.88 },
        { phase: "strike_window", user_t: 2.2, reference_t: 11.25, confidence: 0.8 },
        { phase: "finish", user_t: 3, reference_t: 12, confidence: 0.92 },
      ],
      spatial: {
        comparison_space: "body_local",
        pelvis_centered: true,
        facing_yaw_deg: 8,
        uniform_scale: 0.97,
        mirrored: false,
        scale_label: "body_size_normalized",
      },
      quality: { coverage: 0.94, confidence: 0.86, abstention_reasons: [] },
    },
    cues: [
      {
        id: "lead_knee_load",
        rank: 1,
        phase: "load",
        headline: "Load through the lead knee",
        instruction: "Keep the lead knee flexed through the load phase.",
        visual: {
          kind: "angle_arc",
          joint_names: ["left_hip", "left_knee", "left_ankle"],
          preferred_view: "side",
          reference_overlay: "ghost",
        },
        measurement: {
          domain: "kinematics",
          metric_id: "lead_knee_flexion_deg",
          user_value: 42,
          reference_value: 30,
          delta: 12,
          unit: "deg",
          provenance: "model_estimated",
          authority: "preview",
          confidence: 0.84,
          evidence_locators: [userMotion, referenceMotion],
        },
        claim_boundary: {
          kinematics: "supported",
          contact: "not_verified",
          causality: "not_established",
          reason: "Body motion is comparable; ball contact and outcome causality are not verified.",
        },
      },
    ],
    abstention_reasons: [],
    source_artifacts: [userMotion, referenceMotion],
  };
}

function embedOnlyComparison(): CoachingComparison {
  const userMotion = evidence("user_motion");
  const rubric = evidence("published_rubric", {
    source_id: "official_serve_rubric",
    uri: "https://example.test/official-serve-guidance",
    json_pointer: "/",
    gate_id: null,
    gate_status: "not_applicable",
    authority_band: "preview",
  });
  return {
    schema_version: 1,
    artifact_type: "racketsport_coaching_comparison",
    status: "kinematics_only",
    comparison_basis: "published_rubric",
    user: {
      clip_id: "owner_serve_001",
      player_id: 1,
      replay_manifest_sha256: HASH_A,
      interval: { t0: 1, t1: 3 },
    },
    reference: {
      athlete_name: "Professional reference",
      reference_set_id: "official_embed_reference_v1",
      exemplar_id: "official_serve_embed_001",
      source_url: "https://example.test/watch-page",
      embed_url: "https://example.test/embed/video",
      display_rights: "official_embed_only",
      expert_reviewed: true,
    },
    shot: {
      label: "serve",
      label_source: "human_reviewed",
      confidence: 1,
      context_tags: ["right_handed", "baseline"],
    },
    user_phase_anchors: [
      { phase: "ready", user_t: 1, confidence: 0.95 },
      { phase: "load", user_t: 1.4, confidence: 0.9 },
      { phase: "forward", user_t: 1.8, confidence: 0.88 },
      { phase: "strike_window", user_t: 2.2, confidence: 0.8 },
      { phase: "finish", user_t: 3, confidence: 0.92 },
    ],
    alignment: null,
    cues: [
      {
        id: "stable_base",
        rank: 1,
        phase: "load",
        headline: "Build a quieter base",
        instruction: "Hold your base through the forward swing.",
        visual: {
          kind: "stance_outline",
          joint_names: ["left_ankle", "right_ankle"],
          preferred_view: "front",
          reference_overlay: "none",
        },
        measurement: {
          domain: "kinematics",
          metric_id: "stance_stability_review",
          user_value: 0.61,
          reference_value: null,
          delta: null,
          unit: "score",
          provenance: "model_estimated",
          authority: "preview",
          confidence: 0.78,
          evidence_locators: [userMotion, rubric],
        },
        claim_boundary: {
          kinematics: "supported",
          contact: "not_verified",
          causality: "not_established",
          reason: "The user motion is estimated; the official video is embed-only and supplies no derived pro motion.",
        },
      },
    ],
    abstention_reasons: ["professional_reference_motion_not_permissioned"],
    source_artifacts: [userMotion, rubric],
  };
}

function linkOnlyComparison(): CoachingComparison {
  const payload = embedOnlyComparison();
  payload.reference = {
    athlete_name: "Ben Johns",
    reference_set_id: "official_link_reference_v1",
    exemplar_id: "ben_serve_lesson_001",
    source_url: "https://example.test/official-ben-serve-lesson",
    display_rights: "official_link_only",
    expert_reviewed: true,
  };
  return payload;
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

describe("coaching comparison contract", () => {
  it("parses a permissioned reference-motion comparison and maps the five semantic phases", () => {
    const parsed = parseCoachingComparison(JSON.stringify(motionComparison()));

    expect(comparisonHasReferenceMotion(parsed)).toBe(true);
    expect(parsed.cues).toHaveLength(1);
    expect(mapUserTimeToReferenceTime(parsed.alignment!, 0)).toBe(10);
    expect(mapUserTimeToReferenceTime(parsed.alignment!, 1.6)).toBeCloseTo(10.6, 10);
    expect(mapUserTimeToReferenceTime(parsed.alignment!, 4)).toBe(12);
  });

  it("binds a comparison to the exact replay clip and manifest bytes", () => {
    const parsed = parseCoachingComparison(motionComparison());

    expect(bindCoachingComparisonToManifest(parsed, { clip: "owner_serve_001" }, HASH_A)).toBe(parsed);
    expect(() => bindCoachingComparisonToManifest(parsed, { clip: "different_clip" }, HASH_A)).toThrow(
      "coach comparison clip mismatch",
    );
    expect(() => bindCoachingComparisonToManifest(parsed, { clip: "owner_serve_001" }, HASH_B)).toThrow(
      "coach comparison manifest hash mismatch",
    );
  });

  it("hashes the exact UTF-8 manifest bytes used for comparison binding", async () => {
    expect(await sha256Utf8("abc")).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  });

  it("rejects local-review-only motion from the user-facing comparison path", () => {
    const payload = clone(motionComparison());
    if ("replay_manifest_url" in payload.reference) payload.reference.display_rights = "local_review_only";

    expect(() => parseCoachingComparison(payload)).toThrow("requires a cleared reference motion asset");
  });

  it("accepts an honest official-embed-only rubric comparison without a ghost, alignment, or derived pro number", () => {
    const parsed = parseCoachingComparison(embedOnlyComparison());

    expect(comparisonHasReferenceMotion(parsed)).toBe(false);
    expect(parsed.reference.display_rights).toBe("official_embed_only");
    expect(parsed.alignment).toBeNull();
    expect(parsed.cues[0].visual.reference_overlay).toBe("none");
    expect(parsed.cues[0].measurement.reference_value).toBeNull();
    expect(parsed.cues[0].measurement.delta).toBeNull();
  });

  it("accepts an official-link-only rubric without inventing an iframe or motion asset", () => {
    const parsed = parseCoachingComparison(linkOnlyComparison());

    expect(parsed.reference.display_rights).toBe("official_link_only");
    expect("embed_url" in parsed.reference).toBe(false);
    expect("replay_manifest_url" in parsed.reference).toBe(false);
    expect(comparisonHasReferenceMotion(parsed)).toBe(false);
  });

  it("rejects an embed URL on an official-link-only reference", () => {
    const payload = clone(linkOnlyComparison()) as CoachingComparison & {
      reference: CoachingComparison["reference"] & { embed_url?: string };
    };
    payload.reference.embed_url = "https://example.test/iframe";

    expect(() => parseCoachingComparison(payload)).toThrow("embed_url is not allowed");
  });

  it("keeps the checked-in JSON schema strict and capped at three cues", () => {
    const schemaPath = resolve(process.cwd(), "../../docs/racketsport/coaching_comparison_schema.json");
    const schema = JSON.parse(readFileSync(schemaPath, "utf8")) as Record<string, any>;

    expect(schema.additionalProperties).toBe(false);
    expect(schema.properties.cues.maxItems).toBe(3);
    expect(schema.properties.alignment.oneOf).toContainEqual({ type: "null" });
  });

  it("rejects more than three actionable cues", () => {
    const payload = clone(motionComparison());
    payload.cues = [payload.cues[0], payload.cues[0], payload.cues[0], payload.cues[0]];

    expect(() => parseCoachingComparison(payload)).toThrow("at most 3");
  });

  it("rejects an unknown field at any parsed contract boundary", () => {
    const payload = clone(motionComparison()) as CoachingComparison & { invented_score?: number };
    payload.invented_score = 99;

    expect(() => parseCoachingComparison(payload)).toThrow("invented_score is not allowed");
  });

  it("rejects reordered semantic phases", () => {
    const payload = clone(motionComparison());
    const alignment = payload.alignment!;
    alignment.phase_anchors[1].phase = "forward";

    expect(() => parseCoachingComparison(payload)).toThrow("phase_anchors[1].phase must be load");
  });

  it("rejects non-monotonic user or reference phase time", () => {
    const payload = clone(motionComparison());
    payload.alignment!.phase_anchors[2].reference_t = payload.alignment!.phase_anchors[1].reference_t;

    expect(() => parseCoachingComparison(payload)).toThrow("reference phase-anchor times must be strictly increasing");
  });

  it("requires five ordered user seek phases even when the professional reference is link-only", () => {
    const payload = clone(linkOnlyComparison());
    payload.user_phase_anchors = payload.user_phase_anchors.slice(0, 4);

    expect(() => parseCoachingComparison(payload)).toThrow("requires exactly five ordered user phase anchors");
  });

  it("requires motion alignment user times to match the auditable user seek anchors", () => {
    const payload = clone(motionComparison());
    payload.user_phase_anchors[2].user_t = 1.75;

    expect(() => parseCoachingComparison(payload)).toThrow("must match user_phase_anchors");
  });

  it("rejects phase anchors outside either compared source interval", () => {
    const payload = clone(motionComparison());
    payload.alignment!.phase_anchors[4].user_t = 3.1;

    expect(() => parseCoachingComparison(payload)).toThrow("user_t must lie inside the user interval");
  });

  it("rejects a reference ghost when the only professional asset is an official embed or published rubric", () => {
    const payload = clone(embedOnlyComparison());
    payload.cues[0].visual.reference_overlay = "ghost";

    expect(() => parseCoachingComparison(payload)).toThrow("cannot render a reference ghost");
  });

  it("rejects derived reference motion and phase alignment from an official-embed-only source", () => {
    const payload = clone(embedOnlyComparison());
    payload.comparison_basis = "reference_motion";

    expect(() => parseCoachingComparison(payload)).toThrow("requires a cleared reference motion asset");
  });

  it("requires numeric reference values and deltas for a real reference-motion comparison", () => {
    const payload = clone(motionComparison());
    payload.cues[0].measurement.reference_value = null;
    payload.cues[0].measurement.delta = null;

    expect(() => parseCoachingComparison(payload)).toThrow("requires numeric reference_value and delta");
  });

  it("rejects an internally inconsistent numeric delta", () => {
    const payload = clone(motionComparison());
    payload.cues[0].measurement.delta = 11;

    expect(() => parseCoachingComparison(payload)).toThrow("delta must equal user_value - reference_value");
  });

  it("rejects supported contact without verified gate-passing contact evidence", () => {
    const payload = clone(motionComparison());
    payload.cues[0].claim_boundary.contact = "supported";
    payload.cues[0].measurement.domain = "contact";
    payload.cues[0].measurement.evidence_locators.push(evidence("contact"));
    payload.status = "ready";

    expect(() => parseCoachingComparison(payload)).toThrow("without verified, gate-passing contact evidence");
  });

  it("accepts ready contact only when contact evidence is independently gate-passing and verified", () => {
    const payload = clone(motionComparison());
    payload.cues[0].claim_boundary.contact = "supported";
    payload.cues[0].measurement.domain = "contact";
    payload.cues[0].measurement.provenance = "measured";
    payload.cues[0].measurement.evidence_locators.push(
      evidence("contact", {
        provenance_band: "measured",
        authority_band: "verified",
        gate_status: "pass",
      }),
    );
    payload.status = "ready";

    expect(parseCoachingComparison(payload).status).toBe("ready");
  });

  it("rejects contact-domain measurements that remain not verified", () => {
    const payload = clone(motionComparison());
    payload.cues[0].measurement.domain = "contact";

    expect(() => parseCoachingComparison(payload)).toThrow("contact measurement while contact is not independently supported");
  });

  it("rejects actionable cues whose measurement is too close to call", () => {
    const payload = clone(motionComparison());
    payload.cues[0].measurement.authority = "too_close_to_call";

    expect(() => parseCoachingComparison(payload)).toThrow("cannot be actionable when its measurement is too_close_to_call");
  });

  it("requires abstained comparisons to emit no cues and preserve a reason", () => {
    const invalid = clone(embedOnlyComparison());
    invalid.status = "abstained";
    expect(() => parseCoachingComparison(invalid)).toThrow("must not emit actionable cues");

    const valid = clone(embedOnlyComparison());
    valid.status = "abstained";
    valid.cues = [];
    valid.user_phase_anchors = [];
    valid.alignment = null;
    valid.abstention_reasons = ["user_phase_alignment_unavailable"];
    expect(parseCoachingComparison(valid).status).toBe("abstained");

    valid.abstention_reasons = [];
    expect(() => parseCoachingComparison(valid)).toThrow("must explain at least one abstention reason");
  });
});
