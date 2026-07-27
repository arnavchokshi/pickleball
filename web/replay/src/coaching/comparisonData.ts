export const COACHING_COMPARISON_PHASES = [
  "ready",
  "load",
  "forward",
  "strike_window",
  "finish",
] as const;

export type CoachingComparisonPhase = (typeof COACHING_COMPARISON_PHASES)[number];
export type CoachingComparisonStatus = "ready" | "kinematics_only" | "abstained";
export type CoachingComparisonBasis = "reference_motion" | "published_rubric" | "self_relative";
export type EvidenceRole =
  | "user_motion"
  | "reference_motion"
  | "published_rubric"
  | "shot_label"
  | "phase_alignment"
  | "contact"
  | "expert_review";
export type EvidenceProvenance = "measured" | "model_estimated" | "physics_predicted" | "missing";
export type EvidenceAuthority = "verified" | "preview" | "low_confidence" | "too_close_to_call";
export type EvidenceGateStatus = "pass" | "unpassed" | "not_applicable";

export type ComparisonInterval = {
  t0: number;
  t1: number;
};

export type ComparisonUserClip = {
  clip_id: string;
  player_id: number;
  replay_manifest_sha256: string;
  interval: ComparisonInterval;
};

export type ComparisonMotionReferenceClip = {
  athlete_name: string;
  reference_set_id: string;
  exemplar_id: string;
  source_url: string;
  replay_manifest_url: string;
  replay_manifest_sha256: string;
  player_id: number;
  interval: ComparisonInterval;
  handedness: "left" | "right";
  display_rights: "cleared" | "local_review_only";
  expert_reviewed: boolean;
};

export type ComparisonEmbedOnlyReferenceClip = {
  athlete_name: string;
  reference_set_id: string;
  exemplar_id: string;
  source_url: string;
  embed_url: string;
  display_rights: "official_embed_only";
  expert_reviewed: boolean;
};

export type ComparisonLinkOnlyReferenceClip = {
  athlete_name: string;
  reference_set_id: string;
  exemplar_id: string;
  source_url: string;
  display_rights: "official_link_only";
  expert_reviewed: boolean;
};

export type ComparisonReferenceClip =
  | ComparisonMotionReferenceClip
  | ComparisonEmbedOnlyReferenceClip
  | ComparisonLinkOnlyReferenceClip;

export type ComparisonShot = {
  label: string;
  label_source: "human_reviewed" | "model_estimated";
  confidence: number;
  context_tags: string[];
};

export type ComparisonPhaseAnchor = {
  phase: CoachingComparisonPhase;
  user_t: number;
  reference_t: number;
  confidence: number;
};

export type ComparisonUserPhaseAnchor = {
  phase: CoachingComparisonPhase;
  user_t: number;
  confidence: number;
};

export type ComparisonSpatialAlignment = {
  comparison_space: "body_local" | "court_world";
  pelvis_centered: boolean;
  facing_yaw_deg: number;
  uniform_scale: number;
  mirrored: boolean;
  scale_label: "metric" | "body_size_normalized";
};

export type ComparisonAlignmentQuality = {
  coverage: number;
  confidence: number;
  abstention_reasons: string[];
};

export type CoachingComparisonAlignment = {
  method: "monotonic_phase_map_v1";
  phase_anchors: ComparisonPhaseAnchor[];
  spatial: ComparisonSpatialAlignment;
  quality: ComparisonAlignmentQuality;
};

export type ComparisonEvidenceReference = {
  source_id: string;
  artifact_type: string;
  uri: string;
  sha256: string;
  json_pointer: string;
  role: EvidenceRole;
  provenance_band: EvidenceProvenance;
  authority_band: EvidenceAuthority;
  gate_id: string | null;
  gate_status: EvidenceGateStatus;
};

export type ComparisonCueVisual = {
  kind: "joint_arrow" | "angle_arc" | "phase_timing" | "stance_outline";
  joint_names: string[];
  preferred_view: "front" | "side" | "rear";
  reference_overlay: "ghost" | "none";
};

export type ComparisonCueMeasurement = {
  domain: "kinematics" | "contact";
  metric_id: string;
  user_value: number;
  reference_value: number | null;
  delta: number | null;
  unit: string;
  provenance: "model_estimated" | "measured";
  authority: EvidenceAuthority;
  confidence: number;
  evidence_locators: ComparisonEvidenceReference[];
};

export type ComparisonClaimBoundary = {
  kinematics: "supported" | "abstained";
  contact: "supported" | "not_verified" | "abstained";
  causality: "not_established";
  reason: string;
};

export type CoachingComparisonCue = {
  id: string;
  rank: 1 | 2 | 3;
  phase: CoachingComparisonPhase;
  headline: string;
  instruction: string;
  visual: ComparisonCueVisual;
  measurement: ComparisonCueMeasurement;
  claim_boundary: ComparisonClaimBoundary;
};

export type CoachingComparison = {
  schema_version: 1;
  artifact_type: "racketsport_coaching_comparison";
  status: CoachingComparisonStatus;
  comparison_basis: CoachingComparisonBasis;
  user: ComparisonUserClip;
  reference: ComparisonReferenceClip;
  shot: ComparisonShot;
  user_phase_anchors: ComparisonUserPhaseAnchor[];
  alignment: CoachingComparisonAlignment | null;
  cues: CoachingComparisonCue[];
  abstention_reasons: string[];
  source_artifacts: ComparisonEvidenceReference[];
};

const EVIDENCE_ROLES = [
  "user_motion",
  "reference_motion",
  "published_rubric",
  "shot_label",
  "phase_alignment",
  "contact",
  "expert_review",
] as const;
const PROVENANCE_BANDS = ["measured", "model_estimated", "physics_predicted", "missing"] as const;
const AUTHORITY_BANDS = ["verified", "preview", "low_confidence", "too_close_to_call"] as const;
const GATE_STATUSES = ["pass", "unpassed", "not_applicable"] as const;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;
const ID_PATTERN = /^[A-Za-z0-9._-]+$/;
const SOURCE_ID_PATTERN = /^[a-z0-9_]+$/;
const DELTA_ABSOLUTE_TOLERANCE = 1e-9;
const TIME_EPSILON_SECONDS = 1e-9;

export function parseCoachingComparison(input: unknown): CoachingComparison {
  const value = parseMaybeJson(input);
  assertRecord(value, "coaching_comparison");
  assertExactKeys(
    value,
    [
      "schema_version",
      "artifact_type",
      "status",
      "comparison_basis",
      "user",
      "reference",
      "shot",
      "user_phase_anchors",
      "alignment",
      "cues",
      "abstention_reasons",
      "source_artifacts",
    ],
    "coaching_comparison",
  );
  if (value.schema_version !== 1) throw new Error("coaching_comparison.schema_version must be 1");
  if (value.artifact_type !== "racketsport_coaching_comparison") {
    throw new Error("coaching_comparison.artifact_type must be racketsport_coaching_comparison");
  }

  const comparison: CoachingComparison = {
    schema_version: 1,
    artifact_type: "racketsport_coaching_comparison",
    status: readEnum(value.status, "coaching_comparison.status", ["ready", "kinematics_only", "abstained"] as const),
    comparison_basis: readEnum(
      value.comparison_basis,
      "coaching_comparison.comparison_basis",
      ["reference_motion", "published_rubric", "self_relative"] as const,
    ),
    user: readUserClip(value.user),
    reference: readReferenceClip(value.reference),
    shot: readShot(value.shot),
    user_phase_anchors: readArray(value.user_phase_anchors, "coaching_comparison.user_phase_anchors").map(readUserPhaseAnchor),
    alignment: value.alignment === null ? null : readAlignment(value.alignment),
    cues: readArray(value.cues, "coaching_comparison.cues").map(readCue),
    abstention_reasons: readUniqueStrings(value.abstention_reasons, "coaching_comparison.abstention_reasons"),
    source_artifacts: readArray(value.source_artifacts, "coaching_comparison.source_artifacts").map((entry, index) =>
      readEvidenceReference(entry, `coaching_comparison.source_artifacts[${index}]`),
    ),
  };
  validateComparison(comparison);
  return comparison;
}

export function bindCoachingComparisonToManifest(
  comparison: CoachingComparison,
  manifest: { clip: string },
  manifestSha256: string,
): CoachingComparison {
  if (comparison.user.clip_id !== manifest.clip) {
    throw new Error(
      `coach comparison clip mismatch: expected ${manifest.clip}, received ${comparison.user.clip_id}`,
    );
  }
  if (comparison.user.replay_manifest_sha256 !== manifestSha256) {
    throw new Error("coach comparison manifest hash mismatch");
  }
  return comparison;
}

export async function sha256Utf8(value: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

/**
 * Map one user source-video PTS to the synchronized reference PTS.
 *
 * The mapping is piecewise linear between the five semantic phase anchors and
 * clamps outside the compared motion window. It is intentionally not a free
 * dynamic-time warp: timing differences between semantic phases remain visible.
 */
export function mapUserTimeToReferenceTime(
  alignment: Pick<CoachingComparisonAlignment, "phase_anchors">,
  userTime: number,
): number {
  if (!Number.isFinite(userTime)) throw new Error("userTime must be finite");
  const anchors = alignment.phase_anchors;
  validatePhaseAnchors(anchors, null, null);
  if (userTime <= anchors[0].user_t) return anchors[0].reference_t;
  const last = anchors[anchors.length - 1];
  if (userTime >= last.user_t) return last.reference_t;

  for (let index = 1; index < anchors.length; index += 1) {
    const right = anchors[index];
    if (userTime > right.user_t) continue;
    const left = anchors[index - 1];
    const alpha = (userTime - left.user_t) / (right.user_t - left.user_t);
    return cleanNumber(left.reference_t + (right.reference_t - left.reference_t) * alpha);
  }
  return last.reference_t;
}

export function comparisonHasReferenceMotion(
  comparison: Pick<CoachingComparison, "comparison_basis" | "reference" | "alignment">,
): comparison is Pick<CoachingComparison, "comparison_basis"> & {
  reference: ComparisonMotionReferenceClip;
  alignment: CoachingComparisonAlignment;
} {
  return (
    comparison.comparison_basis !== "published_rubric" &&
    isMotionReference(comparison.reference) &&
    comparison.alignment !== null
  );
}

/**
 * The legacy/public replay surface may only render motion with cleared display
 * rights. The standalone mesh studio has a separate loopback-only gate for
 * explicitly local-review material.
 */
export function assertPublicCoachingComparisonDisplay(comparison: CoachingComparison): CoachingComparison {
  if (comparisonHasReferenceMotion(comparison) && comparison.reference.display_rights !== "cleared") {
    throw new Error("local-review-only reference motion cannot be rendered on the public replay surface");
  }
  return comparison;
}

function readUserClip(input: unknown): ComparisonUserClip {
  const path = "coaching_comparison.user";
  assertRecord(input, path);
  assertExactKeys(input, ["clip_id", "player_id", "replay_manifest_sha256", "interval"], path);
  return {
    clip_id: readNonEmptyString(input.clip_id, `${path}.clip_id`),
    player_id: readNonNegativeInteger(input.player_id, `${path}.player_id`),
    replay_manifest_sha256: readSha256(input.replay_manifest_sha256, `${path}.replay_manifest_sha256`),
    interval: readInterval(input.interval, `${path}.interval`),
  };
}

function readReferenceClip(input: unknown): ComparisonReferenceClip {
  const path = "coaching_comparison.reference";
  assertRecord(input, path);
  if (input.display_rights === "official_link_only") {
    assertExactKeys(
      input,
      ["athlete_name", "reference_set_id", "exemplar_id", "source_url", "display_rights", "expert_reviewed"],
      path,
    );
    return {
      athlete_name: readNonEmptyString(input.athlete_name, `${path}.athlete_name`),
      reference_set_id: readNonEmptyString(input.reference_set_id, `${path}.reference_set_id`),
      exemplar_id: readNonEmptyString(input.exemplar_id, `${path}.exemplar_id`),
      source_url: readNonEmptyString(input.source_url, `${path}.source_url`),
      display_rights: "official_link_only",
      expert_reviewed: readBoolean(input.expert_reviewed, `${path}.expert_reviewed`),
    };
  }
  if (input.display_rights === "official_embed_only") {
    assertExactKeys(
      input,
      [
        "athlete_name",
        "reference_set_id",
        "exemplar_id",
        "source_url",
        "embed_url",
        "display_rights",
        "expert_reviewed",
      ],
      path,
    );
    return {
      athlete_name: readNonEmptyString(input.athlete_name, `${path}.athlete_name`),
      reference_set_id: readNonEmptyString(input.reference_set_id, `${path}.reference_set_id`),
      exemplar_id: readNonEmptyString(input.exemplar_id, `${path}.exemplar_id`),
      source_url: readNonEmptyString(input.source_url, `${path}.source_url`),
      embed_url: readNonEmptyString(input.embed_url, `${path}.embed_url`),
      display_rights: "official_embed_only",
      expert_reviewed: readBoolean(input.expert_reviewed, `${path}.expert_reviewed`),
    };
  }
  assertExactKeys(
    input,
    [
      "athlete_name",
      "reference_set_id",
      "exemplar_id",
      "source_url",
      "replay_manifest_url",
      "replay_manifest_sha256",
      "player_id",
      "interval",
      "handedness",
      "display_rights",
      "expert_reviewed",
    ],
    path,
  );
  return {
    athlete_name: readNonEmptyString(input.athlete_name, `${path}.athlete_name`),
    reference_set_id: readNonEmptyString(input.reference_set_id, `${path}.reference_set_id`),
    exemplar_id: readNonEmptyString(input.exemplar_id, `${path}.exemplar_id`),
    source_url: readNonEmptyString(input.source_url, `${path}.source_url`),
    replay_manifest_url: readNonEmptyString(input.replay_manifest_url, `${path}.replay_manifest_url`),
    replay_manifest_sha256: readSha256(input.replay_manifest_sha256, `${path}.replay_manifest_sha256`),
    player_id: readNonNegativeInteger(input.player_id, `${path}.player_id`),
    interval: readInterval(input.interval, `${path}.interval`),
    handedness: readEnum(input.handedness, `${path}.handedness`, ["left", "right"] as const),
    display_rights: readEnum(input.display_rights, `${path}.display_rights`, ["cleared", "local_review_only"] as const),
    expert_reviewed: readBoolean(input.expert_reviewed, `${path}.expert_reviewed`),
  };
}

function readShot(input: unknown): ComparisonShot {
  const path = "coaching_comparison.shot";
  assertRecord(input, path);
  assertExactKeys(input, ["label", "label_source", "confidence", "context_tags"], path);
  return {
    label: readNonEmptyString(input.label, `${path}.label`),
    label_source: readEnum(input.label_source, `${path}.label_source`, ["human_reviewed", "model_estimated"] as const),
    confidence: readProbability(input.confidence, `${path}.confidence`),
    context_tags: readUniqueStrings(input.context_tags, `${path}.context_tags`),
  };
}

function readAlignment(input: unknown): CoachingComparisonAlignment {
  const path = "coaching_comparison.alignment";
  assertRecord(input, path);
  assertExactKeys(input, ["method", "phase_anchors", "spatial", "quality"], path);
  if (input.method !== "monotonic_phase_map_v1") {
    throw new Error(`${path}.method must be monotonic_phase_map_v1`);
  }
  return {
    method: "monotonic_phase_map_v1",
    phase_anchors: readArray(input.phase_anchors, `${path}.phase_anchors`).map(readPhaseAnchor),
    spatial: readSpatialAlignment(input.spatial),
    quality: readAlignmentQuality(input.quality),
  };
}

function readPhaseAnchor(input: unknown, index: number): ComparisonPhaseAnchor {
  const path = `coaching_comparison.alignment.phase_anchors[${index}]`;
  assertRecord(input, path);
  assertExactKeys(input, ["phase", "user_t", "reference_t", "confidence"], path);
  return {
    phase: readEnum(input.phase, `${path}.phase`, COACHING_COMPARISON_PHASES),
    user_t: readNonNegativeNumber(input.user_t, `${path}.user_t`),
    reference_t: readNonNegativeNumber(input.reference_t, `${path}.reference_t`),
    confidence: readProbability(input.confidence, `${path}.confidence`),
  };
}

function readUserPhaseAnchor(input: unknown, index: number): ComparisonUserPhaseAnchor {
  const path = `coaching_comparison.user_phase_anchors[${index}]`;
  assertRecord(input, path);
  assertExactKeys(input, ["phase", "user_t", "confidence"], path);
  return {
    phase: readEnum(input.phase, `${path}.phase`, COACHING_COMPARISON_PHASES),
    user_t: readNonNegativeNumber(input.user_t, `${path}.user_t`),
    confidence: readProbability(input.confidence, `${path}.confidence`),
  };
}

function readSpatialAlignment(input: unknown): ComparisonSpatialAlignment {
  const path = "coaching_comparison.alignment.spatial";
  assertRecord(input, path);
  assertExactKeys(
    input,
    ["comparison_space", "pelvis_centered", "facing_yaw_deg", "uniform_scale", "mirrored", "scale_label"],
    path,
  );
  const uniformScale = readNumber(input.uniform_scale, `${path}.uniform_scale`);
  if (uniformScale <= 0) throw new Error(`${path}.uniform_scale must be greater than 0`);
  return {
    comparison_space: readEnum(input.comparison_space, `${path}.comparison_space`, ["body_local", "court_world"] as const),
    pelvis_centered: readBoolean(input.pelvis_centered, `${path}.pelvis_centered`),
    facing_yaw_deg: readNumber(input.facing_yaw_deg, `${path}.facing_yaw_deg`),
    uniform_scale: uniformScale,
    mirrored: readBoolean(input.mirrored, `${path}.mirrored`),
    scale_label: readEnum(input.scale_label, `${path}.scale_label`, ["metric", "body_size_normalized"] as const),
  };
}

function readAlignmentQuality(input: unknown): ComparisonAlignmentQuality {
  const path = "coaching_comparison.alignment.quality";
  assertRecord(input, path);
  assertExactKeys(input, ["coverage", "confidence", "abstention_reasons"], path);
  return {
    coverage: readProbability(input.coverage, `${path}.coverage`),
    confidence: readProbability(input.confidence, `${path}.confidence`),
    abstention_reasons: readUniqueStrings(input.abstention_reasons, `${path}.abstention_reasons`),
  };
}

function readCue(input: unknown, index: number): CoachingComparisonCue {
  const path = `coaching_comparison.cues[${index}]`;
  assertRecord(input, path);
  assertExactKeys(input, ["id", "rank", "phase", "headline", "instruction", "visual", "measurement", "claim_boundary"], path);
  const id = readNonEmptyString(input.id, `${path}.id`);
  if (!ID_PATTERN.test(id)) throw new Error(`${path}.id must contain only letters, numbers, dot, underscore, or hyphen`);
  const rank = readInteger(input.rank, `${path}.rank`);
  if (rank < 1 || rank > 3) throw new Error(`${path}.rank must be between 1 and 3`);
  return {
    id,
    rank: rank as 1 | 2 | 3,
    phase: readEnum(input.phase, `${path}.phase`, COACHING_COMPARISON_PHASES),
    headline: readNonEmptyString(input.headline, `${path}.headline`),
    instruction: readNonEmptyString(input.instruction, `${path}.instruction`),
    visual: readCueVisual(input.visual, `${path}.visual`),
    measurement: readCueMeasurement(input.measurement, `${path}.measurement`),
    claim_boundary: readClaimBoundary(input.claim_boundary, `${path}.claim_boundary`),
  };
}

function readCueVisual(input: unknown, path: string): ComparisonCueVisual {
  assertRecord(input, path);
  assertExactKeys(input, ["kind", "joint_names", "preferred_view", "reference_overlay"], path);
  const kind = readEnum(input.kind, `${path}.kind`, ["joint_arrow", "angle_arc", "phase_timing", "stance_outline"] as const);
  const jointNames = readUniqueStrings(input.joint_names, `${path}.joint_names`, 8);
  if (kind !== "phase_timing" && jointNames.length === 0) {
    throw new Error(`${path}.joint_names must name at least one joint for ${kind}`);
  }
  return {
    kind,
    joint_names: jointNames,
    preferred_view: readEnum(input.preferred_view, `${path}.preferred_view`, ["front", "side", "rear"] as const),
    reference_overlay: readEnum(input.reference_overlay, `${path}.reference_overlay`, ["ghost", "none"] as const),
  };
}

function readCueMeasurement(input: unknown, path: string): ComparisonCueMeasurement {
  assertRecord(input, path);
  assertExactKeys(
    input,
    [
      "domain",
      "metric_id",
      "user_value",
      "reference_value",
      "delta",
      "unit",
      "provenance",
      "authority",
      "confidence",
      "evidence_locators",
    ],
    path,
  );
  const metricId = readNonEmptyString(input.metric_id, `${path}.metric_id`);
  if (!ID_PATTERN.test(metricId)) {
    throw new Error(`${path}.metric_id must contain only letters, numbers, dot, underscore, or hyphen`);
  }
  const locators = readArray(input.evidence_locators, `${path}.evidence_locators`).map((entry, index) =>
    readEvidenceReference(entry, `${path}.evidence_locators[${index}]`),
  );
  if (!locators.length) throw new Error(`${path}.evidence_locators must contain at least one item`);
  const referenceValue = readNullableNumber(input.reference_value, `${path}.reference_value`);
  const delta = readNullableNumber(input.delta, `${path}.delta`);
  if ((referenceValue === null) !== (delta === null)) {
    throw new Error(`${path}.reference_value and delta must either both be numbers or both be null`);
  }
  return {
    domain: readEnum(input.domain, `${path}.domain`, ["kinematics", "contact"] as const),
    metric_id: metricId,
    user_value: readNumber(input.user_value, `${path}.user_value`),
    reference_value: referenceValue,
    delta,
    unit: readNonEmptyString(input.unit, `${path}.unit`),
    provenance: readEnum(input.provenance, `${path}.provenance`, ["model_estimated", "measured"] as const),
    authority: readEnum(input.authority, `${path}.authority`, AUTHORITY_BANDS),
    confidence: readProbability(input.confidence, `${path}.confidence`),
    evidence_locators: locators,
  };
}

function readClaimBoundary(input: unknown, path: string): ComparisonClaimBoundary {
  assertRecord(input, path);
  assertExactKeys(input, ["kinematics", "contact", "causality", "reason"], path);
  if (input.causality !== "not_established") throw new Error(`${path}.causality must be not_established`);
  return {
    kinematics: readEnum(input.kinematics, `${path}.kinematics`, ["supported", "abstained"] as const),
    contact: readEnum(input.contact, `${path}.contact`, ["supported", "not_verified", "abstained"] as const),
    causality: "not_established",
    reason: readNonEmptyString(input.reason, `${path}.reason`),
  };
}

function readEvidenceReference(input: unknown, path: string): ComparisonEvidenceReference {
  assertRecord(input, path);
  assertExactKeys(
    input,
    [
      "source_id",
      "artifact_type",
      "uri",
      "sha256",
      "json_pointer",
      "role",
      "provenance_band",
      "authority_band",
      "gate_id",
      "gate_status",
    ],
    path,
  );
  const sourceId = readNonEmptyString(input.source_id, `${path}.source_id`);
  if (!SOURCE_ID_PATTERN.test(sourceId)) throw new Error(`${path}.source_id must match ^[a-z0-9_]+$`);
  const jsonPointer = readString(input.json_pointer, `${path}.json_pointer`);
  if (!jsonPointer.startsWith("/")) throw new Error(`${path}.json_pointer must start with /`);
  return {
    source_id: sourceId,
    artifact_type: readNonEmptyString(input.artifact_type, `${path}.artifact_type`),
    uri: readNonEmptyString(input.uri, `${path}.uri`),
    sha256: readSha256(input.sha256, `${path}.sha256`),
    json_pointer: jsonPointer,
    role: readEnum(input.role, `${path}.role`, EVIDENCE_ROLES),
    provenance_band: readEnum(input.provenance_band, `${path}.provenance_band`, PROVENANCE_BANDS),
    authority_band: readEnum(input.authority_band, `${path}.authority_band`, AUTHORITY_BANDS),
    gate_id: input.gate_id === null ? null : readNonEmptyString(input.gate_id, `${path}.gate_id`),
    gate_status: readEnum(input.gate_status, `${path}.gate_status`, GATE_STATUSES),
  };
}

function readInterval(input: unknown, path: string): ComparisonInterval {
  assertRecord(input, path);
  assertExactKeys(input, ["t0", "t1"], path);
  const interval = {
    t0: readNonNegativeNumber(input.t0, `${path}.t0`),
    t1: readNonNegativeNumber(input.t1, `${path}.t1`),
  };
  if (interval.t1 <= interval.t0) throw new Error(`${path}.t1 must be greater than t0`);
  return interval;
}

function validateComparison(comparison: CoachingComparison): void {
  if (comparison.cues.length > 3) throw new Error("coaching_comparison.cues must contain at most 3 items");
  if (comparison.source_artifacts.length < 2) {
    throw new Error("coaching_comparison.source_artifacts must contain at least 2 items");
  }
  assertUnique(
    comparison.source_artifacts.map((source) => source.source_id),
    "coaching_comparison.source_artifacts source_id",
  );
  const motionReference = isMotionReference(comparison.reference) ? comparison.reference : null;
  const hasMotionReference = motionReference !== null;
  if (comparison.status !== "abstained" || comparison.user_phase_anchors.length > 0) {
    validateUserPhaseAnchors(comparison.user_phase_anchors, comparison.user.interval);
  }
  if (comparison.comparison_basis === "published_rubric") {
    requireEvidenceRoles(comparison.source_artifacts, ["user_motion", "published_rubric"], "coaching_comparison.source_artifacts");
    if (comparison.alignment !== null) {
      throw new Error("published_rubric coaching comparison cannot derive a motion alignment");
    }
  } else {
    if (!hasMotionReference) {
      throw new Error(`${comparison.comparison_basis} coaching comparison requires a reference motion asset`);
    }
    requireEvidenceRoles(comparison.source_artifacts, ["user_motion", "reference_motion"], "coaching_comparison.source_artifacts");
    if (comparison.status !== "abstained" && comparison.alignment === null) {
      throw new Error(`${comparison.comparison_basis} coaching comparison requires a five-phase motion alignment`);
    }
  }
  if (!hasMotionReference && comparison.comparison_basis !== "published_rubric") {
    throw new Error(`${comparison.reference.display_rights} reference can only support a published_rubric comparison`);
  }
  if (comparison.alignment !== null) {
    if (motionReference === null) throw new Error("motion alignment requires a reference motion asset");
    validatePhaseAnchors(comparison.alignment.phase_anchors, comparison.user.interval, motionReference.interval);
    validateAlignmentUserAnchors(comparison.alignment.phase_anchors, comparison.user_phase_anchors);
    validateSpatialAlignment(comparison.alignment.spatial);
  }

  const cueIds = comparison.cues.map((cue) => cue.id);
  assertUnique(cueIds, "coaching_comparison.cues id");
  comparison.cues.forEach((cue, index) => validateCue(cue, index, comparison.comparison_basis));

  if (comparison.status === "abstained") {
    if (comparison.cues.length !== 0) throw new Error("abstained coaching comparison must not emit actionable cues");
    if (!comparison.abstention_reasons.length) {
      throw new Error("abstained coaching comparison must explain at least one abstention reason");
    }
    return;
  }

  const localUnreviewedPreview =
    motionReference?.display_rights === "local_review_only" &&
    !comparison.reference.expert_reviewed &&
    comparison.status === "kinematics_only";
  if (!comparison.reference.expert_reviewed && !localUnreviewedPreview) {
    throw new Error("non-abstained coaching comparison requires an expert-reviewed reference exemplar");
  }
  if (
    localUnreviewedPreview &&
    comparison.cues.some(
      (cue) => cue.measurement.authority === "verified" || cue.claim_boundary.contact === "supported",
    )
  ) {
    throw new Error("unreviewed local reference cues must remain preview-only kinematics");
  }
  if (!comparison.cues.length) throw new Error(`${comparison.status} coaching comparison must emit at least one cue`);
  if (
    comparison.alignment !== null &&
    (comparison.alignment.quality.coverage <= 0 || comparison.alignment.quality.confidence <= 0)
  ) {
    throw new Error("non-abstained coaching comparison requires positive alignment coverage and confidence");
  }

  const supportedContact = comparison.cues.some((cue) => cue.claim_boundary.contact === "supported");
  if (comparison.status === "kinematics_only" && supportedContact) {
    throw new Error("kinematics_only coaching comparison cannot claim supported contact");
  }
  if (comparison.status === "ready" && !supportedContact) {
    throw new Error("ready coaching comparison requires at least one cue with independently supported contact evidence");
  }
}

function validateUserPhaseAnchors(
  anchors: ComparisonUserPhaseAnchor[],
  userInterval: ComparisonInterval,
): void {
  if (anchors.length !== COACHING_COMPARISON_PHASES.length) {
    throw new Error("non-abstained coaching comparison requires exactly five ordered user phase anchors");
  }
  for (let index = 0; index < COACHING_COMPARISON_PHASES.length; index += 1) {
    const anchor = anchors[index];
    const expectedPhase = COACHING_COMPARISON_PHASES[index];
    if (anchor.phase !== expectedPhase) {
      throw new Error(`coaching comparison user_phase_anchors[${index}].phase must be ${expectedPhase}`);
    }
    if (index > 0 && anchor.user_t <= anchors[index - 1].user_t) {
      throw new Error("coaching comparison user phase times must be strictly increasing");
    }
    if (!timeInside(anchor.user_t, userInterval)) {
      throw new Error(`coaching comparison user_phase_anchors[${index}].user_t must lie inside the user interval`);
    }
  }
}

function validateAlignmentUserAnchors(
  alignmentAnchors: ComparisonPhaseAnchor[],
  userAnchors: ComparisonUserPhaseAnchor[],
): void {
  if (userAnchors.length !== alignmentAnchors.length) {
    throw new Error("motion alignment and user_phase_anchors must have the same five phases");
  }
  alignmentAnchors.forEach((anchor, index) => {
    const userAnchor = userAnchors[index];
    if (anchor.phase !== userAnchor.phase || !numbersNearlyEqual(anchor.user_t, userAnchor.user_t)) {
      throw new Error(`coaching comparison alignment phase_anchors[${index}].user_t must match user_phase_anchors`);
    }
  });
}

function validatePhaseAnchors(
  anchors: ComparisonPhaseAnchor[],
  userInterval: ComparisonInterval | null,
  referenceInterval: ComparisonInterval | null,
): void {
  if (anchors.length !== COACHING_COMPARISON_PHASES.length) {
    throw new Error("coaching comparison alignment requires exactly five semantic phase anchors");
  }
  for (let index = 0; index < COACHING_COMPARISON_PHASES.length; index += 1) {
    const anchor = anchors[index];
    const expectedPhase = COACHING_COMPARISON_PHASES[index];
    if (anchor.phase !== expectedPhase) {
      throw new Error(`coaching comparison phase_anchors[${index}].phase must be ${expectedPhase}`);
    }
    if (index > 0) {
      const previous = anchors[index - 1];
      if (anchor.user_t <= previous.user_t) {
        throw new Error("coaching comparison user phase-anchor times must be strictly increasing");
      }
      if (anchor.reference_t <= previous.reference_t) {
        throw new Error("coaching comparison reference phase-anchor times must be strictly increasing");
      }
    }
    if (userInterval && !timeInside(anchor.user_t, userInterval)) {
      throw new Error(`coaching comparison phase_anchors[${index}].user_t must lie inside the user interval`);
    }
    if (referenceInterval && !timeInside(anchor.reference_t, referenceInterval)) {
      throw new Error(`coaching comparison phase_anchors[${index}].reference_t must lie inside the reference interval`);
    }
  }
}

function validateSpatialAlignment(spatial: ComparisonSpatialAlignment): void {
  if (spatial.comparison_space === "body_local" && !spatial.pelvis_centered) {
    throw new Error("body_local coaching comparison must declare pelvis_centered=true");
  }
  if (spatial.comparison_space === "court_world") {
    if (spatial.pelvis_centered) throw new Error("court_world coaching comparison cannot be pelvis centered");
    if (spatial.scale_label !== "metric" || Math.abs(spatial.uniform_scale - 1) > DELTA_ABSOLUTE_TOLERANCE) {
      throw new Error("court_world coaching comparison must preserve metric scale with uniform_scale=1");
    }
  }
}

function validateCue(cue: CoachingComparisonCue, index: number, basis: CoachingComparisonBasis): void {
  const path = `coaching_comparison.cues[${index}]`;
  if (cue.rank !== index + 1) throw new Error(`${path}.rank must be ${index + 1}; cues must already be priority ordered`);
  if (cue.claim_boundary.kinematics !== "supported") {
    throw new Error(`${path} cannot be actionable when its kinematics claim is abstained`);
  }
  if (cue.measurement.authority === "too_close_to_call") {
    throw new Error(`${path} cannot be actionable when its measurement is too_close_to_call`);
  }
  if (cue.measurement.reference_value !== null && cue.measurement.delta !== null) {
    const expectedDelta = cue.measurement.user_value - cue.measurement.reference_value;
    if (!numbersNearlyEqual(cue.measurement.delta, expectedDelta)) {
      throw new Error(`${path}.measurement.delta must equal user_value - reference_value`);
    }
  }
  if (basis === "published_rubric") {
    requireEvidenceRoles(cue.measurement.evidence_locators, ["user_motion", "published_rubric"], `${path}.measurement.evidence_locators`);
    if (cue.visual.reference_overlay !== "none") {
      throw new Error(`${path} cannot render a reference ghost from a published rubric or embed-only reference`);
    }
  } else {
    requireEvidenceRoles(cue.measurement.evidence_locators, ["user_motion", "reference_motion"], `${path}.measurement.evidence_locators`);
    if (cue.measurement.reference_value === null || cue.measurement.delta === null) {
      throw new Error(`${path} requires numeric reference_value and delta for ${basis}`);
    }
  }

  const contactEvidence = cue.measurement.evidence_locators.filter((source) => source.role === "contact");
  const independentlySupportedContact = contactEvidence.some(
    (source) =>
      source.gate_status === "pass" &&
      source.authority_band === "verified" &&
      source.provenance_band !== "missing",
  );
  if (cue.claim_boundary.contact === "supported" && !independentlySupportedContact) {
    throw new Error(`${path} claims supported contact without verified, gate-passing contact evidence`);
  }
  if (cue.measurement.domain === "contact" && cue.claim_boundary.contact !== "supported") {
    throw new Error(`${path} contains a contact measurement while contact is not independently supported`);
  }
}

function requireEvidenceRoles(
  evidence: ComparisonEvidenceReference[],
  roles: EvidenceRole[],
  path: string,
): void {
  for (const role of roles) {
    if (!evidence.some((source) => source.role === role && source.provenance_band !== "missing")) {
      throw new Error(`${path} must include non-missing ${role} evidence`);
    }
  }
}

function parseMaybeJson(input: unknown): unknown {
  if (typeof input !== "string") return input;
  try {
    return JSON.parse(input) as unknown;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`invalid coaching comparison JSON: ${message}`);
  }
}

function assertRecord(value: unknown, path: string): asserts value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${path} must be an object`);
  }
}

function assertExactKeys(value: Record<string, unknown>, allowed: readonly string[], path: string): void {
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) throw new Error(`${path}.${key} is not allowed`);
  }
  for (const key of allowed) {
    if (!(key in value)) throw new Error(`${path}.${key} is required`);
  }
}

function readArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${path} must be an array`);
  return value;
}

function readString(value: unknown, path: string): string {
  if (typeof value !== "string") throw new Error(`${path} must be a string`);
  return value;
}

function readNonEmptyString(value: unknown, path: string): string {
  const text = readString(value, path);
  if (!text.trim()) throw new Error(`${path} must be a non-empty string`);
  return text;
}

function readBoolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") throw new Error(`${path} must be a boolean`);
  return value;
}

function readNumber(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${path} must be a finite number`);
  return value;
}

function readNullableNumber(value: unknown, path: string): number | null {
  return value === null ? null : readNumber(value, path);
}

function readNonNegativeNumber(value: unknown, path: string): number {
  const number = readNumber(value, path);
  if (number < 0) throw new Error(`${path} must be nonnegative`);
  return number;
}

function readInteger(value: unknown, path: string): number {
  const number = readNumber(value, path);
  if (!Number.isInteger(number)) throw new Error(`${path} must be an integer`);
  return number;
}

function readNonNegativeInteger(value: unknown, path: string): number {
  const number = readInteger(value, path);
  if (number < 0) throw new Error(`${path} must be nonnegative`);
  return number;
}

function readProbability(value: unknown, path: string): number {
  const number = readNumber(value, path);
  if (number < 0 || number > 1) throw new Error(`${path} must be between 0 and 1`);
  return number;
}

function readEnum<const T extends readonly string[]>(value: unknown, path: string, values: T): T[number] {
  if (typeof value !== "string" || !values.includes(value)) {
    throw new Error(`${path} must be one of ${values.join(", ")}`);
  }
  return value as T[number];
}

function readUniqueStrings(value: unknown, path: string, maxItems?: number): string[] {
  const values = readArray(value, path).map((entry, index) => readNonEmptyString(entry, `${path}[${index}]`));
  if (maxItems !== undefined && values.length > maxItems) throw new Error(`${path} must contain at most ${maxItems} items`);
  assertUnique(values, path);
  return values;
}

function readSha256(value: unknown, path: string): string {
  const hash = readString(value, path);
  if (!SHA256_PATTERN.test(hash)) throw new Error(`${path} must be a lowercase SHA-256 digest`);
  return hash;
}

function assertUnique(values: readonly string[], path: string): void {
  if (new Set(values).size !== values.length) throw new Error(`${path} values must be unique`);
}

function timeInside(value: number, interval: ComparisonInterval): boolean {
  return value >= interval.t0 - TIME_EPSILON_SECONDS && value <= interval.t1 + TIME_EPSILON_SECONDS;
}

function numbersNearlyEqual(left: number, right: number): boolean {
  const scale = Math.max(1, Math.abs(left), Math.abs(right));
  return Math.abs(left - right) <= DELTA_ABSOLUTE_TOLERANCE * scale;
}

function cleanNumber(value: number): number {
  return Object.is(value, -0) ? 0 : value;
}

function isMotionReference(reference: ComparisonReferenceClip): reference is ComparisonMotionReferenceClip {
  return "replay_manifest_url" in reference;
}
