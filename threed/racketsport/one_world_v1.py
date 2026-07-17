"""Deterministic preview-only same-run fusion for ``one_world_v1``.

The pass is intentionally standalone.  It reads immutable artifacts from one
explicit run directory and writes only a new preview artifact.  It never imports
the pipeline runner and never promotes its inputs or its own output.
"""

from __future__ import annotations

import hashlib
import bisect
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .coordinates import (
    CoordinateSpace,
    camera_to_world_points,
    invert_extrinsics,
    project_world_points,
    translation_to_metres,
)
from .schemas import (
    CameraIntrinsics,
    ConfidenceProvenance,
    CourtExtrinsics,
    FiniteFloat,
    Matrix3,
    SE3,
    StrictArtifact,
    TrustBand,
    Vector2,
    Vector3,
)


BALL_RADIUS_M = 0.0371
WRIST_RADIUS_M = 0.12
WRIST_INDICES = (9, 10)
CONTACT_MAX_RAW_DISTANCE_M = 1.2
POP_LIKELIHOOD_LOG_BOUND = 0.20
DEFAULT_BALL_SIGMA_M = 0.20
DEFAULT_TRACK_SIGMA_M = 0.25
FLOAT_EPS = 1e-12


class OneWorldInputRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact: str
    path: str
    sha256: str
    generation: str
    schema_version: int | None
    consumed_fields: list[str]
    trust_band: TrustBand | None
    missing_reason: str | None


class OneWorldRuleProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule: str
    input_refs: list[str]
    nominal_weights: dict[str, FiniteFloat]
    effective_weights: dict[str, FiniteFloat]
    discounts: list[str]
    robust_kernel: str
    correction_cap: str | None
    degraded_reasons: list[str]


class OneWorldPlayerState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_id: int
    root_world: Vector3
    covariance_m2: Matrix3
    joints_world: list[Vector3] | None
    joint_conf: list[FiniteFloat] | None
    placement_tier: Literal["trackI_refined", "placement_fused", "tracks_world_xy"]
    confidence: FiniteFloat
    trust_band: TrustBand
    confidence_provenance: ConfidenceProvenance
    provenance: OneWorldRuleProvenance


class OneWorldBallState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_xyz: Vector3
    covariance_m2: Matrix3
    xy_observed_px: Vector2 | None
    confidence: FiniteFloat
    source_generation: str
    estimate_tier: Literal["arc_measured", "physics_predicted", "ray_court_projection"]
    ray_origin_world: Vector3 | None
    ray_direction_world: Vector3 | None
    altitude_unknown: bool
    approx: bool
    render_only: Literal[True]
    not_for_detection_metrics: Literal[True]
    trust_band: TrustBand
    confidence_provenance: ConfidenceProvenance
    provenance: OneWorldRuleProvenance


class OneWorldPaddleState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_id: int
    status: Literal["resolved", "unresolved", "unresolved_legacy_wrist_proxy"]
    pose_world: SE3 | None
    display_pose_world: SE3 | None
    display_tier: Literal["resolved", "unresolved_best_evidence", "unresolved_legacy_wrist_proxy"]
    winning_hypothesis: str | None
    retained_hypotheses: list[dict[str, Any]]
    score_components: dict[str, FiniteFloat]
    score_margin: FiniteFloat | None
    ambiguity_margin_px: FiniteFloat | None
    confidence: FiniteFloat
    trust_band: TrustBand
    provenance: OneWorldRuleProvenance


class OneWorldFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_idx: int
    t: FiniteFloat
    players: list[OneWorldPlayerState]
    ball: OneWorldBallState | None
    paddles: list[OneWorldPaddleState]
    missing: list[str]


class ContactHypothesisEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_id: int
    wrist_index: int
    center_distance_m: FiniteFloat
    ball_term: FiniteFloat
    wrist_term: FiniteFloat
    event_confidence_term: FiniteFloat
    marker_reliability: FiniteFloat
    declared_hitter_multiplier: FiniteFloat
    audio_bounded_multiplier: FiniteFloat
    combined_likelihood: FiniteFloat
    discounts: list[str]


class ContactEvidenceVector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upstream_sources: dict[str, Any]
    visual_event_exists: bool
    ball_track_supported: bool
    wrist_candidates_supported: bool
    audio_bounded_multiplier: FiniteFloat
    hypotheses: list[ContactHypothesisEvidence]
    null_likelihood: FiniteFloat


class RacketPoseHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pose_se3: SE3
    confidence: FiniteFloat
    frame_conf: FiniteFloat
    reprojection_error_px: FiniteFloat
    source: str


class RacketPoseHypothesisFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: FiniteFloat
    primary_pose: RacketPoseHypothesis
    alt_pose: RacketPoseHypothesis | None
    candidate_reprojection_errors_px: list[FiniteFloat]
    ambiguity_margin_px: FiniteFloat | None
    ambiguous: bool


class RacketPoseHypothesisPlayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    paddle_dims_in: dict[str, FiniteFloat]
    frames: list[RacketPoseHypothesisFrame]


class RacketPoseHypotheses(StrictArtifact):
    artifact_type: Literal["racketsport_racket_pose_hypotheses"]
    fps: FiniteFloat
    world_frame: Literal["camera"]
    translation_unit: Literal["cm", "m"]
    players: list[RacketPoseHypothesisPlayer]


class OneWorldContactRefinement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_index: int
    frame: int
    t: FiniteFloat
    status: str
    raw_player_id: int | None
    hitter_id: int | None
    hitter_confidence: FiniteFloat
    hitter_band: Literal["resolved", "too_close_to_call", "unsupported"]
    per_player_wrist_likelihoods: dict[str, list[FiniteFloat]]
    contact_evidence_vector: ContactEvidenceVector
    raw_ball_world: Vector3 | None
    refined_ball_world: Vector3 | None
    raw_wrist_volume_residual_m: FiniteFloat | None
    refined_wrist_volume_residual_m: FiniteFloat | None
    displacement_m: FiniteFloat | None
    confidence: FiniteFloat
    trust_band: TrustBand
    provenance: OneWorldRuleProvenance


class OneWorldBounceRefinement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_index: int
    frame: int
    t: FiniteFloat
    status: str
    raw_ball_world: Vector3 | None
    refined_ball_world: Vector3 | None
    signed_plane_residual_before_m: FiniteFloat | None
    signed_plane_residual_after_m: FiniteFloat | None
    out_of_court_bounds: bool
    confidence: FiniteFloat
    trust_band: TrustBand
    provenance: OneWorldRuleProvenance


class OneWorldEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_index: int
    type: Literal["paddle_contact", "floor_bounce", "net_contact", "net_cross"]
    t: FiniteFloat
    frame: int
    world_location_raw: Vector3 | None
    world_location_refined: Vector3 | None
    hitter_id: int | None
    confidence: FiniteFloat
    trust_band: TrustBand
    evidence: ContactEvidenceVector | dict[str, Any]
    provenance: OneWorldRuleProvenance


class OneWorldSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    placement_tier_counts: dict[str, int]
    missing_counts: dict[str, int]
    ball_contact_distance_m: dict[str, FiniteFloat | int | None]
    bounce_plane_residual_m: dict[str, FiniteFloat | int | None]
    world_coverage: dict[str, Any]
    paddle_resolution: dict[str, FiniteFloat | int]
    reprojection_consistency: dict[str, Any]
    regression_kills: list[str]
    warnings: list[str]


class OneWorldV1(StrictArtifact):
    artifact_type: Literal["racketsport_one_world_v1"]
    world_frame: Literal["court_Z0"]
    coordinate_space: Literal["world_court_netcenter_z_up_m"]
    fps: FiniteFloat
    VERIFIED: Literal[0]
    preview_only: Literal[True]
    render_only: Literal[True]
    not_for_detection_metrics: Literal[True]
    not_for_training: Literal[True]
    raw_inputs_mutated: Literal[False]
    inputs: list[OneWorldInputRef]
    frames: list[OneWorldFrame]
    contacts: list[OneWorldContactRefinement]
    bounces: list[OneWorldBounceRefinement]
    events: list[OneWorldEvent]
    summary: OneWorldSummary
    trust_band: TrustBand


class OneWorldMetricDistribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    median: FiniteFloat | None
    p90_nearest_rank: FiniteFloat | None


class OneWorldMetrics(StrictArtifact):
    artifact_type: Literal["racketsport_one_world_v1_metrics"]
    VERIFIED: Literal[0]
    preview_only: Literal[True]
    render_only: Literal[True]
    not_for_detection_metrics: Literal[True]
    not_for_training: Literal[True]
    baseline_source: str
    fused_source: str
    metrics: dict[str, Any]


class OneWorldValidation(StrictArtifact):
    artifact_type: Literal["racketsport_one_world_v1_validation"]
    VERIFIED: Literal[0]
    preview_only: Literal[True]
    render_only: Literal[True]
    not_for_detection_metrics: Literal[True]
    not_for_training: Literal[True]
    valid: bool
    checks: dict[str, bool]
    errors: list[str]
    warnings: list[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: BaseModel | Mapping[str, Any]) -> str:
    value = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else dict(payload)
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clip01(value: Any) -> float:
    return min(1.0, max(0.0, _finite(value)))


def _frame_id(frame: Mapping[str, Any], fps: float, fallback: int | None = None) -> int:
    if frame.get("frame_idx") is not None:
        return int(frame["frame_idx"])
    if frame.get("frame") is not None:
        return int(frame["frame"])
    if frame.get("frame_float") is not None:
        return int(round(_finite(frame["frame_float"])))
    if frame.get("t") is not None:
        return int(round(_finite(frame["t"]) * fps))
    if fallback is not None:
        return fallback
    raise ValueError("frame has no frame_idx/frame/frame_float/t")


def _validate_time(frame: Mapping[str, Any], fps: float, frame_id: int) -> None:
    if frame.get("t") is None:
        return
    tolerance = max(1e-6, 0.25 / fps)
    if abs(_finite(frame["t"]) - frame_id / fps) > tolerance + 1e-12:
        raise ValueError(f"frame/time mismatch at frame {frame_id}")


def _player_maps(payload: Mapping[str, Any] | None, fps: float) -> dict[int, dict[int, dict[str, Any]]]:
    result: dict[int, dict[int, dict[str, Any]]] = {}
    if payload is None:
        return result
    for player in payload.get("players", []):
        if not isinstance(player, Mapping):
            continue
        pid = int(player["id"])
        rows: dict[int, dict[str, Any]] = {}
        for index, raw in enumerate(player.get("frames", [])):
            if not isinstance(raw, dict):
                continue
            fid = _frame_id(raw, fps, index)
            _validate_time(raw, fps, fid)
            rows[fid] = raw
        result[pid] = rows
    return result


def _frame_map(payload: Mapping[str, Any] | None, fps: float, key: str = "frames") -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    if payload is None:
        return result
    for index, raw in enumerate(payload.get(key, [])):
        if not isinstance(raw, dict):
            continue
        fid = _frame_id(raw, fps, index)
        _validate_time(raw, fps, fid)
        result[fid] = raw
    return result


def _identity_covariance(sigma: float) -> list[list[float]]:
    variance = float(sigma) ** 2
    return [[variance, 0.0, 0.0], [0.0, variance, 0.0], [0.0, 0.0, variance]]


def _preview_band(reason: str = "one_world_v1 is permanently preview-only") -> TrustBand:
    return TrustBand(
        stage="WORLD",
        gate_id="ns04_independent_world_fusion_gate",
        gate_status="VERIFIED=0",
        badge="preview",
        reason=reason,
        evidence_path=None,
    )


def _normalize_band(raw: Mapping[str, Any] | None, *, fallback_reason: str) -> TrustBand:
    if not raw:
        return _preview_band(fallback_reason)
    badge = str(raw.get("badge", "preview"))
    if badge == "verified":
        badge = "preview"
    if badge not in {"preview", "low_confidence"}:
        badge = "low_confidence"
    return TrustBand(
        stage=str(raw.get("stage", "WORLD")),
        gate_id=str(raw.get("gate_id", "ns04_independent_world_fusion_gate")),
        gate_status=str(raw.get("gate_status", raw.get("status", "VERIFIED=0"))),
        badge=badge,  # type: ignore[arg-type]
        reason=str(raw.get("reason", raw.get("note", fallback_reason))),
        evidence_path=str(raw["evidence_path"]) if raw.get("evidence_path") is not None else None,
    )


def _low_band(reason: str) -> TrustBand:
    return TrustBand(stage="WORLD", gate_id="ns04_ball_continuity_display_only", gate_status="VERIFIED=0", badge="low_confidence", reason=reason, evidence_path=None)


def ray_court_projection(
    xy: Sequence[float], calibration: Mapping[str, Any]
) -> tuple[list[float], list[float], list[float]] | None:
    """Typed pinhole ray and z=0 court-plane intersection for display only."""

    intrinsics = calibration.get("intrinsics")
    extrinsics = calibration.get("extrinsics")
    if not isinstance(intrinsics, Mapping) or not isinstance(extrinsics, Mapping):
        return None
    ray_camera = np.asarray([(_finite(xy[0]) - _finite(intrinsics.get("cx"))) / _finite(intrinsics.get("fx"), 1.0), (_finite(xy[1]) - _finite(intrinsics.get("cy"))) / _finite(intrinsics.get("fy"), 1.0), 1.0], dtype=np.float64)
    camera_to_world_R, origin = invert_extrinsics(extrinsics["R"], extrinsics["t"])
    direction = camera_to_world_R @ ray_camera
    direction /= max(float(np.linalg.norm(direction)), FLOAT_EPS)
    if abs(float(direction[2])) <= 1e-9:
        return None
    scale = -float(origin[2]) / float(direction[2])
    if scale <= 0.0:
        return None
    intersection = origin + scale * direction
    return origin.tolist(), direction.tolist(), intersection.tolist()


def _provenance(
    rule: str,
    refs: Sequence[str],
    nominal: Mapping[str, float],
    effective: Mapping[str, float],
    *,
    discounts: Sequence[str] = (),
    cap: str | None = None,
    degraded: Sequence[str] = (),
) -> OneWorldRuleProvenance:
    return OneWorldRuleProvenance(
        rule=rule,
        input_refs=list(refs),
        nominal_weights={str(k): float(v) for k, v in nominal.items()},
        effective_weights={str(k): float(v) for k, v in effective.items()},
        discounts=list(discounts),
        robust_kernel="Huber(delta=1.5), three deterministic IRLS iterations",
        correction_cap=cap,
        degraded_reasons=list(degraded),
    )


def _confidence_provenance(band: str, predictor: str, sigma: float | None = None) -> ConfidenceProvenance:
    return ConfidenceProvenance(
        band=band,
        display_band="preview" if band != "low_confidence" else "low_confidence",
        predictor=predictor,
        horizon_frames=0,
        predicted_sigma_m=sigma,
    )


def _huber_influence(normalized_residual: float) -> float:
    magnitude = abs(float(normalized_residual))
    return 1.0 if magnitude <= 1.5 or magnitude == 0.0 else 1.5 / magnitude


def _nearest_rank(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "p90_nearest_rank": _nearest_rank(values, 0.90),
    }


def _observed_stride(player_frames: Mapping[int, Any]) -> float:
    ids = sorted(player_frames)
    diffs = [b - a for a, b in zip(ids, ids[1:]) if b > a]
    return float(statistics.median(diffs)) if diffs else 1.0


def interpolate_wrist(
    player_frames: Mapping[int, Mapping[str, Any]],
    frame_id: int,
    wrist_index: int,
    fps: float,
) -> tuple[list[float], float, float, list[str]] | None:
    """Return an exact or guarded latent wrist observation."""

    exact = player_frames.get(frame_id)
    if exact is not None:
        joints = exact.get("joints_world", [])
        confs = exact.get("joint_conf", [])
        if len(joints) > wrist_index and len(confs) > wrist_index:
            confidence = _clip01(confs[wrist_index])
            if confidence > 0.0:
                sigma = 0.02 + 0.10 * (1.0 - confidence)
                return [float(v) for v in joints[wrist_index]], confidence, sigma, []

    ids = sorted(player_frames)
    lower = max((candidate for candidate in ids if candidate < frame_id), default=None)
    upper = min((candidate for candidate in ids if candidate > frame_id), default=None)
    if lower is None or upper is None:
        return None
    stride = _observed_stride(player_frames)
    gap = upper - lower
    if gap > min(0.10 * fps, 2.0 * stride) + FLOAT_EPS:
        return None
    left = player_frames[lower]
    right = player_frames[upper]
    lj, rj = left.get("joints_world", []), right.get("joints_world", [])
    lc, rc = left.get("joint_conf", []), right.get("joint_conf", [])
    if min(len(lj), len(rj), len(lc), len(rc)) <= wrist_index:
        return None
    c0, c1 = _clip01(lc[wrist_index]), _clip01(rc[wrist_index])
    if min(c0, c1) < 0.5:
        return None
    p0 = np.asarray(lj[wrist_index], dtype=np.float64)
    p1 = np.asarray(rj[wrist_index], dtype=np.float64)
    elapsed = gap / fps
    if elapsed <= 0.0 or float(np.linalg.norm(p1 - p0)) / elapsed > 15.0:
        return None
    alpha = (frame_id - lower) / gap
    point = (1.0 - alpha) * p0 + alpha * p1
    confidence = min(c0, c1) * math.exp(-gap / (2.0 * stride))
    sigma2 = (0.02 + 0.10 * (1.0 - confidence)) ** 2 + (0.04 * gap / stride) ** 2
    return point.tolist(), confidence, math.sqrt(sigma2), ["interpolated_latent_wrist"]


def soft_surface_refinement(
    world_xyz: Sequence[float],
    *,
    plane_point: Sequence[float],
    plane_normal: Sequence[float],
    ball_confidence: float,
    bounce_confidence: float,
    sigma_ball_m: float,
    sigma_cal_m: float,
    sigma_event_m: float,
    calibration_multiplier: float,
) -> tuple[list[float], float, float, float, float]:
    """Finite soft bounce update; it never assigns ``z = BALL_RADIUS_M``."""

    x = np.asarray(world_xyz, dtype=np.float64)
    point = np.asarray(plane_point, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError("surface normal must be nonzero")
    normal = normal / norm
    residual_before = float(np.dot(normal, x - point) - BALL_RADIUS_M)
    combined_sigma = math.sqrt(sigma_cal_m**2 + sigma_event_m**2)
    surface_quality = _clip01(bounce_confidence) ** 2 * calibration_multiplier
    ball_quality = _clip01(ball_confidence) ** 2
    w_surface = surface_quality / max(combined_sigma**2, 0.005**2)
    w_ball = ball_quality / max(sigma_ball_m**2, 0.03**2)
    gain = w_surface / max(w_surface + w_ball, FLOAT_EPS)
    magnitude = _huber_influence(residual_before / max(combined_sigma, FLOAT_EPS)) * residual_before * gain
    cap = min(0.15, 2.0 * sigma_ball_m)
    magnitude = min(cap, max(-cap, magnitude))
    refined = x - normal * magnitude
    residual_after = float(np.dot(normal, refined - point) - BALL_RADIUS_M)
    return refined.tolist(), residual_before, residual_after, w_surface, w_ball


def _inside_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = float(point[0]), float(point[1])
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or FLOAT_EPS) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def lift_camera_pose_to_world(
    pose: Mapping[str, Any], calibration: Mapping[str, Any], *, input_unit: str
) -> dict[str, list[list[float]] | list[float]]:
    """Typed camera-frame centimetre/metre pose lift used by gen-2 paddles."""

    extrinsics = calibration["extrinsics"]
    translation_m = translation_to_metres(pose["t"], input_unit=input_unit)
    world_t = camera_to_world_points([translation_m], extrinsics["R"], extrinsics["t"])[0]
    camera_to_world_R, _ = invert_extrinsics(extrinsics["R"], extrinsics["t"])
    world_R = camera_to_world_R @ np.asarray(pose["R"], dtype=np.float64)
    return {"R": world_R.tolist(), "t": world_t.tolist()}


def resolve_two_hypothesis_sequence(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Two-state Viterbi over independent energies; reprojection is carried only.

    Each input frame supplies ``hypotheses`` with ids ``primary``/``alt`` and
    independent ``wrist``, ``contact``, and optional ``momentum`` energies.
    Reprojection fields are deliberately ignored by all score arithmetic.
    """

    if not frames:
        return {"status": "unresolved", "reason": "missing_hypotheses", "path": [], "display_path": [], "display_rule": "none", "margin": None}
    states = ("primary", "alt")
    costs: dict[str, tuple[float, list[str]]] = {}
    term_count = 0
    wrist_frames = 0
    contact_exists = False
    independent_extra = False
    for frame_index, frame in enumerate(frames):
        by_id = {str(row["id"]): row for row in frame.get("hypotheses", [])}
        if set(by_id) != set(states):
            return {"status": "unresolved", "reason": "missing_two_valid_hypotheses", "path": [], "display_path": [], "display_rule": "none", "margin": None}
        next_costs: dict[str, tuple[float, list[str]]] = {}
        for state in states:
            row = by_id[state]
            components = [row.get("wrist"), row.get("contact"), row.get("momentum")]
            unary = sum(_finite(value) for value in components if value is not None)
            if frame_index == 0:
                next_costs[state] = (unary, [state])
                continue
            candidates = []
            for previous in states:
                transition = _finite(frame.get("transition", {}).get(f"{previous}->{state}"), 0.0)
                candidates.append((costs[previous][0] + transition + unary, costs[previous][1] + [state]))
            next_costs[state] = min(candidates, key=lambda item: (item[0], item[1]))
        costs = next_costs
        sample = by_id["primary"]
        if sample.get("wrist") is not None:
            wrist_frames += 1
        if sample.get("contact") is not None:
            contact_exists = True
        if sample.get("momentum") is not None or wrist_frames >= 3:
            independent_extra = True
        term_count += sum(sample.get(key) is not None for key in ("wrist", "contact", "momentum"))
    ranked = sorted(((cost, path, state) for state, (cost, path) in costs.items()), key=lambda row: (row[0], row[2]))
    best, second = ranked[0], ranked[1]
    margin = (second[0] - best[0]) / max(term_count, 1)
    if abs(second[0] - best[0]) <= 1e-12:
        reprojection_totals = []
        for _, path, state in ranked:
            total = 0.0
            for frame, selected in zip(frames, path):
                row = next(item for item in frame["hypotheses"] if item["id"] == selected)
                total += _finite(row.get("reprojection_error_px"), 1e12) if row.get("reprojection_error_px") is not None else 1e12
            reprojection_totals.append((total, path, state))
        display = min(reprojection_totals, key=lambda row: (row[0], row[2]))[1]
        return {"status": "unresolved", "reason": "energy_tie", "path": [], "display_path": display, "display_rule": "display_tiebreak_reproj", "margin": margin}
    if margin < 0.25 or wrist_frames < 3 or not contact_exists or not independent_extra:
        return {"status": "unresolved", "reason": "independent_evidence_requirements", "path": [], "display_path": best[1], "display_rule": "best_independent_evidence_below_resolve_bar", "margin": margin}
    return {"status": "resolved", "reason": None, "path": best[1], "display_path": best[1], "display_rule": "resolved", "margin": margin}


class _RunReader:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir.resolve()
        self.refs: list[OneWorldInputRef] = []
        self.hashes_before: dict[Path, str] = {}

    def optional(
        self,
        filename: str,
        *,
        generation: str,
        consumed_fields: Sequence[str],
        trust_band: TrustBand | None = None,
    ) -> dict[str, Any] | None:
        path = self.run_dir / filename
        if not path.exists():
            self.refs.append(
                OneWorldInputRef(
                    artifact=filename.removesuffix(".json"),
                    path=filename,
                    sha256="",
                    generation="missing",
                    schema_version=None,
                    consumed_fields=list(consumed_fields),
                    trust_band=trust_band,
                    missing_reason="file_absent",
                )
            )
            return None
        digest = sha256_file(path)
        payload = _load_object(path)
        self.hashes_before[path] = digest
        self.refs.append(
            OneWorldInputRef(
                artifact=filename.removesuffix(".json"),
                path=filename,
                sha256=digest,
                generation=generation,
                schema_version=int(payload["schema_version"]) if payload.get("schema_version") is not None else None,
                consumed_fields=list(consumed_fields),
                trust_band=trust_band,
                missing_reason=None,
            )
        )
        return payload

    def assert_immutable(self) -> None:
        for path, digest in self.hashes_before.items():
            if sha256_file(path) != digest:
                raise RuntimeError(f"raw input mutated during build: {path.name}")


def _select_fps(*payloads: Mapping[str, Any] | None) -> float:
    values = [_finite(payload.get("fps", payload.get("frame_rate"))) for payload in payloads if payload is not None and payload.get("fps", payload.get("frame_rate")) is not None]
    if not values:
        raise ValueError("no input declares fps/frame_rate")
    fps = values[0]
    if fps <= 0.0 or any(abs(value - fps) > 1e-9 for value in values[1:]):
        raise ValueError(f"input fps mismatch: {values}")
    return fps


def _arc_map(arc: Mapping[str, Any] | None, fps: float, generation: str) -> dict[int, dict[str, Any]]:
    if arc is None:
        return {}
    rows = arc.get("samples", []) if generation == "ball_arc_render" else arc.get("frames", [])
    result: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            continue
        frame_id = _frame_id(raw, fps, index)
        world = raw.get("world_xyz")
        if world is None:
            continue
        result[frame_id] = {
            **raw,
            "frame": frame_id,
            "conf": raw.get("conf", raw.get("confidence", 0.0)),
            "sigma_m": raw.get("sigma_m", DEFAULT_BALL_SIGMA_M),
            "source": raw.get("source", generation),
        }
    return result


def _select_placement(
    reader: _RunReader,
    tracks: Mapping[str, Any] | None,
    body: Mapping[str, Any] | None,
    fps: float,
    warnings: list[str],
) -> tuple[str, Mapping[str, Any] | None]:
    refined = reader.optional(
        "placement_trajectory_refined.json",
        generation="trackI_refined",
        consumed_fields=["players", "placement_trajectory_refinement.provenance.inputs"],
    )
    if refined is not None:
        valid = (
            refined.get("artifact_type") == "placement_trajectory_refined"
            and refined.get("world_frame") == "court_Z0"
            and refined.get("coordinate_space") == "world_court_netcenter_z_up_m"
            and abs(_finite(refined.get("fps")) - fps) <= 1e-9
        )
        expected = sorted(int(player["id"]) for player in (tracks or {}).get("players", []))
        actual = sorted(int(player["id"]) for player in refined.get("players", []))
        valid = valid and (not expected or expected == actual)
        provenance = refined.get("placement_trajectory_refinement", {}).get("provenance", {}).get("inputs", {})
        current_by_name = {
            "tracks": reader.run_dir / "tracks.json",
            "skeleton3d": reader.run_dir / ("smpl_motion.json" if (reader.run_dir / "smpl_motion.json").exists() else "skeleton3d.json"),
        }
        for name, current in current_by_name.items():
            recorded = provenance.get(name)
            if recorded is not None and current.exists() and str(recorded.get("sha256", "")) != sha256_file(current):
                valid = False
        if valid:
            return "trackI_refined", refined
        warnings.append("placement_trajectory_refined_identity_mismatch_fallback")
    placement = reader.optional(
        "placement.json",
        generation="placement_fused",
        consumed_fields=["players.frames.smoothed_world_xy", "players.frames.covariance_m2"],
    )
    if placement is not None:
        if abs(_finite(placement.get("fps"), fps) - fps) > 1e-9:
            raise ValueError("placement fps mismatch")
        return "placement_fused", placement
    return "tracks_world_xy", tracks


def _audio_multiplier(event: Mapping[str, Any], audio: Mapping[str, Any] | None, fps: float) -> tuple[float, list[str]]:
    discounts = ["audio_review_only_not_gate_verified=0.20"]
    if audio is None or audio.get("status") != "ran":
        return 1.0, discounts + ["audio_missing_or_blocked_neutral"]
    frame = int(event.get("frame", round(_finite(event.get("t")) * fps)))
    matching = [
        onset
        for onset in audio.get("onsets", [])
        if abs(int(round(_finite(onset.get("time_s", onset.get("analysis_time_s"))) * fps)) - frame) <= 1
    ]
    if not matching:
        return 1.0, discounts + ["no_nearby_audio_onset_neutral"]
    ratio = max(_finite(row.get("features", {}).get("pop_band_ratio"), 0.5) for row in matching)
    log_multiplier = min(POP_LIKELIHOOD_LOG_BOUND, max(-POP_LIKELIHOOD_LOG_BOUND, 0.20 * (2.0 * ratio - 1.0)))
    return math.exp(log_multiplier), discounts


def _project_error(
    world: Sequence[float],
    observed_xy: Sequence[float] | None,
    calibration: Mapping[str, Any] | None,
) -> float | None:
    if observed_xy is None or calibration is None or calibration.get("extrinsics") is None or calibration.get("intrinsics") is None:
        return None
    try:
        projected = project_world_points(
            CourtExtrinsics.model_validate(calibration["extrinsics"]),
            CameraIntrinsics.model_validate(calibration["intrinsics"]),
            [world],
            input_space=CoordinateSpace.WORLD_COURT_NETCENTER_Z_UP_M,
            output_space=CoordinateSpace.PIXELS_UNDISTORTED_NATIVE,
            reference_space=CoordinateSpace.PIXELS_RAW_NATIVE,
        )[0]
    except (ValueError, KeyError, TypeError):
        return None
    return math.dist([float(projected[0]), float(projected[1])], [float(observed_xy[0]), float(observed_xy[1])])


def build_one_world(run_dir: Path) -> OneWorldV1:
    """Build a deterministic permanently-preview one-world artifact."""

    reader = _RunReader(run_dir)
    warnings: list[str] = []
    regression_kills: list[str] = []

    calibration = reader.optional("court_calibration.json", generation="calibration", consumed_fields=["homography", "intrinsics", "extrinsics", "metric_confidence", "trust_band"])
    trust_payload = reader.optional("trust_bands.json", generation="trust", consumed_fields=["court"])
    band_raw = (calibration or {}).get("trust_band") or (trust_payload or {}).get("court")
    band = _normalize_band(band_raw, fallback_reason="missing or legacy calibration trust; fail-safe preview")
    tracks = reader.optional("tracks.json", generation="tracks", consumed_fields=["fps", "players", "rally_spans"])
    repair_summary = reader.optional("repair_summary.json", generation="player_repair", consumed_fields=["input_paths.tracks_path", "summary.confidence_repairs"])
    ball_track = reader.optional("ball_track.json", generation="ball_track", consumed_fields=["fps", "frames", "bounces"])
    if ball_track is None:
        raise FileNotFoundError(f"{Path(run_dir) / 'ball_track.json'} is required")
    body = reader.optional("smpl_motion.json", generation="smpl_motion", consumed_fields=["fps", "players", "model", "world_frame"])
    if body is None:
        body = reader.optional("skeleton3d.json", generation="skeleton3d", consumed_fields=["fps", "players", "world_frame"])
    fps = _select_fps(tracks, ball_track, body)
    placement_tier, placement = _select_placement(reader, tracks, body, fps, warnings)

    arc_render = reader.optional("ball_arc_render.json", generation="ball_arc_render", consumed_fields=["samples", "segments", "solver_status"])
    if arc_render is not None:
        arc_generation, arc = "ball_arc_render", arc_render
    else:
        arc_generation = "ball_track_arc_solved"
        arc = reader.optional("ball_track_arc_solved.json", generation=arc_generation, consumed_fields=["frames", "anchors", "segments", "status", "kill_reasons"])
    contacts_refined = reader.optional("contact_windows_refined_v1.json", generation="contact_windows_refined_v1", consumed_fields=["events"])
    if contacts_refined is not None:
        contacts_generation, contact_payload = "contact_windows_refined_v1", contacts_refined
    else:
        contacts_generation = "contact_windows"
        contact_payload = reader.optional("contact_windows.json", generation=contacts_generation, consumed_fields=["events"])
    audio = reader.optional("audio_onsets_v2.json", generation="audio_onsets_v2", consumed_fields=["status", "not_gate_verified", "trusted_for_contact", "onsets"])
    paddle_pose = reader.optional("racket_pose.json", generation="racket_pose_gen2", consumed_fields=["players", "world_frame", "translation_unit"])
    paddle_hypotheses = reader.optional("racket_pose_hypotheses.json", generation="racket_hypotheses_gen2", consumed_fields=["players", "frames"])
    legacy_paddle = None
    if paddle_pose is None or paddle_hypotheses is None:
        legacy_paddle = reader.optional("racket_pose_estimate.json", generation="racket_pose_gen1", consumed_fields=["players", "render_only", "trust"])
    zones = reader.optional("court_zones.json", generation="court_zones", consumed_fields=["zones.court"])
    rallies = reader.optional("rally_spans.json", generation="rally_spans", consumed_fields=["spans"])
    net_plane = reader.optional("net_plane.json", generation="net_plane", consumed_fields=["plane", "endpoints"])
    virtual_world = reader.optional("virtual_world.json", generation="baseline", consumed_fields=["fps", "players", "ball", "paddles", "summary"])

    if calibration is not None:
        declared = calibration.get("coordinate_frame")
        if declared not in {None, "court_netcenter_z_up_m"}:
            raise ValueError(f"unsupported calibration coordinate_frame: {declared}")

    track_maps = _player_maps(tracks, fps)
    placement_maps = _player_maps(placement, fps)
    body_maps = _player_maps(body, fps)
    if track_maps and body_maps and sorted(track_maps) != sorted(body_maps):
        raise ValueError("tracks/BODY player identity mismatch")
    if track_maps and placement_maps and sorted(track_maps) != sorted(placement_maps):
        raise ValueError("tracks/placement player identity mismatch")
    raw_ball_map = _frame_map(ball_track, fps)
    selected_ball_map = _arc_map(arc, fps, arc_generation)
    supported_ball_ids = sorted(selected_ball_map)
    legacy_paddle_maps = _player_maps(legacy_paddle, fps)
    gen2_hypotheses: RacketPoseHypotheses | None = None
    gen2_states: dict[tuple[int, int], OneWorldPaddleState] = {}
    ambiguous_paddles = 0
    resolved_paddles = 0
    if paddle_pose is not None and paddle_hypotheses is not None:
        gen2_hypotheses = RacketPoseHypotheses.model_validate(paddle_hypotheses)
        if abs(float(gen2_hypotheses.fps) - fps) > 1e-9:
            raise ValueError("racket hypotheses fps mismatch")
        contact_frames = {
            int(event.get("frame", round(_finite(event.get("t")) * fps)))
            for event in (contact_payload or {}).get("events", [])
            if isinstance(event, Mapping) and event.get("type") == "contact"
        }
        for player in gen2_hypotheses.players:
            prepared: list[dict[str, Any]] = []
            lifted_by_frame: dict[int, dict[str, Any]] = {}
            for row in player.frames:
                if row.alt_pose is None:
                    continue
                frame_id = int(round(float(row.t) * fps))
                primary_world = lift_camera_pose_to_world(row.primary_pose.pose_se3.model_dump(), calibration or {}, input_unit=gen2_hypotheses.translation_unit)
                alt_world = lift_camera_pose_to_world(row.alt_pose.pose_se3.model_dump(), calibration or {}, input_unit=gen2_hypotheses.translation_unit)
                wrist_candidates = [
                    interpolate_wrist(body_maps.get(player.id, {}), frame_id, wrist_index, fps)
                    for wrist_index in WRIST_INDICES
                ]
                wrist_candidates = [candidate for candidate in wrist_candidates if candidate is not None]
                ball_row = selected_ball_map.get(frame_id)
                ball_world = ball_row.get("world_xyz") if ball_row else None
                hypotheses_for_solver = []
                for hypothesis_id, lifted in (("primary", primary_world), ("alt", alt_world)):
                    handle = lifted["t"]
                    wrist_energy = min((math.dist(handle, candidate[0]) / 0.15 for candidate in wrist_candidates), default=None)
                    contact_energy = math.dist(handle, ball_world) / 0.10 if frame_id in contact_frames and ball_world is not None else None
                    source_pose = row.primary_pose if hypothesis_id == "primary" else row.alt_pose
                    hypotheses_for_solver.append({"id": hypothesis_id, "wrist": wrist_energy, "contact": contact_energy, "momentum": None, "reprojection_error_px": source_pose.reprojection_error_px if source_pose else None})
                prepared.append({"hypotheses": hypotheses_for_solver, "transition": {}})
                lifted_by_frame[frame_id] = {"row": row, "primary": primary_world, "alt": alt_world}
            resolution = resolve_two_hypothesis_sequence(prepared)
            if prepared:
                ambiguous_paddles += 1
            if resolution["status"] == "resolved":
                resolved_paddles += 1
            for path_index, (frame_id, values) in enumerate(sorted(lifted_by_frame.items())):
                row = values["row"]
                winner = resolution["path"][path_index] if resolution["status"] == "resolved" else None
                pose = values[winner] if winner is not None else None
                display_choice = resolution["display_path"][path_index] if resolution.get("display_path") else None
                display_pose = values[display_choice] if display_choice is not None else None
                gen2_states[(player.id, frame_id)] = OneWorldPaddleState(
                    player_id=player.id,
                    status="resolved" if winner is not None else "unresolved",
                    pose_world=SE3.model_validate(pose) if pose is not None else None,
                    display_pose_world=SE3.model_validate(display_pose) if display_pose is not None else None,
                    display_tier="resolved" if winner is not None else "unresolved_best_evidence",
                    winning_hypothesis=winner,
                    retained_hypotheses=[
                        {"id": "primary", "pose_world": values["primary"], "reprojection_error_px": row.primary_pose.reprojection_error_px},
                        {"id": "alt", "pose_world": values["alt"], "reprojection_error_px": row.alt_pose.reprojection_error_px if row.alt_pose else None},
                    ],
                    score_components={"independent_term_count": float(sum(component is not None for prepared_row in prepared for hypothesis in prepared_row["hypotheses"][:1] for component in (hypothesis.get("wrist"), hypothesis.get("contact"), hypothesis.get("momentum"))))},
                    score_margin=resolution.get("margin"),
                    ambiguity_margin_px=row.ambiguity_margin_px,
                    confidence=max(float(row.primary_pose.frame_conf), float(row.alt_pose.frame_conf if row.alt_pose else 0.0)),
                    trust_band=band,
                    provenance=_provenance("OW-D-PADDLE-TWO-HYPOTHESIS", ["racket_pose", "racket_pose_hypotheses", "smpl_motion", arc_generation], {}, {}, discounts=["reprojection_carried_not_scored", str(resolution.get("display_rule"))], degraded=[] if winner is not None else [str(resolution.get("reason"))]),
                )
    repaired_player_frames: set[tuple[int, int]] = set()
    if repair_summary is not None and tracks is not None:
        recorded_track = repair_summary.get("input_paths", {}).get("tracks_path")
        recorded_hash = repair_summary.get("input_paths", {}).get("tracks_sha256")
        path_hash_matches = recorded_hash == sha256_file(reader.run_dir / "tracks.json") if recorded_hash else bool(recorded_track)
        if path_hash_matches:
            for marker in repair_summary.get("summary", {}).get("confidence_repairs", []):
                if marker.get("repaired"):
                    repaired_player_frames.add((int(marker["player_id"]), int(marker.get("frame_index", marker.get("frame", -1)))))
        else:
            warnings.append("repair_summary_tracks_hash_mismatch_ignored")

    frame_ids: set[int] = set(raw_ball_map) | set(selected_ball_map)
    frame_ids.update(frame_id for _, frame_id in gen2_states)
    for mapping in (track_maps, placement_maps, body_maps, legacy_paddle_maps):
        for rows in mapping.values():
            frame_ids.update(rows)
    if not frame_ids:
        frame_ids.add(0)

    frames: list[OneWorldFrame] = []
    placement_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    player_reproj_before: list[float] = []
    player_reproj_after: list[float] = []
    ball_reproj_before: list[float] = []
    ball_reproj_after: list[float] = []

    for frame_id in sorted(frame_ids):
        player_states: list[OneWorldPlayerState] = []
        missing: list[str] = []
        for player_id in sorted(set(track_maps) | set(placement_maps)):
            track_frame = track_maps.get(player_id, {}).get(frame_id)
            place_frame = placement_maps.get(player_id, {}).get(frame_id)
            body_frame = body_maps.get(player_id, {}).get(frame_id)
            if place_frame is None:
                missing.append(f"player:{player_id}:placement")
                missing_counts["player_placement"] += 1
                continue
            if placement_tier == "trackI_refined":
                refinement = place_frame.get("placement_trajectory_refinement", {})
                root = refinement.get("refined_transl_world", place_frame.get("transl_world"))
                covariance = refinement.get("covariance_m2", _identity_covariance(DEFAULT_TRACK_SIGMA_M))
                joints = place_frame.get("joints_world")
                confs = place_frame.get("joint_conf")
                nominal = {"trackI_refined": 1.0}
                effective = dict(nominal)
                discounts: list[str] = []
            else:
                pxy = place_frame.get("smoothed_world_xy", place_frame.get("world_xy"))
                if pxy is None:
                    missing.append(f"player:{player_id}:placement_xy")
                    missing_counts["player_placement_xy"] += 1
                    continue
                covariance2 = place_frame.get("covariance_m2")
                if covariance2 is None or len(covariance2) < 2:
                    covariance2 = [[DEFAULT_TRACK_SIGMA_M**2, 0.0], [0.0, DEFAULT_TRACK_SIGMA_M**2]]
                cp = np.asarray(covariance2, dtype=np.float64)[:2, :2]
                wp = np.linalg.pinv(cp)
                p = np.asarray(pxy, dtype=np.float64)
                nominal = {"placement": float(np.trace(wp)), "body_correlated": 0.0}
                effective = {"placement": float(np.trace(wp)), "body_correlated": 0.0}
                root_z = 0.0
                joints = None
                confs = None
                discounts = []
                if body_frame is not None and body_frame.get("transl_world") is not None:
                    b = np.asarray(body_frame["transl_world"], dtype=np.float64)[:2]
                    body_conf = statistics.median([_clip01(v) for v in body_frame.get("joint_conf", [0.0])[:17]])
                    sigma_b = 0.15 + 0.25 * (1.0 - body_conf)
                    wb = np.eye(2, dtype=np.float64) * (0.25 / sigma_b**2)
                    x = np.linalg.solve(wp + wb, wp @ p + wb @ b)
                    delta = x - p
                    norm = float(np.linalg.norm(delta))
                    if norm > 0.15:
                        x = p + delta * (0.15 / norm)
                    posterior2 = np.linalg.pinv(wp + wb)
                    nominal["body_correlated"] = float(np.trace(wb))
                    effective["body_correlated"] = float(np.trace(wb))
                    root_z = _finite(body_frame["transl_world"][2])
                    joints = [[float(v) for v in joint] for joint in body_frame.get("joints_world", [])]
                    shift = x - np.asarray(body_frame["transl_world"], dtype=np.float64)[:2]
                    for joint in joints:
                        joint[0] += float(shift[0])
                        joint[1] += float(shift[1])
                    confs = [_clip01(v) for v in body_frame.get("joint_conf", [])]
                else:
                    x = p
                    posterior2 = cp
                    discounts.append("missing_body_no_joint_display_fill")
                root = [float(x[0]), float(x[1]), root_z]
                covariance = [
                    [float(posterior2[0, 0]), float(posterior2[0, 1]), 0.0],
                    [float(posterior2[1, 0]), float(posterior2[1, 1]), 0.0],
                    [0.0, 0.0, 0.20**2],
                ]
            confidence = _clip01((track_frame or {}).get("conf", 0.0))
            if (player_id, frame_id) in repaired_player_frames:
                confidence *= 0.25
                discounts.append("player_repaired=0.25")
                effective = {key: value * 0.25 for key, value in effective.items()}
            elif repair_summary is None:
                discounts.append("legacy_repair_provenance_unavailable=0.80")
                effective = {key: value * 0.80 for key, value in effective.items()}
            state = OneWorldPlayerState(
                player_id=player_id,
                root_world=root,
                covariance_m2=covariance,
                joints_world=joints,
                joint_conf=confs,
                placement_tier=placement_tier,  # type: ignore[arg-type]
                confidence=confidence,
                trust_band=band,
                confidence_provenance=_confidence_provenance("preview", f"one_world_{placement_tier}", math.sqrt(max(float(covariance[0][0]), 0.0))),
                provenance=_provenance("OW-A-PLAYER-PLACEMENT", [placement_tier, "tracks", "smpl_motion"], nominal, effective, discounts=discounts, cap="||x-placement|| <= 0.15 m"),
            )
            player_states.append(state)
            placement_counts[placement_tier] += 1
            if track_frame is not None and track_frame.get("bbox") is not None:
                observed = [(float(track_frame["bbox"][0]) + float(track_frame["bbox"][2])) / 2.0, float(track_frame["bbox"][3])]
                before_world = [float(track_frame.get("world_xy", root[:2])[0]), float(track_frame.get("world_xy", root[:2])[1]), 0.0]
                before_error = _project_error(before_world, observed, calibration)
                after_error = _project_error(root, observed, calibration)
                if before_error is not None and after_error is not None:
                    kill_threshold = max(2.0, 0.10 * before_error)
                    if after_error - before_error > kill_threshold:
                        regression_kills.append(f"player:{player_id}:frame:{frame_id}:reprojection_regression")
                        state.root_world = before_world
                        state.joints_world = None
                        state.provenance.degraded_reasons.append("reprojection_regression")
                        after_error = before_error
                    player_reproj_before.append(before_error)
                    player_reproj_after.append(after_error)

        arc_frame = selected_ball_map.get(frame_id)
        raw_frame = raw_ball_map.get(frame_id)
        ball_state: OneWorldBallState | None = None
        if arc_frame is not None and arc_frame.get("world_xyz") is not None:
            confidence = _clip01(arc_frame.get("conf"))
            sigma = max(0.005, _finite(arc_frame.get("sigma_m"), DEFAULT_BALL_SIGMA_M))
            approx = bool(arc_frame.get("approx", False))
            discounts = ["render_only_arc=0.35"]
            reliability = 0.35
            if approx:
                discounts.append("approx=0.25")
                reliability *= 0.25
            ball_state = OneWorldBallState(
                world_xyz=[float(v) for v in arc_frame["world_xyz"]],
                covariance_m2=_identity_covariance(sigma),
                xy_observed_px=[float(v) for v in raw_frame["xy"]] if raw_frame and raw_frame.get("xy") is not None else None,
                confidence=confidence,
                source_generation=arc_generation,
                estimate_tier="arc_measured",
                ray_origin_world=None,
                ray_direction_world=None,
                altitude_unknown=False,
                approx=approx,
                render_only=True,
                not_for_detection_metrics=True,
                trust_band=band,
                confidence_provenance=_confidence_provenance("preview", arc_generation, sigma),
                provenance=_provenance("OW-B-BALL-PRESERVE", [arc_generation, "ball_track"], {"arc": confidence**2}, {"arc": confidence**2 * reliability}, discounts=discounts, cap=None),
            )
            observed = ball_state.xy_observed_px
            error = _project_error(ball_state.world_xyz, observed, calibration)
            if error is not None:
                ball_reproj_before.append(error)
                ball_reproj_after.append(error)
        else:
            insertion = bisect.bisect_left(supported_ball_ids, frame_id)
            lower = supported_ball_ids[insertion - 1] if insertion > 0 else None
            upper = supported_ball_ids[insertion] if insertion < len(supported_ball_ids) else None
            if lower is not None and upper is not None and lower < frame_id < upper and upper - lower <= int(round(0.5 * fps)):
                alpha = (frame_id - lower) / (upper - lower)
                p0 = np.asarray(selected_ball_map[lower]["world_xyz"], dtype=np.float64)
                p1 = np.asarray(selected_ball_map[upper]["world_xyz"], dtype=np.float64)
                dt = (upper - lower) / fps
                world = (1.0 - alpha) * p0 + alpha * p1
                world[2] += 0.5 * 9.80665 * alpha * (1.0 - alpha) * dt**2
                horizon = min(frame_id - lower, upper - frame_id)
                sigma = 0.20 + 0.04 * horizon
                ball_state = OneWorldBallState(world_xyz=world.tolist(), covariance_m2=_identity_covariance(sigma), xy_observed_px=[float(v) for v in raw_frame["xy"]] if raw_frame and raw_frame.get("xy") is not None else None, confidence=min(_clip01(selected_ball_map[lower].get("conf")), _clip01(selected_ball_map[upper].get("conf"))) * math.exp(-horizon / max(0.5 * fps, 1.0)), source_generation="ballistic_bridge", estimate_tier="physics_predicted", ray_origin_world=None, ray_direction_world=None, altitude_unknown=False, approx=True, render_only=True, not_for_detection_metrics=True, trust_band=_low_band("bounded ballistic bridge is display-only, never measured or metric-eligible"), confidence_provenance=ConfidenceProvenance(band="physics_predicted", display_band="low_confidence", predictor="bounded_ballistic_bridge", horizon_frames=horizon, predicted_sigma_m=sigma), provenance=_provenance("OW-E-BALL-CONTINUITY-BRIDGE", [arc_generation], {}, {}, discounts=["not_measured", "not_metric_eligible"], cap="adjacent support gap <=0.5s; no extrapolation"))
            elif raw_frame is not None and raw_frame.get("xy") is not None and bool(raw_frame.get("visible", False)) and _clip01(raw_frame.get("conf")) > 0.0 and calibration is not None:
                ray = ray_court_projection(raw_frame["xy"], calibration)
                if ray is not None:
                    origin, direction, intersection = ray
                    sigma = 3.0
                    ball_state = OneWorldBallState(world_xyz=intersection, covariance_m2=_identity_covariance(sigma), xy_observed_px=[float(v) for v in raw_frame["xy"]], confidence=_clip01(raw_frame.get("conf")) * 0.20, source_generation="camera_ray_court_projection", estimate_tier="ray_court_projection", ray_origin_world=origin, ray_direction_world=direction, altitude_unknown=True, approx=True, render_only=True, not_for_detection_metrics=True, trust_band=_low_band("2D camera ray court intersection; altitude unknown; display-only"), confidence_provenance=ConfidenceProvenance(band="model_estimated", display_band="low_confidence", predictor="typed_camera_ray_court_projection", horizon_frames=0, predicted_sigma_m=sigma), provenance=_provenance("OW-E-BALL-CONTINUITY-RAY", ["ball_track", "court_calibration"], {}, {}, discounts=["altitude_unknown", "not_measured", "not_metric_eligible"], cap=None))
            if ball_state is None:
                missing.append("ball:world_xyz")
                missing_counts["ball_world_xyz"] += 1

        paddles: list[OneWorldPaddleState] = []
        paddles.extend(state for (player_id, fid), state in sorted(gen2_states.items()) if fid == frame_id)
        for player_id, mapping in sorted(legacy_paddle_maps.items()):
            raw = mapping.get(frame_id)
            if raw is None:
                continue
            paddles.append(
                OneWorldPaddleState(
                    player_id=player_id,
                    status="unresolved_legacy_wrist_proxy",
                    pose_world=None,
                    display_pose_world=SE3.model_validate(raw.get("pose_se3")) if raw.get("pose_se3") is not None else None,
                    display_tier="unresolved_legacy_wrist_proxy",
                    winning_hypothesis=None,
                    retained_hypotheses=[{"id": "legacy_wrist_proxy", "pose_se3": raw.get("pose_se3"), "reprojection_error_px": raw.get("reprojection_error_px")}],
                    score_components={},
                    score_margin=None,
                    ambiguity_margin_px=None,
                    confidence=_clip01(raw.get("conf")),
                    trust_band=_normalize_band(raw.get("trust_band"), fallback_reason="legacy wrist proxy is unresolved"),
                    provenance=_provenance("OW-D-PADDLE-LEGACY", ["racket_pose_estimate"], {}, {}, discounts=["legacy_wrist_proxy_never_resolved"], degraded=["missing_two_ippe_hypotheses"]),
                )
            )
        paddle_ids = {state.player_id for state in paddles}
        for player_id in sorted(set(track_maps) - paddle_ids):
            missing.append(f"paddle:{player_id}:pose")
            missing_counts["paddle_pose"] += 1
        frames.append(OneWorldFrame(frame_idx=frame_id, t=frame_id / fps, players=player_states, ball=ball_state, paddles=paddles, missing=sorted(missing)))

    body_wrist_maps = body_maps
    contact_rows: list[OneWorldContactRefinement] = []
    frame_output_map = {frame.frame_idx: frame for frame in frames}
    for event_index, event in enumerate((contact_payload or {}).get("events", [])):
        if not isinstance(event, Mapping) or event.get("type") != "contact":
            continue
        frame_id = int(event.get("frame", round(_finite(event.get("t")) * fps)))
        raw_player = int(event["player_id"]) if event.get("player_id") is not None else None
        ball_frame = selected_ball_map.get(frame_id)
        ball_world = [float(v) for v in ball_frame["world_xyz"]] if ball_frame and ball_frame.get("world_xyz") is not None else None
        ball_conf = _clip01((ball_frame or {}).get("conf"))
        ball_sigma = max(0.005, _finite((ball_frame or {}).get("sigma_m"), DEFAULT_BALL_SIGMA_M))
        event_conf = _clip01(event.get("confidence"))
        audio_multiplier, audio_discounts = _audio_multiplier(event, audio, fps)
        evidence_hypotheses: list[ContactHypothesisEvidence] = []
        likelihoods: dict[str, list[float]] = {}
        wrist_cache: dict[tuple[int, int], tuple[list[float], float, float, list[str]]] = {}
        if ball_world is not None:
            for player_id in sorted(body_wrist_maps):
                player_values: list[float] = []
                for wrist_index in WRIST_INDICES:
                    wrist = interpolate_wrist(body_wrist_maps[player_id], frame_id, wrist_index, fps)
                    if wrist is None:
                        player_values.append(0.0)
                        continue
                    wrist_cache[(player_id, wrist_index)] = wrist
                    wrist_world, wrist_conf, wrist_sigma, wrist_discounts = wrist
                    distance = math.dist(ball_world, wrist_world)
                    variance = ball_sigma**2 + wrist_sigma**2 + (WRIST_RADIUS_M + BALL_RADIUS_M) ** 2 / 3.0
                    mahalanobis = min(distance**2 / max(variance, FLOAT_EPS), 25.0)
                    ball_term = ball_conf**2
                    wrist_term = wrist_conf**2
                    event_term = event_conf**2
                    marker_reliability = 0.35
                    discounts = ["render_only_arc=0.35", *wrist_discounts, *audio_discounts]
                    if bool((ball_frame or {}).get("approx", False)):
                        marker_reliability *= 0.25
                        discounts.append("approx=0.25")
                    if wrist_discounts:
                        marker_reliability *= 0.25
                    declared_multiplier = 1.25 if raw_player == player_id else 1.0
                    combined = math.exp(-0.5 * mahalanobis) * ball_term * wrist_term * event_term * marker_reliability * declared_multiplier * audio_multiplier
                    player_values.append(combined)
                    evidence_hypotheses.append(ContactHypothesisEvidence(player_id=player_id, wrist_index=wrist_index, center_distance_m=distance, ball_term=ball_term, wrist_term=wrist_term, event_confidence_term=event_term, marker_reliability=marker_reliability, declared_hitter_multiplier=declared_multiplier, audio_bounded_multiplier=audio_multiplier, combined_likelihood=combined, discounts=discounts))
                likelihoods[str(player_id)] = player_values
        player_scores = {int(pid): sum(values) for pid, values in likelihoods.items()}
        total = 0.05 + sum(player_scores.values())
        probabilities = {pid: score / total for pid, score in player_scores.items()}
        ranked = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
        best_pid = ranked[0][0] if ranked else None
        best_p = ranked[0][1] if ranked else 0.0
        second_p = ranked[1][1] if len(ranked) > 1 else 0.0
        nearest_for_best = min((row for row in evidence_hypotheses if row.player_id == best_pid), key=lambda row: (row.center_distance_m, row.wrist_index), default=None)
        status = "unsupported"
        hitter_band: Literal["resolved", "too_close_to_call", "unsupported"] = "unsupported"
        hitter_id: int | None = None
        refined_ball: list[float] | None = None
        raw_residual: float | None = None
        refined_residual: float | None = None
        displacement: float | None = None
        degraded: list[str] = []
        if ball_world is None:
            degraded.append("missing_ball_track_support")
        elif not wrist_cache:
            degraded.append("missing_wrist_support")
        elif nearest_for_best is None or nearest_for_best.center_distance_m > CONTACT_MAX_RAW_DISTANCE_M:
            degraded.append("no_player_wrist_within_1.2m")
        elif best_p < 0.55 or best_p - second_p < 0.15:
            status = "too_close_to_call"
            hitter_band = "too_close_to_call"
            degraded.append("hitter_probability_or_margin_below_threshold")
        else:
            hitter_id = best_pid
            hitter_band = "resolved"
            status = "refined"
            wrist_row = max((row for row in evidence_hypotheses if row.player_id == best_pid), key=lambda row: (row.combined_likelihood, -row.wrist_index))
            wrist_world, wrist_conf, wrist_sigma, _ = wrist_cache[(best_pid, wrist_row.wrist_index)]
            raw_residual = max(0.0, math.dist(ball_world, wrist_world) - WRIST_RADIUS_M - BALL_RADIUS_M)
            ball_weight = ball_conf**2 / max(ball_sigma**2, 0.005**2)
            joint_weight = (ball_conf * wrist_conf * event_conf) ** 2 * wrist_row.marker_reliability / max(ball_sigma**2 + wrist_sigma**2, 0.005**2)
            combined = (ball_weight * np.asarray(ball_world) + joint_weight * np.asarray(wrist_world)) / max(ball_weight + joint_weight, FLOAT_EPS)
            delta = combined - np.asarray(ball_world)
            influence = _huber_influence(float(np.linalg.norm(delta)) / max(math.sqrt(ball_sigma**2 + wrist_sigma**2), FLOAT_EPS))
            delta *= influence
            cap = min(0.35, 2.0 * ball_sigma)
            if float(np.linalg.norm(delta)) > cap:
                delta *= cap / float(np.linalg.norm(delta))
            candidate = (np.asarray(ball_world) + delta).tolist()
            observed_xy = (raw_ball_map.get(frame_id) or {}).get("xy")
            before_px = _project_error(ball_world, observed_xy, calibration)
            after_px = _project_error(candidate, observed_xy, calibration)
            if before_px is not None and after_px is not None and after_px - before_px > max(2.0, 0.10 * before_px):
                hitter_id = None
                hitter_band = "unsupported"
                status = "unsupported"
                degraded.append("reprojection_regression")
                regression_kills.append(f"contact:{event_index}:reprojection_regression")
            else:
                refined_ball = candidate
                displacement = math.dist(ball_world, refined_ball)
                refined_residual = max(0.0, math.dist(refined_ball, wrist_world) - WRIST_RADIUS_M - BALL_RADIUS_M)
        if raw_residual is None and nearest_for_best is not None:
            raw_residual = max(0.0, nearest_for_best.center_distance_m - WRIST_RADIUS_M - BALL_RADIUS_M)
        contact_rows.append(
            OneWorldContactRefinement(
                event_index=event_index,
                frame=frame_id,
                t=_finite(event.get("t"), frame_id / fps),
                status=status,
                raw_player_id=raw_player,
                hitter_id=hitter_id,
                hitter_confidence=best_p,
                hitter_band=hitter_band,
                per_player_wrist_likelihoods=likelihoods,
                contact_evidence_vector=ContactEvidenceVector(upstream_sources=dict(event.get("sources", {})), visual_event_exists=True, ball_track_supported=ball_world is not None, wrist_candidates_supported=bool(wrist_cache), audio_bounded_multiplier=audio_multiplier, hypotheses=evidence_hypotheses, null_likelihood=0.05),
                raw_ball_world=ball_world,
                refined_ball_world=refined_ball,
                raw_wrist_volume_residual_m=raw_residual,
                refined_wrist_volume_residual_m=refined_residual,
                displacement_m=displacement,
                confidence=best_p if hitter_id is not None else 0.0,
                trust_band=band,
                provenance=_provenance("OW-C-CONTACT-COLOCATION", [contacts_generation, arc_generation, "smpl_motion", "audio_onsets_v2"], {"null": 0.05}, {"null": 0.05, "audio_multiplier": audio_multiplier}, discounts=audio_discounts, cap="raw distance <=1.2m; displacement <=min(0.35m,2*sigma_ball)", degraded=degraded),
            )
        )

    court_polygon = ((zones or {}).get("zones", {}).get("court") or [])
    bounce_rows: list[OneWorldBounceRefinement] = []
    bounce_events = list(ball_track.get("bounces", []))
    if not bounce_events and arc_generation == "ball_track_arc_solved":
        bounce_events = [
            {
                **anchor,
                "confidence": 1.0 if anchor.get("status") in {"used", "accepted", "fixed"} else 0.5,
                "p_bounce": 1.0,
                "source": anchor.get("source", "ball_arc_bounce_anchor"),
                "uncertainty_m": anchor.get("sigma_m"),
            }
            for anchor in (arc or {}).get("anchors", [])
            if isinstance(anchor, Mapping) and anchor.get("kind") == "bounce"
        ]
    for event_index, event in enumerate(bounce_events):
        if not isinstance(event, Mapping):
            continue
        frame_id = int(event.get("frame", round(_finite(event.get("t")) * fps)))
        arc_frame = selected_ball_map.get(frame_id)
        anchor_world = event.get("world_xyz")
        raw_world = [float(v) for v in anchor_world] if anchor_world is not None else ([float(v) for v in arc_frame["world_xyz"]] if arc_frame and arc_frame.get("world_xyz") is not None else None)
        derived_from_world_xy = False
        if raw_world is None and event.get("world_xy") is not None:
            # A 2-D court-plane bounce proposal may seed only a high-uncertainty
            # preview prior.  Its z is deliberately offset so the soft solver
            # cannot masquerade as an exact plane observation.
            raw_world = [float(event["world_xy"][0]), float(event["world_xy"][1]), BALL_RADIUS_M + 0.10]
            derived_from_world_xy = True
        status = "unsupported"
        refined = None
        before = after = None
        w_surface = w_ball = 0.0
        degraded: list[str] = []
        if raw_world is not None:
            ball_conf = _clip01((arc_frame or {}).get("conf", event.get("confidence", event.get("p_bounce", 0.0))))
            bounce_conf = _clip01(event.get("confidence", event.get("p_bounce", 0.0)))
            sigma_ball = 0.20 if derived_from_world_xy else max(0.005, _finite((arc_frame or {}).get("sigma_m"), DEFAULT_BALL_SIGMA_M))
            metric_conf = str((calibration or {}).get("metric_confidence", "missing"))
            sigma_cal = {"high": 0.02, "med": 0.05, "low": 0.12}.get(metric_conf, 0.20)
            if band.badge != "verified":
                sigma_cal = max(sigma_cal, 0.12)
            calibration_multiplier = 0.30 if band.badge == "preview" else 0.20 if band.badge == "low_confidence" else 1.0
            sigma_event = max(_finite(event.get("uncertainty_m"), 0.12), 0.05)
            refined, before, after, w_surface, w_ball = soft_surface_refinement(raw_world, plane_point=[0.0, 0.0, 0.0], plane_normal=[0.0, 0.0, 1.0], ball_confidence=ball_conf, bounce_confidence=bounce_conf, sigma_ball_m=sigma_ball, sigma_cal_m=sigma_cal, sigma_event_m=sigma_event, calibration_multiplier=calibration_multiplier)
            status = "soft_refined_preview"
            if derived_from_world_xy:
                degraded.append("world_xy_only_high_uncertainty_preview_lift")
        else:
            degraded.append("missing_ball_world_support")
        out_of_bounds = bool(raw_world is not None and court_polygon and not _inside_polygon(raw_world[:2], court_polygon))
        bounce_rows.append(OneWorldBounceRefinement(event_index=event_index, frame=frame_id, t=_finite(event.get("t"), frame_id / fps), status=status, raw_ball_world=raw_world, refined_ball_world=refined, signed_plane_residual_before_m=before, signed_plane_residual_after_m=after, out_of_court_bounds=out_of_bounds, confidence=_clip01(event.get("confidence", event.get("p_bounce", 0.0))), trust_band=band, provenance=_provenance("OW-B-BOUNCE-SOFT-PRIOR", ["ball_track", arc_generation, "court_calibration", "court_zones"], {"surface": w_surface, "ball": w_ball}, {"surface": w_surface, "ball": w_ball}, discounts=["calibration_preview_weight"] + (["world_xy_only_preview"] if derived_from_world_xy else []), cap="||delta|| <= min(0.15m,2*sigma_ball); never assign z=r_b", degraded=degraded)))

    viewer_events: list[OneWorldEvent] = [
        OneWorldEvent(event_index=row.event_index, type="paddle_contact", t=row.t, frame=row.frame, world_location_raw=row.raw_ball_world, world_location_refined=row.refined_ball_world, hitter_id=row.hitter_id, confidence=row.confidence, trust_band=row.trust_band, evidence=row.contact_evidence_vector, provenance=row.provenance)
        for row in contact_rows
    ]
    viewer_events.extend(
        OneWorldEvent(event_index=row.event_index, type="floor_bounce", t=row.t, frame=row.frame, world_location_raw=row.raw_ball_world, world_location_refined=row.refined_ball_world, hitter_id=None, confidence=row.confidence, trust_band=row.trust_band, evidence={"surface_nominal_weights": row.provenance.nominal_weights, "surface_effective_weights": row.provenance.effective_weights, "soft_constraint": True}, provenance=row.provenance)
        for row in bounce_rows
    )
    for source_index, event in enumerate((contact_payload or {}).get("events", [])):
        raw_type = str(event.get("type", "")) if isinstance(event, Mapping) else ""
        if raw_type not in {"into_net", "net_contact", "net_cross"}:
            continue
        frame_id = int(event.get("frame", round(_finite(event.get("t")) * fps)))
        ball_row = selected_ball_map.get(frame_id)
        raw_world = [float(v) for v in ball_row["world_xyz"]] if ball_row and ball_row.get("world_xyz") is not None else None
        refined_world = raw_world
        nominal: dict[str, float] = {}
        effective: dict[str, float] = {}
        degraded: list[str] = []
        event_type: Literal["net_contact", "net_cross"] = "net_cross" if raw_type == "net_cross" else "net_contact"
        if event_type == "net_contact" and raw_world is not None and net_plane is not None:
            plane = net_plane.get("plane", {})
            refined_world, _, _, ws, wb = soft_surface_refinement(raw_world, plane_point=plane.get("point", [0.0, 0.0, 0.0]), plane_normal=plane.get("normal", [0.0, 1.0, 0.0]), ball_confidence=_clip01(ball_row.get("conf")), bounce_confidence=_clip01(event.get("confidence")), sigma_ball_m=max(0.005, _finite(ball_row.get("sigma_m"), DEFAULT_BALL_SIGMA_M)), sigma_cal_m=0.12, sigma_event_m=0.12, calibration_multiplier=0.30)
            nominal = effective = {"surface": ws, "ball": wb}
        elif event_type == "net_cross":
            degraded.append("net_cross_time_residual_only_no_positional_pull")
        else:
            degraded.append("missing_net_or_ball_world_support")
        provenance = _provenance("OW-B-NET-CROSS-NO-PULL" if event_type == "net_cross" else "OW-B-NET-CONTACT-SOFT-PRIOR", [contacts_generation, arc_generation, "net_plane"], nominal, effective, discounts=["preview_surface_prior"], cap="no positional pull" if event_type == "net_cross" else "finite soft pull; never snap", degraded=degraded)
        viewer_events.append(OneWorldEvent(event_index=source_index, type=event_type, t=_finite(event.get("t"), frame_id / fps), frame=frame_id, world_location_raw=raw_world, world_location_refined=refined_world, hitter_id=None, confidence=_clip01(event.get("confidence")), trust_band=band, evidence={"sources": dict(event.get("sources", {})), "time_residual_s": _finite(event.get("t"), frame_id / fps) - frame_id / fps, "surface_nominal_weights": nominal, "surface_effective_weights": effective}, provenance=provenance))
    viewer_events.sort(key=lambda row: (row.frame, row.type, row.event_index))

    # Frozen half-open rally coverage.
    spans = (rallies or {}).get("spans", (tracks or {}).get("rally_spans", []))
    rally_ids: set[int] = set()
    for span in spans:
        start = int(math.ceil(_finite(span.get("t0")) * fps - 1e-9))
        end = int(math.ceil(_finite(span.get("t1")) * fps - 1e-9))
        rally_ids.update(range(start, end))
    expected_players = sorted(track_maps)
    qualifying = 0
    qualifying_with_predicted = 0
    for frame_id in sorted(rally_ids):
        players_ok = bool(expected_players) and all(frame_id in placement_maps.get(pid, {}) and _clip01(track_maps.get(pid, {}).get(frame_id, {}).get("conf")) >= 0.5 for pid in expected_players)
        ball_ok = frame_id in selected_ball_map and _clip01(selected_ball_map[frame_id].get("conf")) >= 0.5
        qualifying += int(players_ok and ball_ok)
        output_ball = frame_output_map.get(frame_id).ball if frame_output_map.get(frame_id) is not None else None
        qualifying_with_predicted += int(players_ok and output_ball is not None and output_ball.confidence >= 0.5)

    supported_contact_values = [row.refined_wrist_volume_residual_m for row in contact_rows if row.hitter_band == "resolved" and row.refined_wrist_volume_residual_m is not None]
    bounce_before = [abs(row.signed_plane_residual_before_m) for row in bounce_rows if row.signed_plane_residual_before_m is not None]
    bounce_after = [abs(row.signed_plane_residual_after_m) for row in bounce_rows if row.signed_plane_residual_after_m is not None]
    reprojection = {
        "ball": {"baseline": _distribution(ball_reproj_before), "fused": _distribution(ball_reproj_after)},
        "player": {"baseline": _distribution(player_reproj_before), "fused": _distribution(player_reproj_after)},
        "paddle": {"baseline": _distribution([]), "fused": _distribution([]), "unsupported_reason": "no true corners in target run"},
    }
    summary = OneWorldSummary(
        placement_tier_counts=dict(sorted(placement_counts.items())),
        missing_counts=dict(sorted(missing_counts.items())),
        ball_contact_distance_m={**_distribution([float(v) for v in supported_contact_values]), "abstained_count": sum(row.hitter_band != "resolved" for row in contact_rows)},
        bounce_plane_residual_m={"count": len(bounce_after), "baseline_median": statistics.median(bounce_before) if bounce_before else None, "baseline_p90_nearest_rank": _nearest_rank(bounce_before, 0.90), "median": statistics.median(bounce_after) if bounce_after else None, "p90_nearest_rank": _nearest_rank(bounce_after, 0.90)},
        world_coverage={"confidence_threshold": 0.5, "rally_frame_count": len(rally_ids), "complete_frame_count": qualifying, "coverage_fraction": qualifying / len(rally_ids) if rally_ids else 0.0, "coverage_measured": {"complete_frame_count": qualifying, "coverage_fraction": qualifying / len(rally_ids) if rally_ids else 0.0}, "coverage_with_predicted": {"complete_frame_count": qualifying_with_predicted, "coverage_fraction": qualifying_with_predicted / len(rally_ids) if rally_ids else 0.0}, "ball_estimate_tier_counts": dict(sorted(Counter(frame.ball.estimate_tier for frame in frames if frame.ball is not None).items()))},
        paddle_resolution={"ambiguous_denominator": ambiguous_paddles, "resolved_count": resolved_paddles, "resolved_fraction": resolved_paddles / ambiguous_paddles if ambiguous_paddles else 0.0, "unsupported_legacy_wrist_proxy_count": sum(state.status == "unresolved_legacy_wrist_proxy" for frame in frames for state in frame.paddles)},
        reprojection_consistency=reprojection,
        regression_kills=sorted(set(regression_kills)),
        warnings=sorted(set(warnings + ["VERIFIED=0", "preview_band", "render_only", "not_for_detection_metrics", "not_for_training"])),
    )
    output = OneWorldV1(schema_version=1, artifact_type="racketsport_one_world_v1", world_frame="court_Z0", coordinate_space="world_court_netcenter_z_up_m", fps=fps, VERIFIED=0, preview_only=True, render_only=True, not_for_detection_metrics=True, not_for_training=True, raw_inputs_mutated=False, inputs=reader.refs, frames=frames, contacts=contact_rows, bounces=bounce_rows, events=viewer_events, summary=summary, trust_band=band)
    reader.assert_immutable()
    return output


def build_metrics(run_dir: Path, fused_path: Path) -> OneWorldMetrics:
    """Apply the frozen §5 reporting convention to baseline and fused data."""

    from runs.lanes.oneworld_design_20260716.baseline_probe import build_report as build_baseline

    fused = OneWorldV1.model_validate_json(fused_path.read_text(encoding="utf-8"))
    baseline = build_baseline(run_dir.resolve(), 0.5)["metrics"]
    supported = [float(row.refined_wrist_volume_residual_m) for row in fused.contacts if row.hitter_band == "resolved" and row.refined_wrist_volume_residual_m is not None]
    abstention_reasons = Counter(reason for row in fused.contacts if row.hitter_band != "resolved" for reason in row.provenance.degraded_reasons)
    event10 = next((row for row in fused.contacts if row.event_index == 10), None)
    metric_bounces = [row for row in fused.bounces if "world_xy_only_high_uncertainty_preview_lift" not in row.provenance.degraded_reasons]
    b_before = [abs(float(row.signed_plane_residual_before_m)) for row in metric_bounces if row.signed_plane_residual_before_m is not None]
    b_after = [abs(float(row.signed_plane_residual_after_m)) for row in metric_bounces if row.signed_plane_residual_after_m is not None]
    metrics = {
        "M1_ball_at_contact_wrist_volume": {
            "baseline": baseline["ball_at_contact_to_hitter_wrist"]["wrist_volume_residual_m"],
            "baseline_count": baseline["ball_at_contact_to_hitter_wrist"]["computable_event_count"],
            "fused_supported": _distribution(supported),
            "abstained_count": sum(row.hitter_band != "resolved" for row in fused.contacts),
            "abstention_reasons": dict(sorted(abstention_reasons.items())),
            "event_index_10": event10.model_dump(mode="json") if event10 else None,
        },
        "M2_bounce_to_plane": {"baseline": _distribution(b_before), "fused": _distribution(b_after), "soft_constraint": True},
        "M3_world_coverage_at_0_5": {"baseline": baseline["world_coverage"], "fused": fused.summary.world_coverage, "metric_eligibility": "coverage_measured only; predicted tiers are display-only"},
        "M4_paddle_ambiguity": fused.summary.paddle_resolution,
        "M5_reprojection_consistency": fused.summary.reprojection_consistency,
        "hitter_inference_audit": [
            {
                "event_index": row.event_index,
                "frame": row.frame,
                "declared_player_id": row.raw_player_id,
                "fused_hitter_id": row.hitter_id,
                "confidence": row.hitter_confidence,
                "band": row.hitter_band,
                "per_player_wrist_likelihoods": row.per_player_wrist_likelihoods,
            }
            for row in fused.contacts
        ],
    }
    return OneWorldMetrics(schema_version=1, artifact_type="racketsport_one_world_v1_metrics", VERIFIED=0, preview_only=True, render_only=True, not_for_detection_metrics=True, not_for_training=True, baseline_source=str((run_dir / "virtual_world.json").resolve()), fused_source=str(fused_path.resolve()), metrics=metrics)


def validate_one_world(artifact_path: Path, run_dir: Path) -> OneWorldValidation:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}
    try:
        artifact = OneWorldV1.model_validate_json(artifact_path.read_text(encoding="utf-8"))
        checks["schema"] = True
    except Exception as exc:
        return OneWorldValidation(schema_version=1, artifact_type="racketsport_one_world_v1_validation", VERIFIED=0, preview_only=True, render_only=True, not_for_detection_metrics=True, not_for_training=True, valid=False, checks={"schema": False}, errors=[str(exc)], warnings=[])
    checks["preview_policy"] = artifact.VERIFIED == 0 and artifact.preview_only and artifact.render_only and artifact.not_for_detection_metrics and artifact.not_for_training and not artifact.raw_inputs_mutated
    if not checks["preview_policy"]:
        errors.append("preview policy invariant failed")
    hash_ok = True
    for ref in artifact.inputs:
        if ref.missing_reason is not None:
            continue
        path = run_dir / ref.path
        if not path.exists() or sha256_file(path) != ref.sha256:
            hash_ok = False
            errors.append(f"input hash mismatch: {ref.path}")
    checks["input_sha256s"] = hash_ok
    checks["band_inheritance"] = artifact.trust_band.badge != "verified" and all(frame.ball is None or frame.ball.trust_band.badge != "verified" for frame in artifact.frames) and all(player.trust_band.badge != "verified" for frame in artifact.frames for player in frame.players)
    if not checks["band_inheritance"]:
        errors.append("trust band was upgraded")
    checks["absence_semantics"] = all((frame.ball is not None) or ("ball:world_xyz" in frame.missing) for frame in artifact.frames)
    if not checks["absence_semantics"]:
        errors.append("missing ball lacks explicit absence sentinel")
    checks["reprojection_kills"] = all(row.refined_ball_world is None for row in artifact.contacts if "reprojection_regression" in row.provenance.degraded_reasons)
    if not checks["reprojection_kills"]:
        errors.append("reprojection-regressed refinement survived")
    checks["no_snap_policy"] = all("never assign z=r_b" in (row.provenance.correction_cap or "") for row in artifact.bounces if row.refined_ball_world is not None)
    checks["raw_inputs_mutated"] = not artifact.raw_inputs_mutated
    return OneWorldValidation(schema_version=1, artifact_type="racketsport_one_world_v1_validation", VERIFIED=0, preview_only=True, render_only=True, not_for_detection_metrics=True, not_for_training=True, valid=not errors and all(checks.values()), checks=checks, errors=errors, warnings=warnings)


__all__ = [
    "BALL_RADIUS_M",
    "CONTACT_MAX_RAW_DISTANCE_M",
    "ContactEvidenceVector",
    "OneWorldContactRefinement",
    "OneWorldMetrics",
    "OneWorldV1",
    "OneWorldValidation",
    "build_metrics",
    "build_one_world",
    "canonical_json",
    "interpolate_wrist",
    "lift_camera_pose_to_world",
    "resolve_two_hypothesis_sequence",
    "sha256_file",
    "soft_surface_refinement",
    "validate_one_world",
]
