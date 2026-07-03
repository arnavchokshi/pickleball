"""CAL-3 no-tap court-keypoint evaluation plumbing.

The report generated here is an evaluator readiness artifact, not CAL-3 proof.
It validates the checkpoint/evidence inputs locally and can run the trained
heatmap checkpoint on H100 when CUDA is available.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from threed.racketsport.court_calibration import project_image_points_to_world, project_planar_points
from threed.racketsport.court_keypoint_labels import VISIBLE, load_partial_court_keypoints
from threed.racketsport.court_keypoint_net import (
    PICKLEBALL_KEYPOINTS,
    PICKLEBALL_KEYPOINT_BY_NAME,
    court_keypoint_probabilities,
    decode_subpixel_heatmap,
    keypoints_to_solvepnp_correspondences,
    make_court_keypoint_heatmap_model,
    validate_heatmap_prediction_payload,
)
from threed.racketsport.court_partial_geometry import NET_TOP_KEYPOINT_NAMES, fit_visible_floor_homography
from threed.racketsport.schemas import CourtCalibration, CourtLineEvidence, validate_artifact_file


DEFAULT_ACCEPTED_CLIPS: tuple[str, ...] = (
    "burlington_gold_0300_low_steep_corner",
    "wolverine_mixed_0200_mid_steep_corner",
    "outdoor_webcam_iynbd_1500_long_high_baseline",
    "indoor_doubles_fwuks_0500_long_mid_baseline",
)
DEFAULT_COURT_RETIRED_CLIPS: frozenset[str] = frozenset({"burlington_gold_0300_low_steep_corner"})
DEFAULT_TOP_NET_MATCH_THRESHOLD_PX = 12.0
DEFAULT_THRESHOLD_SWEEP: tuple[float, ...] = (0.5, 0.12, 0.1, 0.08, 0.05, 0.02)
CALIBRATION_REPROJECTION_P95_GATE_PX = 80.0
TEMPORAL_JITTER_P95_GATE_PX = 50.0
LINE_TRIPLET_RESIDUAL_GATE_PX = 30.0
WORLD_ERROR_P95_GATE_M = 3.0
BORDER_KEYPOINT_RATIO_GATE = 0.15
BORDER_MARGIN_RATIO = 0.02
PARTIAL_VISIBLE_FLOOR_MEDIAN_GATE_PX = 15.0
PARTIAL_VISIBLE_FLOOR_P95_GATE_PX = 30.0
PARTIAL_VISIBLE_CORNER_MEDIAN_GATE_PX = 20.0
PARTIAL_HIGH_CONFIDENCE_ERROR_GATE_PX = 30.0
PARTIAL_HIGH_CONFIDENCE_THRESHOLD = 0.70
PARTIAL_TEMPORAL_STABILITY_GATE_PX = 10.0
OUTER_CORNER_NAMES: tuple[str, str, str, str] = (
    "near_left_corner",
    "near_right_corner",
    "far_right_corner",
    "far_left_corner",
)


class TopNetPoint(BaseModel):
    model_config = ConfigDict(extra="allow")

    x: float
    y: float


class TopNetReviewPoints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    artifact_type: Literal["pickleball_human_top_net_review_points"]
    clip: str
    click_coordinate_space: dict[str, float | int]
    click_points: dict[str, TopNetPoint]
    evidence_scale: dict[str, float | int] | None = None
    evidence_points: dict[str, list[float]] | None = None
    source_review_input: str | None = None
    notes: str = ""


class TopNetReviewMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    best_coordinate_space: Literal["click_points", "evidence_points", "missing"]
    max_endpoint_delta_px: float | None = Field(default=None, ge=0.0)
    threshold_px: float = Field(ge=0.0)
    notes: list[str] = Field(default_factory=list)


class MetricSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    mean: float = Field(ge=0.0)
    median: float = Field(ge=0.0)
    p95: float = Field(ge=0.0)
    max: float = Field(ge=0.0)


class BorderSanity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    near_border_keypoint_count: int = Field(ge=0)
    confident_keypoint_count: int = Field(ge=0)
    near_border_keypoint_ratio: float = Field(ge=0.0)
    margin_px: float = Field(ge=0.0)


class LineCornerConsistency(BaseModel):
    model_config = ConfigDict(extra="forbid")

    triplet_count: int = Field(ge=0)
    triplet_pass_count: int = Field(ge=0)
    triplet_pass_ratio: float = Field(ge=0.0, le=1.0)
    triplet_residual_px: MetricSummary | None = None
    corner_quad_frame_count: int = Field(ge=0)
    corner_quad_valid_count: int = Field(ge=0)
    corner_quad_valid_ratio: float = Field(ge=0.0, le=1.0)


class CourtZoneSanity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    projected_keypoint_count: int = Field(ge=0)
    outside_court_count: int = Field(ge=0)
    outside_court_ratio: float = Field(ge=0.0)
    world_error_m: MetricSummary | None = None


class CourtKeypointThresholdValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(ge=0.0, le=1.0)
    frame_count: int = Field(ge=0)
    total_confident_keypoints: int = Field(ge=0)
    frame_min_count: int = Field(ge=0)
    frame_max_count: int = Field(ge=0)
    frames_with_solvepnp_min4: int = Field(ge=0)
    calibration_reprojection_error_px: MetricSummary | None = None
    temporal_jitter_px: MetricSummary | None = None
    line_corner_consistency: LineCornerConsistency
    court_zone_sanity: CourtZoneSanity
    border_sanity: BorderSanity
    diagnostic_gate_passed: bool
    blockers: list[str] = Field(default_factory=list)


class CourtKeypointClipValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip: str
    thresholds: list[CourtKeypointThresholdValidation]
    best_threshold: float | None = None
    best_threshold_gate_passed: bool = False
    best_threshold_blockers: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PartialVisibleGate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    partial_label_path: str
    proposal_path: str | None = None
    visible_keypoint_count: int = Field(ge=0)
    missing_keypoints: list[str] = Field(default_factory=list)
    visible_floor_fit_verified: bool
    visible_floor_fit_error_px: MetricSummary | None = None
    all_visible_error_px: MetricSummary | None = None
    floor_visible_error_px: MetricSummary | None = None
    visible_corner_error_px: MetricSummary | None = None
    net_top_visible_error_px: MetricSummary | None = None
    high_confidence_over_30px_count: int = Field(default=0, ge=0)
    self_verification_promotion_allowed: bool | None = None
    temporal_stability_px: MetricSummary | None = None
    gate_passed: bool
    blockers: list[str] = Field(default_factory=list)


class CourtKeypointClipReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip: str
    status: Literal["ready_for_h100", "ran_not_verified", "blocked", "skipped_retired_for_court"]
    run_dir: str
    input_video: str | None = None
    court_line_evidence: str | None = None
    top_net_review_points: str | None = None
    court_calibration: str | None = None
    court_zones: str | None = None
    net_plane: str | None = None
    retired_for_court: bool = False
    evidence_ready: bool = False
    accepted_line_ids: list[str] = Field(default_factory=list)
    missing_required_line_ids: list[str] = Field(default_factory=list)
    missing_required_net_ids: list[str] = Field(default_factory=list)
    top_net_review_match: TopNetReviewMatch
    missing_artifacts: list[str] = Field(default_factory=list)
    planned_frame_indexes: list[int] = Field(default_factory=list)
    prediction_artifact: str | None = None
    prediction_frame_count: int = Field(default=0, ge=0)
    max_confident_keypoint_count: int = Field(default=0, ge=0)
    max_solvepnp_correspondence_count: int = Field(default=0, ge=0)
    validation: CourtKeypointClipValidation | None = None
    partial_label_path: str | None = None
    partial_visible_gate: PartialVisibleGate | None = None
    notes: list[str] = Field(default_factory=list)


class CourtKeypointNoTapEvalSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_clip_count: int = Field(ge=0)
    skipped_clip_count: int = Field(ge=0)
    blocked_clip_count: int = Field(ge=0)
    ready_clip_count: int = Field(ge=0)
    ran_clip_count: int = Field(ge=0)
    diagnostic_gate_passed_clip_count: int = Field(default=0, ge=0)
    diagnostic_gate_blocked_clip_count: int = Field(default=0, ge=0)
    partial_clip_count: int = Field(default=0, ge=0)
    partial_gate_passed_clip_count: int = Field(default=0, ge=0)
    partial_gate_blocked_clip_count: int = Field(default=0, ge=0)
    h100_gate_command: list[str]


class CourtKeypointNoTapEvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    artifact_type: Literal["court_keypoint_no_tap_eval_report"]
    status: Literal["ready_for_h100", "ran_not_verified", "blocked"]
    claim_scope: Literal["dry_run_plumbing", "h100_court_keypoint_eval_not_cal3_verified"]
    run_root: str
    checkpoint: str
    metrics: str | None = None
    dry_run: bool
    device: str
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    thresholds: list[float] = Field(default_factory=lambda: list(DEFAULT_THRESHOLD_SWEEP))
    verified: bool
    not_cal3_verified: bool
    summary: CourtKeypointNoTapEvalSummary
    clips: list[CourtKeypointClipReport]
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _never_claim_cal3_verified(self) -> "CourtKeypointNoTapEvalReport":
        if self.verified or not self.not_cal3_verified:
            raise ValueError("court-keypoint no-tap evaluator must not claim CAL-3 verification")
        if self.dry_run and self.claim_scope != "dry_run_plumbing":
            raise ValueError("dry-run reports must use dry_run_plumbing claim scope")
        if not self.dry_run and self.claim_scope != "h100_court_keypoint_eval_not_cal3_verified":
            raise ValueError("non-dry-run reports must use h100 not-verified claim scope")
        return self


@dataclass(frozen=True)
class SelectedClipInputs:
    active: list[CourtKeypointClipReport]
    skipped: dict[str, CourtKeypointClipReport]


def select_active_clip_inputs(
    run_root: str | Path,
    *,
    accepted_clips: Sequence[str] = DEFAULT_ACCEPTED_CLIPS,
    retired_clips: set[str] | frozenset[str] = DEFAULT_COURT_RETIRED_CLIPS,
    frames_per_clip: int = 5,
    top_net_match_threshold_px: float = DEFAULT_TOP_NET_MATCH_THRESHOLD_PX,
) -> SelectedClipInputs:
    """Return court-review-active clips with schema-valid reviewed evidence."""

    root = Path(run_root)
    active: list[CourtKeypointClipReport] = []
    skipped: dict[str, CourtKeypointClipReport] = {}

    for clip in accepted_clips:
        run_dir = root / clip
        paths = _clip_paths(run_dir)
        base_report = {
            "clip": clip,
            "run_dir": str(run_dir),
            "input_video": str(paths["input_video"]),
            "court_line_evidence": str(paths["court_line_evidence"]),
            "top_net_review_points": str(paths["top_net_review_points"]),
            "court_calibration": str(paths["court_calibration"]),
            "court_zones": str(paths["court_zones"]),
            "net_plane": str(paths["net_plane"]),
            "planned_frame_indexes": list(range(max(0, int(frames_per_clip)))),
        }
        if clip in retired_clips:
            skipped[clip] = CourtKeypointClipReport(
                **base_report,
                status="skipped_retired_for_court",
                retired_for_court=True,
                top_net_review_match=_missing_top_net_match(
                    "clip is retired for court calibration review",
                    threshold_px=top_net_match_threshold_px,
                ),
                notes=["retired_for_court_calibration"],
            )
            continue

        missing = [name for name, path in paths.items() if not path.is_file()]
        notes: list[str] = []
        evidence_ready = False
        accepted_line_ids: list[str] = []
        missing_line_ids: list[str] = []
        missing_net_ids: list[str] = []
        top_net_match = _missing_top_net_match("not evaluated", threshold_px=top_net_match_threshold_px)

        if not missing:
            try:
                evidence = _load_court_line_evidence(paths["court_line_evidence"])
                review_points = _load_top_net_review_points(paths["top_net_review_points"])
                evidence_ready = evidence.aggregate.auto_calibration_ready
                accepted_line_ids = list(evidence.aggregate.accepted_line_ids)
                missing_line_ids = list(evidence.aggregate.missing_required_line_ids)
                missing_net_ids = list(evidence.aggregate.missing_required_net_ids)
                top_net_match = compare_reviewed_top_net(
                    evidence,
                    review_points,
                    threshold_px=top_net_match_threshold_px,
                )
                notes.extend(evidence.aggregate.reasons)
            except Exception as exc:
                notes.append(f"artifact validation failed: {exc}")

        blocked = bool(missing) or not evidence_ready or not top_net_match.passed
        report = CourtKeypointClipReport(
            **base_report,
            status="blocked" if blocked else "ready_for_h100",
            retired_for_court=False,
            evidence_ready=evidence_ready,
            accepted_line_ids=accepted_line_ids,
            missing_required_line_ids=missing_line_ids,
            missing_required_net_ids=missing_net_ids,
            top_net_review_match=top_net_match,
            missing_artifacts=missing,
            notes=notes,
        )
        if blocked:
            skipped[clip] = report
        else:
            active.append(report)

    return SelectedClipInputs(active=active, skipped=skipped)


def compare_reviewed_top_net(
    evidence: CourtLineEvidence,
    review_points: TopNetReviewPoints,
    *,
    threshold_px: float = DEFAULT_TOP_NET_MATCH_THRESHOLD_PX,
) -> TopNetReviewMatch:
    """Compare court evidence top-net endpoints against human review points."""

    if not evidence.net_observations:
        return _missing_top_net_match("court_line_evidence has no net observations", threshold_px=threshold_px)

    top_net = next((item for item in evidence.net_observations if item.net_id == "top_net"), None)
    if top_net is None:
        return _missing_top_net_match("court_line_evidence has no top_net observation", threshold_px=threshold_px)
    if len(top_net.image_points) != 3:
        return _missing_top_net_match("top_net observation must contain left, center, right", threshold_px=threshold_px)

    observed = [top_net.image_points[0], top_net.image_points[2]]
    candidates: dict[str, list[list[float]]] = {}
    click_left = review_points.click_points.get("left")
    click_right = review_points.click_points.get("right")
    if click_left is not None and click_right is not None:
        candidates["click_points"] = [[float(click_left.x), float(click_left.y)], [float(click_right.x), float(click_right.y)]]

    evidence_points = review_points.evidence_points or {}
    raw_left = evidence_points.get("left")
    raw_right = evidence_points.get("right")
    if _is_point(raw_left) and _is_point(raw_right):
        candidates["evidence_points"] = [[float(raw_left[0]), float(raw_left[1])], [float(raw_right[0]), float(raw_right[1])]]

    if not candidates:
        return _missing_top_net_match("top_net review points are missing left/right endpoints", threshold_px=threshold_px)

    deltas = {
        name: max(_distance(observed[0], pair[0]), _distance(observed[1], pair[1]))
        for name, pair in candidates.items()
    }
    best_space, best_delta = min(deltas.items(), key=lambda item: item[1])
    return TopNetReviewMatch(
        passed=best_delta <= threshold_px,
        best_coordinate_space=best_space,  # type: ignore[arg-type]
        max_endpoint_delta_px=float(best_delta),
        threshold_px=float(threshold_px),
        notes=[] if best_delta <= threshold_px else [f"top_net_review_delta_px={best_delta:.3f}"],
    )


def build_partial_visible_clip_reports(
    *,
    eval_root: str | Path,
    include_partial: Sequence[str],
    partial_proposal_root: str | Path | None = None,
) -> list[CourtKeypointClipReport]:
    reports: list[CourtKeypointClipReport] = []
    root = Path(eval_root)
    proposal_root = Path(partial_proposal_root) if partial_proposal_root is not None else None
    for clip in include_partial:
        clip_dir = root / clip
        partial_path = clip_dir / "labels" / "court_keypoints_partial.json"
        base_report = {
            "clip": clip,
            "run_dir": str(clip_dir),
            "input_video": str(clip_dir / "source.mp4"),
            "partial_label_path": str(partial_path),
            "top_net_review_match": _missing_top_net_match(
                "partial visible-label lane uses 15-point partial labels instead of top-net-only review",
                threshold_px=DEFAULT_TOP_NET_MATCH_THRESHOLD_PX,
            ),
        }
        if not partial_path.is_file():
            reports.append(
                CourtKeypointClipReport(
                    **base_report,
                    status="blocked",
                    missing_artifacts=[str(partial_path)],
                    notes=["partial_court_keypoints_missing"],
                )
            )
            continue
        proposal_path = _resolve_partial_proposal_path(clip=clip, proposal_root=proposal_root)
        gate = _build_partial_visible_gate(partial_path=partial_path, proposal_path=proposal_path)
        reports.append(
            CourtKeypointClipReport(
                **base_report,
                status="ran_not_verified",
                evidence_ready=gate.gate_passed,
                partial_visible_gate=gate,
                notes=[
                    "partial visible-label gate; not CAL-3 verified",
                    *gate.blockers,
                ],
            )
        )
    return reports


def _build_partial_visible_gate(*, partial_path: Path, proposal_path: Path | None) -> PartialVisibleGate:
    labels = load_partial_court_keypoints(partial_path)
    visible_points_by_frame = _partial_visible_points_by_frame(labels)
    visible_points = _median_points_by_name(visible_points_by_frame)
    missing = [point.name for point in PICKLEBALL_KEYPOINTS if point.name not in visible_points]
    floor_fit = fit_visible_floor_homography(labels)
    floor_fit_errors = [
        float(floor_fit.per_keypoint_residual_px[name])
        for name in floor_fit.used_keypoints
        if name in floor_fit.per_keypoint_residual_px
    ]
    temporal = _partial_temporal_stability(visible_points_by_frame)

    proposal_payload: Mapping[str, Any] | None = None
    proposal_points: dict[str, tuple[float, float, float]] = {}
    if proposal_path is not None and proposal_path.is_file():
        proposal_payload = json.loads(proposal_path.read_text(encoding="utf-8"))
        proposal_points = _proposal_points_native_px(proposal_payload, native_size=labels.source_resolution)

    errors = {
        name: _distance(visible_xy, proposal_points[name][:2])
        for name, visible_xy in visible_points.items()
        if name in proposal_points
    }
    high_confidence_wrong = sum(
        1
        for name, error in errors.items()
        if proposal_points.get(name, (0.0, 0.0, 0.0))[2] >= PARTIAL_HIGH_CONFIDENCE_THRESHOLD
        and error > PARTIAL_HIGH_CONFIDENCE_ERROR_GATE_PX
    )
    self_verification = proposal_payload.get("self_verification") if isinstance(proposal_payload, Mapping) else None
    promotion_allowed = (
        bool(self_verification.get("promotion_allowed"))
        if isinstance(self_verification, Mapping)
        else None
    )

    all_visible = _metric_summary([errors[name] for name in visible_points if name in errors])
    floor_visible = _metric_summary(
        [errors[name] for name in visible_points if name in errors and name not in NET_TOP_KEYPOINT_NAMES]
    )
    visible_corners = _metric_summary(
        [errors[name] for name in OUTER_CORNER_NAMES if name in visible_points and name in errors]
    )
    net_top_visible = _metric_summary(
        [errors[name] for name in NET_TOP_KEYPOINT_NAMES if name in visible_points and name in errors]
    )
    floor_fit_summary = _metric_summary(floor_fit_errors)

    blockers = _partial_gate_blockers(
        proposal_path=proposal_path,
        floor_fit_verified=floor_fit.fit_verified,
        floor_visible=floor_visible,
        visible_corners=visible_corners,
        high_confidence_wrong=high_confidence_wrong,
        promotion_allowed=promotion_allowed,
        temporal=temporal,
    )
    return PartialVisibleGate(
        partial_label_path=str(partial_path),
        proposal_path=str(proposal_path) if proposal_path is not None and proposal_path.is_file() else None,
        visible_keypoint_count=len(visible_points),
        missing_keypoints=missing,
        visible_floor_fit_verified=floor_fit.fit_verified,
        visible_floor_fit_error_px=floor_fit_summary,
        all_visible_error_px=all_visible,
        floor_visible_error_px=floor_visible,
        visible_corner_error_px=visible_corners,
        net_top_visible_error_px=net_top_visible,
        high_confidence_over_30px_count=high_confidence_wrong,
        self_verification_promotion_allowed=promotion_allowed,
        temporal_stability_px=temporal,
        gate_passed=not blockers,
        blockers=blockers,
    )


def _partial_gate_blockers(
    *,
    proposal_path: Path | None,
    floor_fit_verified: bool,
    floor_visible: MetricSummary | None,
    visible_corners: MetricSummary | None,
    high_confidence_wrong: int,
    promotion_allowed: bool | None,
    temporal: MetricSummary | None,
) -> list[str]:
    blockers: list[str] = []
    if not floor_fit_verified:
        blockers.append("visible_floor_fit_not_verified")
    if proposal_path is None or not proposal_path.is_file():
        blockers.append("proposal_missing")
    if floor_visible is None:
        blockers.append("floor_visible_error_missing")
    else:
        if floor_visible.median > PARTIAL_VISIBLE_FLOOR_MEDIAN_GATE_PX:
            blockers.append("visible_floor_median_gt_15")
        if floor_visible.p95 > PARTIAL_VISIBLE_FLOOR_P95_GATE_PX:
            blockers.append("visible_floor_p95_gt_30")
    if visible_corners is None:
        blockers.append("visible_corner_error_missing")
    elif visible_corners.median > PARTIAL_VISIBLE_CORNER_MEDIAN_GATE_PX:
        blockers.append("visible_corner_median_gt_20")
    if high_confidence_wrong > 0:
        blockers.append("high_confidence_wrong_proposals")
    if promotion_allowed is not True:
        blockers.append("self_verification_not_promotable")
    if temporal is not None and temporal.p95 > PARTIAL_TEMPORAL_STABILITY_GATE_PX:
        blockers.append("temporal_stability_gt_10")
    return blockers


def _partial_visible_points_by_frame(labels: Any) -> list[dict[str, tuple[float, float]]]:
    label_w, label_h = labels.label_coordinate_space
    scale_x = labels.source_resolution[0] / label_w
    scale_y = labels.source_resolution[1] / label_h
    frames: list[dict[str, tuple[float, float]]] = []
    for frame in labels.frames:
        points: dict[str, tuple[float, float]] = {}
        for name, xy in frame.keypoints.items():
            if frame.visibility_by_keypoint.get(name) != VISIBLE:
                continue
            points[name] = (float(xy[0]) * scale_x, float(xy[1]) * scale_y)
        if points:
            frames.append(points)
    return frames


def _median_points_by_name(frames: Sequence[Mapping[str, tuple[float, float]]]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for point in PICKLEBALL_KEYPOINTS:
        xs = [frame[point.name][0] for frame in frames if point.name in frame]
        ys = [frame[point.name][1] for frame in frames if point.name in frame]
        if xs:
            out[point.name] = (_percentile(xs, 50.0), _percentile(ys, 50.0))
    return out


def _partial_temporal_stability(frames: Sequence[Mapping[str, tuple[float, float]]]) -> MetricSummary:
    distances: list[float] = []
    medians = _median_points_by_name(frames)
    for frame in frames:
        for name, xy in frame.items():
            if name in medians:
                distances.append(_distance(xy, medians[name]))
    return _metric_summary(distances) or MetricSummary(count=0, mean=0.0, median=0.0, p95=0.0, max=0.0)


def _proposal_points_native_px(
    payload: Mapping[str, Any],
    *,
    native_size: tuple[float, float],
) -> dict[str, tuple[float, float, float]]:
    proposal_size = _proposal_image_size(payload)
    scale_x = 1.0
    scale_y = 1.0
    if proposal_size is not None and proposal_size[0] > 0.0 and proposal_size[1] > 0.0:
        scale_x = native_size[0] / proposal_size[0]
        scale_y = native_size[1] / proposal_size[1]
    out: dict[str, tuple[float, float, float]] = {}
    for name, item in (payload.get("keypoints") or {}).items():
        if not isinstance(item, Mapping):
            continue
        xy = item.get("xy") or item.get("image_xy") or item.get("point")
        if not _is_point(xy):
            continue
        out[str(name)] = (
            float(xy[0]) * scale_x,
            float(xy[1]) * scale_y,
            float(item.get("confidence", payload.get("solver_confidence") or 0.0)),
        )
    return out


def _proposal_image_size(payload: Mapping[str, Any]) -> tuple[float, float] | None:
    source = payload.get("source")
    if isinstance(source, Mapping):
        image_size = source.get("image_size") or source.get("source_size")
        if _is_point(image_size):
            return (float(image_size[0]), float(image_size[1]))
    for key in ("image_size", "source_size"):
        value = payload.get(key)
        if _is_point(value):
            return (float(value[0]), float(value[1]))
    return None


def _resolve_partial_proposal_path(*, clip: str, proposal_root: Path | None) -> Path | None:
    candidates: list[Path] = []
    if proposal_root is not None:
        candidates.extend(
            [
                proposal_root / "court_corner_proposals.json",
                proposal_root / clip / "court_corner_proposals.json",
            ]
        )
    lowered = clip.lower()
    if "img_1605" in lowered or "img1605" in lowered:
        candidates.append(Path("runs/img1605_court_detector_hardened/court_corner_proposals.json"))
    candidates.extend(
        [
            Path("runs") / clip / "court_corner_proposals.json",
            Path("runs") / clip.replace("owner_", "") / "court_corner_proposals.json",
        ]
    )
    return next((path for path in candidates if path.is_file()), None)


def build_court_keypoint_prediction_validation(
    *,
    prediction_payload: Mapping[str, Any],
    calibration_payload: CourtCalibration | Mapping[str, Any],
    thresholds: Sequence[float] | None = None,
) -> CourtKeypointClipValidation:
    """Evaluate no-tap predictions against the current court calibration evidence.

    This is a diagnostic validation path. Passing these checks is not a CAL-3
    claim because it compares against existing calibration artifacts instead of
    independent real keypoint ground truth.
    """

    calibration = (
        calibration_payload
        if isinstance(calibration_payload, CourtCalibration)
        else CourtCalibration.model_validate(calibration_payload)
    )
    threshold_values = _threshold_values(thresholds, default_min_confidence=0.5)
    frames = _prediction_frames(prediction_payload)
    source_width, source_height = _prediction_source_size(prediction_payload, calibration)
    clip = str(prediction_payload.get("clip") or "unknown")
    threshold_reports = [
        _build_threshold_validation(
            frames,
            calibration=calibration,
            threshold=threshold,
            source_width=source_width,
            source_height=source_height,
        )
        for threshold in threshold_values
    ]
    best = _best_threshold_validation(threshold_reports)
    return CourtKeypointClipValidation(
        clip=clip,
        thresholds=threshold_reports,
        best_threshold=best.threshold if best is not None else None,
        best_threshold_gate_passed=bool(best and best.diagnostic_gate_passed),
        best_threshold_blockers=list(best.blockers) if best is not None else ["no_thresholds_evaluated"],
        notes=[
            "diagnostic validation only; compares checkpoint predictions against existing calibration evidence",
            "not CAL-3 verified without independent real keypoint ground truth and broader clip coverage",
        ],
    )


def _build_threshold_validation(
    frames: list[Mapping[str, Any]],
    *,
    calibration: CourtCalibration,
    threshold: float,
    source_width: float,
    source_height: float,
) -> CourtKeypointThresholdValidation:
    frame_counts: list[int] = []
    frame_filtered: list[dict[str, tuple[float, float]]] = []
    reprojection_errors: list[float] = []
    world_errors: list[float] = []
    outside_court_count = 0
    projected_keypoint_count = 0
    near_border_count = 0
    total_confident = 0

    for frame in frames:
        predictions = validate_heatmap_prediction_payload({"keypoints": frame.get("keypoints", {})})
        filtered: dict[str, tuple[float, float]] = {}
        for name, prediction in predictions.items():
            if prediction.confidence < threshold:
                continue
            x, y = prediction.image_xy
            filtered[name] = (x, y)
            total_confident += 1
            expected_image = project_planar_points(
                calibration.homography,
                [PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m[:2]],
            )[0]
            reprojection_errors.append(_distance((x, y), expected_image))
            if _near_border(x, y, source_width=source_width, source_height=source_height):
                near_border_count += 1
            try:
                world_xy = project_image_points_to_world(calibration.homography, [(x, y)])[0]
                expected_world = PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m[:2]
                world_errors.append(_distance(world_xy, expected_world))
                projected_keypoint_count += 1
                if not _inside_pickleball_court(world_xy):
                    outside_court_count += 1
            except ValueError:
                outside_court_count += 1
        frame_counts.append(len(filtered))
        frame_filtered.append(filtered)

    temporal_jitter = _temporal_jitter(frame_filtered)
    line_corner = _line_corner_consistency(frame_filtered)
    zone = CourtZoneSanity(
        projected_keypoint_count=projected_keypoint_count,
        outside_court_count=outside_court_count,
        outside_court_ratio=_ratio(outside_court_count, projected_keypoint_count),
        world_error_m=_metric_summary(world_errors),
    )
    border = BorderSanity(
        near_border_keypoint_count=near_border_count,
        confident_keypoint_count=total_confident,
        near_border_keypoint_ratio=_ratio(near_border_count, total_confident),
        margin_px=max(source_width, source_height) * BORDER_MARGIN_RATIO,
    )
    calibration_reprojection = _metric_summary(reprojection_errors)
    blockers = _threshold_blockers(
        frame_counts=frame_counts,
        calibration_reprojection=calibration_reprojection,
        temporal_jitter=temporal_jitter,
        line_corner=line_corner,
        zone=zone,
        border=border,
    )
    return CourtKeypointThresholdValidation(
        threshold=threshold,
        frame_count=len(frames),
        total_confident_keypoints=total_confident,
        frame_min_count=min(frame_counts, default=0),
        frame_max_count=max(frame_counts, default=0),
        frames_with_solvepnp_min4=sum(1 for count in frame_counts if count >= 4),
        calibration_reprojection_error_px=calibration_reprojection,
        temporal_jitter_px=temporal_jitter,
        line_corner_consistency=line_corner,
        court_zone_sanity=zone,
        border_sanity=border,
        diagnostic_gate_passed=not blockers,
        blockers=blockers,
    )


def _threshold_blockers(
    *,
    frame_counts: list[int],
    calibration_reprojection: MetricSummary | None,
    temporal_jitter: MetricSummary | None,
    line_corner: LineCornerConsistency,
    zone: CourtZoneSanity,
    border: BorderSanity,
) -> list[str]:
    blockers: list[str] = []
    if not frame_counts or max(frame_counts, default=0) == 0:
        blockers.append("no_confident_keypoints")
    if not any(count >= 4 for count in frame_counts):
        blockers.append("no_frames_with_solvepnp_min4")
    if calibration_reprojection is None:
        blockers.append("no_calibration_reprojection_samples")
    elif calibration_reprojection.p95 > CALIBRATION_REPROJECTION_P95_GATE_PX:
        blockers.append("calibration_reprojection_p95_too_high")
    if temporal_jitter is not None and temporal_jitter.p95 > TEMPORAL_JITTER_P95_GATE_PX:
        blockers.append("temporal_jitter_p95_too_high")
    if line_corner.triplet_count > 0 and line_corner.triplet_pass_ratio < 0.8:
        blockers.append("line_triplet_consistency_too_low")
    if line_corner.corner_quad_frame_count > 0 and line_corner.corner_quad_valid_ratio < 0.8:
        blockers.append("corner_quad_consistency_too_low")
    if zone.world_error_m is not None and zone.world_error_m.p95 > WORLD_ERROR_P95_GATE_M:
        blockers.append("court_world_error_p95_too_high")
    if zone.outside_court_ratio > 0.0:
        blockers.append("predicted_world_points_outside_court")
    if border.near_border_keypoint_ratio > BORDER_KEYPOINT_RATIO_GATE:
        blockers.append("border_keypoint_ratio_too_high")
    return blockers


def _best_threshold_validation(
    threshold_reports: list[CourtKeypointThresholdValidation],
) -> CourtKeypointThresholdValidation | None:
    if not threshold_reports:
        return None
    return max(
        threshold_reports,
        key=lambda item: (
            item.diagnostic_gate_passed,
            item.frames_with_solvepnp_min4,
            -len(item.blockers),
            item.total_confident_keypoints,
            -(item.calibration_reprojection_error_px.p95 if item.calibration_reprojection_error_px else math.inf),
        ),
    )


LINE_TRIPLETS: tuple[tuple[str, str, str, str], ...] = (
    ("near_baseline", "near_left_corner", "near_baseline_center", "near_right_corner"),
    ("far_baseline", "far_left_corner", "far_baseline_center", "far_right_corner"),
    ("near_nvz", "near_nvz_left", "near_nvz_center", "near_nvz_right"),
    ("far_nvz", "far_nvz_left", "far_nvz_center", "far_nvz_right"),
    ("net", "net_left_sideline", "net_center", "net_right_sideline"),
)

def _line_corner_consistency(frame_filtered: list[dict[str, tuple[float, float]]]) -> LineCornerConsistency:
    residuals: list[float] = []
    pass_count = 0
    corner_frame_count = 0
    corner_valid_count = 0
    for frame in frame_filtered:
        for _, left, center, right in LINE_TRIPLETS:
            if left not in frame or center not in frame or right not in frame:
                continue
            residual = _point_line_distance(frame[center], frame[left], frame[right])
            residuals.append(residual)
            if residual <= LINE_TRIPLET_RESIDUAL_GATE_PX:
                pass_count += 1
        if all(name in frame for name in OUTER_CORNER_NAMES):
            corner_frame_count += 1
            corners = [frame[name] for name in OUTER_CORNER_NAMES]
            if _quad_is_valid(corners):
                corner_valid_count += 1
    return LineCornerConsistency(
        triplet_count=len(residuals),
        triplet_pass_count=pass_count,
        triplet_pass_ratio=_ratio(pass_count, len(residuals)),
        triplet_residual_px=_metric_summary(residuals),
        corner_quad_frame_count=corner_frame_count,
        corner_quad_valid_count=corner_valid_count,
        corner_quad_valid_ratio=_ratio(corner_valid_count, corner_frame_count),
    )


def _temporal_jitter(frame_filtered: list[dict[str, tuple[float, float]]]) -> MetricSummary | None:
    by_name: dict[str, list[tuple[float, float]]] = {}
    for frame in frame_filtered:
        for name, point in frame.items():
            by_name.setdefault(name, []).append(point)
    distances: list[float] = []
    for points in by_name.values():
        if len(points) < 2:
            continue
        median_x = _percentile([point[0] for point in points], 50)
        median_y = _percentile([point[1] for point in points], 50)
        distances.extend(_distance(point, (median_x, median_y)) for point in points)
    return _metric_summary(distances)


def _prediction_frames(prediction_payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    frames = prediction_payload.get("frames")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ValueError("prediction payload must contain frames")
    return [frame for frame in frames if isinstance(frame, Mapping)]


def _prediction_source_size(
    prediction_payload: Mapping[str, Any],
    calibration: CourtCalibration,
) -> tuple[float, float]:
    source_size = prediction_payload.get("source_size")
    if _is_point(source_size):
        return float(source_size[0]), float(source_size[1])
    if calibration.image_size is not None and len(calibration.image_size) == 2:
        return float(calibration.image_size[0]), float(calibration.image_size[1])
    return float(calibration.intrinsics.cx) * 2.0, float(calibration.intrinsics.cy) * 2.0


def _load_court_calibration(path: Path) -> CourtCalibration:
    parsed = validate_artifact_file("court_calibration", path)
    if not isinstance(parsed, CourtCalibration):
        raise ValueError("court_calibration artifact did not parse as CourtCalibration")
    return parsed


def build_court_keypoint_no_tap_eval_report(
    *,
    run_root: str | Path,
    checkpoint: str | Path,
    metrics: str | Path | None,
    out: str | Path,
    dry_run: bool,
    device: str,
    frames_per_clip: int = 5,
    min_confidence: float = 0.5,
    thresholds: Sequence[float] | None = None,
    accepted_clips: Sequence[str] | None = None,
    eval_root: str | Path | None = None,
    include_partial: Sequence[str] = (),
    partial_proposal_root: str | Path | None = None,
    detector_v2_proposal_root: str | Path | None = None,
) -> CourtKeypointNoTapEvalReport:
    root = Path(run_root)
    checkpoint_path = Path(checkpoint)
    metrics_path = Path(metrics) if metrics is not None else None
    out_path = Path(out)
    confidence_threshold = _unit_interval(min_confidence, "min_confidence")
    threshold_values = _threshold_values(thresholds, confidence_threshold)
    notes = _validate_checkpoint_inputs(checkpoint_path, metrics_path)
    selected = select_active_clip_inputs(
        root,
        accepted_clips=DEFAULT_ACCEPTED_CLIPS if accepted_clips is None else accepted_clips,
        frames_per_clip=frames_per_clip,
    )
    partial_reports = (
        build_partial_visible_clip_reports(
            eval_root=eval_root,
            include_partial=include_partial,
            partial_proposal_root=partial_proposal_root,
        )
        if eval_root is not None and include_partial
        else []
    )

    active_reports = list(selected.active)
    skipped_reports = list(selected.skipped.values())
    if active_reports and not dry_run:
        active_reports = _run_checkpoint_on_active_clips(
            active_reports,
            checkpoint=checkpoint_path,
            device=device,
            frames_per_clip=frames_per_clip,
            min_confidence=confidence_threshold,
            thresholds=threshold_values,
            prediction_dir=out_path.parent / "court_keypoint_predictions",
            notes=notes,
        )

    clips = [*active_reports, *partial_reports, *skipped_reports]
    status: Literal["ready_for_h100", "ran_not_verified", "blocked"]
    if not active_reports and not partial_reports:
        status = "blocked"
    elif dry_run:
        status = "ready_for_h100"
    else:
        status = "ran_not_verified"

    report = CourtKeypointNoTapEvalReport(
        schema_version=1,
        artifact_type="court_keypoint_no_tap_eval_report",
        status=status,
        claim_scope="dry_run_plumbing" if dry_run else "h100_court_keypoint_eval_not_cal3_verified",
        run_root=str(root),
        checkpoint=str(checkpoint_path),
        metrics=str(metrics_path) if metrics_path is not None else None,
        dry_run=dry_run,
        device=device,
        min_confidence=confidence_threshold,
        thresholds=threshold_values,
        verified=False,
        not_cal3_verified=True,
        summary=CourtKeypointNoTapEvalSummary(
            active_clip_count=len(active_reports),
            skipped_clip_count=sum(1 for clip in clips if clip.status == "skipped_retired_for_court"),
            blocked_clip_count=sum(1 for clip in clips if clip.status == "blocked"),
            ready_clip_count=sum(1 for clip in clips if clip.status == "ready_for_h100"),
            ran_clip_count=sum(1 for clip in clips if clip.status == "ran_not_verified"),
            diagnostic_gate_passed_clip_count=sum(
                1 for clip in clips if clip.validation is not None and clip.validation.best_threshold_gate_passed
            ),
            diagnostic_gate_blocked_clip_count=sum(
                1 for clip in clips if clip.validation is not None and not clip.validation.best_threshold_gate_passed
            ),
            partial_clip_count=len(partial_reports),
            partial_gate_passed_clip_count=sum(
                1 for clip in partial_reports if clip.partial_visible_gate is not None and clip.partial_visible_gate.gate_passed
            ),
            partial_gate_blocked_clip_count=sum(
                1 for clip in partial_reports if clip.partial_visible_gate is not None and not clip.partial_visible_gate.gate_passed
            ),
            h100_gate_command=h100_gate_command(
                run_root=root,
                checkpoint=checkpoint_path,
                metrics=metrics_path,
                out=out_path,
                frames_per_clip=frames_per_clip,
                min_confidence=confidence_threshold,
                thresholds=threshold_values,
                eval_root=eval_root,
                include_partial=include_partial,
                partial_proposal_root=partial_proposal_root,
                detector_v2_proposal_root=detector_v2_proposal_root,
            ),
        ),
        clips=clips,
        notes=notes,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return CourtKeypointNoTapEvalReport.model_validate_json(out_path.read_text(encoding="utf-8"))


def h100_gate_command(
    *,
    run_root: str | Path,
    checkpoint: str | Path,
    metrics: str | Path | None,
    out: str | Path,
    frames_per_clip: int,
    min_confidence: float = 0.5,
    thresholds: Sequence[float] | None = None,
    eval_root: str | Path | None = None,
    include_partial: Sequence[str] = (),
    partial_proposal_root: str | Path | None = None,
    detector_v2_proposal_root: str | Path | None = None,
) -> list[str]:
    h100_out = _h100_out_path(Path(out))
    confidence_threshold = _unit_interval(min_confidence, "min_confidence")
    threshold_values = _threshold_values(thresholds, confidence_threshold)
    command = [
        "python",
        "scripts/racketsport/evaluate_court_keypoint_no_tap.py",
        "--run-root",
        str(run_root),
        "--checkpoint",
        str(checkpoint),
    ]
    if metrics is not None:
        command.extend(["--metrics", str(metrics)])
    command.extend(
        [
            "--out",
            str(h100_out),
            "--frames-per-clip",
            str(frames_per_clip),
            "--device",
            "cuda",
            "--min-confidence",
            f"{confidence_threshold:g}",
        ]
    )
    if threshold_values:
        command.extend(["--thresholds", ",".join(f"{threshold:g}" for threshold in threshold_values)])
    if eval_root is not None:
        command.extend(["--eval-root", str(eval_root)])
    for clip in include_partial:
        command.extend(["--include-partial", str(clip)])
    if partial_proposal_root is not None:
        command.extend(["--partial-proposal-root", str(partial_proposal_root)])
    if detector_v2_proposal_root is not None:
        command.extend(["--detector-v2-proposal-root", str(detector_v2_proposal_root)])
    return command


def render_court_keypoint_no_tap_eval_markdown(report: CourtKeypointNoTapEvalReport) -> str:
    lines = [
        "# Court Keypoint No-Tap Diagnostic",
        "",
        f"- status: `{report.status}`",
        f"- claim scope: `{report.claim_scope}`",
        "- verdict: not CAL-3 verified",
        f"- active clips: {report.summary.active_clip_count}",
        f"- skipped clips: {report.summary.skipped_clip_count}",
        f"- diagnostic gate passed clips: {report.summary.diagnostic_gate_passed_clip_count}",
        f"- diagnostic gate blocked clips: {report.summary.diagnostic_gate_blocked_clip_count}",
        "",
        "| clip | threshold | frames>=4 | reproj p95 px | temporal p95 px | border ratio | world p95 m | gate | blockers |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for clip in report.clips:
        if clip.validation is None:
            for threshold in report.thresholds:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            clip.clip,
                            f"{threshold:g}",
                            "n/a",
                            "n/a",
                            "n/a",
                            "n/a",
                            "n/a",
                            clip.status,
                            ", ".join(clip.notes) or "not run",
                        ]
                    )
                    + " |"
                )
            continue
        for threshold_report in clip.validation.thresholds:
            reproj = threshold_report.calibration_reprojection_error_px
            temporal = threshold_report.temporal_jitter_px
            world = threshold_report.court_zone_sanity.world_error_m
            lines.append(
                "| "
                + " | ".join(
                    [
                        clip.clip,
                        f"{threshold_report.threshold:g}",
                        str(threshold_report.frames_with_solvepnp_min4),
                        _format_optional_float(reproj.p95 if reproj is not None else None),
                        _format_optional_float(temporal.p95 if temporal is not None else None),
                        f"{threshold_report.border_sanity.near_border_keypoint_ratio:.3f}",
                        _format_optional_float(world.p95 if world is not None else None),
                        "pass" if threshold_report.diagnostic_gate_passed else "blocked",
                        ", ".join(threshold_report.blockers) or "none",
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Burlington is skipped for court calibration review when present in the retired court clip set.",
            "- Passing this diagnostic would still require independent real keypoint labels and broader validation before any CAL-3 claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _h100_out_path(out: Path) -> Path:
    if "dry_run" not in out.stem:
        return out
    return out.with_name(out.name.replace("dry_run", "h100"))


def _clip_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "input_video": run_dir / "tracknet_smoke_0000_0010" / "input_0000_0010.mp4",
        "court_line_evidence": run_dir / "court_line_evidence.json",
        "top_net_review_points": run_dir / "top_net_review_points.json",
        "court_calibration": run_dir / "court_calibration.json",
        "court_zones": run_dir / "court_zones.json",
        "net_plane": run_dir / "net_plane.json",
    }


def _load_court_line_evidence(path: Path) -> CourtLineEvidence:
    parsed = validate_artifact_file("court_line_evidence", path)
    if not isinstance(parsed, CourtLineEvidence):
        raise ValueError("court_line_evidence artifact did not parse as CourtLineEvidence")
    return parsed


def _load_top_net_review_points(path: Path) -> TopNetReviewPoints:
    with path.open("r", encoding="utf-8") as handle:
        return TopNetReviewPoints.model_validate(json.load(handle))


def _validate_checkpoint_inputs(checkpoint: Path, metrics: Path | None) -> list[str]:
    notes: list[str] = []
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing court-keypoint checkpoint: {checkpoint}")
    if metrics is None:
        notes.append("pretraining metrics JSON not supplied")
        return notes
    if not metrics.is_file():
        raise FileNotFoundError(f"missing court-keypoint metrics JSON: {metrics}")
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("court-keypoint metrics JSON must be an object")
    if payload.get("status") != "trained_not_phase_verified":
        raise ValueError("court-keypoint metrics status must remain trained_not_phase_verified")
    note = str(payload.get("note") or "")
    if "not a verified CAL-3" not in note and "not a verified cal-3" not in note.lower():
        notes.append("metrics note does not explicitly say the checkpoint is not CAL-3 verified")
    return notes


def _run_checkpoint_on_active_clips(
    clips: list[CourtKeypointClipReport],
    *,
    checkpoint: Path,
    device: str,
    frames_per_clip: int,
    min_confidence: float,
    thresholds: Sequence[float],
    prediction_dir: Path,
    notes: list[str],
) -> list[CourtKeypointClipReport]:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("non-dry-run court-keypoint eval requires torch, numpy, and opencv-python") from exc

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("court-keypoint H100 gate requested cuda but CUDA is not available")

    payload = _load_trusted_court_keypoint_checkpoint(torch, checkpoint, device=device)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise ValueError("court-keypoint checkpoint must contain a model state dict")
    keypoint_names = [str(name) for name in payload.get("keypoint_names", [point.name for point in PICKLEBALL_KEYPOINTS])]
    image_size = payload.get("image_size", [160, 90])
    if not _is_point(image_size):
        raise ValueError("court-keypoint checkpoint image_size must be [width, height]")
    model_width, model_height = int(image_size[0]), int(image_size[1])

    model_architecture = str(payload.get("model_architecture", "local_conv_v1"))
    model = make_court_keypoint_heatmap_model(len(keypoint_names), architecture=model_architecture)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    heatmap_activation = str(payload.get("heatmap_activation", "raw"))

    prediction_dir.mkdir(parents=True, exist_ok=True)
    updated: list[CourtKeypointClipReport] = []
    for clip in clips:
        video_path = Path(clip.input_video or "")
        prediction_payload = _predict_video_keypoints(
            cv2=cv2,
            np=np,
            torch=torch,
            model=model,
            device=device,
            video_path=video_path,
            clip=clip.clip,
            keypoint_names=keypoint_names,
            model_width=model_width,
            model_height=model_height,
            frames_per_clip=frames_per_clip,
            min_confidence=min_confidence,
            heatmap_activation=heatmap_activation,
        )
        prediction_path = prediction_dir / f"{clip.clip}_court_keypoints.json"
        prediction_path.write_text(json.dumps(prediction_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        frame_summaries = prediction_payload["frames"]
        validation = None
        if clip.court_calibration is not None:
            calibration = _load_court_calibration(Path(clip.court_calibration))
            validation = build_court_keypoint_prediction_validation(
                prediction_payload=prediction_payload,
                calibration_payload=calibration,
                thresholds=thresholds,
            )
        updated.append(
            clip.model_copy(
                update={
                    "status": "ran_not_verified",
                    "prediction_artifact": str(prediction_path),
                    "prediction_frame_count": len(frame_summaries),
                    "max_confident_keypoint_count": max(
                        (frame["confident_keypoint_count"] for frame in frame_summaries),
                        default=0,
                    ),
                    "max_solvepnp_correspondence_count": max(
                        (frame["solvepnp_correspondence_count"] for frame in frame_summaries),
                        default=0,
                    ),
                    "validation": validation,
                    "notes": [*clip.notes, "checkpoint inference ran; report remains not CAL-3 verified"],
                }
            )
        )
    notes.append("checkpoint inference ran on active clips; no CAL-3 verification claim is made")
    return updated


def _load_trusted_court_keypoint_checkpoint(torch: Any, checkpoint: Path, *, device: str) -> Any:
    map_location = device if device == "cuda" else "cpu"
    # This repo-owned checkpoint stores training metadata such as pathlib paths.
    return torch.load(checkpoint, map_location=map_location, weights_only=False)


def _predict_video_keypoints(
    *,
    cv2: Any,
    np: Any,
    torch: Any,
    model: Any,
    device: str,
    video_path: Path,
    clip: str,
    keypoint_names: list[str],
    model_width: int,
    model_height: int,
    frames_per_clip: int,
    min_confidence: float,
    heatmap_activation: str,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        indexes = _sample_frame_indexes(frame_count, frames_per_clip)
        frames: list[dict[str, Any]] = []
        for frame_index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
            if not ok:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (model_width, model_height), interpolation=cv2.INTER_AREA)
            arr = resized.astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits = model(tensor).detach().cpu()[0]
                pred = (
                    court_keypoint_probabilities(logits, activation=heatmap_activation)
                    if heatmap_activation in {"sigmoid", "spatial_softmax"}
                    else logits
                )
            keypoints = _decode_model_output(
                pred,
                keypoint_names=keypoint_names,
                source_width=source_width,
                source_height=source_height,
                model_width=model_width,
                model_height=model_height,
            )
            valid = validate_heatmap_prediction_payload({"keypoints": keypoints})
            confident_count = sum(1 for item in valid.values() if item.confidence >= min_confidence)
            correspondence_count = 0
            try:
                correspondence_count = len(
                    keypoints_to_solvepnp_correspondences(valid, min_confidence=min_confidence).keypoint_names
                )
            except ValueError:
                correspondence_count = 0
            frames.append(
                {
                    "frame_index": frame_index,
                    "keypoints": keypoints,
                    "confident_keypoint_count": confident_count,
                    "solvepnp_correspondence_count": correspondence_count,
                }
            )
    finally:
        capture.release()

    return {
        "schema_version": 1,
        "artifact_type": "court_keypoint_no_tap_predictions",
        "clip": clip,
        "video": str(video_path),
        "coordinate_space": "source_video_pixels",
        "model_input_size": [model_width, model_height],
        "source_size": [source_width, source_height],
        "heatmap_activation": heatmap_activation,
        "min_confidence": min_confidence,
        "frames": frames,
        "verified": False,
        "not_cal3_verified": True,
    }


def _decode_model_output(
    pred: Any,
    *,
    keypoint_names: list[str],
    source_width: int,
    source_height: int,
    model_width: int,
    model_height: int,
) -> dict[str, dict[str, Any]]:
    scale_x = source_width / float(model_width)
    scale_y = source_height / float(model_height)
    keypoints: dict[str, dict[str, Any]] = {}
    for idx, name in enumerate(keypoint_names):
        decoded = decode_subpixel_heatmap(pred[idx].tolist())
        confidence = max(0.0, min(1.0, float(decoded.score)))
        keypoints[name] = {
            "xy": [decoded.x * scale_x, decoded.y * scale_y],
            "confidence": confidence,
            "heatmap_score": float(decoded.score),
        }
    return keypoints


def _sample_frame_indexes(frame_count: int, frames_per_clip: int) -> list[int]:
    if frame_count <= 0 or frames_per_clip <= 0:
        return []
    sample_count = min(frame_count, frames_per_clip)
    if sample_count == 1:
        return [0]
    return sorted({int(round(idx * (frame_count - 1) / (sample_count - 1))) for idx in range(sample_count)})


def _missing_top_net_match(note: str, *, threshold_px: float) -> TopNetReviewMatch:
    return TopNetReviewMatch(
        passed=False,
        best_coordinate_space="missing",
        max_endpoint_delta_px=None,
        threshold_px=float(threshold_px),
        notes=[note],
    )


def _is_point(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2 and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item))
        for item in value
    )


def _unit_interval(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _threshold_values(thresholds: Sequence[float] | None, default_min_confidence: float) -> list[float]:
    raw_values = list(thresholds) if thresholds is not None else list(DEFAULT_THRESHOLD_SWEEP)
    if default_min_confidence not in raw_values:
        raw_values.insert(0, default_min_confidence)
    values: list[float] = []
    for index, raw_value in enumerate(raw_values):
        threshold = _unit_interval(raw_value, f"thresholds[{index}]")
        if threshold not in values:
            values.append(threshold)
    return values


def _metric_summary(values: Sequence[float]) -> MetricSummary | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    return MetricSummary(
        count=len(clean),
        mean=sum(clean) / len(clean),
        median=_percentile(clean, 50),
        p95=_percentile(clean, 95),
        max=max(clean),
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _near_border(x: float, y: float, *, source_width: float, source_height: float) -> bool:
    margin = max(source_width, source_height) * BORDER_MARGIN_RATIO
    return x <= margin or y <= margin or x >= source_width - margin or y >= source_height - margin


def _inside_pickleball_court(world_xy: Sequence[float], *, tolerance_m: float = 0.15) -> bool:
    x, y = float(world_xy[0]), float(world_xy[1])
    half_width_m = 10.0 * 0.3048
    half_length_m = 22.0 * 0.3048
    return (
        -half_width_m - tolerance_m <= x <= half_width_m + tolerance_m
        and -half_length_m - tolerance_m <= y <= half_length_m + tolerance_m
    )


def _point_line_distance(
    point: Sequence[float],
    line_start: Sequence[float],
    line_end: Sequence[float],
) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(line_start[0]), float(line_start[1])
    bx, by = float(line_end[0]), float(line_end[1])
    dx = bx - ax
    dy = by - ay
    denominator = math.hypot(dx, dy)
    if math.isclose(denominator, 0.0):
        return math.hypot(px - ax, py - ay)
    return abs(dy * px - dx * py + bx * ay - by * ax) / denominator


def _quad_is_valid(points: Sequence[Sequence[float]]) -> bool:
    if len(points) != 4:
        return False
    area = _polygon_area(points)
    if abs(area) < 1000.0:
        return False
    signs: list[int] = []
    for idx in range(4):
        a = points[idx]
        b = points[(idx + 1) % 4]
        c = points[(idx + 2) % 4]
        cross = (float(b[0]) - float(a[0])) * (float(c[1]) - float(b[1])) - (
            float(b[1]) - float(a[1])
        ) * (float(c[0]) - float(b[0]))
        if math.isclose(cross, 0.0, abs_tol=1e-6):
            return False
        signs.append(1 if cross > 0 else -1)
    return len(set(signs)) == 1


def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    area = 0.0
    for idx, point in enumerate(points):
        next_point = points[(idx + 1) % len(points)]
        area += float(point[0]) * float(next_point[1]) - float(next_point[0]) * float(point[1])
    return area / 2.0


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))
