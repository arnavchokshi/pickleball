"""Semantic court-line evidence scoring.

The module intentionally scores image-space candidates against named projected
template lines. Raw candidates can come from Hough/LSD, a learned segmenter, or
manual/debug labels, but the output stays as semantic evidence for calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal, Sequence

from .schemas import CourtLineEvidence, CourtLineObservation, NetLineObservation


Point2 = tuple[float, float]
Segment2 = tuple[Point2, Point2]
Sport = Literal["pickleball", "tennis"]
REQUIRED_COURT_LINE_IDS_BY_SPORT: dict[Sport, tuple[str, ...]] = {
    "pickleball": ("near_nvz", "far_nvz", "near_centerline", "far_centerline"),
    "tennis": ("near_service_line", "far_service_line"),
}
REQUIRED_COURT_NET_IDS_BY_SPORT: dict[Sport, tuple[str, ...]] = {
    "pickleball": ("top_net",),
    "tennis": ("top_net",),
}


@dataclass(frozen=True)
class LineCandidateScore:
    distance_px: float
    p95_distance_px: float
    angle_delta_deg: float
    visible_fraction: float
    score: float
    confidence: float


def required_court_line_ids(sport: Sport) -> tuple[str, ...]:
    try:
        return REQUIRED_COURT_LINE_IDS_BY_SPORT[sport]
    except KeyError as exc:
        raise ValueError(f"unsupported sport for court line evidence: {sport}") from exc


def required_court_net_ids(sport: Sport) -> tuple[str, ...]:
    try:
        return REQUIRED_COURT_NET_IDS_BY_SPORT[sport]
    except KeyError as exc:
        raise ValueError(f"unsupported sport for court net evidence: {sport}") from exc


def score_line_candidate(expected_segment: Sequence[Sequence[float]], candidate_segment: Sequence[Sequence[float]]) -> LineCandidateScore:
    """Score a candidate image segment against a projected semantic template line."""

    expected = _segment(expected_segment)
    candidate = _segment(candidate_segment)
    ex0, ex1 = expected
    ca0, ca1 = candidate
    expected_length = _distance(ex0, ex1)
    candidate_length = _distance(ca0, ca1)
    if expected_length <= 0.0 or candidate_length <= 0.0:
        return LineCandidateScore(
            distance_px=float("inf"),
            p95_distance_px=float("inf"),
            angle_delta_deg=180.0,
            visible_fraction=0.0,
            score=0.0,
            confidence=0.0,
        )

    ux = (ex1[0] - ex0[0]) / expected_length
    uy = (ex1[1] - ex0[1]) / expected_length
    distances = [_point_line_distance(point, ex0, (ux, uy)) for point in candidate]
    distance_px = sum(distances) / len(distances)
    p95_distance_px = max(distances)
    projections = [_project_along(point, ex0, (ux, uy)) for point in candidate]
    visible_fraction = _overlap_fraction(projections, expected_length)
    angle_delta_deg = _angle_delta_deg(expected, candidate)

    distance_score = math.exp(-((distance_px / 12.0) ** 2))
    angle_score = math.exp(-((angle_delta_deg / 12.0) ** 2))
    overlap_score = visible_fraction
    length_ratio = candidate_length / expected_length
    length_score = min(length_ratio, 1.0)
    score = _clamp01(0.45 * distance_score + 0.25 * angle_score + 0.25 * overlap_score + 0.05 * length_score)
    if length_ratio > 1.08:
        overlength_penalty = math.exp(-(((length_ratio - 1.08) / 0.28) ** 2))
        score *= overlength_penalty

    return LineCandidateScore(
        distance_px=distance_px,
        p95_distance_px=p95_distance_px,
        angle_delta_deg=angle_delta_deg,
        visible_fraction=visible_fraction,
        score=score,
        confidence=score,
    )


def select_best_line_observation(
    *,
    line_id: str,
    expected_segment: Sequence[Sequence[float]],
    candidate_segments: Iterable[Sequence[Sequence[float]]],
    frame_indexes: Iterable[int] = (),
    source: str = "template_line_candidate",
    min_confidence: float = 0.5,
    max_distance_px: float = 24.0,
    min_visible_fraction: float = 0.2,
) -> CourtLineObservation | None:
    """Return the best semantic observation for one projected court line."""

    scored_candidates = [
        (score_line_candidate(expected_segment, candidate_segment), _segment(candidate_segment))
        for candidate_segment in candidate_segments
    ]
    if not scored_candidates:
        return None
    score, segment = max(scored_candidates, key=lambda item: item[0].score)
    if (
        score.confidence < min_confidence
        or score.distance_px > max_distance_px
        or score.visible_fraction < min_visible_fraction
    ):
        return None
    return CourtLineObservation(
        line_id=line_id,
        image_segment=[[float(segment[0][0]), float(segment[0][1])], [float(segment[1][0]), float(segment[1][1])]],
        confidence=score.confidence,
        frame_indexes=[int(index) for index in frame_indexes],
        residual_px={"mean": score.distance_px, "p95": score.p95_distance_px},
        visible_fraction=score.visible_fraction,
        source=source,
    )


def aggregate_court_line_evidence(
    *,
    sport: Sport,
    line_observations: Iterable[CourtLineObservation | None],
    keypoint_observations: Iterable[object] = (),
    net_observations: Iterable[NetLineObservation | None] = (),
    required_line_ids: Sequence[str] = (),
    required_net_ids: Sequence[str] = (),
    min_line_confidence: float = 0.5,
    min_net_confidence: float = 0.5,
    max_mean_residual_px: float = 8.0,
    max_p95_residual_px: float = 16.0,
) -> CourtLineEvidence:
    """Aggregate per-line evidence into the no-tap calibration readiness gate."""

    lines = [line for line in line_observations if line is not None]
    nets = [net for net in net_observations if net is not None]
    accepted_lines = [line for line in lines if line.confidence >= min_line_confidence]
    rejected_line_ids = [line.line_id for line in lines if line.confidence < min_line_confidence]
    accepted_line_ids = [line.line_id for line in accepted_lines]
    accepted_net_ids = [net.net_id for net in nets if net.confidence >= min_net_confidence]
    missing_required_line_ids = [line_id for line_id in required_line_ids if line_id not in accepted_line_ids]
    missing_required_net_ids = [net_id for net_id in required_net_ids if net_id not in accepted_net_ids]

    residual_means = [line.residual_px.mean for line in accepted_lines]
    residual_p95s = [line.residual_px.p95 for line in accepted_lines]
    residual_means.extend(net.residual_px.mean for net in nets if net.confidence >= min_net_confidence)
    residual_p95s.extend(net.residual_px.p95 for net in nets if net.confidence >= min_net_confidence)
    mean_residual_px = sum(residual_means) / len(residual_means) if residual_means else 1_000_000.0
    p95_residual_px = max(residual_p95s) if residual_p95s else 1_000_000.0
    temporal_stability_px = _temporal_stability_proxy(accepted_lines, nets)

    reasons: list[str] = []
    reasons.extend(f"missing_{line_id}" for line_id in missing_required_line_ids)
    reasons.extend(f"missing_{net_id}" for net_id in missing_required_net_ids)
    if mean_residual_px > max_mean_residual_px:
        reasons.append("mean_residual_too_high")
    if p95_residual_px > max_p95_residual_px:
        reasons.append("p95_residual_too_high")

    auto_calibration_ready = not reasons
    return CourtLineEvidence(
        schema_version=1,
        sport=sport,
        source="semantic_line_evidence",
        line_observations=lines,
        keypoint_observations=list(keypoint_observations),
        net_observations=nets,
        aggregate={
            "accepted_line_ids": accepted_line_ids,
            "rejected_line_ids": rejected_line_ids,
            "missing_required_line_ids": missing_required_line_ids,
            "missing_required_net_ids": missing_required_net_ids,
            "mean_residual_px": mean_residual_px,
            "p95_residual_px": p95_residual_px,
            "temporal_stability_px": temporal_stability_px,
            "auto_calibration_ready": auto_calibration_ready,
            "reasons": reasons,
        },
    )


def _segment(segment: Sequence[Sequence[float]]) -> Segment2:
    if len(segment) != 2:
        raise ValueError("segment must contain exactly two points")
    first, second = segment
    if len(first) != 2 or len(second) != 2:
        raise ValueError("segment points must be 2D")
    return (float(first[0]), float(first[1])), (float(second[0]), float(second[1]))


def _distance(first: Point2, second: Point2) -> float:
    return math.hypot(second[0] - first[0], second[1] - first[1])


def _point_line_distance(point: Point2, line_origin: Point2, line_unit: Point2) -> float:
    dx = point[0] - line_origin[0]
    dy = point[1] - line_origin[1]
    return abs(dx * line_unit[1] - dy * line_unit[0])


def _project_along(point: Point2, line_origin: Point2, line_unit: Point2) -> float:
    return (point[0] - line_origin[0]) * line_unit[0] + (point[1] - line_origin[1]) * line_unit[1]


def _overlap_fraction(projections: Sequence[float], expected_length: float) -> float:
    start = max(min(projections), 0.0)
    end = min(max(projections), expected_length)
    if end <= start or expected_length <= 0.0:
        return 0.0
    return _clamp01((end - start) / expected_length)


def _angle_delta_deg(first: Segment2, second: Segment2) -> float:
    first_angle = math.degrees(math.atan2(first[1][1] - first[0][1], first[1][0] - first[0][0]))
    second_angle = math.degrees(math.atan2(second[1][1] - second[0][1], second[1][0] - second[0][0]))
    delta = abs(first_angle - second_angle) % 180.0
    return min(delta, 180.0 - delta)


def _temporal_stability_proxy(lines: Sequence[CourtLineObservation], nets: Sequence[NetLineObservation]) -> float:
    residuals = [line.residual_px.p95 for line in lines]
    residuals.extend(net.residual_px.p95 for net in nets)
    return max(residuals) if residuals else 1_000_000.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
