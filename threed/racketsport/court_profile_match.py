"""Court-profile matching helpers for the profiles-first v1 calibration path."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal, Mapping, Sequence

from .court_auto_evidence import detect_image_line_segments
from .court_calibration import project_world_points
from .court_line_evidence import Segment2, select_best_line_observation
from .court_templates import get_court_template
from .profile_registry import COURT_PROFILE_COLOR_DELTA_E_THRESHOLD, CourtProfile, CourtProfileMatch, LabColor
from .schemas import CourtCalibration

OUTER_COURT_LINE_IDS: tuple[str, ...] = ("near_baseline", "far_baseline", "left_sideline", "right_sideline")
COURT_PROFILE_REUSE_MEDIAN_BAR_PX = 4.8
COURT_PROFILE_REUSE_P95_BAR_PX = 12.3
COURT_PROFILE_REUSE_MIN_RECOVERED_LINES = 3


@dataclass(frozen=True)
class FourLineVerification:
    median_px: float
    p95_px: float
    recovered_lines: int
    evaluated_lines: int
    passed: bool
    line_residuals_px: dict[str, float] = field(default_factory=dict)
    missing_line_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CourtProfileReuseDecision:
    outcome: Literal["reuse", "fall_through_refresh_offer", "generic_path", "needs_owner_tag"]
    court_source: Literal["profile_reuse", "generic_path", "needs_owner_tag"]
    profile: CourtProfile | None
    match_confidence: float
    color_delta_e2000: float | None
    verification: FourLineVerification | None
    needs_profile_refresh_offer: bool
    reason: str


def project_outer_court_lines(calibration: CourtCalibration) -> dict[str, Segment2]:
    """Project the four outer court lines through the frozen metric calibration."""

    template = get_court_template(calibration.sport)
    projected: dict[str, Segment2] = {}
    for line_id in OUTER_COURT_LINE_IDS:
        endpoints = template.line_segments_m[line_id]
        points = project_world_points(calibration.extrinsics, calibration.intrinsics, endpoints)
        projected[line_id] = _segment(points)
    return projected


def line_color_lab_from_projected_lines(
    image_bgr: Any,
    calibration: CourtCalibration,
    *,
    samples_per_line: int = 64,
    cv2_module: Any | None = None,
) -> LabColor:
    """Sample pixels along projected outer lines and return their mean CIELAB color."""

    if samples_per_line <= 1:
        raise ValueError("samples_per_line must be greater than 1")
    cv2 = cv2_module or _cv2()
    np = _np()
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("image_bgr must be an HxWx3 BGR image")
    height, width = image.shape[:2]
    samples: list[Any] = []
    for segment in project_outer_court_lines(calibration).values():
        for x, y in _sample_segment(segment, samples_per_line=samples_per_line):
            ix = int(round(x))
            iy = int(round(y))
            if 0 <= ix < width and 0 <= iy < height:
                samples.append(image[iy, ix, :3])
    if not samples:
        raise ValueError("projected court lines produced no in-frame color samples")
    pixels = np.asarray(samples)
    if pixels.dtype != np.uint8:
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(pixels.reshape((-1, 1, 3)), cv2.COLOR_BGR2LAB).reshape((-1, 3))
    return LabColor(
        l=float(lab[:, 0].mean()) * 100.0 / 255.0,
        a=float(lab[:, 1].mean()) - 128.0,
        b=float(lab[:, 2].mean()) - 128.0,
    )


def verify_outer_court_lines(
    image_bgr: Any,
    calibration: CourtCalibration,
    *,
    candidate_segments: Sequence[Segment2] | None = None,
    cv2_module: Any | None = None,
) -> FourLineVerification:
    """Verify projected outer court lines against Hough/line-segment evidence."""

    if candidate_segments is None:
        candidates = detect_image_line_segments(image_bgr, cv2_module=cv2_module or _cv2())
    else:
        candidates = list(candidate_segments)

    line_residuals: dict[str, float] = {}
    p95_residuals: list[float] = []
    missing_line_ids: list[str] = []
    for line_id, expected in project_outer_court_lines(calibration).items():
        observation = select_best_line_observation(
            line_id=line_id,
            expected_segment=expected,
            candidate_segments=candidates,
            source="profile_reuse_four_line",
            min_confidence=0.35,
            max_distance_px=24.0,
            min_visible_fraction=0.35,
        )
        if observation is None:
            missing_line_ids.append(line_id)
            continue
        line_residuals[line_id] = float(observation.residual_px.mean)
        p95_residuals.append(float(observation.residual_px.p95))

    recovered = len(line_residuals)
    median_px = _percentile(list(line_residuals.values()), 50.0) if line_residuals else float("inf")
    p95_px = (
        max(p95_residuals)
        if p95_residuals and recovered >= COURT_PROFILE_REUSE_MIN_RECOVERED_LINES
        else float("inf")
    )
    passed = (
        recovered >= COURT_PROFILE_REUSE_MIN_RECOVERED_LINES
        and median_px <= COURT_PROFILE_REUSE_MEDIAN_BAR_PX
        and p95_px <= COURT_PROFILE_REUSE_P95_BAR_PX
    )
    return FourLineVerification(
        median_px=median_px,
        p95_px=p95_px,
        recovered_lines=recovered,
        evaluated_lines=len(OUTER_COURT_LINE_IDS),
        passed=passed,
        line_residuals_px=line_residuals,
        missing_line_ids=tuple(missing_line_ids),
    )


def decide_court_profile_reuse(
    matches: Sequence[CourtProfileMatch],
    verifications_by_profile_id: Mapping[str, FourLineVerification],
) -> CourtProfileReuseDecision:
    """Apply the pre-ruled reuse conjunction: fingerprint, ΔE2000, and four-line verification."""

    reusable: list[tuple[CourtProfileMatch, FourLineVerification]] = []
    color_matched_failures: list[tuple[CourtProfileMatch, FourLineVerification | None]] = []
    for match in matches:
        color_delta = match.color_delta_e2000
        color_passed = color_delta is not None and color_delta <= COURT_PROFILE_COLOR_DELTA_E_THRESHOLD
        if not color_passed:
            continue
        verification = verifications_by_profile_id.get(match.profile.profile_id)
        if verification is not None and verification.passed:
            reusable.append((match, verification))
        else:
            color_matched_failures.append((match, verification))

    if len(reusable) == 1:
        match, verification = reusable[0]
        return CourtProfileReuseDecision(
            outcome="reuse",
            court_source="profile_reuse",
            profile=match.profile,
            match_confidence=match.match_confidence,
            color_delta_e2000=match.color_delta_e2000,
            verification=verification,
            needs_profile_refresh_offer=False,
            reason="fingerprint_color_and_four_line_passed",
        )
    if len(reusable) > 1:
        best_match, best_verification = reusable[0]
        return CourtProfileReuseDecision(
            outcome="needs_owner_tag",
            court_source="needs_owner_tag",
            profile=None,
            match_confidence=best_match.match_confidence,
            color_delta_e2000=best_match.color_delta_e2000,
            verification=best_verification,
            needs_profile_refresh_offer=False,
            reason="multiple_profiles_passed_same_fingerprint_color_and_geometry",
        )
    if color_matched_failures:
        best_match, best_verification = color_matched_failures[0]
        return CourtProfileReuseDecision(
            outcome="fall_through_refresh_offer",
            court_source="generic_path",
            profile=None,
            match_confidence=best_match.match_confidence,
            color_delta_e2000=best_match.color_delta_e2000,
            verification=best_verification,
            needs_profile_refresh_offer=True,
            reason="fingerprint_and_color_matched_but_four_line_failed",
        )
    return CourtProfileReuseDecision(
        outcome="generic_path",
        court_source="generic_path",
        profile=None,
        match_confidence=0.0,
        color_delta_e2000=None,
        verification=None,
        needs_profile_refresh_offer=False,
        reason="no_fingerprint_and_color_profile_match",
    )


def _sample_segment(segment: Segment2, *, samples_per_line: int) -> list[tuple[float, float]]:
    (x1, y1), (x2, y2) = segment
    return [
        (x1 + (x2 - x1) * index / float(samples_per_line - 1), y1 + (y2 - y1) * index / float(samples_per_line - 1))
        for index in range(samples_per_line)
    ]


def _segment(points: Sequence[Sequence[float]]) -> Segment2:
    if len(points) != 2:
        raise ValueError("segment requires exactly two points")
    return (float(points[0][0]), float(points[0][1])), (float(points[1][0]), float(points[1][1]))


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[int(rank)]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("court profile matching requires opencv-python") from exc
    return cv2


def _np() -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("court profile matching requires numpy") from exc
    return np


__all__ = [
    "COURT_PROFILE_REUSE_MEDIAN_BAR_PX",
    "COURT_PROFILE_REUSE_MIN_RECOVERED_LINES",
    "COURT_PROFILE_REUSE_P95_BAR_PX",
    "CourtProfileReuseDecision",
    "FourLineVerification",
    "OUTER_COURT_LINE_IDS",
    "decide_court_profile_reuse",
    "line_color_lab_from_projected_lines",
    "project_outer_court_lines",
    "verify_outer_court_lines",
]
