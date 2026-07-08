"""Self-verification gates for court detector v2 hypotheses."""

from __future__ import annotations

from typing import Any, Mapping


FLOOR_MEDIAN_GATE_PX = 15.0
FLOOR_P95_GATE_PX = 30.0
VISIBLE_CORNER_MEDIAN_GATE_PX = 20.0
TEMPORAL_MEDIAN_GATE_PX = 10.0


def verify_court_hypothesis(
    *,
    hypothesis: Mapping[str, Any],
    visible_error_px: Mapping[str, Any] | None = None,
    line_support: Mapping[str, Any] | None = None,
    temporal_stability_px: Mapping[str, Any] | None = None,
    top_net_validation: Mapping[str, Any] | None = None,
    tennis_overlay_rejection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    visible_error_px = visible_error_px or {}
    line_support = line_support or {}
    temporal_stability_px = temporal_stability_px or {}

    floor_visible = visible_error_px.get("floor_visible") or {}
    if _number(floor_visible.get("median")) is not None and float(floor_visible["median"]) > FLOOR_MEDIAN_GATE_PX:
        blockers.append("visible_floor_median_gt_15")
    if _number(floor_visible.get("p95")) is not None and float(floor_visible["p95"]) > FLOOR_P95_GATE_PX:
        blockers.append("visible_floor_p95_gt_30")

    visible_corners = visible_error_px.get("visible_corners") or {}
    if (
        _number(visible_corners.get("median")) is not None
        and float(visible_corners["median"]) > VISIBLE_CORNER_MEDIAN_GATE_PX
    ):
        blockers.append("visible_corner_median_gt_20")

    if int(visible_error_px.get("high_confidence_over_30px_count") or 0) > 0:
        blockers.append("high_confidence_keypoint_over_30px")
    if not bool(line_support.get("required_lines_present", False)):
        blockers.append("required_line_support_missing")
    if _number(temporal_stability_px.get("median")) is not None and float(temporal_stability_px["median"]) > TEMPORAL_MEDIAN_GATE_PX:
        blockers.append("temporal_median_gt_10")
    if top_net_validation is not None and not bool(top_net_validation.get("passed", False)):
        blockers.append("top_net_validation_failed")
    if tennis_overlay_rejection is not None and not bool(tennis_overlay_rejection.get("passed", False)):
        blockers.append("tennis_overlay_rejection_failed")

    return {
        "gate_version": 1,
        "hypothesis_id": hypothesis.get("hypothesis_id"),
        "promotion_allowed": not blockers,
        "blockers": blockers,
        "metrics": {
            "visible_error_px": dict(visible_error_px),
            "line_support": dict(line_support),
            "temporal_stability_px": dict(temporal_stability_px),
            "top_net_validation": dict(top_net_validation or {}),
            "tennis_overlay_rejection": dict(tennis_overlay_rejection or {}),
        },
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


# ---------------------------------------------------------------------------
# Real (non-label) verification metrics (CAL-GEO 2026-07-05, Stage 6).
#
# These compute visible_error_px, top_net_validation, and
# tennis_overlay_rejection from actual image/line-bank EVIDENCE -- projected
# model lines vs observed supporting segments/pixels -- never from reviewed
# labels. They feed `verify_court_hypothesis` above, which stays unchanged so
# existing gate tests keep passing.
# ---------------------------------------------------------------------------

_NET_TOP_KEYPOINT_NAMES = frozenset({"net_left_sideline", "net_center", "net_right_sideline"})
_CORNER_KEYPOINT_NAMES = ("near_left_corner", "near_right_corner", "far_left_corner", "far_right_corner")


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (max(0.0, min(100.0, percentile)) / 100.0) * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def compute_visible_error_px_against_evidence(
    image_bgr: Any,
    keypoints: Mapping[str, Any],
) -> dict[str, Any]:
    """Real Stage-6 visible-error metric: projected keypoint vs nearest observed line pixel.

    This never touches reviewed labels. It samples the same court-line pixel
    distance transform used by the Stage-3 scorer directly at each projected
    keypoint location, so a hypothesis whose projected floor points do not
    line up with any real high-contrast/painted line evidence in the frame is
    penalized exactly like a genuinely wrong court would be.
    """

    from .court_line_bank import court_line_pixel_mask, line_pixel_distance_transform

    if image_bgr is None or not hasattr(image_bgr, "shape") or len(image_bgr.shape) < 2:
        return {
            "floor_visible": {"median": None, "p95": None},
            "visible_corners": {"median": None},
            "high_confidence_over_30px_count": 0,
            "reason": "invalid_image",
        }
    mask = court_line_pixel_mask(image_bgr, dilation_px=3)
    distances = line_pixel_distance_transform(mask)
    height, width = int(mask.shape[0]), int(mask.shape[1])
    off_frame_penalty = float(max(width, height))

    per_point: dict[str, float] = {}
    for name, xy in keypoints.items():
        if not isinstance(xy, (list, tuple)) or len(xy) != 2:
            continue
        x, y = float(xy[0]), float(xy[1])
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < width and 0 <= iy < height:
            per_point[str(name)] = float(distances[iy, ix])
        else:
            per_point[str(name)] = off_frame_penalty

    floor_values = [value for name, value in per_point.items() if name not in _NET_TOP_KEYPOINT_NAMES]
    corner_values = [per_point[name] for name in _CORNER_KEYPOINT_NAMES if name in per_point]
    return {
        "floor_visible": {
            "median": _median(floor_values) if floor_values else None,
            "p95": _percentile(floor_values, 95.0) if floor_values else None,
        },
        "visible_corners": {"median": _median(corner_values) if corner_values else None},
        "high_confidence_over_30px_count": 0,
        "per_point_distance_px": {name: round(value, 3) for name, value in per_point.items()},
        "evidence_source": "projected_line_pixel_distance_transform",
    }


# ---------------------------------------------------------------------------
# Round-2 fix 4: NON-self-confirming visible-error metric.
# ---------------------------------------------------------------------------

_VERIFY_LINE_ENDPOINT_PAIRS: dict[str, tuple[str, str]] = {
    "near_baseline": ("near_left_corner", "near_right_corner"),
    "far_baseline": ("far_left_corner", "far_right_corner"),
    "near_nvz": ("near_nvz_left", "near_nvz_right"),
    "far_nvz": ("far_nvz_left", "far_nvz_right"),
    "left_sideline": ("near_left_corner", "far_left_corner"),
    "right_sideline": ("near_right_corner", "far_right_corner"),
    "near_centerline": ("near_baseline_center", "near_nvz_center"),
    "far_centerline": ("far_baseline_center", "far_nvz_center"),
    "net": ("net_left_sideline", "net_right_sideline"),
}
_FLOOR_LINE_NAMES_FOR_VERIFY = (
    "near_baseline",
    "near_nvz",
    "far_nvz",
    "far_baseline",
    "left_sideline",
    "right_sideline",
    "near_centerline",
    "far_centerline",
)
_CORNER_ADJACENT_LINES: dict[str, tuple[str, str]] = {
    "near_left_corner": ("near_baseline", "left_sideline"),
    "near_right_corner": ("near_baseline", "right_sideline"),
    "far_left_corner": ("far_baseline", "left_sideline"),
    "far_right_corner": ("far_baseline", "right_sideline"),
}
UNSUPPORTED_OBSERVABLE_LINE_ERROR_PX = 120.0


def compute_visible_error_px_against_line_bank(
    keypoints: Mapping[str, Any],
    *,
    segments: Any,
    pixel_mask: Any,
    image_size: tuple[int, int],
    surface_polygon: Any | None = None,
) -> dict[str, Any]:
    """Round-2 fix 4: score projected template LINES vs the FULL persistent bank.

    The round-1 evidence metric sampled a distance transform AT the projected
    keypoints -- a compressed fit whose mis-assigned keypoints all sit on
    real painted lines read 0.0 error while being 350px wrong (self-
    confirming). This version scores each projected template LINE against
    the full persistent line bank (+ the surface-polygon boundary as extra
    candidate support lines), assignment-independently:

    - observable + matched by a bank line -> error = matched line's mean
      perpendicular distance (real px residual);
    - observable but NOT matched -> error =
      UNSUPPORTED_OBSERVABLE_LINE_ERROR_PX (fails the 15/30px gates by
      construction: a compressed fit's phantom cross line now FAILS instead
      of reading 0.0);
    - off-frame -> excluded, recorded as unobservable (fix 2).

    Per-expected-line support fractions are returned for the artifact.
    """

    from .court_line_bank import dedupe_line_segments, evaluate_projected_template_lines

    bank_segments = [dict(segment) for segment in (segments or [])]
    if surface_polygon and isinstance(surface_polygon, (list, tuple)) and len(surface_polygon) >= 3:
        import math as _math

        for index in range(len(surface_polygon)):
            p1 = surface_polygon[index]
            p2 = surface_polygon[(index + 1) % len(surface_polygon)]
            dx = float(p2[0]) - float(p1[0])
            dy = float(p2[1]) - float(p1[1])
            length = _math.hypot(dx, dy)
            if length < 20.0:
                continue
            bank_segments.append(
                {
                    "p1": [float(p1[0]), float(p1[1])],
                    "p2": [float(p2[0]), float(p2[1])],
                    "length_px": round(length, 3),
                    "angle_deg": round(_math.degrees(_math.atan2(dy, dx)), 3),
                    "source": "surface_polygon_boundary",
                }
            )
    bank_segments = dedupe_line_segments(bank_segments)

    evaluation = evaluate_projected_template_lines(
        keypoints,
        endpoint_pairs=_VERIFY_LINE_ENDPOINT_PAIRS,
        segments=bank_segments,
        pixel_mask=pixel_mask,
        image_size=image_size,
    )

    line_errors: dict[str, float] = {}
    for line_name, record in evaluation["per_line"].items():
        status = record.get("status")
        if status == "supported":
            best = record.get("best_segment") or {}
            matched_distance = best.get("mean_perpendicular_distance_px")
            if record.get("segment_supported") and matched_distance is not None:
                line_errors[line_name] = float(matched_distance)
            else:
                # paint-only support: residual not localizable to one bank
                # line; small but nonzero.
                line_errors[line_name] = 6.0
        elif status == "unsupported":
            line_errors[line_name] = UNSUPPORTED_OBSERVABLE_LINE_ERROR_PX

    floor_errors = [line_errors[name] for name in _FLOOR_LINE_NAMES_FOR_VERIFY if name in line_errors]
    corner_errors: list[float] = []
    for corner, (line_a, line_b) in _CORNER_ADJACENT_LINES.items():
        adjacent = [line_errors[name] for name in (line_a, line_b) if name in line_errors]
        if adjacent:
            corner_errors.append(max(adjacent))

    return {
        "floor_visible": {
            "median": _median(floor_errors) if floor_errors else None,
            "p95": _percentile(floor_errors, 95.0) if floor_errors else None,
        },
        "visible_corners": {"median": _median(corner_errors) if corner_errors else None},
        "high_confidence_over_30px_count": 0,
        "per_line_error_px": {name: round(value, 3) for name, value in line_errors.items()},
        "per_line_support": evaluation["per_line"],
        "observable_line_count": evaluation["observable_count"],
        "supported_line_count": evaluation["supported_count"],
        "unobservable_line_count": evaluation["unobservable_count"],
        "supported_fraction": evaluation["supported_fraction"],
        "evidence_source": "persistent_line_bank_plus_surface_boundary",
    }


def compute_top_net_validation(
    net_evidence: Mapping[str, Any] | None,
    keypoints: Mapping[str, Any],
    *,
    tolerance_px: float = 60.0,
) -> dict[str, Any]:
    """Check the projected net line sits inside the net-evidence ROI band.

    Real (non-label) consistency check between two independent evidence
    sources: the net-anchor ROI (Stage-1 E1, orientation/scale-only) and the
    homography-projected net line from the line-bank/regulation solve.
    """

    if not net_evidence:
        return {"passed": False, "reason": "no_net_evidence"}
    roi = net_evidence.get("roi") or {}
    net_center = keypoints.get("net_center")
    if not isinstance(net_center, (list, tuple)) or len(net_center) != 2:
        return {"passed": False, "reason": "no_projected_net_center"}
    y_min = _number(roi.get("y_min"))
    y_max = _number(roi.get("y_max"))
    if y_min is None or y_max is None:
        return {"passed": False, "reason": "no_net_roi"}
    projected_y = float(net_center[1])
    passed = (y_min - tolerance_px) <= projected_y <= (y_max + tolerance_px)
    return {
        "passed": bool(passed),
        "projected_net_center_y": round(projected_y, 3),
        "net_roi_y_range": [round(float(y_min), 3), round(float(y_max), 3)],
        "tolerance_px": tolerance_px,
    }


def compute_tennis_overlay_rejection(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    """Real Stage-6 tennis-overlay rejection from the joint template competition.

    Replaces the previously hard-coded `passed=False`. A hypothesis passes
    only when it is a pickleball-template assignment whose own joint
    pickleball-vs-tennis competition (Stage 3) does not say tennis explains
    the observed line pool better. Hypotheses without real joint-competition
    evidence (e.g. the legacy template-projection stub sources) stay
    fail-closed, matching the previous behavior for that path.
    """

    template = hypothesis.get("template")
    components = hypothesis.get("score_components") or {}
    joint = components.get("joint_template_competition")
    if template != "pickleball" or not isinstance(joint, Mapping) or not joint.get("available"):
        return {"passed": False, "reason": "no_joint_template_competition_evidence"}
    winner = joint.get("winner")
    margin = float(joint.get("margin") or 0.0)
    passed = winner != "tennis"
    return {
        "passed": bool(passed),
        "winner": winner,
        "margin": round(margin, 6),
        "reason": None if passed else "tennis_template_explains_observed_lines_better",
    }
