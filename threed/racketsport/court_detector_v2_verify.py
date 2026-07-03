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
