"""Classify ball bounces against regulation court and NVZ lines."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .court_templates import Sport, ft_to_m, get_court_template
from .court_zones import classify_point
from .schemas import BallLineCalls, BallTrack

BOUNDARY_LINE_IDS = ("near_baseline", "far_baseline", "left_sideline", "right_sideline")


def classify_ball_line_calls(
    ball_track: BallTrack | Mapping[str, Any],
    *,
    sport: Sport = "pickleball",
    uncertainty_radius_m: float = 0.05,
    source: str = "cpu_rule_geometry_v1",
    input_ball_track: str | None = None,
) -> dict[str, Any]:
    """Return schema-valid bounce location calls from a BallTrack payload.

    This classifies where a bounce landed. It does not decide player kitchen
    faults, volley legality, serve legality, or contact timing.
    """

    if uncertainty_radius_m < 0.0:
        raise ValueError("uncertainty_radius_m must be non-negative")
    track = BallTrack.model_validate(ball_track)
    calls = [
        classify_bounce(
            bounce.t,
            bounce.world_xy,
            sport=sport,
            uncertainty_radius_m=float(uncertainty_radius_m),
        )
        for bounce in track.bounces
    ]
    summary = _summary(calls)
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_line_calls",
        "sport": sport,
        "source": source,
        "rule_scope": "ball_bounce_location_only",
        "world_frame": "court_Z0",
        "input_ball_track": input_ball_track,
        "uncertainty_radius_m": float(uncertainty_radius_m),
        "calls": calls,
        "summary": summary,
        "not_gate_verified": True,
    }
    BallLineCalls.model_validate(payload)
    return payload


def classify_bounce(
    t: float,
    world_xy: list[float],
    *,
    sport: Sport = "pickleball",
    uncertainty_radius_m: float = 0.05,
) -> dict[str, Any]:
    if uncertainty_radius_m < 0.0:
        raise ValueError("uncertainty_radius_m must be non-negative")
    x, y = float(world_xy[0]), float(world_xy[1])
    template = get_court_template(sport)
    half_width_m = template.width_m / 2.0
    half_length_m = template.length_m / 2.0
    boundary_margin = min(
        x + half_width_m,
        half_width_m - x,
        y + half_length_m,
        half_length_m - y,
    )
    nearest_boundary_line_id = _nearest_line_id(world_xy, template.line_segments_m, BOUNDARY_LINE_IDS)
    reasons: list[str] = []
    court_call = _signed_margin_call(boundary_margin, uncertainty_radius_m)
    if court_call == "unknown":
        reasons.append("boundary_within_uncertainty")
    elif court_call == "in":
        reasons.append("inside_or_on_court")
    else:
        reasons.append("outside_court")
    if boundary_margin == 0.0:
        reasons.append("on_boundary_line")

    zone = classify_point(sport, [x, y]) if court_call != "out" else None
    kitchen_call, kitchen_margin, nearest_kitchen_line_id, kitchen_reasons = _kitchen_call(
        sport=sport,
        x=x,
        y=y,
        half_width_m=half_width_m,
        court_call=court_call,
        uncertainty_radius_m=uncertainty_radius_m,
    )
    reasons.extend(kitchen_reasons)
    confidence = _confidence(court_call, kitchen_call, boundary_margin, kitchen_margin, uncertainty_radius_m)
    return {
        "t": float(t),
        "world_xy": [x, y],
        "court_call": court_call,
        "kitchen_call": kitchen_call,
        "zone": zone,
        "nearest_boundary_line_id": nearest_boundary_line_id,
        "nearest_kitchen_line_id": nearest_kitchen_line_id,
        "boundary_margin_m": float(boundary_margin),
        "kitchen_margin_m": kitchen_margin,
        "uncertainty_radius_m": float(uncertainty_radius_m),
        "confidence": confidence,
        "reasons": reasons,
    }


def _signed_margin_call(margin_m: float, uncertainty_radius_m: float) -> str:
    if uncertainty_radius_m == 0.0 and margin_m >= 0.0:
        return "in"
    if margin_m > uncertainty_radius_m:
        return "in"
    if margin_m < -uncertainty_radius_m:
        return "out"
    return "unknown"


def _kitchen_call(
    *,
    sport: Sport,
    x: float,
    y: float,
    half_width_m: float,
    court_call: str,
    uncertainty_radius_m: float,
) -> tuple[str, float | None, str | None, list[str]]:
    if sport != "pickleball":
        return "unknown", None, None, ["no_nvz_for_sport"]
    if court_call == "out":
        return "unknown", None, _nearest_nvz_line_id(y), ["outside_court"]
    template = get_court_template("pickleball")
    if template.non_volley_zone_ft is None:
        return "unknown", None, None, ["missing_nvz_template"]
    if abs(x) > half_width_m + uncertainty_radius_m:
        return "unknown", None, _nearest_nvz_line_id(y), ["outside_court_width_for_nvz"]
    nvz_m = ft_to_m(template.non_volley_zone_ft)
    margin = nvz_m - abs(y)
    nearest_line = _nearest_nvz_line_id(y)
    if uncertainty_radius_m == 0.0 and margin >= 0.0:
        return "nvz", float(margin), nearest_line, ["inside_or_on_nvz"]
    if margin > uncertainty_radius_m:
        return "nvz", float(margin), nearest_line, ["inside_nvz"]
    if margin < -uncertainty_radius_m:
        return "non_nvz", float(margin), nearest_line, ["outside_nvz"]
    return "unknown", float(margin), nearest_line, ["nvz_boundary_within_uncertainty"]


def _nearest_nvz_line_id(y: float) -> str:
    return "near_nvz" if y < 0.0 else "far_nvz"


def _nearest_line_id(
    point: list[float],
    segments: Mapping[str, tuple[list[float], list[float]]],
    candidate_ids: tuple[str, ...],
) -> str | None:
    best: tuple[float, str] | None = None
    for line_id in candidate_ids:
        segment = segments.get(line_id)
        if segment is None:
            continue
        distance = _distance_to_segment(point, segment[0], segment[1])
        if best is None or distance < best[0]:
            best = (distance, line_id)
    return best[1] if best is not None else None


def _distance_to_segment(point: list[float], start: list[float], end: list[float]) -> float:
    px, py = float(point[0]), float(point[1])
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return math.hypot(px - x1, py - y1)
    u = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    closest_x = x1 + u * dx
    closest_y = y1 + u * dy
    return math.hypot(px - closest_x, py - closest_y)


def _confidence(
    court_call: str,
    kitchen_call: str,
    boundary_margin: float,
    kitchen_margin: float | None,
    uncertainty_radius_m: float,
) -> float:
    if court_call == "unknown":
        return 0.0
    if kitchen_call == "unknown" and kitchen_margin is not None:
        return 0.0
    if uncertainty_radius_m == 0.0:
        return 1.0
    margins = [abs(boundary_margin)]
    if kitchen_margin is not None:
        margins.append(abs(kitchen_margin))
    margin = min(margins)
    return max(0.0, min(1.0, margin / (margin + uncertainty_radius_m)))


def _summary(calls: list[dict[str, Any]]) -> dict[str, Any]:
    court_counts = {"in": 0, "out": 0, "unknown": 0}
    kitchen_counts = {"non_nvz": 0, "nvz": 0, "unknown": 0}
    for call in calls:
        court_counts[call["court_call"]] += 1
        kitchen_counts[call["kitchen_call"]] += 1
    reasons: list[str] = []
    status = "ready"
    if not calls:
        status = "blocked"
        reasons.append("ball_track has no bounces")
    return {
        "status": status,
        "total_bounces": len(calls),
        "court_call_counts": court_counts,
        "kitchen_call_counts": kitchen_counts,
        "reasons": reasons,
    }


__all__ = ["classify_ball_line_calls", "classify_bounce"]
