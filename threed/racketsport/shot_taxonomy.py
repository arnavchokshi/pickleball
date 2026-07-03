"""PB-Vision-compatible shot taxonomy rules over arc-solver outputs.

This is a deterministic rule layer. It does not train a classifier, does not
touch protected evaluation labels, and treats the arc-solver payload as
render/internal-val evidence only.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_shots"
PB_VISION_VERSION = "v4.10.0-compatible-public-taxonomy"
BALL_RADIUS_M = 0.0371
MPS_TO_MPH = 2.2369362920544
DEFAULT_MIN_SHOT_TYPE_CONFIDENCE = 0.45
DEFAULT_LET_BAND_M = 0.08
DEFAULT_COURT_BOUNDS = (-3.048, 3.048, -6.7056, 6.7056)
DEFAULT_NVZ_M = 2.1336
SHOT_TYPES = ("smash", "lob", "dink", "drop", "drive", "atp", "erne", "tweener")


def classify_shots_from_payloads(
    *,
    clip_id: str,
    ball_arc_payload: Mapping[str, Any],
    events_selected_payload: Mapping[str, Any],
    court_zones_payload: Mapping[str, Any] | None = None,
    net_plane_payload: Mapping[str, Any] | None = None,
    tracks_payload: Mapping[str, Any] | None = None,
    min_shot_type_confidence: float = DEFAULT_MIN_SHOT_TYPE_CONFIDENCE,
    let_band_m: float = DEFAULT_LET_BAND_M,
) -> dict[str, Any]:
    """Classify selected contact events into PB-Vision-style shot records."""

    if not clip_id:
        raise ValueError("clip_id is required")
    if min_shot_type_confidence < 0.0 or min_shot_type_confidence > 1.0:
        raise ValueError("min_shot_type_confidence must be in [0, 1]")
    if let_band_m < 0.0:
        raise ValueError("let_band_m must be non-negative")

    selected = _selected_events(events_selected_payload)
    contacts = [event for event in selected if event.get("kind") == "contact"]
    segments = _segments_by_start(ball_arc_payload)
    selected_index = {str(event.get("anchor_id")): index for index, event in enumerate(selected)}
    selected_by_id = {str(event.get("anchor_id")): event for event in selected if event.get("anchor_id") is not None}
    court = _CourtGeometry.from_payload(court_zones_payload)
    track_lookup = _TrackLookup(tracks_payload)

    shots: list[dict[str, Any]] = []
    for contact_index, event in enumerate(contacts, start=1):
        anchor_id = str(event.get("anchor_id", ""))
        segment = segments.get(anchor_id)
        shot = _classify_contact(
            clip_id=clip_id,
            event=event,
            contact_index=contact_index,
            segment=segment,
            selected_events=selected,
            selected_index=selected_index,
            selected_by_id=selected_by_id,
            court=court,
            track_lookup=track_lookup,
            min_shot_type_confidence=min_shot_type_confidence,
            let_band_m=let_band_m,
        )
        shots.append(shot)

    shot_type_counts = Counter(str(shot["shot_type"]) for shot in shots if "shot_type" in shot)
    outcome_counts = Counter(str(shot.get("outcome", {}).get("call", "unknown")) for shot in shots)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip_id": clip_id,
        "taxonomy": {
            "source": "PB Vision public insights JSON-Schema",
            "pb_vision_schema_version": PB_VISION_VERSION,
            "shot_type_enum": list(SHOT_TYPES),
            "serve_return_index_separate": True,
            "rule_layer": "arc_solver_geometry_v1",
        },
        "policy": {
            "internal_val_only": True,
            "not_ground_truth": True,
            "not_for_detection_metrics": True,
            "abstain_below_confidence": float(min_shot_type_confidence),
        },
        "inputs": {
            "ball_arc_artifact_type": ball_arc_payload.get("artifact_type"),
            "events_artifact_type": events_selected_payload.get("artifact_type"),
            "arc_render_only": bool(ball_arc_payload.get("render_only", False)),
            "arc_not_for_detection_metrics": bool(ball_arc_payload.get("not_for_detection_metrics", False)),
        },
        "shots": shots,
        "summary": {
            "shot_count": len(shots),
            "classified_count": sum(1 for shot in shots if "shot_type" in shot),
            "abstained_count": sum(1 for shot in shots if shot.get("shot_type_abstained")),
            "shot_type_counts": dict(sorted(shot_type_counts.items())),
            "outcome_counts": dict(sorted(outcome_counts.items())),
        },
    }


def _classify_contact(
    *,
    clip_id: str,
    event: Mapping[str, Any],
    contact_index: int,
    segment: Mapping[str, Any] | None,
    selected_events: Sequence[Mapping[str, Any]],
    selected_index: Mapping[str, int],
    selected_by_id: Mapping[str, Mapping[str, Any]],
    court: "_CourtGeometry",
    track_lookup: "_TrackLookup",
    min_shot_type_confidence: float,
    let_band_m: float,
) -> dict[str, Any]:
    anchor_id = str(event.get("anchor_id", f"contact_{contact_index:03d}"))
    warnings: list[str] = []
    event_confidence = _event_confidence(event)
    launch = _vec3(event.get("world_xyz")) or [0.0, 0.0, 0.0]
    player_id = event.get("player_id")
    player_position = track_lookup.nearest(player_id, _float(event.get("t"), 0.0))

    if segment is None:
        confidence = round(event_confidence * 0.35, 6)
        warnings.append("missing_outgoing_segment")
        shot = {
            "shot_id": f"{clip_id}:{anchor_id}",
            "event_anchor_id": anchor_id,
            "player_id": player_id,
            "frame": _int(event.get("frame"), 0),
            "t": _round(_float(event.get("t"), 0.0), 9),
            "rally_index": _rally_index(contact_index, None),
            "shot_type_abstained": True,
            "confidence": confidence,
            "confidence_factors": {"event_confidence": event_confidence},
            "launch_world_xyz": launch,
            "launch_zone": court.zone(launch[:2]),
            "outcome": {"call": "unknown", "faults": ["missing_outgoing_segment"]},
            "warnings": warnings,
        }
        if player_position is not None:
            shot["player_position_world_xy"] = player_position
        return shot

    landing = _landing_from_segment(segment)
    launch = _vec3(segment.get("initial_position_m")) or launch
    velocity = _vec3(segment.get("initial_velocity_mps")) or [0.0, 0.0, 0.0]
    speed_mps = _float(segment.get("initial_speed_mps"), _norm(velocity))
    speed_mph = speed_mps * MPS_TO_MPH
    peak_height_m = _peak_height(segment, launch)
    launch_zone = court.zone(launch[:2])
    landing_zone = court.zone(landing["world_xyz"][:2])
    ellipse = _uncertainty_ellipse(event, segment, landing["world_xyz"])
    line_call = court.line_call(landing["world_xyz"][:2], uncertainty_radius_m=ellipse["semi_major_m"])
    confidence, factors = _segment_confidence(event_confidence, segment, speed_mph)

    if speed_mph < 5.0 or speed_mph > 50.0:
        warnings.append("speed_outside_sanity_range_5_50_mph")
    if confidence < min_shot_type_confidence:
        warnings.append("low_segment_confidence")

    features = _ShotFeatures(
        launch=launch,
        landing=landing["world_xyz"],
        velocity=velocity,
        speed_mps=speed_mps,
        speed_mph=speed_mph,
        peak_height_m=peak_height_m,
        launch_zone=launch_zone,
        landing_zone=landing_zone,
        court=court,
        segment=segment,
        contact_index=contact_index,
    )
    shot_type, shot_reasons = _classify_shot_type(features)
    if contact_index == 3 and shot_type not in {"drop", "drive", "lob"}:
        third_shot = "hybrid"
    else:
        third_shot = shot_type

    outcome = _classify_outcome(
        segment=segment,
        landing=landing,
        line_call=line_call,
        selected_events=selected_events,
        selected_index=selected_index,
        selected_by_id=selected_by_id,
        let_band_m=let_band_m,
    )

    shot: dict[str, Any] = {
        "shot_id": f"{clip_id}:{anchor_id}",
        "event_anchor_id": anchor_id,
        "segment_id": _int(segment.get("segment_id"), -1),
        "player_id": player_id,
        "frame": _int(event.get("frame"), _int(segment.get("frame_start"), 0)),
        "t": _round(_float(event.get("t"), _float(segment.get("t0"), 0.0)), 9),
        "rally_index": _rally_index(contact_index, third_shot),
        "speed_mph": _round(speed_mph, 3),
        "peak_height_m": _round(peak_height_m, 6),
        "launch_world_xyz": [_round(value, 6) for value in launch],
        "launch_zone": launch_zone,
        "landing": {
            "world_xyz": [_round(value, 6) for value in landing["world_xyz"]],
            "source": landing["source"],
            "zone": landing_zone,
            "line_call": line_call,
            "uncertainty_ellipse": ellipse,
        },
        "outcome": outcome,
        "confidence": _round(confidence, 6),
        "confidence_factors": factors,
        "rule_evidence": {
            "shot_type_candidate": shot_type,
            "shot_type_reasons": shot_reasons,
            "net_clearance_m": _optional_round(segment.get("net_clearance_m"), 6),
            "net_clearance_ok": segment.get("net_clearance_ok"),
            "initial_velocity_mps": [_round(value, 6) for value in velocity],
        },
        "warnings": warnings,
    }
    if player_position is not None:
        shot["player_position_world_xy"] = player_position
    if confidence >= min_shot_type_confidence and shot_type is not None:
        shot["shot_type"] = shot_type
        shot["shot_type_abstained"] = False
    else:
        shot["shot_type_abstained"] = True
    return shot


def _selected_events(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    selected = payload.get("selected", [])
    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)):
        raise ValueError("events_selected.selected must be a list")
    events: list[Mapping[str, Any]] = []
    for index, event in enumerate(selected):
        if not isinstance(event, Mapping):
            raise ValueError(f"events_selected.selected/{index} must be an object")
        if event.get("selected", True) is False:
            continue
        events.append(event)
    return sorted(events, key=lambda item: (_float(item.get("t"), 0.0), _int(item.get("frame"), 0), str(item.get("anchor_id", ""))))


def _segments_by_start(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    segments = payload.get("segments", [])
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        raise ValueError("ball_track_arc_solved.segments must be a list")
    by_start: dict[str, Mapping[str, Any]] = {}
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise ValueError(f"ball_track_arc_solved.segments/{index} must be an object")
        start = segment.get("start_anchor")
        if start is not None:
            by_start.setdefault(str(start), segment)
    return by_start


def _event_confidence(event: Mapping[str, Any]) -> float:
    for key in ("candidate_confidence", "confidence", "conf"):
        if key in event:
            return _clamp(_float(event.get(key), 0.65), 0.0, 1.0)
    details = event.get("details")
    if isinstance(details, Mapping):
        for key in ("event_confidence", "joint_confidence"):
            if key in details:
                return _clamp(_float(details.get(key), 0.65), 0.0, 1.0)
    return 0.65


def _landing_from_segment(segment: Mapping[str, Any]) -> dict[str, Any]:
    end_anchor = _anchor(segment, end=True)
    if end_anchor is not None and end_anchor.get("kind") == "bounce":
        end_world = _vec3(end_anchor.get("world_xyz"))
        if end_world is not None:
            return {"world_xyz": end_world, "source": "end_bounce_anchor"}

    p0 = _vec3(segment.get("initial_position_m"))
    v0 = _vec3(segment.get("initial_velocity_mps"))
    if p0 is not None and v0 is not None:
        dt = _ground_intersection_dt(p0[2], v0[2], BALL_RADIUS_M)
        if dt is not None:
            return {
                "world_xyz": [p0[0] + v0[0] * dt, p0[1] + v0[1] * dt, BALL_RADIUS_M],
                "source": "ballistic_ground_intersection",
            }

    if end_anchor is not None:
        end_world = _vec3(end_anchor.get("world_xyz"))
        if end_world is not None:
            return {"world_xyz": end_world, "source": "end_anchor_fallback"}
    p0 = p0 or [0.0, 0.0, 0.0]
    return {"world_xyz": p0, "source": "launch_fallback"}


def _ground_intersection_dt(z0: float, vz: float, target_z: float) -> float | None:
    gravity = 9.80665
    discriminant = vz * vz + 2.0 * gravity * (z0 - target_z)
    if discriminant < 0.0:
        return None
    root = math.sqrt(discriminant)
    candidates = [(vz + root) / gravity, (vz - root) / gravity]
    positive = [value for value in candidates if value > 0.02]
    if not positive:
        return None
    return min(positive)


def _anchor(segment: Mapping[str, Any], *, end: bool) -> Mapping[str, Any] | None:
    anchors = segment.get("anchors_used")
    if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)) or not anchors:
        return None
    index = -1 if end else 0
    anchor = anchors[index]
    return anchor if isinstance(anchor, Mapping) else None


def _peak_height(segment: Mapping[str, Any], launch: Sequence[float]) -> float:
    physical = segment.get("physical_sanity")
    if isinstance(physical, Mapping) and physical.get("apex_height_m") is not None:
        return _float(physical.get("apex_height_m"), launch[2])
    velocity = _vec3(segment.get("initial_velocity_mps"))
    if velocity is None:
        return float(launch[2])
    vz = velocity[2]
    if vz <= 0.0:
        return float(launch[2])
    return float(launch[2]) + (vz * vz) / (2.0 * 9.80665)


def _uncertainty_ellipse(event: Mapping[str, Any], segment: Mapping[str, Any], landing: Sequence[float]) -> dict[str, Any]:
    launch = _vec3(segment.get("initial_position_m")) or _vec3(event.get("world_xyz")) or [0.0, 0.0, 0.0]
    event_sigma = max(0.0, _float(event.get("sigma_m"), 0.12))
    end_anchor = _anchor(segment, end=True)
    end_sigma = max(0.0, _float(end_anchor.get("sigma_m"), 0.10)) if end_anchor else 0.10
    endpoint_term = max(0.0, _float(segment.get("endpoint_error_m"), 0.0)) * 0.25
    rmse_term = max(0.0, _float(segment.get("reprojection_rmse_px"), 0.0)) * 0.01
    semi_major = max(0.05, event_sigma, end_sigma, endpoint_term, rmse_term)
    semi_minor = max(0.03, min(semi_major * 0.75, max(event_sigma, end_sigma, rmse_term * 0.5, 0.03)))
    angle_deg = math.degrees(math.atan2(float(landing[1]) - launch[1], float(landing[0]) - launch[0]))
    return {
        "center_xy": [_round(float(landing[0]), 6), _round(float(landing[1]), 6)],
        "semi_major_m": _round(semi_major, 6),
        "semi_minor_m": _round(semi_minor, 6),
        "angle_deg": _round(angle_deg, 3),
        "source": "segment_sigma_endpoint_reprojection_v1",
    }


def _segment_confidence(event_confidence: float, segment: Mapping[str, Any], speed_mph: float) -> tuple[float, dict[str, float]]:
    inliers = max(0, _int(segment.get("inlier_count"), 0))
    outliers = max(0, _int(segment.get("outlier_count"), 0))
    inlier_ratio = inliers / (inliers + outliers) if inliers + outliers > 0 else 0.0
    endpoint_error = max(0.0, _float(segment.get("endpoint_error_m"), 10.0))
    rmse = max(0.0, _float(segment.get("reprojection_rmse_px"), 20.0))
    endpoint_quality = 1.0 / (1.0 + endpoint_error / 2.0)
    reprojection_quality = 1.0 / (1.0 + rmse / 20.0)
    if 5.0 <= speed_mph <= 50.0:
        speed_quality = 1.0
    elif 3.0 <= speed_mph <= 60.0:
        speed_quality = 0.35
    else:
        speed_quality = 0.0
    confidence = (
        0.35 * event_confidence
        + 0.25 * inlier_ratio
        + 0.20 * endpoint_quality
        + 0.15 * reprojection_quality
        + 0.05 * speed_quality
    )
    physical = segment.get("physical_sanity")
    if isinstance(physical, Mapping) and physical.get("violation"):
        confidence -= 0.20
    confidence = _clamp(confidence, 0.0, 1.0)
    return confidence, {
        "event_confidence": _round(event_confidence, 6),
        "inlier_ratio": _round(inlier_ratio, 6),
        "endpoint_quality": _round(endpoint_quality, 6),
        "reprojection_quality": _round(reprojection_quality, 6),
        "speed_quality": _round(speed_quality, 6),
    }


class _ShotFeatures:
    def __init__(
        self,
        *,
        launch: Sequence[float],
        landing: Sequence[float],
        velocity: Sequence[float],
        speed_mps: float,
        speed_mph: float,
        peak_height_m: float,
        launch_zone: str | None,
        landing_zone: str | None,
        court: "_CourtGeometry",
        segment: Mapping[str, Any],
        contact_index: int,
    ) -> None:
        self.launch = launch
        self.landing = landing
        self.velocity = velocity
        self.speed_mps = speed_mps
        self.speed_mph = speed_mph
        self.peak_height_m = peak_height_m
        self.launch_zone = launch_zone
        self.landing_zone = landing_zone
        self.court = court
        self.segment = segment
        self.contact_index = contact_index


def _classify_shot_type(features: _ShotFeatures) -> tuple[str | None, list[str]]:
    reasons: list[str] = []
    launch_x, launch_y, launch_z = [float(value) for value in features.launch]
    landing_x, landing_y, _ = [float(value) for value in features.landing]
    half_width = features.court.half_width
    half_length = features.court.half_length
    nvz = features.court.nvz_m
    x_at_net = _x_at_y(features.launch, features.landing, 0.0)
    launch_outside_sideline = abs(launch_x) > half_width + 0.20
    lands_deep = abs(landing_y) >= half_length - 1.2
    landing_near_kitchen = abs(landing_y) <= nvz + 0.35
    launch_near_kitchen = abs(launch_y) <= nvz + 0.35

    trick_shots_allowed = features.contact_index >= 3
    if (
        trick_shots_allowed
        and x_at_net is not None
        and abs(x_at_net) > half_width + 0.15
        and abs(launch_y) <= nvz + 0.75
    ):
        reasons.append("net-crossing projection outside sideline")
        return "atp", reasons
    if trick_shots_allowed and launch_outside_sideline and abs(launch_y) <= nvz + 0.75 and launch_z >= 0.85:
        reasons.append("launch outside sideline near kitchen")
        return "erne", reasons
    if trick_shots_allowed and abs(launch_y) > half_length and launch_z <= 0.60:
        reasons.append("low launch from behind baseline")
        return "tweener", reasons
    if (
        features.speed_mph >= 35.0
        and launch_z >= 1.30
        and (features.velocity[2] < 0.0 or features.peak_height_m <= launch_z + 0.25)
    ):
        reasons.append("high fast downward contact")
        return "smash", reasons
    if features.peak_height_m >= 3.0 or (features.peak_height_m >= 2.35 and lands_deep):
        reasons.append("high apex with deep landing")
        return "lob", reasons
    if launch_near_kitchen and landing_near_kitchen and features.speed_mps <= 10.0 and features.peak_height_m <= 1.8:
        reasons.append("kitchen-to-kitchen low-speed trajectory")
        return "dink", reasons
    if not launch_near_kitchen and landing_near_kitchen and features.speed_mps <= 11.5 and features.peak_height_m <= 2.3:
        reasons.append("deeper launch landing in or near kitchen")
        return "drop", reasons
    if features.speed_mps >= 5.0:
        reasons.append("default flat/deep trajectory bucket")
        return "drive", reasons
    reasons.append("insufficient geometric support")
    return None, reasons


def _classify_outcome(
    *,
    segment: Mapping[str, Any],
    landing: Mapping[str, Any],
    line_call: Mapping[str, Any],
    selected_events: Sequence[Mapping[str, Any]],
    selected_index: Mapping[str, int],
    selected_by_id: Mapping[str, Mapping[str, Any]],
    let_band_m: float,
) -> dict[str, Any]:
    net_clearance = segment.get("net_clearance_m")
    clearance = _float(net_clearance, math.nan)
    if segment.get("net_clearance_ok") is False or (math.isfinite(clearance) and clearance < 0.0):
        return {
            "call": "net_hit",
            "faults": ["net_hit"],
            "net_clearance_m": _round(clearance, 6) if math.isfinite(clearance) else None,
            "let_candidate": False,
        }

    let_candidate = math.isfinite(clearance) and 0.0 <= clearance <= let_band_m
    if _has_excess_bounce(segment, selected_events, selected_index, selected_by_id):
        faults = ["excess_bounce"]
        if let_candidate:
            faults.append("let_candidate")
        return {
            "call": "excess_bounce",
            "faults": faults,
            "let_candidate": let_candidate,
        }

    if line_call.get("call") == "out":
        out_payload = {
            "direction": line_call.get("direction"),
            "side": line_call.get("side"),
            "landed": landing.get("source") in {"end_bounce_anchor", "ballistic_ground_intersection"},
        }
        faults = ["out"]
        if let_candidate:
            faults.append("let_candidate")
        return {
            "call": "out",
            "faults": faults,
            "out": out_payload,
            "let_candidate": let_candidate,
        }

    faults = ["let_candidate"] if let_candidate else []
    return {
        "call": "in",
        "faults": faults,
        "let_candidate": let_candidate,
    }


def _has_excess_bounce(
    segment: Mapping[str, Any],
    selected_events: Sequence[Mapping[str, Any]],
    selected_index: Mapping[str, int],
    selected_by_id: Mapping[str, Mapping[str, Any]],
) -> bool:
    end_anchor = str(segment.get("end_anchor", ""))
    end_event = selected_by_id.get(end_anchor)
    if end_event is None or end_event.get("kind") != "bounce":
        end_anchor_payload = _anchor(segment, end=True)
        if end_anchor_payload is None or end_anchor_payload.get("kind") != "bounce":
            return False
    end_idx = selected_index.get(end_anchor)
    if end_idx is None:
        return False
    for event in selected_events[end_idx + 1 :]:
        kind = event.get("kind")
        if kind == "contact":
            return False
        if kind == "bounce":
            return True
    return False


def _rally_index(contact_index: int, third_shot: str | None) -> dict[str, Any]:
    if contact_index == 1:
        return {"contact_index": contact_index, "label": "serve"}
    if contact_index == 2:
        return {"contact_index": contact_index, "label": "return"}
    if contact_index == 3:
        payload = {"contact_index": contact_index, "label": "third"}
        if third_shot is not None:
            payload["third_shot"] = third_shot if third_shot in {"drop", "drive", "lob"} else "hybrid"
        return payload
    return {"contact_index": contact_index, "label": "fourth_plus"}


class _CourtGeometry:
    def __init__(self, bounds: tuple[float, float, float, float], zones: Mapping[str, Any]) -> None:
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.zones = zones
        self.half_width = max(abs(self.x_min), abs(self.x_max))
        self.half_length = max(abs(self.y_min), abs(self.y_max))
        self.nvz_m = _infer_nvz_m(zones)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "_CourtGeometry":
        zones = payload.get("zones", {}) if isinstance(payload, Mapping) else {}
        if not isinstance(zones, Mapping):
            zones = {}
        court = zones.get("court")
        if isinstance(court, Sequence) and not isinstance(court, (str, bytes)) and court:
            xs = [_float(point[0], 0.0) for point in court if isinstance(point, Sequence) and len(point) >= 2]
            ys = [_float(point[1], 0.0) for point in court if isinstance(point, Sequence) and len(point) >= 2]
            if xs and ys:
                return cls((min(xs), max(xs), min(ys), max(ys)), zones)
        return cls(DEFAULT_COURT_BOUNDS, zones)

    def zone(self, xy: Sequence[float]) -> str | None:
        point = [float(xy[0]), float(xy[1])]
        for name, polygon in self.zones.items():
            if name == "court":
                continue
            if _point_in_polygon(point, polygon):
                return str(name)
        court = self.zones.get("court")
        if _point_in_polygon(point, court):
            return "court"
        x, y = point
        if self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max:
            if -self.nvz_m <= y <= 0.0:
                return "near_nvz"
            if 0.0 <= y <= self.nvz_m:
                return "far_nvz"
            return "court"
        return None

    def line_call(self, xy: Sequence[float], *, uncertainty_radius_m: float) -> dict[str, Any]:
        x, y = float(xy[0]), float(xy[1])
        margins = {
            ("wide", "left"): x - self.x_min,
            ("wide", "right"): self.x_max - x,
            ("long", "near"): y - self.y_min,
            ("long", "far"): self.y_max - y,
        }
        nearest = min(margins.items(), key=lambda item: item[1])
        margin = float(nearest[1])
        call = "out" if margin < -float(uncertainty_radius_m) else "in"
        return {
            "call": call,
            "boundary_margin_m": _round(margin, 6),
            "uncertainty_radius_m": _round(float(uncertainty_radius_m), 6),
            "direction": nearest[0][0] if call == "out" else None,
            "side": nearest[0][1] if call == "out" else None,
            "within_uncertainty": abs(margin) <= float(uncertainty_radius_m),
        }


class _TrackLookup:
    def __init__(self, payload: Mapping[str, Any] | None) -> None:
        self.players: dict[str, list[Mapping[str, Any]]] = {}
        if not isinstance(payload, Mapping):
            return
        players = payload.get("players", [])
        if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
            return
        for player in players:
            if not isinstance(player, Mapping):
                continue
            player_id = player.get("id", player.get("player_id"))
            frames = player.get("frames", [])
            if player_id is None or not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
                continue
            self.players[str(player_id)] = [frame for frame in frames if isinstance(frame, Mapping)]

    def nearest(self, player_id: Any, t: float) -> dict[str, Any] | None:
        if player_id is None:
            return None
        frames = self.players.get(str(player_id))
        if not frames:
            return None
        frame = min(frames, key=lambda item: abs(_float(item.get("t"), t) - t))
        xy = frame.get("world_xy") or frame.get("track_world_xy")
        if not isinstance(xy, Sequence) or isinstance(xy, (str, bytes)) or len(xy) < 2:
            return None
        return {
            "world_xy": [_round(_float(xy[0], 0.0), 6), _round(_float(xy[1], 0.0), 6)],
            "t": _round(_float(frame.get("t"), t), 9),
            "conf": _round(_float(frame.get("conf"), 0.0), 6),
        }


def _infer_nvz_m(zones: Mapping[str, Any]) -> float:
    for name in ("near_nvz", "far_nvz"):
        polygon = zones.get(name)
        if isinstance(polygon, Sequence) and not isinstance(polygon, (str, bytes)):
            ys = [_float(point[1], 0.0) for point in polygon if isinstance(point, Sequence) and len(point) >= 2]
            if ys:
                return max(abs(value) for value in ys)
    return DEFAULT_NVZ_M


def _point_in_polygon(point: Sequence[float], polygon: Any) -> bool:
    if not isinstance(polygon, Sequence) or isinstance(polygon, (str, bytes)) or len(polygon) < 3:
        return False
    x, y = float(point[0]), float(point[1])
    inside = False
    count = len(polygon)
    for idx in range(count):
        p1 = polygon[idx]
        p2 = polygon[(idx + 1) % count]
        if not (
            isinstance(p1, Sequence)
            and not isinstance(p1, (str, bytes))
            and isinstance(p2, Sequence)
            and not isinstance(p2, (str, bytes))
            and len(p1) >= 2
            and len(p2) >= 2
        ):
            continue
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        if _point_on_segment(x, y, x1, y1, x2, y2):
            return True
        intersects = (y1 > y) != (y2 > y)
        if intersects:
            x_intersection = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_intersection:
                inside = not inside
    return inside


def _point_on_segment(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> bool:
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > 1e-9:
        return False
    return min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9


def _x_at_y(start: Sequence[float], end: Sequence[float], y: float) -> float | None:
    y0 = float(start[1])
    y1 = float(end[1])
    if (y0 < y and y1 < y) or (y0 > y and y1 > y) or abs(y1 - y0) < 1e-9:
        return None
    alpha = (float(y) - y0) / (y1 - y0)
    return float(start[0]) + alpha * (float(end[0]) - float(start[0]))


def _vec3(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    return [_float(value[0], 0.0), _float(value[1], 0.0), _float(value[2], 0.0)]


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _float(value: Any, default: float) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _int(value: Any, default: int) -> int:
    if isinstance(value, bool) or value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _round(value: float, digits: int) -> float:
    return round(float(value), digits)


def _optional_round(value: Any, digits: int) -> float | None:
    if value is None:
        return None
    number = _float(value, math.nan)
    if not math.isfinite(number):
        return None
    return _round(number, digits)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
