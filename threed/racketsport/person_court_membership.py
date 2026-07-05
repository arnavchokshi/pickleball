"""Source-only court-membership evidence for person identity diagnostics."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .court_templates import Sport, get_court_template


TARGET_COURT = "target_court_player_candidate"
APRON = "apron_or_boundary"
ADJACENT = "adjacent_court"
SPECTATOR = "spectator_background"
UNKNOWN = "projection_unknown"
NON_TARGET_CLASSES = {ADJACENT, SPECTATOR, UNKNOWN}


@dataclass(frozen=True)
class CourtMembershipConfig:
    lateral_apron_m: float = 0.75
    longitudinal_apron_m: float = 1.0
    adjacent_lateral_band_m: float = 8.0
    adjacent_longitudinal_band_m: float = 4.0
    target_fraction_threshold: float = 0.5


def classify_world_position(
    world_xy: Sequence[float] | None,
    *,
    sport: Sport = "pickleball",
    config: CourtMembershipConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify one projected foot/root point relative to the target court.

    This is source-only evidence. It does not read labels and does not delete
    detections. The classes are intended for later identity selection gates.
    """

    cfg = _config(config)
    if world_xy is None or len(world_xy) < 2:
        return _classification(UNKNOWN, ["missing_projected_world_xy"], None, None)
    try:
        x = float(world_xy[0])
        y = float(world_xy[1])
    except (TypeError, ValueError):
        return _classification(UNKNOWN, ["invalid_projected_world_xy"], None, None)
    if not math.isfinite(x) or not math.isfinite(y):
        return _classification(UNKNOWN, ["non_finite_projected_world_xy"], None, None)

    template = get_court_template(sport)
    half_width = template.width_m / 2.0
    half_length = template.length_m / 2.0
    outside_x = max(0.0, abs(x) - half_width)
    outside_y = max(0.0, abs(y) - half_length)

    if outside_x <= 0.0 and outside_y <= 0.0:
        return _classification(TARGET_COURT, ["inside_target_court"], outside_x, outside_y)

    if outside_x <= cfg.lateral_apron_m and outside_y <= cfg.longitudinal_apron_m:
        return _classification(APRON, ["inside_apron_margin"], outside_x, outside_y)

    reasons: list[str] = []
    if outside_x > cfg.lateral_apron_m:
        reasons.append("beyond_lateral_apron")
    if outside_y > cfg.longitudinal_apron_m:
        reasons.append("beyond_longitudinal_apron")

    if outside_x <= cfg.adjacent_lateral_band_m and outside_y <= cfg.adjacent_longitudinal_band_m:
        return _classification(ADJACENT, reasons or ["adjacent_court_band"], outside_x, outside_y)

    reasons.append("far_from_target_court")
    return _classification(SPECTATOR, reasons, outside_x, outside_y)


def build_court_membership_artifact(
    human_observations_payload: Mapping[str, Any],
    *,
    sport: Sport = "pickleball",
    config: CourtMembershipConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _config(config)
    observations = _observations(human_observations_payload)
    classified: list[dict[str, Any]] = []
    by_fragment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        membership = classify_world_position(obs.get("projected_world_xy"), sport=sport, config=cfg)
        row = {
            "detection_id": obs.get("detection_id"),
            "fragment_id": str(obs.get("fragment_id") or _track_fragment_id(obs.get("source_tracker_id"))),
            "source_tracker_id": _int_or_none(obs.get("source_tracker_id")),
            "frame_idx": _int_or_none(obs.get("frame_idx")),
            "membership_class": membership["membership_class"],
            "reason_codes": membership["reason_codes"],
            "outside_target_court_m": membership["outside_target_court_m"],
        }
        classified.append(row)
        by_fragment[row["fragment_id"]].append(row)

    fragments = [
        _summarize_fragment(fragment_id, rows, cfg=cfg)
        for fragment_id, rows in sorted(by_fragment.items(), key=lambda item: _fragment_sort_key(item[0]))
    ]
    class_counts = Counter(row["membership_class"] for row in classified)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_person_court_membership",
        "source_only": True,
        "uses_cvat_labels": False,
        "not_gate_verified": True,
        "sport": sport,
        "config": asdict(cfg),
        "observation_count": len(classified),
        "fragment_count": len(fragments),
        "class_counts": dict(sorted(class_counts.items())),
        "observations": classified,
        "fragments": fragments,
    }


def _summarize_fragment(fragment_id: str, rows: list[dict[str, Any]], *, cfg: CourtMembershipConfig) -> dict[str, Any]:
    counts = Counter(str(row["membership_class"]) for row in rows)
    total = len(rows)
    if not rows:
        membership_class = UNKNOWN
    elif counts[TARGET_COURT] / total >= cfg.target_fraction_threshold:
        membership_class = TARGET_COURT
    elif (counts[TARGET_COURT] + counts[APRON]) / total >= cfg.target_fraction_threshold:
        membership_class = APRON
    elif counts[ADJACENT] / total >= cfg.target_fraction_threshold:
        membership_class = ADJACENT
    elif counts[SPECTATOR] / total >= cfg.target_fraction_threshold:
        membership_class = SPECTATOR
    elif counts[UNKNOWN] == total:
        membership_class = UNKNOWN
    else:
        membership_class = UNKNOWN
    reason_codes = sorted({reason for row in rows for reason in row["reason_codes"]})
    source_ids = sorted({row["source_tracker_id"] for row in rows if row.get("source_tracker_id") is not None})
    return {
        "fragment_id": fragment_id,
        "source_tracker_id": source_ids[0] if len(source_ids) == 1 else None,
        "membership_class": membership_class,
        "eligible_for_target_selection": membership_class in {TARGET_COURT, APRON},
        "observation_count": total,
        "class_counts": dict(sorted(counts.items())),
        "reason_codes": reason_codes,
    }


def _classification(class_name: str, reasons: list[str], outside_x: float | None, outside_y: float | None) -> dict[str, Any]:
    return {
        "membership_class": class_name,
        "eligible_for_target_selection": class_name in {TARGET_COURT, APRON},
        "reason_codes": reasons,
        "outside_target_court_m": {
            "x": round(outside_x, 6) if outside_x is not None else None,
            "y": round(outside_y, 6) if outside_y is not None else None,
        },
    }


def _observations(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = payload.get("observations")
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _track_fragment_id(source_tracker_id: Any) -> str:
    value = _int_or_none(source_tracker_id)
    return f"track_{value}" if value is not None else "fragment_unknown"


def _fragment_sort_key(fragment_id: str) -> tuple[int, str]:
    if fragment_id.startswith("track_") and fragment_id.removeprefix("track_").isdigit():
        return int(fragment_id.removeprefix("track_")), fragment_id
    if fragment_id.startswith("frag_") and fragment_id.removeprefix("frag_").isdigit():
        return int(fragment_id.removeprefix("frag_")), fragment_id
    return 10**9, fragment_id


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _config(value: CourtMembershipConfig | Mapping[str, Any] | None) -> CourtMembershipConfig:
    if value is None:
        return CourtMembershipConfig()
    if isinstance(value, CourtMembershipConfig):
        return value
    defaults = asdict(CourtMembershipConfig())
    for key, raw in value.items():
        if key in defaults:
            defaults[key] = raw
    return CourtMembershipConfig(**defaults)


__all__ = [
    "ADJACENT",
    "APRON",
    "CourtMembershipConfig",
    "SPECTATOR",
    "TARGET_COURT",
    "UNKNOWN",
    "build_court_membership_artifact",
    "classify_world_position",
]
