"""Fail-closed BODY grounding quality gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_body_grounding_quality"
DEFAULT_MAX_FOOT_SLIDE_M = 0.03
INDEPENDENT_PHASE_REJECTION_REASONS = {
    "weak_bilateral_unknown_foot",
    "low_body_contact_confidence",
    "phase_penetrates_ground",
}


def build_body_grounding_quality(
    *,
    clip: str,
    grounding_metrics: Mapping[str, Any],
    max_foot_slide_m: float = DEFAULT_MAX_FOOT_SLIDE_M,
) -> dict[str, Any]:
    if max_foot_slide_m <= 0.0:
        raise ValueError("max_foot_slide_m must be positive")

    foot_slide_value = _maybe_float(grounding_metrics.get("max_foot_lock_slide_m"))
    blockers: list[str] = []
    notes: list[str] = []
    if foot_slide_value is None:
        blockers.append("missing_foot_slide_metric")
        notes.append("grounding metrics did not include max_foot_lock_slide_m")
    elif foot_slide_value > max_foot_slide_m:
        blockers.append("foot_slide_gate_failed")

    gate_stream = _validated_gate_stream(grounding_metrics.get("foot_lock_gate_stream"), clip=clip)
    stream_gate_failures = (
        _over_threshold_gate_stream_rows(gate_stream, max_foot_slide_m=max_foot_slide_m)
        if gate_stream is not None
        else []
    )
    if stream_gate_failures:
        blockers.append("foot_lock_gate_stream_over_threshold_phase")
        notes.append("foot_lock_gate_stream contains over-threshold lock-metric phase rows")

    status = "pass" if not blockers else "blocked" if blockers == ["missing_foot_slide_metric"] else "fail"
    foot_slide_passed = (
        foot_slide_value is not None
        and foot_slide_value <= max_foot_slide_m
        and not stream_gate_failures
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip": clip,
        "status": status,
        "foot_slide_gate": {
            "name": "foot_slide_max_m",
            "threshold_m": float(max_foot_slide_m),
            "value_m": foot_slide_value,
            "passed": foot_slide_passed,
        },
        "grounding_metrics": dict(grounding_metrics),
        "blockers": blockers,
        "notes": notes,
    }
    if gate_stream is not None:
        payload["foot_lock_gate_stream"] = gate_stream
    return payload


def _over_threshold_gate_stream_rows(
    gate_stream: Mapping[str, Any],
    *,
    max_foot_slide_m: float,
) -> list[Mapping[str, Any]]:
    summary = gate_stream.get("summary")
    if not isinstance(summary, Mapping):
        return []
    rows = summary.get("phases_over_threshold")
    if not isinstance(rows, list):
        return []
    failures: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        slide_m = _maybe_float(row.get("slide_m"))
        if slide_m is None or slide_m <= max_foot_slide_m:
            continue
        rejection_reason = row.get("rejection_reason")
        if rejection_reason in INDEPENDENT_PHASE_REJECTION_REASONS:
            continue
        failures.append(row)
    return failures


def write_body_grounding_quality(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _validated_gate_stream(value: Any, *, clip: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("foot_lock_gate_stream must be a mapping when provided")
    if value.get("artifact_type") != "foot_lock_gate_stream":
        raise ValueError("foot_lock_gate_stream.artifact_type must be foot_lock_gate_stream")
    for key in ("phase_rows", "frame_rows", "summary", "artifact_size_policy"):
        if key not in value:
            raise ValueError(f"foot_lock_gate_stream missing {key}")
    phase_rows = value.get("phase_rows")
    frame_rows = value.get("frame_rows")
    if not isinstance(phase_rows, list) or not all(isinstance(row, Mapping) for row in phase_rows):
        raise ValueError("foot_lock_gate_stream.phase_rows must be a list of objects")
    if not isinstance(frame_rows, list) or not all(isinstance(row, Mapping) for row in frame_rows):
        raise ValueError("foot_lock_gate_stream.frame_rows must be a list of objects")
    for row in phase_rows:
        for key in (
            "clip",
            "player_id",
            "foot",
            "phase_id",
            "start_frame_index",
            "end_frame_index",
            "slide_m",
            "max_contributing_frame_index",
            "anchor_position_xyz",
            "contact_source",
            "foot_assignment",
            "weak",
            "demoted",
            "split",
        ):
            if key not in row:
                raise ValueError(f"foot_lock_gate_stream.phase_rows[] missing {key}")
    for row in frame_rows:
        for key in (
            "clip",
            "player_id",
            "foot",
            "phase_id",
            "frame_idx",
            "contact_state",
            "selected_foot",
            "lock_anchor_xyz",
            "body_root_world",
            "output_source",
            "divergence_flag",
            "speed_cap_flag",
            "residuals",
            "source_counts",
            "foot_pin_correction_m",
        ):
            if key not in row:
                raise ValueError(f"foot_lock_gate_stream.frame_rows[] missing {key}")
    normalized = dict(value)
    normalized["clip"] = clip
    normalized["phase_rows"] = [
        {**dict(row), "clip": clip if row.get("clip") in {None, "", "unknown"} else row.get("clip")}
        for row in phase_rows
    ]
    normalized["frame_rows"] = [
        {**dict(row), "clip": clip if row.get("clip") in {None, "", "unknown"} else row.get("clip")}
        for row in frame_rows
    ]
    return normalized
