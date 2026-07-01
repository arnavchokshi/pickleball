"""Fail-closed BODY grounding quality gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_body_grounding_quality"
DEFAULT_MAX_FOOT_SLIDE_M = 0.03


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

    status = "pass" if not blockers else "blocked" if "missing_foot_slide_metric" in blockers else "fail"
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip": clip,
        "status": status,
        "foot_slide_gate": {
            "name": "foot_slide_max_m",
            "threshold_m": float(max_foot_slide_m),
            "value_m": foot_slide_value,
            "passed": foot_slide_value is not None and foot_slide_value <= max_foot_slide_m,
        },
        "grounding_metrics": dict(grounding_metrics),
        "blockers": blockers,
        "notes": notes,
    }


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
