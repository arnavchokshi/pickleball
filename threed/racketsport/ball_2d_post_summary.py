"""Truthful M2 2D post-processing summary builder."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


STATUS_TESTED = "TESTED-ON-REAL-DATA"
STATUS_NOT_STARTED = "NOT-STARTED"
POSTPROCESS_ARTIFACT_TYPE = "racketsport_ball_2d_postprocess_summary"
MODEL_FUSION_ARTIFACT_TYPE = "racketsport_ball_model_fusion"
COURT_METRIC_ARTIFACT_TYPE = "racketsport_ball_target_court_metric_filter"
MAX_SPEED_ARTIFACT_TYPE = "racketsport_ball_world_speed_gate"
RANSAC_ARTIFACT_TYPE = "racketsport_ball_ransac_arc_recovery"
LOCAL_SEARCH_ARTIFACT_TYPE = "racketsport_ball_local_search_filter"
KALMAN_RTS_ARTIFACT_TYPE = "racketsport_ball_kalman_rts_smoother"


def build_ball_2d_postprocess_summary(
    *,
    model_consensus_summary_paths: Sequence[str | Path] | None = None,
    court_gating_summary_path: str | Path | None = None,
    max_speed_summary_path: str | Path | None = None,
    ransac_summary_path: str | Path | None = None,
    local_search_summary_path: str | Path | None = None,
    kalman_rts_summary_path: str | Path | None = None,
    primary_model: str | None = None,
    verifier_model: str | None = None,
) -> dict[str, Any]:
    """Build the M2 postprocess summary from real component summary files.

    The output deliberately distinguishes present evidence from missing
    requirements. It does not promote diagnostic filters into spec components
    unless their artifact type matches the component being summarized.
    """

    consensus_paths = list(model_consensus_summary_paths or [])
    model_consensus = _model_consensus_component(
        consensus_paths,
        primary_model=primary_model,
        verifier_model=verifier_model,
    )
    court_gating = _court_gating_component(court_gating_summary_path)
    max_speed_gate = _single_numeric_component(
        max_speed_summary_path,
        artifact_type=MAX_SPEED_ARTIFACT_TYPE,
        output_key="max_world_speed_mps",
        source_keys=("max_world_speed_mps",),
    )
    ransac = _single_numeric_component(
        ransac_summary_path,
        artifact_type=RANSAC_ARTIFACT_TYPE,
        output_key="max_residual_px",
        source_keys=("max_residual_px", "max_ransac_residual_px"),
    )
    local_search = _single_numeric_component(
        local_search_summary_path,
        artifact_type=LOCAL_SEARCH_ARTIFACT_TYPE,
        output_key="recovery_heatmap_threshold",
        source_keys=("recovery_heatmap_threshold",),
    )
    kalman_rts = _kalman_rts_component(kalman_rts_summary_path)

    components = {
        "model_consensus": model_consensus,
        "court_gating": court_gating,
        "max_speed_gate": max_speed_gate,
        "ransac": ransac,
        "local_search": local_search,
        "kalman_rts": kalman_rts,
    }
    missing = [name for name, component in components.items() if not bool(component.get("evidence_present"))]

    return {
        "schema_version": 1,
        "artifact_type": POSTPROCESS_ARTIFACT_TYPE,
        "status": STATUS_TESTED if _has_any_input(components) else STATUS_NOT_STARTED,
        **components,
        "missing_components": missing,
        "not_ground_truth": True,
    }


def write_ball_2d_postprocess_summary(
    *,
    out: str | Path,
    model_consensus_summary_paths: Sequence[str | Path] | None = None,
    court_gating_summary_path: str | Path | None = None,
    max_speed_summary_path: str | Path | None = None,
    ransac_summary_path: str | Path | None = None,
    local_search_summary_path: str | Path | None = None,
    kalman_rts_summary_path: str | Path | None = None,
    primary_model: str | None = None,
    verifier_model: str | None = None,
) -> dict[str, Any]:
    summary = build_ball_2d_postprocess_summary(
        model_consensus_summary_paths=model_consensus_summary_paths,
        court_gating_summary_path=court_gating_summary_path,
        max_speed_summary_path=max_speed_summary_path,
        ransac_summary_path=ransac_summary_path,
        local_search_summary_path=local_search_summary_path,
        kalman_rts_summary_path=kalman_rts_summary_path,
        primary_model=primary_model,
        verifier_model=verifier_model,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _model_consensus_component(
    paths: Sequence[str | Path],
    *,
    primary_model: str | None,
    verifier_model: str | None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for path in paths:
        payload = _load_optional_json(path)
        if payload is None:
            continue
        radius = _finite_or_none(payload.get("outlier_distance_px"))
        if payload.get("artifact_type") == MODEL_FUSION_ARTIFACT_TYPE and radius is not None:
            candidates.append(
                {
                    "path": str(path),
                    "payload": payload,
                    "radius": radius,
                }
            )
    chosen = _choose_consensus_candidate(candidates)
    if chosen is None:
        return {
            "evidence_present": False,
            "primary": _normalize_name(primary_model),
            "verifier": _normalize_name(verifier_model),
            "radius_px_1080p": None,
            "evidence_path": None,
            "source_artifact_type": None,
        }

    payload = chosen["payload"]
    return {
        "evidence_present": True,
        "primary": _normalize_name(primary_model),
        "verifier": _normalize_name(verifier_model),
        "radius_px_1080p": chosen["radius"],
        "evidence_path": chosen["path"],
        "source_artifact_type": payload.get("artifact_type"),
        "primary_ball_track": payload.get("primary_ball_track"),
        "stable_ball_track": payload.get("stable_ball_track"),
        "verifier_ball_tracks": payload.get("verifier_ball_tracks"),
        "uses_human_clicks": payload.get("uses_human_clicks"),
    }


def _court_gating_component(path: str | Path | None) -> dict[str, Any]:
    payload = _load_optional_json(path)
    margin = _finite_or_none(payload.get("court_margin_m")) if payload is not None else None
    if payload is None or payload.get("artifact_type") != COURT_METRIC_ARTIFACT_TYPE or margin is None:
        return {
            "evidence_present": False,
            "margin_m": None,
            "evidence_path": str(path) if path is not None else None,
            "source_artifact_type": payload.get("artifact_type") if payload is not None else None,
        }
    return {
        "evidence_present": True,
        "margin_m": margin,
        "evidence_path": str(path),
        "source_artifact_type": payload.get("artifact_type"),
        "source_ball_track": payload.get("source_ball_track"),
        "uses_human_clicks": payload.get("uses_human_clicks"),
    }


def _single_numeric_component(
    path: str | Path | None,
    *,
    artifact_type: str,
    output_key: str,
    source_keys: Sequence[str],
) -> dict[str, Any]:
    payload = _load_optional_json(path)
    value = _first_finite(*(payload.get(key) for key in source_keys)) if payload is not None else None
    if payload is None or payload.get("artifact_type") != artifact_type or value is None:
        return {
            "evidence_present": False,
            output_key: None,
            "evidence_path": str(path) if path is not None else None,
            "source_artifact_type": payload.get("artifact_type") if payload is not None else None,
        }
    return {
        "evidence_present": True,
        output_key: value,
        "evidence_path": str(path),
        "source_artifact_type": payload.get("artifact_type"),
        "uses_human_clicks": payload.get("uses_human_clicks"),
    }


def _kalman_rts_component(path: str | Path | None) -> dict[str, Any]:
    payload = _load_optional_json(path)
    gap = _int_or_none(payload.get("max_gap_fill_frames")) if payload is not None else None
    jitter = _finite_or_none(payload.get("jitter_px_std")) if payload is not None else None
    if payload is None or payload.get("artifact_type") != KALMAN_RTS_ARTIFACT_TYPE or gap is None or jitter is None:
        return {
            "evidence_present": False,
            "max_gap_fill_frames": None,
            "jitter_px_std": None,
            "evidence_path": str(path) if path is not None else None,
            "source_artifact_type": payload.get("artifact_type") if payload is not None else None,
        }
    return {
        "evidence_present": True,
        "max_gap_fill_frames": gap,
        "jitter_px_std": jitter,
        "evidence_path": str(path),
        "source_artifact_type": payload.get("artifact_type"),
        "uses_human_clicks": payload.get("uses_human_clicks"),
    }


def _choose_consensus_candidate(candidates: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    within_spec = [candidate for candidate in candidates if float(candidate["radius"]) <= 60.0]
    pool = within_spec or list(candidates)
    return max(pool, key=lambda item: float(item["radius"]))


def _has_any_input(components: Mapping[str, Mapping[str, Any]]) -> bool:
    return any(component.get("evidence_path") is not None for component in components.values())


def _load_optional_json(path: str | Path | None) -> Mapping[str, Any] | None:
    if path is None or not Path(path).is_file():
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _first_finite(*values: Any) -> float | None:
    for value in values:
        numeric = _finite_or_none(value)
        if numeric is not None:
            return numeric
    return None


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _normalize_name(value: str | None) -> str | None:
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


__all__ = [
    "POSTPROCESS_ARTIFACT_TYPE",
    "build_ball_2d_postprocess_summary",
    "write_ball_2d_postprocess_summary",
]
