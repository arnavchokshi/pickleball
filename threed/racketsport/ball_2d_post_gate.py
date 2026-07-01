"""M2 2D post-processing gate report for BALL-only tracking."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


M2_STATUS_TESTED = "TESTED-ON-REAL-DATA"
M2_STATUS_SCAFFOLD = "SCAFFOLD"
M2_STATUS_NOT_STARTED = "NOT-STARTED"
MIN_F1 = 0.90
MIN_RECALL = 0.75
MAX_HIDDEN_FP_RATE = 0.05
CONSENSUS_RADIUS_PX_1080P = 60.0
COURT_MARGIN_M = 0.5
MAX_WORLD_SPEED_MPS = 30.0
MAX_RANSAC_RESIDUAL_PX = 5.0
RECOVERY_HEATMAP_THRESHOLD = 0.25
MAX_GAP_FILL_FRAMES = 6
MAX_JITTER_PX_STD = 2.0


def build_ball_2d_post_gate_report(
    *,
    m1_detector_report_path: str | Path | None = None,
    postprocess_summary_path: str | Path | None = None,
    benchmark_paths: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    m1_report = _load_optional_json(m1_detector_report_path)
    post_summary = _load_optional_json(postprocess_summary_path)
    benchmarks = [path for path in (benchmark_paths or []) if Path(path).is_file()]

    m1 = _m1_summary(m1_report)
    postprocess = _postprocess_summary(post_summary)
    metrics = _metrics_summary(benchmarks)
    status = _status(has_summary=post_summary is not None, has_benchmarks=bool(benchmarks), has_m1=m1_report is not None)

    violations: list[str] = []
    _extend_unique(violations, m1["violations"])
    _extend_unique(violations, postprocess["violations"])
    _extend_unique(violations, metrics["violations"])

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_2d_post_gate_report",
        "milestone": "M2 2D post",
        "status": status,
        "gate_result": "fail" if violations else "pass",
        "blocked_reason": "ball_2d_post_gate_failed" if violations else None,
        "required_thresholds": {
            "model_consensus_radius_px_1080p": CONSENSUS_RADIUS_PX_1080P,
            "court_margin_m": COURT_MARGIN_M,
            "max_world_ball_speed_mps": MAX_WORLD_SPEED_MPS,
            "max_ransac_arc_residual_px": MAX_RANSAC_RESIDUAL_PX,
            "recovery_heatmap_threshold": RECOVERY_HEATMAP_THRESHOLD,
            "max_kalman_rts_gap_fill_frames": MAX_GAP_FILL_FRAMES,
            "max_jitter_px_std": MAX_JITTER_PX_STD,
            "f1_min": MIN_F1,
            "recall_min": MIN_RECALL,
            "hidden_false_positive_rate_max": MAX_HIDDEN_FP_RATE,
        },
        "m1_detector_gate": m1["summary"],
        "postprocess": postprocess["summary"],
        "metrics": metrics["summary"],
        "violations": violations,
        "not_ground_truth": True,
    }


def write_ball_2d_post_gate_report(
    *,
    out: str | Path,
    m1_detector_report_path: str | Path | None = None,
    postprocess_summary_path: str | Path | None = None,
    benchmark_paths: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    report = build_ball_2d_post_gate_report(
        m1_detector_report_path=m1_detector_report_path,
        postprocess_summary_path=postprocess_summary_path,
        benchmark_paths=benchmark_paths,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _m1_summary(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {
            "summary": {"path_present": False, "gate_result": None, "status": None, "blocked_reason": None},
            "violations": ["missing_m1_detector_gate_report"],
        }
    violations = []
    if report.get("artifact_type") != "racketsport_ball_detector_gate_report":
        violations.append("m1_detector_gate_artifact_type_invalid")
    if report.get("gate_result") != "pass":
        violations.append("m1_detector_gate_not_passed")
    return {
        "summary": {
            "path_present": True,
            "artifact_type": report.get("artifact_type"),
            "gate_result": report.get("gate_result"),
            "status": report.get("status"),
            "blocked_reason": report.get("blocked_reason"),
        },
        "violations": violations,
    }


def _postprocess_summary(summary: Mapping[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {
            "summary": {
                "path_present": False,
                "model_consensus": None,
                "court_gating": None,
                "max_speed_gate": None,
                "ransac": None,
                "local_search": None,
                "kalman_rts": None,
            },
            "violations": ["missing_postprocess_summary"],
        }
    violations: list[str] = []
    model_consensus = _mapping_or_empty(summary.get("model_consensus"))
    court_gating = _mapping_or_empty(summary.get("court_gating"))
    max_speed_gate = _mapping_or_empty(summary.get("max_speed_gate"))
    ransac = _mapping_or_empty(summary.get("ransac"))
    local_search = _mapping_or_empty(summary.get("local_search"))
    kalman_rts = _mapping_or_empty(summary.get("kalman_rts"))

    if _evidence_missing(model_consensus):
        violations.append("model_consensus_evidence_missing")
    else:
        if str(model_consensus.get("primary")).lower() != "tracknet" or str(model_consensus.get("verifier")).lower() != "wasb":
            violations.append("model_consensus_missing_tracknet_wasb")
        radius = _finite_or_none(model_consensus.get("radius_px_1080p"))
        if radius is None or radius > CONSENSUS_RADIUS_PX_1080P:
            violations.append("model_consensus_radius_over_60px")

    if _evidence_missing(court_gating):
        violations.append("court_gating_evidence_missing")
    else:
        margin = _finite_or_none(court_gating.get("margin_m"))
        if margin is None or not math.isclose(margin, COURT_MARGIN_M, rel_tol=1e-6, abs_tol=1e-6):
            violations.append("court_margin_not_0_5m")

    if _evidence_missing(max_speed_gate):
        violations.append("max_speed_gate_evidence_missing")
    else:
        max_speed = _finite_or_none(max_speed_gate.get("max_world_speed_mps"))
        if max_speed is None or max_speed > MAX_WORLD_SPEED_MPS:
            violations.append("max_speed_gate_over_30mps")

    if _evidence_missing(ransac):
        violations.append("ransac_evidence_missing")
    else:
        residual = _finite_or_none(ransac.get("max_residual_px"))
        if residual is None or residual > MAX_RANSAC_RESIDUAL_PX:
            violations.append("ransac_residual_over_5px")

    if _evidence_missing(local_search):
        violations.append("local_search_evidence_missing")
    else:
        recovery_threshold = _finite_or_none(local_search.get("recovery_heatmap_threshold"))
        if recovery_threshold is None or not math.isclose(
            recovery_threshold,
            RECOVERY_HEATMAP_THRESHOLD,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            violations.append("local_search_recovery_threshold_not_0_25")

    if _evidence_missing(kalman_rts):
        violations.append("kalman_rts_evidence_missing")
    else:
        gap_fill = _int_or_none(kalman_rts.get("max_gap_fill_frames"))
        if gap_fill is None or gap_fill > MAX_GAP_FILL_FRAMES:
            violations.append("kalman_rts_gap_fill_over_6_frames")
        jitter = _finite_or_none(kalman_rts.get("jitter_px_std"))
        if jitter is None or jitter >= MAX_JITTER_PX_STD:
            violations.append("jitter_target_not_below_2px")

    return {
        "summary": {
            "path_present": True,
            "artifact_type": summary.get("artifact_type"),
            "model_consensus": dict(model_consensus),
            "court_gating": dict(court_gating),
            "max_speed_gate": dict(max_speed_gate),
            "ransac": dict(ransac),
            "local_search": dict(local_search),
            "kalman_rts": dict(kalman_rts),
        },
        "violations": sorted(set(violations)),
    }


def _metrics_summary(paths: Sequence[str | Path]) -> dict[str, Any]:
    if not paths:
        return {
            "summary": {
                "path_count": 0,
                "candidate_count": 0,
                "best_candidate": None,
                "best_f1": None,
                "best_recall": None,
                "best_hidden_false_positive_rate": None,
                "best_teleport_count": None,
                "best_max_visible_gap_frames": None,
            },
            "violations": ["missing_m2_benchmark"],
        }
    scored: list[dict[str, Any]] = []
    for path in paths:
        payload = _load_json(path)
        aggregate = payload.get("aggregate")
        candidates = aggregate if isinstance(aggregate, Mapping) else {}
        for name, metrics in candidates.items():
            if not isinstance(metrics, Mapping):
                continue
            scored.append(
                {
                    "path": str(path),
                    "name": str(name),
                    "f1": _first_finite(metrics.get("micro_label_f1_at_20px"), metrics.get("micro_label_f1")),
                    "recall": _first_finite(
                        metrics.get("micro_visible_recall_at_20px"),
                        metrics.get("micro_visible_hit_recall"),
                        metrics.get("mean_visible_hit_recall"),
                    ),
                    "hidden_fp_rate": _first_finite(
                        metrics.get("micro_hidden_false_positive_rate"),
                        metrics.get("mean_hidden_false_positive_rate"),
                    ),
                    "teleport_count": _int_or_none(metrics.get("total_teleport_count")),
                    "max_visible_gap_frames": _first_finite(metrics.get("mean_max_visible_gap_frames")),
                }
            )
    best = max(scored, key=lambda item: item["f1"] if item["f1"] is not None else -math.inf, default=None)
    best_f1 = best.get("f1") if best else None
    best_recall = best.get("recall") if best else None
    best_hidden_fp = best.get("hidden_fp_rate") if best else None
    best_teleports = best.get("teleport_count") if best else None
    best_gap = best.get("max_visible_gap_frames") if best else None
    violations: list[str] = []
    if best_f1 is None or best_f1 < MIN_F1:
        violations.append("post_f1_below_0_90")
    if best_recall is None or best_recall < MIN_RECALL:
        violations.append("post_recall_below_0_75")
    if best_hidden_fp is None or best_hidden_fp > MAX_HIDDEN_FP_RATE:
        violations.append("post_hidden_fp_rate_over_0_05")
    if best_teleports is None:
        violations.append("teleport_count_missing")
    elif best_teleports > 0:
        violations.append("teleport_count_nonzero")
    if best_gap is None:
        violations.append("visible_gap_metric_missing")
    elif best_gap > MAX_GAP_FILL_FRAMES:
        violations.append("visible_gap_over_6_frames")
    return {
        "summary": {
            "path_count": len(paths),
            "paths": [str(path) for path in paths],
            "candidate_count": len(scored),
            "best_candidate": best.get("name") if best else None,
            "best_candidate_path": best.get("path") if best else None,
            "best_f1": best_f1,
            "best_recall": best_recall,
            "best_hidden_false_positive_rate": best_hidden_fp,
            "best_teleport_count": best_teleports,
            "best_max_visible_gap_frames": best_gap,
        },
        "violations": violations,
    }


def _status(*, has_summary: bool, has_benchmarks: bool, has_m1: bool) -> str:
    if has_benchmarks:
        return M2_STATUS_TESTED
    if has_summary or has_m1:
        return M2_STATUS_SCAFFOLD
    return M2_STATUS_NOT_STARTED


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _evidence_missing(component: Mapping[str, Any]) -> bool:
    return component.get("evidence_present") is False


def _load_optional_json(path: str | Path | None) -> Mapping[str, Any] | None:
    if path is None or not Path(path).is_file():
        return None
    return _load_json(path)


def _load_json(path: str | Path) -> Mapping[str, Any]:
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


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _finite_like(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


def _finite_or_none(value: Any) -> float | None:
    return float(value) if _finite_like(value) else None


def _extend_unique(target: list[str], values: Sequence[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


__all__ = [
    "M2_STATUS_NOT_STARTED",
    "M2_STATUS_SCAFFOLD",
    "M2_STATUS_TESTED",
    "build_ball_2d_post_gate_report",
    "write_ball_2d_post_gate_report",
]
