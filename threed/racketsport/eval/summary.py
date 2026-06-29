from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from threed.racketsport.schemas import PhaseEvalMetrics


SUMMARY_STATUS_KEYS = ("pass", "fail", "blocked", "not_measured", "missing_metrics", "malformed_metrics")
EVAL_STATUS_KEYS = ("pass", "fail", "blocked", "not_measured")
METRIC_STATUS_KEYS = ("measured", "not_measured")
GATE_RESULT_KEYS = ("pass", "fail", "not_measured")


def build_eval_run_summary(
    phase_dirs: Iterable[str | Path],
    *,
    metrics_filename: str = "metrics.json",
    strict: bool = True,
) -> dict[str, Any]:
    phase_paths = sorted(Path(phase_dir) for phase_dir in phase_dirs)
    if not phase_paths:
        raise ValueError("at least one phase directory is required")

    phases: list[dict[str, Any]] = []
    missing_metrics_files: list[str] = []
    malformed_metrics_files: list[dict[str, str]] = []
    status_counts: Counter[str] = Counter()
    metric_status_counts: Counter[str] = Counter()
    gate_result_counts: Counter[str] = Counter()
    clip_status_counts: Counter[str] = Counter()

    for phase_dir in phase_paths:
        metrics_path = phase_dir / metrics_filename
        if not metrics_path.is_file():
            missing_metrics_files.append(str(metrics_path))
            status_counts["missing_metrics"] += 1
            phases.append(_missing_phase_record(phase_dir=phase_dir, metrics_path=metrics_path))
            continue

        try:
            payload = _load_phase_metrics(metrics_path)
        except ValueError as exc:
            malformed = {
                "phase_dir": str(phase_dir),
                "metrics_path": str(metrics_path),
                "error": str(exc),
            }
            malformed_metrics_files.append(malformed)
            if strict:
                raise
            status_counts["malformed_metrics"] += 1
            phases.append(_malformed_phase_record(phase_dir=phase_dir, metrics_path=metrics_path, error=str(exc)))
            continue

        phase_record = _phase_record(phase_dir=phase_dir, metrics_path=metrics_path, payload=payload)
        phases.append(phase_record)
        status_counts[phase_record["status"]] += 1
        metric_status_counts.update(phase_record["metric_status_counts"])
        gate_result_counts.update(phase_record["gate_result_counts"])
        clip_status_counts.update(phase_record["clip_status_counts"])

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_eval_run_summary",
        "execution": {
            "cpu_only": True,
            "runs_evaluations": False,
            "uses_gpu": False,
            "mutates_metrics": False,
        },
        "summary": {
            "phase_count": len(phases),
            "metrics_file_count": len(phases) - len(missing_metrics_files) - len(malformed_metrics_files),
            "missing_metrics_file_count": len(missing_metrics_files),
            "malformed_metrics_file_count": len(malformed_metrics_files),
            "status_counts": _fixed_counts(status_counts, SUMMARY_STATUS_KEYS),
            "metric_status_counts": _fixed_counts(metric_status_counts, METRIC_STATUS_KEYS),
            "gate_result_counts": _fixed_counts(gate_result_counts, GATE_RESULT_KEYS),
            "clip_status_counts": _fixed_counts(clip_status_counts, EVAL_STATUS_KEYS),
        },
        "phases": phases,
        "highest_risk_phases": _highest_risk_phases(phases),
        "missing_metrics_files": missing_metrics_files,
        "malformed_metrics_files": malformed_metrics_files,
    }


def write_eval_run_summary(out: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_phase_metrics(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"failed to parse {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"failed to parse {path}: metrics payload must be a JSON object")

    try:
        return PhaseEvalMetrics.model_validate(raw).model_dump(mode="json")
    except Exception as exc:
        raise ValueError(f"failed to parse {path}: {exc}") from exc


def _missing_phase_record(*, phase_dir: Path, metrics_path: Path) -> dict[str, Any]:
    phase = phase_dir.name
    return {
        "phase": phase,
        "evaluator": None,
        "phase_dir": str(phase_dir),
        "metrics_path": str(metrics_path),
        "metrics_exists": False,
        "status": "missing_metrics",
        "summary": None,
        "required_artifacts": [],
        "metric_status_counts": _fixed_counts(Counter(), METRIC_STATUS_KEYS),
        "gate_result_counts": _fixed_counts(Counter(), GATE_RESULT_KEYS),
        "clip_status_counts": _fixed_counts(Counter(), EVAL_STATUS_KEYS),
        "risk_score": 80,
        "risk_reasons": ["metrics.json missing"],
    }


def _malformed_phase_record(*, phase_dir: Path, metrics_path: Path, error: str) -> dict[str, Any]:
    phase = phase_dir.name
    return {
        "phase": phase,
        "evaluator": None,
        "phase_dir": str(phase_dir),
        "metrics_path": str(metrics_path),
        "metrics_exists": True,
        "status": "malformed_metrics",
        "summary": {"error": error},
        "required_artifacts": [],
        "metric_status_counts": _fixed_counts(Counter(), METRIC_STATUS_KEYS),
        "gate_result_counts": _fixed_counts(Counter(), GATE_RESULT_KEYS),
        "clip_status_counts": _fixed_counts(Counter(), EVAL_STATUS_KEYS),
        "risk_score": 100,
        "risk_reasons": ["metrics.json malformed"],
    }


def _phase_record(*, phase_dir: Path, metrics_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    metrics = _collect_metrics(payload)
    metric_status_counts = _metric_status_counts(metrics)
    gate_result_counts = _gate_result_counts(metrics)
    clip_status_counts = Counter(clip.get("status") for clip in _as_list(payload.get("clips")))
    risk_score, risk_reasons = _risk(payload=payload, gate_result_counts=gate_result_counts)

    return {
        "phase": payload["phase"],
        "evaluator": payload["evaluator"],
        "phase_dir": str(phase_dir),
        "metrics_path": str(metrics_path),
        "metrics_exists": True,
        "status": payload["status"],
        "summary": payload["summary"],
        "required_artifacts": payload["required_artifacts"],
        "metric_status_counts": _fixed_counts(metric_status_counts, METRIC_STATUS_KEYS),
        "gate_result_counts": _fixed_counts(gate_result_counts, GATE_RESULT_KEYS),
        "clip_status_counts": _fixed_counts(clip_status_counts, EVAL_STATUS_KEYS),
        "risk_score": risk_score,
        "risk_reasons": risk_reasons,
    }


def _collect_metrics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    metrics.extend(metric for metric in _as_mapping(payload.get("metrics")).values() if isinstance(metric, dict))
    for clip in _as_list(payload.get("clips")):
        if isinstance(clip, dict):
            metrics.extend(metric for metric in _as_mapping(clip.get("metrics")).values() if isinstance(metric, dict))
    return metrics


def _metric_status_counts(metrics: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for metric in metrics:
        status = metric.get("status")
        if status in METRIC_STATUS_KEYS:
            counts[status] += 1
    return counts


def _gate_result_counts(metrics: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for metric in metrics:
        if metric.get("status") == "not_measured" or metric.get("passed") is None:
            counts["not_measured"] += 1
        elif metric.get("passed") is True:
            counts["pass"] += 1
        elif metric.get("passed") is False:
            counts["fail"] += 1
    return counts


def _risk(*, payload: dict[str, Any], gate_result_counts: Counter[str]) -> tuple[int, list[str]]:
    status = payload["status"]
    reasons: list[str] = []
    score = 0

    if status == "fail":
        score += 100
        reasons.append("phase status fail")
    elif status == "blocked":
        score += 75
        reasons.append("phase status blocked")
    elif status == "not_measured":
        score += 50
        reasons.append("phase status not_measured")

    failed_gates = gate_result_counts["fail"]
    not_measured_gates = gate_result_counts["not_measured"]
    if failed_gates:
        score += failed_gates * 10
        reasons.append(f"{failed_gates} failed gate(s)")
    if not_measured_gates:
        score += not_measured_gates * 3
        reasons.append(f"{not_measured_gates} not measured gate(s)")

    summary = _as_mapping(payload.get("summary"))
    failed_clips = _as_int(summary.get("failed_clips"))
    blocked_clips = _as_int(summary.get("blocked_clips"))
    if failed_clips:
        score += failed_clips * 5
        reasons.append(f"{failed_clips} failed clip(s)")
    if blocked_clips:
        score += blocked_clips * 4
        reasons.append(f"{blocked_clips} blocked clip(s)")

    return score, reasons


def _highest_risk_phases(phases: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    ranked = sorted(phases, key=lambda phase: (-phase["risk_score"], phase["phase"], phase["metrics_path"]))
    return [
        {
            "phase": phase["phase"],
            "status": phase["status"],
            "risk_score": phase["risk_score"],
            "risk_reasons": phase["risk_reasons"],
            "phase_dir": phase["phase_dir"],
            "metrics_path": phase["metrics_path"],
        }
        for phase in ranked[:limit]
    ]


def _fixed_counts(counts: Counter[str], keys: tuple[str, ...]) -> dict[str, int]:
    return {key: int(counts.get(key, 0)) for key in keys}


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
