"""M1 offline detector gate report for BALL-only tracking."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schemas import BallTrack, validate_artifact_file


M1_STATUS_TESTED = "TESTED-ON-REAL-DATA"
M1_STATUS_SCAFFOLD = "SCAFFOLD"
M1_STATUS_NOT_STARTED = "NOT-STARTED"
MIN_F1 = 0.90
MIN_RECALL = 0.75
MAX_HIDDEN_FP_RATE = 0.05
EXPECTED_TRACKNET_SOURCE = "https://github.com/qaz812345/TrackNetV3"
EXPECTED_WASB_SOURCE = "https://github.com/nttcom/WASB-SBDT"


def build_ball_detector_gate_report(
    *,
    model_manifest_path: str | Path | None = None,
    ball_track_path: str | Path | None = None,
    benchmark_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = _load_optional_json(model_manifest_path)
    benchmark = _load_optional_json(benchmark_path)
    metadata = _load_optional_json(metadata_path)
    track = _load_optional_ball_track(ball_track_path)

    model_manifest = _model_manifest_summary(manifest)
    confidence = _confidence_summary(track, metadata)
    metrics = _metrics_summary(benchmark)
    status = _status(has_track=track is not None, has_benchmark=benchmark is not None, has_manifest=manifest is not None)

    violations: list[str] = []
    _extend_unique(violations, model_manifest["violations"])
    _extend_unique(violations, confidence["violations"])
    _extend_unique(violations, metrics["violations"])

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_detector_gate_report",
        "milestone": "M1 Offline detector",
        "status": status,
        "gate_result": "fail" if violations else "pass",
        "blocked_reason": "ball_detector_gate_failed" if violations else None,
        "required_thresholds": {
            "f1_min": MIN_F1,
            "blur_occlusion_recall_min": MIN_RECALL,
            "hidden_false_positive_rate_max": MAX_HIDDEN_FP_RATE,
            "requires_precision_recall_at_10px": True,
            "confidence_semantics": "real heatmap peak value (0..1)",
        },
        "model_manifest": model_manifest["summary"],
        "confidence": confidence["summary"],
        "metrics": metrics["summary"],
        "violations": violations,
        "not_ground_truth": True,
    }


def write_ball_detector_gate_report(
    *,
    out: str | Path,
    model_manifest_path: str | Path | None = None,
    ball_track_path: str | Path | None = None,
    benchmark_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    report = build_ball_detector_gate_report(
        model_manifest_path=model_manifest_path,
        ball_track_path=ball_track_path,
        benchmark_path=benchmark_path,
        metadata_path=metadata_path,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _model_manifest_summary(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    if manifest is None:
        return {
            "summary": {
                "path_present": False,
                "tracknetv3": _missing_model_summary(),
                "inpaintnet": _missing_model_summary(),
                "wasb": _missing_model_summary(),
            },
            "violations": ["missing_model_manifest"],
        }
    models = manifest.get("models")
    entries = models if isinstance(models, list) else []
    tracknet = _find_model(entries, ["tracknetv3"], exclude=["inpaint"])
    inpaintnet = _find_model(entries, ["inpaintnet", "tracknetv3_inpaintnet"])
    wasb = _find_model(entries, ["wasb"])
    violations: list[str] = []
    summaries: dict[str, dict[str, Any]] = {}
    for label, model, source_prefix, fine_tuned_violation in (
        ("tracknetv3", tracknet, EXPECTED_TRACKNET_SOURCE, "tracknet_not_pickleball_finetuned"),
        ("inpaintnet", inpaintnet, EXPECTED_TRACKNET_SOURCE, "inpaintnet_not_pickleball_finetuned"),
        ("wasb", wasb, EXPECTED_WASB_SOURCE, "wasb_not_pickleball_finetuned"),
    ):
        summary, model_violations = _model_summary(
            model,
            source_prefix=source_prefix,
            missing_violation=f"missing_{label}_manifest_entry",
            fine_tuned_violation=fine_tuned_violation,
        )
        summaries[label] = summary
        _extend_unique(violations, model_violations)
    return {"summary": {"path_present": True, **summaries}, "violations": violations}


def _model_summary(
    model: Mapping[str, Any] | None,
    *,
    source_prefix: str,
    missing_violation: str,
    fine_tuned_violation: str,
) -> tuple[dict[str, Any], list[str]]:
    if model is None:
        return _missing_model_summary(), [missing_violation]
    source = str(model.get("source") or "")
    sha256 = model.get("sha256")
    fine_tuned = model.get("fine_tuned_on_pickleball")
    status = model.get("status")
    source_official = source_prefix in source
    violations: list[str] = []
    if not source_official:
        violations.append(f"{missing_violation}_official_source")
    if not isinstance(sha256, str) or len(sha256) != 64:
        violations.append(f"{missing_violation}_sha256")
    if status not in {"available_on_h100", "downloadable_local_checkpoint", "available_local"}:
        violations.append(f"{missing_violation}_status")
    if fine_tuned is not True:
        violations.append(fine_tuned_violation)
    return (
        {
            "id": model.get("id"),
            "source": source,
            "source_official": source_official,
            "sha256": sha256,
            "status": status,
            "local_path": model.get("local_path"),
            "fine_tuned_on_pickleball": fine_tuned,
            "license": model.get("license"),
        },
        violations,
    )


def _confidence_summary(track: BallTrack | None, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    violations: list[str] = []
    if track is None:
        return {
            "summary": {
                "ball_track_present": False,
                "frame_count": 0,
                "conf_min": None,
                "conf_max": None,
                "conf_mean": None,
                "has_non_binary_confidence": False,
                "metadata_confidence_semantics": None,
            },
            "violations": ["missing_ball_track"],
        }
    confs = [float(frame.conf) for frame in track.frames]
    non_binary = any(value not in {0.0, 1.0} for value in confs)
    semantics = metadata.get("confidence_semantics") if metadata is not None else None
    if not non_binary:
        violations.append("confidence_values_are_binary_only")
    if metadata is None:
        violations.append("missing_detector_run_metadata")
    elif "heatmap" not in str(semantics).lower() or "0..1" not in str(semantics):
        violations.append("confidence_semantics_not_heatmap_peak")
    return {
        "summary": {
            "ball_track_present": True,
            "source": track.source,
            "frame_count": len(track.frames),
            "visible_frame_count": sum(1 for frame in track.frames if frame.visible),
            "conf_min": min(confs, default=None),
            "conf_max": max(confs, default=None),
            "conf_mean": sum(confs) / len(confs) if confs else None,
            "has_non_binary_confidence": non_binary,
            "metadata_confidence_semantics": semantics,
        },
        "violations": violations,
    }


def _metrics_summary(benchmark: Mapping[str, Any] | None) -> dict[str, Any]:
    if benchmark is None:
        return {
            "summary": {
                "path_present": False,
                "artifact_type": None,
                "candidate_count": 0,
                "best_candidate": None,
                "best_f1": None,
                "best_recall": None,
                "best_precision_at_10px": None,
                "best_recall_at_10px": None,
                "best_hidden_false_positive_rate": None,
            },
            "violations": ["missing_detector_benchmark"],
        }
    aggregate = benchmark.get("aggregate")
    candidates = aggregate if isinstance(aggregate, Mapping) else {}
    scored: list[dict[str, Any]] = []
    for name, metrics in candidates.items():
        if not isinstance(metrics, Mapping):
            continue
        scored.append(
            {
                "name": str(name),
                "f1": _first_finite(metrics.get("micro_label_f1_at_20px"), metrics.get("micro_label_f1")),
                "recall": _first_finite(
                    metrics.get("micro_visible_recall_at_20px"),
                    metrics.get("micro_visible_hit_recall"),
                    metrics.get("mean_visible_hit_recall"),
                ),
                "precision_at_10px": _finite_or_none(metrics.get("micro_precision_at_10px")),
                "recall_at_10px": _finite_or_none(metrics.get("micro_recall_at_10px")),
                "hidden_fp_rate": _first_finite(
                    metrics.get("micro_hidden_false_positive_rate"),
                    metrics.get("mean_hidden_false_positive_rate"),
                ),
            }
        )
    best = max(scored, key=lambda item: item["f1"] if item["f1"] is not None else -math.inf, default=None)
    violations: list[str] = []
    best_f1 = best.get("f1") if best else None
    best_recall = best.get("recall") if best else None
    best_precision_10 = best.get("precision_at_10px") if best else None
    best_recall_10 = best.get("recall_at_10px") if best else None
    best_hidden_fp = best.get("hidden_fp_rate") if best else None
    if best_f1 is None or best_f1 < MIN_F1:
        violations.append("detector_f1_below_0_90")
    if best_recall is None or best_recall < MIN_RECALL:
        violations.append("detector_recall_below_0_75")
    if best_hidden_fp is None or best_hidden_fp > MAX_HIDDEN_FP_RATE:
        violations.append("detector_hidden_fp_rate_over_0_05")
    if best_precision_10 is None or best_recall_10 is None:
        violations.append("precision_recall_at_10px_missing")
    return {
        "summary": {
            "path_present": True,
            "artifact_type": benchmark.get("artifact_type"),
            "candidate_count": len(candidates),
            "best_candidate": best.get("name") if best else None,
            "best_f1": best_f1,
            "best_recall": best_recall,
            "best_precision_at_10px": best_precision_10,
            "best_recall_at_10px": best_recall_10,
            "best_hidden_false_positive_rate": best_hidden_fp,
        },
        "violations": violations,
    }


def _status(*, has_track: bool, has_benchmark: bool, has_manifest: bool) -> str:
    if has_track and has_benchmark:
        return M1_STATUS_TESTED
    if has_track or has_benchmark or has_manifest:
        return M1_STATUS_SCAFFOLD
    return M1_STATUS_NOT_STARTED


def _find_model(entries: Sequence[Any], needles: Sequence[str], *, exclude: Sequence[str] = ()) -> Mapping[str, Any] | None:
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        model_id = str(entry.get("id") or "").lower()
        if all(token not in model_id for token in exclude) and any(token in model_id for token in needles):
            return entry
    return None


def _missing_model_summary() -> dict[str, Any]:
    return {
        "id": None,
        "source": None,
        "source_official": False,
        "sha256": None,
        "status": None,
        "local_path": None,
        "fine_tuned_on_pickleball": None,
        "license": None,
    }


def _load_optional_ball_track(path: str | Path | None) -> BallTrack | None:
    if path is None or not Path(path).is_file():
        return None
    artifact = validate_artifact_file("ball_track", path)
    if not isinstance(artifact, BallTrack):
        raise ValueError(f"{path} did not validate as BallTrack")
    return artifact


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


def _finite_like(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


def _finite_or_none(value: Any) -> float | None:
    return float(value) if _finite_like(value) else None


def _extend_unique(target: list[str], values: Sequence[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


__all__ = [
    "M1_STATUS_NOT_STARTED",
    "M1_STATUS_SCAFFOLD",
    "M1_STATUS_TESTED",
    "build_ball_detector_gate_report",
    "write_ball_detector_gate_report",
]
