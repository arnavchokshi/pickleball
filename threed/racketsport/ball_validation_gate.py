"""M8 verifier and validation gate report for BALL-only tracking."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schemas import BallTrack, validate_artifact_file
from .wasb_adapter import WASB_CONFIDENCE_SEMANTICS, WASB_MODEL_ZOO_URL, WASB_REPO_URL


M8_STATUS_TESTED = "TESTED-ON-REAL-DATA"
M8_STATUS_SCAFFOLD = "SCAFFOLD"
M8_STATUS_NOT_STARTED = "NOT-STARTED"
MIN_OFFLINE_F1 = 0.90
MIN_BLUR_OCCLUSION_RECALL = 0.75
MAX_HIDDEN_FP_RATE = 0.05
MIN_FULL_SUITE_CLIPS = 4
EXPECTED_MILESTONES: tuple[str, ...] = ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7")
ALLOWED_STATUSES = {M8_STATUS_TESTED, M8_STATUS_SCAFFOLD, M8_STATUS_NOT_STARTED}
EXPECTED_WASB_CHECKPOINT_SHA256 = "9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb"
EXPECTED_WASB_REPO_COMMIT = "923462cacdeb3353b84ddebdedb3f4b7a8553b0f"


def build_ball_validation_gate_report(
    *,
    milestone_report_paths: Mapping[str, str | Path] | None = None,
    tracknet_benchmark_path: str | Path | None = None,
    wasb_track_path: str | Path | None = None,
    wasb_metadata_path: str | Path | None = None,
    wasb_benchmark_path: str | Path | None = None,
    eval_suite_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed M8 report over BALL milestone and verifier artifacts."""

    milestone_reports = _milestone_summary(milestone_report_paths or {})
    offline_detector = _benchmark_summary(
        tracknet_benchmark_path,
        missing_violation="missing_offline_detector_benchmark",
        f1_violation="offline_detector_f1_below_0_90",
        recall_violation="offline_detector_recall_below_0_75",
        fp_violation="offline_detector_hidden_fp_rate_over_0_05",
    )
    wasb_verifier = _wasb_verifier_summary(wasb_track_path, wasb_metadata_path, wasb_benchmark_path)
    eval_suite = _eval_suite_summary(eval_suite_path)
    status = _status(
        offline_detector_present=offline_detector["summary"]["path_present"],
        wasb_track_valid=wasb_verifier["summary"]["track_source"] == "wasb",
        wasb_run_evidence_valid=wasb_verifier["summary"]["run_evidence_valid"],
        wasb_benchmark_present=wasb_verifier["summary"]["benchmark_path_present"],
        eval_suite_present=eval_suite["summary"]["path_present"],
    )

    violations: list[str] = []
    _extend_unique(violations, milestone_reports["violations"])
    _extend_unique(violations, offline_detector["violations"])
    _extend_unique(violations, wasb_verifier["violations"])
    _extend_unique(violations, eval_suite["violations"])

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_validation_gate_report",
        "milestone": "M8 Verifier + validation",
        "status": status,
        "gate_result": "fail" if violations else "pass",
        "blocked_reason": "ball_validation_gate_failed" if violations else None,
        "required_thresholds": {
            "offline_f1_min": MIN_OFFLINE_F1,
            "blur_occlusion_recall_min": MIN_BLUR_OCCLUSION_RECALL,
            "hidden_false_positive_rate_max": MAX_HIDDEN_FP_RATE,
            "full_suite_clip_count_min": MIN_FULL_SUITE_CLIPS,
            "expected_milestones": list(EXPECTED_MILESTONES),
        },
        "milestones": milestone_reports["summary"],
        "offline_detector": offline_detector["summary"],
        "wasb_verifier": wasb_verifier["summary"],
        "eval_suite": eval_suite["summary"],
        "violations": violations,
        "not_ground_truth": True,
    }


def write_ball_validation_gate_report(
    *,
    out: str | Path,
    milestone_report_paths: Mapping[str, str | Path] | None = None,
    tracknet_benchmark_path: str | Path | None = None,
    wasb_track_path: str | Path | None = None,
    wasb_metadata_path: str | Path | None = None,
    wasb_benchmark_path: str | Path | None = None,
    eval_suite_path: str | Path | None = None,
) -> dict[str, Any]:
    report = build_ball_validation_gate_report(
        milestone_report_paths=milestone_report_paths,
        tracknet_benchmark_path=tracknet_benchmark_path,
        wasb_track_path=wasb_track_path,
        wasb_metadata_path=wasb_metadata_path,
        wasb_benchmark_path=wasb_benchmark_path,
        eval_suite_path=eval_suite_path,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _milestone_summary(paths: Mapping[str, str | Path]) -> dict[str, Any]:
    reports: dict[str, dict[str, Any]] = {}
    violations: list[str] = []
    pass_count = 0
    fail_count = 0
    missing_count = 0
    for key in EXPECTED_MILESTONES:
        raw_path = paths.get(key)
        if raw_path is None or not Path(raw_path).is_file():
            reports[key] = {
                "path": str(raw_path) if raw_path is not None else None,
                "path_present": False,
                "milestone": None,
                "status": None,
                "gate_result": None,
                "blocked_reason": None,
            }
            violations.append(f"missing_milestone_report:{key}")
            missing_count += 1
            continue
        payload = _load_json(raw_path)
        status = payload.get("status")
        gate_result = payload.get("gate_result")
        if status not in ALLOWED_STATUSES:
            violations.append(f"milestone_status_invalid:{key}")
        if gate_result != "pass":
            violations.append(f"milestone_gate_not_passed:{key}")
            fail_count += 1
        else:
            pass_count += 1
        reports[key] = {
            "path": str(raw_path),
            "path_present": True,
            "artifact_type": payload.get("artifact_type"),
            "milestone": payload.get("milestone"),
            "status": status,
            "gate_result": gate_result,
            "blocked_reason": payload.get("blocked_reason"),
        }
    return {
        "summary": {
            "expected": list(EXPECTED_MILESTONES),
            "reports": reports,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "missing_count": missing_count,
        },
        "violations": violations,
    }


def _benchmark_summary(
    path: str | Path | None,
    *,
    missing_violation: str,
    f1_violation: str,
    recall_violation: str,
    fp_violation: str,
) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        return {
            "summary": {
                "path_present": False,
                "artifact_type": None,
                "candidate_count": 0,
                "best_candidate": None,
                "best_f1": None,
                "best_recall": None,
                "best_hidden_false_positive_rate": None,
            },
            "violations": [missing_violation],
        }
    payload = _load_json(path)
    aggregate = payload.get("aggregate")
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
                "hidden_fp_rate": _first_finite(
                    metrics.get("micro_hidden_false_positive_rate"),
                    metrics.get("mean_hidden_false_positive_rate"),
                ),
            }
        )
    best = max(scored, key=lambda item: item["f1"] if item["f1"] is not None else -math.inf, default=None)
    best_f1 = best.get("f1") if best else None
    best_recall = best.get("recall") if best else None
    best_hidden_fp = best.get("hidden_fp_rate") if best else None
    violations: list[str] = []
    if best_f1 is None or best_f1 < MIN_OFFLINE_F1:
        violations.append(f1_violation)
    if best_recall is None or best_recall < MIN_BLUR_OCCLUSION_RECALL:
        violations.append(recall_violation)
    if best_hidden_fp is None or best_hidden_fp > MAX_HIDDEN_FP_RATE:
        violations.append(fp_violation)
    return {
        "summary": {
            "path_present": True,
            "path": str(path),
            "artifact_type": payload.get("artifact_type"),
            "candidate_count": len(candidates),
            "best_candidate": best.get("name") if best else None,
            "best_f1": best_f1,
            "best_recall": best_recall,
            "best_hidden_false_positive_rate": best_hidden_fp,
        },
        "violations": violations,
    }


def _wasb_verifier_summary(
    wasb_track_path: str | Path | None,
    wasb_metadata_path: str | Path | None,
    wasb_benchmark_path: str | Path | None,
) -> dict[str, Any]:
    violations: list[str] = []
    track_source = None
    frame_count = None
    visible_frame_count = None
    if wasb_track_path is None or not Path(wasb_track_path).is_file():
        violations.append("missing_wasb_track")
    else:
        track = validate_artifact_file("ball_track", wasb_track_path)
        if not isinstance(track, BallTrack):
            raise ValueError(f"{wasb_track_path} did not validate as BallTrack")
        track_source = track.source
        frame_count = len(track.frames)
        visible_frame_count = sum(1 for frame in track.frames if frame.visible)
        if track.source != "wasb":
            violations.append("wasb_track_source_invalid")

    metadata = _wasb_metadata_summary(
        wasb_metadata_path,
        wasb_track_path=Path(wasb_track_path) if wasb_track_path is not None else None,
        track_frame_count=frame_count,
        track_visible_frame_count=visible_frame_count,
    )
    run_evidence_valid = track_source == "wasb" and not metadata["violations"]

    benchmark = _benchmark_summary(
        wasb_benchmark_path,
        missing_violation="missing_wasb_benchmark",
        f1_violation="wasb_f1_below_0_90",
        recall_violation="wasb_recall_below_0_75",
        fp_violation="wasb_hidden_fp_rate_over_0_05",
    )
    # WASB is a verifier input for M8; its benchmark can fail the detector gates but
    # still be useful evidence. Surface benchmark threshold misses without hiding
    # whether the verifier artifact exists.
    _extend_unique(violations, metadata["violations"])
    _extend_unique(violations, benchmark["violations"])
    return {
        "summary": {
            "track_path_present": wasb_track_path is not None and Path(wasb_track_path).is_file(),
            "track_path": str(wasb_track_path) if wasb_track_path is not None else None,
            "track_source": track_source,
            "frame_count": frame_count,
            "visible_frame_count": visible_frame_count,
            "metadata": metadata["summary"],
            "run_evidence_valid": run_evidence_valid,
            "benchmark_path_present": benchmark["summary"]["path_present"],
            "benchmark": benchmark["summary"],
        },
        "violations": sorted(set(violations)),
    }


def _wasb_metadata_summary(
    path: str | Path | None,
    *,
    wasb_track_path: Path | None,
    track_frame_count: int | None,
    track_visible_frame_count: int | None,
) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        return {
            "summary": {
                "path_present": False,
                "artifact_type": None,
                "status": None,
                "source_mode": None,
                "out": None,
                "predictions_csv": None,
                "predictions_csv_present": False,
                "confidence_semantics": None,
                "visible_threshold": None,
                "official_repo_url": None,
                "official_model_zoo_url": None,
                "device": None,
                "effective_fps": None,
                "processed_frame_count": None,
                "processed_window_count": None,
                "read_frame_count": None,
                "video": None,
                "checkpoint_path": None,
                "checkpoint_sha256": None,
                "wasb_repo": None,
                "wasb_repo_commit": None,
            },
            "violations": ["missing_wasb_run_metadata"],
        }

    payload = _load_json(path)
    runtime_payload = payload.get("runtime")
    runtime = runtime_payload if isinstance(runtime_payload, Mapping) else {}
    checkpoint_payload = runtime.get("wasb_checkpoint") if isinstance(runtime, Mapping) else None
    checkpoint = checkpoint_payload if isinstance(checkpoint_payload, Mapping) else {}

    artifact_type = payload.get("artifact_type")
    status = payload.get("status")
    source_mode = payload.get("source_mode")
    out = payload.get("out")
    predictions_csv = payload.get("predictions_csv")
    predictions_csv_present = _artifact_reference_exists(predictions_csv, artifact_path=Path(path))
    confidence_semantics = payload.get("confidence_semantics")
    visible_threshold = _finite_or_none(payload.get("visible_threshold"))
    official_repo_url = payload.get("official_repo_url")
    official_model_zoo_url = payload.get("official_model_zoo_url")
    metadata_frame_count = _int_or_none(payload.get("frame_count"))
    metadata_visible_frame_count = _int_or_none(payload.get("visible_frame_count"))
    device = runtime.get("device") if isinstance(runtime, Mapping) else None
    effective_fps = _finite_or_none(runtime.get("effective_fps")) if isinstance(runtime, Mapping) else None
    processed_frame_count = _int_or_none(runtime.get("processed_frame_count")) if isinstance(runtime, Mapping) else None
    processed_window_count = _int_or_none(runtime.get("processed_window_count")) if isinstance(runtime, Mapping) else None
    read_frame_count = _int_or_none(runtime.get("read_frame_count")) if isinstance(runtime, Mapping) else None
    video = runtime.get("video") if isinstance(runtime, Mapping) else None
    checkpoint_path = checkpoint.get("path") if isinstance(checkpoint, Mapping) else None
    checkpoint_sha256 = checkpoint.get("sha256") if isinstance(checkpoint, Mapping) else None
    wasb_repo = runtime.get("wasb_repo") if isinstance(runtime, Mapping) else None
    wasb_repo_commit = runtime.get("wasb_repo_commit") if isinstance(runtime, Mapping) else None

    violations: list[str] = []
    if artifact_type != "racketsport_wasb_ball_run":
        violations.append("wasb_metadata_artifact_type_invalid")
    if status != M8_STATUS_TESTED:
        violations.append("wasb_metadata_not_tested_on_real_data")
    if source_mode != "wasb_predict":
        violations.append("wasb_metadata_source_mode_not_predict")
    if payload.get("not_ground_truth") is not True:
        violations.append("wasb_metadata_not_ground_truth_flag_missing")
    if not isinstance(out, str) or not out:
        violations.append("wasb_metadata_out_missing")
    elif wasb_track_path is None or not _paths_match(out, wasb_track_path):
        violations.append("wasb_metadata_out_mismatch")
    if not isinstance(predictions_csv, str) or not predictions_csv:
        violations.append("wasb_predictions_csv_missing")
    elif not predictions_csv_present:
        violations.append("wasb_predictions_csv_not_found")
    if confidence_semantics != WASB_CONFIDENCE_SEMANTICS:
        violations.append("wasb_confidence_semantics_not_heatmap_peak")
    if visible_threshold is None or visible_threshold < 0.5:
        violations.append("wasb_visible_threshold_below_0_50")
    if official_repo_url != WASB_REPO_URL:
        violations.append("wasb_official_repo_url_invalid")
    if official_model_zoo_url != WASB_MODEL_ZOO_URL:
        violations.append("wasb_model_zoo_url_invalid")
    if metadata_frame_count is None or metadata_frame_count != track_frame_count:
        violations.append("wasb_metadata_frame_count_mismatch")
    if metadata_visible_frame_count is None or metadata_visible_frame_count != track_visible_frame_count:
        violations.append("wasb_metadata_visible_frame_count_mismatch")
    if device != "cuda":
        violations.append("wasb_runtime_device_not_cuda")
    if effective_fps is None or effective_fps <= 0:
        violations.append("wasb_runtime_effective_fps_missing")
    if processed_frame_count is None or processed_frame_count <= 0 or processed_frame_count != track_frame_count:
        violations.append("wasb_runtime_processed_frame_count_mismatch")
    if processed_window_count is None or processed_window_count <= 0:
        violations.append("wasb_runtime_processed_window_count_missing")
    if read_frame_count is None or read_frame_count < (processed_frame_count or 0):
        violations.append("wasb_runtime_read_frame_count_invalid")
    if not isinstance(video, str) or not video:
        violations.append("wasb_runtime_video_missing")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        violations.append("wasb_runtime_checkpoint_path_missing")
    if checkpoint_sha256 != EXPECTED_WASB_CHECKPOINT_SHA256:
        violations.append("wasb_runtime_checkpoint_sha256_not_official")
    if not isinstance(wasb_repo, str) or not wasb_repo:
        violations.append("wasb_runtime_repo_path_missing")
    if wasb_repo_commit != EXPECTED_WASB_REPO_COMMIT:
        violations.append("wasb_runtime_repo_commit_not_official")

    return {
        "summary": {
            "path_present": True,
            "path": str(path),
            "artifact_type": artifact_type,
            "status": status,
            "source_mode": source_mode,
            "out": out,
            "predictions_csv": predictions_csv,
            "predictions_csv_present": predictions_csv_present,
            "frame_count": metadata_frame_count,
            "visible_frame_count": metadata_visible_frame_count,
            "confidence_semantics": confidence_semantics,
            "visible_threshold": visible_threshold,
            "official_repo_url": official_repo_url,
            "official_model_zoo_url": official_model_zoo_url,
            "device": device,
            "effective_fps": effective_fps,
            "processed_frame_count": processed_frame_count,
            "processed_window_count": processed_window_count,
            "read_frame_count": read_frame_count,
            "video": video,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
            "wasb_repo": wasb_repo,
            "wasb_repo_commit": wasb_repo_commit,
        },
        "violations": sorted(set(violations)),
    }


def _eval_suite_summary(path: str | Path | None) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        return {
            "summary": {
                "path_present": False,
                "artifact_type": None,
                "clip_count": 0,
                "candidate_count": 0,
                "has_wasb_fusion_candidate": False,
                "wasb_fusion_candidates": [],
            },
            "violations": ["missing_eval_suite_summary"],
        }
    payload = _load_json(path)
    benchmark = payload.get("benchmark")
    aggregate = benchmark.get("aggregate") if isinstance(benchmark, Mapping) else None
    candidates = aggregate if isinstance(aggregate, Mapping) else {}
    candidate_names = [str(name) for name in candidates.keys()]
    wasb_fusion_candidates = [
        name for name in candidate_names if "wasb" in name.lower() and "fusion" in name.lower()
    ]
    clip_count = _int_or_none(payload.get("clip_count")) or 0
    violations: list[str] = []
    if payload.get("artifact_type") != "racketsport_ball_tracking_eval_suite":
        violations.append("eval_suite_artifact_type_invalid")
    if clip_count < MIN_FULL_SUITE_CLIPS:
        violations.append("eval_suite_clip_count_below_4")
    if not candidates:
        violations.append("eval_suite_has_no_benchmark_candidates")
    if not wasb_fusion_candidates:
        violations.append("missing_wasb_fusion_candidate")
    return {
        "summary": {
            "path_present": True,
            "path": str(path),
            "artifact_type": payload.get("artifact_type"),
            "status": payload.get("status"),
            "clip_count": clip_count,
            "candidate_count": len(candidates),
            "has_wasb_fusion_candidate": bool(wasb_fusion_candidates),
            "wasb_fusion_candidates": wasb_fusion_candidates,
        },
        "violations": sorted(set(violations)),
    }


def _status(
    *,
    offline_detector_present: bool,
    wasb_track_valid: bool,
    wasb_run_evidence_valid: bool,
    wasb_benchmark_present: bool,
    eval_suite_present: bool,
) -> str:
    if (
        offline_detector_present
        and wasb_track_valid
        and wasb_run_evidence_valid
        and wasb_benchmark_present
        and eval_suite_present
    ):
        return M8_STATUS_TESTED
    if offline_detector_present or wasb_track_valid or wasb_benchmark_present or eval_suite_present:
        return M8_STATUS_SCAFFOLD
    return M8_STATUS_NOT_STARTED


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


def _paths_match(left: str | Path, right: str | Path) -> bool:
    return _path_key(left) == _path_key(right)


def _path_key(value: str | Path) -> str:
    path = Path(value)
    try:
        return str(path.expanduser().resolve(strict=False))
    except OSError:
        return str(path)


def _artifact_reference_exists(reference: Any, *, artifact_path: Path) -> bool:
    if not isinstance(reference, str) or not reference:
        return False
    path = Path(reference)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(artifact_path.parent / path)
    return any(candidate.is_file() for candidate in candidates)


__all__ = [
    "M8_STATUS_NOT_STARTED",
    "M8_STATUS_SCAFFOLD",
    "M8_STATUS_TESTED",
    "build_ball_validation_gate_report",
    "write_ball_validation_gate_report",
]
