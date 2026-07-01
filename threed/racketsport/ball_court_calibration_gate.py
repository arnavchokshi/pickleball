"""M3 court-calibration gate report for BALL-only tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .court_calibration import passes_reprojection_gate
from .io_decode import FrameSource, probe_clip
from .schemas import CourtCalibration, validate_artifact_file


M3_STATUS_TESTED = "TESTED-ON-REAL-DATA"
M3_STATUS_SCAFFOLD = "SCAFFOLD"
TRUSTED_CALIBRATION_SOURCES = {
    "arkit_plane_keypoint_metric_solve_v1",
    "manual_metric_fallback_v1",
}
TRUSTED_INTRINSICS_SOURCES = {
    "arkit",
    "calibrated_charuco",
    "calibrated_checkerboard",
    "device_profile_calibrated",
}
UNTRUSTED_CAPTURE_REASON_TOKENS = (
    "prototype",
    "estimated_intrinsics",
    "corrected_unverified",
    "unverified",
)


def build_ball_court_calibration_gate_report(
    *,
    calibration_path: str | Path,
    video_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed M3 report from a court calibration and optional real clip."""

    video = probe_clip(video_path) if video_path is not None else None
    resolved_calibration_path = Path(calibration_path)
    if not resolved_calibration_path.is_file():
        return _report(
            calibration_path=resolved_calibration_path,
            calibration=None,
            video=video,
            violations=["missing_court_calibration"],
            blocked_reason="missing_court_calibration",
        )

    calibration = validate_artifact_file("court_calibration", resolved_calibration_path)
    if not isinstance(calibration, CourtCalibration):
        raise ValueError(f"{resolved_calibration_path} did not validate as CourtCalibration")

    violations = _court_calibration_violations(calibration)
    return _report(
        calibration_path=resolved_calibration_path,
        calibration=calibration,
        video=video,
        violations=violations,
        blocked_reason="court_calibration_gate_failed" if violations else None,
    )


def write_ball_court_calibration_gate_report(
    *,
    calibration_path: str | Path,
    out: str | Path,
    video_path: str | Path | None = None,
) -> dict[str, Any]:
    report = build_ball_court_calibration_gate_report(calibration_path=calibration_path, video_path=video_path)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _court_calibration_violations(calibration: CourtCalibration) -> list[str]:
    violations: list[str] = []

    def add(reason: str) -> None:
        if reason not in violations:
            violations.append(reason)

    if not passes_reprojection_gate(calibration.reprojection_error_px):
        add("reprojection_gate_failed")

    metric_field_values = {
        "coordinate_frame": calibration.coordinate_frame,
        "T_world_court": calibration.T_world_court,
        "per_keypoint_residual_px": calibration.per_keypoint_residual_px,
        "metric_confidence": calibration.metric_confidence,
        "gsd_model": calibration.gsd_model,
        "source": calibration.source,
        "solved_over_frames": calibration.solved_over_frames,
    }
    if any(value is None for value in metric_field_values.values()):
        add("metric_fields_incomplete")

    source = (calibration.source or "").strip()
    if not source:
        add("calibration_source_missing")
    elif source not in TRUSTED_CALIBRATION_SOURCES:
        add("calibration_source_not_trusted")

    metric_confidence = calibration.metric_confidence
    if metric_confidence is None:
        add("metric_confidence_missing")
    elif metric_confidence != "high":
        add("metric_confidence_not_high")

    intrinsics_source = calibration.intrinsics.source.strip().lower()
    if intrinsics_source not in TRUSTED_INTRINSICS_SOURCES:
        add("intrinsics_not_trusted")

    capture_grade = calibration.capture_quality.grade
    if capture_grade == "poor":
        add("capture_quality_poor")
    elif capture_grade == "warn":
        add("capture_quality_warn")

    for reason in calibration.capture_quality.reasons:
        if _untrusted_capture_reason(reason):
            add(f"capture_quality_unverified:{reason}")

    return violations


def _report(
    *,
    calibration_path: Path,
    calibration: CourtCalibration | None,
    video: FrameSource | None,
    violations: list[str],
    blocked_reason: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_court_calibration_gate_report",
        "milestone": "M3 Court",
        "status": M3_STATUS_TESTED if video is not None else M3_STATUS_SCAFFOLD,
        "gate_result": "fail" if violations else "pass",
        "blocked_reason": blocked_reason,
        "calibration_path": str(calibration_path),
        "video": _video_summary(video) if video is not None else None,
        "reprojection_error_px": (
            calibration.reprojection_error_px.model_dump(mode="json") if calibration is not None else None
        ),
        "reprojection_gate": {
            "median_px_lt": 8.0,
            "p95_px_lt": 15.0,
            "passed": passes_reprojection_gate(calibration.reprojection_error_px) if calibration is not None else False,
        },
        "calibration_source": calibration.source if calibration is not None else None,
        "metric_confidence": calibration.metric_confidence if calibration is not None else None,
        "metric_fields_complete": _metric_fields_complete(calibration) if calibration is not None else False,
        "intrinsics_source": calibration.intrinsics.source if calibration is not None else None,
        "capture_quality": (
            calibration.capture_quality.model_dump(mode="json") if calibration is not None else None
        ),
        "violations": violations,
        "not_ground_truth": True,
    }


def _video_summary(video: FrameSource) -> dict[str, Any]:
    return {
        "path": str(video.path),
        "resolution": [int(video.width), int(video.height)],
        "fps": float(video.fps),
        "duration_s": float(video.duration_s),
        "frame_count": video.frame_count,
        "audio_present": video.audio_sample_rate is not None,
        "audio_sample_rate": video.audio_sample_rate,
    }


def _metric_fields_complete(calibration: CourtCalibration) -> bool:
    return all(
        value is not None
        for value in (
            calibration.coordinate_frame,
            calibration.T_world_court,
            calibration.per_keypoint_residual_px,
            calibration.metric_confidence,
            calibration.gsd_model,
            calibration.source,
            calibration.solved_over_frames,
        )
    )


def _untrusted_capture_reason(reason: str) -> bool:
    lowered = reason.lower()
    return any(token in lowered for token in UNTRUSTED_CAPTURE_REASON_TOKENS)


__all__ = [
    "M3_STATUS_SCAFFOLD",
    "M3_STATUS_TESTED",
    "build_ball_court_calibration_gate_report",
    "write_ball_court_calibration_gate_report",
]
