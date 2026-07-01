"""M0 capture protocol validation for BALL-only tracking."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .io_decode import FrameSource, probe_clip
from .schemas import CaptureSidecar, validate_artifact_file


M0_STATUS_TESTED = "TESTED-ON-REAL-DATA"
SIDECAR_FILENAME = "capture_sidecar.json"


def build_ball_capture_protocol_report(
    *,
    video_path: str | Path,
    sidecar_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed M0 report from a real video and optional sidecar."""

    video = probe_clip(video_path)
    resolved_sidecar_path = _resolve_sidecar_path(video.path, sidecar_path)
    video_violations = _video_protocol_violations(video)
    if resolved_sidecar_path is None:
        return _report(
            video=video,
            sidecar_path=None,
            sidecar=None,
            violations=_unique(["missing_capture_sidecar", *video_violations]),
            blocked_reason="missing_capture_sidecar",
        )

    sidecar = validate_artifact_file("capture_sidecar", resolved_sidecar_path)
    if not isinstance(sidecar, CaptureSidecar):
        raise ValueError(f"{resolved_sidecar_path} did not validate as CaptureSidecar")

    violations = _unique([*video_violations, *_capture_protocol_violations(video=video, sidecar=sidecar)])
    return _report(
        video=video,
        sidecar_path=resolved_sidecar_path,
        sidecar=sidecar,
        violations=violations,
        blocked_reason="capture_protocol_failed" if violations else None,
    )


def write_ball_capture_protocol_report(
    *,
    video_path: str | Path,
    out: str | Path,
    sidecar_path: str | Path | None = None,
) -> dict[str, Any]:
    report = build_ball_capture_protocol_report(video_path=video_path, sidecar_path=sidecar_path)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _capture_protocol_violations(*, video: FrameSource, sidecar: CaptureSidecar) -> list[str]:
    violations: list[str] = []

    def add(reason: str) -> None:
        if reason not in violations:
            violations.append(reason)

    width, height = int(video.width), int(video.height)
    if sidecar.audio_recorded is not True:
        add("audio_missing")
    if sidecar.orientation != "landscape":
        add("orientation_not_landscape")
    if tuple(sidecar.resolution) != (width, height):
        add("sidecar_resolution_mismatch")
    if not math.isclose(float(sidecar.fps), float(video.fps), abs_tol=2.0):
        add("sidecar_fps_mismatch")

    if sidecar.locked.exposure_s > 1 / 500:
        add("shutter_slower_than_1_500")
    elif sidecar.locked.exposure_s > 1 / 1000:
        add("shutter_slower_than_1_1000_target")
    if sidecar.exposure_locked is not True:
        add("exposure_not_locked" if sidecar.exposure_locked is False else "exposure_lock_unknown")
    if sidecar.focus_locked is not True:
        add("focus_not_locked" if sidecar.focus_locked is False else "focus_lock_unknown")
    if sidecar.locked.wb_locked is not True:
        add("white_balance_not_locked")

    if sidecar.hdr_enabled is not False:
        add("hdr_enabled" if sidecar.hdr_enabled is True else "hdr_state_unknown")
    if sidecar.video_stabilization_enabled is not False:
        add(
            "video_stabilization_enabled"
            if sidecar.video_stabilization_enabled is True
            else "video_stabilization_state_unknown"
        )

    if sidecar.tripod_height_m is None:
        add("tripod_height_unknown")
    elif sidecar.tripod_height_m < 1.5:
        add("tripod_height_below_1_5m")
    if sidecar.full_court_visible is not True:
        add("full_court_not_visible" if sidecar.full_court_visible is False else "full_court_visibility_unknown")
    if sidecar.court_lock_passed is not True:
        add("court_lock_failed" if sidecar.court_lock_passed is False else "court_lock_unknown")
    arkit_court_seed_present = sidecar.arkit_camera_pose is not None and sidecar.court_plane is not None
    manual_tap_count = len(sidecar.manual_court_taps)
    manual_court_seed_present = manual_tap_count == 4
    if not arkit_court_seed_present and not manual_court_seed_present:
        if manual_tap_count not in {0, 4}:
            add("manual_court_taps_incomplete")
        else:
            add("court_calibration_seed_missing")
    if sidecar.ball_high_contrast is not True:
        add("ball_low_contrast" if sidecar.ball_high_contrast is False else "ball_contrast_unknown")

    if sidecar.capture_quality.grade == "poor":
        add("capture_quality_poor")
    elif sidecar.capture_quality.grade == "warn":
        add("capture_quality_warn")

    return violations


def _video_protocol_violations(video: FrameSource) -> list[str]:
    violations: list[str] = []
    if min(int(video.width), int(video.height)) < 1080:
        violations.append("resolution_below_1080p")
    if float(video.fps) < 60.0:
        violations.append("fps_below_60")
    if video.audio_sample_rate is None:
        violations.append("audio_missing")
    return violations


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _report(
    *,
    video: FrameSource,
    sidecar_path: Path | None,
    sidecar: CaptureSidecar | None,
    violations: list[str],
    blocked_reason: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_capture_protocol_report",
        "milestone": "M0 Capture",
        "status": M0_STATUS_TESTED,
        "gate_result": "fail" if violations else "pass",
        "blocked_reason": blocked_reason,
        "video": {
            "path": str(video.path),
            "resolution": [int(video.width), int(video.height)],
            "fps": float(video.fps),
            "duration_s": float(video.duration_s),
            "frame_count": video.frame_count,
            "audio_present": video.audio_sample_rate is not None,
            "audio_sample_rate": video.audio_sample_rate,
        },
        "sidecar_path": str(sidecar_path) if sidecar_path is not None else None,
        "sidecar": _sidecar_summary(sidecar) if sidecar is not None else None,
        "violations": violations,
        "not_ground_truth": True,
    }


def _sidecar_summary(sidecar: CaptureSidecar) -> dict[str, Any]:
    return {
        "device_model": sidecar.device_model,
        "device_tier": sidecar.device_tier,
        "resolution": list(sidecar.resolution),
        "fps": int(sidecar.fps),
        "orientation": sidecar.orientation,
        "format": sidecar.format,
        "exposure_s": float(sidecar.locked.exposure_s),
        "hdr_enabled": sidecar.hdr_enabled,
        "video_stabilization_enabled": sidecar.video_stabilization_enabled,
        "exposure_locked": sidecar.exposure_locked,
        "focus_locked": sidecar.focus_locked,
        "wb_locked": sidecar.locked.wb_locked,
        "tripod_height_m": sidecar.tripod_height_m,
        "full_court_visible": sidecar.full_court_visible,
        "court_lock_passed": sidecar.court_lock_passed,
        "arkit_court_seed_present": sidecar.arkit_camera_pose is not None and sidecar.court_plane is not None,
        "manual_court_tap_count": len(sidecar.manual_court_taps),
        "ball_high_contrast": sidecar.ball_high_contrast,
        "audio_recorded": sidecar.audio_recorded,
        "capture_quality": sidecar.capture_quality.model_dump(mode="json"),
    }


def _resolve_sidecar_path(video_path: Path, sidecar_path: str | Path | None) -> Path | None:
    if sidecar_path is not None:
        return Path(sidecar_path)
    sibling = video_path.parent / SIDECAR_FILENAME
    return sibling if sibling.is_file() else None


__all__ = [
    "M0_STATUS_TESTED",
    "build_ball_capture_protocol_report",
    "write_ball_capture_protocol_report",
]
