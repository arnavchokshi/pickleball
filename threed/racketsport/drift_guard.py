"""Tripod bump and calibration drift checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .court_calibration import CALIBRATION_REPROJECTION_P95_GATE_PX, project_planar_points, reprojection_error
from .schemas import DriftLog, ReprojectionError


@dataclass(frozen=True)
class DriftCheck:
    frame_index: int
    reprojection_error_px: ReprojectionError
    recalibration_required: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DriftSequenceStatus:
    checks: list[dict[str, float | int | bool]]
    recalibration_required: bool
    recalibration_from_frame: int | None
    reasons: list[str] = field(default_factory=list)


def should_check_frame(frame_index: int, *, every_n_frames: int) -> bool:
    if every_n_frames <= 0:
        raise ValueError("every_n_frames must be positive")
    return frame_index == 0 or frame_index % every_n_frames == 0


def verify_drift(
    *,
    homography: Iterable[Iterable[float]],
    world_pts: Iterable[Iterable[float]],
    observed_image_pts: Iterable[Iterable[float]],
    frame_index: int,
    p95_gate_px: float = CALIBRATION_REPROJECTION_P95_GATE_PX,
) -> DriftCheck:
    projected = project_planar_points(homography, world_pts)
    error = reprojection_error(observed_image_pts, projected)
    reasons = ["reprojection_drift"] if error.p95 > p95_gate_px else []
    return DriftCheck(
        frame_index=frame_index,
        reprojection_error_px=error,
        recalibration_required=bool(reasons),
        reasons=reasons,
    )


def verify(
    *,
    homography: Iterable[Iterable[float]],
    world_pts: Iterable[Iterable[float]],
    observed_image_pts: Iterable[Iterable[float]],
    frame_index: int,
    p95_gate_px: float = CALIBRATION_REPROJECTION_P95_GATE_PX,
) -> DriftCheck:
    """Doc-compatible wrapper for the Phase 1 drift check."""

    return verify_drift(
        homography=homography,
        world_pts=world_pts,
        observed_image_pts=observed_image_pts,
        frame_index=frame_index,
        p95_gate_px=p95_gate_px,
    )


def evaluate_drift_sequence(
    checks: Iterable[tuple[int, float]],
    *,
    p95_gate_px: float = 8.0,
    consecutive_failures: int = 3,
) -> DriftSequenceStatus:
    """Apply the Stage D tripod-bump rule across cheap verification checks."""

    if p95_gate_px < 0.0:
        raise ValueError("p95_gate_px must be non-negative")
    if consecutive_failures <= 0:
        raise ValueError("consecutive_failures must be positive")

    serialized_checks: list[dict[str, float | int | bool]] = []
    failing_window: list[int] = []
    recalibration_from_frame: int | None = None
    for frame_index, p95_px in checks:
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if p95_px < 0.0:
            raise ValueError("p95_px must be non-negative")
        failed = float(p95_px) > p95_gate_px
        if failed:
            failing_window.append(int(frame_index))
        else:
            failing_window.clear()
        tripped = len(failing_window) >= consecutive_failures
        if tripped and recalibration_from_frame is None:
            recalibration_from_frame = failing_window[-consecutive_failures]
        serialized_checks.append({"frame": int(frame_index), "p95_px": float(p95_px), "tripped": tripped})

    reasons = ["reprojection_drift_3_consecutive"] if recalibration_from_frame is not None else []
    return DriftSequenceStatus(
        checks=serialized_checks,
        recalibration_required=recalibration_from_frame is not None,
        recalibration_from_frame=recalibration_from_frame,
        reasons=reasons,
    )


def build_drift_log_artifact(
    status: DriftSequenceStatus,
    *,
    recalibration_to_frame: int | None = None,
) -> dict[str, object]:
    """Serialize Stage D checks and recalibration spans to drift_log.json."""

    recalibrations: list[dict[str, int | str]] = []
    if status.recalibration_required:
        if status.recalibration_from_frame is None:
            raise ValueError("recalibration_from_frame is required when recalibration_required is true")
        if not status.checks:
            raise ValueError("drift checks are required when recalibration_required is true")
        to_frame = int(recalibration_to_frame) if recalibration_to_frame is not None else int(status.checks[-1]["frame"])
        reason = status.reasons[0] if status.reasons else "reprojection_drift_3_consecutive"
        recalibrations.append(
            {
                "from_frame": int(status.recalibration_from_frame),
                "to_frame": to_frame,
                "reason": reason,
            }
        )

    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_drift_log",
        "checks": status.checks,
        "recalibrations": recalibrations,
    }
    return DriftLog.model_validate(payload).model_dump(mode="json")
