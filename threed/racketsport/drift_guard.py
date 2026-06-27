"""Tripod bump and calibration drift checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .court_calibration import CALIBRATION_REPROJECTION_P95_GATE_PX, project_planar_points, reprojection_error
from .schemas import ReprojectionError


@dataclass(frozen=True)
class DriftCheck:
    frame_index: int
    reprojection_error_px: ReprojectionError
    recalibration_required: bool
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
