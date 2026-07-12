from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from threed.racketsport.court_temporal_lock import (
    COURT_LINES_M,
    CourtLineObservation,
    TemporalCourtLock,
    TemporalCourtLockConfig,
    validate_trusted_calibration,
)


def _calibration(h: np.ndarray | None = None) -> dict[str, object]:
    homography = h if h is not None else np.asarray(
        [[65.0, 0.0, 320.0], [0.0, 28.0, 210.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return {
        "homography": homography.tolist(),
        "image_size": [640, 480],
        "source": "reviewed_fixture",
        "intrinsics": {"dist": [0.0, 0.0, 0.0, 0.0]},
    }


def _project(h: np.ndarray, point: tuple[float, float]) -> np.ndarray:
    projected = h @ np.asarray([point[0], point[1], 1.0], dtype=np.float64)
    return projected[:2] / projected[2]


def _render(h: np.ndarray, *, visible: bool = True) -> np.ndarray:
    frame = np.full((480, 640, 3), (45, 100, 45), dtype=np.uint8)
    if not visible:
        return frame
    for court_a, court_b in COURT_LINES_M.values():
        image_a = tuple(np.rint(_project(h, court_a)).astype(int))
        image_b = tuple(np.rint(_project(h, court_b)).astype(int))
        cv2.line(frame, image_a, image_b, (245, 245, 245), 4, cv2.LINE_AA)
    return frame


def _observations(h: np.ndarray, *, offset_px: float = 0.0) -> list[CourtLineObservation]:
    observations: list[CourtLineObservation] = []
    for line_id, (court_a, court_b) in COURT_LINES_M.items():
        image_a = _project(h, court_a)
        image_b = _project(h, court_b)
        direction = image_b - image_a
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        for t in (0.2, 0.5, 0.8):
            court_xy = (
                court_a[0] + (court_b[0] - court_a[0]) * t,
                court_a[1] + (court_b[1] - court_a[1]) * t,
            )
            image_xy = _project(h, court_xy) + normal * offset_px
            observations.append(
                CourtLineObservation(
                    line_id=line_id,
                    court_xy=court_xy,
                    image_xy=(float(image_xy[0]), float(image_xy[1])),
                    normal=(float(normal[0]), float(normal[1])),
                    variance_px2=0.25,
                )
            )
    return observations


def _corner_errors(h: np.ndarray, truth: np.ndarray) -> list[float]:
    corners = ((-3.048, -6.7056), (3.048, -6.7056), (3.048, 6.7056), (-3.048, 6.7056))
    return [float(np.linalg.norm(_project(h, point) - _project(truth, point))) for point in corners]


def test_synthetic_pan_rendered_template_tracks_under_two_pixels_and_bounds_drift() -> None:
    reference_h = np.asarray(_calibration()["homography"], dtype=np.float64)
    lock = TemporalCourtLock(_calibration(), config=TemporalCourtLockConfig(fps=30.0))
    errors: list[float] = []
    for frame_idx in range(60):
        current_from_reference = np.asarray(
            [[1.0, 0.0, 0.8 * frame_idx], [0.0, 1.0, 0.15 * frame_idx], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        truth_h = current_from_reference @ reference_h
        current_to_reference = np.linalg.inv(current_from_reference)
        frame = lock.step(
            frame_idx,
            frame_bgr=_render(truth_h),
            motion={
                "M": current_to_reference.tolist(),
                "compensated": True,
                "inlier_ratio": 0.95,
            },
        )
        assert frame["H_court_to_image"] is not None
        errors.extend(_corner_errors(np.asarray(frame["H_court_to_image"]), truth_h))
    assert float(np.median(errors)) < 2.0
    assert max(errors[-4:]) < 2.0


def test_static_rendered_template_adds_at_most_point_two_px_p95_jitter() -> None:
    h = np.asarray(_calibration()["homography"], dtype=np.float64)
    lock = TemporalCourtLock(_calibration())
    tracked_corners: list[np.ndarray] = []
    for frame_idx in range(45):
        result = lock.step(
            frame_idx,
            frame_bgr=_render(h),
            motion={"M": np.eye(3).tolist(), "compensated": True, "inlier_ratio": 1.0},
        )
        result_h = np.asarray(result["H_court_to_image"], dtype=np.float64)
        tracked_corners.append(np.concatenate([_project(result_h, point) for point in COURT_LINES_M["near_baseline"]]))
    baseline = tracked_corners[0]
    jitter = [float(np.max(np.abs(corners - baseline))) for corners in tracked_corners]
    assert float(np.percentile(jitter, 95)) <= 0.2


def test_occlusion_grows_covariance_coasts_and_recovers_within_fifteen_frames() -> None:
    h = np.asarray(_calibration()["homography"], dtype=np.float64)
    lock = TemporalCourtLock(
        _calibration(),
        config=TemporalCourtLockConfig(max_coast_frames=20, min_measurements=6),
    )
    first = lock.step(0, observations=_observations(h), motion={"M": np.eye(3).tolist()})
    traces: list[float] = []
    states: list[str] = []
    for frame_idx in range(1, 11):
        result = lock.step(frame_idx, observations=[], motion={"M": np.eye(3).tolist()})
        traces.append(float(result["covariance"]["trace"]))
        states.append(str(result["lock_state"]))
    assert all(b > a for a, b in zip(traces, traces[1:]))
    assert set(states) == {"coasting"}
    recovered = lock.step(11, observations=_observations(h), motion={"M": np.eye(3).tolist()})
    assert recovered["lock_state"] == "locked"
    assert recovered["provenance"]["kind"] == "reset"
    assert recovered["provenance"]["reason"] == "evidence_recovery"


def test_long_missing_window_fails_closed_instead_of_freezing_precise() -> None:
    h = np.asarray(_calibration()["homography"], dtype=np.float64)
    lock = TemporalCourtLock(
        _calibration(),
        config=TemporalCourtLockConfig(max_coast_frames=2, min_measurements=6),
    )
    lock.step(0, observations=_observations(h), motion={"M": np.eye(3).tolist()})
    lock.step(1, observations=[], motion=None)
    lock.step(2, observations=[], motion=None)
    absent = lock.step(3, observations=[], motion=None)
    assert absent["lock_state"] == "absent"
    assert absent["provenance"]["kind"] == "missing"
    assert absent["provenance"]["reason"] == "coast_limit_exceeded"


def test_hard_cut_starts_new_generation_and_never_warps_old_homography() -> None:
    h = np.asarray(_calibration()["homography"], dtype=np.float64)
    lock = TemporalCourtLock(_calibration())
    before = lock.step(0, observations=_observations(h), motion={"M": np.eye(3).tolist()})
    cut = lock.step(1, observations=_observations(h), hard_cut=True)
    after = lock.step(2, observations=_observations(h), motion={"M": np.eye(3).tolist()})
    assert before["provenance"]["reference_generation"] == 0
    assert cut["provenance"]["reference_generation"] == 1
    assert cut["provenance"]["kind"] == "reset"
    assert cut["H_court_to_image"] is None
    assert after["lock_state"] == "absent"
    assert after["H_court_to_image"] is None


def test_camera_motion_direction_is_inverse_m_times_reference_h() -> None:
    h = np.asarray(_calibration()["homography"], dtype=np.float64)
    current_from_reference = np.asarray(
        [[1.0, 0.0, 4.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    lock = TemporalCourtLock(_calibration())
    result = lock.step(
        0,
        observations=[],
        motion={"M": np.linalg.inv(current_from_reference).tolist(), "compensated": True},
    )
    assert np.asarray(result["H_court_to_image"]) == pytest.approx(current_from_reference @ h)


def test_robust_update_is_partial_and_does_not_snap_to_measurement() -> None:
    h = np.asarray(_calibration()["homography"], dtype=np.float64)
    lock = TemporalCourtLock(
        _calibration(),
        config=TemporalCourtLockConfig(measurement_deadband_px=0.0, min_measurements=6),
    )
    result = lock.step(0, observations=_observations(h, offset_px=2.0), motion={"M": np.eye(3).tolist()})
    moved = np.linalg.norm(_project(np.asarray(result["H_court_to_image"]), (0.0, 0.0)) - _project(h, (0.0, 0.0)))
    assert 0.0 < moved < 2.0
    assert result["evidence"]["update"]["gain_cap"] == 0.75


def test_trusted_seed_validation_rejects_auto_proposal() -> None:
    calibration = _calibration()
    calibration["source"] = "auto_preview_proposal"
    with pytest.raises(ValueError, match="not a trusted"):
        validate_trusted_calibration(calibration)
