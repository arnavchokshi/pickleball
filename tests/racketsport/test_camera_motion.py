from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from threed.racketsport.camera_motion import (
    CameraMotionParams,
    build_court_mask,
    detect_reference_features,
    estimate_camera_motion,
    mask_people_for_frame,
    transform_sanity_reason,
    validate_camera_motion_payload,
    write_camera_motion_json,
)
import threed.racketsport.camera_motion as camera_motion_module


CLI_PATH = "scripts/racketsport/estimate_camera_motion.py"


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 30.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def _textured_court(width: int = 180, height: int = 120) -> np.ndarray:
    rng = np.random.default_rng(1234)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (40, 120, 60)
    cv2.rectangle(image, (12, 12), (width - 13, height - 13), (235, 235, 235), 2)
    cv2.line(image, (width // 2, 12), (width // 2, height - 13), (235, 235, 235), 1)
    cv2.line(image, (12, height // 2), (width - 13, height // 2), (235, 235, 235), 1)
    for _ in range(180):
        x = int(rng.integers(18, width - 18))
        y = int(rng.integers(18, height - 18))
        color = int(rng.integers(80, 190))
        cv2.circle(image, (x, y), 1, (color, color, color), -1)
    return image


def _translated_frames(offsets: list[tuple[float, float]]) -> list[np.ndarray]:
    base = _textured_court()
    height, width = base.shape[:2]
    frames = []
    for dx, dy in offsets:
        affine = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        frames.append(cv2.warpAffine(base, affine, (width, height), borderValue=(40, 120, 60)))
    return frames


def _calibration(path: Path, width: int = 180, height: int = 120, reference: int = 0) -> Path:
    payload = {
        "schema_version": 1,
        "image_size": [width, height],
        "solved_over_frames": [reference],
        "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "image_pts": [[12.0, 12.0], [width - 13.0, 12.0], [width - 13.0, height - 13.0], [12.0, height - 13.0]],
        "world_pts": [[12.0, 12.0, 0.0], [width - 13.0, 12.0, 0.0], [width - 13.0, height - 13.0, 0.0], [12.0, height - 13.0, 0.0]],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _tracks(path: Path, bbox: list[float]) -> Path:
    payload = {
        "schema_version": 1,
        "fps": 30.0,
        "players": [{"id": 1, "frames": [{"frame_idx": 0, "bbox": bbox, "conf": 0.9}]}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_known_synthetic_pan_recovers_frame_to_reference_transform(tmp_path: Path) -> None:
    offsets = [(0.0, 0.0), (4.0, 2.0), (8.0, -3.0), (12.0, 5.0)]
    video = tmp_path / "pan.avi"
    _write_video(video, _translated_frames(offsets))
    calibration = _calibration(tmp_path / "court_calibration.json")

    payload = estimate_camera_motion(
        video,
        calibration,
        params=CameraMotionParams(min_homography_inliers=8, max_ransac_reproj_px=2.0),
    )

    validate_camera_motion_payload(payload)
    assert payload["reference_frame_idx"] == 0
    assert payload["summary"]["n_compensated"] == len(offsets)
    for frame, (dx, dy) in zip(payload["frames"], offsets, strict=True):
        assert frame["compensated"] is True
        matrix = np.asarray(frame["M"], dtype=np.float64)
        assert np.allclose(matrix[0, 2], -dx, atol=1.75)
        assert np.allclose(matrix[1, 2], -dy, atol=1.75)
        assert frame["rms_px"] <= 2.0


def test_scaled_processing_preserves_original_pixel_transform_units(tmp_path: Path) -> None:
    offsets = [(0.0, 0.0), (6.0, -4.0), (10.0, 3.0)]
    video = tmp_path / "scaled_pan.avi"
    _write_video(video, _translated_frames(offsets))
    calibration = _calibration(tmp_path / "court_calibration.json")

    payload = estimate_camera_motion(
        video,
        calibration,
        params=CameraMotionParams(
            processing_scale=0.5,
            temporal_smoothing=False,
            min_homography_inliers=8,
            max_ransac_reproj_px=2.0,
        ),
    )

    for frame, (dx, dy) in zip(payload["frames"], offsets, strict=True):
        matrix = np.asarray(frame["M"], dtype=np.float64)
        assert np.allclose(matrix[0, 2], -dx, atol=2.0)
        assert np.allclose(matrix[1, 2], -dy, atol=2.0)


def test_flow_mad_filter_removes_large_moving_player_outliers() -> None:
    ref = np.array(
        [
            [20.0, 20.0],
            [40.0, 20.0],
            [60.0, 20.0],
            [20.0, 45.0],
            [40.0, 45.0],
            [60.0, 45.0],
            [35.0, 35.0],
            [55.0, 35.0],
        ],
        dtype=np.float32,
    )
    cur = ref + np.array([4.0, -2.0], dtype=np.float32)
    cur[-2:] += np.array([[38.0, 20.0], [-42.0, -18.0]], dtype=np.float32)
    params = CameraMotionParams(flow_mad_z=3.0, min_flow_mad_survivors=4)

    filtered_ref, filtered_cur, stats = camera_motion_module._mad_filter_flow_tracks(ref, cur, params)

    assert len(filtered_ref) == 6
    assert len(filtered_cur) == 6
    assert stats["flow_track_count"] == 8
    assert stats["flow_mad_filtered_count"] == 2
    assert np.allclose(filtered_cur - filtered_ref, np.array([4.0, -2.0]), atol=1e-6)


def test_temporal_smoothing_mad_then_gaussian_reduces_camera_path_spike() -> None:
    translations = [0.0, -1.0, -2.0, -28.0, -4.0, -5.0, -6.0]
    frames = []
    for frame_idx, tx in enumerate(translations):
        matrix = [[1.0, 0.0, tx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        frames.append(
            {
                "frame_idx": frame_idx,
                "M": matrix,
                "inlier_count": 20,
                "rms_px": 0.2,
                "compensated": True,
                "model": "similarity",
            }
        )
    params = CameraMotionParams(
        temporal_smoothing=True,
        temporal_mad_z=3.0,
        temporal_gaussian_sigma_frames=0.8,
        temporal_gaussian_radius_frames=2,
    )

    smoothed, stats = camera_motion_module._smooth_camera_motion_frames(frames, params)
    raw_path = np.array([[frame["M"][0][2], frame["M"][1][2]] for frame in frames], dtype=np.float64)
    smooth_path = np.array([[frame["M"][0][2], frame["M"][1][2]] for frame in smoothed], dtype=np.float64)

    assert abs(smooth_path[3, 0] + 3.0) < abs(raw_path[3, 0] + 3.0)
    assert _path_jerk_rms(smooth_path) < _path_jerk_rms(raw_path)
    assert stats["temporal_mad_replaced_count"] >= 1
    assert stats["temporal_smoothed_frame_count"] == len(frames)


def test_default_params_are_hardened_and_legacy_params_preserve_baseline_switch() -> None:
    default = CameraMotionParams()
    legacy = CameraMotionParams.legacy()

    assert default.estimator_mode == "hardened"
    assert default.flow_mad_filter is True
    assert default.temporal_smoothing is True
    assert legacy.estimator_mode == "legacy"
    assert legacy.flow_mad_filter is False
    assert legacy.temporal_smoothing is False
    assert legacy.use_person_masks is True


def test_motion_probe_scores_pan_above_static_and_reports_sampling_budget(tmp_path: Path) -> None:
    assert hasattr(camera_motion_module, "estimate_camera_motion_probe")
    probe_fn = camera_motion_module.estimate_camera_motion_probe
    static_video = tmp_path / "static.avi"
    pan_video = tmp_path / "pan.avi"
    calibration = _calibration(tmp_path / "court_calibration.json")
    _write_video(static_video, _translated_frames([(0.0, 0.0) for _ in range(6)]))
    _write_video(pan_video, _translated_frames([(float(i * 4), 0.0) for i in range(6)]))

    params = CameraMotionParams(
        processing_scale=0.5,
        temporal_smoothing=False,
        min_homography_inliers=8,
        max_ransac_reproj_px=2.0,
    )
    static_probe = probe_fn(static_video, calibration, params=params, frame_step=2, max_probe_frames=4)
    pan_probe = probe_fn(pan_video, calibration, params=params, frame_step=2, max_probe_frames=4)

    assert static_probe["motion_score"] <= 0.05
    assert pan_probe["motion_score"] > static_probe["motion_score"] + 6.0
    assert pan_probe["sampled_frame_count"] <= 4
    assert pan_probe["frame_step"] == 2
    assert pan_probe["threshold"] > 0.0
    assert pan_probe["enabled"] == (pan_probe["motion_score"] > pan_probe["threshold"])
    assert pan_probe["forced"] == "auto"
    assert pan_probe["verified"] is False
    assert pan_probe["not_gate_verified"] is True


def test_dense_flow_backend_adapter_uses_synthetic_flow_without_weights() -> None:
    reference_points = np.array([[[16.0, 12.0]], [[40.0, 12.0]], [[16.0, 32.0]], [[40.0, 32.0]]], dtype=np.float32)
    flow = np.zeros((48, 64, 2), dtype=np.float32)
    flow[..., 0] = 5.0
    flow[..., 1] = -3.0
    mask = np.full((48, 64), 255, dtype=np.uint8)
    params = CameraMotionParams(min_similarity_inliers=4)

    ref, cur, stats = camera_motion_module._tracks_from_dense_flow(reference_points, flow, mask)
    estimate = camera_motion_module._fit_similarity(cur, ref, params)
    status = camera_motion_module.raft_small_backend_status()

    assert stats["flow_backend"] == "dense"
    assert np.allclose(cur - ref, np.array([5.0, -3.0]), atol=1e-6)
    assert estimate is not None
    assert estimate.matrix[0, 2] == np.float64(-5.0)
    assert estimate.matrix[1, 2] == np.float64(3.0)
    assert status["backend"] == "raft-small"
    assert status["status"] in {"enabled", "not_enabled_pending_weights", "not_available"}


def test_feature_poor_black_frames_fail_closed_with_identity(tmp_path: Path) -> None:
    video = tmp_path / "black.avi"
    _write_video(video, [np.zeros((80, 120, 3), dtype=np.uint8) for _ in range(4)])
    calibration = _calibration(tmp_path / "court_calibration.json", width=120, height=80)

    payload = estimate_camera_motion(video, calibration)

    validate_camera_motion_payload(payload)
    assert payload["summary"]["n_compensated"] == 0
    for frame in payload["frames"]:
        assert frame["compensated"] is False
        assert frame["model"] == "identity"
        assert frame["reason"] == "too_few_reference_features"
        assert frame["M"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_person_mask_excludes_moving_blob_region_from_reference_features(tmp_path: Path) -> None:
    image = np.zeros((100, 140, 3), dtype=np.uint8)
    image[:] = (30, 110, 50)
    for x in range(44, 92, 6):
        for y in range(24, 72, 6):
            cv2.rectangle(image, (x, y), (x + 2, y + 2), (245, 245, 245), -1)
    for x in range(12, 128, 18):
        cv2.line(image, (x, 82), (x + 8, 90), (240, 240, 240), 1)
    calibration = json.loads(_calibration(tmp_path / "court_calibration.json", width=140, height=100).read_text())
    base_mask = build_court_mask(calibration, (100, 140), apron_px=0)
    person_tracks = {0: [[40.0, 20.0, 96.0, 76.0]]}

    masked = mask_people_for_frame(base_mask, person_tracks, frame_idx=0, padding_px=4)
    points = detect_reference_features(image, masked, CameraMotionParams(max_corners=80, quality_level=0.01))

    assert len(points) > 0
    for x, y in points.reshape(-1, 2):
        assert not (36.0 <= x <= 100.0 and 16.0 <= y <= 80.0)


def test_implausible_similarity_scale_is_rejected_by_transform_sanity() -> None:
    court_polygon = np.array([[10.0, 10.0], [170.0, 10.0], [170.0, 110.0], [10.0, 110.0]])
    collapsed = np.array([[0.25, 0.0, 350.0], [0.0, 0.25, 350.0], [0.0, 0.0, 1.0]])

    reason = transform_sanity_reason(collapsed, court_polygon, CameraMotionParams())

    assert reason == "implausible_area_scale"


def test_estimation_is_byte_deterministic(tmp_path: Path) -> None:
    video = tmp_path / "pan.avi"
    _write_video(video, _translated_frames([(0.0, 0.0), (3.0, 1.0), (6.0, 2.0)]))
    calibration = _calibration(tmp_path / "court_calibration.json")
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"

    payload_a = estimate_camera_motion(video, calibration)
    payload_b = estimate_camera_motion(video, calibration)
    write_camera_motion_json(payload_a, out_a)
    write_camera_motion_json(payload_b, out_b)

    assert out_a.read_bytes() == out_b.read_bytes()


def test_camera_motion_payload_round_trips_through_schema_validation(tmp_path: Path) -> None:
    video = tmp_path / "pan.avi"
    _write_video(video, _translated_frames([(0.0, 0.0), (5.0, -2.0)]))
    calibration = _calibration(tmp_path / "court_calibration.json")

    payload = estimate_camera_motion(video, calibration)
    validate_camera_motion_payload(payload)
    reloaded = json.loads(json.dumps(payload))
    validate_camera_motion_payload(reloaded)
    assert reloaded["artifact_type"] == "racketsport_camera_motion"
    assert reloaded["verified"] is False
    assert reloaded["not_gate_verified"] is True


def test_estimate_camera_motion_cli_help_is_registered() -> None:
    completed = subprocess.run(
        [sys.executable, CLI_PATH, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--calibration" in completed.stdout
    assert "--reference-frame" in completed.stdout
    assert "--estimator" in completed.stdout
    assert "--no-person-mask" in completed.stdout
    assert "--flow-backend" in completed.stdout


def _path_jerk_rms(path_xy: np.ndarray) -> float:
    second_diff = np.diff(np.asarray(path_xy, dtype=np.float64), n=2, axis=0)
    return float(np.sqrt(np.mean(np.sum(np.square(second_diff), axis=1))))
