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
