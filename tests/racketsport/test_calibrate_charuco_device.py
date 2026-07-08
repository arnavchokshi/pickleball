from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from threed.racketsport.profile_registry import load_profile_registry


def test_calibrate_charuco_device_recovers_synthetic_barrel_distortion_and_persists_profile(tmp_path: Path) -> None:
    pytest.importorskip("cv2.aruco")
    videos, expected_dist = _write_synthetic_charuco_videos(tmp_path)
    profiles_root = tmp_path / "profiles"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/calibrate_charuco_device.py",
            "--account-id",
            "owner_1",
            "--device-key",
            "iphone16-owner",
            "--profile-id",
            "iphone16_owner",
            "--display-name",
            "iPhone 16 Owner",
            "--lens",
            "wide",
            "--zoom",
            "1.0",
            "--profiles-root",
            str(profiles_root),
            "--source-clip-id",
            "synthetic_charuco_wide",
            "--rms-threshold",
            "1.0",
            "--spread-threshold",
            "0.20",
            *[item for video in videos for item in ("--video", str(video))],
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    summary = json.loads(result.stdout)
    recovered = summary["intrinsics"]["dist"]
    assert summary["status"] == "persisted"
    assert summary["rms_reprojection_px"] <= 1.0
    assert summary["spread"]["k1_relative_spread"] < 0.20
    assert summary["spread"]["k2_relative_spread"] < 0.20
    assert recovered[0] < 0.0
    assert recovered[0] == pytest.approx(expected_dist[0], abs=0.04)
    assert recovered[1] == pytest.approx(expected_dist[1], abs=0.04)

    registry = load_profile_registry("owner_1", profiles_root=profiles_root)
    profile = registry.device_profiles["iphone16_owner"]
    entry = profile.intrinsics_by_lens_zoom[0]
    assert entry.lens == "wide"
    assert entry.zoom == pytest.approx(1.0)
    assert entry.intrinsics.dist[:2] == pytest.approx(recovered[:2])
    assert entry.source_trace.source_clip_id == "synthetic_charuco_wide"


def _write_synthetic_charuco_videos(tmp_path: Path) -> tuple[list[Path], list[float]]:
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard((5, 7), 0.04, 0.03, dictionary)
    width, height = 1600, 1200
    camera_matrix = np.asarray([[1100.0, 0.0, width / 2.0], [0.0, 1100.0, height / 2.0], [0.0, 0.0, 1.0]])
    dist = np.asarray([-0.18, 0.08, 0.0005, -0.0003, 0.0], dtype=np.float64)
    board_image = board.generateImage((600, 840), marginSize=0)
    right_bottom = board.getRightBottomCorner()
    board_corners = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [right_bottom[0], 0.0, 0.0],
            [right_bottom[0], right_bottom[1], 0.0],
            [0.0, right_bottom[1], 0.0],
        ],
        dtype=np.float32,
    )
    source_corners = np.asarray([[0.0, 0.0], [599.0, 0.0], [599.0, 839.0], [0.0, 839.0]], dtype=np.float32)
    distort_map = _distort_map(width, height, camera_matrix, dist)

    videos: list[Path] = []
    for distance_m, spread in ((0.55, 0.08), (0.75, 0.12), (1.0, 0.17)):
        path = tmp_path / f"charuco_distance_{distance_m:.2f}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (width, height), isColor=True)
        assert writer.isOpened()
        frame_count = 0
        for ix, tx in enumerate((-spread, 0.0, spread)):
            for iy, ty in enumerate((-spread * 0.6, spread * 0.6)):
                for tilt_deg in (-22.0, 0.0, 22.0):
                    frame = _charuco_frame(
                        width=width,
                        height=height,
                        camera_matrix=camera_matrix,
                        board_image=board_image,
                        board_corners=board_corners,
                        source_corners=source_corners,
                        distance_m=distance_m,
                        tx=tx,
                        ty=ty,
                        rx=math.radians(tilt_deg),
                        ry=math.radians((ix - 1) * 18.0),
                        rz=math.radians((iy - 0.5) * 16.0),
                        distort_map=distort_map,
                    )
                    if frame is None:
                        continue
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
                    frame_count += 1
        writer.release()
        assert frame_count >= 8
        videos.append(path)
    return videos, dist[:4].tolist()


def _charuco_frame(
    *,
    width: int,
    height: int,
    camera_matrix: np.ndarray,
    board_image: np.ndarray,
    board_corners: np.ndarray,
    source_corners: np.ndarray,
    distance_m: float,
    tx: float,
    ty: float,
    rx: float,
    ry: float,
    rz: float,
    distort_map: tuple[np.ndarray, np.ndarray],
) -> np.ndarray | None:
    rvec, _ = cv2.Rodrigues(_rotation_matrix(rx, ry, rz))
    tvec = np.asarray([[tx], [ty], [distance_m]], dtype=np.float64)
    projected, _ = cv2.projectPoints(board_corners, rvec, tvec, camera_matrix, None)
    dest_corners = projected.reshape(-1, 2).astype(np.float32)
    if (
        dest_corners[:, 0].min() < 20.0
        or dest_corners[:, 0].max() > width - 20.0
        or dest_corners[:, 1].min() < 20.0
        or dest_corners[:, 1].max() > height - 20.0
    ):
        return None
    homography = cv2.getPerspectiveTransform(source_corners, dest_corners)
    canvas = np.full((height, width), 255, dtype=np.uint8)
    warped = cv2.warpPerspective(board_image, homography, (width, height), borderValue=255)
    canvas[warped < 250] = warped[warped < 250]
    return cv2.remap(canvas, distort_map[0], distort_map[1], cv2.INTER_LINEAR, borderValue=255)


def _distort_map(width: int, height: int, camera_matrix: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    distorted_pixels = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2).astype(np.float32)
    undistorted = cv2.undistortPoints(distorted_pixels, camera_matrix, dist, P=camera_matrix).reshape(height, width, 2)
    return undistorted[..., 0].astype(np.float32), undistorted[..., 1].astype(np.float32)


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    rot_x = np.asarray([[1.0, 0.0, 0.0], [0.0, math.cos(rx), -math.sin(rx)], [0.0, math.sin(rx), math.cos(rx)]])
    rot_y = np.asarray([[math.cos(ry), 0.0, math.sin(ry)], [0.0, 1.0, 0.0], [-math.sin(ry), 0.0, math.cos(ry)]])
    rot_z = np.asarray([[math.cos(rz), -math.sin(rz), 0.0], [math.sin(rz), math.cos(rz), 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x
