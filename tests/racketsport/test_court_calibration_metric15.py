from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from threed.racketsport.court_calibration_metric15 import (
    METRIC15_SOURCE_TAG,
    ReviewedCourtKeypoints,
    aggregate_reviewed_keypoints_native_px,
    fit_single_view_metric_camera,
    load_reviewed_court_keypoints_15pt,
    metric_calibration_from_reviewed_keypoints_15pt,
)
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS
from threed.racketsport.schemas import PICKLEBALL_COURT_KEYPOINT_NAMES, CourtCalibration, validate_artifact_file

OBJECT_POINTS = [list(point.world_xyz_m) for point in PICKLEBALL_KEYPOINTS]
NATIVE_SIZE = (1920.0, 1080.0)


def _look_at_pose(cam_pos: tuple[float, float, float], target: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    cam = np.asarray(cam_pos, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    world_up = np.asarray([0.0, 0.0, 1.0])
    forward = tgt - cam
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ cam
    return rotation, translation


def _project(object_points, rotation, translation, fx, cx, cy, dist=None):
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = translation.reshape(3, 1)
    k = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_arr = None if dist is None else np.asarray(dist, dtype=np.float64)
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float64), rvec, tvec, k, dist_arr
    )
    return projected.reshape(-1, 2).tolist()


# A realistic behind-baseline broadcast-style camera pose: elevated, looking down
# the court, court fills a healthy fraction of the frame (the failure mode this test
# guards against is a degenerate/edge-of-frame synthetic pose that makes single-view
# calibration artificially well- or ill-conditioned).
CAM_POS = (0.0, -13.0, 5.5)
TARGET = (0.0, 0.0, 0.0)
TRUE_FX = 1450.0
TRUE_CX, TRUE_CY = NATIVE_SIZE[0] / 2.0, NATIVE_SIZE[1] / 2.0


def test_synthetic_round_trip_zero_distortion_recovers_focal_and_pose():
    rotation, translation = _look_at_pose(CAM_POS, TARGET)
    image_points = _project(OBJECT_POINTS, rotation, translation, TRUE_FX, TRUE_CX, TRUE_CY)

    fit = fit_single_view_metric_camera(OBJECT_POINTS, image_points, NATIVE_SIZE)

    assert fit.fx == pytest.approx(TRUE_FX, rel=1e-3)
    assert fit.fy == pytest.approx(TRUE_FX, rel=1e-3)
    assert fit.cx == pytest.approx(TRUE_CX)
    assert fit.cy == pytest.approx(TRUE_CY)
    assert fit.reprojection_error_px.median < 0.05
    assert fit.reprojection_error_px.p95 < 0.1
    assert len(fit.per_point_residual_px) == 15
    assert any("principal point fixed" in note for note in fit.identifiability_notes)


def test_synthetic_round_trip_recovers_radial_distortion_when_present():
    rotation, translation = _look_at_pose(CAM_POS, TARGET)
    true_dist = [-0.15, 0.04, 0.0, 0.0]
    image_points = _project(OBJECT_POINTS, rotation, translation, TRUE_FX, TRUE_CX, TRUE_CY, dist=true_dist)

    fit = fit_single_view_metric_camera(OBJECT_POINTS, image_points, NATIVE_SIZE)

    assert fit.distortion_model == "k1_k2_calibrate_camera_seeded"
    assert fit.fx == pytest.approx(TRUE_FX, rel=1e-3)
    assert fit.k1 == pytest.approx(true_dist[0], abs=5e-3)
    assert fit.k2 == pytest.approx(true_dist[1], abs=5e-3)
    assert fit.reprojection_error_px.median < 0.05
    assert any("accepted" in note for note in fit.identifiability_notes)


def test_zero_distortion_input_does_not_spuriously_accept_distortion():
    rotation, translation = _look_at_pose(CAM_POS, TARGET)
    image_points = _project(OBJECT_POINTS, rotation, translation, TRUE_FX, TRUE_CX, TRUE_CY)

    fit = fit_single_view_metric_camera(OBJECT_POINTS, image_points, NATIVE_SIZE)

    # With no real distortion in the data, k1/k2 should stay at (or converge to)
    # ~zero rather than fitting noise -- whichever distortion_model wins.
    assert abs(fit.k1) < 1e-3
    assert abs(fit.k2) < 1e-3


def test_fit_requires_minimum_correspondences():
    with pytest.raises(ValueError, match="at least"):
        fit_single_view_metric_camera(OBJECT_POINTS[:4], [[0.0, 0.0]] * 4, NATIVE_SIZE)


def _reviewed_payload(*, label_space=(1280, 720), source_res=(1920, 1080), frame_count=8, omit_size=False):
    rotation, translation = _look_at_pose(CAM_POS, TARGET)
    native_points = _project(OBJECT_POINTS, rotation, translation, TRUE_FX, TRUE_CX, TRUE_CY)
    scale_x = label_space[0] / source_res[0]
    scale_y = label_space[1] / source_res[1]
    label_points = [[x * scale_x, y * scale_y] for x, y in native_points]

    items = []
    for idx in range(frame_count):
        keypoints = {
            name: label_points[i] for i, name in enumerate(PICKLEBALL_COURT_KEYPOINT_NAMES)
        }
        items.append(
            {
                "frame": f"frame_{idx + 1:06d}.jpg",
                "keypoints": keypoints,
                "review_id": f"court_keypoints_manual_15pt_{idx:04d}",
                "status": "reviewed" if idx == 0 else "reviewed_static_camera_copy",
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_court_keypoint_labels",
        "clip": "synthetic_test_clip",
        "annotation": {"items": items},
        "review": {"status": "reviewed"},
        "frames": {
            "frame_count": frame_count,
            "frame_dir": "does/not/matter",
            "sample_every_frames": 30,
            "source_resolution": list(source_res),
        },
    }
    if not omit_size:
        payload["frames"]["label_coordinate_space"] = list(label_space)
    return payload, native_points


def test_loader_rejects_missing_declared_label_coordinate_space(tmp_path):
    payload, _ = _reviewed_payload(omit_size=True)
    path = tmp_path / "court_keypoints.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="label_coordinate_space"):
        load_reviewed_court_keypoints_15pt(path)


def test_aggregate_rescales_from_declared_label_space_to_native():
    payload, native_points = _reviewed_payload(label_space=(1280, 720), source_res=(1920, 1080))

    # Build frames the way the loader would (bypassing file IO here).
    from threed.racketsport.court_calibration_metric15 import ReviewedKeypointFrame

    frames = []
    for item in payload["annotation"]["items"]:
        frames.append(
            ReviewedKeypointFrame(
                frame=item["frame"],
                status=item["status"],
                keypoints={name: tuple(item["keypoints"][name]) for name in PICKLEBALL_COURT_KEYPOINT_NAMES},
            )
        )
    reviewed = ReviewedCourtKeypoints(
        clip=payload["clip"],
        label_coordinate_space=(1280.0, 720.0),
        source_resolution=(1920.0, 1080.0),
        frames=frames,
    )

    aggregated, stdev, native_size = aggregate_reviewed_keypoints_native_px(reviewed)

    assert native_size == (1920.0, 1080.0)
    for idx, name in enumerate(PICKLEBALL_COURT_KEYPOINT_NAMES):
        assert aggregated[name] == pytest.approx(native_points[idx], abs=1e-6)
        assert stdev[name]["x_stdev_px"] == pytest.approx(0.0, abs=1e-6)


def test_metric_calibration_from_reviewed_keypoints_end_to_end(tmp_path):
    payload, _ = _reviewed_payload()
    path = tmp_path / "court_keypoints.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    calibration = metric_calibration_from_reviewed_keypoints_15pt(path, sport="pickleball")

    assert isinstance(calibration, CourtCalibration)
    assert calibration.intrinsics.source == METRIC15_SOURCE_TAG
    assert calibration.intrinsics.fx == pytest.approx(TRUE_FX, rel=1e-2)
    assert calibration.image_size == (1920, 1080)
    assert calibration.coordinate_frame == "court_netcenter_z_up_m"
    assert calibration.per_keypoint_residual_px is not None
    assert len(calibration.per_keypoint_residual_px) == 15
    assert max(calibration.per_keypoint_residual_px) < 1.0
    assert calibration.reprojection_error_px.median < 0.5
    assert calibration.metric_confidence == "high"
    # frame_000001 (1-based) -> native frame 0, frame_000002 -> native frame 30, ...
    assert calibration.solved_over_frames == [idx * 30 for idx in range(8)]
    assert calibration.source == METRIC15_SOURCE_TAG
    assert "single_view_planar_full_calibration" in calibration.capture_quality.reasons


def test_metric_calibration_rejects_non_pickleball_sport(tmp_path):
    payload, _ = _reviewed_payload()
    path = tmp_path / "court_keypoints.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="pickleball only"):
        metric_calibration_from_reviewed_keypoints_15pt(path, sport="tennis")


def test_loader_skips_incomplete_frames(tmp_path):
    payload, _ = _reviewed_payload(frame_count=2)
    # Drop one keypoint from the second frame -- it should be skipped, not crash.
    del payload["annotation"]["items"][1]["keypoints"][PICKLEBALL_COURT_KEYPOINT_NAMES[0]]
    path = tmp_path / "court_keypoints.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    reviewed = load_reviewed_court_keypoints_15pt(path)

    assert len(reviewed.frames) == 1
    assert reviewed.frames[0].frame == "frame_000001.jpg"


def test_loader_raises_when_no_frame_has_all_keypoints(tmp_path):
    payload, _ = _reviewed_payload(frame_count=1)
    del payload["annotation"]["items"][0]["keypoints"][PICKLEBALL_COURT_KEYPOINT_NAMES[0]]
    path = tmp_path / "court_keypoints.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="no reviewed frame"):
        load_reviewed_court_keypoints_15pt(path)


def test_calibrate_cli_writes_metric15_calibration_from_reviewed_keypoints(tmp_path):
    payload, _ = _reviewed_payload()
    keypoints_path = tmp_path / "court_keypoints.json"
    keypoints_path.write_text(json.dumps(payload), encoding="utf-8")
    out_dir = tmp_path / "calib"

    subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/calibrate.py",
            "--reviewed-court-keypoints",
            str(keypoints_path),
            "--sport",
            "pickleball",
            "--out",
            str(out_dir),
        ],
        check=True,
    )

    calibration = validate_artifact_file("court_calibration", out_dir / "court_calibration.json")

    assert isinstance(calibration, CourtCalibration)
    assert calibration.source == METRIC15_SOURCE_TAG
    assert calibration.intrinsics.source == METRIC15_SOURCE_TAG
    assert (out_dir / "court_zones.json").is_file()
    assert (out_dir / "net_plane.json").is_file()


def test_calibrate_cli_rejects_reviewed_keypoints_combined_with_sidecar(tmp_path):
    payload, _ = _reviewed_payload()
    keypoints_path = tmp_path / "court_keypoints.json"
    keypoints_path.write_text(json.dumps(payload), encoding="utf-8")
    sidecar_path = tmp_path / "capture_sidecar.json"
    sidecar_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    out_dir = tmp_path / "calib"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/calibrate.py",
            "--sidecar",
            str(sidecar_path),
            "--reviewed-court-keypoints",
            str(keypoints_path),
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "standalone" in result.stderr
