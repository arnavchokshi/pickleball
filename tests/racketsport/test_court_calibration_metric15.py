from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

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

    assert fit.distortion_model == "k1_k2_radial_bounded_cv_selected"
    assert fit.fx == pytest.approx(TRUE_FX, rel=1e-3)
    assert fit.k1 == pytest.approx(true_dist[0], abs=5e-3)
    assert fit.k2 == pytest.approx(true_dist[1], abs=5e-3)
    assert fit.reprojection_error_px.median < 0.05
    assert any("accepted" in note for note in fit.identifiability_notes)
    # Accepted on held-out evidence, not on the residual of the fitted points.
    assert fit.held_out_median_px is not None
    assert fit.held_out_median_px < 0.5
    assert any("leave_one_out" in note for note in fit.identifiability_notes)
    scored = {record["distortion_model"] for record in fit.model_selection or []}
    assert {"zero_distortion_grid_search_focal", "k1_radial_bounded_cv_selected", "k1_k2_radial_bounded_cv_selected"} <= scored


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


def test_real_burlington_fixture_preserves_legacy_numeric_payload_and_adds_typed_contract() -> None:
    repo = Path(__file__).resolve().parents[2]
    fixture = (
        repo
        / "eval_clips/ball/burlington_gold_0300_low_steep_corner/labels/court_keypoints.json"
    )
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == (
        "537dbe0299a7069f258d373ada99758ff9238e6718d6bc6481365b614daa030f"
    )

    calibration = metric_calibration_from_reviewed_keypoints_15pt(fixture)
    payload = calibration.model_dump(mode="json")
    contract = payload.pop("coordinate_contract")

    # Digest refreshed 2026-07-26 (CAL lane). The previous pin,
    # 08f6c8ce1151bb0c654d3895910898cb51c694b61bddbaa59c574bf3754ee3a9, was
    # already failing at f29145a: it was frozen when the 3 net keypoints were
    # declared at z=0, and a later court-taxonomy change raised them to the net
    # top (post_net_height_m) without re-freezing, silently degrading this fit
    # from 6.39px to 26.4px median. The metric fit now selects the net label
    # height on held-out residual and this clip's reviewed labels select the
    # ground net line, so the fixture is meaningful again.
    legacy_digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert legacy_digest == "00945af38d16553ef7ef24f21114f30c46027def3e105a69a70804aae30e95c1"
    assert payload["world_pts"][9:12] == [[-3.048, 0.0, 0.0], [0.0, 0.0, 0.0], [3.048, 0.0, 0.0]]
    assert payload["intrinsics"]["dist"][0] == pytest.approx(-0.1789, abs=1e-3)
    assert contract == {
        "camera_matrix_K": [
            [payload["intrinsics"]["fx"], 0.0, payload["intrinsics"]["cx"]],
            [0.0, payload["intrinsics"]["fy"], payload["intrinsics"]["cy"]],
            [0.0, 0.0, 1.0],
        ],
        "camera_matrix_input_space": "camera_m",
        "camera_matrix_output_space": "pixels_undistorted_native",
        "extrinsics_convention": "world_to_camera_opencv_column",
        "extrinsics_input_space": "world_court_netcenter_z_up_m",
        "extrinsics_output_space": "camera_m",
        "homography_convention": "world_xy_to_image_column",
        "homography_input_space": "world_xy_homography_m",
        "homography_output_space": "pixels_raw_native",
        "homography_pixel_convention": "raw_pixels",
    }

    conflicting = calibration.model_dump(mode="json")
    conflicting["coordinate_contract"]["camera_matrix_K"][0][0] += 1.0
    with pytest.raises(ValueError, match="camera_matrix_K conflicts"):
        CourtCalibration.model_validate(conflicting)


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


# --- Held-out model selection (CAL lane, 2026-07-26) -------------------------------

NET_INDEXES = [
    index
    for index, name in enumerate(PICKLEBALL_COURT_KEYPOINT_NAMES)
    if name in ("net_left_sideline", "net_center", "net_right_sideline")
]


def _object_points_with_net_at(height_m: float) -> list[list[float]]:
    points = [list(point) for point in OBJECT_POINTS]
    for index in NET_INDEXES:
        points[index][2] = height_m
    return points


def _net_variants() -> dict[str, list[list[float]]]:
    return {
        "net_top_as_declared": _object_points_with_net_at(0.9144),
        "ground_net_line": _object_points_with_net_at(0.0),
    }


@pytest.mark.parametrize(
    ("truth_height_m", "expected_variant"),
    [(0.0, "ground_net_line"), (0.9144, "net_top_as_declared")],
)
def test_net_label_height_is_selected_from_held_out_residual(truth_height_m, expected_variant):
    """The pixels decide which world height the label source meant -- both ways.

    This is the defect that made the reviewed 15-pt solve worse than a line solve
    on the same video: the taxonomy declares the 3 net keypoints at the net top,
    every reviewed label set in this repo marks the ground net line, and a 0.9m
    world-model error on 3 of 15 points is a 40-120px systematic residual that no
    distortion coefficient can absorb.
    """

    rotation, translation = _look_at_pose(CAM_POS, TARGET)
    truth = _object_points_with_net_at(truth_height_m)
    image_points = _project(truth, rotation, translation, TRUE_FX, TRUE_CX, TRUE_CY)

    fit = fit_single_view_metric_camera(
        _object_points_with_net_at(0.9144),
        image_points,
        NATIVE_SIZE,
        object_point_variants=_net_variants(),
    )

    assert fit.object_point_variant == expected_variant
    assert fit.object_points_m is not None
    assert fit.object_points_m[NET_INDEXES[0]][2] == pytest.approx(truth_height_m)
    assert fit.fx == pytest.approx(TRUE_FX, rel=1e-3)
    assert fit.reprojection_error_px.median < 0.05
    variant_scores = {
        record["object_point_variant"]: record["held_out_median_px"]
        for record in fit.model_selection or []
        if record.get("role") == "object_point_variant_candidate"
    }
    assert set(variant_scores) == {"net_top_as_declared", "ground_net_line"}
    losers = [score for name, score in variant_scores.items() if name != expected_variant]
    assert min(losers) > 10.0 * max(1e-6, variant_scores[expected_variant])


def test_wrong_net_height_makes_the_fit_much_worse():
    """Quantifies the defect: same pixels, only the declared net height differs."""

    rotation, translation = _look_at_pose(CAM_POS, TARGET)
    truth = _object_points_with_net_at(0.0)
    image_points = _project(truth, rotation, translation, TRUE_FX, TRUE_CX, TRUE_CY)

    right = fit_single_view_metric_camera(truth, image_points, NATIVE_SIZE)
    wrong = fit_single_view_metric_camera(
        _object_points_with_net_at(0.9144), image_points, NATIVE_SIZE
    )

    assert right.reprojection_error_px.median < 0.05
    assert wrong.reprojection_error_px.median > 5.0
    # And the wrong world model cannot be rescued by distortion.
    assert wrong.k1 == 0.0 or abs(wrong.reprojection_error_px.median) > 5.0


def test_accepted_radial_model_is_always_invertible_over_the_frame():
    """Unconstrained cv2.calibrateCamera returns folded radial maps from 15 points."""

    from threed.racketsport.camera_distortion import is_radially_invertible, max_normalized_radius

    rotation, translation = _look_at_pose(CAM_POS, TARGET)
    for true_dist in ([-0.15, 0.04, 0.0, 0.0], [-0.32, 0.11, 0.0, 0.0], None):
        image_points = _project(
            OBJECT_POINTS, rotation, translation, TRUE_FX, TRUE_CX, TRUE_CY, dist=true_dist
        )
        fit = fit_single_view_metric_camera(OBJECT_POINTS, image_points, NATIVE_SIZE)
        max_radius = max_normalized_radius(NATIVE_SIZE, fit.fx, fit.fy, fit.cx, fit.cy)
        assert is_radially_invertible(fit.k1, fit.k2, max_radius)


def test_selection_is_recorded_for_audit_not_asserted():
    rotation, translation = _look_at_pose(CAM_POS, TARGET)
    image_points = _project(OBJECT_POINTS, rotation, translation, TRUE_FX, TRUE_CX, TRUE_CY)

    fit = fit_single_view_metric_camera(OBJECT_POINTS, image_points, NATIVE_SIZE)

    assert fit.model_selection
    accepted = [record for record in fit.model_selection if record.get("accepted")]
    assert len(accepted) == 1
    assert accepted[0]["distortion_model"] == fit.distortion_model
    assert fit.held_out_median_px == pytest.approx(accepted[0]["held_out_median_px"])
