from __future__ import annotations

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from threed.racketsport.overlapping_court_calibration import (
    HSVPaintRange,
    LineClusterConfig,
    apply_near_side_net_crop,
    clustered_hough_boundaries,
    detect_hsv_paint_hough_segments,
    fit_joint_camera_point_line_lm,
    fit_joint_distorted_camera_lm,
    fit_metric_plane_camera_lm,
    _mobilenet_v3_keypoint_checkpoint_evidence,
    hsv_paint_mask,
    image_points_to_world_plane_with_distortion_fit,
    load_labelme_court_keypoints,
    shadow_removal_preprocess,
    project_world_points_with_distortion_fit,
    refine_image_to_world_homography_lm,
)


YELLOW = HSVPaintRange(name="pickleball_yellow", lower=(24, 90, 90), upper=(40, 255, 255))


def _court_with_white_tennis_and_yellow_pickleball() -> np.ndarray:
    image = np.zeros((260, 360, 3), dtype=np.uint8)
    image[:, :] = (55, 115, 70)
    # White tennis lines that should be erased by the yellow HSV mask.
    cv2.line(image, (18, 40), (340, 40), (245, 245, 245), 5, cv2.LINE_AA)
    cv2.line(image, (18, 220), (340, 220), (245, 245, 245), 5, cv2.LINE_AA)
    cv2.line(image, (40, 20), (40, 240), (245, 245, 245), 5, cv2.LINE_AA)
    cv2.line(image, (320, 20), (320, 240), (245, 245, 245), 5, cv2.LINE_AA)
    # Yellow pickleball boundaries.
    for p1, p2 in [
        ((70, 80), (290, 80)),
        ((70, 210), (290, 210)),
        ((70, 80), (70, 210)),
        ((290, 80), (290, 210)),
    ]:
        cv2.line(image, p1, p2, (40, 230, 230), 4, cv2.LINE_AA)
    return image


def test_hsv_paint_mask_isolates_colored_pickleball_lines_from_white_tennis_lines() -> None:
    image = _court_with_white_tennis_and_yellow_pickleball()

    mask = hsv_paint_mask(image, ranges=[YELLOW])

    yellow_pixels = int(np.count_nonzero(mask[78:83, 70:290]))
    white_pixels = int(np.count_nonzero(mask[38:43, 18:340]))
    assert yellow_pixels > 700
    assert white_pixels < 20


def test_near_side_net_crop_removes_far_side_line_noise() -> None:
    mask = np.zeros((220, 320), dtype=np.uint8)
    cv2.line(mask, (40, 45), (280, 45), 255, 5)
    cv2.line(mask, (40, 180), (280, 180), 255, 5)
    cropped, evidence = apply_near_side_net_crop(
        mask,
        net_evidence={"top_tape_line": [[20, 105], [300, 105]]},
        preserve_margin_px=4,
    )

    assert evidence["applied"] is True
    assert int(np.count_nonzero(cropped[:95, :])) == 0
    assert int(np.count_nonzero(cropped[170:190, :])) > 100


def test_clustered_hough_boundaries_returns_four_colored_boundary_clusters() -> None:
    image = _court_with_white_tennis_and_yellow_pickleball()
    mask = hsv_paint_mask(image, ranges=[YELLOW])

    boundaries = clustered_hough_boundaries(
        mask,
        config=LineClusterConfig(min_line_length_px=60, hough_threshold=30),
    )

    assert len(boundaries.clusters) == 4
    orientations = sorted(cluster.orientation for cluster in boundaries.clusters)
    assert orientations.count("cross") == 2
    assert orientations.count("longitudinal") == 2
    assert boundaries.raw_segment_count >= 4


def test_detect_hsv_paint_hough_segments_reports_mask_and_group_evidence() -> None:
    image = _court_with_white_tennis_and_yellow_pickleball()

    evidence = detect_hsv_paint_hough_segments(image, ranges=[YELLOW])

    assert evidence["available"] is True
    assert evidence["technology_id"] == "opencv_hsv_paint_hough"
    assert evidence["candidate_count"] >= 4
    assert evidence["paint_mask"]["support_ratio"] > 0.005
    assert evidence["boundary_cluster_count"] == 4


def test_mobilenet_v3_checkpoint_evidence_uses_sibling_holdout_metrics_without_rescoring_all_rows(tmp_path) -> None:
    run_dir = (
        tmp_path
        / "runs"
        / "overlapping_court_calibration_20260703"
        / "mobilenet_v3_direct_regressor"
        / "mobilenet_trial"
    )
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "mobilenet_v3_court_keypoint_regressor.pt"
    checkpoint.write_bytes(b"placeholder; sibling metrics should be authoritative")
    metrics = {
        "artifact_type": "mobilenet_v3_court_keypoint_regressor_training_report",
        "status": "trained_not_cal3_verified",
        "verified": False,
        "not_cal3_verified": True,
        "diagnostic_only": True,
        "promotes_calibration": False,
        "checkpoint": str(checkpoint),
        "train_row_count": 24,
        "holdout_row_count": 8,
        "train_clip_names": ["clip_a", "clip_b", "clip_c"],
        "holdout_clip_names": ["clip_d"],
        "evaluation": {
            "artifact_type": "mobilenet_v3_court_keypoint_regressor_eval",
            "status": "scored",
            "diagnostic_only": True,
            "promotes_calibration": False,
            "gate_passed": False,
            "median_error_px": 12.5,
            "mean_error_px": 20.0,
            "p95_error_px": 44.0,
            "pck_at_5px": 0.1,
            "evaluated_keypoint_count": 120,
            "promotion_blockers": ["diagnostic_only", "not_cal3_verified", "gate_failed"],
        },
    }
    (run_dir / "mobilenet_v3_court_keypoint_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    evidence = _mobilenet_v3_keypoint_checkpoint_evidence(
        repo_root=tmp_path,
        eval_root=tmp_path / "missing_eval_root",
    )

    assert evidence["status"] == "scored"
    assert evidence["candidate_count"] == 1
    assert evidence["scored_candidate_count"] == 1
    assert evidence["best_candidate"]["source"] == "sibling_training_metrics"
    assert evidence["best_candidate"]["checkpoint"] == str(checkpoint)
    assert evidence["best_candidate"]["median_error_px"] == 12.5
    assert evidence["best_candidate"]["holdout_row_count"] == 8
    assert evidence["best_candidate"]["train_row_count"] == 24
    assert evidence["promotes_calibration"] is False


def test_labelme_loader_accepts_point_shapes_for_canonical_court_keypoints(tmp_path) -> None:
    labelme_path = tmp_path / "labelme.json"
    labelme_path.write_text(
        json.dumps(
            {
                "imagePath": "frame_000001.jpg",
                "imageWidth": 1280,
                "imageHeight": 720,
                "shapes": [
                    {"label": "near_left_corner", "shape_type": "point", "points": [[10.5, 20.25]]},
                    {"label": "near_right_corner", "shape_type": "point", "points": [[110.5, 20.25]]},
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_labelme_court_keypoints(labelme_path)

    assert loaded["image_size"] == [1280, 720]
    assert loaded["keypoints"]["near_left_corner"] == [10.5, 20.25]
    assert loaded["keypoints"]["near_right_corner"] == [110.5, 20.25]
    assert loaded["missing_keypoints"][0] == "near_baseline_center"


def test_lm_homography_refinement_reduces_world_residual_from_corner_seed() -> None:
    world_points = [
        [-10.0, -22.0],
        [10.0, -22.0],
        [10.0, 22.0],
        [-10.0, 22.0],
        [-10.0, -7.0],
        [10.0, -7.0],
        [-10.0, 7.0],
        [10.0, 7.0],
        [0.0, -22.0],
        [0.0, 22.0],
        [0.0, -7.0],
        [0.0, 7.0],
    ]
    true_h_world_to_image = np.array(
        [
            [22.0, 3.0, 520.0],
            [1.5, -13.0, 360.0],
            [0.002, -0.008, 1.0],
        ],
        dtype=float,
    )
    image_points = []
    for x, y in world_points:
        raw = true_h_world_to_image @ np.array([x, y, 1.0])
        image_points.append([float(raw[0] / raw[2]), float(raw[1] / raw[2])])
    # Add deterministic label-like noise to interior points only; the four corners are
    # the manual seed and the full-floor LM should absorb the review noise better.
    noisy_world_points = [list(point) for point in world_points]
    for index in range(4, len(noisy_world_points)):
        noisy_world_points[index][0] += 0.12 if index % 2 else -0.10
        noisy_world_points[index][1] += -0.08 if index % 3 else 0.09

    result = refine_image_to_world_homography_lm(
        image_points,
        noisy_world_points,
        seed_point_indexes=(0, 1, 2, 3),
    )

    assert result["method"] == "scipy_least_squares_lm"
    assert result["optimized_mean_residual_ft"] < result["initial_mean_residual_ft"]
    assert result["optimized_mean_residual_ft"] < 0.12
    assert result["target_mean_residual_ft"] == 0.2


def _look_at_pose(cam_pos: tuple[float, float, float], target: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    cam = np.asarray(cam_pos, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    forward = tgt - cam
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ cam
    return rotation, translation


def _project_camera(
    object_points_m: list[list[float]],
    *,
    rotation: np.ndarray,
    translation: np.ndarray,
    fx: float,
    cx: float,
    cy: float,
    dist: list[float] | None = None,
) -> list[list[float]]:
    rvec, _ = cv2.Rodrigues(rotation)
    k = np.asarray([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    projected, _ = cv2.projectPoints(
        np.asarray(object_points_m, dtype=np.float64),
        rvec,
        translation.reshape(3, 1),
        k,
        None if dist is None else np.asarray(dist, dtype=np.float64),
    )
    return projected.reshape(-1, 2).tolist()


COURT_OBJECT_POINTS_M = [
    [-3.048, -6.7056, 0.0],
    [3.048, -6.7056, 0.0],
    [3.048, 6.7056, 0.0],
    [-3.048, 6.7056, 0.0],
    [-3.048, -2.1336, 0.0],
    [3.048, -2.1336, 0.0],
    [-3.048, 2.1336, 0.0],
    [3.048, 2.1336, 0.0],
    [0.0, -6.7056, 0.0],
    [0.0, 6.7056, 0.0],
    [0.0, -2.1336, 0.0],
    [0.0, 2.1336, 0.0],
]


def test_joint_distorted_camera_lm_recovers_radial_distortion_and_reduces_reprojection_error() -> None:
    image_size = (1920, 1080)
    rotation, translation = _look_at_pose((0.0, -10.5, 4.6), (0.0, 0.0, 0.0))
    true_points = _project_camera(
        COURT_OBJECT_POINTS_M,
        rotation=rotation,
        translation=translation,
        fx=1420.0,
        cx=image_size[0] / 2.0,
        cy=image_size[1] / 2.0,
        dist=[-0.18, 0.055, 0.0, 0.0],
    )

    fit = fit_joint_distorted_camera_lm(COURT_OBJECT_POINTS_M, true_points, image_size=image_size)

    assert fit["method"] == "joint_focal_pose_radial_lm"
    assert fit["optimized_reprojection_rmse_px"] < 0.08
    assert fit["optimized_reprojection_rmse_px"] < fit["initial_reprojection_rmse_px"]
    assert fit["intrinsics"]["fx"] == pytest.approx(1420.0, rel=0.02)
    assert fit["distortion"]["k1"] == pytest.approx(-0.18, abs=0.03)
    assert fit["distortion"]["k2"] == pytest.approx(0.055, abs=0.04)


def test_point_line_lm_uses_line_evidence_to_reduce_clean_geometry_error() -> None:
    image_size = (1920, 1080)
    rotation, translation = _look_at_pose((0.0, -10.5, 4.6), (0.0, 0.0, 0.0))
    clean_points = _project_camera(
        COURT_OBJECT_POINTS_M,
        rotation=rotation,
        translation=translation,
        fx=1420.0,
        cx=image_size[0] / 2.0,
        cy=image_size[1] / 2.0,
        dist=[-0.12, 0.025, 0.0, 0.0],
    )
    noisy_points = [[x + 9.0, y - 6.0] if idx in {4, 5, 8, 10} else [x, y] for idx, (x, y) in enumerate(clean_points)]
    line_observations = [
        {
            "world_line_m": [COURT_OBJECT_POINTS_M[0], COURT_OBJECT_POINTS_M[1]],
            "image_segment_px": [clean_points[0], clean_points[1]],
            "sampled_image_points_px": [
                clean_points[0],
                [(clean_points[0][0] + clean_points[1][0]) / 2.0, (clean_points[0][1] + clean_points[1][1]) / 2.0],
                clean_points[1],
            ],
            "name": "near_baseline",
        },
        {
            "world_line_m": [COURT_OBJECT_POINTS_M[2], COURT_OBJECT_POINTS_M[3]],
            "image_segment_px": [clean_points[2], clean_points[3]],
            "sampled_image_points_px": [
                clean_points[2],
                [(clean_points[2][0] + clean_points[3][0]) / 2.0, (clean_points[2][1] + clean_points[3][1]) / 2.0],
                clean_points[3],
            ],
            "name": "far_baseline",
        },
        {
            "world_line_m": [COURT_OBJECT_POINTS_M[0], COURT_OBJECT_POINTS_M[3]],
            "image_segment_px": [clean_points[0], clean_points[3]],
            "sampled_image_points_px": [
                clean_points[0],
                [(clean_points[0][0] + clean_points[3][0]) / 2.0, (clean_points[0][1] + clean_points[3][1]) / 2.0],
                clean_points[3],
            ],
            "name": "left_sideline",
        },
        {
            "world_line_m": [COURT_OBJECT_POINTS_M[1], COURT_OBJECT_POINTS_M[2]],
            "image_segment_px": [clean_points[1], clean_points[2]],
            "sampled_image_points_px": [
                clean_points[1],
                [(clean_points[1][0] + clean_points[2][0]) / 2.0, (clean_points[1][1] + clean_points[2][1]) / 2.0],
                clean_points[2],
            ],
            "name": "right_sideline",
        },
    ]

    point_only = fit_joint_distorted_camera_lm(COURT_OBJECT_POINTS_M, noisy_points, image_size=image_size)
    point_line = fit_joint_camera_point_line_lm(
        COURT_OBJECT_POINTS_M,
        noisy_points,
        image_size=image_size,
        line_observations=line_observations,
        line_weight=0.8,
    )
    point_only_projection = project_world_points_with_distortion_fit(COURT_OBJECT_POINTS_M, point_only)
    point_line_projection = project_world_points_with_distortion_fit(COURT_OBJECT_POINTS_M, point_line)

    point_only_clean_rmse = np.sqrt(np.mean(np.sum((np.asarray(point_only_projection) - np.asarray(clean_points)) ** 2, axis=1)))
    point_line_clean_rmse = np.sqrt(np.mean(np.sum((np.asarray(point_line_projection) - np.asarray(clean_points)) ** 2, axis=1)))

    assert point_line["method"] == "joint_point_line_focal_pose_radial_lm"
    assert point_line["line_residual_mode"] == "sampled_line_pixels_to_projected_model_line"
    assert point_line["line_observation_count"] == 4
    assert point_line["line_pixel_sample_count"] == 12
    assert point_line["line_pixel_samples_per_observation"] == 3
    assert point_line_clean_rmse < point_only_clean_rmse


def test_camera_fit_backprojects_image_points_to_world_court_plane() -> None:
    image_size = (1920, 1080)
    rotation, translation = _look_at_pose((0.0, -10.5, 4.6), (0.0, 0.0, 0.0))
    image_points = _project_camera(
        COURT_OBJECT_POINTS_M,
        rotation=rotation,
        translation=translation,
        fx=1420.0,
        cx=image_size[0] / 2.0,
        cy=image_size[1] / 2.0,
        dist=[-0.12, 0.025, 0.0, 0.0],
    )
    fit = fit_joint_distorted_camera_lm(COURT_OBJECT_POINTS_M, image_points, image_size=image_size)

    backprojected = image_points_to_world_plane_with_distortion_fit(image_points, fit)
    mean_error_m = float(
        np.mean(
            np.linalg.norm(
                np.asarray(backprojected, dtype=np.float64)[:, :2]
                - np.asarray(COURT_OBJECT_POINTS_M, dtype=np.float64)[:, :2],
                axis=1,
            )
        )
    )

    assert mean_error_m < 0.02
    assert max(abs(point[2]) for point in backprojected) < 1e-6


def test_metric_plane_camera_lm_reduces_world_residual_for_noisy_floor_points() -> None:
    image_size = (1920, 1080)
    rotation, translation = _look_at_pose((0.0, -10.5, 4.6), (0.0, 0.0, 0.0))
    clean_points = _project_camera(
        COURT_OBJECT_POINTS_M,
        rotation=rotation,
        translation=translation,
        fx=1420.0,
        cx=image_size[0] / 2.0,
        cy=image_size[1] / 2.0,
        dist=[-0.12, 0.025, 0.0, 0.0],
    )
    noisy_points = [[x, y] for x, y in clean_points]
    for index, (dx, dy) in {
        4: (11.0, -8.0),
        5: (8.0, 5.0),
        8: (-9.0, 7.0),
        10: (13.0, -6.0),
    }.items():
        noisy_points[index][0] += dx
        noisy_points[index][1] += dy

    pixel_fit = fit_joint_distorted_camera_lm(COURT_OBJECT_POINTS_M, noisy_points, image_size=image_size)
    metric_fit = fit_metric_plane_camera_lm(COURT_OBJECT_POINTS_M, noisy_points, image_size=image_size)
    pixel_backprojected = image_points_to_world_plane_with_distortion_fit(noisy_points, pixel_fit)
    metric_backprojected = image_points_to_world_plane_with_distortion_fit(noisy_points, metric_fit)

    pixel_world_rmse = np.sqrt(
        np.mean(
            np.sum(
                (np.asarray(pixel_backprojected)[:, :2] - np.asarray(COURT_OBJECT_POINTS_M)[:, :2]) ** 2,
                axis=1,
            )
        )
    )
    metric_world_rmse = np.sqrt(
        np.mean(
            np.sum(
                (np.asarray(metric_backprojected)[:, :2] - np.asarray(COURT_OBJECT_POINTS_M)[:, :2]) ** 2,
                axis=1,
            )
        )
    )

    assert metric_fit["method"] == "metric_plane_focal_pose_radial_soft_l1_lm"
    assert metric_fit["objective"] == "world_plane_backprojection_m"
    assert metric_world_rmse < pixel_world_rmse


def test_shadow_removal_preprocess_recovers_shadowed_white_line_support() -> None:
    image = np.zeros((180, 300, 3), dtype=np.uint8)
    image[:, :] = (70, 130, 70)
    cv2.line(image, (35, 95), (265, 95), (245, 245, 245), 4, cv2.LINE_AA)
    image[:, 120:190] = (image[:, 120:190].astype(np.float32) * 0.35).astype(np.uint8)

    before_mask = (image.min(axis=2) > 210).astype(np.uint8) * 255
    processed, evidence = shadow_removal_preprocess(image)
    after_mask = (processed.min(axis=2) > 210).astype(np.uint8) * 255

    assert evidence["available"] is True
    assert evidence["mode"] == "lab_luminance_shadow_compensation"
    assert int(np.count_nonzero(after_mask[92:99, 120:190])) > int(np.count_nonzero(before_mask[92:99, 120:190])) + 100
