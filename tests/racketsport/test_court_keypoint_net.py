from __future__ import annotations

import pytest

from threed.racketsport.court_templates import FT_TO_M
from threed.racketsport.court_keypoint_net import (
    PICKLEBALL_KEYPOINTS,
    decode_subpixel_heatmap,
    keypoint_labels_from_court_corners,
    keypoints_to_solvepnp_correspondences,
    refine_keypoint_xy_with_planar_homography,
    validate_heatmap_prediction_payload,
    validate_synthetic_render_config,
    validate_training_plan_config,
)


def _ft(value_m: float) -> float:
    return value_m / FT_TO_M


def test_pickleball_keypoint_taxonomy_includes_corners_nvz_and_centerline_intersections() -> None:
    names = [point.name for point in PICKLEBALL_KEYPOINTS]

    assert names == [
        "near_left_corner",
        "near_baseline_center",
        "near_right_corner",
        "far_right_corner",
        "far_baseline_center",
        "far_left_corner",
        "near_nvz_left",
        "near_nvz_center",
        "near_nvz_right",
        "net_left_sideline",
        "net_center",
        "net_right_sideline",
        "far_nvz_left",
        "far_nvz_center",
        "far_nvz_right",
    ]

    by_name = {point.name: point for point in PICKLEBALL_KEYPOINTS}
    assert by_name["near_left_corner"].world_xyz_m[2] == pytest.approx(0.0)
    assert _ft(by_name["near_left_corner"].world_xyz_m[0]) == pytest.approx(-10.0)
    assert _ft(by_name["near_left_corner"].world_xyz_m[1]) == pytest.approx(-22.0)
    assert _ft(by_name["near_baseline_center"].world_xyz_m[0]) == pytest.approx(0.0)
    assert _ft(by_name["near_baseline_center"].world_xyz_m[1]) == pytest.approx(-22.0)
    assert _ft(by_name["far_baseline_center"].world_xyz_m[0]) == pytest.approx(0.0)
    assert _ft(by_name["far_baseline_center"].world_xyz_m[1]) == pytest.approx(22.0)
    assert _ft(by_name["near_nvz_center"].world_xyz_m[0]) == pytest.approx(0.0)
    assert _ft(by_name["near_nvz_center"].world_xyz_m[1]) == pytest.approx(-7.0)
    assert _ft(by_name["net_center"].world_xyz_m[0]) == pytest.approx(0.0)
    assert _ft(by_name["net_center"].world_xyz_m[1]) == pytest.approx(0.0)
    assert _ft(by_name["far_nvz_center"].world_xyz_m[1]) == pytest.approx(7.0)


def test_synthetic_render_and_training_plan_configs_validate_training_bounds() -> None:
    render = validate_synthetic_render_config(
        {
            "viewpoint_count": 200,
            "height_m": [1.0, 4.0],
            "tilt_deg": [10.0, 80.0],
            "focal_mm_eq": [28.0, 90.0],
            "image_size": [1920, 1080],
            "augmentations": ["shadows", "glare", "occluded_corners"],
        }
    )
    plan = validate_training_plan_config(
        {
            "synthetic": render,
            "tennis_frames": 8800,
            "pickleball_frames": 300,
            "device": "cuda",
            "checkpoint_policy": "courtkeynet_mit",
        }
    )

    assert render.viewpoint_count == 200
    assert render.height_m == pytest.approx((1.0, 4.0))
    assert render.tilt_deg == pytest.approx((10.0, 80.0))
    assert plan.device == "cuda"
    assert plan.checkpoint_policy == "courtkeynet_mit"
    assert plan.pickleball_frames == 300


def test_training_plan_rejects_unknown_devices_policies_and_out_of_recipe_ranges() -> None:
    with pytest.raises(ValueError, match="viewpoint_count"):
        validate_synthetic_render_config({"viewpoint_count": 49})

    with pytest.raises(ValueError, match="height_m"):
        validate_synthetic_render_config({"viewpoint_count": 50, "height_m": [0.9, 4.0]})

    with pytest.raises(ValueError, match="augmentations"):
        validate_synthetic_render_config({"viewpoint_count": 50, "augmentations": "shadows"})

    with pytest.raises(ValueError, match="augmentations"):
        validate_synthetic_render_config({"viewpoint_count": 50, "augmentations": [""]})

    with pytest.raises(ValueError, match="augmentations"):
        validate_synthetic_render_config({"viewpoint_count": 50, "augmentations": [None]})

    with pytest.raises(ValueError, match="augmentations"):
        validate_synthetic_render_config({"viewpoint_count": 50, "augmentations": [3]})

    render = validate_synthetic_render_config({"viewpoint_count": 50})
    with pytest.raises(ValueError, match="device"):
        validate_training_plan_config({"synthetic": render, "device": "tpu"})

    with pytest.raises(ValueError, match="checkpoint_policy"):
        validate_training_plan_config({"synthetic": render, "checkpoint_policy": "unknown_external_checkpoint"})


def test_decode_subpixel_heatmap_uses_parabolic_offset_without_numpy() -> None:
    decoded = decode_subpixel_heatmap(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 4.0, 1.0],
            [0.0, 1.0, 2.0, 0.5],
        ]
    )

    assert decoded.x == pytest.approx(1.75)
    assert decoded.y == pytest.approx(1.1666666667)
    assert decoded.score == pytest.approx(4.0)

    with pytest.raises(ValueError, match="rectangular"):
        decode_subpixel_heatmap([[0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="finite"):
        decode_subpixel_heatmap([[float("nan")]])


def test_validate_heatmap_prediction_payload_decodes_known_keypoints() -> None:
    payload = validate_heatmap_prediction_payload(
        {
            "keypoints": {
                "near_left_corner": {
                    "heatmap": [
                        [0.1, 0.2],
                        [0.3, 0.9],
                    ],
                    "confidence": 0.92,
                },
                "near_nvz_center": {"xy": [101.25, 202.5], "confidence": 0.8},
            }
        }
    )

    assert payload["near_left_corner"].image_xy == pytest.approx((1.0, 1.0))
    assert payload["near_left_corner"].heatmap_score == pytest.approx(0.9)
    assert payload["near_left_corner"].confidence == pytest.approx(0.92)
    assert payload["near_nvz_center"].image_xy == pytest.approx((101.25, 202.5))

    with pytest.raises(ValueError, match="unknown keypoint"):
        validate_heatmap_prediction_payload({"keypoints": {"baseline": {"xy": [0.0, 0.0]}}})

    with pytest.raises(ValueError, match="confidence"):
        validate_heatmap_prediction_payload({"keypoints": {"net_center": {"xy": [0.0, 0.0], "confidence": 1.2}}})


def test_keypoints_to_solvepnp_correspondences_filters_confidence_and_preserves_taxonomy_order() -> None:
    payload = validate_heatmap_prediction_payload(
        {
            "keypoints": {
                "far_right_corner": {"xy": [500.0, 100.0], "confidence": 0.95},
                "near_left_corner": {"xy": [100.0, 900.0], "confidence": 0.99},
                "near_right_corner": {"xy": [900.0, 900.0], "confidence": 0.98},
                "far_left_corner": {"xy": [200.0, 100.0], "confidence": 0.94},
                "net_center": {"xy": [510.0, 520.0], "confidence": 0.2},
            }
        }
    )

    correspondences = keypoints_to_solvepnp_correspondences(payload, min_confidence=0.9)

    assert correspondences.keypoint_names == (
        "near_left_corner",
        "near_right_corner",
        "far_right_corner",
        "far_left_corner",
    )
    assert list(correspondences.image_points_px[0]) == pytest.approx([100.0, 900.0])
    assert list(correspondences.image_points_px[1]) == pytest.approx([900.0, 900.0])
    assert list(correspondences.image_points_px[2]) == pytest.approx([500.0, 100.0])
    assert list(correspondences.image_points_px[3]) == pytest.approx([200.0, 100.0])
    assert _ft(correspondences.object_points_m[0][0]) == pytest.approx(-10.0)
    assert _ft(correspondences.object_points_m[0][1]) == pytest.approx(-22.0)
    assert correspondences.object_points_m[0][2] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="at least 4"):
        keypoints_to_solvepnp_correspondences(payload, min_confidence=0.96)


def test_keypoint_labels_from_court_corners_expands_full_pickleball_layout() -> None:
    labels = keypoint_labels_from_court_corners(
        {
            "near_left": [100.0, 900.0],
            "near_right": [900.0, 900.0],
            "far_right": [700.0, 100.0],
            "far_left": [300.0, 100.0],
        }
    )

    assert set(labels) == {point.name for point in PICKLEBALL_KEYPOINTS}
    assert labels["near_left_corner"] == pytest.approx([100.0, 900.0])
    assert labels["near_right_corner"] == pytest.approx([900.0, 900.0])
    assert labels["far_right_corner"] == pytest.approx([700.0, 100.0])
    assert labels["far_left_corner"] == pytest.approx([300.0, 100.0])
    assert labels["near_baseline_center"][0] == pytest.approx(500.0)
    assert labels["far_baseline_center"][0] == pytest.approx(500.0)
    assert labels["net_center"][0] == pytest.approx(500.0)
    assert labels["net_center"][1] < labels["near_baseline_center"][1]
    assert labels["net_center"][1] > labels["far_baseline_center"][1]


def test_refine_keypoint_xy_with_planar_homography_recovers_scattered_outliers() -> None:
    labels = keypoint_labels_from_court_corners(
        {
            "near_left": [120.0, 900.0],
            "near_right": [930.0, 880.0],
            "far_right": [700.0, 110.0],
            "far_left": [310.0, 120.0],
        }
    )
    raw = {name: list(xy) for name, xy in labels.items()}
    raw["net_left_sideline"] = [1700.0, 80.0]
    raw["net_center"] = [20.0, 930.0]
    raw["far_baseline_center"] = [1800.0, 950.0]
    raw["far_nvz_center"] = [10.0, 10.0]

    refined = refine_keypoint_xy_with_planar_homography(raw, max_inlier_error_px=3.0, min_inliers=8)

    for name, expected_xy in labels.items():
        assert refined[name] == pytest.approx(expected_xy, abs=1e-6)
