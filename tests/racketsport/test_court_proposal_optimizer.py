from __future__ import annotations

import math

import numpy as np
import pytest

from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS
from threed.racketsport.court_proposal_optimizer import (
    NET_TOP_KEYPOINT_NAMES,
    RefinementConfig,
    accept_refinement,
    diagnose_planar_keypoint_refinement,
    magsac_homography_from_points,
    refine_homography_with_lines,
    score_homography_support,
    synthesize_hybrid_intersection_points,
)
from threed.racketsport.court_templates import get_court_template


EXACT_H = np.asarray(
    [[118.0, 11.0, 960.0], [4.0, 44.0, 520.0], [0.012, 0.019, 1.0]],
    dtype=np.float64,
)


def _project(homography: np.ndarray, xy: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([xy, np.ones(len(xy))])
    projected = (homography @ homogeneous.T).T
    return projected[:, :2] / projected[:, 2:3]


def _synthetic_evidence(*, heldout_shift: float = 0.0) -> tuple[dict[str, object], dict[str, object]]:
    template = get_court_template("pickleball")
    lines: dict[str, object] = {}
    for index, (line_id, endpoints) in enumerate(template.line_segments_m.items()):
        if line_id == "net":
            continue
        projected = _project(EXACT_H, np.asarray(endpoints, dtype=np.float64)[:, :2])
        optimize = projected + np.asarray([0.2 * math.sin(index), 0.2 * math.cos(index)])
        heldout = projected + np.asarray([0.0, heldout_shift])
        lines[line_id] = {
            "optimize": optimize.tolist(),
            "heldout": heldout.tolist(),
            "confidence": 0.9,
        }
    points = {}
    for index, point in enumerate(PICKLEBALL_KEYPOINTS):
        projected = _project(EXACT_H, np.asarray([[point.world_xyz_m[0], point.world_xyz_m[1]]]))[0]
        points[point.name] = {
            "xy": (projected + np.asarray([0.5 * math.sin(index), 0.5 * math.cos(index)])).tolist(),
            "confidence": 0.9,
        }
    return lines, points


def test_accept_refinement_rejects_better_line_residual_when_pixel_support_worsens() -> None:
    before = {"line_rmse_px": 20.0, "pixel_support": 0.70, "p95_px": 200.0, "median_px": 100.0}
    after = {"line_rmse_px": 10.0, "pixel_support": 0.40, "p95_px": 220.0, "median_px": 90.0}

    accepted, reasons = accept_refinement(before, after)

    assert accepted is False
    assert "pixel_support_worsened" in reasons


def test_accept_refinement_accepts_when_line_pixel_and_tail_improve() -> None:
    before = {"line_rmse_px": 20.0, "pixel_support": 0.70, "p95_px": 200.0, "median_px": 100.0}
    after = {"line_rmse_px": 12.0, "pixel_support": 0.78, "p95_px": 170.0, "median_px": 90.0}

    accepted, reasons = accept_refinement(before, after)

    assert accepted is True
    assert reasons == []


def test_zero_evidence_returns_seed_with_named_reason_and_covariance_inflation() -> None:
    result = refine_homography_with_lines(EXACT_H.tolist(), semantic_lines={}, line_distance_map=None)

    assert result["accepted"] is False
    assert np.asarray(result["homography_image_from_court"]) == pytest.approx(EXACT_H)
    assert result["reject_reasons"] == ["insufficient_line_evidence"]
    assert result["covariance_inflation_required"] is True


def test_exact_h_plus_one_pixel_scale_noise_improves_or_stays_within_tolerance() -> None:
    lines, points = _synthetic_evidence()
    seed = EXACT_H.copy()
    seed[0, 2] += 6.0
    seed[1, 2] -= 4.0

    result = refine_homography_with_lines(
        seed.tolist(),
        lines,
        None,
        points,
        config=RefinementConfig(max_condition_number=1e15),
    )

    assert result["accepted"] is True, result
    assert result["selection"] == "refined"
    assert result["scores_after"]["p90_px"] <= result["scores_before"]["p90_px"] + 0.25
    assert result["scores_after"]["median_px"] < result["scores_before"]["median_px"]
    assert result["telemetry"]["geometry_synthesized_point_count"] >= 4


def test_observation_bootstrap_stability_guard_fires_on_line_driven_slide() -> None:
    lines, points = _synthetic_evidence()
    for row in lines.values():
        row["optimize"] = (np.asarray(row["optimize"]) + np.asarray([8.0, 0.0])).tolist()
        row["heldout"] = (np.asarray(row["heldout"]) + np.asarray([8.0, 0.0])).tolist()

    result = refine_homography_with_lines(
        EXACT_H.tolist(),
        lines,
        None,
        points,
        config=RefinementConfig(max_condition_number=1e15, max_corner_shift_px=100.0),
    )

    assert result["accepted"] is False
    assert result["selection_reason"] == "seed_wins_observation_bootstrap_stability"
    assert np.asarray(result["homography_image_from_court"]) == pytest.approx(EXACT_H)
    assert result["covariance_inflation_required"] is True
    assert all(
        row["observation_bootstrap_ratio_vs_seed"] > 1.08
        for row in result["telemetry"]["guard_line_search"]
        if row["selection_reason"] == "seed_wins_observation_bootstrap_stability"
    )


def test_band_refined_crossing_synthesizes_ground_truth_point_with_covariance() -> None:
    template = get_court_template("pickleball")

    def hybrid_segment(line_id: str) -> dict[str, object]:
        world = np.asarray(template.line_segments_m[line_id], dtype=np.float64)[:, :2]
        endpoints = _project(EXACT_H, world)
        direction = endpoints[1] - endpoints[0]
        direction /= np.linalg.norm(direction)
        normal = np.asarray([-direction[1], direction[0]])
        covariance = 0.01 * np.outer(normal, normal)
        samples = []
        for fraction in np.linspace(0.0, 1.0, 5):
            xy = endpoints[0] + fraction * (endpoints[1] - endpoints[0])
            samples.append(
                {
                    "xy": xy.tolist(),
                    "normal_covariance_px2": covariance.tolist(),
                    "provenance": "band_refined",
                }
            )
        return {"endpoints": endpoints.tolist(), "sampled_points": samples}

    semantic_lines = {
        "near_baseline": {"optimize_hybrid": [hybrid_segment("near_baseline")]},
        "left_sideline": {"optimize_hybrid": [hybrid_segment("left_sideline")]},
    }

    points = synthesize_hybrid_intersection_points(semantic_lines)

    assert len(points) == 1
    expected_world = np.asarray([-3.048, -6.7056])
    expected_image = _project(EXACT_H, expected_world[None, :])[0]
    covariance = np.asarray(points[0]["covariance_px2"])
    assert points[0]["world_xy"] == pytest.approx(expected_world)
    assert points[0]["image_xy"] == pytest.approx(expected_image, abs=1e-6)
    assert covariance == pytest.approx(covariance.T, abs=1e-12)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)
    assert float(np.trace(covariance)) < 0.01


def test_one_gross_floor_outlier_is_rejected_by_magsac() -> None:
    points = []
    for point in PICKLEBALL_KEYPOINTS:
        if point.name in NET_TOP_KEYPOINT_NAMES:
            continue
        world = [point.world_xyz_m[0], point.world_xyz_m[1]]
        image = _project(EXACT_H, np.asarray([world]))[0].tolist()
        points.append({"name": point.name, "world_xy": world, "image_xy": image, "confidence": 1.0})
    points[-1]["image_xy"] = [25.0, 25.0]

    result = magsac_homography_from_points(points, threshold_px=2.0)

    assert result["success"] is True
    assert result["method"] in {"cv2.USAC_MAGSAC", "cv2.LMEDS", "cv2.RANSAC"}
    assert result["inlier_count"] <= len(points) - 1
    recovered = np.asarray(result["homography"])
    court = np.asarray([[0.0, 0.0], [-3.048, -6.7056], [3.048, 6.7056]])
    assert _project(recovered, court) == pytest.approx(_project(EXACT_H, court), abs=1e-4)


def test_three_net_top_points_never_enter_planar_optimizer_fit() -> None:
    lines, points = _synthetic_evidence()
    seed = EXACT_H.copy()
    seed[0, 2] += 3.0

    result = refine_homography_with_lines(
        seed.tolist(), lines, None, points, config=RefinementConfig(max_condition_number=1e15)
    )

    assert result["telemetry"]["excluded_net_top_point_count"] == 3
    assert result["telemetry"]["net_top_point_count_in_planar_fit"] == 0


def test_wrong_parallel_line_family_scores_worse_than_correct_identity() -> None:
    lines, points = _synthetic_evidence()
    correct = score_homography_support(EXACT_H.tolist(), lines, None, points)
    wrong = EXACT_H.copy()
    wrong[1, 2] += 28.0
    wrong_score = score_homography_support(wrong.tolist(), lines, None, points)

    assert wrong_score["median_px"] > correct["median_px"] + 10.0
    assert wrong_score["pixel_support"] < correct["pixel_support"]


def test_distorted_camera_requires_typed_space_and_optimizes_undistorted() -> None:
    lines, points = _synthetic_evidence()
    calibration = {
        "intrinsics": {"fx": 1180.0, "fy": 1180.0, "cx": 960.0, "cy": 540.0, "dist": [-0.15, 0.03, 0.0, 0.0]},
        "extrinsics": {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 12.0]},
    }

    missing_type = refine_homography_with_lines(EXACT_H.tolist(), lines, None, points, calibration=calibration)
    typed = refine_homography_with_lines(
        EXACT_H.tolist(),
        lines,
        None,
        points,
        calibration=calibration,
        coordinate_space="pixels_undistorted_native",
        config=RefinementConfig(max_condition_number=1e18, max_corner_shift_px=2000.0),
    )

    assert missing_type["reject_reasons"] == ["coordinate_space_missing_for_distorted_calibration"]
    assert typed["telemetry"]["coordinate_space"] == "pixels_undistorted_native"
    assert typed["telemetry"]["distortion_present"] is True


def test_keypoint_diagnosis_reproduces_insufficient_candidate_noop() -> None:
    raw = {
        point.name: _project(EXACT_H, np.asarray([[point.world_xyz_m[0], point.world_xyz_m[1]]]))[0].tolist()
        for point in PICKLEBALL_KEYPOINTS[:7]
    }

    diagnosis = diagnose_planar_keypoint_refinement(raw)

    assert diagnosis["candidate_count"] == 7
    assert diagnosis["fallback_reason"] == "insufficient_candidates"
    assert diagnosis["output_delta_px"]["max"] == 0.0


def test_all15_planar_refinement_is_worse_than_floor12_when_net_tops_are_nonplanar() -> None:
    raw = {}
    for point in PICKLEBALL_KEYPOINTS:
        projected = _project(EXACT_H, np.asarray([[point.world_xyz_m[0], point.world_xyz_m[1]]]))[0]
        if point.name in NET_TOP_KEYPOINT_NAMES:
            projected = projected + np.asarray([0.0, -24.0])
        raw[point.name] = projected.tolist()
    floor = {name: xy for name, xy in raw.items() if name not in NET_TOP_KEYPOINT_NAMES}

    all15 = diagnose_planar_keypoint_refinement(raw, max_inlier_error_px=30.0, min_inliers=8)
    floor12 = diagnose_planar_keypoint_refinement(floor, max_inlier_error_px=30.0, min_inliers=8)
    floor_truth = {
        point.name: _project(EXACT_H, np.asarray([[point.world_xyz_m[0], point.world_xyz_m[1]]]))[0]
        for point in PICKLEBALL_KEYPOINTS
        if point.name not in NET_TOP_KEYPOINT_NAMES
    }

    def floor_error(result: dict[str, object]) -> float:
        output = result["refined_output"]
        return float(np.median([np.linalg.norm(np.asarray(output[name]) - truth) for name, truth in floor_truth.items()]))

    assert all15["net_top_points_entered_planar_fit"] is True
    assert all15["net_top_in_best_inliers"] is True
    assert floor12["net_top_points_entered_planar_fit"] is False
    assert floor_error(all15) > floor_error(floor12) + 0.5
