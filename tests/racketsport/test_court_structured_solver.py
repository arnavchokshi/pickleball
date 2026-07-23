from __future__ import annotations

import math

import numpy as np
import pytest

from threed.racketsport.court_structured_solver import (
    FLOOR_KEYPOINT_NAMES,
    FLOOR_WORLD_XY_M,
    NET_TOP_KEYPOINT_NAMES,
    solve_best_floor_court,
)


def _project(homography: np.ndarray, xy: tuple[float, float]) -> list[float]:
    source = np.asarray([xy[0], xy[1], 1.0], dtype=np.float64)
    projected = homography @ source
    return [float(projected[0] / projected[2]), float(projected[1] / projected[2])]


def _perspective_homography() -> np.ndarray:
    return np.asarray(
        [
            [76.0, 13.0, 651.0],
            [-9.0, 48.0, 382.0],
            [0.007, -0.011, 1.0],
        ],
        dtype=np.float64,
    )


def _observations(
    homography: np.ndarray,
    *,
    include_alternates: bool = False,
) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for index, name in enumerate(FLOOR_KEYPOINT_NAMES):
        exact = _project(homography, FLOOR_WORLD_XY_M[name])
        rows = [
            {
                "id": f"{name}:true",
                "xy": exact,
                "confidence": 0.92 - index * 0.005,
                "visibility": 0.98,
                "covariance": [[1.5, 0.1], [0.1, 2.0]],
            }
        ]
        if include_alternates:
            rows.append(
                {
                    "id": f"{name}:outlier",
                    "xy": [exact[0] + 90.0 + index, exact[1] - 70.0],
                    "confidence": 0.62,
                    "visibility": 1.0,
                    "covariance": 4.0,
                }
            )
        result[name] = rows
    return result


def test_robust_floor_solver_uses_one_homography_and_ignores_outliers_duplicates_and_net() -> None:
    homography = _perspective_homography()
    observations = _observations(homography, include_alternates=True)
    duplicate_xy = observations["near_left_corner"][0]["xy"]
    observations["near_right_corner"].insert(
        0,
        {
            "id": "near_right:duplicate_wrong",
            "xy": duplicate_xy,
            "confidence": 0.97,
            "visibility": 1.0,
            "covariance": 1.0,
        },
    )
    observations["net_center"] = [
        {
            "id": "net:high",
            "xy": [640.0, 350.0],
            "confidence": 0.999,
            "visibility": 1.0,
            "covariance": 0.25,
        }
    ]

    result = solve_best_floor_court(observations)

    assert result["status"] == "solved_best_effort"
    assert result["measurement_valid"] is False
    assert result["authority_state"] == "review_only"
    assert result["solution_role"] == "best_effort"
    assert result["verified"] is False
    assert set(result["projected_floor_keypoints"]) == set(FLOOR_KEYPOINT_NAMES)
    assert set(result["point_confidence"]) == set(FLOOR_KEYPOINT_NAMES)
    assert set(result["projected_floor_keypoints"]).isdisjoint(NET_TOP_KEYPOINT_NAMES)
    for name in FLOOR_KEYPOINT_NAMES:
        expected = _project(homography, FLOOR_WORLD_XY_M[name])
        assert result["projected_floor_keypoints"][name]["xy"] == pytest.approx(expected, abs=1.0e-5)

    inlier_ids = {item["candidate_id"] for item in result["inliers"]}
    assert "near_right:duplicate_wrong" not in inlier_ids
    assert len(result["inliers"]) == len(FLOOR_KEYPOINT_NAMES)
    ignored = {(item["candidate_id"], item["reason"]) for item in result["ignored_observations"]}
    assert ("net:high", "net_top_excluded_floor_only_solver") in ignored
    assert any(candidate_id.endswith(":outlier") for candidate_id, _reason in ignored)
    assert any(
        candidate_id == "near_right:duplicate_wrong"
        and reason in {"duplicate_image_location", "residual_outlier", "alternate_candidate_not_selected"}
        for candidate_id, reason in ignored
    )
    assert result["score_components"]["inlier_count"] == 12
    assert 0.0 < result["court_confidence"] < 1.0


def test_top_two_cap_confidence_visibility_and_covariance_control_candidate_priority() -> None:
    homography = _perspective_homography()
    observations = _observations(homography)
    name = "near_left_corner"
    exact = observations[name][0]["xy"]
    observations[name] = [
        {
            "id": "huge-covariance",
            "xy": [exact[0] + 55.0, exact[1] - 40.0],
            "confidence": 0.99,
            "visibility": 1.0,
            "covariance": 2500.0,
        },
        {
            "id": "true",
            "xy": exact,
            "confidence": 0.88,
            "visibility": 1.0,
            "covariance": 1.0,
        },
        {
            "id": "third",
            "xy": [exact[0] + 200.0, exact[1] + 200.0],
            "confidence": 0.01,
            "visibility": 1.0,
            "covariance": 1.0,
        },
        {
            "id": "invisible",
            "xy": exact,
            "confidence": 1.0,
            "visibility": False,
            "covariance": 0.1,
        },
    ]

    result = solve_best_floor_court(observations, max_hypotheses=31)

    assert {item["candidate_id"] for item in result["inliers"]} >= {"true"}
    ignored = {item["candidate_id"]: item["reason"] for item in result["ignored_observations"]}
    assert ignored["third"] == "below_top2_confidence"
    assert ignored["invisible"] == "visibility_zero"
    search = result["diagnostics"]["hypothesis_search"]
    assert search["hypothesis_cap"] == 31
    assert search["hypotheses_retained_cap"] <= 31
    assert search["valid_homographies_scored"] <= 31


def test_swapped_semantic_candidates_are_outvoted_by_global_regulation_consensus() -> None:
    homography = _perspective_homography()
    observations = _observations(homography)
    left = observations["far_left_corner"][0]["xy"]
    right = observations["far_right_corner"][0]["xy"]
    observations["far_left_corner"].insert(
        0,
        {
            "id": "swapped-left",
            "xy": right,
            "confidence": 0.97,
            "visibility": 1.0,
            "covariance": 1.0,
        },
    )
    observations["far_right_corner"].insert(
        0,
        {
            "id": "swapped-right",
            "xy": left,
            "confidence": 0.97,
            "visibility": 1.0,
            "covariance": 1.0,
        },
    )

    result = solve_best_floor_court(observations)

    inlier_ids = {item["candidate_id"] for item in result["inliers"]}
    assert "swapped-left" not in inlier_ids
    assert "swapped-right" not in inlier_ids
    assert len(result["inliers"]) == 12
    assert result["residual_stats_px"]["p90"] == pytest.approx(0.0, abs=1.0e-5)


def test_prior_fallback_projects_all_floor_points_but_never_claims_measurement_authority() -> None:
    homography = _perspective_homography()
    observations = {
        name: [
            {
                "id": name,
                "xy": _project(homography, FLOOR_WORLD_XY_M[name]),
                "confidence": 0.8,
                "visibility": 1.0,
                "covariance": 4.0,
            }
        ]
        for name in FLOOR_KEYPOINT_NAMES[:3]
    }

    result = solve_best_floor_court(observations, prior_homography=homography)

    assert result["status"] == "solved_best_effort"
    assert result["measurement_valid"] is False
    assert result["authority_state"] == "review_only"
    assert result["selected_hypothesis"]["source"].startswith("prior_homography")
    assert set(result["projected_floor_keypoints"]) == set(FLOOR_KEYPOINT_NAMES)
    assert result["diagnostics"]["prior_homography"]["accepted"] is True
    assert result["court_confidence"] < 0.25


def test_prior_only_with_no_observations_is_explicitly_low_confidence_best_effort() -> None:
    homography = _perspective_homography()

    result = solve_best_floor_court({}, prior_homography=homography)

    assert result["status"] == "prior_only_best_effort"
    assert result["inliers"] == []
    assert result["court_confidence"] == pytest.approx(0.05)
    assert set(result["projected_floor_keypoints"]) == set(FLOOR_KEYPOINT_NAMES)
    assert all(confidence < 0.05 for confidence in result["point_confidence"].values())
    assert result["measurement_valid"] is False


def test_insufficient_observations_without_prior_fail_closed_without_inventing_points() -> None:
    observations = {
        "near_left_corner": {
            "xy": [10.0, 20.0],
            "confidence": 0.9,
            "visibility": 1.0,
            "covariance": 1.0,
        },
        "made_up_point": {
            "xy": [30.0, 40.0],
            "confidence": 1.0,
            "visibility": 1.0,
            "covariance": 1.0,
        },
    }

    result = solve_best_floor_court(observations)

    assert result["status"] == "insufficient_floor_hypothesis"
    assert result["homography_image_from_court"] is None
    assert result["projected_floor_keypoints"] == {}
    assert result["point_confidence"] == {}
    assert result["court_confidence"] == 0.0
    assert result["measurement_valid"] is False
    assert any(
        item["semantic"] == "made_up_point" and item["reason"] == "unknown_semantic"
        for item in result["ignored_observations"]
    )


def test_projective_diagnostics_do_not_assume_axis_aligned_or_equal_image_lengths() -> None:
    angle = math.radians(61.0)
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    perspective = np.asarray(
        [
            [48.0, 5.0, 820.0],
            [3.0, 92.0, 510.0],
            [0.015, -0.021, 1.0],
        ]
    )
    homography = perspective @ rotation

    result = solve_best_floor_court(_observations(homography))

    diagnostics = result["diagnostics"]
    assert diagnostics["projective"]["finite"] is True
    assert diagnostics["projective"]["invertible"] is True
    assert diagnostics["convexity"]["outer_court_convex"] is True
    assert diagnostics["convexity"]["self_intersecting"] is False
    assert diagnostics["diagonal_center"]["diagonals_intersect_finitely"] is True
    assert diagnostics["diagonal_center"]["diagonal_center_residual_px"] == pytest.approx(
        0.0, abs=1.0e-6
    )
    assert diagnostics["order"]["all_passed"] is True
    assert "no image-angle or image-length equality assumptions" in diagnostics["diagnostic_note"]
