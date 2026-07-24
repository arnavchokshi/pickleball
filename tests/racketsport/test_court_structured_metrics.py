from __future__ import annotations

import math

import pytest

from threed.racketsport.court_structured_metrics import (
    evaluate_raw_vs_structured_court_outputs,
    evaluate_structured_court_outputs,
)


def _template_points() -> dict[str, list[float]]:
    """A perspective trapezoid that preserves every planar template invariant exactly."""

    return {
        "near_left_corner": [0.0, 400.0],
        "near_baseline_center": [200.0, 400.0],
        "near_right_corner": [400.0, 400.0],
        "near_nvz_left": [100.0 / 3.0, 300.0],
        "near_nvz_center": [200.0, 300.0],
        "near_nvz_right": [1100.0 / 3.0, 300.0],
        "far_nvz_left": [200.0 / 3.0, 200.0],
        "far_nvz_center": [200.0, 200.0],
        "far_nvz_right": [1000.0 / 3.0, 200.0],
        "far_left_corner": [100.0, 100.0],
        "far_baseline_center": [200.0, 100.0],
        "far_right_corner": [300.0, 100.0],
    }


def _translated(points: dict[str, list[float]], dx: float, dy: float) -> dict[str, list[float]]:
    return {name: [xy[0] + dx, xy[1] + dy] for name, xy in points.items()}


def test_exact_name_error_pck_missing_policy_p95_and_per_viewpoint() -> None:
    records = [
        {
            "sample_id": "diagonal/1",
            "viewpoint": "diagonal",
            "ground_truth": {"left": [0.0, 0.0], "right": [10.0, 0.0]},
            "prediction": {
                "keypoints": {"left": [3.0, 4.0]},
                "confidences": {"left": 0.8, "right": 0.2},
                "whole_court_confidence": 0.7,
            },
        },
        {
            "sample_id": "baseline/1",
            "viewpoint": "baseline",
            "ground_truth": {"left": [0.0, 0.0], "right": [10.0, 0.0]},
            "prediction": {"keypoints": {"left": [0.0, 0.0], "right": [10.0, 0.0]}},
        },
    ]

    metrics = evaluate_structured_court_outputs(records, ece_bin_count=5)
    point = metrics["point_metrics"]
    assert point["labeled_count"] == 4
    assert point["matched_count"] == 3
    assert point["missing_prediction_count"] == 1
    assert point["pck_at_5px"] == pytest.approx(0.75)
    assert point["pck_at_10px"] == pytest.approx(0.75)
    assert point["pck_at_2px"] == pytest.approx(0.5)
    assert point["p90_error_px"] == pytest.approx(4.0)
    assert point["p95_error_px"] == pytest.approx(4.5)
    assert point["max_error_px"] == pytest.approx(5.0)
    assert metrics["samples"][0]["point_errors_px"] == {"left": 5.0, "right": None}
    assert metrics["configuration"]["semantic_matching"] == "exact_name_only"

    diagonal = metrics["per_viewpoint"]["diagonal"]["point_metrics"]
    baseline = metrics["per_viewpoint"]["baseline"]["point_metrics"]
    assert diagonal["pck_at_5px"] == pytest.approx(0.5)
    assert baseline["pck_at_5px"] == pytest.approx(1.0)
    assert point["per_semantic_name"]["right"]["missing_prediction_count"] == 1


def test_exact_semantic_matching_does_not_hide_left_right_swap() -> None:
    metrics = evaluate_structured_court_outputs(
        [
            {
                "sample_id": "swap",
                "viewpoint": "baseline",
                "ground_truth": {"left": [0.0, 0.0], "right": [100.0, 0.0]},
                "prediction": {"keypoints": {"left": [100.0, 0.0], "right": [0.0, 0.0]}},
            }
        ]
    )
    assert metrics["point_metrics"]["mean_error_px"] == 100.0
    assert metrics["point_metrics"]["pck_at_10px"] == 0.0


def test_template_topology_simple_polygon_and_duplicate_collapse_are_scored() -> None:
    ground_truth = _template_points()
    good = _translated(ground_truth, 2.0, 0.0)
    bad = {name: list(xy) for name, xy in ground_truth.items()}
    bad["near_left_corner"], bad["near_right_corner"] = (
        bad["near_right_corner"],
        bad["near_left_corner"],
    )
    bad["near_nvz_center"] = list(bad["near_nvz_left"])

    metrics = evaluate_structured_court_outputs(
        [
            {
                "sample_id": "valid",
                "viewpoint": "high",
                "ground_truth": ground_truth,
                "prediction": {"keypoints": good},
            },
            {
                "sample_id": "invalid",
                "viewpoint": "low",
                "ground_truth": ground_truth,
                "prediction": {"keypoints": bad},
            },
        ],
        collapse_distance_px=0.1,
    )

    structure = metrics["structure"]
    assert structure["duplicate_or_collapse_pair_count"] == 1
    assert structure["samples_with_duplicate_or_collapse_count"] == 1
    assert structure["boundary_polygon_available_count"] == 2
    assert structure["boundary_polygon_self_intersection_count"] == 1
    assert structure["topology_valid_count"] == 1
    assert structure["centerline_nvz_template_valid_count"] == 1
    assert structure["topology_valid_rate"] == pytest.approx(0.5)

    valid_topology = metrics["samples"][0]["topology"]
    invalid_topology = metrics["samples"][1]["topology"]
    assert valid_topology["topology_valid"] is True
    assert all(valid_topology["template_invariant_checks"].values())
    assert invalid_topology["boundary_polygon_self_intersects"] is True
    assert invalid_topology["template_invariant_valid"] is False
    assert invalid_topology["topology_valid"] is False


def test_ignored_observation_outlier_recovery_uses_declared_or_computed_gt_error() -> None:
    metrics = evaluate_structured_court_outputs(
        [
            {
                "sample_id": "outliers",
                "viewpoint": "baseline",
                "ground_truth": {"a": [0.0, 0.0], "b": [10.0, 0.0]},
                "prediction": {
                    "keypoints": {"a": [0.0, 0.0], "b": [10.0, 0.0]},
                    "ignored_observations": [
                        {"name": "a", "xy": [6.0, 8.0]},
                        {"name": "b", "gt_error_px": 4.0},
                        {"name": "missing", "xy": [99.0, 99.0]},
                    ],
                },
            }
        ],
        outlier_threshold_px=5.0,
    )
    recovery = metrics["outlier_recovery"]
    assert recovery["available"] is True
    assert recovery["ignored_observations_with_gt_error"] == 2
    assert recovery["recovered_outlier_count"] == 1
    assert recovery["ignored_inlier_count"] == 1
    assert recovery["recovery_precision"] == pytest.approx(0.5)
    sample = metrics["samples"][0]["outlier_recovery"]
    assert sample["ignored_observations_without_gt_error"] == 1
    assert sample["observations"][0]["gt_error_px"] == 10.0


def test_point_and_whole_court_brier_and_ece_bins() -> None:
    ground_truth = _template_points()
    metrics = evaluate_structured_court_outputs(
        [
            {
                "sample_id": "calibrated",
                "viewpoint": "baseline",
                "ground_truth": ground_truth,
                "prediction": {
                    "keypoints": _translated(ground_truth, 3.0, 0.0),
                    "confidences": {name: 0.8 for name in ground_truth},
                    "whole_court_confidence": 0.9,
                },
            },
            {
                "sample_id": "overconfident",
                "viewpoint": "baseline",
                "ground_truth": {"one": [0.0, 0.0]},
                "prediction": {
                    "keypoints": {"one": [20.0, 0.0]},
                    "confidences": {"one": 0.8},
                    "whole_court_confidence": 0.9,
                },
            },
        ],
        ece_bin_count=5,
    )

    point_calibration = metrics["point_confidence_calibration"]
    assert point_calibration["available"] is True
    assert point_calibration["count"] == 13
    assert point_calibration["brier_score"] == pytest.approx((12 * 0.04 + 0.64) / 13)
    assert point_calibration["ece"] == pytest.approx(abs(0.8 - 12 / 13))
    assert sum(row["count"] for row in point_calibration["bins"]) == 13

    whole = metrics["whole_court_calibration"]
    assert whole["available"] is True
    assert whole["count"] == 2
    assert whole["brier_score"] == pytest.approx((0.1**2 + 0.9**2) / 2.0)
    assert whole["ece"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "record, message",
    [
        (
            {
                "ground_truth": {"a": [0.0, 0.0]},
                "prediction": {"keypoints": {"a": [0.0, 0.0]}, "confidences": {"a": 1.1}},
            },
            "between 0 and 1",
        ),
        (
            {"ground_truth": {"a": [0.0]}, "prediction": {"keypoints": {"a": [0.0, 0.0]}}},
            "two-item coordinate",
        ),
    ],
)
def test_invalid_structured_metric_inputs_fail_loudly(record: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_structured_court_outputs([record])


def test_empty_input_reports_unavailable_metrics_without_dividing_by_zero() -> None:
    metrics = evaluate_structured_court_outputs([])
    assert metrics["sample_count"] == 0
    assert metrics["point_metrics"]["pck_at_5px"] is None
    assert metrics["point_confidence_calibration"]["available"] is False
    assert metrics["whole_court_calibration"]["available"] is False
    assert metrics["per_viewpoint"] == {}
    assert math.isfinite(metrics["configuration"]["pck_thresholds_px"][0])


def test_paired_raw_vs_structured_metrics_report_strata_and_sample_bootstrap() -> None:
    records = [
        {
            "sample_id": "owner/frame_1",
            "viewpoint": "straight",
            "strata": {
                "subgroup": "owner",
                "source": "camera_a",
                "source_group": "venue_a",
                "visibility": "full_floor_12",
            },
            "ground_truth": {"left": [0.0, 0.0], "right": [10.0, 0.0]},
            "raw_prediction": {
                "keypoints": {"left": [1.0, 0.0], "right": [18.0, 0.0]}
            },
            "structured_prediction": {
                "keypoints": {"left": [2.0, 0.0], "right": [14.0, 0.0]}
            },
        },
        {
            "sample_id": "external/frame_1",
            "viewpoint": "diagonal",
            "strata": {
                "subgroup": "external",
                "source": "dataset_b",
                "source_group": "venue_b",
                "visibility": "partial_floor",
            },
            "ground_truth": {"center": [0.0, 0.0]},
            "raw_prediction": {"keypoints": {}},
            "structured_prediction": {"keypoints": {"center": [20.0, 0.0]}},
        },
    ]

    metrics = evaluate_raw_vs_structured_court_outputs(
        records,
        bootstrap_resamples=200,
        bootstrap_seed=13,
    )

    assert metrics["evaluated_taxonomy"] == "canonical_floor_points_exact_semantic_name"
    assert metrics["raw"]["point_metrics"]["pck_at_2px"] == pytest.approx(1 / 3)
    assert metrics["raw"]["point_metrics"]["pck_at_5px"] == pytest.approx(1 / 3)
    assert metrics["raw"]["point_metrics"]["pck_at_10px"] == pytest.approx(2 / 3)
    assert metrics["structured"]["point_metrics"]["pck_at_5px"] == pytest.approx(2 / 3)
    assert metrics["structured"]["point_metrics"]["p90_error_px"] == pytest.approx(16.8)
    assert metrics["structured"]["point_metrics"]["max_error_px"] == 20.0
    point = metrics["paired_deltas"]["point_estimates"]
    assert point["pck_at_5px_structured_minus_raw"] == pytest.approx(1 / 3)
    assert point["paired_error_count"] == 2
    assert point["mean_paired_error_reduction_px"] == pytest.approx(1.5)
    assert metrics["paired_deltas"]["bootstrap_unit"] == "sample"
    assert metrics["paired_deltas"]["bootstrap_resamples"] == 200
    assert metrics["paired_deltas"]["bootstrap_95_intervals"][
        "pck_at_5px_structured_minus_raw"
    ]["available"] is True
    assert set(metrics["strata"]["subgroup"]) == {"external", "owner"}
    assert set(metrics["strata"]["source"]) == {"camera_a", "dataset_b"}
    assert set(metrics["strata"]["viewpoint"]) == {"diagonal", "straight"}
    assert set(metrics["strata"]["visibility"]) == {"full_floor_12", "partial_floor"}
    assert metrics["task88_policy"].startswith("historical_development_only")
