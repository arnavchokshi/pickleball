"""Metrics for semantically structured pickleball-court predictions.

The evaluator deliberately compares points by canonical semantic name.  It never rematches a
prediction to the nearest ground-truth point, because doing so can hide left/right or near/far
label swaps.  Missing predictions remain PCK failures while pixel-error percentiles are reported
only over same-name pairs that actually exist.

Input records have this compact schema::

    {
        "sample_id": "clip/frame",
        "viewpoint": "high_baseline",
        "ground_truth": {"near_left_corner": [x, y], ...},
        "prediction": {
            "keypoints": {"near_left_corner": [x, y], ...},
            "confidences": {"near_left_corner": 0.9, ...},       # optional
            "ignored_observations": [                            # optional
                {"name": "near_left_corner", "xy": [x, y]},
                {"name": "far_right_corner", "gt_error_px": 31.0},
            ],
            "whole_court_confidence": 0.8,                       # optional
        },
    }

The planar topology checks are projective invariants of the regulation template: boundary
corners must form a simple polygon; baseline/NVZ centers must lie between their named endpoints;
and the two NVZ intersections must occur in order along both sidelines and the centerline.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


BOUNDARY_POLYGON = (
    "near_left_corner",
    "near_right_corner",
    "far_right_corner",
    "far_left_corner",
)

TEMPLATE_ROW_TRIPLETS = (
    ("near_left_corner", "near_baseline_center", "near_right_corner"),
    ("near_nvz_left", "near_nvz_center", "near_nvz_right"),
    ("far_nvz_left", "far_nvz_center", "far_nvz_right"),
    ("far_left_corner", "far_baseline_center", "far_right_corner"),
)

TEMPLATE_ORDERED_LINES = (
    (
        "left_sideline",
        ("near_left_corner", "near_nvz_left", "far_nvz_left", "far_left_corner"),
    ),
    (
        "right_sideline",
        ("near_right_corner", "near_nvz_right", "far_nvz_right", "far_right_corner"),
    ),
    (
        "centerline",
        ("near_baseline_center", "near_nvz_center", "far_nvz_center", "far_baseline_center"),
    ),
)


def evaluate_structured_court_outputs(
    records: Sequence[Mapping[str, Any]],
    *,
    pck_thresholds_px: tuple[float, ...] = (2.0, 5.0, 10.0),
    collapse_distance_px: float = 1.0,
    topology_tolerance_px: float = 3.0,
    outlier_threshold_px: float = 5.0,
    ece_bin_count: int = 10,
) -> dict[str, Any]:
    """Evaluate structured court records and return aggregate plus per-viewpoint metrics."""

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of mappings")
    thresholds = tuple(_positive_float(value, "pck_thresholds_px") for value in pck_thresholds_px)
    if not thresholds or any(left >= right for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError("pck_thresholds_px must contain increasing positive thresholds")
    collapse_distance_px = _nonnegative_float(collapse_distance_px, "collapse_distance_px")
    topology_tolerance_px = _nonnegative_float(topology_tolerance_px, "topology_tolerance_px")
    outlier_threshold_px = _nonnegative_float(outlier_threshold_px, "outlier_threshold_px")
    if isinstance(ece_bin_count, bool) or not isinstance(ece_bin_count, int) or ece_bin_count <= 0:
        raise ValueError("ece_bin_count must be a positive integer")

    normalized = [_normalize_record(record, index=index) for index, record in enumerate(records)]
    overall = _summarize_records(
        normalized,
        thresholds=thresholds,
        collapse_distance_px=collapse_distance_px,
        topology_tolerance_px=topology_tolerance_px,
        outlier_threshold_px=outlier_threshold_px,
        ece_bin_count=ece_bin_count,
        include_samples=True,
    )
    by_viewpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in normalized:
        by_viewpoint[record["viewpoint"]].append(record)
    overall["per_viewpoint"] = {
        viewpoint: _summarize_records(
            grouped,
            thresholds=thresholds,
            collapse_distance_px=collapse_distance_px,
            topology_tolerance_px=topology_tolerance_px,
            outlier_threshold_px=outlier_threshold_px,
            ece_bin_count=ece_bin_count,
            include_samples=False,
        )
        for viewpoint, grouped in sorted(by_viewpoint.items())
    }
    strata: dict[str, Any] = {}
    for dimension in ("subgroup", "source", "source_group", "viewpoint", "visibility"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in normalized:
            value = record["strata"].get(dimension)
            if value is not None:
                grouped[value].append(record)
        strata[dimension] = {
            value: _summarize_records(
                rows,
                thresholds=thresholds,
                collapse_distance_px=collapse_distance_px,
                topology_tolerance_px=topology_tolerance_px,
                outlier_threshold_px=outlier_threshold_px,
                ece_bin_count=ece_bin_count,
                include_samples=False,
            )
            for value, rows in sorted(grouped.items())
        }
    overall["strata"] = strata
    overall["configuration"] = {
        "pck_thresholds_px": list(thresholds),
        "collapse_distance_px": collapse_distance_px,
        "topology_tolerance_px": topology_tolerance_px,
        "outlier_threshold_px": outlier_threshold_px,
        "ece_bin_count": ece_bin_count,
        "semantic_matching": "exact_name_only",
        "missing_prediction_pck_policy": "incorrect",
    }
    return overall


def evaluate_raw_vs_structured_court_outputs(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 13,
) -> dict[str, Any]:
    """Compare raw and structured floor predictions using exact semantic identities.

    PCK includes every labeled point in its denominator, so a missing prediction is a miss.
    Pixel-error deltas use only points available from both methods. Bootstrap confidence intervals
    resample complete samples to preserve the within-frame correlation among court points.
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of mappings")
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples < 0
    ):
        raise ValueError("bootstrap_resamples must be a nonnegative integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("bootstrap_seed must be an integer")

    normalized_pairs = [_normalize_paired_record(record, index=index) for index, record in enumerate(records)]
    raw_records = [_method_record(record, "raw_prediction") for record in normalized_pairs]
    structured_records = [_method_record(record, "structured_prediction") for record in normalized_pairs]
    raw = evaluate_structured_court_outputs(raw_records)
    structured = evaluate_structured_court_outputs(structured_records)
    paired = _paired_delta_summary(
        normalized_pairs,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )

    strata: dict[str, Any] = {}
    for dimension in ("subgroup", "source", "source_group", "viewpoint", "visibility"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in normalized_pairs:
            value = record["strata"].get(dimension)
            if value is not None:
                grouped[value].append(record)
        strata[dimension] = {
            value: {
                "sample_count": len(group),
                "raw": evaluate_structured_court_outputs(
                    [_method_record(record, "raw_prediction") for record in group]
                )["point_metrics"],
                "structured": evaluate_structured_court_outputs(
                    [_method_record(record, "structured_prediction") for record in group]
                )["point_metrics"],
                "paired_point_estimates": _paired_delta_summary(
                    group,
                    bootstrap_resamples=0,
                    bootstrap_seed=bootstrap_seed,
                )["point_estimates"],
            }
            for value, group in sorted(grouped.items())
        }

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_raw_vs_structured_exact_semantic_metrics",
        "sample_count": len(normalized_pairs),
        "evaluated_taxonomy": "canonical_floor_points_exact_semantic_name",
        "raw": raw,
        "structured": structured,
        "paired_deltas": paired,
        "strata": strata,
        "task88_policy": "historical_development_only_not_fold_or_promotion_evidence",
    }


def _normalize_paired_record(record: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(f"record {index} must be a mapping")
    raw_prediction = record.get("raw_prediction")
    structured_prediction = record.get("structured_prediction")
    if not isinstance(raw_prediction, Mapping) or not isinstance(structured_prediction, Mapping):
        raise ValueError(
            f"record {index} requires raw_prediction and structured_prediction mappings"
        )

    base = {
        key: value
        for key, value in record.items()
        if key not in {"raw_prediction", "structured_prediction"}
    }
    raw = _normalize_record({**base, "prediction": raw_prediction}, index=index)
    structured = _normalize_record({**base, "prediction": structured_prediction}, index=index)
    if raw["ground_truth"] != structured["ground_truth"]:
        raise AssertionError("paired normalization changed ground truth")
    return {
        "sample_id": raw["sample_id"],
        "viewpoint": raw["viewpoint"],
        "strata": raw["strata"],
        "ground_truth": raw["ground_truth"],
        "raw_prediction": {
            "keypoints": raw["keypoints"],
            "confidences": raw["confidences"],
            "whole_court_confidence": raw["whole_court_confidence"],
            "ignored_observations": raw["ignored_observations"],
        },
        "structured_prediction": {
            "keypoints": structured["keypoints"],
            "confidences": structured["confidences"],
            "whole_court_confidence": structured["whole_court_confidence"],
            "ignored_observations": structured["ignored_observations"],
        },
    }


def _method_record(record: Mapping[str, Any], method: str) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "viewpoint": record["viewpoint"],
        "strata": record["strata"],
        "ground_truth": record["ground_truth"],
        "prediction": record[method],
    }


def _paired_point_estimates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    thresholds = (2.0, 5.0, 10.0)
    labeled_count = 0
    hit_deltas = [0 for _threshold in thresholds]
    error_reductions: list[float] = []
    for record in records:
        ground_truth = record["ground_truth"]
        raw = record["raw_prediction"]["keypoints"]
        structured = record["structured_prediction"]["keypoints"]
        labeled_count += len(ground_truth)
        for name, gt_xy in ground_truth.items():
            raw_error = math.dist(gt_xy, raw[name]) if name in raw else None
            structured_error = math.dist(gt_xy, structured[name]) if name in structured else None
            for index, threshold in enumerate(thresholds):
                raw_hit = raw_error is not None and raw_error <= threshold
                structured_hit = structured_error is not None and structured_error <= threshold
                hit_deltas[index] += int(structured_hit) - int(raw_hit)
            if raw_error is not None and structured_error is not None:
                error_reductions.append(raw_error - structured_error)
    return {
        "labeled_count": labeled_count,
        "paired_error_count": len(error_reductions),
        **{
            f"{_pck_key(threshold)}_structured_minus_raw": hit_deltas[index] / labeled_count
            if labeled_count
            else None
            for index, threshold in enumerate(thresholds)
        },
        "mean_paired_error_reduction_px": _mean_or_none(error_reductions),
        "median_paired_error_reduction_px": _percentile_or_none(error_reductions, 0.5),
    }


def _paired_delta_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    point_estimates = _paired_point_estimates(records)
    metric_names = [
        "pck_at_2px_structured_minus_raw",
        "pck_at_5px_structured_minus_raw",
        "pck_at_10px_structured_minus_raw",
        "mean_paired_error_reduction_px",
        "median_paired_error_reduction_px",
    ]
    sampled_values: dict[str, list[float]] = {name: [] for name in metric_names}
    if records and bootstrap_resamples:
        rng = random.Random(bootstrap_seed)
        for _iteration in range(bootstrap_resamples):
            sample = [records[rng.randrange(len(records))] for _row in records]
            estimates = _paired_point_estimates(sample)
            for name in metric_names:
                value = estimates[name]
                if value is not None:
                    sampled_values[name].append(float(value))
    intervals = {
        name: {
            "available": bool(values),
            "lower_95": _percentile_or_none(values, 0.025),
            "upper_95": _percentile_or_none(values, 0.975),
        }
        for name, values in sampled_values.items()
    }
    return {
        "available": bool(records),
        "pairing": "same_sample_same_semantic_name",
        "bootstrap_unit": "sample",
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "direction": {
            "pck": "positive_means_structured_better",
            "error_reduction": "positive_means_structured_lower_error",
        },
        "point_estimates": point_estimates,
        "bootstrap_95_intervals": intervals,
    }


def _normalize_record(record: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(f"record {index} must be a mapping")
    ground_truth = record.get("ground_truth")
    prediction = record.get("prediction")
    if not isinstance(ground_truth, Mapping):
        raise ValueError(f"record {index} requires ground_truth mapping")
    if not isinstance(prediction, Mapping):
        raise ValueError(f"record {index} requires prediction mapping")
    keypoints = prediction.get("keypoints")
    if not isinstance(keypoints, Mapping):
        raise ValueError(f"record {index} prediction requires keypoints mapping")
    sample_id = record.get("sample_id", str(index))
    viewpoint = record.get("viewpoint", "unknown")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"record {index} sample_id must be a non-empty string")
    if not isinstance(viewpoint, str) or not viewpoint:
        raise ValueError(f"record {index} viewpoint must be a non-empty string")

    raw_strata = record.get("strata", {})
    if not isinstance(raw_strata, Mapping):
        raise ValueError(f"record {index} strata must be a mapping")
    normalized_strata: dict[str, str] = {}
    for dimension in ("subgroup", "source", "source_group", "visibility"):
        value = raw_strata.get(dimension, record.get(dimension))
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"record {index} {dimension} must be a non-empty string")
        normalized_strata[dimension] = value
    normalized_strata["viewpoint"] = viewpoint

    normalized_gt = {str(name): _xy(value, f"ground_truth.{name}") for name, value in ground_truth.items() if value is not None}
    normalized_prediction = {
        str(name): _xy(value, f"prediction.keypoints.{name}")
        for name, value in keypoints.items()
        if value is not None
    }
    confidences = prediction.get("confidences", {})
    if not isinstance(confidences, Mapping):
        raise ValueError(f"record {index} prediction.confidences must be a mapping")
    normalized_confidences = {
        str(name): _probability(value, f"prediction.confidences.{name}")
        for name, value in confidences.items()
    }
    whole_court_confidence = prediction.get("whole_court_confidence")
    if whole_court_confidence is not None:
        whole_court_confidence = _probability(
            whole_court_confidence, "prediction.whole_court_confidence"
        )
    ignored = prediction.get("ignored_observations", [])
    if isinstance(ignored, (str, bytes)) or not isinstance(ignored, Sequence):
        raise ValueError(f"record {index} prediction.ignored_observations must be a sequence")
    return {
        "sample_id": sample_id,
        "viewpoint": viewpoint,
        "strata": normalized_strata,
        "ground_truth": normalized_gt,
        "keypoints": normalized_prediction,
        "confidences": normalized_confidences,
        "whole_court_confidence": whole_court_confidence,
        "ignored_observations": list(ignored),
    }


def _summarize_records(
    records: Sequence[dict[str, Any]],
    *,
    thresholds: tuple[float, float],
    collapse_distance_px: float,
    topology_tolerance_px: float,
    outlier_threshold_px: float,
    ece_bin_count: int,
    include_samples: bool,
) -> dict[str, Any]:
    all_errors: list[float] = []
    point_denominator = 0
    missing_predictions = 0
    unexpected_predictions = 0
    threshold_hits = [0 for _threshold in thresholds]
    errors_by_name: dict[str, list[float]] = defaultdict(list)
    denominators_by_name: CounterLike = defaultdict(int)
    missing_by_name: CounterLike = defaultdict(int)
    hits_by_name: dict[str, list[int]] = defaultdict(lambda: [0 for _threshold in thresholds])
    point_calibration_pairs: list[tuple[float, bool]] = []
    whole_court_pairs: list[tuple[float, bool]] = []
    sample_outputs: list[dict[str, Any]] = []
    collapse_pair_count = 0
    samples_with_collapse = 0
    self_intersection_count = 0
    boundary_available_count = 0
    topology_available_count = 0
    topology_valid_count = 0
    template_available_count = 0
    template_valid_count = 0
    ignored_with_error = 0
    recovered_outliers = 0
    ignored_inliers = 0

    for record in records:
        gt = record["ground_truth"]
        predicted = record["keypoints"]
        point_denominator += len(gt)
        unexpected_predictions += len(set(predicted) - set(gt))
        sample_errors: dict[str, float | None] = {}
        for name, gt_xy in sorted(gt.items()):
            denominators_by_name[name] += 1
            pred_xy = predicted.get(name)
            error = None if pred_xy is None else math.dist(gt_xy, pred_xy)
            sample_errors[name] = error
            if error is None:
                missing_predictions += 1
                missing_by_name[name] += 1
            else:
                all_errors.append(error)
                errors_by_name[name].append(error)
                for index, threshold in enumerate(thresholds):
                    if error <= threshold:
                        threshold_hits[index] += 1
                        hits_by_name[name][index] += 1
            if name in record["confidences"]:
                point_calibration_pairs.append(
                    (record["confidences"][name], error is not None and error <= 5.0)
                )

        collapse_pairs = _collapse_pairs(predicted, threshold_px=collapse_distance_px)
        collapse_pair_count += len(collapse_pairs)
        samples_with_collapse += int(bool(collapse_pairs))
        topology = _topology_metrics(
            predicted,
            collapse_pairs=collapse_pairs,
            tolerance_px=topology_tolerance_px,
        )
        if topology["boundary_polygon_available"]:
            boundary_available_count += 1
            self_intersection_count += int(topology["boundary_polygon_self_intersects"])
        if topology["topology_available"]:
            topology_available_count += 1
            topology_valid_count += int(topology["topology_valid"])
        if topology["template_invariant_available"]:
            template_available_count += 1
            template_valid_count += int(topology["template_invariant_valid"])

        recovered = _score_ignored_observations(
            record["ignored_observations"], gt=gt, threshold_px=outlier_threshold_px
        )
        ignored_with_error += recovered["ignored_observations_with_gt_error"]
        recovered_outliers += recovered["recovered_outlier_count"]
        ignored_inliers += recovered["ignored_inlier_count"]

        all_within_5 = bool(gt) and all(
            sample_errors[name] is not None and float(sample_errors[name]) <= 5.0 for name in gt
        )
        topology_ok = not topology["topology_available"] or topology["topology_valid"]
        whole_court_correct = all_within_5 and topology_ok
        if record["whole_court_confidence"] is not None:
            whole_court_pairs.append((record["whole_court_confidence"], whole_court_correct))

        if include_samples:
            sample_outputs.append(
                {
                    "sample_id": record["sample_id"],
                    "viewpoint": record["viewpoint"],
                    "strata": dict(record["strata"]),
                    "point_errors_px": sample_errors,
                    "point_confidence": {
                        name: record["confidences"][name]
                        for name in sorted(gt)
                        if name in record["confidences"]
                    },
                    "whole_court_confidence": record["whole_court_confidence"],
                    "missing_prediction_names": sorted(set(gt) - set(predicted)),
                    "unexpected_prediction_names": sorted(set(predicted) - set(gt)),
                    "collapse_pairs": [list(pair) for pair in collapse_pairs],
                    "topology": topology,
                    "outlier_recovery": recovered,
                    "whole_court_within_5px_and_topology_valid": whole_court_correct,
                }
            )

    per_name: dict[str, Any] = {}
    for name in sorted(denominators_by_name):
        denominator = denominators_by_name[name]
        errors = errors_by_name[name]
        per_name[name] = {
            "labeled_count": denominator,
            "matched_count": len(errors),
            "missing_prediction_count": missing_by_name[name],
            "mean_error_px": _mean_or_none(errors),
            "median_error_px": _percentile_or_none(errors, 0.5),
            "p90_error_px": _percentile_or_none(errors, 0.9),
            "p95_error_px": _percentile_or_none(errors, 0.95),
            "max_error_px": max(errors) if errors else None,
            **{
                _pck_key(threshold): hits_by_name[name][index] / denominator
                for index, threshold in enumerate(thresholds)
            },
        }

    result = {
        "sample_count": len(records),
        "point_metrics": {
            "labeled_count": point_denominator,
            "matched_count": len(all_errors),
            "missing_prediction_count": missing_predictions,
            "unexpected_prediction_name_count": unexpected_predictions,
            "mean_error_px": _mean_or_none(all_errors),
            "median_error_px": _percentile_or_none(all_errors, 0.5),
            "p90_error_px": _percentile_or_none(all_errors, 0.9),
            "p95_error_px": _percentile_or_none(all_errors, 0.95),
            "max_error_px": max(all_errors) if all_errors else None,
            **{
                _pck_key(threshold): threshold_hits[index] / point_denominator
                if point_denominator
                else None
                for index, threshold in enumerate(thresholds)
            },
            "per_semantic_name": per_name,
        },
        "structure": {
            "duplicate_or_collapse_pair_count": collapse_pair_count,
            "samples_with_duplicate_or_collapse_count": samples_with_collapse,
            "boundary_polygon_available_count": boundary_available_count,
            "topology_available_count": topology_available_count,
            "boundary_polygon_self_intersection_count": self_intersection_count,
            "topology_valid_count": topology_valid_count,
            "topology_valid_rate": topology_valid_count / topology_available_count
            if topology_available_count
            else None,
            "centerline_nvz_template_available_count": template_available_count,
            "centerline_nvz_template_valid_count": template_valid_count,
            "centerline_nvz_template_valid_rate": template_valid_count / template_available_count
            if template_available_count
            else None,
        },
        "outlier_recovery": {
            "available": ignored_with_error > 0,
            "ignored_observations_with_gt_error": ignored_with_error,
            "recovered_outlier_count": recovered_outliers,
            "ignored_inlier_count": ignored_inliers,
            "recovery_precision": recovered_outliers / ignored_with_error if ignored_with_error else None,
        },
        "point_confidence_calibration": _calibration_metrics(
            point_calibration_pairs, bin_count=ece_bin_count
        ),
        "whole_court_calibration": _calibration_metrics(
            whole_court_pairs, bin_count=ece_bin_count
        ),
    }
    if include_samples:
        result["samples"] = sample_outputs
    return result


CounterLike = dict[str, int]


def _topology_metrics(
    points: Mapping[str, tuple[float, float]],
    *,
    collapse_pairs: Sequence[tuple[str, str]],
    tolerance_px: float,
) -> dict[str, Any]:
    boundary_available = all(name in points for name in BOUNDARY_POLYGON)
    self_intersects = False
    boundary_area_px2: float | None = None
    boundary_degenerate = False
    if boundary_available:
        polygon = [points[name] for name in BOUNDARY_POLYGON]
        self_intersects = _segments_intersect(polygon[0], polygon[1], polygon[2], polygon[3]) or _segments_intersect(
            polygon[1], polygon[2], polygon[3], polygon[0]
        )
        boundary_area_px2 = abs(_signed_polygon_area(polygon))
        boundary_degenerate = boundary_area_px2 <= max(1e-9, tolerance_px * tolerance_px)

    template_names = {
        name for triplet in TEMPLATE_ROW_TRIPLETS for name in triplet
    } | {name for _line, ordered in TEMPLATE_ORDERED_LINES for name in ordered}
    template_available = template_names.issubset(points)
    invariant_checks: dict[str, bool] = {}
    if template_available:
        for left, center, right in TEMPLATE_ROW_TRIPLETS:
            invariant_checks[f"{center}_between_{left}_and_{right}"] = _point_between_on_line(
                points[left], points[center], points[right], tolerance_px=tolerance_px
            )
        for line_name, ordered_names in TEMPLATE_ORDERED_LINES:
            ordered_points = [points[name] for name in ordered_names]
            invariant_checks[f"{line_name}_ordered_near_to_far"] = _ordered_on_line(
                ordered_points, tolerance_px=tolerance_px
            )
    template_valid = template_available and all(invariant_checks.values())
    collapsed_template_names = any(
        left in template_names and right in template_names for left, right in collapse_pairs
    )
    topology_available = boundary_available and template_available
    topology_valid = (
        topology_available
        and not self_intersects
        and not boundary_degenerate
        and template_valid
        and not collapsed_template_names
    )
    return {
        "boundary_polygon_available": boundary_available,
        "boundary_polygon_self_intersects": self_intersects if boundary_available else None,
        "boundary_polygon_area_px2": boundary_area_px2,
        "boundary_polygon_degenerate": boundary_degenerate if boundary_available else None,
        "template_invariant_available": template_available,
        "template_invariant_valid": template_valid if template_available else None,
        "template_invariant_checks": invariant_checks,
        "topology_available": topology_available,
        "topology_valid": topology_valid if topology_available else None,
    }


def _collapse_pairs(
    points: Mapping[str, tuple[float, float]], *, threshold_px: float
) -> list[tuple[str, str]]:
    names = sorted(points)
    return [
        (left, right)
        for index, left in enumerate(names)
        for right in names[index + 1 :]
        if math.dist(points[left], points[right]) <= threshold_px
    ]


def _point_between_on_line(
    start: tuple[float, float],
    middle: tuple[float, float],
    end: tuple[float, float],
    *,
    tolerance_px: float,
) -> bool:
    length2 = _squared_distance(start, end)
    if length2 <= 1e-12:
        return False
    parameter = _projection_parameter(start, end, middle)
    return -1e-9 <= parameter <= 1.0 + 1e-9 and _point_line_distance(start, end, middle) <= tolerance_px


def _ordered_on_line(points: Sequence[tuple[float, float]], *, tolerance_px: float) -> bool:
    if len(points) < 2 or _squared_distance(points[0], points[-1]) <= 1e-12:
        return False
    parameters = [_projection_parameter(points[0], points[-1], point) for point in points]
    return (
        all(_point_line_distance(points[0], points[-1], point) <= tolerance_px for point in points)
        and all(left + 1e-9 < right for left, right in zip(parameters, parameters[1:], strict=False))
        and parameters[0] >= -1e-9
        and parameters[-1] <= 1.0 + 1e-9
    )


def _score_ignored_observations(
    observations: Sequence[Any],
    *,
    gt: Mapping[str, tuple[float, float]],
    threshold_px: float,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    unavailable = 0
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise ValueError(f"ignored observation {index} must be a mapping")
        name = observation.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"ignored observation {index} requires a semantic name")
        error = observation.get("gt_error_px")
        if error is not None:
            error = _nonnegative_float(error, f"ignored_observations[{index}].gt_error_px")
        elif name in gt and observation.get("xy") is not None:
            error = math.dist(gt[name], _xy(observation["xy"], f"ignored_observations[{index}].xy"))
        else:
            unavailable += 1
            continue
        errors.append({"name": name, "gt_error_px": error, "is_true_outlier": error > threshold_px})
    recovered = sum(row["is_true_outlier"] for row in errors)
    return {
        "ignored_observation_count": len(observations),
        "ignored_observations_with_gt_error": len(errors),
        "ignored_observations_without_gt_error": unavailable,
        "recovered_outlier_count": recovered,
        "ignored_inlier_count": len(errors) - recovered,
        "observations": errors,
    }


def _calibration_metrics(pairs: Sequence[tuple[float, bool]], *, bin_count: int) -> dict[str, Any]:
    if not pairs:
        return {
            "available": False,
            "count": 0,
            "brier_score": None,
            "ece": None,
            "bins": [],
        }
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(bin_count)]
    for confidence, event in pairs:
        bin_index = min(bin_count - 1, int(confidence * bin_count))
        bins[bin_index].append((confidence, event))
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for index, values in enumerate(bins):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        if values:
            mean_confidence = sum(confidence for confidence, _event in values) / len(values)
            accuracy = sum(event for _confidence, event in values) / len(values)
            gap = abs(mean_confidence - accuracy)
            contribution = len(values) / len(pairs) * gap
            ece += contribution
        else:
            mean_confidence = None
            accuracy = None
            gap = None
            contribution = 0.0
        rows.append(
            {
                "bin_index": index,
                "lower_inclusive": lower,
                "upper_inclusive": upper if index == bin_count - 1 else None,
                "upper_exclusive": None if index == bin_count - 1 else upper,
                "count": len(values),
                "mean_confidence": mean_confidence,
                "empirical_accuracy": accuracy,
                "absolute_gap": gap,
                "ece_contribution": contribution,
            }
        )
    return {
        "available": True,
        "count": len(pairs),
        "brier_score": sum((confidence - float(event)) ** 2 for confidence, event in pairs) / len(pairs),
        "ece": ece,
        "bins": rows,
    }


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    def orientation(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    values = (orientation(a, b, c), orientation(a, b, d), orientation(c, d, a), orientation(c, d, b))
    epsilon = 1e-9
    if values[0] * values[1] < -epsilon and values[2] * values[3] < -epsilon:
        return True
    return (
        abs(values[0]) <= epsilon and _point_in_bbox(c, a, b)
        or abs(values[1]) <= epsilon and _point_in_bbox(d, a, b)
        or abs(values[2]) <= epsilon and _point_in_bbox(a, c, d)
        or abs(values[3]) <= epsilon and _point_in_bbox(b, c, d)
    )


def _point_in_bbox(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> bool:
    epsilon = 1e-9
    return (
        min(start[0], end[0]) - epsilon <= point[0] <= max(start[0], end[0]) + epsilon
        and min(start[1], end[1]) - epsilon <= point[1] <= max(start[1], end[1]) + epsilon
    )


def _signed_polygon_area(points: Sequence[tuple[float, float]]) -> float:
    return 0.5 * sum(
        current[0] * following[1] - following[0] * current[1]
        for current, following in zip(points, (*points[1:], points[0]), strict=True)
    )


def _projection_parameter(
    start: tuple[float, float], end: tuple[float, float], point: tuple[float, float]
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    return ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (dx * dx + dy * dy)


def _point_line_distance(
    start: tuple[float, float], end: tuple[float, float], point: tuple[float, float]
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    return abs(dx * (start[1] - point[1]) - (start[0] - point[0]) * dy) / math.hypot(dx, dy)


def _squared_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2


def _xy(value: Any, field: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{field} must be a two-item coordinate")
    return (_finite_float(value[0], field), _finite_float(value[1], field))


def _probability(value: Any, field: str) -> float:
    result = _finite_float(value, field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return result


def _positive_float(value: Any, field: str) -> float:
    result = _finite_float(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _nonnegative_float(value: Any, field: str) -> float:
    result = _finite_float(value, field)
    if result < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pck_key(threshold: float) -> str:
    value = str(int(threshold)) if float(threshold).is_integer() else str(threshold).replace(".", "_")
    return f"pck_at_{value}px"


def _percentile_or_none(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
