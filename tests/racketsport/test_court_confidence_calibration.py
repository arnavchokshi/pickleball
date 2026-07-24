from __future__ import annotations

import pytest

from threed.racketsport.court_confidence_calibration import (
    IsotonicConfidenceCalibrator,
    TemperatureConfidenceCalibrator,
    apply_point_calibration,
    confidence_calibration_report,
    fit_isotonic_confidence,
    fit_temperature_confidence,
    select_zero_false_accept_threshold,
)


def test_isotonic_fit_repairs_nonmonotone_empirical_accuracy() -> None:
    calibrator = fit_isotonic_confidence(
        [0.1, 0.2, 0.3, 0.4, 0.8, 0.9],
        [0, 1, 0, 1, 1, 1],
    )
    assert list(calibrator.probabilities) == sorted(calibrator.probabilities)
    assert calibrator.predict(0.05) <= calibrator.predict(0.5) <= calibrator.predict(0.95)


def test_isotonic_round_trip_is_exact() -> None:
    fitted = fit_isotonic_confidence([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    loaded = IsotonicConfidenceCalibrator.from_dict(fitted.to_dict())
    assert loaded == fitted


def test_apply_calibration_preserves_semantic_names() -> None:
    fitted = fit_isotonic_confidence([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    result = apply_point_calibration({"near_left_corner": 0.15, "far_right_corner": 0.85}, fitted)
    assert set(result) == {"near_left_corner", "far_right_corner"}
    assert result["near_left_corner"] <= result["far_right_corner"]


def test_invalid_serialized_calibration_is_rejected() -> None:
    with pytest.raises(ValueError, match="nondecreasing"):
        IsotonicConfidenceCalibrator.from_dict(
            {
                "calibrator": "isotonic_pav",
                "upper_bounds": [0.2, 0.8],
                "probabilities": [0.9, 0.1],
                "sample_count": 2,
                "positive_count": 1,
            }
        )


def test_temperature_scaling_round_trip_and_probability_order() -> None:
    fitted = fit_temperature_confidence([-3.0, -1.0, 1.0, 3.0], [0, 0, 1, 1])
    loaded = TemperatureConfidenceCalibrator.from_dict(fitted.to_dict())

    assert loaded == fitted
    assert loaded.temperature > 0.0
    assert loaded.predict_probability(0.1) < loaded.predict_probability(0.9)


def test_calibration_report_and_zero_false_accept_threshold() -> None:
    report = confidence_calibration_report([0.1, 0.3, 0.8, 0.9], [0, 0, 1, 1], bin_count=4)
    threshold = select_zero_false_accept_threshold(
        [0.1, 0.3, 0.8, 0.9],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
    )

    assert report["sample_count"] == 4
    assert 0.0 <= report["brier_score"] <= 1.0
    assert 0.0 <= report["ece"] <= 1.0
    assert threshold == pytest.approx(0.8)
