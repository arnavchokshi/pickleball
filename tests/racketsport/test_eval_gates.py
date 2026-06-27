from __future__ import annotations

import pytest

from threed.racketsport.eval.metrics import NumericGate, evaluate_numeric_gates, metric


def test_evaluate_numeric_gates_marks_passes_and_failures_from_values() -> None:
    gated = evaluate_numeric_gates(
        {"reprojection_median_px": 4.9, "coverage_overall": 0.72},
        {
            "reprojection_median_px": NumericGate(name="calibration_median_px", op="<", threshold=5.0, unit="px"),
            "coverage_overall": NumericGate(name="habit_coverage_min", op=">=", threshold=0.8, unit="ratio"),
        },
    )

    assert gated["reprojection_median_px"].value == 4.9
    assert gated["reprojection_median_px"].gate == "calibration_median_px: < 5.0"
    assert gated["reprojection_median_px"].passed is True
    assert gated["coverage_overall"].value == 0.72
    assert gated["coverage_overall"].gate == "habit_coverage_min: >= 0.8"
    assert gated["coverage_overall"].passed is False


def test_evaluate_numeric_gates_accepts_eval_metrics_and_value_dicts() -> None:
    gated = evaluate_numeric_gates(
        {
            "players_detected": metric(value=2, unit="players", gate=">= 1", passed=True),
            "track_frames": {"value": 12},
        },
        {
            "players_detected": NumericGate(name="min_players", op=">=", threshold=1, unit="players"),
            "track_frames": NumericGate(name="min_track_frames", op=">=", threshold=10, unit="frames"),
        },
    )

    assert gated["players_detected"].value == 2
    assert gated["players_detected"].unit == "players"
    assert gated["players_detected"].passed is True
    assert gated["track_frames"].value == 12
    assert gated["track_frames"].unit == "frames"
    assert gated["track_frames"].passed is True


def test_evaluate_numeric_gates_marks_missing_values_not_measured() -> None:
    gated = evaluate_numeric_gates(
        {"contact_events": None},
        {
            "contact_events": NumericGate(name="min_contacts", op=">=", threshold=1, unit="events"),
            "bounce_events": NumericGate(name="min_bounces", op=">=", threshold=1, unit="events"),
        },
    )

    assert gated["contact_events"].value is None
    assert gated["contact_events"].status == "not_measured"
    assert gated["contact_events"].passed is None
    assert gated["bounce_events"].value is None
    assert gated["bounce_events"].status == "not_measured"
    assert gated["bounce_events"].passed is None


def test_numeric_gate_rejects_unknown_operator() -> None:
    with pytest.raises(ValueError, match="unsupported numeric gate operator"):
        NumericGate(name="bad_gate", op="!=", threshold=1)
