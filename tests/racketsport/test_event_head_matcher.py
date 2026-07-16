from __future__ import annotations

from threed.racketsport.event_head.matcher import Event, event_metrics, greedy_match


def test_type_aware_greedy_match_is_one_to_one() -> None:
    predictions = [Event(9, 1, 0.9), Event(11, 1, 0.8), Event(10, 2, 0.99)]
    truth = [Event(10, 1)]
    result = greedy_match(predictions, truth, tolerance_frames=2)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 0)


def test_metrics_report_frame_and_ms_tolerance() -> None:
    metrics = event_metrics([Event(12, 2)], [Event(10, 2)], tolerance_frames=2, fps=25.0)
    assert metrics["per_class"]["BOUNCE"]["f1"] == 1.0
    assert metrics["tolerance_ms"] == 80.0
