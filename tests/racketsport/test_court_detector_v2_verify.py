from __future__ import annotations

from threed.racketsport.court_detector_v2_verify import verify_court_hypothesis


def test_verifier_blocks_large_visible_floor_residuals() -> None:
    result = verify_court_hypothesis(
        hypothesis={"hypothesis_id": "bad", "promotion_allowed": False},
        visible_error_px={
            "floor_visible": {"median": 581.0, "p95": 786.0},
            "visible_corners": {"median": 604.0},
        },
        line_support={"required_lines_present": False},
        temporal_stability_px={"median": 0.0},
    )

    assert result["promotion_allowed"] is False
    assert "visible_floor_median_gt_15" in result["blockers"]
    assert "visible_floor_p95_gt_30" in result["blockers"]
    assert "required_line_support_missing" in result["blockers"]


def test_verifier_promotes_clean_visible_residuals() -> None:
    result = verify_court_hypothesis(
        hypothesis={"hypothesis_id": "good", "promotion_allowed": False},
        visible_error_px={
            "floor_visible": {"median": 4.0, "p95": 12.0},
            "visible_corners": {"median": 8.0},
            "high_confidence_over_30px_count": 0,
        },
        line_support={"required_lines_present": True},
        temporal_stability_px={"median": 2.0},
        top_net_validation={"passed": True},
        tennis_overlay_rejection={"passed": True},
    )

    assert result["promotion_allowed"] is True
    assert result["blockers"] == []
