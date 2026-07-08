from __future__ import annotations

from threed.racketsport.court_proposals import (
    GEO_R3_IDENTITY_LINK_MEDIAN_PX,
    GEO_R3_TRIGGER_TEMPORAL_MEDIAN_PX,
    _geor3_select_identity_vote,
    _geor3_temporal_trigger_fires,
)


_POINT_NAMES = (
    "near_left_corner",
    "near_baseline_center",
    "near_right_corner",
    "near_nvz_left",
    "near_nvz_center",
    "near_nvz_right",
    "net_left_sideline",
    "net_center",
    "net_right_sideline",
    "far_nvz_left",
    "far_nvz_center",
    "far_nvz_right",
)


def _keypoints(dx: float, dy: float = 0.0) -> dict[str, tuple[float, float]]:
    return {name: (100.0 + index * 13.0 + dx, 200.0 + index * 7.0 + dy) for index, name in enumerate(_POINT_NAMES)}


def _hypothesis(hypothesis_id: str, *, dx: float, score: float) -> dict[str, object]:
    return {
        "hypothesis_id": hypothesis_id,
        "template": "pickleball",
        "keypoints": _keypoints(dx),
        "score": score,
        "evidence_score": 1.0 / (1.0 + score),
    }


def test_geor3_temporal_trigger_is_predeclared_and_strict() -> None:
    assert GEO_R3_TRIGGER_TEMPORAL_MEDIAN_PX == 24.0

    assert not _geor3_temporal_trigger_fires({"frame_count": 2, "median": 60.0})
    assert not _geor3_temporal_trigger_fires({"frame_count": 3, "median": 24.0})
    assert _geor3_temporal_trigger_fires({"frame_count": 3, "median": 24.001})


def test_geor3_identity_vote_selects_cross_frame_court_over_per_frame_adjacent_winners() -> None:
    frames = [
        {
            "frame_index": 10,
            "top_pickleball_hypotheses": [
                _hypothesis("adjacent_a", dx=360.0, score=1.0),
                _hypothesis("true_10", dx=0.0, score=2.0),
                _hypothesis("other_10", dx=720.0, score=3.0),
            ],
        },
        {
            "frame_index": 20,
            "top_pickleball_hypotheses": [
                _hypothesis("adjacent_b", dx=520.0, score=1.0),
                _hypothesis("true_20", dx=8.0, score=2.2),
                _hypothesis("other_20", dx=760.0, score=3.0),
            ],
        },
        {
            "frame_index": 30,
            "top_pickleball_hypotheses": [
                _hypothesis("adjacent_c", dx=680.0, score=1.0),
                _hypothesis("other_30", dx=760.0, score=2.0),
                _hypothesis("true_30", dx=16.0, score=2.4),
            ],
        },
    ]

    result = _geor3_select_identity_vote(frames, identity_link_median_px=GEO_R3_IDENTITY_LINK_MEDIAN_PX)

    assert result["selected"] is True
    assert result["support_frame_count"] == 3
    assert result["selected_hypothesis_ids"] == ["true_10", "true_20", "true_30"]
    assert [item["best"]["hypothesis_id"] for item in result["selected_frames"]] == ["true_10", "true_20", "true_30"]


def test_geor3_identity_vote_reports_unresolved_without_majority_support() -> None:
    frames = [
        {"frame_index": 1, "top_pickleball_hypotheses": [_hypothesis("court_a", dx=0.0, score=1.0)]},
        {"frame_index": 2, "top_pickleball_hypotheses": [_hypothesis("court_b", dx=240.0, score=1.0)]},
        {"frame_index": 3, "top_pickleball_hypotheses": [_hypothesis("court_c", dx=480.0, score=1.0)]},
    ]

    result = _geor3_select_identity_vote(frames)

    assert result["selected"] is False
    assert result["support_frame_count"] == 1
    assert result["selected_frames"] == []
