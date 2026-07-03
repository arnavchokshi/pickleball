from __future__ import annotations

from threed.racketsport.court_detector_v2_hypotheses import (
    FLOOR_KEYPOINT_NAMES,
    NET_TOP_KEYPOINT_NAMES,
    generate_court_hypotheses,
)


def test_floor_hypotheses_exclude_top_net_keypoints() -> None:
    assert "net_left_sideline" in NET_TOP_KEYPOINT_NAMES
    assert "net_left_sideline" not in FLOOR_KEYPOINT_NAMES


def test_hypothesis_generation_never_uses_net_top_as_floor_points() -> None:
    hypotheses = generate_court_hypotheses(
        image_size=(1080, 1920),
        net_evidence={"confidence": 0.5, "top_tape_line": [[100, 900], [980, 900]]},
        surface_evidence={"semantic_line_candidates": []},
    )

    assert len(hypotheses) >= 1
    for hypothesis in hypotheses:
        assert not set(hypothesis.get("floor_correspondence_names", [])) & NET_TOP_KEYPOINT_NAMES
        assert hypothesis["promotion_allowed"] is False


def test_hypothesis_generation_uses_surface_line_support() -> None:
    hypotheses = generate_court_hypotheses(
        image_size=(1080, 1920),
        net_evidence={"confidence": 0.5, "top_tape_line": [[100, 900], [980, 900]]},
        surface_evidence={
            "semantic_line_candidates": [
                {"p1": [0, 100], "p2": [100, 100]},
                {"p1": [0, 200], "p2": [100, 200]},
                {"p1": [10, 0], "p2": [10, 300]},
                {"p1": [90, 0], "p2": [90, 300]},
            ]
        },
    )

    kitchen = next(h for h in hypotheses if h["source"] == "kitchen_first_line_seed")
    assert kitchen["line_support"]["required_lines_present"] is True
    assert kitchen["line_support"]["semantic_line_count"] == 4
