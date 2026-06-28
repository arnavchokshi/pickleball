from __future__ import annotations

import pytest

from threed.racketsport.court_line_evidence import (
    aggregate_court_line_evidence,
    score_line_candidate,
    select_best_line_observation,
)
from threed.racketsport.schemas import CourtLineObservation, NetLineObservation


def test_score_line_candidate_prefers_aligned_over_parallel_distractor():
    expected = ((100.0, 300.0), (900.0, 300.0))
    painted_line = ((120.0, 303.0), (880.0, 302.0))
    parallel_distractor = ((120.0, 260.0), (880.0, 260.0))

    good = score_line_candidate(expected, painted_line)
    distractor = score_line_candidate(expected, parallel_distractor)

    assert good.distance_px < 4.0
    assert good.visible_fraction > 0.9
    assert good.score > distractor.score
    assert good.confidence > distractor.confidence


def test_select_best_line_observation_builds_semantic_schema_entry():
    observation = select_best_line_observation(
        line_id="near_nvz",
        expected_segment=((100.0, 300.0), (900.0, 300.0)),
        candidate_segments=[
            ((120.0, 260.0), (880.0, 260.0)),
            ((120.0, 303.0), (880.0, 302.0)),
        ],
        frame_indexes=[10, 11, 12],
        source="hough_template",
    )

    assert isinstance(observation, CourtLineObservation)
    assert observation.line_id == "near_nvz"
    assert observation.confidence > 0.75
    assert observation.visible_fraction > 0.9
    assert observation.frame_indexes == [10, 11, 12]
    assert observation.image_segment[0] == pytest.approx([120.0, 303.0])
    assert observation.image_segment[1] == pytest.approx([880.0, 302.0])


def test_select_best_line_observation_rejects_parallel_non_overlapping_candidate():
    observation = select_best_line_observation(
        line_id="near_nvz",
        expected_segment=((100.0, 300.0), (900.0, 300.0)),
        candidate_segments=[((1100.0, 300.0), (1300.0, 300.0))],
        frame_indexes=[10],
        source="hough_template",
    )

    assert observation is None


def test_aggregate_requires_centerlines_and_top_net_for_no_tap_readiness():
    near_nvz = select_best_line_observation(
        line_id="near_nvz",
        expected_segment=((100.0, 300.0), (900.0, 300.0)),
        candidate_segments=[((120.0, 302.0), (880.0, 302.0))],
        frame_indexes=[4, 5, 6],
    )
    far_nvz = select_best_line_observation(
        line_id="far_nvz",
        expected_segment=((130.0, 180.0), (870.0, 180.0)),
        candidate_segments=[((140.0, 181.0), (860.0, 182.0))],
        frame_indexes=[4, 5, 6],
    )

    evidence = aggregate_court_line_evidence(
        sport="pickleball",
        line_observations=[near_nvz, far_nvz],
        required_line_ids=["near_nvz", "far_nvz", "near_centerline", "far_centerline"],
        required_net_ids=["top_net"],
    )

    assert evidence.aggregate.auto_calibration_ready is False
    assert evidence.aggregate.missing_required_line_ids == ["near_centerline", "far_centerline"]
    assert evidence.aggregate.missing_required_net_ids == ["top_net"]
    assert "missing_near_centerline" in evidence.aggregate.reasons
    assert "missing_top_net" in evidence.aggregate.reasons


def test_aggregate_accepts_centerlines_and_observed_top_net_for_no_tap_readiness():
    line_observations = [
        select_best_line_observation(
            line_id=line_id,
            expected_segment=expected,
            candidate_segments=[candidate],
            frame_indexes=[1, 2, 3],
        )
        for line_id, expected, candidate in [
            ("near_nvz", ((100.0, 300.0), (900.0, 300.0)), ((120.0, 302.0), (880.0, 302.0))),
            ("far_nvz", ((130.0, 180.0), (870.0, 180.0)), ((140.0, 181.0), (860.0, 182.0))),
            ("near_centerline", ((500.0, 300.0), (500.0, 700.0)), ((501.0, 320.0), (500.0, 680.0))),
            ("far_centerline", ((500.0, 180.0), (500.0, 30.0)), ((499.0, 170.0), (501.0, 40.0))),
        ]
    ]
    top_net = NetLineObservation(
        net_id="top_net",
        image_points=[[100.0, 240.0], [500.0, 238.0], [900.0, 240.0]],
        confidence=0.86,
        frame_indexes=[1, 2, 3],
        residual_px={"mean": 2.0, "p95": 3.0},
        source="net_top_roi",
    )

    evidence = aggregate_court_line_evidence(
        sport="pickleball",
        line_observations=line_observations,
        net_observations=[top_net],
        required_line_ids=["near_nvz", "far_nvz", "near_centerline", "far_centerline"],
        required_net_ids=["top_net"],
    )

    assert evidence.aggregate.auto_calibration_ready is True
    assert evidence.aggregate.missing_required_line_ids == []
    assert evidence.aggregate.missing_required_net_ids == []
