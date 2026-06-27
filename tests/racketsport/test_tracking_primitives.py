from __future__ import annotations

import pytest

from threed.racketsport.court_templates import FT_TO_M
from threed.racketsport.person_fast import PersonDetection, court_polygon_filter
from threed.racketsport.track_lock import TrackCandidate, ground_step_plausible, n_lock


def test_court_polygon_filter_rejects_off_court_people_before_tracking():
    detections = [
        PersonDetection(bbox_xyxy=(100.0, 100.0, 140.0, 260.0), confidence=0.92, foot_world_xy=[0.0, 0.0]),
        PersonDetection(
            bbox_xyxy=(500.0, 100.0, 540.0, 260.0),
            confidence=0.89,
            foot_world_xy=[12.0 * FT_TO_M, 0.0],
        ),
    ]

    filtered = court_polygon_filter(detections, sport="pickleball")

    assert filtered == [detections[0]]


def test_n_lock_returns_exactly_requested_candidate_count_by_confidence():
    candidates = [
        TrackCandidate(track_id=10, world_xy=[0.0, 0.0], confidence=0.4),
        TrackCandidate(track_id=11, world_xy=[1.0, 0.0], confidence=0.9),
        TrackCandidate(track_id=12, world_xy=[2.0, 0.0], confidence=0.7),
    ]

    locked = n_lock(candidates, count=2)

    assert [candidate.track_id for candidate in locked] == [11, 12]


def test_n_lock_fails_closed_when_not_enough_candidates():
    with pytest.raises(ValueError, match="need 2 candidates"):
        n_lock([TrackCandidate(track_id=1, world_xy=[0.0, 0.0], confidence=0.9)], count=2)


def test_ground_step_plausible_rejects_metric_teleports():
    assert ground_step_plausible([0.0, 0.0], [0.5, 0.0], max_step_m=1.0)
    assert not ground_step_plausible([0.0, 0.0], [2.5, 0.0], max_step_m=1.0)
