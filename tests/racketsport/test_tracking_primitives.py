from __future__ import annotations

import pytest

from threed.racketsport.court_templates import FT_TO_M
from threed.racketsport.doubles_id import assign_doubles_roles, coach_anchor
from threed.racketsport.person_fast import PersonDetection, court_polygon_filter, person_detection_from_bbox
from threed.racketsport.schemas import CameraIntrinsics, CaptureQuality, CourtCalibration, CourtExtrinsics, ReprojectionError
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


def test_person_detection_from_bbox_uses_bottom_center_as_world_foot_point():
    calibration = CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[10.0, 0.0, 100.0], [0.0, 5.0, 200.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, dist=[], source="arkit"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 15.0],
            camera_height_m=15.0,
        ),
        reprojection_error_px=ReprojectionError(median=0.0, p95=0.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=[],
        world_pts=[],
    )

    detection = person_detection_from_bbox(calibration, bbox_xyxy=(105.0, 150.0, 115.0, 190.0), confidence=0.91)

    assert detection.bbox_xyxy == (105.0, 150.0, 115.0, 190.0)
    assert detection.confidence == pytest.approx(0.91)
    assert detection.foot_world_xy == pytest.approx([1.0, -2.0])
    assert court_polygon_filter([detection], sport="pickleball") == [detection]


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


def test_assign_doubles_roles_uses_court_side_and_lateral_position():
    candidates = [
        TrackCandidate(track_id=1, world_xy=[-2.0, -4.0], confidence=0.9),
        TrackCandidate(track_id=2, world_xy=[2.0, -4.0], confidence=0.9),
        TrackCandidate(track_id=3, world_xy=[-2.0, 4.0], confidence=0.9),
        TrackCandidate(track_id=4, world_xy=[2.0, 4.0], confidence=0.9),
    ]

    roles = assign_doubles_roles(candidates)

    assert roles[1].side == "near"
    assert roles[1].role == "left"
    assert roles[2].role == "right"
    assert roles[3].side == "far"
    assert roles[4].side == "far"


def test_coach_anchor_binds_nearest_track_within_radius():
    candidates = [
        TrackCandidate(track_id=1, world_xy=[0.0, 0.0], confidence=0.7),
        TrackCandidate(track_id=2, world_xy=[2.0, 0.0], confidence=0.9),
    ]

    identity = coach_anchor(candidates, anchor_world_xy=[1.8, 0.1], label="server", max_distance_m=0.5)

    assert identity.track_id == 2
    assert identity.label == "server"

    with pytest.raises(ValueError, match="no track within"):
        coach_anchor(candidates, anchor_world_xy=[10.0, 0.0], label="server", max_distance_m=0.5)
