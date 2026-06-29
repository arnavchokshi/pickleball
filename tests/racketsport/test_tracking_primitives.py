from __future__ import annotations

import pytest

from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts
from threed.racketsport.court_templates import FT_TO_M
from threed.racketsport import doubles_id, track_lock
from threed.racketsport.doubles_id import DoublesIdentity, assign_doubles_roles, coach_anchor
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


def test_court_polygon_filter_can_include_runoff_margin_for_players_near_lines():
    detections = [
        PersonDetection(
            bbox_xyxy=(100.0, 100.0, 140.0, 260.0),
            confidence=0.92,
            foot_world_xy=[(10.0 * FT_TO_M) + 0.5, 0.0],
        ),
        PersonDetection(
            bbox_xyxy=(500.0, 100.0, 540.0, 260.0),
            confidence=0.89,
            foot_world_xy=[(10.0 * FT_TO_M) + 1.5, 0.0],
        ),
    ]

    assert court_polygon_filter(detections, sport="pickleball") == []
    assert court_polygon_filter(detections, sport="pickleball", margin_m=0.75) == [detections[0]]


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
        image_pts=minimal_calibration_image_pts(),
        world_pts=minimal_calibration_world_pts(),
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


def test_update_track_lock_preserves_locked_ids_when_steps_are_plausible():
    previous = [
        TrackCandidate(track_id=10, world_xy=[0.0, 0.0], confidence=0.9),
        TrackCandidate(track_id=20, world_xy=[2.0, 0.0], confidence=0.8),
    ]
    current = [
        TrackCandidate(track_id=20, world_xy=[2.4, 0.0], confidence=0.6),
        TrackCandidate(track_id=30, world_xy=[8.0, 0.0], confidence=0.99),
        TrackCandidate(track_id=10, world_xy=[0.3, 0.1], confidence=0.5),
    ]

    result = track_lock.update_track_lock(previous, current, max_step_m=0.5)

    assert result.accepted
    assert [candidate.track_id for candidate in result.locked] == [10, 20]
    assert result.locked[0].world_xy == pytest.approx([0.3, 0.1])
    assert result.locked[1].world_xy == pytest.approx([2.4, 0.0])
    assert result.notes == ["preserved_locked_ids:10,20"]


def test_update_track_lock_fails_closed_when_locked_track_is_missing():
    previous = [
        TrackCandidate(track_id=10, world_xy=[0.0, 0.0], confidence=0.9),
        TrackCandidate(track_id=20, world_xy=[2.0, 0.0], confidence=0.8),
    ]
    current = [TrackCandidate(track_id=10, world_xy=[0.2, 0.0], confidence=0.7)]

    result = track_lock.update_track_lock(previous, current, max_step_m=1.0)

    assert not result.accepted
    assert result.locked == previous
    assert result.notes == ["missing_locked_track:20"]


def test_update_track_lock_fails_closed_when_locked_track_teleports():
    previous = [TrackCandidate(track_id=10, world_xy=[0.0, 0.0], confidence=0.9)]
    current = [TrackCandidate(track_id=10, world_xy=[2.5, 0.0], confidence=0.7)]

    result = track_lock.update_track_lock(previous, current, max_step_m=1.0)

    assert not result.accepted
    assert result.locked == previous
    assert result.notes == ["implausible_ground_step:10"]


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


def test_apply_coach_anchor_labels_existing_identity_without_recomputing_role():
    candidates = [
        TrackCandidate(track_id=1, world_xy=[-2.0, -4.0], confidence=0.9),
        TrackCandidate(track_id=2, world_xy=[2.0, 4.0], confidence=0.9),
    ]
    identities = {
        1: DoublesIdentity(track_id=1, side="near", role="left"),
        2: DoublesIdentity(track_id=2, side="near", role="left"),
    }

    labeled = doubles_id.apply_coach_anchor(
        identities,
        candidates,
        anchor_world_xy=[2.1, 4.1],
        label="server",
        max_distance_m=0.5,
    )

    assert labeled[2] == DoublesIdentity(track_id=2, side="near", role="left", label="server")
    assert identities[2].label is None

    with pytest.raises(ValueError, match="no identity within"):
        doubles_id.apply_coach_anchor(
            identities,
            candidates,
            anchor_world_xy=[10.0, 0.0],
            label="server",
            max_distance_m=0.5,
        )
