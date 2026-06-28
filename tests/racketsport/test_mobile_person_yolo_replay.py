from __future__ import annotations

import pytest

from threed.racketsport.mobile_person_yolo_replay import ReplayYoloCandidate, _expand_bbox_xywh, _make_linker, _prune_observations
from threed.racketsport.schemas import CameraIntrinsics, CaptureQuality, CourtCalibration, CourtExtrinsics, ReprojectionError


def _observation(x: float, y: float, w: float = 10.0, h: float = 20.0, confidence: float = 0.9) -> dict:
    return {
        "bbox_xywh": [x, y, w, h],
        "confidence": confidence,
        "source": "test",
    }


def test_candidate_defaults_to_predict_iou_tracker() -> None:
    candidate = ReplayYoloCandidate(name="candidate", model="yolo26n.pt", imgsz=416, conf=0.1, iou=0.6)

    assert candidate.tracker == "predict_iou"
    assert candidate.tracker_config is None


def test_center_distance_linker_keeps_identity_when_iou_drops_but_motion_is_plausible() -> None:
    linker = _make_linker("predict_center", max_players=4)

    first = linker.update(frame_index=0, observations=[_observation(0.0, 0.0), _observation(100.0, 0.0)])
    second = linker.update(frame_index=1, observations=[_observation(14.0, 0.0), _observation(114.0, 0.0)])

    assert [det["track_id"] for det in first] == [1, 2]
    assert [det["track_id"] for det in second] == [1, 2]


def test_role_lock_assigns_stable_court_position_ids_without_temporal_linking() -> None:
    linker = _make_linker("predict_role_lock", max_players=4)

    detections = linker.update(
        frame_index=0,
        observations=[
            _observation(300.0, 300.0),
            _observation(100.0, 100.0),
            _observation(300.0, 100.0),
            _observation(100.0, 300.0),
        ],
    )

    by_box = {tuple(det["bbox_xywh"][:2]): det["track_id"] for det in detections}
    assert by_box[(100.0, 100.0)] == 1
    assert by_box[(300.0, 100.0)] == 2
    assert by_box[(100.0, 300.0)] == 3
    assert by_box[(300.0, 300.0)] == 4


def test_court_pruning_prefers_on_court_player_over_high_confidence_spectator() -> None:
    calibration = _identity_court_calibration()
    observations = [
        _observation(11.0, 0.0, w=1.0, h=1.0, confidence=0.99),
        _observation(0.0, 0.0, w=1.0, h=1.0, confidence=0.40),
    ]

    confidence_only = _prune_observations(observations, max_players=1)
    court_pruned = _prune_observations(
        observations,
        max_players=1,
        prune_mode="court",
        court_calibration=calibration,
        court_margin_m=0.5,
    )

    assert confidence_only[0]["bbox_xywh"][0] == pytest.approx(11.0)
    assert court_pruned[0]["bbox_xywh"][0] == pytest.approx(0.0)


def test_bbox_expansion_preserves_bottom_center_foot_point() -> None:
    expanded = _expand_bbox_xywh([10.0, 20.0, 20.0, 40.0], 1.5)

    assert expanded == pytest.approx([5.0, 0.0, 30.0, 60.0])


def _identity_court_calibration() -> CourtCalibration:
    return CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, dist=[], source="test"),
        extrinsics=CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 10.0],
            camera_height_m=10.0,
        ),
        reprojection_error_px=ReprojectionError(median=0.0, p95=0.0),
        capture_quality=CaptureQuality(grade="good", reasons=[]),
        image_pts=[],
        world_pts=[],
    )
