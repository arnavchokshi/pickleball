from __future__ import annotations

import json
import sys
import types

import pytest

import threed.racketsport.mobile_person_yolo_replay as replay
from threed.racketsport.mobile_person_yolo_replay import ReplayYoloCandidate, _expand_bbox_xywh, _make_linker, _prune_observations
from threed.racketsport.schemas import MobilePersonTrackingMetrics
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


def test_replay_yolo_candidate_scores_with_candidate_max_players(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured: dict[str, int] = {}

    class _Array:
        def __init__(self, value):
            self._value = value

        def cpu(self):
            return self

        def numpy(self):
            return self._value

    class _Boxes:
        xyxy = _Array([[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 30.0, 10.0]])
        conf = _Array([0.9, 0.8])

        def __len__(self):
            return 2

    class _Result:
        boxes = _Boxes()
        speed = {"inference": 1.0}

    class _YOLO:
        def __init__(self, model: str) -> None:
            self.model = model

        def predict(self, **kwargs):
            del kwargs
            yield _Result()

    def _score(ground_truth, predictions, *, iou_threshold=0.5, expected_players=None):
        del ground_truth, predictions, iou_threshold
        captured["expected_players"] = expected_players
        return MobilePersonTrackingMetrics(
            schema_version=1,
            artifact_type="racketsport_mobile_person_tracking_metrics",
            clip_id="clip-a",
            candidate="two-player",
            iou_threshold=0.5,
            frames=1,
            gt_detections=2,
            pred_detections=2,
            matches=2,
            false_positives=0,
            false_negatives=0,
            id_switches=0,
            idf1=1.0,
            mota=1.0,
            precision=1.0,
            recall=1.0,
            expected_players=expected_players or 0,
            expected_player_coverage=1.0,
            expected_player_frames=1,
            exact_expected_player_frames=1,
        )

    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_YOLO))
    monkeypatch.setattr(replay, "_video_properties", lambda cv2, video: (30.0, 100, 100, 1))
    monkeypatch.setattr(replay, "score_mobile_person_tracks", _score)

    ground_truth_path = tmp_path / "person_ground_truth.json"
    ground_truth_path.write_text(json.dumps(_person_ground_truth_payload()), encoding="utf-8")

    summary = replay.run_replay_yolo_candidate(
        video_path=tmp_path / "clip.mp4",
        ground_truth_path=ground_truth_path,
        candidate=ReplayYoloCandidate(
            name="two-player",
            model="dummy.pt",
            imgsz=416,
            conf=0.1,
            iou=0.6,
            max_players=2,
        ),
        out_dir=tmp_path / "out",
        render_overlay=False,
    )

    assert captured["expected_players"] == 2
    assert summary["metrics"]["expected_players"] == 2


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


def _person_ground_truth_payload() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_person_ground_truth",
        "clip_id": "clip-a",
        "source_format": "cvat_mot_1_1",
        "source_path": "synthetic.zip",
        "fps": 30.0,
        "frames": [
            {
                "frame_index": 0,
                "source_frame_id": 1,
                "labels": [
                    {
                        "track_id": 1,
                        "bbox_xywh": [0.0, 0.0, 10.0, 10.0],
                        "ignored": False,
                        "visibility": 1.0,
                        "confidence": 1.0,
                        "class_id": 1,
                        "class_name": "player",
                        "person_class": True,
                    },
                    {
                        "track_id": 2,
                        "bbox_xywh": [20.0, 0.0, 10.0, 10.0],
                        "ignored": False,
                        "visibility": 1.0,
                        "confidence": 1.0,
                        "class_id": 1,
                        "class_name": "player",
                        "person_class": True,
                    },
                ],
            }
        ],
        "summary": {
            "frame_count": 1,
            "valid_label_count": 2,
            "ignored_label_count": 0,
            "track_ids": [1, 2],
            "max_valid_players_per_frame": 2,
        },
    }
