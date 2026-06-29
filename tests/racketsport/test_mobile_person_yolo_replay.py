from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

import scripts.racketsport.run_mobile_person_yolo_replay as replay_cli
import threed.racketsport.mobile_person_yolo_replay as replay
from tests.racketsport.calibration_fixtures import minimal_calibration_image_pts, minimal_calibration_world_pts
from threed.racketsport.mobile_person_yolo_replay import (
    ReplayYoloCandidate,
    _closed_set_prune_frames,
    _closed_set_track_summaries,
    _expand_bbox_xywh,
    _make_linker,
    _observations_from_result,
    _prune_observations,
)
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


def test_wide_role_lock_uses_spatial_diversity_instead_of_top_confidence_only() -> None:
    linker = _make_linker("predict_role_lock_wide", max_players=4)

    detections = linker.update(
        frame_index=0,
        observations=[
            _observation(100.0, 100.0, confidence=0.99),
            _observation(103.0, 102.0, confidence=0.98),
            _observation(300.0, 100.0, confidence=0.90),
            _observation(100.0, 300.0, confidence=0.89),
            _observation(300.0, 300.0, confidence=0.05),
        ],
    )

    selected_origins = {tuple(det["bbox_xywh"][:2]) for det in detections}
    assert selected_origins == {(100.0, 100.0), (300.0, 100.0), (100.0, 300.0), (300.0, 300.0)}


def test_closed_set_track_summaries_use_true_percentiles() -> None:
    frames = []
    for frame_index, (confidence, width) in enumerate(
        [(0.1, 10.0), (0.2, 20.0), (0.3, 30.0), (0.4, 40.0), (0.9, 100.0)]
    ):
        frames.append(
            {
                "frame_index": frame_index,
                "detections": [
                    {
                        "track_id": 7,
                        "bbox_xywh": [0.0, 0.0, width, 1.0],
                        "confidence": confidence,
                    }
                ],
            }
        )

    summary = _closed_set_track_summaries(frames, frame_width=500.0)[7]

    assert summary["confidence_p90"] == pytest.approx(0.7)
    assert summary["median_area"] == pytest.approx(30.0)
    assert summary["p90_area"] == pytest.approx(76.0)
    assert summary["median_center"] == pytest.approx([15.0, 0.5])


def test_stable_set_linker_assigns_from_wider_candidate_pool() -> None:
    linker = _make_linker("predict_stable_set", max_players=4)

    first = linker.update(
        frame_index=0,
        observations=[
            _observation(0.0, 0.0),
            _observation(100.0, 0.0),
            _observation(0.0, 100.0),
            _observation(100.0, 100.0),
        ],
    )
    second = linker.update(
        frame_index=1,
        observations=[
            _observation(500.0, 500.0, confidence=0.99),
            _observation(520.0, 500.0, confidence=0.98),
            _observation(540.0, 500.0, confidence=0.97),
            _observation(560.0, 500.0, confidence=0.96),
            _observation(4.0, 0.0, confidence=0.10),
            _observation(104.0, 0.0, confidence=0.10),
            _observation(4.0, 100.0, confidence=0.10),
            _observation(104.0, 100.0, confidence=0.10),
        ],
    )

    assert [det["track_id"] for det in first] == [1, 2, 3, 4]
    assert [det["track_id"] for det in second] == [1, 2, 3, 4]
    assert [det["bbox_xywh"][0] for det in second] == pytest.approx([4.0, 104.0, 4.0, 104.0])


def test_wide_role_stable_linker_filters_before_temporal_assignment() -> None:
    linker = _make_linker("predict_role_lock_wide_stable", max_players=4)

    first = linker.update(
        frame_index=0,
        observations=[
            _observation(0.0, 0.0),
            _observation(100.0, 0.0),
            _observation(0.0, 100.0),
            _observation(100.0, 100.0),
        ],
    )
    second = linker.update(
        frame_index=1,
        observations=[
            _observation(500.0, 500.0, confidence=0.99),
            _observation(4.0, 0.0, confidence=0.10),
            _observation(104.0, 0.0, confidence=0.10),
            _observation(4.0, 100.0, confidence=0.10),
            _observation(104.0, 100.0, confidence=0.10),
        ],
    )

    assert [det["track_id"] for det in first] == [1, 2, 3, 4]
    assert [det["track_id"] for det in second] == [1, 2, 3, 4]
    assert {round(det["bbox_xywh"][0]) for det in second} == {4, 104}


def test_stable_set_linker_uses_appearance_when_tracks_cross() -> None:
    linker = _make_linker("predict_stable_set", max_players=2)

    first = linker.update(
        frame_index=0,
        observations=[
            {**_observation(0.0, 0.0), "appearance_hsv": [1.0, 0.0, 0.0]},
            {**_observation(100.0, 0.0), "appearance_hsv": [0.0, 1.0, 0.0]},
        ],
    )
    second = linker.update(
        frame_index=1,
        observations=[
            {**_observation(96.0, 0.0), "appearance_hsv": [1.0, 0.0, 0.0]},
            {**_observation(4.0, 0.0), "appearance_hsv": [0.0, 1.0, 0.0]},
        ],
    )

    assert [det["track_id"] for det in first] == [1, 2]
    assert [det["track_id"] for det in second] == [1, 2]
    assert [det["bbox_xywh"][0] for det in second] == pytest.approx([96.0, 4.0])


def test_stable_set_linker_replaces_expired_tracks() -> None:
    linker = _make_linker("predict_stable_set", max_players=2, max_age_frames=1)

    first = linker.update(frame_index=0, observations=[_observation(0.0, 0.0), _observation(100.0, 0.0)])
    second = linker.update(frame_index=5, observations=[_observation(300.0, 0.0), _observation(400.0, 0.0)])

    assert [det["track_id"] for det in first] == [1, 2]
    assert len(second) == 2
    assert [det["track_id"] for det in second] == [3, 4]
    assert [det["bbox_xywh"][0] for det in second] == pytest.approx([300.0, 400.0])


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


def test_observations_from_result_can_keep_more_candidates_than_final_players() -> None:
    class _Array:
        def __init__(self, value):
            self._value = value

        def cpu(self):
            return self

        def numpy(self):
            return self._value

    class _Boxes:
        xyxy = _Array(
            [
                [0.0, 0.0, 10.0, 20.0],
                [20.0, 0.0, 30.0, 20.0],
                [40.0, 0.0, 50.0, 20.0],
            ]
        )
        conf = _Array([0.9, 0.8, 0.7])

        def __len__(self):
            return 3

    result = types.SimpleNamespace(boxes=_Boxes())

    observations = _observations_from_result(result, max_players=2, output_limit=3)

    assert len(observations) == 3
    assert [obs["confidence"] for obs in observations] == pytest.approx([0.9, 0.8, 0.7])


def test_closed_set_pruning_keeps_long_stable_tracks_and_remaps_ids() -> None:
    frames = []
    for frame_index in range(20):
        detections = [
            {"track_id": 10, "bbox_xywh": [10.0, 20.0, 20.0, 50.0], "confidence": 0.90, "source": "test", "role": None},
            {"track_id": 20, "bbox_xywh": [110.0, 25.0, 20.0, 50.0], "confidence": 0.88, "source": "test", "role": None},
            {"track_id": 30, "bbox_xywh": [210.0, 30.0, 20.0, 50.0], "confidence": 0.86, "source": "test", "role": None},
            {"track_id": 40, "bbox_xywh": [310.0, 35.0, 20.0, 50.0], "confidence": 0.84, "source": "test", "role": None},
        ]
        if frame_index < 4:
            detections.append(
                {
                    "track_id": 99,
                    "bbox_xywh": [390.0, 15.0, 20.0, 50.0],
                    "confidence": 0.99,
                    "source": "test",
                    "role": None,
                }
            )
        frames.append({"frame_index": frame_index, "detections": detections})

    pruned = _closed_set_prune_frames(frames, max_players=4, mode="quality", frame_width=420.0)

    selected_ids = {
        detection["track_id"]
        for frame in pruned
        for detection in frame["detections"]
    }
    assert selected_ids == {1, 2, 3, 4}
    assert all(len(frame["detections"]) == 4 for frame in pruned[4:])
    assert all(
        detection["source"] == "test+closed_set_quality"
        for frame in pruned
        for detection in frame["detections"]
    )


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


def test_run_mobile_person_yolo_replay_cli_forwards_args_and_prints_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_run_replay_yolo_candidate(**kwargs):
        captured.update(kwargs)
        return {"metrics": {"idf1": 0.875, "expected_players": kwargs["candidate"].max_players}}

    monkeypatch.setattr(replay_cli, "run_replay_yolo_candidate", fake_run_replay_yolo_candidate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/racketsport/run_mobile_person_yolo_replay.py",
            "--video",
            str(tmp_path / "clip.mp4"),
            "--ground-truth",
            str(tmp_path / "person_ground_truth.json"),
            "--model",
            "models_coreml/yolo26n_img416_int8/yolo26n.mlpackage",
            "--candidate",
            "coreml-smoke",
            "--out-dir",
            str(tmp_path / "out"),
            "--imgsz",
            "512",
            "--conf",
            "0.2",
            "--max-players",
            "2",
            "--tracker",
            "predict_center",
            "--no-overlay",
        ],
    )

    assert replay_cli.main() == 0

    output = json.loads(capsys.readouterr().out)
    assert output == {"expected_players": 2, "idf1": 0.875}
    assert captured["video_path"] == tmp_path / "clip.mp4"
    assert captured["ground_truth_path"] == tmp_path / "person_ground_truth.json"
    assert captured["out_dir"] == tmp_path / "out"
    assert captured["max_frames"] is None
    assert captured["render_overlay"] is False
    candidate = captured["candidate"]
    assert isinstance(candidate, ReplayYoloCandidate)
    assert candidate.name == "coreml-smoke"
    assert candidate.model == "models_coreml/yolo26n_img416_int8/yolo26n.mlpackage"
    assert candidate.imgsz == 512
    assert candidate.conf == pytest.approx(0.2)
    assert candidate.max_players == 2
    assert candidate.tracker == "predict_center"


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
        image_pts=minimal_calibration_image_pts(),
        world_pts=minimal_calibration_world_pts(),
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
