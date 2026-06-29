from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from threed.racketsport.mobile_person_eval import score_mobile_person_tracks, write_mobile_person_metrics
from threed.racketsport.schemas import (
    MobilePersonTrackingMetrics,
    OnDevicePersonTiming,
    OnDevicePersonTracks,
    PersonGroundTruth,
    validate_artifact_file,
)


def _ground_truth_payload() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_person_ground_truth",
        "clip_id": "clip_a",
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
            },
            {
                "frame_index": 1,
                "source_frame_id": 2,
                "labels": [
                    {
                        "track_id": 1,
                        "bbox_xywh": [1.0, 0.0, 10.0, 10.0],
                        "ignored": False,
                        "visibility": 1.0,
                        "confidence": 1.0,
                        "class_id": 1,
                        "class_name": "player",
                        "person_class": True,
                    },
                    {
                        "track_id": 2,
                        "bbox_xywh": [21.0, 0.0, 10.0, 10.0],
                        "ignored": False,
                        "visibility": 1.0,
                        "confidence": 1.0,
                        "class_id": 1,
                        "class_name": "player",
                        "person_class": True,
                    },
                ],
            },
        ],
        "summary": {
            "frame_count": 2,
            "valid_label_count": 4,
            "ignored_label_count": 0,
            "track_ids": [1, 2],
            "max_valid_players_per_frame": 2,
        },
    }


def _single_player_frame(frame_index: int, track_id: int, x: float) -> dict:
    return {
        "frame_index": frame_index,
        "source_frame_id": frame_index + 1,
        "labels": [
            {
                "track_id": track_id,
                "bbox_xywh": [x, 0.0, 10.0, 10.0],
                "ignored": False,
                "visibility": 1.0,
                "confidence": 1.0,
                "class_id": 1,
                "class_name": "player",
                "person_class": True,
            }
        ],
    }


def _single_prediction_frame(frame_index: int, track_id: int, x: float) -> dict:
    return {
        "frame_index": frame_index,
        "detections": [
            {"track_id": track_id, "bbox_xywh": [x, 0.0, 10.0, 10.0], "confidence": 0.9, "source": "vision"}
        ],
    }


def _predictions_payload(*, second_frame_track_for_gt1: int = 1, include_extra_fp: bool = False) -> dict:
    detections = [
        {"track_id": 1, "bbox_xywh": [0.0, 0.0, 10.0, 10.0], "confidence": 0.9, "source": "vision"},
        {"track_id": 2, "bbox_xywh": [20.0, 0.0, 10.0, 10.0], "confidence": 0.9, "source": "vision"},
    ]
    frame_two = [
        {"track_id": second_frame_track_for_gt1, "bbox_xywh": [1.0, 0.0, 10.0, 10.0], "confidence": 0.9, "source": "vision"},
        {"track_id": 2, "bbox_xywh": [21.0, 0.0, 10.0, 10.0], "confidence": 0.9, "source": "vision"},
    ]
    if include_extra_fp:
        frame_two.append({"track_id": 9, "bbox_xywh": [60.0, 60.0, 10.0, 10.0], "confidence": 0.5, "source": "vision"})
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_on_device_person_tracks",
        "clip_id": "clip_a",
        "candidate": "vision_human_rectangles_iou_v1",
        "device_model": "iPhone15,2",
        "fps": 30.0,
        "frames": [
            {"frame_index": 0, "detections": detections},
            {"frame_index": 1, "detections": frame_two},
        ],
        "summary": {"frame_count": 2, "detection_count": 4 + int(include_extra_fp), "track_ids": [1, 2, 9] if include_extra_fp else [1, 2]},
    }


def test_score_mobile_person_tracks_reports_perfect_identity_when_tracks_are_stable() -> None:
    ground_truth = PersonGroundTruth.model_validate(_ground_truth_payload())
    predictions = OnDevicePersonTracks.model_validate(_predictions_payload())

    metrics = score_mobile_person_tracks(ground_truth, predictions, iou_threshold=0.5)

    assert metrics.idf1 == pytest.approx(1.0)
    assert metrics.mota == pytest.approx(1.0)
    assert metrics.id_switches == 0
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.expected_player_coverage == pytest.approx(1.0)


def test_score_mobile_person_tracks_catches_id_switches_and_false_positives(tmp_path: Path) -> None:
    ground_truth = PersonGroundTruth.model_validate(_ground_truth_payload())
    predictions = OnDevicePersonTracks.model_validate(
        _predictions_payload(second_frame_track_for_gt1=7, include_extra_fp=True)
    )

    metrics = score_mobile_person_tracks(ground_truth, predictions, iou_threshold=0.5)

    assert metrics.id_switches == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0
    assert metrics.idf1 == pytest.approx(6 / 9)
    assert metrics.mota == pytest.approx(0.5)

    out_path = tmp_path / "mobile_person_tracking_metrics.json"
    write_mobile_person_metrics(out_path, metrics)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "racketsport_mobile_person_tracking_metrics"
    parsed = validate_artifact_file("mobile_person_tracking_metrics", out_path)
    assert isinstance(parsed, MobilePersonTrackingMetrics)


def test_score_mobile_person_tracks_uses_best_identity_subset_when_pred_ids_are_fewer() -> None:
    payload = _ground_truth_payload()
    payload["frames"] = [
        _single_player_frame(0, 1, 0.0),
        _single_player_frame(1, 2, 20.0),
        _single_player_frame(2, 2, 21.0),
        _single_player_frame(3, 2, 22.0),
    ]
    payload["summary"] = {
        "frame_count": 4,
        "valid_label_count": 4,
        "ignored_label_count": 0,
        "track_ids": [1, 2],
        "max_valid_players_per_frame": 1,
    }
    predictions_payload = _predictions_payload()
    predictions_payload["frames"] = [
        _single_prediction_frame(0, 7, 0.0),
        _single_prediction_frame(1, 7, 20.0),
        _single_prediction_frame(2, 7, 21.0),
        _single_prediction_frame(3, 7, 22.0),
    ]
    predictions_payload["summary"] = {"frame_count": 4, "detection_count": 4, "track_ids": [7]}
    ground_truth = PersonGroundTruth.model_validate(payload)
    predictions = OnDevicePersonTracks.model_validate(predictions_payload)

    metrics = score_mobile_person_tracks(ground_truth, predictions, iou_threshold=0.5)

    assert metrics.matches == 4
    assert metrics.idf1 == pytest.approx(0.75)


def test_score_mobile_person_tracks_excludes_ignored_region_predictions_from_idf1_denominator() -> None:
    payload = _ground_truth_payload()
    payload["frames"] = [
        {
            "frame_index": 0,
            "source_frame_id": 1,
            "labels": [
                {
                    "track_id": 99,
                    "bbox_xywh": [0.0, 0.0, 10.0, 10.0],
                    "ignored": True,
                    "visibility": 1.0,
                    "confidence": 0.0,
                    "class_id": 2,
                    "class_name": "spectator",
                    "person_class": False,
                }
            ],
        }
    ]
    payload["summary"] = {
        "frame_count": 1,
        "valid_label_count": 0,
        "ignored_label_count": 1,
        "track_ids": [],
        "max_valid_players_per_frame": 0,
    }
    predictions_payload = _predictions_payload()
    predictions_payload["frames"] = [_single_prediction_frame(0, 1, 0.0)]
    predictions_payload["summary"] = {"frame_count": 1, "detection_count": 1, "track_ids": [1]}
    ground_truth = PersonGroundTruth.model_validate(payload)
    predictions = OnDevicePersonTracks.model_validate(predictions_payload)

    metrics = score_mobile_person_tracks(ground_truth, predictions, iou_threshold=0.5)

    assert metrics.pred_detections == 0
    assert metrics.false_positives == 0
    assert metrics.idf1 == pytest.approx(0.0)


def test_score_mobile_person_tracks_cli_writes_metrics(tmp_path: Path) -> None:
    gt_path = tmp_path / "person_ground_truth.json"
    pred_path = tmp_path / "on_device_person_tracks.json"
    out_path = tmp_path / "metrics.json"
    gt_path.write_text(json.dumps(_ground_truth_payload()), encoding="utf-8")
    pred_path.write_text(json.dumps(_predictions_payload()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/score_mobile_person_tracks.py",
            "--ground-truth",
            str(gt_path),
            "--predictions",
            str(pred_path),
            "--out",
            str(out_path),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert str(out_path) in result.stdout
    parsed = validate_artifact_file("mobile_person_tracking_metrics", out_path)
    assert isinstance(parsed, MobilePersonTrackingMetrics)
    assert parsed.idf1 == pytest.approx(1.0)


def test_on_device_person_timing_schema_validates_phone_runtime_payload(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_on_device_person_timing",
        "clip_id": "clip_a",
        "candidate": "vision_human_rectangles_iou_v1",
        "mode": "replay",
        "device_model": "iPhone15,2",
        "os_version": "26.5",
        "wall_clock_seconds": 2.0,
        "dropped_frame_count": 3,
        "model_load_ms": None,
        "mlpackage_size_mb": None,
        "started_thermal_state": "nominal",
        "ended_thermal_state": "fair",
        "samples": [
            {"frame_index": 0, "latency_ms": 8.0, "processed": True},
            {"frame_index": 1, "latency_ms": 12.0, "processed": True},
        ],
        "summary": {
            "processed_frame_count": 2,
            "dropped_frame_count": 3,
            "sustained_processed_fps": 1.0,
            "p50_latency_ms": 8.0,
            "p95_latency_ms": 12.0,
        },
    }
    path = tmp_path / "timing.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    parsed = validate_artifact_file("on_device_person_timing", path)

    assert isinstance(parsed, OnDevicePersonTiming)
    assert parsed.mode == "replay"
    assert parsed.summary.p95_latency_ms == pytest.approx(12.0)


def test_on_device_person_track_schema_rejects_impossible_values() -> None:
    payload = _predictions_payload()
    payload["fps"] = -30.0
    payload["frames"][0]["frame_index"] = -1
    payload["frames"][0]["detections"][0]["track_id"] = 0

    with pytest.raises(ValidationError):
        OnDevicePersonTracks.model_validate(payload)


def test_on_device_person_timing_schema_rejects_impossible_values() -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_on_device_person_timing",
        "clip_id": "clip_a",
        "candidate": "vision_human_rectangles_iou_v1",
        "mode": "live",
        "wall_clock_seconds": -1.0,
        "dropped_frame_count": -1,
        "samples": [{"frame_index": -1, "latency_ms": -5.0, "processed": True}],
        "summary": {
            "processed_frame_count": -1,
            "dropped_frame_count": -1,
            "sustained_processed_fps": -1.0,
            "p50_latency_ms": -1.0,
            "p95_latency_ms": -1.0,
        },
    }

    with pytest.raises(ValidationError):
        OnDevicePersonTiming.model_validate(payload)
