from __future__ import annotations

import pytest

from threed.racketsport.person_detector_oracle import detections_payload_to_candidates, score_detector_oracle
from threed.racketsport.schemas import (
    PersonGroundTruth,
    PersonGroundTruthFrame,
    PersonGroundTruthSummary,
    PersonLabel,
)


def test_detections_payload_to_candidates_converts_xyxy_and_sorts_by_confidence() -> None:
    payload = {
        "frames": [
            {
                "frame": 0,
                "detections": [
                    {"bbox": [20.0, 10.0, 30.0, 40.0], "conf": 0.4},
                    {"bbox": [5.0, 6.0, 15.0, 26.0], "conf": 0.9},
                ],
            }
        ]
    }

    candidates = detections_payload_to_candidates(payload)

    assert candidates == {
        0: [
            {"bbox_xywh": [5.0, 6.0, 10.0, 20.0], "confidence": 0.9},
            {"bbox_xywh": [20.0, 10.0, 10.0, 30.0], "confidence": 0.4},
        ]
    }


def test_score_detector_oracle_reports_topn_recall_and_per_track_recall() -> None:
    gt = _ground_truth()
    candidates = {
        0: [
            {"bbox_xywh": [50.0, 0.0, 10.0, 10.0], "confidence": 0.99},
            {"bbox_xywh": [100.0, 0.0, 10.0, 10.0], "confidence": 0.80},
            {"bbox_xywh": [0.0, 0.0, 10.0, 10.0], "confidence": 0.70},
        ],
        1: [
            {"bbox_xywh": [0.0, 0.0, 10.0, 10.0], "confidence": 0.60},
        ],
    }

    report = score_detector_oracle(
        gt,
        candidates,
        candidate_limits=(1, 2, 3),
        iou_thresholds=(0.5,),
    )

    assert report["gt_detections"] == 3
    assert report["candidate_limits"]["1"]["iou_0.50"]["recall"] == pytest.approx(1 / 3)
    assert report["candidate_limits"]["2"]["iou_0.50"]["recall"] == pytest.approx(2 / 3)
    assert report["candidate_limits"]["3"]["iou_0.50"]["recall"] == pytest.approx(1.0)
    per_track = report["candidate_limits"]["2"]["iou_0.50"]["per_track"]
    assert per_track["1"] == {"hits": 1, "total": 2, "recall": 0.5}
    assert per_track["2"] == {"hits": 1, "total": 1, "recall": 1.0}


def _ground_truth() -> PersonGroundTruth:
    return PersonGroundTruth(
        schema_version=1,
        artifact_type="racketsport_person_ground_truth",
        clip_id="clip",
        source_format="cvat_mot_1_1",
        source_path="synthetic.zip",
        fps=30.0,
        frames=[
            PersonGroundTruthFrame(
                frame_index=0,
                source_frame_id=1,
                labels=[
                    PersonLabel(track_id=1, bbox_xywh=(0.0, 0.0, 10.0, 10.0)),
                    PersonLabel(track_id=2, bbox_xywh=(100.0, 0.0, 10.0, 10.0)),
                ],
            ),
            PersonGroundTruthFrame(
                frame_index=1,
                source_frame_id=2,
                labels=[PersonLabel(track_id=1, bbox_xywh=(0.0, 0.0, 10.0, 10.0))],
            ),
        ],
        summary=PersonGroundTruthSummary(
            frame_count=2,
            valid_label_count=3,
            ignored_label_count=0,
            track_ids=[1, 2],
            max_valid_players_per_frame=2,
        ),
    )
