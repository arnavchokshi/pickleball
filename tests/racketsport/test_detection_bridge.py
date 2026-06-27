from __future__ import annotations

import json
import subprocess
import sys

import pytest

from threed.racketsport.detection_bridge import player_labels_to_detections


def test_player_labels_to_detections_groups_accepted_players_by_zero_based_frame():
    payload = {
        "annotation": {
            "items": [
                {
                    "frame": "frame_000002.jpg",
                    "bbox_xyxy": [10, 20, 30, 60],
                    "confidence": 0.9,
                    "status": "accepted",
                    "id": "p1",
                    "review_id": "person_2_0",
                },
                {
                    "frame": "frame_000002.jpg",
                    "bbox": [40, 50, 20, 30],
                    "confidence": 0.8,
                    "status": "accepted",
                    "id": "p2",
                },
                {
                    "frame": "frame_000003.jpg",
                    "bbox_xyxy": [11, 22, 33, 66],
                    "confidence": 0.3,
                    "status": "uncertain",
                    "id": "p3",
                },
            ]
        }
    }

    detections = player_labels_to_detections(payload, fps=60.0)

    assert detections == {
        "schema_version": 1,
        "artifact_type": "racketsport_person_detections",
        "source": "player_labels",
        "fps": 60.0,
        "frames": [
            {
                "frame": 1,
                "detections": [
                    {
                        "bbox_xyxy": [10.0, 20.0, 30.0, 60.0],
                        "conf": 0.9,
                        "class": "person",
                        "source_id": "p1",
                        "source_review_id": "person_2_0",
                    },
                    {
                        "bbox_xyxy": [40.0, 50.0, 60.0, 80.0],
                        "conf": 0.8,
                        "class": "person",
                        "source_id": "p2",
                    },
                ],
            }
        ],
        "counts": {"accepted": 2, "skipped_status": 1, "skipped_invalid": 0},
        "qualitative_status": "prototype_teacher_detections_not_verified",
    }


def test_player_labels_to_detections_can_preserve_label_ids_when_requested():
    payload = {
        "items": [
            {
                "frame": 0,
                "bbox_xyxy": [10, 20, 30, 60],
                "confidence": 0.9,
                "status": "accepted",
                "id": "p7",
            }
        ]
    }

    detections = player_labels_to_detections(payload, fps=30.0, preserve_label_ids=True)

    assert detections["frames"][0]["detections"][0]["temp_track_id"] == "p7"


def test_player_labels_to_detections_fails_closed_without_fps():
    with pytest.raises(ValueError, match="fps must be positive"):
        player_labels_to_detections({"items": []}, fps=0.0)


def test_convert_player_labels_cli_writes_detections_with_manifest_fps(tmp_path):
    labels = tmp_path / "players.json"
    manifest = tmp_path / "prototype_autolabel_manifest.json"
    out = tmp_path / "detections.json"
    labels.write_text(
        json.dumps(
            {
                "annotation": {
                    "items": [
                        {
                            "frame": "frame_000001.jpg",
                            "bbox_xyxy": [1, 2, 3, 4],
                            "confidence": 0.7,
                            "status": "accepted",
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text(json.dumps({"clip": {"metadata": {"frame_rate_fps": 59.94}}}), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/convert_player_labels_to_detections.py",
            "--players",
            str(labels),
            "--manifest",
            str(manifest),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(out.read_text(encoding="utf-8"))
    assert output["fps"] == pytest.approx(59.94)
    assert output["counts"]["accepted"] == 1
    assert "accepted=1" in completed.stderr
