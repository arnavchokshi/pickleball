from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.racket_candidates import racket_labels_to_candidates
from threed.racketsport.racket_true_corners import (
    build_paddle_true_corner_review,
    true_corner_labels_to_candidates,
)
from threed.racketsport.schemas import RacketCandidates, validate_artifact_file


def test_racket_labels_to_candidates_builds_ordered_four_corner_boxes() -> None:
    payload = {
        "annotation": {
            "items": [
                {
                    "frame": "frame_000002.jpg",
                    "bbox_xyxy": [100, 200, 140, 280],
                    "confidence": 0.7,
                    "status": "uncertain",
                    "player_id": "p3",
                    "source": "yolo26m_teacher",
                    "class_name": "tennis racket",
                },
                {
                    "frame": "frame_000003.jpg",
                    "bbox": [200, 100, 20, 50],
                    "confidence": 0.1,
                    "status": "uncertain",
                    "player_id": "p3",
                    "source": "yolo26m_teacher",
                    "class_name": "tennis racket",
                },
            ]
        }
    }

    candidates, counts = racket_labels_to_candidates(payload, fps=30.0, min_confidence=0.2)

    assert counts == {"accepted": 1, "skipped_status": 0, "skipped_confidence": 1, "skipped_invalid": 0}
    parsed = RacketCandidates.model_validate(candidates)
    assert parsed.artifact_type == "racketsport_racket_candidates"
    assert parsed.fps == pytest.approx(30.0)
    assert parsed.players[0].id == 3
    assert parsed.players[0].paddle_dims_in == {"length": 16.0, "width": 8.0}
    frame = parsed.players[0].frames[0]
    assert frame.t == pytest.approx(1.0 / 30.0)
    assert frame.corners_px == [[100.0, 200.0], [140.0, 200.0], [140.0, 280.0], [100.0, 280.0]]
    assert frame.conf == pytest.approx(0.7)
    assert frame.source == "label_bbox:yolo26m_teacher"


def test_racket_labels_to_candidates_fails_closed_without_valid_items() -> None:
    payload = {"items": [{"frame": "frame_000001.jpg", "status": "rejected", "bbox_xyxy": [1, 2, 3, 4]}]}

    with pytest.raises(ValueError, match="no racket candidates accepted"):
        racket_labels_to_candidates(payload, fps=30.0)


def test_build_racket_candidates_cli_writes_registered_artifact_with_manifest_fps(tmp_path: Path) -> None:
    labels = tmp_path / "racket_pose.json"
    manifest = tmp_path / "prototype_autolabel_manifest.json"
    out = tmp_path / "racket_candidates.json"
    labels.write_text(
        json.dumps(
            {
                "annotation": {
                    "items": [
                        {
                            "frame": "frame_000001.jpg",
                            "bbox_xyxy": [10, 20, 50, 100],
                            "confidence": 0.8,
                            "status": "uncertain",
                            "player_id": None,
                            "source": "yolo26m_teacher",
                            "class_name": "tennis racket",
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
            "scripts/racketsport/build_racket_candidates.py",
            "--racket-labels",
            str(labels),
            "--manifest",
            str(manifest),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert str(out) in completed.stdout
    assert "accepted=1" in completed.stderr
    parsed = validate_artifact_file("racket_candidates", out)
    assert isinstance(parsed, RacketCandidates)
    assert parsed.fps == pytest.approx(59.94)
    assert parsed.players[0].id == 0


def test_true_corner_labels_to_candidates_marks_reviewed_corners_as_non_box_evidence() -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_paddle_true_corner_labels",
        "clip": "clip_001",
        "fps": 60.0,
        "label_source": "human_true_corner_review",
        "players": [
            {
                "id": 4,
                "paddle_dims_in": {"length": 16.0, "width": 8.0},
                "frames": [
                    {
                        "frame_index": 12,
                        "corners_px": [[101.0, 201.0], [141.0, 203.0], [138.0, 282.0], [98.0, 279.0]],
                        "conf": 0.95,
                        "reviewer": "qa",
                        "evidence_type": "true_corners",
                    }
                ],
            }
        ],
    }

    candidates, summary = true_corner_labels_to_candidates(payload)

    assert summary == {
        "accepted": 1,
        "skipped_invalid": 0,
        "source": "true_corners:human_true_corner_review",
    }
    parsed = RacketCandidates.model_validate(candidates)
    assert parsed.fps == pytest.approx(60.0)
    assert parsed.players[0].id == 4
    assert parsed.players[0].frames[0].t == pytest.approx(12.0 / 60.0)
    assert parsed.players[0].frames[0].source == "true_corners:human_true_corner_review"


def test_true_corner_labels_reject_box_sources_and_missing_reviewer() -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_paddle_true_corner_labels",
        "clip": "clip_001",
        "fps": 60.0,
        "label_source": "label_bbox:yolo26m_teacher",
        "players": [
            {
                "id": 4,
                "paddle_dims_in": {"length": 16.0, "width": 8.0},
                "frames": [
                    {
                        "frame_index": 12,
                        "corners_px": [[101.0, 201.0], [141.0, 203.0], [138.0, 282.0], [98.0, 279.0]],
                        "conf": 0.95,
                        "evidence_type": "true_corners",
                    }
                ],
            }
        ],
    }

    with pytest.raises(ValueError, match="label_source must not be box-derived"):
        true_corner_labels_to_candidates(payload)


def test_paddle_true_corner_review_lists_required_labels_for_box_candidates() -> None:
    candidates = {
        "schema_version": 1,
        "artifact_type": "racketsport_racket_candidates",
        "fps": 30.0,
        "players": [
            {
                "id": 2,
                "paddle_dims_in": {"length": 16.0, "width": 8.0},
                "frames": [
                    {
                        "t": 0.5,
                        "corners_px": [[10.0, 20.0], [30.0, 20.0], [30.0, 70.0], [10.0, 70.0]],
                        "conf": 0.8,
                        "source": "label_bbox:yolo26m_teacher",
                    }
                ],
            }
        ],
    }

    review = build_paddle_true_corner_review(clip="clip_001", racket_candidates=candidates)

    assert review["artifact_type"] == "racketsport_paddle_true_corner_review"
    assert review["status"] == "blocked_missing_true_corner_labels"
    assert review["trusted_for_rkt_promotion"] is False
    assert review["required_label_count"] == 1
    assert review["true_corner_label_count"] == 0
    assert review["box_derived_candidate_count"] == 1
    assert review["required_labels"][0]["frame_index"] == 15
    assert review["required_labels"][0]["candidate_source"] == "label_bbox:yolo26m_teacher"
    assert review["promotion_blockers"] == [
        "box_candidates_are_not_true_paddle_corners",
        "missing_reviewed_true_corner_labels",
        "missing_reference_or_cad_gt",
    ]
