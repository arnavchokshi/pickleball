from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import threed.racketsport.racket_candidates as racket_candidate_module
from threed.racketsport.racket_candidates import racket_labels_to_candidates
from threed.racketsport.racket_true_corners import (
    build_paddle_true_corner_review,
    is_true_corner_source,
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


def test_cvat_paddle_boxes_convert_to_review_only_box_candidates() -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_video_annotations",
        "clip_id": "clip_cvat",
        "source_format": "cvat_video_1_1",
        "source_path": "cvat.zip",
        "task": {
            "task_id": 42,
            "name": "clip_cvat",
            "size": 3,
            "mode": "interpolation",
            "start_frame": 0,
            "stop_frame": 2,
            "original_size": [1920, 1080],
            "source": "clip.mp4",
            "dumped": None,
        },
        "frames": [
            {
                "frame_index": 0,
                "boxes": [
                    {
                        "track_id": 3,
                        "label": "player",
                        "frame_index": 0,
                        "bbox_xyxy": [10.0, 20.0, 110.0, 220.0],
                        "bbox_xywh": [10.0, 20.0, 100.0, 200.0],
                        "keyframe": True,
                        "occluded": False,
                        "source": "manual",
                    }
                ],
            },
            {
                "frame_index": 1,
                "boxes": [
                    {
                        "track_id": 5,
                        "label": "paddle",
                        "frame_index": 1,
                        "bbox_xyxy": [300.0, 400.0, 330.0, 450.0],
                        "bbox_xywh": [300.0, 400.0, 30.0, 50.0],
                        "keyframe": True,
                        "occluded": False,
                        "source": "manual",
                    },
                    {
                        "track_id": 6,
                        "label": "paddle",
                        "frame_index": 1,
                        "bbox_xyxy": [500.0, 600.0, 520.0, 640.0],
                        "bbox_xywh": [500.0, 600.0, 20.0, 40.0],
                        "keyframe": False,
                        "occluded": True,
                        "source": "manual",
                    },
                ],
            },
            {"frame_index": 2, "boxes": []},
        ],
        "tracks": [],
        "summary": {
            "frame_count": 3,
            "visible_box_count": 3,
            "outside_box_count": 0,
            "labels": ["player", "paddle"],
            "track_count_by_label": {"player": 1, "paddle": 2},
            "visible_box_count_by_label": {"player": 1, "paddle": 2},
        },
    }

    candidates, counts = racket_candidate_module.cvat_paddle_boxes_to_candidates(payload, fps=30.0)

    assert counts == {"accepted": 2, "skipped_label": 1, "skipped_invalid": 0}
    parsed = RacketCandidates.model_validate(candidates)
    assert parsed.fps == pytest.approx(30.0)
    assert [player.id for player in parsed.players] == [5, 6]
    assert parsed.players[0].frames[0].t == pytest.approx(1.0 / 30.0)
    assert parsed.players[0].frames[0].corners_px == [
        [300.0, 400.0],
        [330.0, 400.0],
        [330.0, 450.0],
        [300.0, 450.0],
    ]
    assert parsed.players[0].frames[0].source == "label_bbox:cvat_video:paddle"
    assert parsed.players[1].frames[0].source == "label_bbox:cvat_video:paddle_occluded"


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


def test_build_racket_candidates_cli_converts_cvat_paddle_boxes_as_box_derived(tmp_path: Path) -> None:
    reviewed_boxes = tmp_path / "reviewed_boxes.json"
    out = tmp_path / "racket_candidates.json"
    reviewed_boxes.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_cvat_video_annotations",
                "clip_id": "clip_cvat",
                "source_format": "cvat_video_1_1",
                "source_path": "cvat.zip",
                "task": {
                    "size": 2,
                    "start_frame": 0,
                    "stop_frame": 1,
                    "original_size": [1920, 1080],
                },
                "frames": [
                    {"frame_index": 0, "boxes": []},
                    {
                        "frame_index": 1,
                        "boxes": [
                            {
                                "track_id": 8,
                                "label": "paddle",
                                "frame_index": 1,
                                "bbox_xyxy": [100.0, 120.0, 140.0, 180.0],
                                "bbox_xywh": [100.0, 120.0, 40.0, 60.0],
                                "keyframe": True,
                                "occluded": False,
                                "source": "manual",
                            }
                        ],
                    },
                ],
                "tracks": [],
                "summary": {
                    "frame_count": 2,
                    "visible_box_count": 1,
                    "outside_box_count": 0,
                    "labels": ["paddle"],
                    "track_count_by_label": {"paddle": 1},
                    "visible_box_count_by_label": {"paddle": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_racket_candidates.py",
            "--cvat-reviewed-boxes",
            str(reviewed_boxes),
            "--fps",
            "30",
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "accepted=1" in completed.stderr
    parsed = validate_artifact_file("racket_candidates", out)
    assert isinstance(parsed, RacketCandidates)
    assert parsed.players[0].id == 8
    assert parsed.players[0].frames[0].source == "label_bbox:cvat_video:paddle"


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


def test_true_corner_labels_reject_fractional_player_and_frame_ids() -> None:
    base_player = {
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
    updates = (
        {"id": 4.5},
        {"frames": [{**base_player["frames"][0], "frame_index": 12.9}]},
    )
    for update in updates:
        payload = {
            "schema_version": 1,
            "artifact_type": "racketsport_paddle_true_corner_labels",
            "clip": "clip_001",
            "fps": 60.0,
            "label_source": "human_true_corner_review",
            "players": [{**base_player, **update}],
        }

        with pytest.raises(ValueError, match="no reviewed true-corner labels accepted"):
            true_corner_labels_to_candidates(payload)


def test_cad_gt_counts_as_cad_not_reference_gt() -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_paddle_true_corner_labels",
        "clip": "clip_001",
        "fps": 60.0,
        "label_source": "measured_cad_pose",
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
                        "evidence_type": "cad_gt",
                    }
                ],
            }
        ],
    }

    candidates, summary = true_corner_labels_to_candidates(payload)
    review = build_paddle_true_corner_review(
        clip="clip_001",
        racket_candidates=candidates,
        true_corner_candidates=candidates,
    )

    assert summary["source"] == "cad_gt:measured_cad_pose"
    assert review["true_corner_source_evidence_counts"]["synthetic_or_cad"] == 1
    assert review["true_corner_source_evidence_counts"]["reference_gt"] == 0


def test_true_corner_source_policy_does_not_count_ambiguous_manual_sources() -> None:
    assert is_true_corner_source("true_corners:human_review") is True
    assert is_true_corner_source("mask_corner:segmenter") is True
    assert is_true_corner_source("aruco_gt:april_tag_reference") is True
    assert is_true_corner_source("cad_gt:synthetic_paddle") is True
    assert is_true_corner_source("label_bbox:yolo26m_teacher") is False
    assert is_true_corner_source("manual_review") is False

    ambiguous = {
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
                        "source": "manual_review",
                    }
                ],
            }
        ],
    }

    empty_candidates = {
        "schema_version": 1,
        "artifact_type": "racketsport_racket_candidates",
        "fps": 30.0,
        "players": [],
    }
    review = build_paddle_true_corner_review(
        clip="clip_001",
        racket_candidates=empty_candidates,
        true_corner_candidates=ambiguous,
    )

    assert review["true_corner_label_count"] == 0
    assert review["true_corner_source_evidence_counts"]["true_corners_or_pose"] == 0
    assert review["true_corner_source_evidence_counts"]["keypoint_or_mask"] == 0


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
