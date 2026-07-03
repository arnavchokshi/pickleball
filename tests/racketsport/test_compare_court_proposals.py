from __future__ import annotations

import json

import pytest

from scripts.racketsport.compare_court_proposals_to_reviewed_keypoints import (
    compare_proposal_to_reviewed_keypoints,
)
from threed.racketsport.court_keypoint_labels import build_partial_court_keypoint_label_payload
from threed.racketsport.schemas import PICKLEBALL_COURT_KEYPOINT_NAMES


IMG1605_PROGRESS = "runs/court_keypoint_review_20260701/local_court_keypoint_review_progress.json"
IMG1605_PROPOSAL = "runs/court_recheck_20260703T003107Z/img1605/court_corner_proposals.json"
IMG1605_CLIP = "owner_IMG_1605_8a193402780b"


def _reviewed_payload(label_points: dict[str, list[float]]) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_keypoint_labels",
        "clip": "owner_IMG_1605_8a193402780b",
        "annotation": {
            "items": [
                {
                    "frame": "frame_000151.jpg",
                    "keypoints": label_points,
                    "review_id": "img1605_court_keypoints_000151",
                    "status": "reviewed",
                }
            ]
        },
        "review": {"status": "reviewed"},
        "frames": {
            "frame_count": 1,
            "frame_dir": "does/not/matter",
            "label_coordinate_space": [540, 960],
            "source_resolution": [1080, 1920],
        },
    }


def test_compare_proposal_to_reviewed_keypoints_rescales_labels_and_scores_groups(tmp_path):
    label_points = {
        name: [20.0 + idx * 3.0, 40.0 + idx * 5.0]
        for idx, name in enumerate(PICKLEBALL_COURT_KEYPOINT_NAMES)
    }
    proposal_points_native = {
        name: {"xy": [xy[0] * 2.0, xy[1] * 2.0], "confidence": 0.9}
        for name, xy in label_points.items()
    }
    reviewed_path = tmp_path / "court_keypoints.json"
    proposal_path = tmp_path / "court_corner_proposals.json"
    reviewed_path.write_text(json.dumps(_reviewed_payload(label_points)), encoding="utf-8")
    proposal_path.write_text(
        json.dumps({"source": {"image_size": [1080, 1920]}, "keypoints": proposal_points_native}),
        encoding="utf-8",
    )

    summary = compare_proposal_to_reviewed_keypoints(
        reviewed_keypoints_path=reviewed_path,
        proposal_path=proposal_path,
    )

    assert summary["matched_keypoint_count"] == 15
    assert summary["groups"]["all"]["median_px"] == pytest.approx(0.0)
    assert summary["groups"]["planar_12"]["count"] == 12
    assert summary["groups"]["net_top_3"]["count"] == 3
    assert summary["missing_proposal_keypoints"] == []


def test_compare_proposal_to_reviewed_keypoints_uses_corner_fallback(tmp_path):
    label_points = {
        name: [20.0 + idx * 3.0, 40.0 + idx * 5.0]
        for idx, name in enumerate(PICKLEBALL_COURT_KEYPOINT_NAMES)
    }
    reviewed_path = tmp_path / "court_keypoints.json"
    proposal_path = tmp_path / "court_corner_proposals.json"
    reviewed_path.write_text(json.dumps(_reviewed_payload(label_points)), encoding="utf-8")
    proposal_path.write_text(
        json.dumps(
            {
                "source": {"image_size": [1080, 1920]},
                "corners": {
                    "near_left": {"xy": [label_points["near_left_corner"][0] * 2.0, label_points["near_left_corner"][1] * 2.0]},
                    "near_right": {"xy": [label_points["near_right_corner"][0] * 2.0, label_points["near_right_corner"][1] * 2.0]},
                    "far_right": {"xy": [label_points["far_right_corner"][0] * 2.0, label_points["far_right_corner"][1] * 2.0]},
                    "far_left": {"xy": [label_points["far_left_corner"][0] * 2.0, label_points["far_left_corner"][1] * 2.0]},
                },
            }
        ),
        encoding="utf-8",
    )

    summary = compare_proposal_to_reviewed_keypoints(
        reviewed_keypoints_path=reviewed_path,
        proposal_path=proposal_path,
    )

    assert summary["matched_keypoint_count"] == 4
    assert summary["groups"]["corners_4"]["median_px"] == pytest.approx(0.0)
    assert len(summary["missing_proposal_keypoints"]) == 11


def test_compare_img1605_partial_labels_rejects_current_bad_proposal(tmp_path):
    progress = json.loads(open(IMG1605_PROGRESS, encoding="utf-8").read())
    clip_payload = progress["clips"][IMG1605_CLIP]
    partial_path = tmp_path / "court_keypoints_partial.json"
    partial_path.write_text(
        json.dumps(
            build_partial_court_keypoint_label_payload(
                clip=IMG1605_CLIP,
                reviewer=clip_payload["reviewer"],
                items=clip_payload["items"],
                source_resolution=[1080, 1920],
                label_coordinate_space=[1080, 1920],
                available_review_frame_count=3,
                sample_every_frames=None,
                frame_dir="eval_clips/ball/owner_IMG_1605_8a193402780b/labels/court_keypoint_partial_frames",
                reviewed_at_utc="2026-07-03T00:00:00+00:00",
            )
        ),
        encoding="utf-8",
    )

    summary = compare_proposal_to_reviewed_keypoints(
        reviewed_keypoints_path=partial_path,
        proposal_path=IMG1605_PROPOSAL,
    )

    assert summary["matched_keypoint_count"] == 14
    assert summary["missing_reviewed_keypoints"] == ["near_left_corner"]
    assert summary["groups"]["all_visible"]["median_px"] > 100.0
    assert summary["groups"]["floor_visible"]["median_px"] > 200.0
    assert summary["groups"]["visible_corners"]["median_px"] > 300.0
    assert summary["verdict"] == "mandatory_user_confirmation_only"
    assert "visible_keypoint_residual_too_high" in summary["rejection_reasons"]
