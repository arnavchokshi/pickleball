from __future__ import annotations

import json
from pathlib import Path

from threed.racketsport.court_keypoint_labels import (
    MISSING_OCCLUDED_OR_OFF_FRAME,
    PARTIAL_LABEL_ARTIFACT_TYPE,
    VISIBLE,
    build_partial_court_keypoint_label_payload,
    load_partial_court_keypoints,
)


IMG1605_PROGRESS = Path("runs/court_keypoint_review_20260701/local_court_keypoint_review_progress.json")
IMG1605_CLIP = "owner_IMG_1605_8a193402780b"


def test_img1605_progress_builds_partial_visible_label_payload(tmp_path: Path) -> None:
    progress = json.loads(IMG1605_PROGRESS.read_text(encoding="utf-8"))
    clip_payload = progress["clips"][IMG1605_CLIP]
    item = clip_payload["items"][0]

    payload = build_partial_court_keypoint_label_payload(
        clip=IMG1605_CLIP,
        reviewer=clip_payload["reviewer"],
        items=[item],
        source_resolution=[1080, 1920],
        label_coordinate_space=[1080, 1920],
        available_review_frame_count=3,
        sample_every_frames=None,
        frame_dir="eval_clips/ball/owner_IMG_1605_8a193402780b/labels/court_keypoint_partial_frames",
        reviewed_at_utc="2026-07-03T00:00:00+00:00",
    )

    assert payload["artifact_type"] == PARTIAL_LABEL_ARTIFACT_TYPE
    assert payload["clip"] == IMG1605_CLIP
    assert payload["review"]["status"] == "reviewed_partial"
    assert payload["review"]["not_full_metric15_calibration"] is True
    assert payload["frames"]["source_resolution"] == [1080, 1920]
    assert payload["frames"]["label_coordinate_space"] == [1080, 1920]

    exported_item = payload["annotation"]["items"][0]
    assert exported_item["frame"] == "frame_000151.jpg"
    assert exported_item["status"] == "reviewed_partial_visible"
    assert len(exported_item["keypoints"]) == 14
    assert "near_left_corner" not in exported_item["keypoints"]
    assert exported_item["visibility_by_keypoint"]["near_left_corner"] == MISSING_OCCLUDED_OR_OFF_FRAME
    assert exported_item["visibility_by_keypoint"]["far_left_corner"] == VISIBLE

    out_path = tmp_path / "court_keypoints_partial.json"
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    reviewed = load_partial_court_keypoints(out_path)
    assert reviewed.clip == IMG1605_CLIP
    assert reviewed.source_resolution == (1080.0, 1920.0)
    assert reviewed.label_coordinate_space == (1080.0, 1920.0)
    assert len(reviewed.frames) == 1
    assert reviewed.frames[0].visibility_by_keypoint["near_left_corner"] == MISSING_OCCLUDED_OR_OFF_FRAME
    assert set(reviewed.frames[0].keypoints) == set(exported_item["keypoints"])
