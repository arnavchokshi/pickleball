from __future__ import annotations

import json
from pathlib import Path

import pytest

from threed.racketsport.court_keypoint_labels import (
    build_partial_court_keypoint_label_payload,
    load_partial_court_keypoints,
)
from threed.racketsport.court_partial_geometry import fit_visible_floor_homography


IMG1605_PROGRESS = Path("runs/court_keypoint_review_20260701/local_court_keypoint_review_progress.json")
IMG1605_CLIP = "owner_IMG_1605_8a193402780b"


def _img1605_partial_labels(tmp_path: Path):
    progress = json.loads(IMG1605_PROGRESS.read_text(encoding="utf-8"))
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
    return load_partial_court_keypoints(partial_path)


def test_visible_floor_homography_infers_img1605_occluded_corner(tmp_path):
    partial = _img1605_partial_labels(tmp_path)

    fit = fit_visible_floor_homography(partial)

    assert fit.used_keypoints == [
        "near_baseline_center",
        "near_right_corner",
        "far_right_corner",
        "far_baseline_center",
        "far_left_corner",
        "near_nvz_left",
        "near_nvz_center",
        "near_nvz_right",
        "far_nvz_left",
        "far_nvz_center",
        "far_nvz_right",
    ]
    assert fit.excluded_keypoints == ["net_left_sideline", "net_center", "net_right_sideline"]
    assert fit.residual_summary_px["median_px"] < 5.0
    assert fit.residual_summary_px["max_px"] < 10.0
    assert fit.inferred_keypoints["near_left_corner"]["source"] == "visible_floor_homography"
    assert fit.inferred_keypoints["near_left_corner"]["xy"][0] < 0.0
    assert fit.inferred_keypoints["near_left_corner"]["xy"][1] == pytest.approx(1207.0, abs=10.0)


def test_visible_floor_homography_requires_four_floor_points(tmp_path):
    partial = _img1605_partial_labels(tmp_path)
    partial.frames[0].keypoints.clear()
    partial.frames[0].visibility_by_keypoint.update(
        {
            "net_left_sideline": "visible",
            "net_center": "visible",
            "net_right_sideline": "visible",
        }
    )
    partial.frames[0].keypoints.update(
        {
            "net_left_sideline": (509.0, 1106.0),
            "net_center": (717.0, 1131.0),
            "net_right_sideline": (1018.0, 1173.0),
        }
    )

    with pytest.raises(ValueError, match="at least 4 visible floor keypoints"):
        fit_visible_floor_homography(partial)
