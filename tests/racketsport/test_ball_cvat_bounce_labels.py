from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from threed.racketsport.ball_cvat_bounce_labels import build_cvat_reviewed_bounce_inout_labels
from threed.racketsport.ball_manual_court_inout import manual_court_projection_from_corners
from threed.racketsport.court_calibration import project_planar_points


def _court_corners_payload(image_size: tuple[int, int] = (1920, 1080)) -> dict[str, object]:
    # Matches _cvat_payload()'s task.original_size by default: since
    # build_cvat_reviewed_bounce_inout_labels() now derives its
    # target_image_size from the CVAT task's own declared original_size
    # (Task #20), declaring the corners already in that same space keeps
    # this fixture's existing scale=1 numeric expectations intact.
    return {
        "annotation": {
            "items": [
                {
                    "court_corners": {
                        "near_left": [100.0, 500.0],
                        "near_right": [900.0, 500.0],
                        "far_right": [700.0, 100.0],
                        "far_left": [300.0, 100.0],
                    },
                    "frame": "frame_000001.jpg",
                    "image_size": list(image_size),
                    "source": "human_review",
                    "status": "corrected_unverified",
                }
            ]
        }
    }


def _image_for_world_xy(world_xy: list[float]) -> list[float]:
    projection = manual_court_projection_from_corners(_court_corners_payload(), sport="pickleball")
    return project_planar_points(projection["homography"], [world_xy])[0]


def _ball_box(frame_index: int, center_xy: list[float]) -> dict[str, object]:
    x, y = center_xy
    return {
        "track_id": 12,
        "label": "ball",
        "frame_index": frame_index,
        "bbox_xyxy": [x - 4.0, y - 4.0, x + 4.0, y + 4.0],
        "bbox_xywh": [x - 4.0, y - 4.0, 8.0, 8.0],
        "keyframe": True,
        "occluded": False,
        "source": "manual",
    }


def _cvat_payload(points: list[list[float]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_cvat_video_annotations",
        "clip_id": "clip_a",
        "source_format": "cvat_video_1_1",
        "source_path": "cvat_upload/exports/clip_a.zip",
        "task": {
            "task_id": 1,
            "name": "clip_a",
            "size": len(points),
            "mode": "interpolation",
            "start_frame": 0,
            "stop_frame": len(points) - 1,
            "original_size": [1920, 1080],
            "source": "source.mp4",
        },
        "frames": [
            {"frame_index": frame_index, "boxes": [_ball_box(frame_index, point)]}
            for frame_index, point in enumerate(points)
        ],
        "tracks": [
            {
                "track_id": 12,
                "label": "ball",
                "visible_box_count": len(points),
                "outside_box_count": 0,
                "keyframe_count": len(points),
                "first_visible_frame": 0,
                "last_visible_frame": len(points) - 1,
            }
        ],
        "summary": {
            "frame_count": len(points),
            "visible_box_count": len(points),
            "outside_box_count": 0,
            "labels": ["ball"],
            "track_count_by_label": {"ball": 1},
            "visible_box_count_by_label": {"ball": len(points)},
        },
    }


def test_build_cvat_reviewed_bounce_inout_labels_derives_ground_contact_and_call() -> None:
    points = [
        _image_for_world_xy([0.0, 1.0]),
        _image_for_world_xy([0.0, 0.2]),
        _image_for_world_xy([0.0, 0.0]),
        _image_for_world_xy([0.0, 0.2]),
        _image_for_world_xy([0.0, 1.0]),
    ]

    bounces, inout = build_cvat_reviewed_bounce_inout_labels(
        _cvat_payload(points),
        _court_corners_payload(),
        fps=60.0,
        min_vertical_delta_px=2.0,
    )

    assert bounces["artifact_type"] == "racketsport_reviewed_ball_bounces"
    assert bounces["status"] == "derived_from_human_reviewed_cvat_boxes"
    assert bounces["not_ground_truth"] is True
    assert bounces["reviewed_item_count"] == 1
    assert bounces["bounces"][0]["frame"] == 2
    assert bounces["bounces"][0]["review_id"] == "cvat_bounce_0000"
    assert inout["artifact_type"] == "racketsport_reviewed_ball_inout"
    assert inout["calls"] == [{"frame": 2, "t": pytest.approx(2.0 / 60.0), "call": "in", "review_id": "cvat_bounce_0000"}]


def test_build_cvat_reviewed_bounce_inout_labels_rescales_corners_to_cvat_original_size() -> None:
    """Task #20: this consumer must rescale declared corner pixels onto the
    CVAT task's own original_size rather than assuming they already match --
    build sample points in the *target* (CVAT) space via a correctly
    rescaled homography, then confirm feeding the small-preview-declared
    corners through build_cvat_reviewed_bounce_inout_labels (which derives
    target_image_size from labels.task.original_size automatically) recovers
    the same "in" call rather than the double/half-scaled miscall."""

    preview_size = (960, 540)
    native_size = (1920, 1080)
    small_corners = _court_corners_payload(image_size=preview_size)

    correct_projection = manual_court_projection_from_corners(small_corners, sport="pickleball", target_image_size=native_size)
    native_points = [
        project_planar_points(correct_projection["homography"], [world_xy])[0]
        for world_xy in ([0.0, 1.0], [0.0, 0.2], [0.0, 0.0], [0.0, 0.2], [0.0, 1.0])
    ]

    bounces, inout = build_cvat_reviewed_bounce_inout_labels(
        _cvat_payload(native_points),  # task.original_size == native_size == (1920, 1080)
        small_corners,
        fps=60.0,
        min_vertical_delta_px=2.0,
    )

    assert bounces["reviewed_item_count"] == 1
    assert inout["calls"] == [{"frame": 2, "t": pytest.approx(2.0 / 60.0), "call": "in", "review_id": "cvat_bounce_0000"}]
    assert inout["derivation"]["projection"]["declared_image_size"] == list(preview_size)
    assert inout["derivation"]["projection"]["target_image_size"] == list(native_size)
    assert inout["derivation"]["projection"]["corner_pixel_scale_applied"] == pytest.approx([2.0, 2.0])


def test_build_cvat_reviewed_bounce_inout_labels_rejects_non_cvat_skeleton() -> None:
    skeleton = {
        "schema_version": 1,
        "artifact_type": "racketsport_data1_label_skeleton",
        "status": "missing_human_review",
        "not_ground_truth": True,
        "items": [],
    }

    with pytest.raises(ValueError, match="CvatVideoAnnotations"):
        build_cvat_reviewed_bounce_inout_labels(skeleton, _court_corners_payload(), fps=60.0)


def test_build_cvat_reviewed_bounce_inout_cli_writes_outputs(tmp_path: Path) -> None:
    points = [
        _image_for_world_xy([0.0, 1.0]),
        _image_for_world_xy([0.0, 0.2]),
        _image_for_world_xy([0.0, 0.0]),
        _image_for_world_xy([0.0, 0.2]),
        _image_for_world_xy([0.0, 1.0]),
    ]
    cvat = tmp_path / "reviewed_boxes.json"
    corners = tmp_path / "court_corners.json"
    reviewed_bounces = tmp_path / "reviewed_bounces.json"
    reviewed_inout = tmp_path / "reviewed_inout.json"
    cvat.write_text(json.dumps(_cvat_payload(points)), encoding="utf-8")
    corners.write_text(json.dumps(_court_corners_payload()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/build_cvat_reviewed_ball_bounce_inout.py",
            "--cvat-labels",
            str(cvat),
            "--court-corners",
            str(corners),
            "--fps",
            "60",
            "--out-bounces",
            str(reviewed_bounces),
            "--out-inout",
            str(reviewed_inout),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(reviewed_bounces.read_text(encoding="utf-8"))["bounces"][0]["frame"] == 2
    assert json.loads(reviewed_inout.read_text(encoding="utf-8"))["calls"][0]["call"] == "in"
