from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.racketsport import court_keypoint_review_server
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_review_task(root: Path, clip: str = "clip_a") -> None:
    task_dir = root / "runs" / "court_keypoint_review_20260701" / "cvat_tasks" / clip
    images = task_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    for frame in ("frame_000001.jpg", "frame_000002.jpg"):
        (images / frame).write_bytes(b"jpg")
    _write_json(
        task_dir / "task.json",
        {
            "schema_version": 1,
            "artifact_type": "racketsport_cvat_task",
            "status": "ready_for_cvat_review",
            "clip": clip,
            "images": [
                {
                    "file_name": "frame_000001.jpg",
                    "frame": "frame_000001.jpg",
                    "review_id": "court_keypoints_manual_15pt_0000",
                    "target_file": "court_keypoints.json",
                },
                {
                    "file_name": "frame_000002.jpg",
                    "frame": "frame_000002.jpg",
                    "review_id": "court_keypoints_manual_15pt_0001",
                    "target_file": "court_keypoints.json",
                },
            ],
        },
    )
    _write_json(
        root / "runs" / "court_keypoint_review_20260701" / "label_frames" / clip / "label_frame_manifest.json",
        {
            "schema_version": 1,
            "clip": clip,
            "frames": ["frame_000001.jpg", "frame_000002.jpg"],
            "frame_count": 2,
            "max_width": 1280,
            "sample_every_frames": 30,
            "source_resolution": [1920, 1080],
        },
    )


def _full_keypoints(offset: float = 0.0) -> dict[str, list[float]]:
    return {
        point.name: [float(index * 3 + 10 + offset), float(index * 2 + 20 + offset)]
        for index, point in enumerate(PICKLEBALL_KEYPOINTS)
    }


def test_manifest_discovers_court_keypoint_review_tasks(tmp_path: Path) -> None:
    _write_review_task(tmp_path)

    manifest = court_keypoint_review_server._manifest(tmp_path)

    assert manifest["review_type"] == "court_keypoint_review"
    assert manifest["progress_save_path"] == "runs/court_keypoint_review_20260701/local_court_keypoint_review_progress.json"
    assert [point["name"] for point in manifest["keypoints"]] == [point.name for point in PICKLEBALL_KEYPOINTS]
    assert manifest["clips"][0]["clip"] == "clip_a"
    assert manifest["clips"][0]["source_resolution"] == [1920, 1080]
    assert manifest["clips"][0]["label_coordinate_space"] == [1280, 720]
    assert manifest["clips"][0]["images"][0]["url"].endswith(
        "/asset?path=runs/court_keypoint_review_20260701/cvat_tasks/clip_a/images/frame_000001.jpg"
    )


def test_write_review_progress_exports_complete_reviewed_label_files(tmp_path: Path) -> None:
    _write_review_task(tmp_path)
    now = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    summary = court_keypoint_review_server._write_review_progress(
        tmp_path,
        {
            "schema_version": 1,
            "review_type": "court_keypoint_review",
            "clips": {
                "clip_a": {
                    "items": [
                        {
                            "frame": "frame_000001.jpg",
                            "review_id": "r1",
                            "status": "reviewed",
                            "keypoints": _full_keypoints(),
                        },
                        {
                            "frame": "frame_000002.jpg",
                            "review_id": "r2",
                            "status": "reviewed_static_camera_copy",
                            "keypoints": _full_keypoints(100.0),
                        },
                    ]
                }
            },
        },
        now=now,
    )

    label_path = tmp_path / "eval_clips" / "ball" / "clip_a" / "labels" / "court_keypoints.json"
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    assert summary["status"] == "saved"
    assert summary["exported_clip_count"] == 1
    assert summary["exported"][0]["label_path"] == str(label_path.relative_to(tmp_path))
    assert summary["exported"][0]["independent_reviewed_count"] == 1
    assert summary["exported"][0]["static_camera_copy_count"] == 1
    assert payload["review"] == {
        "status": "reviewed",
        "reviewer": "local_court_keypoint_review",
        "reviewed_at_utc": "2026-07-01T12:00:00+00:00",
        "independent_reviewed_count": 1,
        "static_camera_copy_count": 1,
    }
    assert payload["frames"]["frame_dir"] == "eval_clips/ball/clip_a/labels/court_keypoint_frames"
    assert payload["frames"]["source_resolution"] == [1920, 1080]
    assert payload["frames"]["label_coordinate_space"] == [1280, 720]
    assert len(payload["annotation"]["items"]) == 2
    # Per-item status must stay distinct through export -- the copy status is never
    # collapsed into "reviewed", and the independent review is never demoted either.
    items_by_frame = {item["frame"]: item for item in payload["annotation"]["items"]}
    assert items_by_frame["frame_000001.jpg"]["status"] == "reviewed"
    assert items_by_frame["frame_000002.jpg"]["status"] == "reviewed_static_camera_copy"
    assert payload["annotation"]["items"][1]["keypoints"]["near_left_corner"] == [110.0, 120.0]
    assert (tmp_path / "eval_clips" / "ball" / "clip_a" / "labels" / "court_keypoint_frames" / "frame_000001.jpg").read_bytes() == b"jpg"


def test_write_review_progress_rejects_unknown_item_status_without_writing(tmp_path: Path) -> None:
    _write_review_task(tmp_path)

    with pytest.raises(ValueError, match="status must be one of"):
        court_keypoint_review_server._write_review_progress(
            tmp_path,
            {
                "schema_version": 1,
                "review_type": "court_keypoint_review",
                "clips": {
                    "clip_a": {
                        "items": [
                            {
                                "frame": "frame_000001.jpg",
                                "status": "definitely_reviewed_trust_me",
                                "keypoints": _full_keypoints(),
                            },
                        ]
                    }
                },
            },
        )

    assert not (tmp_path / "runs" / "court_keypoint_review_20260701" / "local_court_keypoint_review_progress.json").exists()
    assert not (tmp_path / "eval_clips" / "ball" / "clip_a" / "labels" / "court_keypoints.json").exists()


def test_write_review_progress_accepts_all_enum_statuses(tmp_path: Path) -> None:
    _write_review_task(tmp_path)

    for status in ("in_progress", "reviewed", "reviewed_static_camera_copy"):
        summary = court_keypoint_review_server._write_review_progress(
            tmp_path,
            {
                "schema_version": 1,
                "review_type": "court_keypoint_review",
                "clips": {
                    "clip_a": {"items": [{"frame": "frame_000001.jpg", "status": status, "keypoints": _full_keypoints()}]}
                },
            },
        )
        # A full 15-point frame is always exportable regardless of its status label; only
        # the exported status text (and provenance counts) should vary with the input.
        assert summary["exported_clip_count"] == 1

    label_path = tmp_path / "eval_clips" / "ball" / "clip_a" / "labels" / "court_keypoints.json"
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    assert payload["annotation"]["items"][0]["status"] == "reviewed_static_camera_copy"
    assert payload["review"]["independent_reviewed_count"] == 0
    assert payload["review"]["static_camera_copy_count"] == 1


def test_write_review_progress_exports_completed_frames_even_when_clip_has_unlabeled_review_frames(tmp_path: Path) -> None:
    _write_review_task(tmp_path)

    summary = court_keypoint_review_server._write_review_progress(
        tmp_path,
        {
            "schema_version": 1,
            "review_type": "court_keypoint_review",
            "clips": {"clip_a": {"items": [{"frame": "frame_000001.jpg", "keypoints": _full_keypoints()}]}},
        },
    )

    label_path = tmp_path / "eval_clips" / "ball" / "clip_a" / "labels" / "court_keypoints.json"
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    assert summary["status"] == "saved_partial"
    assert summary["exported_clip_count"] == 1
    assert summary["incomplete"]["clip_a"]["missing_frame_count"] == 1
    assert payload["frames"]["frame_count"] == 1
    assert [item["frame"] for item in payload["annotation"]["items"]] == ["frame_000001.jpg"]
    exported_frames = sorted(
        path.name for path in (tmp_path / "eval_clips" / "ball" / "clip_a" / "labels" / "court_keypoint_frames").glob("*.jpg")
    )
    assert exported_frames == ["frame_000001.jpg"]


def test_write_review_progress_saves_partial_progress_without_exporting(tmp_path: Path) -> None:
    _write_review_task(tmp_path)
    partial = _full_keypoints()
    partial.pop("far_nvz_right")

    summary = court_keypoint_review_server._write_review_progress(
        tmp_path,
        {
            "schema_version": 1,
            "review_type": "court_keypoint_review",
            "clips": {"clip_a": {"items": [{"frame": "frame_000001.jpg", "keypoints": partial}]}},
        },
    )

    assert summary["status"] == "saved_partial"
    assert summary["exported_clip_count"] == 0
    assert summary["incomplete"]["clip_a"]["missing_frame_count"] == 1
    assert summary["incomplete"]["clip_a"]["missing_keypoint_count"] == 1
    assert (tmp_path / "runs" / "court_keypoint_review_20260701" / "local_court_keypoint_review_progress.json").is_file()
    assert not (tmp_path / "eval_clips" / "ball" / "clip_a" / "labels" / "court_keypoints.json").exists()


def test_write_review_progress_rejects_unknown_clip_without_writing(tmp_path: Path) -> None:
    _write_review_task(tmp_path)

    with pytest.raises(ValueError, match="unknown court keypoint review clip"):
        court_keypoint_review_server._write_review_progress(
            tmp_path,
            {
                "schema_version": 1,
                "review_type": "court_keypoint_review",
                "clips": {"../../evil": {"items": []}},
            },
        )

    assert not (tmp_path / "runs" / "court_keypoint_review_20260701" / "local_court_keypoint_review_progress.json").exists()
    assert not (tmp_path.parent / "evil").exists()


def test_court_keypoint_review_html_exposes_canvas_labeler() -> None:
    html = court_keypoint_review_server.HTML

    assert "courtKeypointLabeler" in html
    assert "keypointRail" in html
    assert "imageStage" in html
    assert "Save progress" in html


def test_court_keypoint_review_server_runs_by_script_path() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/racketsport/court_keypoint_review_server.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Serve a local UI for reviewed 15-point court keypoint labels" in result.stdout
