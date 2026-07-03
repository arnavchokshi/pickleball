from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.racketsport import court_keypoint_review_server
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS


def _start_court_keypoint_server(tmp_path: Path, *, token: str = "test-review-token"):
    server = court_keypoint_review_server.ThreadingHTTPServer(
        ("127.0.0.1", 0), court_keypoint_review_server.CourtKeypointReviewHandler
    )
    server.repo_root = tmp_path  # type: ignore[attr-defined]
    server.save_token = token  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_court_keypoint_server(server, thread) -> None:  # noqa: ANN001
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


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

    partial_label_path = tmp_path / "eval_clips" / "ball" / "clip_a" / "labels" / "court_keypoints_partial.json"
    partial_payload = json.loads(partial_label_path.read_text(encoding="utf-8"))
    assert partial_payload["artifact_type"] == "racketsport_court_keypoint_partial_labels"
    assert partial_payload["review"]["status"] == "reviewed_partial"
    assert partial_payload["review"]["not_full_metric15_calibration"] is True
    assert partial_payload["annotation"]["items"][0]["status"] == "reviewed_partial_visible"
    assert partial_payload["annotation"]["items"][0]["visibility_by_keypoint"]["far_nvz_right"] == "missing_occluded_or_off_frame"
    assert partial_payload["annotation"]["items"][0]["visibility_by_keypoint"]["near_left_corner"] == "visible"


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


def test_write_review_progress_export_survives_concurrent_saves_without_corruption(tmp_path: Path) -> None:
    # Regression test for review_harden_20260702.md finding 2: this is the
    # more damaging save surface (it writes eval_clips/ball/*/labels/
    # court_keypoints.json directly), so concurrent saves for the same clip
    # must never leave the progress JSON, label JSON, or exported frame in a
    # torn/interleaved state.
    _write_review_task(tmp_path)
    errors: list[BaseException] = []

    def worker(offset: float) -> None:
        try:
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
                                    "status": "reviewed",
                                    "keypoints": _full_keypoints(offset),
                                }
                            ]
                        }
                    },
                },
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(float(i),)) for i in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    progress_path = tmp_path / "runs" / "court_keypoint_review_20260701" / "local_court_keypoint_review_progress.json"
    label_path = tmp_path / "eval_clips" / "ball" / "clip_a" / "labels" / "court_keypoints.json"
    # json.loads raising would mean a torn/interleaved write landed on disk.
    progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
    label_payload = json.loads(label_path.read_text(encoding="utf-8"))
    assert progress_payload["clips"]["clip_a"]["items"][0]["frame"] == "frame_000001.jpg"
    assert len(label_payload["annotation"]["items"]) == 1
    frame_path = tmp_path / "eval_clips" / "ball" / "clip_a" / "labels" / "court_keypoint_frames" / "frame_000001.jpg"
    assert frame_path.read_bytes() == b"jpg"
    leftover_temp = list(frame_path.parent.glob(".*.tmp*"))
    assert leftover_temp == []


def test_copy_export_frames_removes_only_frames_outside_new_set(tmp_path: Path) -> None:
    _write_review_task(tmp_path)
    entry = court_keypoint_review_server._task_entries(tmp_path)["clip_a"]
    out_dir = tmp_path / "eval_clips" / "ball" / "clip_a" / "labels" / "court_keypoint_frames"
    out_dir.mkdir(parents=True)
    (out_dir / "frame_000009.jpg").write_bytes(b"stale")

    court_keypoint_review_server._copy_export_frames(
        tmp_path,
        entry,
        images=[entry["images"][0]],
        exported_frame_dir=Path("eval_clips/ball/clip_a/labels/court_keypoint_frames"),
    )

    remaining = sorted(path.name for path in out_dir.glob("*.jpg"))
    assert remaining == ["frame_000001.jpg"]
    assert not list(out_dir.glob(".*.tmp*"))


def test_court_keypoint_post_save_rejects_missing_or_wrong_token(tmp_path: Path) -> None:
    _write_review_task(tmp_path)
    server, thread = _start_court_keypoint_server(tmp_path, token="correct-horse-battery-staple")
    try:
        port = server.server_address[1]
        body = json.dumps(
            {
                "schema_version": 1,
                "review_type": "court_keypoint_review",
                "clips": {"clip_a": {"items": [{"frame": "frame_000001.jpg", "keypoints": _full_keypoints()}]}},
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/save",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(request, timeout=5)
        assert exc_info.value.code == 401

        bad_request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/save",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "X-Review-Token": "wrong-token"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info_bad:
            urllib.request.urlopen(bad_request, timeout=5)
        assert exc_info_bad.value.code == 401

        assert not (
            tmp_path / "runs" / "court_keypoint_review_20260701" / "local_court_keypoint_review_progress.json"
        ).exists()
    finally:
        _stop_court_keypoint_server(server, thread)


def test_court_keypoint_post_save_accepts_correct_token(tmp_path: Path) -> None:
    _write_review_task(tmp_path)
    server, thread = _start_court_keypoint_server(tmp_path, token="correct-horse-battery-staple")
    try:
        port = server.server_address[1]
        body = json.dumps(
            {
                "schema_version": 1,
                "review_type": "court_keypoint_review",
                "clips": {"clip_a": {"items": [{"frame": "frame_000001.jpg", "keypoints": _full_keypoints()}]}},
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/save",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "X-Review-Token": "correct-horse-battery-staple"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200
            result = json.loads(response.read().decode("utf-8"))
        assert result["status"] == "saved_partial"
        assert (
            tmp_path / "runs" / "court_keypoint_review_20260701" / "local_court_keypoint_review_progress.json"
        ).is_file()
    finally:
        _stop_court_keypoint_server(server, thread)


def test_court_keypoint_served_page_embeds_save_token_not_placeholder(tmp_path: Path) -> None:
    _write_review_task(tmp_path)
    server, thread = _start_court_keypoint_server(tmp_path, token="page-embedded-token")
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            html = response.read().decode("utf-8")
        assert "page-embedded-token" in html
        assert court_keypoint_review_server.SAVE_TOKEN_PLACEHOLDER not in html
    finally:
        _stop_court_keypoint_server(server, thread)
