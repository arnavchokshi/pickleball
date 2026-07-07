from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.racketsport.train_court_keypoint_heatmap import load_real_court_keypoint_labels
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS
from threed.racketsport.label_review import PROTOTYPE_GATE_CLIPS, export_cvat_tasks, export_review_bundle, import_corrected_labels


def _write_frame_pack(frames_root: Path, clip: str) -> None:
    clip_frames = frames_root / clip
    clip_frames.mkdir(parents=True)
    for name in ("frame_000001.jpg", "frame_000002.jpg"):
        (clip_frames / name).write_bytes(b"fake jpeg")
    (clip_frames / "label_frame_manifest.json").write_text(json.dumps({"frames": ["frame_000001.jpg", "frame_000002.jpg"]}), encoding="utf-8")


def _write_draft(drafts_root: Path, clip: str, label_file: str, items: list[dict]) -> None:
    labels = drafts_root / clip / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    (labels / label_file).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "draft_manual_annotation",
                "annotation": {"target_file": label_file, "items": items, "notes": []},
            }
        ),
        encoding="utf-8",
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, str]:
    clip = PROTOTYPE_GATE_CLIPS[0]
    drafts_root = tmp_path / "runs" / "label_drafts" / "prototype_gate"
    frames_root = tmp_path / "runs" / "label_frames"
    _write_frame_pack(frames_root, clip)
    _write_draft(
        drafts_root,
        clip,
        "court_corners.json",
        [
            {"review_id": "corner_review_1", "frame": "frame_000001.jpg", "status": "uncertain", "confidence": 0.42},
            {"review_id": "corner_confident_2", "frame": "frame_000002.jpg", "status": "accepted", "confidence": 0.98},
        ],
    )
    return drafts_root, frames_root, clip


def test_export_review_bundle_and_cvat_task(tmp_path: Path) -> None:
    drafts_root, frames_root, clip = _inputs(tmp_path)
    bundle = tmp_path / "review_bundle"

    summary = export_review_bundle(drafts_root=drafts_root, frames_root=frames_root, out=bundle)

    manifest = json.loads((bundle / "review_manifest.json").read_text(encoding="utf-8"))
    correction = json.loads((bundle / "corrections" / clip / "court_corners.json").read_text(encoding="utf-8"))
    assert summary["review_item_count"] == 1
    assert manifest["prototype_gate_clips"] == list(PROTOTYPE_GATE_CLIPS)
    assert manifest["clips"][0]["review_items"][0]["reason"] == "status=uncertain"
    assert (bundle / "images" / clip / "frame_000001.jpg").read_bytes() == b"fake jpeg"
    assert correction["status"] == "draft_prototype_corrections"
    assert correction["review_items"] == ["corner_review_1"]

    cvat = export_cvat_tasks(review_manifest=bundle / "review_manifest.json", out=tmp_path / "cvat")
    task = json.loads((tmp_path / "cvat" / clip / "task.json").read_text(encoding="utf-8"))
    assert cvat["task_count"] == 1
    assert task["labels"][0]["name"] == "court_corner"
    assert task["images"][0]["review_id"] == "corner_review_1"


def test_export_review_bundle_and_cvat_task_support_court_keypoint_review(tmp_path: Path) -> None:
    clip = PROTOTYPE_GATE_CLIPS[0]
    drafts_root = tmp_path / "runs" / "label_drafts" / "prototype_gate"
    frames_root = tmp_path / "runs" / "label_frames"
    _write_frame_pack(frames_root, clip)
    _write_draft(
        drafts_root,
        clip,
        "court_keypoints.json",
        [
            {
                "review_id": "court_keypoints_review_1",
                "frame": "frame_000001.jpg",
                "status": "uncertain",
                "confidence": 0.25,
            }
        ],
    )
    bundle = tmp_path / "review_bundle"

    summary = export_review_bundle(drafts_root=drafts_root, frames_root=frames_root, out=bundle)
    cvat = export_cvat_tasks(review_manifest=bundle / "review_manifest.json", out=tmp_path / "cvat")

    correction = json.loads((bundle / "corrections" / clip / "court_keypoints.json").read_text(encoding="utf-8"))
    task = json.loads((tmp_path / "cvat" / clip / "task.json").read_text(encoding="utf-8"))
    assert summary["review_item_count"] == 1
    assert cvat["status"] == "ready_for_cvat_review"
    assert correction["review_items"] == ["court_keypoints_review_1"]
    assert {"name": "court_keypoint", "attributes": ["keypoint_name"]} in task["labels"]
    assert task["images"][0]["target_file"] == "court_keypoints.json"


def test_export_cvat_tasks_exposes_blur_ready_ball_label_spec(tmp_path: Path) -> None:
    review_manifest = tmp_path / "review_manifest.json"
    frame = tmp_path / "frame_000060.jpg"
    frame.write_bytes(b"fake owner frame")
    review_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_owner_capture_review_manifest",
                "status": "candidate_prediction",
                "capture_id": "owner_IMG_1605_8a193402780b",
                "clips": [
                    {
                        "clip": "owner_IMG_1605_8a193402780b",
                        "review_items": [
                            {
                                "clip": "owner_IMG_1605_8a193402780b",
                                "frame": "frame_000060.jpg",
                                "frame_index": 60,
                                "image_path": str(frame),
                                "review_id": "ball_000060_000089",
                                "source_image_exists": True,
                                "target_file": "ball_track.json",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    export_cvat_tasks(review_manifest=review_manifest, out=tmp_path / "cvat")

    task = json.loads(
        (tmp_path / "cvat" / "owner_IMG_1605_8a193402780b" / "task.json").read_text(encoding="utf-8")
    )
    ball_label = next(label for label in task["labels"] if label["name"] == "ball")
    assert ball_label["attributes"] == [
        "visibility",
        "visibility_level",
        "center_convention",
        "blur_angle_deg",
        "blur_length_px",
        "blur_width_px",
        "blur_label_quality",
    ]
    assert ball_label["attribute_values"]["visibility_level"] == ["clear", "partial", "full", "out_of_frame"]
    assert ball_label["wbce_weights"] == {"clear": 1, "full": 3, "out_of_frame": 3, "partial": 2}
    assert "legacy_visible" in ball_label["legacy_visibility_mapping"]


def test_export_review_bundle_blocks_missing_review_images(tmp_path: Path) -> None:
    drafts_root, _frames_root, clip = _inputs(tmp_path)
    missing_frames_root = tmp_path / "missing_frames"
    bundle = tmp_path / "review_bundle"

    summary = export_review_bundle(drafts_root=drafts_root, frames_root=missing_frames_root, out=bundle)
    manifest = json.loads((bundle / "review_manifest.json").read_text(encoding="utf-8"))

    assert summary["status"] == "blocked_missing_review_images"
    assert manifest["missing_source_image_count"] == 1
    assert manifest["clips"][0]["review_items"][0]["source_image_exists"] is False
    assert not (bundle / "images" / clip / "frame_000001.jpg").exists()


def test_export_review_frames_cli_rejects_missing_review_images(tmp_path: Path) -> None:
    drafts_root, _frames_root, _clip = _inputs(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/export_review_frames.py",
            "--drafts-root",
            str(drafts_root),
            "--frames-root",
            str(tmp_path / "missing_frames"),
            "--out",
            str(tmp_path / "review_bundle"),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert json.loads(completed.stdout)["status"] == "blocked_missing_review_images"


def test_export_review_bundle_reports_no_review_items(tmp_path: Path) -> None:
    clip = PROTOTYPE_GATE_CLIPS[0]
    drafts_root = tmp_path / "runs" / "label_drafts" / "prototype_gate"
    frames_root = tmp_path / "runs" / "label_frames"
    _write_frame_pack(frames_root, clip)
    _write_draft(
        drafts_root,
        clip,
        "court_corners.json",
        [{"review_id": "corner_confident", "frame": "frame_000001.jpg", "status": "accepted", "confidence": 0.98}],
    )

    summary = export_review_bundle(drafts_root=drafts_root, frames_root=frames_root, out=tmp_path / "review_bundle")

    assert summary["status"] == "no_review_items"
    assert summary["review_item_count"] == 0
    assert summary["clips"] == []


def test_export_cvat_tasks_blocks_manifest_with_missing_images(tmp_path: Path) -> None:
    drafts_root, _frames_root, _clip = _inputs(tmp_path)
    bundle = tmp_path / "review_bundle"
    export_review_bundle(drafts_root=drafts_root, frames_root=tmp_path / "missing_frames", out=bundle)

    summary = export_cvat_tasks(review_manifest=bundle / "review_manifest.json", out=tmp_path / "cvat")

    assert summary["status"] == "blocked_missing_review_images"
    assert summary["missing_source_image_count"] == 1
    assert summary["tasks"][0]["image_count"] == 0
    task = json.loads(Path(summary["tasks"][0]["task_dir"]).joinpath("task.json").read_text(encoding="utf-8"))
    assert task["status"] == "blocked_missing_review_images"
    assert task["missing_source_image_count"] == 1


def test_export_cvat_tasks_cli_rejects_missing_images(tmp_path: Path) -> None:
    drafts_root, _frames_root, _clip = _inputs(tmp_path)
    bundle = tmp_path / "review_bundle"
    export_review_bundle(drafts_root=drafts_root, frames_root=tmp_path / "missing_frames", out=bundle)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/export_cvat_tasks.py",
            "--review-manifest",
            str(bundle / "review_manifest.json"),
            "--out",
            str(tmp_path / "cvat"),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert json.loads(completed.stdout)["status"] == "blocked_missing_review_images"


def test_import_corrected_labels_blocks_missing_draft_targets(tmp_path: Path) -> None:
    corrections_root = tmp_path / "corrections"
    correction_path = corrections_root / "clip_without_draft" / "court_corners.json"
    correction_path.parent.mkdir(parents=True)
    correction_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "clip": "clip_without_draft",
                "target_file": "court_corners.json",
                "items": [{"review_id": "corner_review_1"}],
            }
        ),
        encoding="utf-8",
    )

    summary = import_corrected_labels(drafts_root=tmp_path / "drafts", corrections_root=corrections_root)

    assert summary["status"] == "blocked_missing_drafts"
    assert summary["imported_item_count"] == 0
    assert summary["missing_draft_count"] == 1
    assert summary["skipped_corrections"][0]["correction_path"] == str(correction_path)


def test_import_corrections_roundtrip_cli(tmp_path: Path) -> None:
    drafts_root, frames_root, clip = _inputs(tmp_path)
    bundle = tmp_path / "review_bundle"
    export_review_bundle(drafts_root=drafts_root, frames_root=frames_root, out=bundle)
    correction_path = bundle / "corrections" / clip / "court_corners.json"
    correction = json.loads(correction_path.read_text(encoding="utf-8"))
    correction["items"] = [
        {
            "review_id": "corner_review_1",
            "frame": "frame_000001.jpg",
            "source": "human_review",
            "court_corners": {"far_left": [1, 2], "far_right": [3, 4], "near_right": [5, 6], "near_left": [7, 8]},
        }
    ]
    correction_path.write_text(json.dumps(correction), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/import_cvat_labels.py",
            "--drafts-root",
            str(drafts_root),
            "--corrections-root",
            str(bundle / "corrections"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["imported_item_count"] == 1
    draft = json.loads((drafts_root / clip / "labels" / "court_corners.json").read_text(encoding="utf-8"))
    assert draft["status"] == "draft_manual_annotation"
    assert draft["annotation"]["items"][0]["status"] == "corrected_unverified"
    assert draft["annotation"]["items"][0]["court_corners"]["near_right"] == [5, 6]


def test_import_reviewed_court_keypoint_corrections_preserves_training_ready_review_status(tmp_path: Path) -> None:
    clip = PROTOTYPE_GATE_CLIPS[0]
    drafts_root = tmp_path / "runs" / "label_drafts" / "prototype_gate"
    labels = drafts_root / clip / "labels"
    labels.mkdir(parents=True)
    (labels.parent / "source.mp4").write_bytes(b"fake video")
    keypoints = {
        point.name: [float(index + 10), float(index + 20)]
        for index, point in enumerate(PICKLEBALL_KEYPOINTS)
    }
    (labels / "court_keypoints.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "draft_manual_annotation",
                "annotation": {
                    "target_file": "court_keypoints.json",
                    "items": [
                        {
                            "review_id": "court_keypoints_review_1",
                            "frame": "frame_000001.jpg",
                            "status": "uncertain",
                            "confidence": 0.25,
                        }
                    ],
                },
                "frames": {
                    "frame_dir": str(tmp_path / "runs" / "label_frames" / clip),
                    "source_resolution": [1920, 1080],
                    "label_coordinate_space": [1920, 1080],
                },
            }
        ),
        encoding="utf-8",
    )
    corrections = tmp_path / "review_bundle" / "corrections"
    correction_path = corrections / clip / "court_keypoints.json"
    correction_path.parent.mkdir(parents=True)
    correction_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "clip": clip,
                "target_file": "court_keypoints.json",
                "review": {"status": "reviewed", "reviewer": "court-keypoint-review"},
                "items": [
                    {
                        "review_id": "court_keypoints_review_1",
                        "frame": "frame_000001.jpg",
                        "source": "human_review",
                        "keypoints": keypoints,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = import_corrected_labels(drafts_root=drafts_root, corrections_root=corrections)
    rows = load_real_court_keypoint_labels(drafts_root)

    draft = json.loads((labels / "court_keypoints.json").read_text(encoding="utf-8"))
    assert summary["imported_item_count"] == 1
    assert draft["review"] == {"status": "reviewed", "reviewer": "court-keypoint-review"}
    assert draft["annotation"]["items"][0]["status"] == "reviewed"
    assert rows[0]["clip"] == clip
    assert rows[0]["label_source"] == "reviewed_15_keypoint_court_labels"
    assert rows[0]["keypoints"]["near_left_corner"] == [10.0, 20.0]


def test_import_corrections_cli_rejects_noop_missing_draft_by_default(tmp_path: Path) -> None:
    corrections_root = tmp_path / "corrections"
    correction_path = corrections_root / "clip_without_draft" / "court_corners.json"
    correction_path.parent.mkdir(parents=True)
    correction_path.write_text(
        json.dumps({"clip": "clip_without_draft", "target_file": "court_corners.json", "items": [{"review_id": "corner_review_1"}]}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/import_cvat_labels.py",
            "--drafts-root",
            str(tmp_path / "drafts"),
            "--corrections-root",
            str(corrections_root),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["status"] == "blocked_missing_drafts"


def test_import_corrections_cli_can_allow_missing_drafts_explicitly(tmp_path: Path) -> None:
    corrections_root = tmp_path / "corrections"
    correction_path = corrections_root / "clip_without_draft" / "court_corners.json"
    correction_path.parent.mkdir(parents=True)
    correction_path.write_text(
        json.dumps({"clip": "clip_without_draft", "target_file": "court_corners.json", "items": [{"review_id": "corner_review_1"}]}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/import_cvat_labels.py",
            "--drafts-root",
            str(tmp_path / "drafts"),
            "--corrections-root",
            str(corrections_root),
            "--allow-missing-drafts",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "missing_drafts_allowed"
    assert payload["missing_draft_count"] == 1
