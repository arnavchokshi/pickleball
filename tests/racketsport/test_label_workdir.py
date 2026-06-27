from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.racketsport.init_label_workdir import init_label_workdir
from threed.racketsport.testclips import REQUIRED_LABEL_FILES, build_testclip_manifest


def _write_metadata(clip_dir: Path) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "camera_height": "mid",
        "camera_angle": "side_fence",
        "play_type": "doubles",
        "environment": "outdoor",
        "frame_rate_fps": 120,
        "duration_s": 90.0,
        "racket_gt": True,
    }
    (clip_dir / "clip_metadata.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_frame_manifest(frames_root: Path, clip_name: str) -> None:
    clip_frames = frames_root / clip_name
    clip_frames.mkdir(parents=True)
    for name in ("frame_000001.jpg", "frame_000002.jpg"):
        (clip_frames / name).write_bytes(b"fake jpeg")
    manifest = {
        "schema_version": 1,
        "clip": clip_name,
        "frame_count": 2,
        "frames": ["frame_000001.jpg", "frame_000002.jpg"],
        "sample_every_frames": 30,
    }
    (clip_frames / "label_frame_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_init_label_workdir_writes_drafts_outside_dataset_labels(tmp_path):
    root = tmp_path / "data" / "testclips"
    clip = root / "candidate_001"
    _write_metadata(clip)
    frames_root = tmp_path / "runs" / "label_frames"
    _write_frame_manifest(frames_root, "candidate_001")
    out = tmp_path / "runs" / "label_drafts"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/init_label_workdir.py",
            "--root",
            str(root),
            "--frames-root",
            str(frames_root),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    draft_labels = out / "candidate_001" / "labels"
    dataset_labels = clip / "labels"

    assert payload["draft_ready"] is True
    assert payload["dataset_labels_written"] is False
    assert payload["clips"][0]["draft_labels_dir"] == str(draft_labels)
    assert sorted(path.name for path in draft_labels.glob("*.json")) == sorted(REQUIRED_LABEL_FILES)
    assert not dataset_labels.exists()

    manifest = build_testclip_manifest(root)
    assert manifest.ready_clips == 0
    assert manifest.clips[0].missing_label_files == list(REQUIRED_LABEL_FILES)


def test_draft_label_templates_include_manual_annotation_context(tmp_path):
    root = tmp_path / "data" / "testclips"
    clip = root / "candidate_001"
    _write_metadata(clip)
    frames_root = tmp_path / "runs" / "label_frames"
    _write_frame_manifest(frames_root, "candidate_001")
    out = tmp_path / "runs" / "label_drafts"

    subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/init_label_workdir.py",
            "--root",
            str(root),
            "--frames-root",
            str(frames_root),
            "--out",
            str(out),
        ],
        check=True,
    )

    draft = json.loads((out / "candidate_001" / "labels" / "events.json").read_text(encoding="utf-8"))

    assert draft["schema_version"] == 1
    assert draft["status"] == "draft_manual_annotation"
    assert draft["clip"]["name"] == "candidate_001"
    assert draft["clip"]["metadata"]["camera_angle"] == "side_fence"
    assert draft["frames"]["manifest_path"] == str(frames_root / "candidate_001" / "label_frame_manifest.json")
    assert draft["frames"]["frame_count"] == 2
    assert draft["frames"]["frames"] == [
        {
            "name": "frame_000001.jpg",
            "path": str(frames_root / "candidate_001" / "frame_000001.jpg"),
        },
        {
            "name": "frame_000002.jpg",
            "path": str(frames_root / "candidate_001" / "frame_000002.jpg"),
        },
    ]
    assert draft["annotation"]["target_file"] == "events.json"
    assert draft["annotation"]["items"] == []


def test_init_label_workdir_refuses_dataset_root_output(tmp_path):
    root = tmp_path / "data" / "testclips"
    _write_metadata(root / "candidate_001")

    with pytest.raises(ValueError, match="refusing to write draft labels into dataset labels path"):
        init_label_workdir(root=root, out=root)
