from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
