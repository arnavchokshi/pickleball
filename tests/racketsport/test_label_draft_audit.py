from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.racketsport.audit_label_drafts import audit_label_drafts
from scripts.racketsport.init_label_workdir import init_label_workdir
from threed.racketsport.testclips import REQUIRED_LABEL_FILES


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
    manifest = {
        "schema_version": 1,
        "clip": clip_name,
        "frame_count": 2,
        "frames": ["frame_000001.jpg", "frame_000002.jpg"],
        "sample_every_frames": 30,
    }
    (clip_frames / "label_frame_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _build_ready_drafts(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "data" / "testclips"
    _write_metadata(root / "candidate_001")
    frames_root = tmp_path / "runs" / "label_frames"
    _write_frame_manifest(frames_root, "candidate_001")
    drafts_root = tmp_path / "runs" / "label_drafts"
    init_label_workdir(root=root, out=drafts_root, frames_root=frames_root)
    return root, drafts_root, frames_root


def test_audit_label_drafts_reports_ready_complete_drafts(tmp_path):
    root, drafts_root, frames_root = _build_ready_drafts(tmp_path)

    report = audit_label_drafts(root=root, drafts_root=drafts_root, frames_root=frames_root)

    assert report["status"] == "ready"
    assert report["dataset_labels_written"] is False
    assert report["counts"] == {
        "total_clips": 1,
        "ready_clips": 1,
        "not_ready_clips": 0,
        "missing_required_drafts": 0,
        "invalid_drafts": 0,
        "warnings": 0,
    }
    clip = report["clips"]["candidate_001"]
    assert clip["status"] == "ready"
    assert clip["missing_required_label_drafts"] == []
    assert clip["frame_pack"] == {
        "manifest_path": str(frames_root / "candidate_001" / "label_frame_manifest.json"),
        "present": True,
        "frame_count": 2,
        "sample_every_frames": 30,
        "warnings": [],
    }
    assert clip["drafts"]["events.json"]["present"] is True
    assert clip["drafts"]["events.json"]["json_valid"] is True
    assert clip["drafts"]["events.json"]["schema_version_ok"] is True
    assert clip["drafts"]["events.json"]["status_ok"] is True
    assert clip["drafts"]["events.json"]["target_file_ok"] is True
    assert clip["drafts"]["events.json"]["annotation_item_count"] == 0
    assert clip["drafts"]["events.json"]["errors"] == []


def test_audit_label_drafts_reports_missing_required_draft(tmp_path):
    root, drafts_root, frames_root = _build_ready_drafts(tmp_path)
    (drafts_root / "candidate_001" / "labels" / "ball.json").unlink()

    report = audit_label_drafts(root=root, drafts_root=drafts_root, frames_root=frames_root)

    assert report["status"] == "not_ready"
    assert report["counts"]["ready_clips"] == 0
    assert report["counts"]["missing_required_drafts"] == 1
    clip = report["clips"]["candidate_001"]
    assert clip["missing_required_label_drafts"] == ["ball.json"]
    assert clip["drafts"]["ball.json"] == {
        "path": str(drafts_root / "candidate_001" / "labels" / "ball.json"),
        "present": False,
        "json_valid": False,
        "schema_version_ok": False,
        "status_ok": False,
        "target_file_ok": False,
        "annotation_item_count": 0,
        "warnings": [],
        "errors": ["missing required label draft"],
    }


def test_audit_label_drafts_reports_malformed_json(tmp_path):
    root, drafts_root, frames_root = _build_ready_drafts(tmp_path)
    malformed = drafts_root / "candidate_001" / "labels" / "events.json"
    malformed.write_text("{not-json", encoding="utf-8")

    report = audit_label_drafts(root=root, drafts_root=drafts_root, frames_root=frames_root)

    assert report["status"] == "not_ready"
    assert report["counts"]["invalid_drafts"] == 1
    draft = report["clips"]["candidate_001"]["drafts"]["events.json"]
    assert draft["present"] is True
    assert draft["json_valid"] is False
    assert draft["annotation_item_count"] == 0
    assert any(error.startswith("invalid JSON:") for error in draft["errors"])


def test_audit_label_drafts_reports_target_file_mismatch(tmp_path):
    root, drafts_root, frames_root = _build_ready_drafts(tmp_path)
    draft_path = drafts_root / "candidate_001" / "labels" / "events.json"
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    draft["annotation"]["target_file"] = "ball.json"
    draft["annotation"]["items"] = [{"frame": "frame_000001.jpg"}]
    draft_path.write_text(json.dumps(draft), encoding="utf-8")

    report = audit_label_drafts(root=root, drafts_root=drafts_root, frames_root=frames_root)

    assert report["status"] == "not_ready"
    draft_report = report["clips"]["candidate_001"]["drafts"]["events.json"]
    assert draft_report["json_valid"] is True
    assert draft_report["target_file_ok"] is False
    assert draft_report["annotation_item_count"] == 1
    assert draft_report["errors"] == ["annotation.target_file must equal events.json"]


def test_audit_label_drafts_cli_emits_json_summary(tmp_path):
    root, drafts_root, frames_root = _build_ready_drafts(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/audit_label_drafts.py",
            "--root",
            str(root),
            "--drafts-root",
            str(drafts_root),
            "--frames-root",
            str(frames_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["status"] == "ready"
    assert payload["required_label_files"] == list(REQUIRED_LABEL_FILES)
