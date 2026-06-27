from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from threed.racketsport.testclips import REQUIRED_LABEL_FILES, build_testclip_manifest


def _write_labels(clip_dir: Path, labels: list[str]) -> None:
    labels_dir = clip_dir / "labels"
    labels_dir.mkdir(parents=True)
    for label in labels:
        (labels_dir / label).write_text("{}", encoding="utf-8")


def test_testclip_manifest_marks_clip_ready_when_required_labels_exist(tmp_path):
    root = tmp_path / "testclips"
    clip = root / "baseline_corner_001"
    _write_labels(clip, list(REQUIRED_LABEL_FILES))

    manifest = build_testclip_manifest(root)

    assert manifest.total_clips == 1
    assert manifest.ready_clips == 1
    assert manifest.not_ready_clips == 0
    assert manifest.is_ready is True
    assert manifest.clips[0].name == "baseline_corner_001"
    assert manifest.clips[0].missing_label_files == []
    assert manifest.clips[0].is_ready is True


def test_testclip_manifest_reports_missing_labels_and_counts(tmp_path):
    root = tmp_path / "testclips"
    clip = root / "side_fence_001"
    _write_labels(clip, ["court_corners.json", "players.json"])

    manifest = build_testclip_manifest(root)

    assert manifest.total_clips == 1
    assert manifest.ready_clips == 0
    assert manifest.not_ready_clips == 1
    assert manifest.is_ready is False
    assert manifest.label_file_counts["court_corners.json"] == 1
    assert manifest.label_file_counts["ball.json"] == 0
    assert "ball.json" in manifest.clips[0].missing_label_files
    assert "manual_metrics.json" in manifest.clips[0].missing_label_files


def test_validate_testclips_cli_emits_json_summary(tmp_path):
    root = tmp_path / "testclips"
    _write_labels(root / "ready", list(REQUIRED_LABEL_FILES))
    _write_labels(root / "missing_events", [label for label in REQUIRED_LABEL_FILES if label != "events.json"])

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/validate_testclips.py",
            "--root",
            str(root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert payload["schema_version"] == 1
    assert payload["root"] == str(root)
    assert payload["total_clips"] == 2
    assert payload["ready_clips"] == 1
    assert payload["not_ready_clips"] == 1
    assert payload["is_ready"] is False
    assert payload["clips"][1]["name"] == "missing_events"
    assert payload["clips"][1]["missing_label_files"] == ["events.json"]
