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


def _write_metadata(
    clip_dir: Path,
    *,
    camera_height: str = "mid",
    camera_angle: str = "side_fence",
    play_type: str = "doubles",
    environment: str = "indoor",
    frame_rate_fps: int = 60,
    duration_s: float = 90.0,
    racket_gt: bool = False,
) -> None:
    payload = {
        "schema_version": 1,
        "camera_height": camera_height,
        "camera_angle": camera_angle,
        "play_type": play_type,
        "environment": environment,
        "frame_rate_fps": frame_rate_fps,
        "duration_s": duration_s,
        "racket_gt": racket_gt,
    }
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "clip_metadata.json").write_text(json.dumps(payload), encoding="utf-8")


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


def test_testclip_manifest_reads_clip_metadata_and_reports_matrix_gaps(tmp_path):
    root = tmp_path / "testclips"
    clip = root / "side_fence_001"
    _write_labels(clip, list(REQUIRED_LABEL_FILES))
    _write_metadata(clip, camera_height="mid", camera_angle="side_fence", environment="outdoor")

    manifest = build_testclip_manifest(root)

    assert manifest.clips[0].metadata_present is True
    assert manifest.clips[0].metadata_errors == []
    assert manifest.clips[0].metadata.camera_angle == "side_fence"
    assert manifest.coverage_counts["camera_angle"]["side_fence"] == 1
    assert manifest.meets_dataset_matrix is False
    assert "need at least 24 clips with valid metadata" in manifest.coverage_gaps


def test_testclip_manifest_passes_matrix_when_minimum_coverage_exists(tmp_path):
    root = tmp_path / "testclips"
    rows = []
    for idx in range(24):
        rows.append(
            {
                "camera_height": ["low", "mid", "high"][idx % 3],
                "camera_angle": ["shallow_baseline", "steep_corner", "side_fence", "near_overhead"][idx % 4],
                "play_type": "doubles" if idx < 10 else "singles_drill" if idx < 20 else "messy_real_world",
                "environment": "indoor" if idx < 12 else "outdoor",
                "frame_rate_fps": 120 if idx < 6 else 240 if idx < 8 else 60,
                "duration_s": 960.0 if idx < 4 else 90.0,
                "racket_gt": idx < 3,
            }
        )
    for idx, metadata in enumerate(rows):
        clip = root / f"clip_{idx:02d}"
        _write_labels(clip, list(REQUIRED_LABEL_FILES))
        _write_metadata(clip, **metadata)

    manifest = build_testclip_manifest(root)

    assert manifest.ready_clips == 24
    assert manifest.metadata_ready_clips == 24
    assert manifest.meets_dataset_matrix is True
    assert manifest.coverage_gaps == []


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
    assert payload["dataset_ready"] is False


def test_validate_testclips_cli_exits_zero_only_when_labels_and_matrix_are_ready(tmp_path):
    root = tmp_path / "testclips"
    for idx in range(24):
        clip = root / f"clip_{idx:02d}"
        _write_labels(clip, list(REQUIRED_LABEL_FILES))
        _write_metadata(
            clip,
            camera_height=["low", "mid", "high"][idx % 3],
            camera_angle=["shallow_baseline", "steep_corner", "side_fence", "near_overhead"][idx % 4],
            play_type="doubles" if idx < 10 else "singles_drill" if idx < 20 else "messy_real_world",
            environment="indoor" if idx < 12 else "outdoor",
            frame_rate_fps=120 if idx < 6 else 240 if idx < 8 else 60,
            duration_s=960.0 if idx < 4 else 90.0,
            racket_gt=idx < 3,
        )

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
    assert completed.returncode == 0
    assert payload["is_ready"] is True
    assert payload["meets_dataset_matrix"] is True
    assert payload["dataset_ready"] is True
