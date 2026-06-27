from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.racketsport.report_testclip_coverage import build_coverage_report
from threed.racketsport.testclips import REQUIRED_LABEL_FILES


def _write_labels(clip_dir: Path, labels: list[str]) -> None:
    labels_dir = clip_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for label in labels:
        (labels_dir / label).write_text("{}", encoding="utf-8")


def _write_metadata(
    clip_dir: Path,
    *,
    camera_height: str = "mid",
    camera_angle: str = "side_fence",
    play_type: str = "doubles",
    environment: str = "outdoor",
    frame_rate_fps: int = 120,
    duration_s: float = 90.0,
    racket_gt: bool = False,
) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
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
    (clip_dir / "clip_metadata.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_frame_manifest(frames_root: Path, clip_name: str, frame_count: int) -> None:
    clip_frames = frames_root / clip_name
    clip_frames.mkdir(parents=True, exist_ok=True)
    frames = [f"frame_{idx:06d}.jpg" for idx in range(1, frame_count + 1)]
    for name in frames:
        (clip_frames / name).write_bytes(b"fake jpeg")
    payload = {
        "schema_version": 1,
        "clip": clip_name,
        "frame_count": frame_count,
        "frames": frames,
        "sample_every_frames": 30,
    }
    (clip_frames / "label_frame_manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_build_coverage_report_keeps_ready_false_and_counts_frame_packs(tmp_path):
    root = tmp_path / "data" / "testclips"
    frames_root = tmp_path / "runs" / "label_frames"
    _write_metadata(root / "candidate_needs_labels", camera_angle="side_fence")
    _write_metadata(root / "ready_but_matrix_gap", camera_angle="steep_corner", racket_gt=True)
    _write_labels(root / "ready_but_matrix_gap", list(REQUIRED_LABEL_FILES))
    _write_frame_manifest(frames_root, "candidate_needs_labels", 3)

    report = build_coverage_report(root=root, frames_root=frames_root)

    assert report["schema_version"] == 1
    assert report["ready"] is False
    assert report["label_readiness"]["ready"] is False
    assert report["label_readiness"]["ready_clips"] == 1
    assert report["label_readiness"]["not_ready_clips"] == 1
    assert report["matrix"]["ready"] is False
    assert "need at least 24 clips with valid metadata" in report["matrix"]["missing_coverage"]
    assert report["frame_packs"]["total_frames"] == 3
    assert report["frame_packs"]["clips_with_frame_packs"] == 1
    assert report["clips"]["candidate_needs_labels"]["ready"] is False
    assert report["clips"]["candidate_needs_labels"]["missing_label_files"] == list(REQUIRED_LABEL_FILES)
    assert report["clips"]["candidate_needs_labels"]["frame_pack"]["frame_count"] == 3
    assert report["clips"]["ready_but_matrix_gap"]["ready"] is True
    assert report["clips"]["ready_but_matrix_gap"]["frame_pack"]["frame_count"] == 0


def test_report_cli_prints_json_by_default_and_does_not_write_dataset_labels(tmp_path):
    root = tmp_path / "data" / "testclips"
    frames_root = tmp_path / "runs" / "label_frames"
    _write_metadata(root / "candidate_needs_labels")
    _write_frame_manifest(frames_root, "candidate_needs_labels", 2)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/report_testclip_coverage.py",
            "--root",
            str(root),
            "--frames-root",
            str(frames_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)

    assert completed.returncode == 0
    assert payload["ready"] is False
    assert payload["label_readiness"]["not_ready_clips"] == 1
    assert payload["frame_packs"]["total_frames"] == 2
    assert not (root / "candidate_needs_labels" / "labels").exists()


def test_report_cli_can_write_markdown_triage_report(tmp_path):
    root = tmp_path / "data" / "testclips"
    frames_root = tmp_path / "runs" / "label_frames"
    markdown_path = tmp_path / "coverage.md"
    _write_metadata(root / "candidate_needs_labels")
    _write_frame_manifest(frames_root, "candidate_needs_labels", 4)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/report_testclip_coverage.py",
            "--root",
            str(root),
            "--frames-root",
            str(frames_root),
            "--markdown-out",
            str(markdown_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert completed.returncode == 0
    assert payload["markdown_report"] == str(markdown_path)
    assert "# DATA-1 Test Clip Coverage Triage" in markdown
    assert "Ready: false" in markdown
    assert "candidate_needs_labels" in markdown
    assert "need at least 24 clips with valid metadata" in markdown
    assert "| candidate_needs_labels | false | 9 | 4 |" in markdown
    assert not (root / "candidate_needs_labels" / "labels").exists()
