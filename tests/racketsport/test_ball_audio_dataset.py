from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.racketsport.validate_ball_audio_dataset import validate_manifest


def _write_manifest(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"schema_version": 1, "entries": entries}), encoding="utf-8")


def test_ball_audio_dataset_accepts_synthetic_ball_and_audio_sources(tmp_path):
    ball_path = tmp_path / "tracks" / "clip_001_ball.json"
    audio_path = tmp_path / "audio" / "clip_001_pop.wav"
    ball_path.parent.mkdir()
    audio_path.parent.mkdir()
    ball_path.write_text("{}", encoding="utf-8")
    audio_path.write_bytes(b"tiny synthetic wav placeholder")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {
                "id": "clip_001_ball",
                "path": "tracks/clip_001_ball.json",
                "split": "train",
                "source_type": "ball_track",
                "frame_rate": 60,
            },
            {
                "id": "clip_001_audio",
                "path": "audio/clip_001_pop.wav",
                "split": "train",
                "source_type": "pop_audio",
                "sample_rate": 44100,
            },
        ],
    )

    summary = validate_manifest(manifest_path)

    assert summary["valid"] is True
    assert summary["entry_count"] == 2
    assert summary["coverage_counts"]["source_type"] == {"ball_track": 1, "pop_audio": 1}
    assert summary["coverage_counts"]["split"] == {"test": 0, "train": 2, "val": 0}
    assert summary["coverage_gaps"] == []


def test_ball_audio_dataset_rejects_unsafe_relative_paths(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {
                "id": "unsafe",
                "path": "../outside.wav",
                "split": "train",
                "source_type": "pop_audio",
                "sample_rate": 44100,
            }
        ],
    )

    summary = validate_manifest(manifest_path)

    assert summary["valid"] is False
    assert "entries/0/path: must be relative and stay within the manifest directory" in summary["errors"]


def test_ball_audio_dataset_reports_gaps_without_failing_cli(tmp_path):
    ball_path = tmp_path / "tracks" / "clip_001_ball.json"
    ball_path.parent.mkdir()
    ball_path.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {
                "id": "clip_001_ball",
                "path": "tracks/clip_001_ball.json",
                "split": "val",
                "source_type": "ball_track",
                "frame_rate": 120,
            }
        ],
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/validate_ball_audio_dataset.py",
            str(manifest_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert completed.returncode == 0
    assert payload["valid"] is True
    assert payload["dataset_ready"] is False
    assert payload["coverage_counts"]["source_type"] == {"ball_track": 1, "pop_audio": 0}
    assert payload["coverage_gaps"] == ["missing pop_audio entries"]


def test_ball_audio_dataset_rejects_duplicate_entry_ids(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {"id": "dup", "path": "first.json", "split": "train", "source_type": "ball_track"},
            {"id": "dup", "path": "second.json", "split": "train", "source_type": "pop_audio", "sample_rate": 44100},
        ],
    )

    summary = validate_manifest(manifest_path)

    assert summary["valid"] is False
    assert "entries/1/id duplicate entry id: dup" in summary["errors"]
