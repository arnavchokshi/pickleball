from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.racketsport.validate_ball_audio_dataset import validate_manifest


def _write_manifest(path: Path, entries: list[dict[str, object]], **metadata: object) -> None:
    path.write_text(json.dumps({"schema_version": 1, "entries": entries, **metadata}), encoding="utf-8")


def test_ball_audio_dataset_accepts_tracknet_roboflow_audio_and_augmentation_metadata(tmp_path):
    for directory in ("roboflow", "tracknet", "audio"):
        (tmp_path / directory).mkdir()
    (tmp_path / "roboflow" / "frame_001.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tracknet" / "frame_002.json").write_text("{}", encoding="utf-8")
    (tmp_path / "audio" / "pop_001.wav").write_bytes(b"tiny synthetic wav placeholder")
    (tmp_path / "audio" / "negative_001.wav").write_bytes(b"tiny synthetic wav placeholder")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {
                "id": "rf_visible",
                "path": "roboflow/frame_001.json",
                "split": "train",
                "source_type": "roboflow_ball_xy",
                "class_label": "ball_visible",
                "label_format": "roboflow_xy",
                "frame_rate": 60,
                "augmentations": ["motion_blur", "color_jitter", "h264_artifact"],
            },
            {
                "id": "tn_occluded",
                "path": "tracknet/frame_002.json",
                "split": "val",
                "source_type": "tracknet_ball_xy",
                "class_label": "ball_occluded",
                "label_format": "tracknet_xy",
                "frame_rate": 120,
                "visibility": 0,
            },
            {
                "id": "pop_audio",
                "path": "audio/pop_001.wav",
                "split": "test",
                "source_type": "pop_audio",
                "class_label": "pop",
                "sample_rate": 44100,
                "audio_format": "wav",
                "duration_ms": 100,
                "augmentations": ["noise_snr", "rir", "specaugment", "mixup"],
            },
            {
                "id": "negative_audio",
                "path": "audio/negative_001.wav",
                "split": "train",
                "source_type": "pop_audio",
                "class_label": "negative",
                "sample_rate": 44100,
                "audio_format": "wav",
            },
        ],
        dataset_id="tiny_ball_audio_dataset",
    )

    summary = validate_manifest(manifest_path)

    assert summary["valid"] is True
    assert summary["dataset_ready"] is True
    assert summary["entry_count"] == 4
    assert summary["coverage_counts"]["source_type"] == {
        "pop_audio": 2,
        "roboflow_ball_xy": 1,
        "tracknet_ball_xy": 1,
    }
    assert summary["coverage_counts"]["class_label"] == {
        "ball_occluded": 1,
        "ball_visible": 1,
        "negative": 1,
        "pop": 1,
    }
    assert summary["coverage_counts"]["split"] == {"test": 1, "train": 2, "val": 1}
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
                "class_label": "pop",
                "sample_rate": 44100,
            }
        ],
    )

    summary = validate_manifest(manifest_path)

    assert summary["valid"] is False
    assert "entries/0/path: must be relative and stay within the manifest directory" in summary["errors"]


def test_ball_audio_dataset_reports_split_class_and_source_gaps_without_failing_cli(tmp_path):
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
                "split": "train",
                "source_type": "roboflow_ball_xy",
                "class_label": "ball_visible",
                "label_format": "roboflow_xy",
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
    assert payload["coverage_counts"]["source_type"] == {
        "pop_audio": 0,
        "roboflow_ball_xy": 1,
        "tracknet_ball_xy": 0,
    }
    assert payload["coverage_counts"]["class_label"]["ball_visible"] == 1
    assert "missing val entries" in payload["coverage_gaps"]
    assert "missing test entries" in payload["coverage_gaps"]
    assert "missing source types: pop_audio, tracknet_ball_xy" in payload["coverage_gaps"]
    assert "missing key classes: ball_occluded, negative, pop" in payload["coverage_gaps"]


def test_ball_audio_dataset_rejects_duplicate_entry_ids(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {
                "id": "dup",
                "path": "first.json",
                "split": "train",
                "source_type": "roboflow_ball_xy",
                "class_label": "ball_visible",
                "label_format": "roboflow_xy",
            },
            {
                "id": "dup",
                "path": "second.json",
                "split": "train",
                "source_type": "pop_audio",
                "class_label": "pop",
                "sample_rate": 44100,
            },
        ],
    )

    summary = validate_manifest(manifest_path)

    assert summary["valid"] is False
    assert "entries/1/id duplicate entry id: dup" in summary["errors"]


def test_ball_audio_dataset_rejects_audio_sample_rates_other_than_44100(tmp_path):
    audio_path = tmp_path / "audio" / "pop.wav"
    audio_path.parent.mkdir()
    audio_path.write_bytes(b"placeholder")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        [
            {
                "id": "bad_rate",
                "path": "audio/pop.wav",
                "split": "train",
                "source_type": "pop_audio",
                "class_label": "pop",
                "sample_rate": 48000,
            }
        ],
    )

    summary = validate_manifest(manifest_path)

    assert summary["valid"] is False
    assert "entries/0/sample_rate: pop_audio sources must be 44100 Hz" in summary["errors"]


def test_ball_audio_dataset_schema_file_is_valid_json():
    schema_path = Path("docs/racketsport/ball_audio_dataset_schema.json")

    completed = subprocess.run(
        [sys.executable, "-m", "json.tool", str(schema_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
