from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.racketsport.validate_shot_dataset import validate_manifest


SHOT_LABELS = (
    "serve",
    "fh_drive",
    "bh_drive",
    "dink",
    "lob",
    "overhead",
    "third_shot_drop",
    "reset_block",
)


def _write_manifest(root: Path, payload: dict) -> Path:
    manifest = root / "shot_dataset_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def _entry(
    entry_id: str,
    shot_label: str,
    source_type: str,
    split: str,
    path: str,
    **metadata: object,
) -> dict:
    return {
        "id": entry_id,
        "shot_label": shot_label,
        "source_type": source_type,
        "split": split,
        "path": path,
        **metadata,
    }


def test_validate_shot_dataset_accepts_manifest_with_all_shot_classes_and_sources(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    entries = []
    source_cycle = ("audio_snapped_pose", "manual_review", "synthetic_aug")
    split_cycle = ("train", "val", "test")
    for index, shot_label in enumerate(SHOT_LABELS):
        clip_path = clips / f"{shot_label}.json"
        clip_path.write_text("{}", encoding="utf-8")
        entries.append(
            _entry(
                f"shot_{index:03d}",
                shot_label,
                source_cycle[index % len(source_cycle)],
                split_cycle[index % len(split_cycle)],
                f"clips/{shot_label}.json",
                fps=120,
                contact_time_ms=250.0,
                window_ms=900,
                player_id="player_a",
                notes="tiny fixture",
            )
        )
    manifest = _write_manifest(
        tmp_path,
        {"schema_version": 1, "dataset_id": "tiny_shot_dataset", "entries": entries},
    )

    summary = validate_manifest(manifest)

    assert summary["valid"] is True
    assert summary["dataset_ready"] is True
    assert summary["entry_count"] == len(SHOT_LABELS)
    assert summary["coverage_counts"]["shot_label"] == {label: 1 for label in SHOT_LABELS}
    assert summary["coverage_counts"]["source_type"] == {
        "audio_snapped_pose": 3,
        "manual_review": 3,
        "synthetic_aug": 2,
    }
    assert summary["coverage_counts"]["split"] == {"test": 2, "train": 3, "val": 3}
    assert summary["coverage_gaps"] == []


def test_validate_shot_dataset_reports_coverage_gaps_without_failing_cli(tmp_path):
    clip = tmp_path / "clips" / "serve.json"
    clip.parent.mkdir()
    clip.write_text("{}", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_id": "train_only_seed",
            "entries": [
                _entry(
                    "serve_001",
                    "serve",
                    "audio_snapped_pose",
                    "train",
                    "clips/serve.json",
                )
            ],
        },
    )

    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/validate_shot_dataset.py", str(manifest)],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert completed.returncode == 0
    assert payload["valid"] is True
    assert payload["dataset_ready"] is False
    assert payload["coverage_counts"]["split"] == {"test": 0, "train": 1, "val": 0}
    assert payload["coverage_counts"]["shot_label"]["serve"] == 1
    assert payload["coverage_counts"]["shot_label"]["fh_drive"] == 0
    assert "missing val entries" in payload["coverage_gaps"]
    assert "missing test entries" in payload["coverage_gaps"]
    assert (
        "missing key shot classes: bh_drive, dink, fh_drive, lob, overhead, reset_block, third_shot_drop"
        in payload["coverage_gaps"]
    )


def test_validate_shot_dataset_rejects_duplicate_ids_unsafe_paths_and_bad_enums(tmp_path):
    inside = tmp_path / "clips" / "inside.json"
    inside.parent.mkdir()
    inside.write_text("{}", encoding="utf-8")
    outside = tmp_path.parent / "outside_shot.json"
    outside.write_text("{}", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_id": "invalid_seed",
            "entries": [
                _entry("dup", "serve", "audio_snapped_pose", "train", "clips/inside.json"),
                _entry("dup", "smash", "teacher_guess", "holdout", "../outside_shot.json"),
            ],
        },
    )

    summary = validate_manifest(manifest)

    assert summary["valid"] is False
    assert "entries/1/id duplicate entry id: dup" in summary["errors"]
    assert "entries/1/path: must be relative and stay within the manifest directory" in summary["errors"]
    assert (
        "entries/1/shot_label: must be one of bh_drive, dink, fh_drive, lob, overhead, reset_block, serve, third_shot_drop"
        in summary["errors"]
    )
    assert "entries/1/source_type: must be one of audio_snapped_pose, manual_review, synthetic_aug" in summary["errors"]
    assert "entries/1/split: must be one of test, train, val" in summary["errors"]


def test_validate_shot_dataset_cli_exits_one_for_invalid_manifest(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_id": "bad",
            "entries": [
                _entry("missing", "dink", "manual_review", "train", "missing.json")
            ],
        },
    )

    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/validate_shot_dataset.py", str(manifest)],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert completed.returncode == 1
    assert payload["valid"] is False
    assert "entries/0/path: file does not exist: missing.json" in payload["errors"]
