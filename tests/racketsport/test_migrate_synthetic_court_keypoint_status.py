from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.racketsport.migrate_synthetic_court_keypoint_status import (
    migrate_label_file,
    migrate_synthetic_corpus,
    sha256_file,
)


def test_run_migrate_synthetic_court_keypoint_status_cli_help_runs_from_repo_root() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/migrate_synthetic_court_keypoint_status.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()
    assert "--root" in completed.stdout


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_label_payload(status: str = "reviewed_static_camera_copy") -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_keypoint_labels",
        "clip": "synthetic_court_000000",
        "annotation": {
            "items": [
                {
                    "frame": "frame_000000.jpg",
                    "status": status,
                    "keypoints": {"near_left_corner": [1.0, 2.0]},
                    "provenance": {"synthetic": True, "human_reviewed": False},
                }
            ]
        },
        "review": {
            "status": "reviewed",
            "static_camera_copy_count": 1 if status == "reviewed_static_camera_copy" else 0,
            "synthetic_count": 0,
        },
    }


def _real_label_payload() -> dict:
    """A real owner-approved static-camera-copy row (NOT synthetic provenance) -- the migration
    must refuse to touch this even if it lived under the same root by mistake."""
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_keypoint_labels",
        "clip": "burlington_gold_0300_low_steep_corner",
        "annotation": {
            "items": [
                {
                    "frame": "frame_000010.jpg",
                    "status": "reviewed_static_camera_copy",
                    "keypoints": {"near_left_corner": [3.0, 4.0]},
                }
            ]
        },
        "review": {"status": "reviewed"},
    }


def test_migrate_label_file_rewrites_synthetic_status_and_review_counts(tmp_path: Path) -> None:
    label_path = tmp_path / "synthetic_court_000000" / "labels" / "court_keypoints.json"
    _write_json(label_path, _synthetic_label_payload())

    result = migrate_label_file(label_path)

    assert result == "migrated"
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    assert payload["annotation"]["items"][0]["status"] == "synthetic"
    assert payload["review"]["static_camera_copy_count"] == 0
    assert payload["review"]["synthetic_count"] == 1


def test_migrate_label_file_is_idempotent(tmp_path: Path) -> None:
    label_path = tmp_path / "synthetic_court_000000" / "labels" / "court_keypoints.json"
    _write_json(label_path, _synthetic_label_payload())
    migrate_label_file(label_path)

    result = migrate_label_file(label_path)

    assert result == "already_migrated"
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    assert payload["annotation"]["items"][0]["status"] == "synthetic"


def test_migrate_label_file_refuses_non_synthetic_provenance_rows(tmp_path: Path) -> None:
    label_path = tmp_path / "burlington_gold_0300_low_steep_corner" / "labels" / "court_keypoints.json"
    _write_json(label_path, _real_label_payload())

    with pytest.raises(ValueError, match="refusing to migrate"):
        migrate_label_file(label_path)

    # The real row must be left byte-for-byte untouched.
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    assert payload["annotation"]["items"][0]["status"] == "reviewed_static_camera_copy"


def test_migrate_synthetic_corpus_updates_manifest_label_sha256(tmp_path: Path) -> None:
    root = tmp_path / "court_synthetic"
    label_rel = "synthetic_court_000000/labels/court_keypoints.json"
    label_path = root / label_rel
    _write_json(label_path, _synthetic_label_payload())
    stale_sha256 = "0" * 64
    manifest_path = root / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "artifact_type": "synthetic_court_keypoint_corpus_manifest",
            "schema_notes": ["existing note"],
            "samples": [{"sample_id": "synthetic_court_000000", "label_path": label_rel, "label_sha256": stale_sha256}],
        },
    )

    report = migrate_synthetic_corpus(root)

    assert report["migrated"] == 1
    assert report["already_migrated"] == 0
    assert report["label_sha256_updates"] == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["samples"][0]["label_sha256"] == sha256_file(label_path)
    assert manifest["samples"][0]["label_sha256"] != stale_sha256
    assert any("CAL-R2 provenance-fix migration" in note for note in manifest["schema_notes"])
    assert "existing note" in manifest["schema_notes"]


def test_migrate_synthetic_corpus_is_a_no_op_when_already_migrated(tmp_path: Path) -> None:
    root = tmp_path / "court_synthetic"
    label_rel = "synthetic_court_000000/labels/court_keypoints.json"
    label_path = root / label_rel
    _write_json(label_path, _synthetic_label_payload(status="synthetic"))
    correct_sha256 = sha256_file(label_path)
    manifest_path = root / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "samples": [{"sample_id": "synthetic_court_000000", "label_path": label_rel, "label_sha256": correct_sha256}],
        },
    )
    manifest_mtime_before = manifest_path.stat().st_mtime_ns

    report = migrate_synthetic_corpus(root)

    assert report["migrated"] == 0
    assert report["already_migrated"] == 1
    assert report["label_sha256_updates"] == 0
    # No-op run must not rewrite the manifest file at all.
    assert manifest_path.stat().st_mtime_ns == manifest_mtime_before
