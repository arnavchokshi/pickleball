from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_manifest(root: Path, payload: dict) -> Path:
    manifest = root / "racket_dataset_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def _entry(entry_id: str, source_type: str, path: str, **metadata: object) -> dict:
    return {"id": entry_id, "source_type": source_type, "path": path, **metadata}


def test_validate_racket_dataset_accepts_valid_mixed_sources(tmp_path):
    (tmp_path / "racketvision").mkdir()
    (tmp_path / "synthetic").mkdir()
    (tmp_path / "aruco").mkdir()
    (tmp_path / "racketvision" / "rv_clip.json").write_text("{}", encoding="utf-8")
    (tmp_path / "synthetic" / "blend_pose.json").write_text("{}", encoding="utf-8")
    (tmp_path / "aruco" / "trial_001.json").write_text("{}", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_id": "tiny_racket_sources",
            "sources": [
                _entry(
                    "rv_clip",
                    "racketvision",
                    "racketvision/rv_clip.json",
                    camera={"camera_id": "cam_a", "view": "side"},
                    fps=120,
                ),
                _entry("blend_pose", "synthetic_blenderproc", "synthetic/blend_pose.json", fps=240),
                _entry(
                    "aruco_trial",
                    "aruco_gt",
                    "aruco/trial_001.json",
                    marker={"dictionary": "DICT_4X4_50", "marker_size_m": 0.04},
                ),
            ],
        },
    )

    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/validate_racket_dataset.py", str(manifest)],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert payload["valid"] is True
    assert payload["total_sources"] == 3
    assert payload["source_type_counts"] == {
        "aruco_gt": 1,
        "racketvision": 1,
        "synthetic_blenderproc": 1,
    }
    assert payload["coverage_summary"]["has_aruco_gt"] is True
    assert payload["coverage_summary"]["gaps"] == []


def test_validate_racket_dataset_reports_missing_aruco_without_failing(tmp_path):
    (tmp_path / "racketvision").mkdir()
    (tmp_path / "racketvision" / "rv_clip.json").write_text("{}", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_id": "no_aruco_yet",
            "sources": [_entry("rv_clip", "racketvision", "racketvision/rv_clip.json")],
        },
    )

    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/validate_racket_dataset.py", str(manifest)],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert payload["valid"] is True
    assert payload["source_type_counts"] == {
        "aruco_gt": 0,
        "racketvision": 1,
        "synthetic_blenderproc": 0,
    }
    assert payload["coverage_summary"]["has_aruco_gt"] is False
    assert payload["coverage_summary"]["gaps"] == [
        "no aruco_gt entries registered; ArUco-GT coverage is missing"
    ]


def test_validate_racket_dataset_rejects_unsafe_paths(tmp_path):
    outside = tmp_path.parent / "outside_racket.json"
    outside.write_text("{}", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_id": "unsafe",
            "sources": [_entry("escape", "racketvision", "../outside_racket.json")],
        },
    )

    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/validate_racket_dataset.py", str(manifest)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "sources/0/path must be relative and stay within the manifest directory" in completed.stderr
