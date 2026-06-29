from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_manifest(root: Path, payload: dict) -> Path:
    manifest = root / "pose_dataset_manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def _source(source_id: str, source_type: str, path: str, split: str, **metadata: object) -> dict:
    return {"id": source_id, "source_type": source_type, "path": path, "split": split, **metadata}


def _run_validator(manifest: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/racketsport/validate_pose_dataset.py", str(manifest)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_validate_pose_dataset_accepts_complete_body_ladder_and_eval_sources(tmp_path):
    for directory in ("bedlam2", "athletepose3d", "caltennis", "rich", "amass", "emdb"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "manifest.json").write_text(
            json.dumps({"source": directory, "frames": [{"t": 0.0}]}),
            encoding="utf-8",
        )
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_id": "tiny_pose_sources",
            "sources": [
                _source(
                    "bedlam2_seed",
                    "bedlam2",
                    "bedlam2/manifest.json",
                    "train",
                    fps=30,
                    frame_count=8000000,
                    joint_set="SMPL-X",
                    license="research",
                ),
                _source("athletepose3d_train", "athletepose3d", "athletepose3d/manifest.json", "train", fps=120),
                _source("caltennis_val", "caltennis", "caltennis/manifest.json", "val", fps=60),
                _source("rich_contact", "rich", "rich/manifest.json", "train", joint_set="SMPL-X"),
                _source("amass_prior", "amass", "amass/manifest.json", "train", notes=["motion prior"]),
                _source("emdb_eval", "emdb_eval", "emdb/manifest.json", "eval"),
            ],
        },
    )

    completed = _run_validator(manifest)

    payload = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert payload["valid"] is True
    assert payload["dataset_ready"] is True
    assert payload["total_sources"] == 6
    assert payload["source_type_counts"] == {
        "amass": 1,
        "athletepose3d": 1,
        "bedlam2": 1,
        "caltennis": 1,
        "emdb_eval": 1,
        "rich": 1,
    }
    assert payload["split_counts"] == {"eval": 1, "test": 0, "train": 4, "val": 1}
    assert payload["coverage_summary"]["fine_tune_ladder"] == [
        "bedlam2",
        "athletepose3d",
        "caltennis",
        "rich",
        "amass",
    ]
    assert payload["coverage_summary"]["gaps"] == []


def test_validate_pose_dataset_does_not_mark_empty_source_manifests_ready(tmp_path):
    for directory in ("bedlam2", "athletepose3d", "caltennis", "rich", "amass", "emdb"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "manifest.json").write_text("{}", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_id": "placeholder_pose_sources",
            "sources": [
                _source("bedlam2_seed", "bedlam2", "bedlam2/manifest.json", "train"),
                _source("athletepose3d_train", "athletepose3d", "athletepose3d/manifest.json", "train"),
                _source("caltennis_val", "caltennis", "caltennis/manifest.json", "val"),
                _source("rich_contact", "rich", "rich/manifest.json", "train"),
                _source("amass_prior", "amass", "amass/manifest.json", "train"),
                _source("emdb_eval", "emdb_eval", "emdb/manifest.json", "eval"),
            ],
        },
    )

    completed = _run_validator(manifest)
    payload = json.loads(completed.stdout)

    assert completed.returncode == 0
    assert payload["valid"] is True
    assert payload["coverage_summary"]["gaps"] == []
    assert payload["dataset_ready"] is False
    assert payload["content_gaps"] == ["sources contain placeholder JSON only: 6"]


def test_validate_pose_dataset_reports_fine_tune_ladder_gaps_without_failing(tmp_path):
    (tmp_path / "bedlam2").mkdir()
    (tmp_path / "bedlam2" / "manifest.json").write_text("{}", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_id": "incomplete_pose_sources",
            "sources": [_source("bedlam2_seed", "bedlam2", "bedlam2/manifest.json", "train")],
        },
    )

    completed = _run_validator(manifest)

    payload = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert payload["valid"] is True
    assert payload["dataset_ready"] is False
    assert payload["coverage_summary"]["missing_fine_tune_sources"] == [
        "athletepose3d",
        "caltennis",
        "rich",
        "amass",
    ]
    assert payload["coverage_summary"]["gaps"] == [
        "missing fine-tune ladder source: athletepose3d",
        "missing fine-tune ladder source: caltennis",
        "missing fine-tune ladder source: rich",
        "missing fine-tune ladder source: amass",
        "no emdb_eval entries registered for eval coverage",
    ]


def test_validate_pose_dataset_rejects_unsafe_paths(tmp_path):
    outside = tmp_path.parent / "outside_pose.json"
    outside.write_text("{}", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_id": "unsafe",
            "sources": [_source("escape", "bedlam2", "../outside_pose.json", "train")],
        },
    )

    completed = _run_validator(manifest)

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "sources/0/path must be relative and stay within the manifest directory" in completed.stderr


def test_validate_pose_dataset_rejects_duplicate_source_ids(tmp_path):
    (tmp_path / "first.json").write_text("{}", encoding="utf-8")
    (tmp_path / "second.json").write_text("{}", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_id": "duplicates",
            "sources": [
                _source("dup", "bedlam2", "first.json", "train"),
                _source("dup", "athletepose3d", "second.json", "train"),
            ],
        },
    )

    completed = _run_validator(manifest)

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "sources/1/id duplicate source id: dup" in completed.stderr


def test_pose_dataset_schema_file_is_valid_json():
    schema_path = Path("docs/racketsport/pose_dataset_schema.json")

    completed = subprocess.run(
        [sys.executable, "-m", "json.tool", str(schema_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
