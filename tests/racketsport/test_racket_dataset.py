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


def _run_validator(manifest: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/racketsport/validate_racket_dataset.py", str(manifest)],
        check=False,
        capture_output=True,
        text=True,
    )


def _annotations(*, pose: bool = False, masks: str | None = None) -> dict:
    annotations: dict[str, object] = {
        "keypoints": {
            "names": ["top", "bottom", "handle"],
            "dimensions": 2,
        },
        "corners": {
            "names": ["top_left", "top_right", "bottom_right", "bottom_left"],
            "dimensions": 2,
        },
    }
    if pose:
        annotations["pose_labels"] = {
            "format": "rvec_tvec",
            "coordinate_frame": "camera",
        }
    if masks is not None:
        annotations["segmentation_masks"] = {
            "format": "png",
            "path": masks,
        }
    return annotations


def test_validate_racket_dataset_accepts_valid_mixed_sources(tmp_path):
    (tmp_path / "racketvision").mkdir()
    (tmp_path / "synthetic").mkdir()
    (tmp_path / "aruco").mkdir()
    (tmp_path / "racketvision" / "rv_clip.json").write_text("{}", encoding="utf-8")
    (tmp_path / "synthetic" / "blend_pose.json").write_text("{}", encoding="utf-8")
    (tmp_path / "synthetic" / "blend_mask.png").write_text("mask", encoding="utf-8")
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
                    split="train",
                    annotations=_annotations(),
                    camera={"camera_id": "cam_a", "view": "side"},
                    fps=120,
                    frame_count=400,
                ),
                _entry(
                    "blend_pose",
                    "synthetic_blenderproc",
                    "synthetic/blend_pose.json",
                    split="val",
                    annotations=_annotations(pose=True, masks="synthetic/blend_mask.png"),
                    fps=240,
                    frame_count=50000,
                ),
                _entry(
                    "aruco_trial",
                    "aruco_gt",
                    "aruco/trial_001.json",
                    split="eval",
                    annotations=_annotations(pose=True),
                    marker={"dictionary": "DICT_4X4_50", "marker_size_m": 0.04},
                    fps=120,
                ),
                _entry(
                    "heldout_rv",
                    "racketvision",
                    "racketvision/rv_clip.json",
                    split="test",
                    annotations=_annotations(),
                ),
            ],
        },
    )

    completed = _run_validator(manifest)

    payload = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert payload["valid"] is True
    assert payload["dataset_ready"] is True
    assert payload["total_sources"] == 4
    assert payload["source_type_counts"] == {
        "aruco_gt": 1,
        "racketvision": 2,
        "synthetic_blenderproc": 1,
    }
    assert payload["split_counts"] == {"eval": 1, "test": 1, "train": 1, "val": 1}
    assert payload["annotation_counts"] == {
        "corners": 4,
        "keypoints": 4,
        "pose_labels": 2,
        "segmentation_masks": 1,
    }
    assert payload["coverage_summary"]["has_aruco_gt_eval"] is True
    assert payload["coverage_summary"]["gaps"] == []


def test_validate_racket_dataset_reports_missing_aruco_eval_without_failing(tmp_path):
    (tmp_path / "racketvision").mkdir()
    (tmp_path / "racketvision" / "rv_clip.json").write_text("{}", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_id": "no_aruco_yet",
            "sources": [
                _entry(
                    "rv_clip",
                    "racketvision",
                    "racketvision/rv_clip.json",
                    split="train",
                    annotations=_annotations(),
                )
            ],
        },
    )

    completed = _run_validator(manifest)

    payload = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert payload["valid"] is True
    assert payload["dataset_ready"] is False
    assert payload["source_type_counts"] == {
        "aruco_gt": 0,
        "racketvision": 1,
        "synthetic_blenderproc": 0,
    }
    assert payload["coverage_summary"]["has_aruco_gt_eval"] is False
    assert "no aruco_gt eval entries registered for racket face-angle GT coverage" in payload[
        "coverage_summary"
    ]["gaps"]


def test_validate_racket_dataset_reports_missing_split_and_source_coverage_without_failing(tmp_path):
    (tmp_path / "synthetic").mkdir()
    (tmp_path / "synthetic" / "blend_pose.json").write_text("{}", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_id": "coverage_gaps",
            "sources": [
                _entry(
                    "blend_pose",
                    "synthetic_blenderproc",
                    "synthetic/blend_pose.json",
                    split="train",
                    annotations=_annotations(pose=True),
                )
            ],
        },
    )

    completed = _run_validator(manifest)

    payload = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert payload["valid"] is True
    assert payload["dataset_ready"] is False
    assert payload["split_counts"] == {"eval": 0, "test": 0, "train": 1, "val": 0}
    assert payload["coverage_summary"]["missing_source_types"] == ["racketvision", "aruco_gt"]
    assert payload["coverage_summary"]["missing_splits"] == ["eval", "test", "val"]
    assert payload["coverage_summary"]["gaps"] == [
        "missing source type: racketvision",
        "missing source type: aruco_gt",
        "missing split: eval",
        "missing split: test",
        "missing split: val",
        "no aruco_gt eval entries registered for racket face-angle GT coverage",
    ]


def test_validate_racket_dataset_rejects_unsafe_paths(tmp_path):
    outside = tmp_path.parent / "outside_racket.json"
    outside.write_text("{}", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_id": "unsafe",
            "sources": [
                _entry(
                    "escape",
                    "racketvision",
                    "../outside_racket.json",
                    split="train",
                    annotations=_annotations(),
                )
            ],
        },
    )

    completed = _run_validator(manifest)

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "sources/0/path must be relative and stay within the manifest directory" in completed.stderr


def test_validate_racket_dataset_rejects_duplicate_source_ids(tmp_path):
    (tmp_path / "first.json").write_text("{}", encoding="utf-8")
    (tmp_path / "second.json").write_text("{}", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_id": "duplicates",
            "sources": [
                _entry("dup", "racketvision", "first.json", split="train", annotations=_annotations()),
                _entry("dup", "racketvision", "second.json", split="val", annotations=_annotations()),
            ],
        },
    )

    completed = _run_validator(manifest)

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "sources/1/id duplicate source id: dup" in completed.stderr


def test_validate_racket_dataset_rejects_invalid_keypoint_schema(tmp_path):
    (tmp_path / "racketvision").mkdir()
    (tmp_path / "racketvision" / "rv_clip.json").write_text("{}", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "dataset_id": "bad_keypoints",
            "sources": [
                _entry(
                    "rv_clip",
                    "racketvision",
                    "racketvision/rv_clip.json",
                    split="train",
                    annotations={
                        "keypoints": {"names": ["top", "side"], "dimensions": 2},
                        "corners": {
                            "names": ["top_left", "top_right", "bottom_right", "bottom_left"],
                            "dimensions": 2,
                        },
                    },
                )
            ],
        },
    )

    completed = _run_validator(manifest)

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "sources/0/annotations/keypoints/names: must include top, bottom, handle" in completed.stderr


def test_racket_dataset_schema_file_is_valid_json():
    schema_path = Path("docs/racketsport/racket_dataset_schema.json")

    completed = subprocess.run(
        [sys.executable, "-m", "json.tool", str(schema_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
