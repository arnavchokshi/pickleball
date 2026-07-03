from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from threed.racketsport.eval_guard import EvalClipLeakError
from threed.racketsport.person_reid_dataset import (
    PersonReIDClipSpec,
    PersonReIDDatasetConfig,
    clip_specs_from_import_manifest,
    export_person_reid_crop_dataset,
)


def test_export_person_reid_crop_dataset_builds_train_query_gallery(tmp_path: Path) -> None:
    train_video, train_gt = _write_clip(tmp_path, "train_clip")
    val_video, val_gt = _write_clip(tmp_path, "val_clip")

    out_dir = tmp_path / "reid_dataset"
    manifest = export_person_reid_crop_dataset(
        clips=[
            PersonReIDClipSpec("train_clip", train_video, train_gt),
            PersonReIDClipSpec("val_clip", val_video, val_gt),
        ],
        out_dir=out_dir,
        config=PersonReIDDatasetConfig(val_clips=("val_clip",), crop_padding_px=0, query_every=2),
    )

    assert manifest["artifact_type"] == "racketsport_person_reid_crop_dataset"
    assert manifest["uses_cvat_labels"] is True
    assert manifest["source_only"] is False
    assert manifest["promote_trk"] is False
    assert manifest["identity_count"] == 4
    assert manifest["split_counts"] == {"train": 8, "query": 4, "gallery": 4}
    assert manifest["clip_counts"]["train_clip"] == {"train": 8, "query": 0, "gallery": 0}
    assert manifest["clip_counts"]["val_clip"] == {"train": 0, "query": 4, "gallery": 4}

    query_rows = [row for row in manifest["rows"] if row["split"] == "query"]
    gallery_rows = [row for row in manifest["rows"] if row["split"] == "gallery"]
    assert {row["camid"] for row in query_rows} == {0}
    assert {row["camid"] for row in gallery_rows} == {1}
    assert {row["identity_key"] for row in query_rows} == {row["identity_key"] for row in gallery_rows}

    first = manifest["rows"][0]
    crop_path = out_dir / first["relative_image_path"]
    assert crop_path.is_file()
    crop = cv2.imread(str(crop_path))
    assert crop is not None
    assert crop.shape[:2] == (16, 10)
    assert json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))["crop_count"] == 16


def test_export_person_reid_crop_dataset_guards_strict_holdout_clip_before_writing_any_crops(
    tmp_path: Path,
) -> None:
    """Regression test for review finding F3 (2026-07-02, HIGH).

    The crop-dataset builder must refuse a protected eval clip that would land in the
    train split, and must do so *before* creating any output directory or writing any
    crop file -- a downstream trainer's guard on the already-materialized manifest is
    not enough, because another script or manual job can consume the crops before that
    trainer ever runs.
    """

    protected_video, protected_gt = _write_clip(tmp_path, "outdoor_webcam_iynbd_1500_long_high_baseline")
    val_video, val_gt = _write_clip(tmp_path, "val_clip")

    out_dir = tmp_path / "reid_dataset"
    with pytest.raises(EvalClipLeakError, match="outdoor_webcam_iynbd_1500_long_high_baseline"):
        export_person_reid_crop_dataset(
            clips=[
                PersonReIDClipSpec("outdoor_webcam_iynbd_1500_long_high_baseline", protected_video, protected_gt),
                PersonReIDClipSpec("val_clip", val_video, val_gt),
            ],
            out_dir=out_dir,
            config=PersonReIDDatasetConfig(val_clips=("val_clip",), crop_padding_px=0, query_every=2),
        )

    # Nothing was materialized -- not images/train, not even the output directory --
    # because the guard ran before any file was written, not just before this
    # particular clip's crops.
    assert not out_dir.exists()


def test_export_person_reid_crop_dataset_refuses_internal_val_only_clip_as_train_split(tmp_path: Path) -> None:
    """Burlington/Wolverine may be query/gallery-only, never actual train-split data."""

    protected_video, protected_gt = _write_clip(tmp_path, "burlington_gold_0300_low_steep_corner")
    val_video, val_gt = _write_clip(tmp_path, "val_clip")

    out_dir = tmp_path / "reid_dataset"
    with pytest.raises(EvalClipLeakError, match="burlington_gold_0300_low_steep_corner"):
        export_person_reid_crop_dataset(
            clips=[
                PersonReIDClipSpec("burlington_gold_0300_low_steep_corner", protected_video, protected_gt),
                PersonReIDClipSpec("val_clip", val_video, val_gt),
            ],
            out_dir=out_dir,
            config=PersonReIDDatasetConfig(val_clips=("val_clip",), crop_padding_px=0, query_every=2),
        )
    assert not out_dir.exists()


def test_export_person_reid_crop_dataset_allows_internal_val_only_clip_as_query_gallery(tmp_path: Path) -> None:
    """Burlington/Wolverine ARE allowed as query/gallery-only (val_clips) data, and the
    guard's internal-val allowance is recorded in the manifest for audit."""

    train_video, train_gt = _write_clip(tmp_path, "train_clip")
    protected_video, protected_gt = _write_clip(tmp_path, "burlington_gold_0300_low_steep_corner")

    out_dir = tmp_path / "reid_dataset"
    manifest = export_person_reid_crop_dataset(
        clips=[
            PersonReIDClipSpec("train_clip", train_video, train_gt),
            PersonReIDClipSpec("burlington_gold_0300_low_steep_corner", protected_video, protected_gt),
        ],
        out_dir=out_dir,
        config=PersonReIDDatasetConfig(
            val_clips=("burlington_gold_0300_low_steep_corner",), crop_padding_px=0, query_every=2
        ),
    )

    assert manifest["clip_counts"]["burlington_gold_0300_low_steep_corner"]["train"] == 0
    assert manifest["eval_guard"]["status"] == "internal_val_used"
    assert manifest["eval_guard"]["internal_val_uses"][0]["clip_id"] == "burlington_gold_0300_low_steep_corner"


def test_clip_specs_from_import_manifest_reads_cvat_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "clip_id": "clip_a",
                        "source_video": "video.mp4",
                        "person_ground_truth": "person_ground_truth.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = clip_specs_from_import_manifest(manifest_path)

    assert specs == [PersonReIDClipSpec("clip_a", Path("video.mp4"), Path("person_ground_truth.json"))]


def test_person_reid_dataset_config_requires_val_clip() -> None:
    with pytest.raises(ValueError, match="val clip"):
        PersonReIDDatasetConfig()


def test_person_reid_crop_and_train_cli_help() -> None:
    for script, expected in (
        ("scripts/racketsport/build_person_reid_crop_dataset.py", "Build a labeled person ReID crop dataset"),
        ("scripts/racketsport/train_person_osnet_reid.py", "Fine-tune OSNet"),
    ):
        completed = subprocess.run(
            [sys.executable, script, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0
        assert expected in completed.stdout


def _write_clip(tmp_path: Path, clip_id: str) -> tuple[Path, Path]:
    video_path = tmp_path / f"{clip_id}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (48, 32))
    assert writer.isOpened()
    try:
        for frame_index in range(4):
            frame = np.zeros((32, 48, 3), dtype=np.uint8)
            frame[:, :24] = (0, 0, 80 + frame_index)
            frame[:, 24:] = (80 + frame_index, 0, 0)
            writer.write(frame)
    finally:
        writer.release()

    gt_path = tmp_path / f"{clip_id}_person_ground_truth.json"
    frames = []
    for frame_index in range(4):
        frames.append(
            {
                "frame_index": frame_index,
                "source_frame_id": frame_index + 1,
                "labels": [
                    {
                        "track_id": 1,
                        "bbox_xywh": [2.0, 3.0, 10.0, 16.0],
                        "ignored": False,
                        "visibility": 1.0,
                        "confidence": 1.0,
                        "class_name": "player",
                    },
                    {
                        "track_id": 2,
                        "bbox_xywh": [28.0, 4.0, 10.0, 16.0],
                        "ignored": False,
                        "visibility": 1.0,
                        "confidence": 1.0,
                        "class_name": "player",
                    },
                ],
            }
        )
    gt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_person_ground_truth",
                "clip_id": clip_id,
                "source_format": "cvat_video_1_1",
                "source_path": f"{clip_id}.zip",
                "fps": 10.0,
                "frames": frames,
                "summary": {
                    "frame_count": 4,
                    "valid_label_count": 8,
                    "ignored_label_count": 0,
                    "track_ids": [1, 2],
                    "max_valid_players_per_frame": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    return video_path, gt_path
