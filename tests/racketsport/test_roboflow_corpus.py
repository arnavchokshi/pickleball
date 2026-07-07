from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("torch")

from threed.racketsport.roboflow_corpus import (
    ADJACENT_SPORT_SLUGS,
    aggregate_roboflow_corpus,
    load_smoke_samples,
    normalize_dataset_entry,
)


CLI_PATH = "scripts/racketsport/aggregate_roboflow_corpus.py"


def test_normalize_dataset_entry_converts_coco_boxes_to_index_samples(tmp_path: Path) -> None:
    dataset = tmp_path / "data" / "workspace__pickleball-demo__v1"
    train = dataset / "train"
    train.mkdir(parents=True)
    _write_image(train / "video1_000001_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg", color=(120, 30, 10))
    _write_image(train / "video1_000002_jpg.rf.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.jpg", color=(10, 120, 30))
    _write_coco(
        train / "_annotations.coco.json",
        images=[
            {
                "id": 1,
                "file_name": "video1_000001_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg",
                "width": 64,
                "height": 48,
                "extra": {"name": "video1_000001.jpg"},
            },
            {
                "id": 2,
                "file_name": "video1_000002_jpg.rf.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.jpg",
                "width": 64,
                "height": 48,
                "extra": {"name": "video1_000002.jpg"},
            },
        ],
        annotations=[
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10.0, 20.0, 8.0, 6.0]},
            {"id": 11, "image_id": 1, "category_id": 2, "bbox": [3.0, 4.0, 20.0, 35.0]},
            {"id": 12, "image_id": 2, "category_id": 3, "bbox": [0.0, 0.0, 64.0, 48.0]},
            {"id": 13, "image_id": 2, "category_id": 4, "bbox": [30.0, 30.0, 12.0, 10.0]},
        ],
    )
    entry = {
        "slug": "workspace/pickleball-demo",
        "status": "downloaded",
        "local_path": str(dataset),
        "content_category_guess": "1_ball",
        "temporal_hint": "likely_video_sequential",
        "temporal_hint_detail": {"hint": "likely_video_sequential"},
        "classes": ["ball", "Player", "Court", "Paddle"],
    }

    index = normalize_dataset_entry(entry, repo_root=tmp_path)

    assert index["source"]["slug"] == "workspace/pickleball-demo"
    assert index["bucket"] == "core_pickleball"
    assert index["statistics"]["sample_count"] == 2
    assert index["statistics"]["label_counts_by_taxonomy"] == {
        "ball": 1,
        "court": 1,
        "paddle": 1,
        "person": 1,
    }
    first = index["samples"][0]
    assert first["image_path"].endswith("video1_000001_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg")
    assert Path(first["image_path"]).is_file()
    assert first["temporal"]["kind"] == "temporal_sequence"
    assert first["temporal"]["sequence_id"].endswith(":train:video1")
    ball = first["labels"]["ball"][0]
    assert ball == {
        "t": 0.0,
        "xy": [14.0, 23.0],
        "conf": 1.0,
        "visible": True,
        "source_bbox_xywh": [10.0, 20.0, 8.0, 6.0],
        "original_category": "ball",
        "annotation_id": 10,
    }
    assert "visibility_level" not in ball


def test_aggregate_roboflow_corpus_buckets_fork_dead_and_adjacent_sources(tmp_path: Path) -> None:
    data_root = tmp_path / "data" / "roboflow_universe_20260706"
    output_dir = data_root / "aggregated"
    lane_dir = tmp_path / "runs" / "lanes" / "p10_roboflow_aggregate_20260706"
    eval_root = tmp_path / "eval_clips" / "ball"
    eval_root.mkdir(parents=True)

    core_large = _dataset_with_images(data_root, "core-large__pickleball__v1", ["000001", "000002"])
    core_fork = _dataset_with_images(
        data_root,
        "core-fork__pickleball__v1",
        ["000001"],
        copied_from=next((core_large / "train").glob("000001_jpg.rf.*.jpg")),
    )
    adjacent = _dataset_with_images(
        data_root,
        "pickleball-kjawm__tennis-ball-detection-sxi3e-inzuo__v1",
        ["000010"],
    )
    manifest = {
        "schema_version": 1,
        "entries": [
            _manifest_entry("core-large/pickleball", core_large, 2),
            _manifest_entry("core-fork/pickleball", core_fork, 1),
            _manifest_entry("pickleball-kjawm/tennis-ball-detection-sxi3e-inzuo", adjacent, 1),
            {
                "slug": "missing/no-version",
                "status": "failed",
                "version": None,
                "classes": ["Ball"],
                "image_count_downloaded": 0,
                "note": "no exported version",
            },
        ],
    }
    manifest_path = data_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = aggregate_roboflow_corpus(
        manifest_path=manifest_path,
        output_dir=output_dir,
        lane_dir=lane_dir,
        eval_root=eval_root,
        repo_root=tmp_path,
    )

    card = json.loads((output_dir / "corpus_card.json").read_text(encoding="utf-8"))
    corpus_index = json.loads((output_dir / "corpus_index.json").read_text(encoding="utf-8"))

    assert result["corpus_card_path"] == str(output_dir / "corpus_card.json")
    assert card["bucket_counts_by_source"] == {
        "adjacent_sport_aux": 1,
        "core_pickleball": 1,
        "excluded_dead": 1,
        "excluded_duplicate": 1,
    }
    assert card["fork_duplicate_mappings"] == [
        {
            "duplicate_slug": "core-fork/pickleball",
            "kept_slug": "core-large/pickleball",
            "overlap_count": 1,
            "overlap_ratio_of_duplicate": 1.0,
            "reason": "exact_image_sha256_overlap",
        }
    ]
    assert card["leakage_check"]["eval_collision_count"] == 0
    assert "pickleball-kjawm/tennis-ball-detection-sxi3e-inzuo" in ADJACENT_SPORT_SLUGS
    assert all(Path(sample["image_path"]).is_file() for sample in corpus_index["samples"])
    assert not list(output_dir.rglob("*.jpg"))


def test_loader_smoke_reads_real_roboflow_samples_across_sources() -> None:
    manifest_path = Path("data/roboflow_universe_20260706/manifest.json")
    if not manifest_path.is_file():
        pytest.skip("Roboflow universe manifest is not present in this checkout")

    smoke = load_smoke_samples(manifest_path, repo_root=Path("."), limit=50, min_datasets=5)

    assert smoke["opened_samples"] >= 50
    assert smoke["dataset_count"] >= 5
    assert smoke["label_kinds_seen"] >= ["ball"]
    assert smoke["missing_paths"] == []


def test_aggregate_roboflow_corpus_cli_help_is_indexed() -> None:
    completed = subprocess.run(
        [sys.executable, CLI_PATH, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--manifest" in completed.stdout
    assert "--output-dir" in completed.stdout
    assert "index-based" in completed.stdout


def test_scaffold_index_covers_aggregate_roboflow_corpus_cli() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/racketsport/list_scaffold_tools.py",
            "--root",
            ".",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    by_path = {tool["command_path"]: tool for tool in payload["tools"]}

    assert by_path[CLI_PATH]["category"] == "dataset"
    assert by_path[CLI_PATH]["workstream"] == "DATA"
    assert by_path[CLI_PATH]["task_prefix"] == "P1-0"
    assert by_path[CLI_PATH]["direct_cli_reference_test"] == "tests/racketsport/test_roboflow_corpus.py"


def _write_image(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color=color).save(path)


def _write_coco(path: Path, *, images: list[dict], annotations: list[dict]) -> None:
    categories = [
        {"id": 1, "name": "ball", "supercategory": "none"},
        {"id": 2, "name": "Player", "supercategory": "none"},
        {"id": 3, "name": "Court", "supercategory": "none"},
        {"id": 4, "name": "Paddle", "supercategory": "none"},
    ]
    path.write_text(
        json.dumps(
            {
                "info": {},
                "licenses": [],
                "categories": categories,
                "images": images,
                "annotations": annotations,
            }
        ),
        encoding="utf-8",
    )


def _dataset_with_images(
    data_root: Path,
    dirname: str,
    frame_ids: list[str],
    *,
    copied_from: Path | None = None,
) -> Path:
    dataset = data_root / dirname
    train = dataset / "train"
    train.mkdir(parents=True, exist_ok=True)
    images: list[dict] = []
    annotations: list[dict] = []
    for index, frame_id in enumerate(frame_ids, start=1):
        filename = f"{frame_id}_jpg.rf.{frame_id * 8}.jpg"
        path = train / filename
        if copied_from is not None and index == 1:
            path.write_bytes(copied_from.read_bytes())
        else:
            _write_image(path, color=(index * 40 % 255, index * 70 % 255, index * 90 % 255))
        images.append(
            {
                "id": index,
                "file_name": filename,
                "width": 64,
                "height": 48,
                "extra": {"name": f"{frame_id}.jpg"},
            }
        )
        annotations.append({"id": index, "image_id": index, "category_id": 1, "bbox": [1.0, 2.0, 3.0, 4.0]})
    _write_coco(train / "_annotations.coco.json", images=images, annotations=annotations)
    return dataset


def _manifest_entry(slug: str, dataset: Path, count: int) -> dict:
    return {
        "slug": slug,
        "status": "downloaded",
        "version": 1,
        "classes": ["ball"],
        "content_category_guess": "1_ball",
        "image_count_downloaded": count,
        "local_path": str(dataset),
        "temporal_hint": "likely_video_sequential",
        "temporal_hint_detail": {"hint": "likely_video_sequential"},
    }
