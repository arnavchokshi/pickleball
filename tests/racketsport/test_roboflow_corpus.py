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
    ProtectedEvalHashCollisionError,
    RoboflowBallPretrainDataset,
    aggregate_roboflow_corpus,
    file_sha256,
    image_dhash,
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


def test_ball_pretrain_dataset_resolves_index_paths_and_uses_unknown_visibility_weight(tmp_path: Path) -> None:
    image = tmp_path / "source" / "train" / "frame_000001.jpg"
    _write_image(image, color=(20, 80, 140))
    corpus_index = tmp_path / "aggregated" / "corpus_index.json"
    _write_ball_corpus_index(
        corpus_index,
        [
            _ball_sample(
                sample_id="core:train:1",
                image_path=image,
                split="train",
                bucket="core_pickleball",
                xy=[11.0, 13.0],
                temporal={"kind": "isolated_still", "sequence_id": None, "sequence_order": None},
            )
        ],
    )

    dataset = RoboflowBallPretrainDataset(
        corpus_index,
        split_role="train",
        image_size=(32, 32),
        frames_in=3,
        protected_eval_hashes={"synthetic": [0xFFFFFFFFFFFFFFFF]},
        expected_protected_eval_hash_count=1,
        skip_list_path=tmp_path / "skip_list.json",
    )
    item = dataset[0]

    assert dataset.summary["selected_sample_count"] == 1
    assert dataset.summary["visibility_policy"]["unknown_visibility_wbce_weight"] == 1
    assert item["sample_id"] == "core:train:1"
    assert item["image_path"] == str(image)
    assert tuple(item["input"].shape) == (9, 32, 32)
    assert tuple(item["target"].shape) == (1, 32, 32)
    assert item["visibility_level"] is None
    assert float(item["wbce_weight"]) == pytest.approx(1.0)
    assert item["temporal_sample_kind"] == "still_aux_repeated"
    assert item["window_sample_ids"] == ["core:train:1", "core:train:1", "core:train:1"]
    assert json.loads((tmp_path / "skip_list.json").read_text(encoding="utf-8"))["skip_count"] == 0


def test_ball_pretrain_dataset_rewrites_absolute_image_roots_for_vm_paths(tmp_path: Path) -> None:
    vm_root = tmp_path / "vm_checkout"
    actual_image = vm_root / "data" / "roboflow_universe_20260706" / "source" / "train" / "frame_000001.jpg"
    _write_image(actual_image, color=(30, 90, 150))
    old_image_path = Path("/Users/arnavchokshi/Desktop/pickleball") / actual_image.relative_to(vm_root)
    sample = _ball_sample(
        sample_id="core:train:rewritten",
        image_path=actual_image,
        split="train",
        bucket="core_pickleball",
        xy=[11.0, 13.0],
        temporal={"kind": "isolated_still", "sequence_id": None, "sequence_order": None},
    )
    sample["image_path"] = str(old_image_path)
    corpus_index = tmp_path / "aggregated" / "corpus_index.json"
    _write_ball_corpus_index(corpus_index, [sample])

    dataset = RoboflowBallPretrainDataset(
        corpus_index,
        split_role="train",
        image_size=(32, 32),
        frames_in=3,
        protected_eval_hashes={"synthetic": [0xFFFFFFFFFFFFFFFF]},
        expected_protected_eval_hash_count=1,
        skip_list_path=tmp_path / "skip_list.json",
        image_path_rewrites={"/Users/arnavchokshi/Desktop/pickleball": str(vm_root)},
    )
    item = dataset[0]

    assert item["image_path"] == str(actual_image)
    assert dataset.summary["image_path_rewrites"] == {
        "/Users/arnavchokshi/Desktop/pickleball": str(vm_root)
    }


def test_ball_pretrain_dataset_tensor_matches_wasb_official_preprocessing(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from threed.racketsport import wasb_adapter

    image = tmp_path / "source" / "train" / "frame_000001.png"
    _write_gradient_image(image, width=64, height=48)
    corpus_index = tmp_path / "aggregated" / "corpus_index.json"
    _write_ball_corpus_index(
        corpus_index,
        [
            _ball_sample(
                sample_id="core:train:official",
                image_path=image,
                split="train",
                bucket="core_pickleball",
                xy=[16.0, 12.0],
                temporal={"kind": "isolated_still", "sequence_id": None, "sequence_order": None},
            )
        ],
    )

    dataset = RoboflowBallPretrainDataset(
        corpus_index,
        split_role="train",
        frames_in=3,
        protected_eval_hashes={"synthetic": [0xFFFFFFFFFFFFFFFF]},
        expected_protected_eval_hash_count=1,
        skip_list_path=tmp_path / "skip_list.json",
    )
    item = dataset[0]
    frame = np.asarray(Image.open(image).convert("RGB"))
    trans_input = wasb_adapter._wasb_official_input_affine(64, 48, cv2=cv2, np=np)
    expected = wasb_adapter._preprocess_wasb_window(
        [frame, frame, frame],
        trans_input,
        cv2=cv2,
        np=np,
        torch=torch,
        input_preprocessing="official",
    )

    assert torch.allclose(item["input"], expected, atol=1e-6)


def test_ball_pretrain_dataset_heatmap_argmax_uses_wasb_official_affine(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from threed.racketsport import wasb_adapter

    image = tmp_path / "source" / "train" / "frame_000001.png"
    _write_gradient_image(image, width=64, height=48)
    corpus_index = tmp_path / "aggregated" / "corpus_index.json"
    _write_ball_corpus_index(
        corpus_index,
        [
            _ball_sample(
                sample_id="core:train:label-geometry",
                image_path=image,
                split="train",
                bucket="core_pickleball",
                xy=[16.0, 12.0],
                temporal={"kind": "isolated_still", "sequence_id": None, "sequence_order": None},
            )
        ],
    )

    dataset = RoboflowBallPretrainDataset(
        corpus_index,
        split_role="train",
        frames_in=3,
        protected_eval_hashes={"synthetic": [0xFFFFFFFFFFFFFFFF]},
        expected_protected_eval_hash_count=1,
        skip_list_path=tmp_path / "skip_list.json",
    )
    item = dataset[0]
    peak_index = int(item["target"][0].flatten().argmax().item())
    peak_xy = torch.tensor([peak_index % 512, peak_index // 512], dtype=torch.float32)
    trans_input = wasb_adapter._wasb_official_input_affine(64, 48, cv2=cv2, np=np)
    expected_xy = torch.tensor(
        wasb_adapter._wasb_affine_transform_xy([16.0, 12.0], trans_input, np=np),
        dtype=torch.float32,
    )

    assert torch.allclose(item["target_xy_px"], expected_xy, atol=1e-6)
    assert torch.allclose(peak_xy, expected_xy.round(), atol=1.0)


def test_ball_pretrain_dataset_uses_corpus_card_leakage_summary_by_default(tmp_path: Path) -> None:
    image = tmp_path / "source" / "train" / "frame_000001.jpg"
    _write_image(image, color=(40, 90, 130))
    corpus_index = tmp_path / "aggregated" / "corpus_index.json"
    sample = _ball_sample(
        sample_id="core:train:card",
        image_path=image,
        split="train",
        bucket="core_pickleball",
        xy=[11.0, 13.0],
        temporal={"kind": "isolated_still", "sequence_id": None, "sequence_order": None},
    )
    sample["leakage"] = {"eval_collision": False, "collisions": []}
    _write_ball_corpus_index(corpus_index, [sample])
    _write_corpus_card(corpus_index.with_name("corpus_card.json"))

    dataset = RoboflowBallPretrainDataset(
        corpus_index,
        split_role="train",
        image_size=(32, 32),
        frames_in=3,
        eval_root=tmp_path / "missing_eval_root",
        skip_list_path=tmp_path / "skip_list.json",
    )

    assert dataset.summary["protected_eval_hash_check"]["hash_count"] == 35
    assert dataset.summary["protected_eval_hash_check"]["hash_source"] == str(
        corpus_index.with_name("corpus_card.json")
    )


def test_ball_pretrain_dataset_fails_loud_on_eval_hash_collision(tmp_path: Path) -> None:
    image = tmp_path / "source" / "train" / "frame_000001.jpg"
    _write_image(image, color=(200, 20, 20))
    collision_hash = image_dhash(image)
    corpus_index = tmp_path / "aggregated" / "corpus_index.json"
    _write_ball_corpus_index(
        corpus_index,
        [
            _ball_sample(
                sample_id="core:train:1",
                image_path=image,
                split="train",
                bucket="core_pickleball",
                xy=[16.0, 15.0],
                temporal={"kind": "isolated_still", "sequence_id": None, "sequence_order": None},
            )
        ],
    )

    with pytest.raises(ProtectedEvalHashCollisionError, match="protected eval hash collision"):
        RoboflowBallPretrainDataset(
            corpus_index,
            split_role="train",
            image_size=(32, 32),
            frames_in=3,
            protected_eval_hashes={"eval_clip": [collision_hash]},
            expected_protected_eval_hash_count=1,
            skip_list_path=tmp_path / "skip_list.json",
            collision_hamming_threshold=0,
        )


def test_ball_pretrain_dataset_builds_sequence_windows_and_still_aux_samples(tmp_path: Path) -> None:
    paths = []
    for frame in range(3):
        path = tmp_path / "source" / "train" / f"seq_{frame:06d}.jpg"
        _write_image(path, color=(20 + frame * 20, 90, 120))
        paths.append(path)
    still = tmp_path / "source" / "train" / "still.jpg"
    _write_image(still, color=(100, 40, 140))
    corpus_index = tmp_path / "aggregated" / "corpus_index.json"
    _write_ball_corpus_index(
        corpus_index,
        [
            _ball_sample(
                sample_id=f"seq:{frame}",
                image_path=path,
                split="train",
                bucket="core_pickleball",
                xy=[10.0 + frame, 12.0 + frame],
                temporal={
                    "kind": "temporal_sequence",
                    "sequence_id": "seq-a",
                    "sequence_order": frame,
                    "frame_number": frame,
                    "sequence_group_size": 3,
                },
            )
            for frame, path in enumerate(paths)
        ]
        + [
            _ball_sample(
                sample_id="still:0",
                image_path=still,
                split="train",
                bucket="core_pickleball",
                xy=[20.0, 21.0],
                temporal={"kind": "isolated_still", "sequence_id": None, "sequence_order": None},
            )
        ],
    )

    dataset = RoboflowBallPretrainDataset(
        corpus_index,
        split_role="train",
        image_size=(32, 32),
        frames_in=3,
        protected_eval_hashes={"synthetic": [0xFFFFFFFFFFFFFFFF]},
        expected_protected_eval_hash_count=1,
        skip_list_path=tmp_path / "skip_list.json",
    )
    by_id = {dataset[index]["sample_id"]: dataset[index] for index in range(len(dataset))}

    assert by_id["seq:1"]["temporal_sample_kind"] == "sequence_window"
    assert by_id["seq:1"]["window_sample_ids"] == ["seq:0", "seq:1", "seq:2"]
    assert by_id["still:0"]["temporal_sample_kind"] == "still_aux_repeated"
    assert by_id["still:0"]["window_sample_ids"] == ["still:0", "still:0", "still:0"]
    assert dataset.summary["temporal_sample_counts"] == {
        "sequence_window": 3,
        "still_aux_repeated": 1,
    }


def test_ball_pretrain_dataset_applies_core_to_aux_mixing_ratio(tmp_path: Path) -> None:
    samples = []
    for index, bucket in enumerate(["core_pickleball", "core_pickleball", "adjacent_sport_aux"], start=1):
        image = tmp_path / "source" / "train" / f"{bucket}_{index}.jpg"
        _write_image(image, color=(index * 40, index * 30, index * 20))
        samples.append(
            _ball_sample(
                sample_id=f"{bucket}:{index}",
                image_path=image,
                split="train",
                bucket=bucket,
                xy=[12.0, 14.0],
                temporal={"kind": "isolated_still", "sequence_id": None, "sequence_order": None},
            )
        )
    corpus_index = tmp_path / "aggregated" / "corpus_index.json"
    _write_ball_corpus_index(corpus_index, samples)

    dataset = RoboflowBallPretrainDataset(
        corpus_index,
        split_role="train",
        image_size=(32, 32),
        frames_in=3,
        core_to_aux_ratio=2,
        protected_eval_hashes={"synthetic": [0xFFFFFFFFFFFFFFFF]},
        expected_protected_eval_hash_count=1,
        skip_list_path=tmp_path / "skip_list.json",
    )

    assert [dataset[index]["bucket"] for index in range(len(dataset))] == [
        "core_pickleball",
        "core_pickleball",
        "adjacent_sport_aux",
    ]
    assert dataset.summary["mixing"]["core_to_aux_ratio"] == 2


def _write_image(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color=color).save(path)


def _write_gradient_image(path: Path, *, width: int, height: int) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:height, 0:width]
    array = np.stack(
        [
            (xx * 5 + yy * 3) % 256,
            (xx * 7 + 11) % 256,
            (yy * 13 + 17) % 256,
        ],
        axis=2,
    ).astype(np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


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


def _write_ball_corpus_index(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_roboflow_public_pretrain_corpus_index",
                "index_policy": "index_based_original_image_paths_no_image_copies",
                "hash": {"collision_hamming_threshold": 3},
                "sample_count": len(samples),
                "samples": samples,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_corpus_card(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_roboflow_public_pretrain_corpus_card",
                "leakage_check": {
                    "eval_collision_count": 0,
                    "eval_hash_counts": {
                        "burlington": 5,
                        "indoor": 15,
                        "outdoor": 10,
                        "wolverine": 5,
                    },
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _ball_sample(
    *,
    sample_id: str,
    image_path: Path,
    split: str,
    bucket: str,
    xy: list[float],
    temporal: dict,
) -> dict:
    return {
        "sample_id": sample_id,
        "source_slug": sample_id.split(":")[0],
        "bucket": bucket,
        "split": split,
        "image_path": str(image_path),
        "width": 64,
        "height": 48,
        "labels": {
            "ball": [
                {
                    "t": 0.0,
                    "xy": xy,
                    "conf": 1.0,
                    "visible": True,
                    "source_bbox_xywh": [xy[0] - 2.0, xy[1] - 2.0, 4.0, 4.0],
                    "original_category": "ball",
                    "annotation_id": 1,
                }
            ],
            "court": [],
            "other": [],
            "paddle": [],
            "person": [],
        },
        "label_kinds": ["ball"],
        "temporal": temporal,
        "hashes": {
            "dhash": f"{image_dhash(image_path):016x}",
            "sha256": file_sha256(image_path),
        },
    }
