from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from scripts.racketsport.build_real_court_corpus import CANONICAL_KEYPOINTS
from scripts.racketsport.train_court_keypoint_heatmap import load_real_court_keypoint_labels


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/racketsport/build_real_court_corpus.py"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_coco_dataset(
    root: Path,
    name: str,
    *,
    license_name: str = "CC BY 4.0",
    keypoint_names: tuple[str, ...] = CANONICAL_KEYPOINTS,
    image_count: int = 1,
    color_seed: int = 0,
) -> None:
    dataset = root / name
    split_dir = dataset / "train"
    split_dir.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    for image_index in range(image_count):
        image_path = split_dir / f"court_{image_index:02d}.png"
        Image.new("RGB", (96, 64), ((31 + color_seed + image_index) % 255, 102, 77)).save(image_path)
        keypoint_values: list[float | int] = []
        for index in range(len(keypoint_names)):
            keypoint_values.extend([float(5 + index * 3), float(8 + index * 2), 2])
        images.append({"id": image_index + 1, "file_name": image_path.name, "width": 96, "height": 64})
        annotations.append(
            {
                "id": image_index + 1,
                "image_id": image_index + 1,
                "category_id": 1,
                "bbox": [0, 0, 96, 64],
                "keypoints": keypoint_values,
                "num_keypoints": len(keypoint_names),
            }
        )
    _write_json(
        dataset / "train" / "_annotations.coco.json",
        {
            "images": images,
            "categories": [
                {
                    "id": 1,
                    "name": "court",
                    "keypoints": list(keypoint_names),
                    "skeleton": [],
                }
            ],
            "annotations": annotations,
        },
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"entries": []}
    manifest["entries"].append(
        {
            "status": "downloaded",
            "local_path": str(dataset.resolve()),
            "slug": f"test/{name}",
            "project": name,
            "classes": ["court"],
            "content_category_guess": "2_court",
            "license_as_recorded": license_name,
        }
    )
    _write_json(
        manifest_path,
        manifest,
    )


def _mapping(name: str) -> dict:
    return {
        "schema_version": 1,
        "canonical_keypoints": list(CANONICAL_KEYPOINTS),
        "datasets": {
            name: {
                "verdict": "direct-map",
                "mapping_confidence": "high",
                "mapping_rationale": "Toy direct mapping.",
                "keypoint_mapping": {point: point for point in CANONICAL_KEYPOINTS},
                "viewpoint_character": ["elevated"],
                "source_group": "toy_source",
                "human_annotated": True,
                "include_default": True,
            }
        },
    }


def test_direct_cli_builds_loader_compatible_symlink_corpus(tmp_path: Path) -> None:
    dataset_root = tmp_path / "roboflow"
    dataset_name = "toy__court__v1"
    _write_coco_dataset(dataset_root, dataset_name)
    mapping_path = tmp_path / "mapping.json"
    _write_json(mapping_path, _mapping(dataset_name))
    lane_dir = tmp_path / "lane"
    corpus_root = lane_dir / "real_court_corpus"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset-root",
            str(dataset_root),
            "--mapping-table",
            str(mapping_path),
            "--lane-dir",
            str(lane_dir),
            "--output-root",
            str(corpus_root),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary == {
        "audit_dataset_count": 1,
        "corpus_rows": 1,
        "dataset_count": 1,
        "output_root": str(corpus_root),
    }
    rows = load_real_court_keypoint_labels(corpus_root)
    assert len(rows) == 1
    assert rows[0]["label_status"] == "reviewed"
    assert set(rows[0]["keypoints"]) == set(CANONICAL_KEYPOINTS)
    assert rows[0]["image_path"] is not None
    assert Path(rows[0]["image_path"]).is_symlink()
    assert Path(rows[0]["image_path"]).is_file()

    label_payload = json.loads(next(corpus_root.glob("*/labels/court_keypoints.json")).read_text(encoding="utf-8"))
    provenance = label_payload["annotation"]["items"][0]["provenance"]
    assert provenance["dataset"] == dataset_name
    assert provenance["split"] == "train"
    assert provenance["license"] == "CC BY 4.0"
    assert provenance["original_image"].endswith("court_00.png")


def test_cli_excludes_noncommercial_dataset_from_default_corpus(tmp_path: Path) -> None:
    dataset_root = tmp_path / "roboflow"
    dataset_name = "toy__court_nc__v1"
    _write_coco_dataset(dataset_root, dataset_name, license_name="BY-NC-SA 4.0")
    mapping_path = tmp_path / "mapping.json"
    _write_json(mapping_path, _mapping(dataset_name))
    lane_dir = tmp_path / "lane"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset-root",
            str(dataset_root),
            "--mapping-table",
            str(mapping_path),
            "--lane-dir",
            str(lane_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["corpus_rows"] == 0
    stats = json.loads((lane_dir / "corpus_stats.json").read_text(encoding="utf-8"))
    assert stats["excluded_counts"] == {"noncommercial_or_unknown_license": 1}
    license_card = json.loads((lane_dir / "license_card.json").read_text(encoding="utf-8"))
    assert license_card["quarantined_noncommercial"] == [dataset_name]


def test_partial_cli_emits_exact_null_schema_external_status_and_loader_proofs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "roboflow"
    planar = tuple(name for name in CANONICAL_KEYPOINTS if not name.startswith("net_"))
    source_planar = tuple(f"source_{index}" for index in range(len(planar)))
    names = ("toy__direct__v1", "toy__partial_a__v1", "toy__partial_b__v1")
    _write_coco_dataset(dataset_root, names[0], image_count=5, color_seed=0)
    _write_coco_dataset(dataset_root, names[1], keypoint_names=source_planar, image_count=5, color_seed=40)
    _write_coco_dataset(dataset_root, names[2], keypoint_names=source_planar, image_count=5, color_seed=80)
    mapping = _mapping(names[0])
    for dataset_name in names[1:]:
        mapping["datasets"][dataset_name] = {
            "verdict": "partial-map",
            "mapping_confidence": "high",
            "mapping_rationale": "Toy direct planar mapping.",
            "keypoint_mapping": dict(zip(source_planar, planar)),
            "viewpoint_character": ["low", "broadcast"],
            "source_group": dataset_name,
            "include_default": False,
        }
    mapping_path = tmp_path / "mapping.json"
    _write_json(mapping_path, mapping)
    guard_report = tmp_path / "parent_guard.json"
    guard_root = tmp_path / "guard"
    guard_root.mkdir()
    Image.new("RGB", (17, 13), (240, 10, 10)).save(guard_root / "protected.png")
    _write_json(
        guard_report,
        {"guard_image_files_hashed": 1, "guard_unique_hashes": 1, "guard_hash_matches_included": 0},
    )
    lane_dir = tmp_path / "lane"
    corpus_root = lane_dir / "real_court_corpus_partial"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset-root",
            str(dataset_root),
            "--mapping-table",
            str(mapping_path),
            "--lane-dir",
            str(lane_dir),
            "--output-root",
            str(corpus_root),
            "--partial-rows",
            "--reuse-guard-report",
            str(guard_report),
            "--guard-image-root",
            str(guard_root),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["corpus_rows"] == 15
    rows = load_real_court_keypoint_labels(corpus_root)
    assert len(rows) == 15
    assert {row["label_status"] for row in rows} == {"reviewed_external_dataset"}
    assert sorted({len(row["keypoints"]) for row in rows}) == [12, 15]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in corpus_root.glob("*/labels/court_keypoints.json")]
    partial_item = next(
        item
        for payload in payloads
        for item in payload["annotation"]["items"]
        if item["provenance"]["dataset"] == names[1]
    )
    assert set(partial_item["keypoints"]) == set(CANONICAL_KEYPOINTS)
    assert {name: partial_item["keypoints"][name] for name in CANONICAL_KEYPOINTS if name.startswith("net_")} == {
        "net_left_sideline": None,
        "net_center": None,
        "net_right_sideline": None,
    }
    proof = json.loads((lane_dir / "loader_contract_proof.json").read_text(encoding="utf-8"))
    assert proof["loaded_rows"] == 15
    assert proof["labeled_keypoint_histogram"] == {"12": 10, "15": 5}
    assert proof["schema_errors"] == 0
    assert proof["training_summary_label_buckets"]["labels_external_dataset_frame_count"] == 15
    assert proof["training_summary_label_buckets"]["labels_independent_human_frames"] == 0
    assert proof["partial_roundtrip"]["loader_omits_null_channels"] is True
    assert len(proof["random_12_point_net_null_spot_proof"]) == 5
    split = json.loads((lane_dir / "split_proposal.json").read_text(encoding="utf-8"))
    assert split["method"] == "by_dataset"
    assert len(split["val_datasets"]) == 2
    assert all(len(paths) == 5 for paths in json.loads((lane_dir / "corpus_stats.json").read_text())["emission_overlays"].values())
