from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
from PIL import Image

from scripts.racketsport.train_court_keypoint_heatmap import load_real_court_keypoint_labels
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/racketsport/adapt_roboflow_court_keypoints.py"
WORKSPACE = "chetan-rajagiri-9abfm__pickleball-court-v2__v1"
SOURCE_NAMES = [
    "far_left_baseline",
    "far_center_baseline",
    "far_right_baseline",
    "far_left_kitchen",
    "far_center_kitchen",
    "far_right_kitchen",
    "net_left",
    "net_center",
    "net_right",
    "near_left_kitchen",
    "near_center_kitchen",
    "near_right_kitchen",
    "near_left_baseline",
    "near_center_baseline",
    "near_right_baseline",
]


def _write_coco_fixture(dataset_root: Path) -> None:
    split = dataset_root / WORKSPACE / "train"
    split.mkdir(parents=True)
    file_name = "frame_000001_jpg.rf.0123456789abcdef0123456789abcdef.jpg"
    Image.new("RGB", (640, 640), (44, 91, 70)).save(split / file_name)

    # Chetan source far_* is image-near and source near_* is image-far. These
    # points form an exact projective regulation layout after that correction.
    source_points = {
        "far_left_baseline": [50.0, 580.0],
        "far_center_baseline": [320.0, 580.0],
        "far_right_baseline": [590.0, 580.0],
        "far_left_kitchen": [50.0, 416.0],
        "far_center_kitchen": [320.0, 416.0],
        "far_right_kitchen": [590.0, 416.0],
        "net_left": [150.0, 340.0],
        "net_center": [320.0, 340.0],
        "net_right": [490.0, 340.0],
        "near_left_kitchen": [50.0, 264.0],
        "near_center_kitchen": [320.0, 264.0],
        "near_right_kitchen": [590.0, 264.0],
        "near_left_baseline": [50.0, 100.0],
        "near_center_baseline": [320.0, 100.0],
        "near_right_baseline": [590.0, 100.0],
    }
    keypoints = [
        value
        for name in SOURCE_NAMES
        for value in (*source_points[name], 2)
    ]
    payload = {
        "images": [{"id": 1, "file_name": file_name, "width": 640, "height": 640}],
        "annotations": [
            {
                "id": 11,
                "image_id": 1,
                "category_id": 1,
                "keypoints": keypoints,
                "num_keypoints": 15,
                "bbox": [40, 90, 560, 500],
            }
        ],
        "categories": [
            {"id": 0, "name": "pickleball-court-v2"},
            {"id": 1, "name": "court", "keypoints": SOURCE_NAMES, "skeleton": []},
        ],
    }
    (split / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")


def test_direct_cli_emits_canonical_partial_loader_contract(tmp_path: Path) -> None:
    dataset_root = tmp_path / "roboflow"
    lane_dir = tmp_path / "roboflow_court_adapter_20260723"
    _write_coco_fixture(dataset_root)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset-root",
            str(dataset_root),
            "--lane-dir",
            str(lane_dir),
            "--owner-pack-root",
            str(tmp_path / "absent_owner_pack"),
            "--mode",
            "diagnostic",
            "--validation-samples",
            "1",
            "--max-per-apparent-venue",
            "5",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = json.loads(completed.stdout)
    assert stdout["status"] == "PASS"
    assert stdout["verified"] is False
    assert stdout["adapted_usable_labels"] == 1

    label_paths = list((lane_dir / "adapted_corpus").glob("*/labels/court_keypoints.json"))
    assert len(label_paths) == 1
    payload = json.loads(label_paths[0].read_text(encoding="utf-8"))
    item = payload["annotation"]["items"][0]
    canonical_order = [point.name for point in PICKLEBALL_KEYPOINTS]
    assert list(item["keypoints"]) == canonical_order
    assert item["keypoints"]["far_left_corner"] == [50.0, 100.0]
    assert item["keypoints"]["near_left_corner"] == [50.0, 580.0]
    assert item["keypoints"]["net_left_sideline"] is None
    assert item["keypoints"]["net_center"] is None
    assert item["keypoints"]["net_right_sideline"] is None
    assert item["provenance"]["mapping_mode"] == "semantic_direct_with_source_depth_reversal"
    assert item["pseudo_label_status"] == "PENDING_SPOTCHECK"

    with pytest.raises(ValueError, match="diagnostic-only"):
        load_real_court_keypoint_labels(lane_dir / "adapted_corpus")
    rows = load_real_court_keypoint_labels(
        lane_dir / "adapted_corpus",
        allow_pending_diagnostic_only=True,
    )
    assert len(rows) == 1
    assert len(rows[0]["keypoints"]) == 12
    assert set(rows[0]["keypoints"]).isdisjoint(
        {"net_left_sideline", "net_center", "net_right_sideline"}
    )

    report = json.loads((lane_dir / "report.json").read_text(encoding="utf-8"))
    assert report["counts"]["court_workspaces_enumerated"] == 1
    assert report["counts"]["adapted_usable_labels_after_dedup_sanity_and_venue_cap"] == 1
    assert report["loader_validation"]["status"] == "PASS"
    assert report["loader_validation"]["default_training_loader_passed"] is False
    assert report["exclusions"]["selected_owner_pack_collision_count"] == 0
    assert len(list((lane_dir / "validation_pngs").glob("*.png"))) == 1
    assert (lane_dir / "PROPOSED_LEDGER_ROW.json").is_file()
