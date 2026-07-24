from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.racketsport.build_court_v31_protocol import (
    CorpusSpec,
    build_protocol,
    load_protocol_partition_rows,
)
from threed.racketsport.schemas import PICKLEBALL_COURT_KEYPOINT_NAMES


def _keypoints() -> dict[str, list[float] | None]:
    floor = {
        "near_left_corner": [40.0, 330.0],
        "near_baseline_center": [320.0, 330.0],
        "near_right_corner": [600.0, 330.0],
        "far_right_corner": [450.0, 60.0],
        "far_baseline_center": [320.0, 60.0],
        "far_left_corner": [190.0, 60.0],
        "near_nvz_left": [130.0, 235.0],
        "near_nvz_center": [320.0, 235.0],
        "near_nvz_right": [510.0, 235.0],
        "far_nvz_left": [235.0, 135.0],
        "far_nvz_center": [320.0, 135.0],
        "far_nvz_right": [405.0, 135.0],
    }
    return {name: floor.get(name) for name in PICKLEBALL_COURT_KEYPOINT_NAMES}


def _write_row(
    root: Path,
    *,
    clip: str,
    group: str,
    image: np.ndarray,
    image_format: str,
    source_identity: str,
) -> None:
    frame_dir = root / clip / "frames"
    label_dir = root / clip / "labels"
    frame_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    suffix = ".jpg" if image_format == "JPEG" else ".png"
    frame_name = f"frame_000001{suffix}"
    Image.fromarray(image).save(frame_dir / frame_name, format=image_format, quality=78)
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_court_keypoint_labels",
        "clip": clip,
        "annotation": {
            "items": [
                {
                    "frame": frame_name,
                    "status": "reviewed",
                    "keypoints": _keypoints(),
                    "provenance": {
                        "venue_id": group,
                        "workspace": "fixture_workspace",
                        "source_sha256": source_identity,
                    },
                }
            ]
        },
        "frames": {
            "frame_dir": f"{clip}/frames",
            "path_base": "corpus_root",
            "source_resolution": [640, 360],
            "label_coordinate_space": [640, 360],
        },
        "review": {"status": "reviewed", "reviewer": "fixture"},
    }
    (label_dir / "court_keypoints.json").write_text(json.dumps(payload), encoding="utf-8")


def _image(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(360, 640, 3), dtype=np.uint8)


def test_source_and_duplicate_components_never_cross_five_fold_partitions(tmp_path: Path) -> None:
    """Direct coverage for scripts/racketsport/build_court_v31_protocol.py."""

    left = tmp_path / "left"
    right = tmp_path / "right"
    exact = _image(1)
    perceptual = _image(2)
    _write_row(
        left,
        clip="exact_a",
        group="exact_group_a",
        image=exact,
        image_format="PNG",
        source_identity="1" * 64,
    )
    _write_row(
        right,
        clip="exact_b",
        group="exact_group_b",
        image=exact,
        image_format="PNG",
        source_identity="2" * 64,
    )
    _write_row(
        left,
        clip="perceptual_a",
        group="perceptual_group_a",
        image=perceptual,
        image_format="PNG",
        source_identity="3" * 64,
    )
    _write_row(
        right,
        clip="perceptual_b",
        group="perceptual_group_b",
        image=perceptual,
        image_format="JPEG",
        source_identity="4" * 64,
    )
    for index, root in enumerate((left, right, left), start=5):
        _write_row(
            root,
            clip=f"unique_{index}",
            group=f"unique_group_{index}",
            image=_image(index),
            image_format="PNG",
            source_identity=str(index) * 64,
        )

    corpora = [
        CorpusSpec("left", left, "external"),
        CorpusSpec("right", right, "owner"),
    ]
    report = build_protocol(corpora, fold_count=5, seed=13, phash_distance=4)

    assert report["counts"]["usable_rows"] == 7
    assert report["counts"]["declared_source_groups"] == 7
    assert report["counts"]["dedup_connected_components"] == 5
    assert report["counts"]["exact_cross_group_edges"] == 1
    assert report["counts"]["perceptual_cross_group_edges"] == 1
    assert report["task88"]["included_in_rows"] is False
    assert report["task88"]["role"] == "historical_development_only"

    by_key = {row["row_key"]: row for row in report["rows"]}
    assert by_key["left/exact_a/frame_000001"]["fold_bucket"] == by_key[
        "right/exact_b/frame_000001"
    ]["fold_bucket"]
    assert by_key["left/perceptual_a/frame_000001"]["fold_bucket"] == by_key[
        "right/perceptual_b/frame_000001"
    ]["fold_bucket"]

    all_keys = set(by_key)
    for fold in report["folds"]:
        partitions = {name: set(keys) for name, keys in fold["partitions"].items()}
        assert set().union(*partitions.values()) == all_keys
        assert partitions["train"].isdisjoint(partitions["validation"])
        assert partitions["train"].isdisjoint(partitions["test"])
        assert partitions["validation"].isdisjoint(partitions["test"])
        assert fold["leakage_audit"] == {
            "passed": True,
            "source_group_cross_partition_count": 0,
            "exact_content_cross_partition_pair_count": 0,
            "perceptual_near_duplicate_cross_partition_pair_count": 0,
        }

    assert build_protocol(corpora, fold_count=5, seed=13, phash_distance=4) == report

    manifest_path = tmp_path / "protocol.json"
    manifest_path.write_text(json.dumps(report), encoding="utf-8")
    held_out = load_protocol_partition_rows(manifest_path, fold_index=0, partition="test")
    assert len(held_out) == report["folds"][0]["counts"]["test"]
    assert all(row["protocol_partition"] == "test" for row in held_out)
    assert all(row["source_group"] and row["viewpoint"] and row["visibility"] for row in held_out)


def test_cli_help_exposes_frozen_protocol_controls() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/racketsport/build_court_v31_protocol.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--fold-count" in completed.stdout
    assert "--phash-distance" in completed.stdout
    assert hashlib.sha256(completed.stdout.encode()).hexdigest()
