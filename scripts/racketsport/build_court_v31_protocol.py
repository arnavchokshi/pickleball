#!/usr/bin/env python3
"""Build the source-grouped court v3.1 five-fold evaluation protocol.

Rows are never randomly split. Declared source/venue groups are first joined with exact-content
and perceptual-near-duplicate edges, then complete connected components are assigned to folds.
This keeps duplicate observations in one partition without discarding owner label upgrades.
Task 88 is deliberately absent: it remains historical development evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport.train_court_model_v2 import load_real_training_rows  # noqa: E402
from threed.racketsport.schemas import PICKLEBALL_COURT_KEYPOINT_NAMES  # noqa: E402


NET_TOP_NAMES = frozenset({"net_left_sideline", "net_center", "net_right_sideline"})
FLOOR_NAMES = tuple(name for name in PICKLEBALL_COURT_KEYPOINT_NAMES if name not in NET_TOP_NAMES)
DEFAULT_FOLD_COUNT = 5
DEFAULT_SEED = 13
DEFAULT_PHASH_DISTANCE = 4


@dataclass(frozen=True)
class CorpusSpec:
    corpus_id: str
    root: Path
    subgroup: str


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_index(frame_name: str) -> int:
    try:
        return int(Path(frame_name).stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot parse frame index from {frame_name}") from exc


def _perceptual_hashes(path: Path) -> tuple[str, str]:
    import cv2
    import numpy as np

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"could not decode image: {path}")
    phash_image = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(phash_image)[:8, :8]
    median = float(np.median(low.reshape(-1)[1:]))
    phash_bits = (low > median).reshape(-1)
    phash = 0
    for bit in phash_bits:
        phash = (phash << 1) | int(bool(bit))

    dhash_image = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    dhash_bits = (dhash_image[:, 1:] > dhash_image[:, :-1]).reshape(-1)
    dhash = 0
    for bit in dhash_bits:
        dhash = (dhash << 1) | int(bool(bit))
    return f"{phash:016x}", f"{dhash:016x}"


def _viewpoint(keypoints: dict[str, Any]) -> str:
    required = ("near_left_corner", "near_right_corner", "far_left_corner", "far_right_corner")
    if not all(keypoints.get(name) is not None for name in required):
        return "unknown_partial"
    near_left, near_right, far_left, far_right = (keypoints[name] for name in required)
    near_center_x = (float(near_left[0]) + float(near_right[0])) / 2.0
    far_center_x = (float(far_left[0]) + float(far_right[0])) / 2.0
    near_width = math.dist(near_left, near_right)
    far_width = math.dist(far_left, far_right)
    normalized_offset = abs(near_center_x - far_center_x) / max(near_width, far_width, 1.0)
    return "straight" if normalized_offset <= 0.10 else "diagonal"


def _visibility(keypoints: dict[str, Any]) -> str:
    visible = sum(keypoints.get(name) is not None for name in FLOOR_NAMES)
    if visible == len(FLOOR_NAMES):
        return "full_floor_12"
    if visible >= 9:
        return "partial_floor_9_to_11"
    if visible >= 4:
        return "partial_floor_4_to_8"
    return "sparse_floor_below_4"


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _collect_rows(corpus: CorpusSpec) -> list[dict[str, Any]]:
    root = corpus.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    eligible = load_real_training_rows([root])
    eligible_keys = {
        (str(row["clip"]), int(row["frame_index"]))
        for row in eligible
    }
    rows: list[dict[str, Any]] = []
    for label_path in sorted(root.glob("*/labels/court_keypoints.json")):
        payload = _read_object(label_path)
        clip = str(payload.get("clip") or label_path.parent.parent.name)
        frames = payload.get("frames")
        annotation = payload.get("annotation")
        if not isinstance(frames, dict) or not isinstance(annotation, dict):
            raise ValueError(f"malformed label payload: {label_path}")
        frame_dir_raw = frames.get("frame_dir")
        if not isinstance(frame_dir_raw, str) or not frame_dir_raw:
            raise ValueError(f"missing frames.frame_dir: {label_path}")
        frame_dir = Path(frame_dir_raw)
        if not frame_dir.is_absolute():
            frame_dir = root / frame_dir
        items = annotation.get("items")
        if not isinstance(items, list):
            raise ValueError(f"missing annotation.items: {label_path}")
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("frame"), str):
                raise ValueError(f"malformed annotation item: {label_path}")
            frame_name = item["frame"]
            frame_index = _frame_index(frame_name)
            if (clip, frame_index) not in eligible_keys:
                continue
            image_path = frame_dir / frame_name
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            keypoints = item.get("keypoints")
            provenance = item.get("provenance", {})
            if not isinstance(keypoints, dict) or not isinstance(provenance, dict):
                raise ValueError(f"malformed keypoints/provenance: {label_path}#{frame_name}")
            declared_group = (
                provenance.get("apparent_venue_id")
                or provenance.get("venue_id")
                or provenance.get("source_group")
                or provenance.get("dataset")
                or clip
            )
            source = provenance.get("workspace") or provenance.get("dataset") or corpus.corpus_id
            if not isinstance(declared_group, str) or not declared_group:
                raise ValueError(f"missing declared source group: {label_path}#{frame_name}")
            if not isinstance(source, str) or not source:
                raise ValueError(f"missing source: {label_path}#{frame_name}")
            actual_sha = _sha256(image_path)
            declared_identities = {
                value
                for value in (
                    provenance.get("source_sha256"),
                    provenance.get("image_sha256"),
                    provenance.get("frame_sha256"),
                    actual_sha,
                )
                if isinstance(value, str) and len(value) == 64
            }
            phash, dhash = _perceptual_hashes(image_path)
            row_key = f"{corpus.corpus_id}/{clip}/frame_{frame_index:06d}"
            rows.append(
                {
                    "row_key": row_key,
                    "corpus_id": corpus.corpus_id,
                    "corpus_root": _repo_path(root),
                    "clip": clip,
                    "frame_index": frame_index,
                    "image_path": _repo_path(image_path),
                    "actual_image_sha256": actual_sha,
                    "declared_content_sha256": sorted(declared_identities),
                    "perceptual_hash64": phash,
                    "difference_hash64": dhash,
                    "declared_source_group": f"{corpus.corpus_id}:{declared_group}",
                    "source": source,
                    "subgroup": corpus.subgroup,
                    "viewpoint": _viewpoint(keypoints),
                    "visibility": _visibility(keypoints),
                    "visible_floor_keypoint_count": sum(
                        keypoints.get(name) is not None for name in FLOOR_NAMES
                    ),
                    "visible_net_keypoint_count": sum(
                        keypoints.get(name) is not None for name in NET_TOP_NAMES
                    ),
                    "label_status": item.get("status", "reviewed"),
                }
            )
    if len(rows) != len(eligible):
        raise ValueError(
            f"eligible/materialized row mismatch for {corpus.corpus_id}: "
            f"loader={len(eligible)} collected={len(rows)}"
        )
    return rows


def _partition_leakage(
    rows: list[dict[str, Any]],
    partition_by_row: dict[str, str],
    *,
    phash_distance: int,
) -> dict[str, Any]:
    source_group_leaks: set[tuple[str, str, str]] = set()
    exact_pairs = 0
    perceptual_pairs = 0
    for index, left in enumerate(rows):
        left_partition = partition_by_row[left["row_key"]]
        for right in rows[index + 1 :]:
            right_partition = partition_by_row[right["row_key"]]
            if left_partition == right_partition:
                continue
            if left["declared_source_group"] == right["declared_source_group"]:
                source_group_leaks.add(
                    (left["declared_source_group"], left_partition, right_partition)
                )
            if set(left["declared_content_sha256"]) & set(right["declared_content_sha256"]):
                exact_pairs += 1
            distance = (int(left["perceptual_hash64"], 16) ^ int(right["perceptual_hash64"], 16)).bit_count()
            if distance <= phash_distance:
                perceptual_pairs += 1
    return {
        "passed": not source_group_leaks and exact_pairs == 0 and perceptual_pairs == 0,
        "source_group_cross_partition_count": len(source_group_leaks),
        "exact_content_cross_partition_pair_count": exact_pairs,
        "perceptual_near_duplicate_cross_partition_pair_count": perceptual_pairs,
    }


def build_protocol(
    corpora: list[CorpusSpec],
    *,
    fold_count: int = DEFAULT_FOLD_COUNT,
    seed: int = DEFAULT_SEED,
    phash_distance: int = DEFAULT_PHASH_DISTANCE,
) -> dict[str, Any]:
    if not corpora:
        raise ValueError("at least one corpus is required")
    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or fold_count < 3:
        raise ValueError("fold_count must be an integer >= 3")
    if not 0 <= phash_distance <= 64:
        raise ValueError("phash_distance must be in [0,64]")
    corpus_ids = [corpus.corpus_id for corpus in corpora]
    if len(set(corpus_ids)) != len(corpus_ids):
        raise ValueError("corpus ids must be unique")

    rows = sorted(
        (row for corpus in corpora for row in _collect_rows(corpus)),
        key=lambda row: row["row_key"],
    )
    if len({row["row_key"] for row in rows}) != len(rows):
        raise ValueError("row keys must be unique")
    disjoint = _DisjointSet(len(rows))
    first_by_group: dict[str, int] = {}
    for index, row in enumerate(rows):
        prior = first_by_group.setdefault(row["declared_source_group"], index)
        disjoint.union(index, prior)

    exact_edges: list[dict[str, Any]] = []
    perceptual_edges: list[dict[str, Any]] = []
    for left_index, left in enumerate(rows):
        left_hashes = set(left["declared_content_sha256"])
        for right_index in range(left_index + 1, len(rows)):
            right = rows[right_index]
            if left["declared_source_group"] == right["declared_source_group"]:
                continue
            shared = sorted(left_hashes & set(right["declared_content_sha256"]))
            phash_hamming = (
                int(left["perceptual_hash64"], 16) ^ int(right["perceptual_hash64"], 16)
            ).bit_count()
            dhash_hamming = (
                int(left["difference_hash64"], 16) ^ int(right["difference_hash64"], 16)
            ).bit_count()
            if shared:
                disjoint.union(left_index, right_index)
                exact_edges.append(
                    {
                        "left": left["row_key"],
                        "right": right["row_key"],
                        "shared_sha256": shared,
                        "phash_hamming": phash_hamming,
                        "dhash_hamming": dhash_hamming,
                    }
                )
            elif phash_hamming <= phash_distance:
                disjoint.union(left_index, right_index)
                perceptual_edges.append(
                    {
                        "left": left["row_key"],
                        "right": right["row_key"],
                        "phash_hamming": phash_hamming,
                        "dhash_hamming": dhash_hamming,
                    }
                )

    component_indices: dict[int, list[int]] = {}
    for index in range(len(rows)):
        component_indices.setdefault(disjoint.find(index), []).append(index)
    components: list[dict[str, Any]] = []
    for indices in component_indices.values():
        row_keys = sorted(rows[index]["row_key"] for index in indices)
        component_id = "dupgrp_" + hashlib.sha256("\n".join(row_keys).encode()).hexdigest()[:16]
        components.append(
            {
                "component_id": component_id,
                "row_keys": row_keys,
                "row_count": len(row_keys),
                "declared_source_groups": sorted(
                    {rows[index]["declared_source_group"] for index in indices}
                ),
                "sources": sorted({rows[index]["source"] for index in indices}),
                "subgroups": sorted({rows[index]["subgroup"] for index in indices}),
            }
        )
    if len(components) < fold_count:
        raise ValueError(
            f"dedup/source grouping produced {len(components)} components for {fold_count} folds"
        )

    rng = random.Random(seed)
    tie_breaks = {component["component_id"]: rng.random() for component in components}
    ordered_components = sorted(
        components,
        key=lambda component: (
            -int(component["row_count"]),
            tie_breaks[component["component_id"]],
            component["component_id"],
        ),
    )
    buckets: list[list[dict[str, Any]]] = [[] for _fold in range(fold_count)]
    bucket_counts = [0 for _fold in range(fold_count)]
    for component in ordered_components:
        fold_index = min(range(fold_count), key=lambda index: (bucket_counts[index], index))
        buckets[fold_index].append(component)
        bucket_counts[fold_index] += int(component["row_count"])

    component_bucket = {
        component["component_id"]: fold_index
        for fold_index, bucket in enumerate(buckets)
        for component in bucket
    }
    row_component = {
        row_key: component["component_id"]
        for component in components
        for row_key in component["row_keys"]
    }
    for row in rows:
        row["dedup_component_id"] = row_component[row["row_key"]]
        row["fold_bucket"] = component_bucket[row_component[row["row_key"]]]

    folds: list[dict[str, Any]] = []
    for fold_index in range(fold_count):
        validation_bucket = (fold_index + 1) % fold_count
        partition_by_row: dict[str, str] = {}
        for row in rows:
            bucket = int(row["fold_bucket"])
            partition_by_row[row["row_key"]] = (
                "test"
                if bucket == fold_index
                else "validation"
                if bucket == validation_bucket
                else "train"
            )
        partitions = {
            partition: sorted(
                row_key for row_key, assigned in partition_by_row.items() if assigned == partition
            )
            for partition in ("train", "validation", "test")
        }
        leakage = _partition_leakage(
            rows,
            partition_by_row,
            phash_distance=phash_distance,
        )
        if not leakage["passed"]:
            raise AssertionError(f"fold {fold_index} leakage audit failed: {leakage}")
        folds.append(
            {
                "fold_index": fold_index,
                "validation_bucket": validation_bucket,
                "partitions": partitions,
                "counts": {name: len(keys) for name, keys in partitions.items()},
                "leakage_audit": leakage,
            }
        )

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_court_v31_source_grouped_five_fold_protocol",
        "status": "FROZEN_EVALUATION_PROTOCOL",
        "verified": False,
        "seed": seed,
        "fold_count": fold_count,
        "corpora": [
            {
                "corpus_id": corpus.corpus_id,
                "root": _repo_path(corpus.root),
                "subgroup": corpus.subgroup,
            }
            for corpus in corpora
        ],
        "counts": {
            "usable_rows": len(rows),
            "declared_source_groups": len({row["declared_source_group"] for row in rows}),
            "dedup_connected_components": len(components),
            "exact_cross_group_edges": len(exact_edges),
            "perceptual_cross_group_edges": len(perceptual_edges),
            "fold_test_rows_min": min(bucket_counts),
            "fold_test_rows_max": max(bucket_counts),
        },
        "deduplication": {
            "policy": "retain_rows_but_bind_connected_duplicates_to_one_partition",
            "exact_identity": "declared source/image/frame SHA-256 plus staged-image SHA-256",
            "perceptual_algorithm": "opencv_dct_phash_8x8_64bit",
            "perceptual_hamming_threshold_inclusive": phash_distance,
            "exact_cross_group_edges": exact_edges,
            "perceptual_cross_group_edges": perceptual_edges,
        },
        "components": sorted(components, key=lambda component: component["component_id"]),
        "rows": rows,
        "folds": folds,
        "evaluation_contract": {
            "semantic_matching": "exact_name_only",
            "floor_taxonomy_point_count": len(FLOOR_NAMES),
            "metrics": [
                "PCK@2px",
                "PCK@5px",
                "PCK@10px",
                "median_error_px",
                "p90_error_px",
                "p95_error_px",
                "max_error_px",
            ],
            "strata": ["subgroup", "source", "source_group", "viewpoint", "visibility"],
            "paired_comparison": "raw_vs_structured_sample_cluster_bootstrap",
            "fold_selection_policy": "all five held-out test folds reported; no best-fold cherry-pick",
        },
        "task88": {
            "name": "CVAT Task 88",
            "included_in_rows": False,
            "role": "historical_development_only",
            "allowed_use": "continuity comparison against earlier experiments",
            "forbidden_use": [
                "fold_assignment",
                "checkpoint_selection",
                "confidence_calibration",
                "promotion_evidence",
            ],
        },
    }


def load_protocol_partition_rows(
    manifest_path: Path,
    *,
    fold_index: int,
    partition: str = "test",
) -> list[dict[str, Any]]:
    """Load one frozen partition and attach its recorded evaluation strata to trainer rows."""

    payload = _read_object(manifest_path)
    if payload.get("artifact_type") != "racketsport_court_v31_source_grouped_five_fold_protocol":
        raise ValueError("unsupported court v3.1 protocol manifest")
    folds = payload.get("folds")
    if not isinstance(folds, list):
        raise ValueError("protocol manifest requires folds")
    fold = next(
        (
            candidate
            for candidate in folds
            if isinstance(candidate, dict) and candidate.get("fold_index") == fold_index
        ),
        None,
    )
    if fold is None:
        raise ValueError(f"protocol fold {fold_index} does not exist")
    if partition not in {"train", "validation", "test"}:
        raise ValueError("partition must be train, validation, or test")
    partitions = fold.get("partitions")
    if not isinstance(partitions, dict) or not isinstance(partitions.get(partition), list):
        raise ValueError(f"protocol fold {fold_index} is missing partition {partition}")
    selected = set(partitions[partition])

    manifest_rows = payload.get("rows")
    corpora = payload.get("corpora")
    if not isinstance(manifest_rows, list) or not isinstance(corpora, list):
        raise ValueError("protocol manifest requires rows and corpora")
    metadata = {
        row["row_key"]: row
        for row in manifest_rows
        if isinstance(row, dict) and isinstance(row.get("row_key"), str)
    }
    if selected - set(metadata):
        raise ValueError("protocol partition references unknown row keys")

    loaded_by_key: dict[str, dict[str, Any]] = {}
    for corpus in corpora:
        if not isinstance(corpus, dict):
            raise ValueError("protocol corpus must be an object")
        corpus_id = corpus.get("corpus_id")
        root_raw = corpus.get("root")
        if not isinstance(corpus_id, str) or not isinstance(root_raw, str):
            raise ValueError("protocol corpus requires corpus_id and root")
        root = Path(root_raw)
        if not root.is_absolute():
            root = ROOT / root
        for row in load_real_training_rows([root]):
            row_key = f"{corpus_id}/{row['clip']}/frame_{int(row['frame_index']):06d}"
            if row_key not in selected:
                continue
            protocol_row = metadata[row_key]
            enriched = dict(row)
            enriched.update(
                {
                    "protocol_row_key": row_key,
                    "protocol_fold_index": fold_index,
                    "protocol_partition": partition,
                    "subgroup": protocol_row["subgroup"],
                    "source": protocol_row["source"],
                    "source_group": protocol_row["declared_source_group"],
                    "viewpoint": protocol_row["viewpoint"],
                    "visibility": protocol_row["visibility"],
                }
            )
            if row_key in loaded_by_key:
                raise ValueError(f"duplicate loaded protocol row: {row_key}")
            loaded_by_key[row_key] = enriched
    missing = sorted(selected - set(loaded_by_key))
    if missing:
        raise ValueError(f"protocol rows could not be loaded: {missing[:5]}")
    return [loaded_by_key[row_key] for row_key in sorted(selected)]


def _parse_corpus(value: str) -> CorpusSpec:
    parts = value.split("=", 2)
    if len(parts) != 3 or any(not part for part in parts):
        raise argparse.ArgumentTypeError("--corpus must be CORPUS_ID=PATH=SUBGROUP")
    return CorpusSpec(parts[0], Path(parts[1]), parts[2])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        action="append",
        type=_parse_corpus,
        help="Repeat as CORPUS_ID=PATH=SUBGROUP. Defaults to curated264 plus owner pack3.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=DEFAULT_FOLD_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--phash-distance", type=int, default=DEFAULT_PHASH_DISTANCE)
    args = parser.parse_args(argv)
    corpora = args.corpus or [
        CorpusSpec(
            "curated264",
            ROOT / "runs/lanes/roboflow_court_adapter_20260723/adapted_corpus",
            "roboflow_external_floor_only",
        ),
        CorpusSpec(
            "owner_pack3",
            ROOT / "runs/lanes/court_labelpack3_owner_ingest_20260723/train",
            "owner_reviewed_pack3",
        ),
    ]
    report = build_protocol(
        corpora,
        fold_count=args.fold_count,
        seed=args.seed,
        phash_distance=args.phash_distance,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=args.out.parent,
        prefix=f".{args.out.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, args.out)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
