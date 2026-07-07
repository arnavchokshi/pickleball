from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageOps

from .ball_tracknet_cvat_dataset import (
    PUBLIC_UNKNOWN_VISIBILITY_POLICY,
    PUBLIC_UNKNOWN_VISIBILITY_WBCE_WEIGHT,
)


ADJACENT_SPORT_SLUGS = {
    "pickleball-detection/pickleball-5pshr",
    "hughs-workspace-plw3g/pickleball-with-players-topw1",
    "pickleball-kjawm/tennis-ball-detection-sxi3e-inzuo",
}

AMBIGUOUS_SPOT_CHECK = {
    "slug": "pickleball-kjawm/tennis-ball-detection-sxi3e-inzuo",
    "sample_count": 12,
    "result": "all_12_visual_samples_are_tennis_broadcast_frames",
}

BUCKET_CORE = "core_pickleball"
BUCKET_ADJACENT = "adjacent_sport_aux"
BUCKET_DUPLICATE = "excluded_duplicate"
BUCKET_DEAD = "excluded_dead"
HASH_TYPE = "dhash_8x8_64bit"
DEFAULT_DEDUP_THRESHOLD = 3
DEFAULT_EVAL_SAMPLE_EVERY_S = 2.0
DEFAULT_PROTECTED_EVAL_HASH_COUNT = 35
DEFAULT_BALL_PRETRAIN_IMAGE_SIZE = (512, 288)
DEFAULT_BALL_PRETRAIN_FRAMES_IN = 3
DEFAULT_BALL_PRETRAIN_HEATMAP_RADIUS_PX = 4.0
DEFAULT_CORE_TO_AUX_RATIO = 8
TRAIN_SOURCE_SPLITS = {"train"}
INTERNAL_VAL_SOURCE_SPLITS = {"valid", "val", "test"}


def normalize_dataset_entry(entry: Mapping[str, Any], *, repo_root: str | Path = ".") -> dict[str, Any]:
    """Normalize one downloaded Roboflow COCO dataset into an index of original image paths."""

    slug = str(entry.get("slug") or "")
    bucket, bucket_reason = initial_bucket_for_entry(entry)
    local_path = _resolve_path(entry.get("local_path"), repo_root=Path(repo_root))
    samples: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    original_class_counts: Counter[str] = Counter()
    annotation_count = 0
    missing_paths: list[str] = []
    parse_warnings: list[str] = []
    taxonomy_mapping: dict[str, str] = {}

    if bucket == BUCKET_DEAD:
        return _empty_dataset_index(
            entry,
            bucket=bucket,
            bucket_reason=bucket_reason,
            reason="source_not_downloaded",
        )

    coco_paths = sorted(local_path.glob("*/_annotations.coco.json"))
    if not coco_paths:
        raise ValueError(f"no COCO annotation files found under {local_path}")

    sequence_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    sample_group_keys: dict[int, tuple[str, str] | None] = {}

    for coco_path in coco_paths:
        split = coco_path.parent.name
        coco = _read_json(coco_path)
        categories = {
            int(category["id"]): str(category.get("name") or category["id"])
            for category in coco.get("categories", [])
            if "id" in category
        }
        for category_name in categories.values():
            taxonomy_mapping[category_name] = taxonomy_for_category(category_name)
        annotations_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for annotation in coco.get("annotations", []):
            if "image_id" not in annotation:
                continue
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        for image in coco.get("images", []):
            if "id" not in image or "file_name" not in image:
                parse_warnings.append(f"{coco_path}: image missing id or file_name")
                continue
            image_id = int(image["id"])
            image_path = coco_path.parent / str(image["file_name"])
            if not image_path.is_file():
                missing_paths.append(str(image_path))
            original_name = _original_image_name(image)
            prefix, frame_number = _filename_sequence_parts(original_name)
            group_key = (split, prefix) if prefix is not None and frame_number is not None else None
            if group_key is not None:
                sequence_groups[group_key].append(len(samples))
            sample_group_keys[len(samples)] = group_key
            labels = {"ball": [], "person": [], "court": [], "paddle": [], "other": []}
            for annotation in annotations_by_image.get(image_id, []):
                category_name = categories.get(int(annotation.get("category_id", -1)), str(annotation.get("category_id")))
                taxonomy = taxonomy_for_category(category_name)
                original_class_counts[category_name] += 1
                label = _normalize_annotation(annotation, category_name=category_name, taxonomy=taxonomy)
                if label is None:
                    continue
                labels[taxonomy].append(label)
                label_counts[taxonomy] += 1
                annotation_count += 1
            sample_id = f"{_safe_slug(slug)}:{split}:{image_id}"
            samples.append(
                {
                    "sample_id": sample_id,
                    "source_slug": slug,
                    "split": split,
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "original_file_name": original_name,
                    "width": int(image.get("width") or 0),
                    "height": int(image.get("height") or 0),
                    "labels": labels,
                    "label_kinds": sorted(kind for kind, values in labels.items() if values),
                    "temporal": {
                        "kind": "isolated_still",
                        "sequence_id": None,
                        "filename_prefix": prefix,
                        "frame_number": frame_number,
                        "sequence_group_size": 1,
                    },
                }
            )

    for group_key, positions in sequence_groups.items():
        if len(positions) < 2:
            continue
        split, prefix = group_key
        sequence_id = f"{_safe_slug(slug)}:{split}:{prefix}"
        ordered = sorted(
            positions,
            key=lambda pos: (
                samples[pos]["temporal"]["frame_number"]
                if samples[pos]["temporal"]["frame_number"] is not None
                else math.inf
            ),
        )
        for order_index, position in enumerate(ordered):
            samples[position]["temporal"].update(
                {
                    "kind": "temporal_sequence",
                    "sequence_id": sequence_id,
                    "sequence_group_size": len(positions),
                    "sequence_order": order_index,
                }
            )

    temporal_counts = Counter(sample["temporal"]["kind"] for sample in samples)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_roboflow_dataset_index",
        "source": _source_summary(entry),
        "bucket": bucket,
        "bucket_reason": bucket_reason,
        "statistics": {
            "sample_count": len(samples),
            "annotation_count": annotation_count,
            "missing_path_count": len(missing_paths),
            "label_counts_by_taxonomy": dict(sorted(label_counts.items())),
            "original_class_counts": dict(sorted(original_class_counts.items())),
            "temporal_counts": dict(sorted(temporal_counts.items())),
        },
        "taxonomy_mapping": dict(sorted(taxonomy_mapping.items())),
        "missing_paths": missing_paths,
        "parse_warnings": parse_warnings,
        "samples": samples,
    }


def aggregate_roboflow_corpus(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    lane_dir: str | Path,
    eval_root: str | Path = "eval_clips/ball",
    repo_root: str | Path = ".",
    dedup_threshold: int = DEFAULT_DEDUP_THRESHOLD,
    eval_sample_every_s: float = DEFAULT_EVAL_SAMPLE_EVERY_S,
) -> dict[str, Any]:
    """Build the index-only aggregated Roboflow corpus artifacts."""

    repo_root = Path(repo_root)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    lane_dir = Path(lane_dir)
    per_dataset_dir = output_dir / "per_dataset"
    subset_dir = output_dir / "subset_indexes"
    per_dataset_dir.mkdir(parents=True, exist_ok=True)
    subset_dir.mkdir(parents=True, exist_ok=True)
    lane_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(manifest_path)
    entries = list(manifest.get("entries", []))
    normalized_indices: list[dict[str, Any]] = []
    parse_failures: list[dict[str, Any]] = []

    for entry in entries:
        try:
            index = normalize_dataset_entry(entry, repo_root=repo_root)
        except Exception as exc:
            parse_failures.append({"slug": entry.get("slug"), "error": str(exc)})
            index = _empty_dataset_index(
                entry,
                bucket=BUCKET_DEAD,
                bucket_reason=f"coco_parse_failed: {exc}",
                reason="coco_parse_failed",
            )
        normalized_indices.append(index)

    _attach_image_hashes(normalized_indices)
    duplicate_mappings = _mark_dataset_duplicates(normalized_indices)
    eval_hashes, eval_hash_source = _load_eval_hashes(
        eval_root=Path(eval_root),
        sample_every_s=eval_sample_every_s,
    )
    corpus_samples, subset_samples, dedup_summary, leakage_summary = _mark_sample_statuses(
        normalized_indices,
        eval_hashes=eval_hashes,
        dedup_threshold=dedup_threshold,
    )

    for index in normalized_indices:
        _write_json(
            per_dataset_dir / f"{_safe_slug(index['source']['slug'])}.index.json",
            index,
            compact=True,
        )

    subset_paths: dict[str, str] = {}
    for kind, samples in sorted(subset_samples.items()):
        path = subset_dir / f"{kind}_index.json"
        _write_json(
            path,
            {
                "schema_version": 1,
                "artifact_type": "racketsport_roboflow_subset_index",
                "label_kind": kind,
                "sample_count": len(samples),
                "samples": samples,
            },
            compact=True,
        )
        subset_paths[kind] = str(path)

    corpus_index = {
        "schema_version": 1,
        "artifact_type": "racketsport_roboflow_public_pretrain_corpus_index",
        "generated_utc": _utc_now(),
        "source_manifest": str(manifest_path),
        "index_policy": "index_based_original_image_paths_no_image_copies",
        "hash": {
            "type": HASH_TYPE,
            "collision_hamming_threshold": dedup_threshold,
            "eval_sample_every_s": eval_sample_every_s,
            "eval_hash_source": eval_hash_source,
        },
        "bucket_counts_by_source": _bucket_counts_by_source(normalized_indices),
        "sample_count": len(corpus_samples),
        "subset_indexes": subset_paths,
        "samples": corpus_samples,
    }
    corpus_index_path = output_dir / "corpus_index.json"
    _write_json(corpus_index_path, corpus_index, compact=True)

    card = _build_corpus_card(
        manifest=manifest,
        normalized_indices=normalized_indices,
        parse_failures=parse_failures,
        duplicate_mappings=duplicate_mappings,
        dedup_summary=dedup_summary,
        leakage_summary=leakage_summary,
        eval_hashes=eval_hashes,
        eval_hash_source=eval_hash_source,
        subset_paths=subset_paths,
        corpus_sample_count=len(corpus_samples),
        dedup_threshold=dedup_threshold,
        eval_sample_every_s=eval_sample_every_s,
    )
    card_json_path = output_dir / "corpus_card.json"
    card_md_path = output_dir / "corpus_card.md"
    _write_json(card_json_path, card)
    card_md_path.write_text(_format_corpus_card_md(card), encoding="utf-8")

    aggregation_summary_path = lane_dir / "aggregation_summary.json"
    _write_json(
        aggregation_summary_path,
        {
            "schema_version": 1,
            "artifact_type": "racketsport_roboflow_aggregation_summary",
            "corpus_index_path": str(corpus_index_path),
            "corpus_card_json_path": str(card_json_path),
            "corpus_card_md_path": str(card_md_path),
            "parse_failures": parse_failures,
            "bucket_counts_by_source": card["bucket_counts_by_source"],
            "dedup": card["dedup"],
            "temporal_split": card["temporal_split"],
            "leakage_check": card["leakage_check"],
        },
    )

    return {
        "objective_result": "PASS" if _p10_gate_passed(card) else "PARTIAL",
        "corpus_index_path": str(corpus_index_path),
        "corpus_card_path": str(card_json_path),
        "corpus_card_md_path": str(card_md_path),
        "aggregation_summary_path": str(aggregation_summary_path),
        "parse_failure_count": len(parse_failures),
        "downloaded_source_count": sum(1 for entry in entries if entry.get("status") == "downloaded"),
        "bucket_counts_by_source": card["bucket_counts_by_source"],
        "dedup": card["dedup"],
        "leakage_check": card["leakage_check"],
    }


def load_smoke_samples(
    manifest_path: str | Path,
    *,
    repo_root: str | Path = ".",
    limit: int = 50,
    min_datasets: int = 5,
) -> dict[str, Any]:
    manifest = _read_json(Path(manifest_path))
    entries = [
        entry
        for entry in manifest.get("entries", [])
        if entry.get("status") == "downloaded" and entry.get("local_path")
    ]
    opened = 0
    datasets_seen: set[str] = set()
    missing_paths: list[str] = []
    label_kinds: set[str] = set()
    per_dataset_budget = max(1, limit // max(1, min_datasets))

    for entry in entries:
        if opened >= limit and len(datasets_seen) >= min_datasets:
            break
        try:
            index = normalize_dataset_entry(entry, repo_root=repo_root)
        except Exception:
            continue
        opened_for_dataset = 0
        for sample in index["samples"]:
            if opened_for_dataset >= per_dataset_budget and len(datasets_seen) < min_datasets:
                break
            if opened >= limit and len(datasets_seen) >= min_datasets:
                break
            image_path = Path(sample["image_path"])
            if not image_path.is_file():
                missing_paths.append(str(image_path))
                continue
            with Image.open(image_path) as image:
                image.verify()
            opened += 1
            opened_for_dataset += 1
            datasets_seen.add(str(entry.get("slug")))
            label_kinds.update(sample.get("label_kinds", []))

    return {
        "opened_samples": opened,
        "dataset_count": len(datasets_seen),
        "datasets": sorted(datasets_seen),
        "label_kinds_seen": sorted(label_kinds),
        "missing_paths": missing_paths,
    }


class ProtectedEvalHashCollisionError(ValueError):
    """Raised when a Roboflow pretrain sample collides with a protected eval hash."""


class RoboflowCorpusIntegrityError(ValueError):
    """Raised when index-backed corpus construction finds missing/corrupt source images."""


@dataclass(frozen=True)
class RoboflowBallPretrainRecord:
    sample: Mapping[str, Any]
    label: Mapping[str, Any]
    window_samples: tuple[Mapping[str, Any], ...]
    temporal_sample_kind: str
    wbce_weight: int


class RoboflowBallPretrainDataset:
    """Torch Dataset over the index-only aggregated Roboflow ball corpus.

    The dataset reads images from their original Roboflow download locations.
    It never copies frames into a training cache. Public labels keep
    ``visibility_level=None`` and use the explicit unknown-visibility WBCE
    weight documented by ``PUBLIC_UNKNOWN_VISIBILITY_POLICY``.
    """

    def __init__(
        self,
        corpus_index_path: str | Path,
        *,
        split_role: str,
        image_size: tuple[int, int] = DEFAULT_BALL_PRETRAIN_IMAGE_SIZE,
        frames_in: int = DEFAULT_BALL_PRETRAIN_FRAMES_IN,
        heatmap_radius_px: float = DEFAULT_BALL_PRETRAIN_HEATMAP_RADIUS_PX,
        core_to_aux_ratio: int | None = DEFAULT_CORE_TO_AUX_RATIO,
        seed: int = 0,
        max_samples: int | None = None,
        include_aux_in_internal_val: bool = False,
        protected_eval_hashes: Mapping[str, Sequence[int | str]] | None = None,
        eval_root: str | Path = "eval_clips/ball",
        eval_sample_every_s: float = DEFAULT_EVAL_SAMPLE_EVERY_S,
        expected_protected_eval_hash_count: int | None = DEFAULT_PROTECTED_EVAL_HASH_COUNT,
        collision_hamming_threshold: int | None = None,
        skip_list_path: str | Path | None = None,
        skip_policy: str = "fail",
        image_path_rewrites: Mapping[str, str] | Sequence[str] | None = None,
    ) -> None:
        if split_role not in {"train", "internal_val"}:
            raise ValueError("split_role must be train or internal_val")
        if frames_in <= 0 or frames_in % 2 == 0:
            raise ValueError("frames_in must be a positive odd integer")
        if image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError("image_size must contain positive width,height")
        if heatmap_radius_px <= 0.0:
            raise ValueError("heatmap_radius_px must be positive")
        if skip_policy not in {"fail", "skip"}:
            raise ValueError("skip_policy must be fail or skip")
        if core_to_aux_ratio is not None and core_to_aux_ratio < 0:
            raise ValueError("core_to_aux_ratio must be >= 0")

        self._torch = _require_torch()
        self._np = _require_numpy()
        self.corpus_index_path = Path(corpus_index_path)
        self.split_role = split_role
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.frames_in = int(frames_in)
        self.heatmap_radius_px = float(heatmap_radius_px)
        self.skip_list_path = Path(skip_list_path) if skip_list_path is not None else None
        self.image_path_rewrites = _normalize_image_path_rewrites(image_path_rewrites)

        payload = _read_json(self.corpus_index_path)
        samples = list(payload.get("samples", []))
        threshold = (
            int(collision_hamming_threshold)
            if collision_hamming_threshold is not None
            else int(payload.get("hash", {}).get("collision_hamming_threshold", DEFAULT_DEDUP_THRESHOLD))
        )
        if protected_eval_hashes is None:
            eval_hash_count, eval_hash_source = _protected_eval_hash_summary_from_corpus_card(self.corpus_index_path)
            eval_hashes = None
        else:
            eval_hashes, eval_hash_source = _protected_eval_hashes(
                protected_eval_hashes=protected_eval_hashes,
                eval_root=Path(eval_root),
                eval_sample_every_s=eval_sample_every_s,
            )
            eval_hash_count = sum(len(values) for values in eval_hashes.values())
        if expected_protected_eval_hash_count is not None and eval_hash_count != expected_protected_eval_hash_count:
            raise ProtectedEvalHashCollisionError(
                "protected eval hash count mismatch: "
                f"expected {expected_protected_eval_hash_count}, got {eval_hash_count} from {eval_hash_source}"
            )

        eligible = [
            sample
            for sample in samples
            if _sample_has_visible_ball(sample)
            and _sample_split_matches_role(sample, split_role=split_role)
            and _sample_bucket_allowed(
                sample,
                split_role=split_role,
                include_aux_in_internal_val=include_aux_in_internal_val,
            )
        ]
        if eval_hashes is None:
            assert_no_index_marked_protected_eval_collisions(eligible, eval_hash_count=eval_hash_count)
        else:
            assert_no_protected_eval_hash_collisions(
                eligible,
                eval_hashes=eval_hashes,
                threshold=threshold,
            )
        selected = _apply_ball_pretrain_mixing(
            eligible,
            split_role=split_role,
            core_to_aux_ratio=core_to_aux_ratio,
            seed=seed,
            max_samples=max_samples,
        )
        sequence_lookup = _sequence_lookup(eligible)
        records = [
            _record_for_sample(
                sample,
                sequence_lookup=sequence_lookup,
                frames_in=self.frames_in,
            )
            for sample in selected
        ]

        skip_records = _find_unreadable_record_images(records, image_path_rewrites=self.image_path_rewrites)
        _write_skip_list(
            self.skip_list_path,
            skip_records=skip_records,
            split_role=split_role,
            corpus_index_path=self.corpus_index_path,
            skip_policy=skip_policy,
        )
        if skip_records and skip_policy == "fail":
            raise RoboflowCorpusIntegrityError(
                f"Roboflow pretrain dataset found {len(skip_records)} unreadable image(s); "
                f"skip list: {self.skip_list_path}"
            )
        if skip_records:
            skipped_ids = {str(item["sample_id"]) for item in skip_records}
            records = [record for record in records if str(record.sample["sample_id"]) not in skipped_ids]

        self.records = tuple(records)
        temporal_counts = Counter(record.temporal_sample_kind for record in self.records)
        bucket_counts = Counter(str(record.sample.get("bucket")) for record in self.records)
        split_counts = Counter(str(record.sample.get("split")) for record in self.records)
        self.summary = {
            "schema_version": 1,
            "artifact_type": "racketsport_roboflow_ball_pretrain_dataset_summary",
            "corpus_index_path": str(self.corpus_index_path),
            "index_policy": payload.get("index_policy", "index_based_original_image_paths_no_image_copies"),
            "image_path_rewrites": self.image_path_rewrites,
            "split_role": split_role,
            "source_splits": sorted(split_counts),
            "selected_sample_count": len(self.records),
            "bucket_counts": dict(sorted(bucket_counts.items())),
            "source_split_counts": dict(sorted(split_counts.items())),
            "temporal_sample_counts": dict(sorted(temporal_counts.items())),
            "image_size": list(self.image_size),
            "frames_in": self.frames_in,
            "heatmap_radius_px": self.heatmap_radius_px,
            "mixing": {
                "core_to_aux_ratio": core_to_aux_ratio,
                "include_aux_in_internal_val": include_aux_in_internal_val,
                "seed": seed,
                "max_samples": max_samples,
            },
            "visibility_policy": {
                "visibility_level": None,
                "unknown_visibility_wbce_weight": PUBLIC_UNKNOWN_VISIBILITY_WBCE_WEIGHT,
                "policy": PUBLIC_UNKNOWN_VISIBILITY_POLICY,
            },
            "protected_eval_hash_check": {
                "hash_count": eval_hash_count,
                "hash_source": eval_hash_source,
                "collision_hamming_threshold": threshold,
                "collision_count": 0,
            },
            "skip_list_path": str(self.skip_list_path) if self.skip_list_path is not None else None,
            "skip_count": len(skip_records),
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample = record.sample
        image_path = _resolve_sample_image_path(sample, self.image_path_rewrites)
        width = int(sample.get("width") or 0)
        height = int(sample.get("height") or 0)
        if width <= 0 or height <= 0:
            with Image.open(image_path) as image:
                width, height = image.size
        target_w, target_h = self.image_size
        x, y = _label_xy(record.label)
        scaled_x = float(x) * float(target_w) / float(width)
        scaled_y = float(y) * float(target_h) / float(height)
        frames = [
            _image_tensor(_resolve_sample_image_path(window_sample, self.image_path_rewrites), image_size=self.image_size)
            for window_sample in record.window_samples
        ]
        torch = self._torch
        input_tensor = torch.cat(frames, dim=0)
        target = _gaussian_heatmap(
            scaled_x,
            scaled_y,
            width=target_w,
            height=target_h,
            radius=self.heatmap_radius_px,
            torch=torch,
        )
        return {
            "sample_id": str(sample["sample_id"]),
            "source_slug": str(sample.get("source_slug")),
            "bucket": str(sample.get("bucket")),
            "source_split": str(sample.get("split")),
            "image_path": str(image_path),
            "window_sample_ids": [str(window_sample["sample_id"]) for window_sample in record.window_samples],
            "temporal_sample_kind": record.temporal_sample_kind,
            "input": input_tensor,
            "target": target,
            "target_xy_px": torch.tensor([scaled_x, scaled_y], dtype=torch.float32),
            "source_xy_px": torch.tensor([float(x), float(y)], dtype=torch.float32),
            "ball_present": torch.tensor(1.0, dtype=torch.float32),
            "wbce_weight": torch.tensor(float(record.wbce_weight), dtype=torch.float32),
            "visibility_level": None,
        }


def load_protected_eval_hashes(
    *,
    eval_root: str | Path = "eval_clips/ball",
    eval_sample_every_s: float = DEFAULT_EVAL_SAMPLE_EVERY_S,
) -> tuple[dict[str, list[int]], str]:
    return _load_eval_hashes(eval_root=Path(eval_root), sample_every_s=eval_sample_every_s)


def assert_no_protected_eval_hash_collisions(
    samples: Sequence[Mapping[str, Any]],
    *,
    eval_hashes: Mapping[str, Sequence[int]],
    threshold: int = DEFAULT_DEDUP_THRESHOLD,
) -> dict[str, Any]:
    collisions: list[dict[str, Any]] = []
    for sample in samples:
        hashes = sample.get("hashes")
        if not isinstance(hashes, Mapping) or not hashes.get("dhash"):
            continue
        collisions.extend(
            _eval_collisions_for_hash(
                sample,
                dhash_value=int(str(hashes["dhash"]), 16),
                eval_hashes=eval_hashes,
                threshold=threshold,
            )
        )
    if collisions:
        first = collisions[0]
        raise ProtectedEvalHashCollisionError(
            "protected eval hash collision in Roboflow pretrain corpus: "
            f"{first['sample_id']} vs {first['eval_clip']} hamming={first['hamming_distance']}"
        )
    return {
        "hash_type": HASH_TYPE,
        "collision_hamming_threshold": threshold,
        "sample_count": len(samples),
        "eval_hash_count": sum(len(values) for values in eval_hashes.values()),
        "collision_count": 0,
    }


def assert_no_index_marked_protected_eval_collisions(
    samples: Sequence[Mapping[str, Any]],
    *,
    eval_hash_count: int,
) -> dict[str, Any]:
    collisions: list[Mapping[str, Any]] = []
    for sample in samples:
        leakage = sample.get("leakage")
        if isinstance(leakage, Mapping) and leakage.get("eval_collision"):
            collisions.append(sample)
    if collisions:
        first = collisions[0]
        raise ProtectedEvalHashCollisionError(
            "protected eval hash collision already marked in Roboflow pretrain corpus index: "
            f"{first.get('sample_id')}"
        )
    return {
        "hash_type": HASH_TYPE,
        "sample_count": len(samples),
        "eval_hash_count": int(eval_hash_count),
        "collision_count": 0,
        "source": "corpus_index_leakage_flags",
    }


def _protected_eval_hash_summary_from_corpus_card(corpus_index_path: Path) -> tuple[int, str]:
    card_path = corpus_index_path.with_name("corpus_card.json")
    if not card_path.is_file():
        raise ProtectedEvalHashCollisionError(
            f"missing corpus_card.json next to {corpus_index_path}; "
            "pass explicit protected_eval_hashes for synthetic tests or regenerate the aggregate corpus card"
        )
    card = _read_json(card_path)
    leakage_check = card.get("leakage_check")
    if not isinstance(leakage_check, Mapping):
        raise ProtectedEvalHashCollisionError(f"corpus card missing leakage_check: {card_path}")
    if int(leakage_check.get("eval_collision_count", -1)) != 0:
        raise ProtectedEvalHashCollisionError(
            f"corpus card reports protected eval collisions: {card_path}"
        )
    counts = leakage_check.get("eval_hash_counts")
    if not isinstance(counts, Mapping):
        raise ProtectedEvalHashCollisionError(f"corpus card missing eval_hash_counts: {card_path}")
    return sum(int(value) for value in counts.values()), str(card_path)


def _protected_eval_hashes(
    *,
    protected_eval_hashes: Mapping[str, Sequence[int | str]] | None,
    eval_root: Path,
    eval_sample_every_s: float,
) -> tuple[dict[str, list[int]], str]:
    if protected_eval_hashes is None:
        return load_protected_eval_hashes(eval_root=eval_root, eval_sample_every_s=eval_sample_every_s)
    normalized: dict[str, list[int]] = {}
    for clip_id, values in protected_eval_hashes.items():
        normalized[str(clip_id)] = [_parse_hash_value(value) for value in values]
    return normalized, "constructor_provided_protected_eval_hashes"


def _parse_hash_value(value: int | str) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    return int(text, 16)


def _sample_split_matches_role(sample: Mapping[str, Any], *, split_role: str) -> bool:
    split = str(sample.get("split") or "").lower()
    if split_role == "train":
        return split in TRAIN_SOURCE_SPLITS
    return split in INTERNAL_VAL_SOURCE_SPLITS


def _sample_bucket_allowed(
    sample: Mapping[str, Any],
    *,
    split_role: str,
    include_aux_in_internal_val: bool,
) -> bool:
    bucket = str(sample.get("bucket") or "")
    if bucket == BUCKET_CORE:
        return True
    if bucket == BUCKET_ADJACENT:
        return split_role == "train" or include_aux_in_internal_val
    return False


def _sample_has_visible_ball(sample: Mapping[str, Any]) -> bool:
    return _ball_label_for_sample(sample) is not None


def _ball_label_for_sample(sample: Mapping[str, Any]) -> Mapping[str, Any] | None:
    labels = sample.get("labels")
    ball_labels: object = None
    if isinstance(labels, Mapping):
        ball_labels = labels.get("ball")
    if not isinstance(ball_labels, Sequence) or isinstance(ball_labels, (str, bytes)):
        return None
    candidates = [label for label in ball_labels if isinstance(label, Mapping) and label.get("xy") is not None]
    if not candidates:
        return None
    return sorted(candidates, key=lambda label: (-_bbox_area(label), str(label.get("annotation_id"))))[0]


def _bbox_area(label: Mapping[str, Any]) -> float:
    bbox = label.get("source_bbox_xywh")
    if isinstance(bbox, Sequence) and len(bbox) == 4:
        try:
            return float(bbox[2]) * float(bbox[3])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _apply_ball_pretrain_mixing(
    samples: Sequence[Mapping[str, Any]],
    *,
    split_role: str,
    core_to_aux_ratio: int | None,
    seed: int,
    max_samples: int | None,
) -> list[Mapping[str, Any]]:
    ordered = sorted(samples, key=lambda sample: str(sample.get("sample_id")))
    if split_role == "train":
        core = [sample for sample in ordered if sample.get("bucket") == BUCKET_CORE]
        aux = [sample for sample in ordered if sample.get("bucket") == BUCKET_ADJACENT]
        if core_to_aux_ratio is None:
            mixed = [*core, *aux]
        elif core_to_aux_ratio == 0:
            mixed = core
        else:
            max_aux = math.ceil(len(core) / float(core_to_aux_ratio)) if core else len(aux)
            mixed = [*core, *aux[:max_aux]]
    else:
        mixed = ordered
    if max_samples is not None:
        max_count = int(max_samples)
        if max_count < 0:
            raise ValueError("max_samples must be non-negative")
        if len(mixed) > max_count:
            rng = random.Random(seed)
            mixed = sorted(rng.sample(mixed, max_count), key=lambda sample: str(sample.get("sample_id")))
    return mixed


def _sequence_lookup(samples: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        temporal = sample.get("temporal")
        if not isinstance(temporal, Mapping):
            continue
        if temporal.get("kind") != "temporal_sequence" or not temporal.get("sequence_id"):
            continue
        grouped[str(temporal["sequence_id"])].append(sample)
    return {
        sequence_id: sorted(
            values,
            key=lambda sample: (
                _temporal_order(sample),
                str(sample.get("sample_id")),
            ),
        )
        for sequence_id, values in grouped.items()
    }


def _temporal_order(sample: Mapping[str, Any]) -> int:
    temporal = sample.get("temporal")
    if isinstance(temporal, Mapping):
        value = temporal.get("sequence_order")
        if value is None:
            value = temporal.get("frame_number")
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0
    return 0


def _record_for_sample(
    sample: Mapping[str, Any],
    *,
    sequence_lookup: Mapping[str, Sequence[Mapping[str, Any]]],
    frames_in: int,
) -> RoboflowBallPretrainRecord:
    label = _ball_label_for_sample(sample)
    if label is None:
        raise ValueError(f"sample has no ball label: {sample.get('sample_id')}")
    temporal = sample.get("temporal")
    temporal_sample_kind = "still_aux_repeated"
    window_samples: list[Mapping[str, Any]]
    if isinstance(temporal, Mapping) and temporal.get("kind") == "temporal_sequence" and temporal.get("sequence_id"):
        group = list(sequence_lookup.get(str(temporal["sequence_id"]), []))
        position = next(
            (idx for idx, candidate in enumerate(group) if candidate.get("sample_id") == sample.get("sample_id")),
            None,
        )
        if position is not None and group:
            radius = frames_in // 2
            window_samples = []
            for offset in range(-radius, radius + 1):
                group_index = min(len(group) - 1, max(0, position + offset))
                window_samples.append(group[group_index])
            temporal_sample_kind = "sequence_window"
        else:
            window_samples = [sample for _ in range(frames_in)]
    else:
        window_samples = [sample for _ in range(frames_in)]
    return RoboflowBallPretrainRecord(
        sample=sample,
        label=label,
        window_samples=tuple(window_samples),
        temporal_sample_kind=temporal_sample_kind,
        wbce_weight=PUBLIC_UNKNOWN_VISIBILITY_WBCE_WEIGHT,
    )


def _find_unreadable_record_images(
    records: Sequence[RoboflowBallPretrainRecord],
    *,
    image_path_rewrites: Mapping[str, str],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        for window_sample in record.window_samples:
            sample_id = str(record.sample.get("sample_id"))
            image_path = _resolve_sample_image_path(window_sample, image_path_rewrites)
            key = (sample_id, str(image_path))
            if key in seen:
                continue
            seen.add(key)
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                issues.append(
                    {
                        "sample_id": sample_id,
                        "window_sample_id": str(window_sample.get("sample_id")),
                        "image_path": str(image_path),
                        "reason": "image_missing_or_unreadable",
                        "error": str(exc),
                    }
                )
    return issues


def _normalize_image_path_rewrites(image_path_rewrites: Mapping[str, str] | Sequence[str] | None) -> dict[str, str]:
    if image_path_rewrites is None:
        return {}
    if isinstance(image_path_rewrites, Mapping):
        items = image_path_rewrites.items()
    else:
        parsed: list[tuple[str, str]] = []
        for item in image_path_rewrites:
            text = str(item)
            if "=" not in text:
                raise ValueError("image_path_rewrites entries must be OLD_PREFIX=NEW_PREFIX")
            old_prefix, new_prefix = text.split("=", 1)
            parsed.append((old_prefix, new_prefix))
        items = parsed
    normalized: dict[str, str] = {}
    for old_prefix, new_prefix in items:
        old = str(old_prefix).rstrip("/")
        new = str(new_prefix).rstrip("/")
        if not old or not new:
            raise ValueError("image_path_rewrites prefixes must be non-empty")
        normalized[old] = new
    return dict(sorted(normalized.items(), key=lambda item: len(item[0]), reverse=True))


def _resolve_sample_image_path(sample: Mapping[str, Any], image_path_rewrites: Mapping[str, str]) -> Path:
    raw = str(sample.get("image_path") or "")
    if not raw:
        raise RoboflowCorpusIntegrityError(f"sample missing image_path: {sample.get('sample_id')}")
    for old_prefix, new_prefix in image_path_rewrites.items():
        if raw == old_prefix:
            return Path(new_prefix)
        if raw.startswith(old_prefix + "/"):
            return Path(new_prefix + raw[len(old_prefix) :])
    return Path(raw)


def _write_skip_list(
    path: Path | None,
    *,
    skip_records: Sequence[Mapping[str, Any]],
    split_role: str,
    corpus_index_path: Path,
    skip_policy: str,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_roboflow_ball_pretrain_skip_list",
        "corpus_index_path": str(corpus_index_path),
        "split_role": split_role,
        "skip_policy": skip_policy,
        "skip_count": len(skip_records),
        "skips": list(skip_records),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _label_xy(label: Mapping[str, Any]) -> tuple[float, float]:
    xy = label.get("xy")
    if not isinstance(xy, Sequence) or len(xy) != 2:
        raise ValueError(f"ball label missing xy: {label}")
    return float(xy[0]), float(xy[1])


def _image_tensor(path: Path, *, image_size: tuple[int, int]) -> Any:
    torch = _require_torch()
    np = _require_numpy()
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _gaussian_heatmap(
    x: float,
    y: float,
    *,
    width: int,
    height: int,
    radius: float,
    torch: Any,
) -> Any:
    xx = torch.arange(width, dtype=torch.float32).view(1, width)
    yy = torch.arange(height, dtype=torch.float32).view(height, 1)
    heatmap = torch.exp(-((xx - float(x)) ** 2 + (yy - float(y)) ** 2) / (2.0 * float(radius) ** 2))
    return heatmap.unsqueeze(0).clamp(0.0, 1.0)


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for RoboflowBallPretrainDataset; use pytest.importorskip('torch') in tests") from exc
    return torch


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required for RoboflowBallPretrainDataset") from exc
    return np


def initial_bucket_for_entry(entry: Mapping[str, Any]) -> tuple[str, str]:
    slug = str(entry.get("slug") or "")
    if entry.get("status") != "downloaded":
        if entry.get("version") is None:
            return BUCKET_DEAD, "no_exported_version_or_dead_download"
        return BUCKET_DEAD, f"download_status_{entry.get('status')}"
    if slug in ADJACENT_SPORT_SLUGS:
        return BUCKET_ADJACENT, "manager_curated_adjacent_sport_or_contaminated_source"
    return BUCKET_CORE, "downloaded_pickleball_related_source"


def taxonomy_for_category(category_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", category_name.lower()).strip()
    tokens = set(normalized.split())
    if "paddle" in tokens or "racket" in tokens or "racquet" in tokens:
        return "paddle"
    if "person" in tokens or "player" in tokens or re.fullmatch(r"team\d?", normalized or ""):
        return "person"
    if (
        "court" in tokens
        or "kitchen" in tokens
        or "net" in tokens
        or "line" in tokens
        or "surface" in tokens
        or normalized.startswith("pb")
    ):
        return "court"
    if "ball" in tokens or "pickleball" in tokens or normalized in {"pickle", "pickle ball"}:
        return "ball"
    return "other"


def image_dhash(path: str | Path, *, hash_size: int = 8) -> int:
    width = hash_size + 1
    height = hash_size
    with Image.open(path) as image:
        gray = ImageOps.grayscale(image).resize((width, height), Image.Resampling.LANCZOS)
        pixels = gray.tobytes()
    value = 0
    bit = 0
    for row in range(height):
        offset = row * width
        for col in range(hash_size):
            if pixels[offset + col] > pixels[offset + col + 1]:
                value |= 1 << bit
            bit += 1
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_annotation(annotation: Mapping[str, Any], *, category_name: str, taxonomy: str) -> dict[str, Any] | None:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, Sequence) or len(bbox) != 4:
        return None
    bbox_xywh = [round(float(value), 6) for value in bbox]
    annotation_id = int(annotation["id"]) if "id" in annotation else None
    if taxonomy == "ball":
        x, y, width, height = bbox_xywh
        return {
            "t": 0.0,
            "xy": [round(x + width / 2.0, 6), round(y + height / 2.0, 6)],
            "conf": 1.0,
            "visible": True,
            "source_bbox_xywh": bbox_xywh,
            "original_category": category_name,
            "annotation_id": annotation_id,
        }
    return {
        "bbox_xywh": bbox_xywh,
        "original_category": category_name,
        "annotation_id": annotation_id,
        "iscrowd": int(annotation.get("iscrowd", 0) or 0),
        "has_segmentation": bool(annotation.get("segmentation")),
    }


def _attach_image_hashes(indices: Sequence[dict[str, Any]]) -> None:
    for index in indices:
        hash_errors = []
        for sample in index.get("samples", []):
            path = Path(sample["image_path"])
            try:
                sample["hashes"] = {
                    "dhash": f"{image_dhash(path):016x}",
                    "sha256": file_sha256(path),
                }
            except Exception as exc:
                sample["hashes"] = None
                sample["index_status"] = "excluded"
                sample["exclude_reason"] = "image_hash_failed"
                hash_errors.append({"sample_id": sample["sample_id"], "image_path": str(path), "error": str(exc)})
        index["hash_errors"] = hash_errors
        index["statistics"]["hash_error_count"] = len(hash_errors)
        index["statistics"]["hashable_sample_count"] = sum(1 for sample in index.get("samples", []) if sample.get("hashes"))


def _mark_dataset_duplicates(indices: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    core_indices = [
        index
        for index in indices
        if index.get("bucket") == BUCKET_CORE and index["statistics"].get("hashable_sample_count", 0) > 0
    ]
    ordered = sorted(
        core_indices,
        key=lambda item: (
            item["statistics"].get("hashable_sample_count", 0),
            item["statistics"].get("sample_count", 0),
            item["source"]["slug"],
        ),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for index in ordered:
        sha_set = _dataset_sha_set(index)
        best: tuple[int, dict[str, Any]] | None = None
        for kept_index in kept:
            overlap = len(sha_set & _dataset_sha_set(kept_index))
            if best is None or overlap > best[0]:
                best = (overlap, kept_index)
        if best is not None and sha_set:
            overlap, kept_index = best
            overlap_ratio = overlap / len(sha_set)
            if overlap >= 1 and overlap_ratio >= 0.90:
                index["bucket"] = BUCKET_DUPLICATE
                index["bucket_reason"] = f"exact_image_sha256_overlap_with_{kept_index['source']['slug']}"
                index["duplicate_of_source_slug"] = kept_index["source"]["slug"]
                mappings.append(
                    {
                        "duplicate_slug": index["source"]["slug"],
                        "kept_slug": kept_index["source"]["slug"],
                        "overlap_count": overlap,
                        "overlap_ratio_of_duplicate": round(overlap_ratio, 6),
                        "reason": "exact_image_sha256_overlap",
                    }
                )
                continue
        kept.append(index)
    return sorted(mappings, key=lambda item: item["duplicate_slug"])


def _mark_sample_statuses(
    indices: Sequence[dict[str, Any]],
    *,
    eval_hashes: Mapping[str, Sequence[int]],
    dedup_threshold: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    tree = HammingBKTree()
    corpus_samples: list[dict[str, Any]] = []
    subset_samples: dict[str, list[dict[str, Any]]] = {kind: [] for kind in ["ball", "person", "court", "paddle", "other"]}
    duplicate_records: list[dict[str, Any]] = []
    leakage_collisions: list[dict[str, Any]] = []
    considered = 0

    for index in _sample_processing_order(indices):
        bucket = index["bucket"]
        for sample in index.get("samples", []):
            hashes = sample.get("hashes")
            sample["bucket"] = bucket
            sample["dedup"] = {"keep": False, "duplicate_of_sample_id": None, "hamming_distance": None}
            sample["leakage"] = {"eval_collision": False, "collisions": []}
            if bucket in {BUCKET_DEAD, BUCKET_DUPLICATE}:
                sample["index_status"] = "excluded"
                sample["exclude_reason"] = "source_bucket_" + bucket
                continue
            if not hashes:
                sample.setdefault("index_status", "excluded")
                sample.setdefault("exclude_reason", "image_hash_failed")
                continue
            considered += 1
            dhash_value = int(hashes["dhash"], 16)
            eval_collisions = _eval_collisions_for_hash(
                sample,
                dhash_value=dhash_value,
                eval_hashes=eval_hashes,
                threshold=dedup_threshold,
            )
            if eval_collisions:
                sample["leakage"] = {"eval_collision": True, "collisions": eval_collisions}
                sample["index_status"] = "excluded"
                sample["exclude_reason"] = "protected_eval_dhash_collision"
                leakage_collisions.extend(eval_collisions)
                continue
            matches = tree.query(dhash_value, dedup_threshold)
            if matches:
                distance, payload = sorted(matches, key=lambda item: (item[0], item[1]["sample_id"]))[0]
                duplicate = {
                    "sample_id": sample["sample_id"],
                    "source_slug": sample["source_slug"],
                    "duplicate_of_sample_id": payload["sample_id"],
                    "duplicate_of_source_slug": payload["source_slug"],
                    "hamming_distance": distance,
                    "hash": hashes["dhash"],
                }
                sample["dedup"] = {
                    "keep": False,
                    "duplicate_of_sample_id": payload["sample_id"],
                    "hamming_distance": distance,
                }
                sample["index_status"] = "excluded"
                sample["exclude_reason"] = "perceptual_duplicate"
                duplicate_records.append(duplicate)
                continue
            sample["dedup"] = {"keep": True, "duplicate_of_sample_id": None, "hamming_distance": None}
            sample["index_status"] = "kept"
            tree.add(dhash_value, {"sample_id": sample["sample_id"], "source_slug": sample["source_slug"]})
            corpus_sample = _merged_sample(sample)
            corpus_samples.append(corpus_sample)
            for kind in sample.get("label_kinds", []):
                subset_samples[kind].append(_subset_sample(sample, kind=kind))

    dedup_summary = {
        "hash_type": HASH_TYPE,
        "collision_hamming_threshold": dedup_threshold,
        "considered_sample_count": considered,
        "kept_sample_count": len(corpus_samples),
        "perceptual_duplicate_sample_count": len(duplicate_records),
        "dedup_rate": round(len(duplicate_records) / considered, 6) if considered else 0.0,
        "duplicate_examples": duplicate_records[:200],
    }
    leakage_summary = {
        "eval_collision_count": len(leakage_collisions),
        "collisions": leakage_collisions[:200],
    }
    return corpus_samples, subset_samples, dedup_summary, leakage_summary


def _build_corpus_card(
    *,
    manifest: Mapping[str, Any],
    normalized_indices: Sequence[dict[str, Any]],
    parse_failures: Sequence[Mapping[str, Any]],
    duplicate_mappings: Sequence[Mapping[str, Any]],
    dedup_summary: Mapping[str, Any],
    leakage_summary: Mapping[str, Any],
    eval_hashes: Mapping[str, Sequence[int]],
    eval_hash_source: str,
    subset_paths: Mapping[str, str],
    corpus_sample_count: int,
    dedup_threshold: int,
    eval_sample_every_s: float,
) -> dict[str, Any]:
    per_source = [_per_source_counts(index) for index in sorted(normalized_indices, key=lambda item: item["source"]["slug"])]
    temporal_counts = Counter()
    sample_bucket_counts = Counter()
    taxonomy_mapping: dict[str, str] = {}
    skipped_no_version = []
    skipped_dead_link = []
    skipped_failed_other = []
    for index in normalized_indices:
        sample_bucket_counts[index["bucket"]] += index["statistics"].get("sample_count", 0)
        temporal_counts.update(index["statistics"].get("temporal_counts", {}))
        taxonomy_mapping.update(index.get("taxonomy_mapping", {}))
        source = index["source"]
        if source.get("status") != "downloaded" and source.get("version") is None:
            error = str(source.get("error") or "").lower()
            if "404" in error or "not found" in error:
                skipped_dead_link.append(source["slug"])
            elif "no published" in error or "never exported" in error:
                skipped_no_version.append(source["slug"])
            else:
                skipped_failed_other.append(source["slug"])
    downloaded_count = sum(1 for index in normalized_indices if index["source"].get("status") == "downloaded")
    parse_failure_rate = len(parse_failures) / downloaded_count if downloaded_count else 0.0
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_roboflow_public_pretrain_corpus_card",
        "generated_utc": _utc_now(),
        "source_manifest_counts": manifest.get("counts", {}),
        "bucket_policy": {
            "buckets": [BUCKET_CORE, BUCKET_ADJACENT, BUCKET_DUPLICATE, BUCKET_DEAD],
            "adjacent_sport_aux_slugs": sorted(ADJACENT_SPORT_SLUGS),
            "ambiguous_spot_check": AMBIGUOUS_SPOT_CHECK,
            "visibility_policy": "public labels keep visibility_level absent; legacy visible=True only",
            "image_copy_policy": "indexes reference original image paths; images are not copied into aggregated outputs",
        },
        "bucket_counts_by_source": _bucket_counts_by_source(normalized_indices),
        "bucket_counts_by_sample": dict(sorted(sample_bucket_counts.items())),
        "per_source_counts": per_source,
        "corpus_index_sample_count": corpus_sample_count,
        "subset_indexes": dict(sorted(subset_paths.items())),
        "dedup": dict(dedup_summary),
        "fork_duplicate_mappings": list(duplicate_mappings),
        "temporal_split": {
            "counts": dict(sorted(temporal_counts.items())),
            "temporal_sequence_fraction": round(
                temporal_counts.get("temporal_sequence", 0) / max(1, sum(temporal_counts.values())),
                6,
            ),
        },
        "leakage_check": {
            "hash_type": HASH_TYPE,
            "eval_sample_every_s": eval_sample_every_s,
            "collision_hamming_threshold": dedup_threshold,
            "eval_hash_source": eval_hash_source,
            "eval_hash_counts": {key: len(value) for key, value in sorted(eval_hashes.items())},
            **dict(leakage_summary),
        },
        "class_taxonomy_mapping": dict(sorted(taxonomy_mapping.items())),
        "skipped_no_version": sorted(skipped_no_version),
        "skipped_dead_link": sorted(skipped_dead_link),
        "skipped_failed_other": sorted(skipped_failed_other),
        "skipped_no_version_or_dead": sorted([*skipped_no_version, *skipped_dead_link, *skipped_failed_other]),
        "parse_failures": list(parse_failures),
        "kill_criteria": {
            "downloaded_source_count": downloaded_count,
            "parse_failure_count": len(parse_failures),
            "parse_failure_rate": round(parse_failure_rate, 6),
            "threshold": 0.20,
            "tripped": parse_failure_rate > 0.20,
        },
    }


def _format_corpus_card_md(card: Mapping[str, Any]) -> str:
    lines = [
        "# Roboflow Public Pretrain Corpus Card",
        "",
        f"- Generated UTC: `{card['generated_utc']}`",
        f"- Corpus samples kept: `{card['corpus_index_sample_count']}`",
        f"- Bucket counts by source: `{json.dumps(card['bucket_counts_by_source'], sort_keys=True)}`",
        f"- Dedup rate: `{card['dedup']['dedup_rate']}` ({card['dedup']['perceptual_duplicate_sample_count']} perceptual duplicates / {card['dedup']['considered_sample_count']} considered)",
        f"- Temporal split: `{json.dumps(card['temporal_split']['counts'], sort_keys=True)}`",
        f"- Leakage check vs protected eval clips: `{card['leakage_check']['eval_collision_count']}` dHash collisions",
        f"- Index policy: `{card['bucket_policy']['image_copy_policy']}`",
        "",
        "## Buckets",
        "",
        "| bucket | source count | sample count |",
        "|---|---:|---:|",
    ]
    for bucket in [BUCKET_CORE, BUCKET_ADJACENT, BUCKET_DUPLICATE, BUCKET_DEAD]:
        lines.append(
            f"| {bucket} | {card['bucket_counts_by_source'].get(bucket, 0)} | {card['bucket_counts_by_sample'].get(bucket, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Fork Duplicate Mappings",
            "",
            "| duplicate source | kept source | overlap | reason |",
            "|---|---|---:|---|",
        ]
    )
    if card["fork_duplicate_mappings"]:
        for item in card["fork_duplicate_mappings"]:
            lines.append(
                f"| {item['duplicate_slug']} | {item['kept_slug']} | {item['overlap_count']} | {item['reason']} |"
            )
    else:
        lines.append("| none | none | 0 | no dataset-level duplicate found |")
    lines.extend(
        [
            "",
            "## Class Taxonomy Mapping",
            "",
            "| original class | normalized taxonomy |",
            "|---|---|",
        ]
    )
    for original, taxonomy in card["class_taxonomy_mapping"].items():
        lines.append(f"| {original} | {taxonomy} |")
    lines.extend(
        [
            "",
            "## Per Source Counts",
            "",
            "| source | bucket | samples | kept | duplicates | leakage | temporal seq | isolated |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source in card["per_source_counts"]:
        temporal = source["temporal_counts"]
        lines.append(
            "| {slug} | {bucket} | {samples} | {kept} | {dupes} | {leaks} | {seq} | {still} |".format(
                slug=source["slug"],
                bucket=source["bucket"],
                samples=source["sample_count"],
                kept=source["kept_sample_count"],
                dupes=source["perceptual_duplicate_sample_count"],
                leaks=source["leakage_collision_sample_count"],
                seq=temporal.get("temporal_sequence", 0),
                still=temporal.get("isolated_still", 0),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _p10_gate_passed(card: Mapping[str, Any]) -> bool:
    return (
        card["leakage_check"]["eval_collision_count"] == 0
        and not card["kill_criteria"]["tripped"]
        and card["bucket_counts_by_source"].get(BUCKET_ADJACENT, 0) >= 3
    )


def _source_summary(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "slug": entry.get("slug"),
        "url": entry.get("url"),
        "workspace": entry.get("workspace"),
        "project": entry.get("project"),
        "status": entry.get("status"),
        "version": entry.get("version"),
        "content_category_guess": entry.get("content_category_guess"),
        "license_as_recorded": entry.get("license_as_recorded") or entry.get("license"),
        "classes": entry.get("classes", []),
        "image_count_downloaded": int(entry.get("image_count_downloaded") or 0),
        "image_count_claimed": entry.get("image_count_claimed"),
        "local_path": entry.get("local_path"),
        "temporal_hint": entry.get("temporal_hint"),
        "temporal_hint_detail": entry.get("temporal_hint_detail"),
        "note": entry.get("note"),
        "error": entry.get("error"),
    }


def _empty_dataset_index(
    entry: Mapping[str, Any],
    *,
    bucket: str,
    bucket_reason: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_roboflow_dataset_index",
        "source": _source_summary(entry),
        "bucket": bucket,
        "bucket_reason": bucket_reason,
        "statistics": {
            "sample_count": 0,
            "annotation_count": 0,
            "missing_path_count": 0,
            "label_counts_by_taxonomy": {},
            "original_class_counts": {},
            "temporal_counts": {},
            "hash_error_count": 0,
            "hashable_sample_count": 0,
        },
        "taxonomy_mapping": {},
        "missing_paths": [],
        "parse_warnings": [reason],
        "hash_errors": [],
        "samples": [],
    }


def _per_source_counts(index: Mapping[str, Any]) -> dict[str, Any]:
    samples = index.get("samples", [])
    return {
        "slug": index["source"]["slug"],
        "status": index["source"].get("status"),
        "version": index["source"].get("version"),
        "bucket": index["bucket"],
        "bucket_reason": index.get("bucket_reason"),
        "sample_count": index["statistics"].get("sample_count", 0),
        "kept_sample_count": sum(1 for sample in samples if sample.get("index_status") == "kept"),
        "perceptual_duplicate_sample_count": sum(
            1 for sample in samples if sample.get("exclude_reason") == "perceptual_duplicate"
        ),
        "leakage_collision_sample_count": sum(
            1 for sample in samples if sample.get("exclude_reason") == "protected_eval_dhash_collision"
        ),
        "label_counts_by_taxonomy": index["statistics"].get("label_counts_by_taxonomy", {}),
        "original_class_counts": index["statistics"].get("original_class_counts", {}),
        "temporal_counts": index["statistics"].get("temporal_counts", {}),
        "missing_path_count": index["statistics"].get("missing_path_count", 0),
        "hash_error_count": index["statistics"].get("hash_error_count", 0),
    }


def _merged_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "source_slug": sample["source_slug"],
        "bucket": sample["bucket"],
        "split": sample["split"],
        "image_path": sample["image_path"],
        "width": sample["width"],
        "height": sample["height"],
        "labels": sample["labels"],
        "label_kinds": sample["label_kinds"],
        "temporal": sample["temporal"],
        "hashes": sample["hashes"],
    }


def _subset_sample(sample: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "source_slug": sample["source_slug"],
        "bucket": sample["bucket"],
        "image_path": sample["image_path"],
        "width": sample["width"],
        "height": sample["height"],
        "temporal": sample["temporal"],
        "labels": sample["labels"][kind],
    }


def _load_eval_hashes(*, eval_root: Path, sample_every_s: float) -> tuple[dict[str, list[int]], str]:
    try:
        from threed.racketsport.online_harvest_ingest import PROTECTED_EVAL_CLIPS, perceptual_hash_video

        source = "threed.racketsport.online_harvest_ingest.perceptual_hash_video"
        hashes: dict[str, list[int]] = {}
        for clip in PROTECTED_EVAL_CLIPS:
            path = eval_root / clip.clip_id / "source.mp4"
            if path.is_file():
                hashes[clip.clip_id] = perceptual_hash_video(path, sample_every_s=sample_every_s)
        return hashes, source
    except Exception as exc:
        return {}, f"eval_hash_unavailable: {exc}"


def _eval_collisions_for_hash(
    sample: Mapping[str, Any],
    *,
    dhash_value: int,
    eval_hashes: Mapping[str, Sequence[int]],
    threshold: int,
) -> list[dict[str, Any]]:
    collisions: list[dict[str, Any]] = []
    for clip_id, values in sorted(eval_hashes.items()):
        for eval_hash in values:
            distance = int(dhash_value ^ int(eval_hash)).bit_count()
            if distance <= threshold:
                collisions.append(
                    {
                        "sample_id": sample["sample_id"],
                        "source_slug": sample["source_slug"],
                        "image_hash": f"{dhash_value:016x}",
                        "eval_clip": clip_id,
                        "eval_hash": f"{int(eval_hash):016x}",
                        "hamming_distance": distance,
                    }
                )
    return collisions


def _dataset_sha_set(index: Mapping[str, Any]) -> set[str]:
    return {sample["hashes"]["sha256"] for sample in index.get("samples", []) if sample.get("hashes")}


def _sample_processing_order(indices: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    bucket_rank = {BUCKET_CORE: 0, BUCKET_ADJACENT: 1, BUCKET_DUPLICATE: 2, BUCKET_DEAD: 3}
    return sorted(
        indices,
        key=lambda index: (
            bucket_rank.get(index["bucket"], 99),
            -int(index["statistics"].get("hashable_sample_count", 0)),
            index["source"]["slug"],
        ),
    )


def _bucket_counts_by_source(indices: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(index["bucket"] for index in indices)
    return {bucket: counts.get(bucket, 0) for bucket in [BUCKET_ADJACENT, BUCKET_CORE, BUCKET_DEAD, BUCKET_DUPLICATE] if counts.get(bucket, 0)}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if compact:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        else:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")


def _resolve_path(value: Any, *, repo_root: Path) -> Path:
    path = Path(str(value)) if value else Path()
    if path.is_absolute():
        return path
    return repo_root / path


def _original_image_name(image: Mapping[str, Any]) -> str:
    extra = image.get("extra")
    if isinstance(extra, Mapping) and extra.get("name"):
        return str(extra["name"])
    file_name = str(image.get("file_name", ""))
    stem = Path(file_name).stem
    stem = re.sub(r"_(?:jpg|jpeg|png)\.rf\.[0-9A-Za-z]+$", "", stem)
    return stem + Path(file_name).suffix


def _filename_sequence_parts(file_name: str) -> tuple[str | None, int | None]:
    stem = Path(file_name).stem
    match = re.match(r"^(.*?)(\d+)$", stem)
    if match is None:
        return None, None
    prefix = match.group(1).rstrip("_-. ")
    return (prefix or "(numeric)", int(match.group(2)))


def _safe_slug(slug: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", slug.strip()).strip("_") or "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class _BKNode:
    value: int
    payload: dict[str, Any]
    children: dict[int, "_BKNode"]


class HammingBKTree:
    def __init__(self) -> None:
        self.root: _BKNode | None = None

    def add(self, value: int, payload: dict[str, Any]) -> None:
        if self.root is None:
            self.root = _BKNode(value=value, payload=payload, children={})
            return
        node = self.root
        while True:
            distance = int(value ^ node.value).bit_count()
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value=value, payload=payload, children={})
                return
            node = child

    def query(self, value: int, max_distance: int) -> list[tuple[int, dict[str, Any]]]:
        if self.root is None:
            return []
        matches: list[tuple[int, dict[str, Any]]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = int(value ^ node.value).bit_count()
            if distance <= max_distance:
                matches.append((distance, node.payload))
            low = distance - max_distance
            high = distance + max_distance
            for edge_distance, child in node.children.items():
                if low <= edge_distance <= high:
                    stack.append(child)
        return matches
