from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageOps


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
