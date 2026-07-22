#!/usr/bin/env python3
"""Export the licensed, pickleball-only Roboflow PERSON index as YOLO.

This is deliberately separate from the CVAT exporter.  It consumes only the
index-backed public Roboflow corpus, fails closed on rights/bucket metadata,
and hashes every selected image against every frame under the protected root
before it creates training inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2  # type: ignore[import-not-found]
import numpy as np


EXPECTED_PERSON_INDEX_SHA256 = "4dbf5c8e7ca328b2a05743f525b0f6e4cbf3b50c5c074da8a436a2e83b358135"
REQUIRED_NC_EXCLUSION = "testing-esifc/pickle-ball-labeling-mff1d"
ALLOWED_LICENSE = "CC BY 4.0"
REQUIRED_BUCKET = "core_pickleball"
DEFAULT_SEED = 20260721
PHASH_SCALES = (16, 32, 64)
PHASH_BITS_PER_SCALE = 64
PHASH_HAMMING_THRESHOLD = 6
FAMILY_BALANCE_MAX_REPETITIONS = 4
CONTENT_SSIM_THRESHOLD = 0.90
CONTENT_ORB_MIN_INLIERS = 40
CONTENT_ORB_MIN_INLIER_RATIO = 0.25
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mov", ".mp4"}
PROTECTED_CLIP_IDS = (
    "burlington_gold_0300_low_steep_corner",
    "indoor_doubles_fwuks_0500_long_mid_baseline",
    "outdoor_webcam_iynbd_1500_long_high_baseline",
    "wolverine_mixed_0200_mid_steep_corner",
)
PROTECTED_TRANSFORMS = (
    "identity",
    "horizontal_flip",
    "center_crop_05pct",
    "center_crop_10pct",
    "center_crop_20pct",
    "letterbox_05pct",
    "letterbox_10pct",
    "letterbox_20pct",
)
ROBUSTNESS_PROBE_TRANSFORMS = PROTECTED_TRANSFORMS[1:]
REVIEW_COLUMNS = (
    "source",
    "sample_id",
    "drawn_box_count",
    "correct_person_box_count",
    "visible_on_court_person_count",
    "annotated_visible_on_court_person_count",
    "frame_complete_yes_no",
    "notes",
)
ADJUDICATED_ORIGINAL_VIDEO_FAMILIES = (
    {
        "family_id": "family:pickleball-od8al/pickleball-seg",
        "original_video_family_id": "original_footage_component:od8al_validation_r2",
        "game_session": "multiple sessions linked by source-level shared original footage",
        "channel": "pickleball-od8al",
        "members": (
            "pickle-es3fs/pickleball-video",
            "pickleball-od8al/pickleball-seg",
            "pickleball-od8al/pickleball-tsgju",
            "pickleball-od8al/pickleball-version2",
            "nigh-workspace/pickleball-player-object-detection-cc2sw",
        ),
        "evidence": (
            "review P1-SPLIT-LEAK: consecutive output_frame frames share the 2021 US Open broadcast",
            "shared Roboflow channel plus overlapping temporal filename prefixes",
            "manifest identifies version2 as a duplicate of pickleball-seg",
            "review P1-SPLIT-LEAK-R2: pickle-es3fs/yt-UF0i_EboHqA-0004 and od8al/output_frame_file2_28 share original footage (pHash<=6, SSIM 0.916809, 2307 ORB homography inliers)",
            "review P1-SPLIT-LEAK-R2: nigh-workspace/PBGAME13_mp4-0206 and od8al/output_frame_file4_844 share original footage (pHash<=6, SSIM 0.904504, 1130 ORB homography inliers)",
            "orchestrator ruling: whole-source holdout-side-wins merge; no frame-level carve-outs",
        ),
    },
)
_BYTE_POPCOUNT = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)


class ProtectedCollisionError(RuntimeError):
    """Raised when an export candidate perceptually collides with protected data."""


class CrossSplitContentLeakError(RuntimeError):
    """Raised when final train/validation/test content shares original footage."""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a guarded Roboflow PERSON subset to a source-held YOLO dataset."
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--exclude-source", action="append", default=[])
    parser.add_argument("--val-source", required=True)
    parser.add_argument("--test-source", required=True)
    parser.add_argument("--group-forks", action="store_true")
    parser.add_argument("--source-balanced", action="store_true")
    parser.add_argument("--audit-samples-per-source", type=int, required=True)
    parser.add_argument("--protected-root", type=Path, required=True)
    parser.add_argument(
        "--review-csv",
        type=Path,
        help="Completed copy of audit/review_template.csv; without it P2 data.yaml is withheld.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        summary = export_roboflow_person_yolo_dataset(
            index_path=args.index,
            bucket=str(args.bucket),
            exclude_sources=tuple(args.exclude_source),
            val_source=str(args.val_source),
            test_source=str(args.test_source),
            group_forks=bool(args.group_forks),
            source_balanced=bool(args.source_balanced),
            audit_samples_per_source=int(args.audit_samples_per_source),
            protected_root=args.protected_root,
            review_csv=args.review_csv,
            out_dir=args.out,
            expected_index_sha256=EXPECTED_PERSON_INDEX_SHA256,
        )
    except (ProtectedCollisionError, CrossSplitContentLeakError) as exc:
        print(f"Roboflow PERSON export refused: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Roboflow PERSON export failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(_compact_summary(summary), sort_keys=True))
    return 0 if summary["training_ready_gate"]["status"] == "PASS" else 2


def export_roboflow_person_yolo_dataset(
    *,
    index_path: Path,
    bucket: str,
    exclude_sources: Sequence[str],
    val_source: str,
    test_source: str,
    group_forks: bool,
    source_balanced: bool,
    audit_samples_per_source: int,
    protected_root: Path,
    out_dir: Path,
    review_csv: Path | None = None,
    expected_index_sha256: str | None = None,
    seed: int = DEFAULT_SEED,
    min_train_images: int = 5_000,
    min_train_source_groups: int = 8,
    phash_hamming_threshold: int = PHASH_HAMMING_THRESHOLD,
    family_balance_max_repetitions: int = FAMILY_BALANCE_MAX_REPETITIONS,
) -> dict[str, Any]:
    """Run the guarded export and return its acceptance-number summary."""

    index_path = Path(index_path)
    protected_root = Path(protected_root)
    review_csv = Path(review_csv) if review_csv is not None else None
    out_dir = Path(out_dir)
    if bucket != REQUIRED_BUCKET:
        raise ValueError(
            f"PERSON P1 hard-refuses bucket {bucket!r}; only {REQUIRED_BUCKET!r} is allowed"
        )
    if not index_path.is_file():
        raise FileNotFoundError(f"missing Roboflow PERSON index: {index_path}")
    if not protected_root.is_dir():
        raise FileNotFoundError(f"missing protected root: {protected_root}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory must be absent or empty: {out_dir}")
    if audit_samples_per_source <= 0:
        raise ValueError("--audit-samples-per-source must be positive")
    if not group_forks:
        raise ValueError("--group-forks is required by the PERSON P1 data policy")
    if not source_balanced:
        raise ValueError("--source-balanced is required by the PERSON P1 data policy")
    if family_balance_max_repetitions < 1:
        raise ValueError("family balance repetition cap must be positive")
    if val_source == test_source:
        raise ValueError("validation and test sources must differ")
    if not 0 <= phash_hamming_threshold <= PHASH_BITS_PER_SCALE:
        raise ValueError("pHash Hamming threshold must be between 0 and 64")

    index_sha256 = file_sha256(index_path)
    if expected_index_sha256 is not None and index_sha256 != expected_index_sha256:
        raise ValueError(
            f"PERSON index SHA-256 mismatch: expected {expected_index_sha256}, got {index_sha256}"
        )
    payload = _read_json(index_path)
    if payload.get("artifact_type") != "racketsport_roboflow_subset_index":
        raise ValueError(f"unexpected index artifact_type: {payload.get('artifact_type')}")
    if payload.get("label_kind") != "person":
        raise ValueError(f"index label_kind must be person, got {payload.get('label_kind')}")
    samples = list(payload.get("samples") or [])
    if int(payload.get("sample_count", -1)) != len(samples):
        raise ValueError("index sample_count does not match samples length")

    manifest_path, corpus_card_path = _metadata_paths(index_path)
    manifest = _read_json(manifest_path)
    corpus_card = _read_json(corpus_card_path)
    metadata = {
        str(entry.get("slug")): entry
        for entry in manifest.get("entries", [])
        if isinstance(entry, Mapping) and entry.get("slug")
    }

    forced_exclusions = {REQUIRED_NC_EXCLUSION}
    requested_exclusions = {str(source) for source in exclude_sources}
    exclusions = forced_exclusions | requested_exclusions
    eligible, exclusion_report = _filter_samples(samples, bucket=bucket, exclusions=exclusions)
    if not eligible:
        raise ValueError("no samples remain after bucket and source exclusions")

    source_names = sorted({str(sample.get("source_slug")) for sample in eligible})
    rights = _validate_rights(source_names, metadata)
    if val_source not in source_names:
        raise ValueError(f"named validation source is not eligible: {val_source}")
    if test_source not in source_names:
        raise ValueError(f"named test source is not eligible: {test_source}")

    family_for_source, fork_families, original_video_family_map = _build_fork_families(
        source_names,
        samples=eligible,
        manifest=manifest,
        corpus_card=corpus_card,
    )
    split_for_source = _split_sources_by_family(
        fork_families,
        val_source=val_source,
        test_source=test_source,
    )
    temporal_lineage_audit = _audit_cross_split_lineage(
        eligible,
        split_for_source=split_for_source,
        family_for_source=family_for_source,
    )

    export_samples, invalid_annotations, dropped_images = _sanitize_samples(eligible)
    if not export_samples:
        raise ValueError("no images retain a valid PERSON box after geometry validation")
    pre_quarantine_source_counts = _source_count_rows(
        export_samples,
        indexed_samples=eligible,
        invalid_annotations=invalid_annotations,
        split_for_source=split_for_source,
        family_for_source=family_for_source,
    )
    pre_quarantine_split_counts = _split_count_rows(pre_quarantine_source_counts)
    pre_quarantine_family_counts = _family_count_rows(
        pre_quarantine_source_counts,
        fork_families,
        family_for_source=family_for_source,
    )
    pre_quarantine_retention = _retention_snapshot(
        pre_quarantine_source_counts,
        status="PROVISIONAL_PENDING_HUMAN_REVIEW",
        min_train_images=min_train_images,
        min_train_source_groups=min_train_source_groups,
    )

    audit_selection = _select_audit_samples(
        export_samples,
        per_source=audit_samples_per_source,
        seed=seed,
    )
    audit_counts = [
        {
            "source": source,
            "available_images": sum(1 for sample in export_samples if sample.get("source_slug") == source),
            "available_original_frame_identities": len(
                {
                    _original_frame_identity(sample)
                    for sample in export_samples
                    if sample.get("source_slug") == source
                }
            ),
            "requested_samples": audit_samples_per_source,
            "staged_unique_samples": len(selected),
            "shortfall": max(0, audit_samples_per_source - len(selected)),
        }
        for source, selected in sorted(audit_selection.items())
    ]

    human_review = _ingest_human_review(review_csv, audit_selection=audit_selection)
    quarantined_sources = set(human_review["quarantined_sources"])
    if human_review["status"] == "COMPLETE":
        materialized_samples = [
            sample
            for sample in export_samples
            if str(sample["source_slug"]) not in quarantined_sources
        ]
        post_quarantine_source_counts = _source_count_rows(
            materialized_samples,
            indexed_samples=eligible,
            invalid_annotations=invalid_annotations,
            split_for_source=split_for_source,
            family_for_source=family_for_source,
        )
        post_quarantine_split_counts = _split_count_rows(post_quarantine_source_counts)
        post_quarantine_family_counts = _family_count_rows(
            post_quarantine_source_counts,
            fork_families,
            family_for_source=family_for_source,
        )
        post_quarantine_retention: dict[str, Any] | None = _retention_snapshot(
            post_quarantine_source_counts,
            status="MEASURED_POST_QUARANTINE",
            min_train_images=min_train_images,
            min_train_source_groups=min_train_source_groups,
        )
    else:
        materialized_samples = export_samples
        post_quarantine_source_counts = None
        post_quarantine_split_counts = None
        post_quarantine_family_counts = None
        post_quarantine_retention = None

    exported_hashes, exported_refs = _hash_export_candidates(export_samples)
    cross_split_content_audit = _audit_cross_split_content(
        materialized_samples,
        all_hashes=exported_hashes,
        all_hash_refs=exported_refs,
        split_for_source=split_for_source,
        family_for_source=family_for_source,
        threshold=phash_hamming_threshold,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "cross_split_content_audit.json", cross_split_content_audit)
    if cross_split_content_audit["verified_leak_count"]:
        _write_json(
            out_dir / "refusal.json",
            {
                "objective_result": "BLOCKED",
                "verdict": "CROSS_SPLIT_CONTENT_LEAK",
                "verified_leak_count": cross_split_content_audit["verified_leak_count"],
                "content_audit": str(out_dir / "cross_split_content_audit.json"),
            },
        )
        raise CrossSplitContentLeakError(
            "content-level final-split scan found "
            f"{cross_split_content_audit['verified_leak_count']} verified cross-split leak(s); "
            f"report: {out_dir / 'cross_split_content_audit.json'}"
        )
    protected_hashes, protected_refs, protected_inventory = _hash_protected_root(
        protected_root, threshold=phash_hamming_threshold
    )
    collision_report = _compare_multiscale_hashes(
        exported_hashes,
        exported_refs,
        protected_hashes,
        protected_refs,
        threshold=phash_hamming_threshold,
    )
    collision_report.update(
        {
            "hash_type": "pHash_DCT_64bit_at_16_32_64_grayscale_scales",
            "scales": list(PHASH_SCALES),
            "bits_per_scale": PHASH_BITS_PER_SCALE,
            "protected_transform_variants": list(PROTECTED_TRANSFORMS),
            "collision_rule": (
                "candidate matches when every scale is within the Hamming threshold for any "
                "identity/flip/center-crop/letterbox protected descriptor"
            ),
            "residual_uncovered_transform_classes": [
                "rotation",
                "perspective warp",
                "off-center crop",
                "heavy occlusion or compositing",
                "combined flip-plus-crop or flip-plus-letterbox",
                "learned or adversarial pixel transforms",
            ],
            "protected_inventory": protected_inventory,
        }
    )

    _write_json(out_dir / "protected_collision_report.json", collision_report)
    if collision_report["collision_pair_count"]:
        _write_json(
            out_dir / "refusal.json",
            {
                "objective_result": "BLOCKED",
                "verdict": "PROTECTED_FRAME_COLLISION",
                "collision_pair_count": collision_report["collision_pair_count"],
                "collision_image_count": collision_report["collision_image_count"],
            },
        )
        raise ProtectedCollisionError(
            "exhaustive multi-scale pHash check found "
            f"{collision_report['collision_pair_count']} protected-frame collision pair(s); "
            f"report: {out_dir / 'protected_collision_report.json'}"
        )

    training_ready_gate = _training_ready_gate(
        human_review=human_review,
        pre_quarantine_retention=pre_quarantine_retention,
        post_quarantine_retention=post_quarantine_retention,
        post_quarantine_split_counts=post_quarantine_split_counts,
        cross_split_content_leak_count=int(
            cross_split_content_audit["verified_leak_count"]
        ),
        collision_pair_count=int(collision_report["collision_pair_count"]),
    )
    rows, materialization = _materialize_yolo(
        materialized_samples,
        out_dir=out_dir,
        split_for_source=split_for_source,
        family_for_source=family_for_source,
    )
    audit_summary = _materialize_audit_pack(
        audit_selection,
        out_dir=out_dir,
        requested_per_source=audit_samples_per_source,
        seed=seed,
        human_review=human_review,
        review_csv=review_csv,
    )
    source_balance = _write_family_balanced_train_list(
        rows,
        out_dir=out_dir,
        seed=seed,
        max_repetitions=family_balance_max_repetitions,
    )
    if training_ready_gate["status"] == "PASS":
        _write_data_yaml(out_dir / "data.yaml")
    _write_json(out_dir / "training_ready_gate.json", training_ready_gate)
    _assert_package_state_consistency(
        out_dir=out_dir,
        human_review=human_review,
        audit_summary=audit_summary,
        training_ready_gate=training_ready_gate,
    )

    active_source_counts = (
        post_quarantine_source_counts
        if post_quarantine_source_counts is not None
        else pre_quarantine_source_counts
    )
    active_split_counts = (
        post_quarantine_split_counts
        if post_quarantine_split_counts is not None
        else pre_quarantine_split_counts
    )
    active_family_counts = (
        post_quarantine_family_counts
        if post_quarantine_family_counts is not None
        else pre_quarantine_family_counts
    )
    if pre_quarantine_retention["verdict"] == "PERSON_RF_POOL_TOO_THIN" or (
        post_quarantine_retention
        and post_quarantine_retention["verdict"] == "PERSON_RF_POOL_TOO_THIN"
    ):
        verdict = "PERSON_RF_POOL_TOO_THIN"
    elif training_ready_gate["status"] == "PASS":
        verdict = "TRAINING_READY"
    elif human_review["status"] != "COMPLETE":
        verdict = "PENDING_HUMAN_REVIEW"
    elif post_quarantine_retention and post_quarantine_retention["verdict"] != "PASS":
        verdict = "PERSON_RF_POOL_TOO_THIN"
    else:
        verdict = "ANNOTATION_QUALITY_REJECTED"
    summary = {
        "schema_version": 2,
        "artifact_type": "racketsport_roboflow_person_yolo_dataset",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "objective_result": "PASS" if training_ready_gate["status"] == "PASS" else "PARTIAL",
        "verdict": verdict,
        "annotation_quality": human_review,
        "input": {
            "index": str(index_path),
            "index_sha256": index_sha256,
            "manifest": str(manifest_path),
            "corpus_card": str(corpus_card_path),
            "bucket": bucket,
            "index_images": len(samples),
            "index_boxes": sum(len(sample.get("labels") or []) for sample in samples),
        },
        "exclusions": {
            **exclusion_report,
            "requested_sources": sorted(requested_exclusions),
            "policy_forced_sources": sorted(forced_exclusions),
            "invalid_annotations": invalid_annotations,
            "invalid_annotation_count": len(invalid_annotations),
            "dropped_images_without_valid_boxes": dropped_images,
            "dropped_image_count": len(dropped_images),
        },
        "retained_source_count": sum(1 for row in active_source_counts if int(row["images"]) > 0),
        "eligible_index_images": len(eligible),
        "eligible_index_boxes": sum(len(sample.get("labels") or []) for sample in eligible),
        "retained_images": len(materialized_samples),
        "retained_boxes": sum(len(sample.get("labels") or []) for sample in materialized_samples),
        "rights": rights,
        "split_counts": active_split_counts,
        "source_counts": active_source_counts,
        "fork_families": active_family_counts,
        "original_video_family_map": original_video_family_map,
        "temporal_lineage_audit": temporal_lineage_audit,
        "cross_split_content_audit": cross_split_content_audit,
        "source_balance": source_balance,
        "retention": {
            "pre_quarantine": pre_quarantine_retention,
            "post_quarantine": post_quarantine_retention,
        },
        "training_ready_gate": training_ready_gate,
        "protected_collision_check": collision_report,
        "materialization": materialization,
        "data_yaml": str(out_dir / "data.yaml") if training_ready_gate["status"] == "PASS" else None,
        "data_yaml_status": (
            "PRESENT_TRAINING_READY"
            if training_ready_gate["status"] == "PASS"
            else "WITHHELD_UNTIL_TRAINING_READY"
        ),
        "dataset_manifest": str(out_dir / "dataset_manifest.json"),
        "audit_pack": audit_summary,
        "audit_sample_counts": audit_counts,
    }
    _write_json(
        out_dir / "source_counts.json",
        {
            "pre_quarantine": pre_quarantine_source_counts,
            "post_quarantine": post_quarantine_source_counts,
        },
    )
    _write_json(
        out_dir / "split_counts.json",
        {
            "pre_quarantine": pre_quarantine_split_counts,
            "post_quarantine": post_quarantine_split_counts,
        },
    )
    _write_json(
        out_dir / "fork_families.json",
        {
            "pre_quarantine": pre_quarantine_family_counts,
            "post_quarantine": post_quarantine_family_counts,
        },
    )
    _write_json(out_dir / "original_video_families.json", original_video_family_map)
    _write_json(out_dir / "temporal_lineage_audit.json", temporal_lineage_audit)
    _write_json(out_dir / "cross_split_content_audit.json", cross_split_content_audit)
    _write_json(out_dir / "source_balance.json", source_balance)
    _write_json(out_dir / "human_review.json", human_review)
    _write_json(
        out_dir / "retention.json",
        {
            "pre_quarantine": pre_quarantine_retention,
            "post_quarantine": post_quarantine_retention,
        },
    )
    _write_json(out_dir / "summary.json", summary)
    _write_attribution(out_dir / "ATTRIBUTION.md", rights)
    return summary


def _filter_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    bucket: str,
    exclusions: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    excluded_by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    excluded_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for raw in samples:
        sample = dict(raw)
        sample_bucket = str(sample.get("bucket") or "<missing>")
        source = str(sample.get("source_slug") or "<missing>")
        boxes = len(sample.get("labels") or [])
        if sample_bucket != bucket:
            excluded_by_bucket[sample_bucket].update(images=1, boxes=boxes)
            continue
        if source in exclusions:
            excluded_by_source[source].update(images=1, boxes=boxes)
            continue
        eligible.append(sample)
    return eligible, {
        "by_bucket": [
            {"bucket": name, "images": counts["images"], "boxes": counts["boxes"]}
            for name, counts in sorted(excluded_by_bucket.items())
        ],
        "by_source": [
            {"source": name, "images": counts["images"], "boxes": counts["boxes"]}
            for name, counts in sorted(excluded_by_source.items())
        ],
    }


def _validate_rights(source_names: Sequence[str], metadata: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in source_names:
        entry = metadata.get(source)
        if entry is None:
            raise ValueError(f"missing rights metadata for selected source: {source}")
        license_name = str(entry.get("license_as_recorded") or "")
        if license_name != ALLOWED_LICENSE:
            raise ValueError(
                f"selected source is not {ALLOWED_LICENSE}: {source} has {license_name or '<missing>'}"
            )
        rows.append(
            {
                "source": source,
                "license": license_name,
                "url": str(entry.get("url") or ""),
            }
        )
    return rows


def _build_fork_families(
    sources: Sequence[str],
    *,
    samples: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    corpus_card: Mapping[str, Any],
) -> tuple[dict[str, str], list[list[str]], dict[str, Any]]:
    parent = {source: source for source in sources}
    evidence_rows: list[dict[str, Any]] = []

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str, *, evidence_type: str, detail: str) -> None:
        if left not in parent or right not in parent:
            return
        evidence_rows.append(
            {
                "left": left,
                "right": right,
                "evidence_type": evidence_type,
                "detail": detail,
            }
        )
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for row in corpus_card.get("fork_duplicate_mappings", []):
        if isinstance(row, Mapping):
            union(
                str(row.get("duplicate_slug")),
                str(row.get("kept_slug")),
                evidence_type="corpus_card_duplicate_mapping",
                detail=str(row.get("reason") or "declared duplicate mapping"),
            )
    source_set = set(sources)
    for entry in manifest.get("entries", []):
        if not isinstance(entry, Mapping):
            continue
        source = str(entry.get("slug") or "")
        if source not in source_set:
            continue
        note = str(entry.get("note") or "")
        if not re.search(r"fork|mirror|duplicate", note, flags=re.IGNORECASE):
            continue
        for other in source_set - {source}:
            other_project = other.partition("/")[2]
            if other in note or (
                other_project
                and re.search(rf"(?<![a-z0-9]){re.escape(other_project)}(?![a-z0-9])", note, flags=re.IGNORECASE)
            ):
                union(
                    source,
                    other,
                    evidence_type="manifest_fork_note",
                    detail=note,
                )

    for adjudication in ADJUDICATED_ORIGINAL_VIDEO_FAMILIES:
        present = [source for source in adjudication["members"] if source in source_set]
        for other in present[1:]:
            union(
                present[0],
                other,
                evidence_type="adjudicated_original_video_game_session",
                detail="; ".join(str(value) for value in adjudication["evidence"]),
            )

    prefixes_by_channel: dict[tuple[str, str], set[str]] = defaultdict(set)
    for sample in samples:
        source = str(sample.get("source_slug") or "")
        if source not in source_set:
            continue
        prefix = _temporal_prefix(sample)
        if prefix:
            prefixes_by_channel[(_source_channel(source), prefix)].add(source)
    for (channel, prefix), matching_sources in sorted(prefixes_by_channel.items()):
        ordered = sorted(matching_sources)
        for other in ordered[1:]:
            union(
                ordered[0],
                other,
                evidence_type="shared_channel_temporal_prefix",
                detail=f"channel={channel}; filename_prefix={prefix}",
            )

    grouped: dict[str, list[str]] = defaultdict(list)
    for source in sources:
        grouped[find(source)].append(source)
    families = sorted((sorted(members) for members in grouped.values()), key=lambda members: members[0])
    family_for_source = {
        source: _canonical_family_id(members) for members in families for source in members
    }
    family_rows: list[dict[str, Any]] = []
    for members in families:
        member_set = set(members)
        adjudicated = next(
            (
                row
                for row in ADJUDICATED_ORIGINAL_VIDEO_FAMILIES
                if set(row["members"]).issubset(member_set)
            ),
            None,
        )
        family_rows.append(
            {
                "family_id": _canonical_family_id(members),
                "members": list(members),
                "original_video_family_id": (
                    adjudicated["original_video_family_id"]
                    if adjudicated is not None
                    else "UNRESOLVED_SOURCE_LEVEL_FAMILY"
                ),
                "game_session": (
                    adjudicated["game_session"] if adjudicated is not None else "UNRESOLVED"
                ),
                "channels": sorted({_source_channel(source) for source in members}),
                "temporal_prefixes_by_source": {
                    source: sorted(
                        {
                            prefix
                            for sample in samples
                            if str(sample.get("source_slug")) == source
                            if (prefix := _temporal_prefix(sample))
                        }
                    )
                    for source in members
                },
                "evidence": [
                    row
                    for row in evidence_rows
                    if row["left"] in member_set and row["right"] in member_set
                ],
            }
        )
    family_map = {
        "schema_version": 1,
        "artifact_type": "racketsport_original_video_game_session_channel_family_map",
        "policy": (
            "union declared forks, adjudicated original-video/game/session families, and sources "
            "sharing a temporal filename prefix inside one channel as whole sources; named "
            "holdout-side-wins with no frame-level carve-outs"
        ),
        "families": family_rows,
    }
    return family_for_source, families, family_map


def _canonical_family_id(members: Sequence[str]) -> str:
    member_set = set(members)
    adjudicated = next(
        (
            row
            for row in ADJUDICATED_ORIGINAL_VIDEO_FAMILIES
            if set(row["members"]).issubset(member_set)
        ),
        None,
    )
    return (
        str(adjudicated["family_id"])
        if adjudicated is not None
        else f"family:{sorted(members)[0]}"
    )


def _source_channel(source: str) -> str:
    return source.partition("/")[0].lower()


def _temporal_prefix(sample: Mapping[str, Any]) -> str:
    temporal = sample.get("temporal")
    if isinstance(temporal, Mapping) and temporal.get("filename_prefix"):
        raw = str(temporal["filename_prefix"])
    else:
        raw = _normalized_original_filename(sample)
        raw = re.sub(r"(?:[_-]?\d+)+$", "", raw)
    return re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")


def _normalized_original_filename(sample: Mapping[str, Any]) -> str:
    name = Path(str(sample.get("image_path") or "")).stem.lower()
    name = re.sub(r"\.rf\.[0-9a-f]+$", "", name)
    name = re.sub(r"_(?:jpg|jpeg|png|bmp|webp)$", "", name)
    return name


def _original_frame_identity(sample: Mapping[str, Any]) -> str:
    return f"{sample.get('source_slug')}:{_normalized_original_filename(sample)}"


def _audit_cross_split_lineage(
    samples: Sequence[Mapping[str, Any]],
    *,
    split_for_source: Mapping[str, str],
    family_for_source: Mapping[str, str],
) -> dict[str, Any]:
    temporal: dict[tuple[str, str], set[str]] = defaultdict(set)
    filenames: dict[tuple[str, str], set[str]] = defaultdict(set)
    for sample in samples:
        source = str(sample.get("source_slug") or "")
        prefix = _temporal_prefix(sample)
        if prefix:
            temporal[(_source_channel(source), prefix)].add(source)
        filename = _normalized_original_filename(sample)
        # Bare camera counters such as 0000 are not a filename lineage by
        # themselves; require an alphabetic stem and let temporal metadata
        # carry numeric-only sequences.
        if filename and re.search(r"[a-z]{3,}", filename):
            filenames[(_source_channel(source), filename)].add(source)

    def overlap_rows(groups: Mapping[tuple[str, str], set[str]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (channel, lineage), sources in sorted(groups.items()):
            if len(sources) < 2:
                continue
            splits = sorted({split_for_source[source] for source in sources})
            rows.append(
                {
                    "channel": channel,
                    "lineage": lineage,
                    "sources": sorted(sources),
                    "family_ids": sorted({family_for_source[source] for source in sources}),
                    "splits": splits,
                }
            )
        return rows

    temporal_rows = overlap_rows(temporal)
    filename_rows = overlap_rows(filenames)
    violations = [
        {"kind": kind, **row}
        for kind, rows in (("temporal_prefix", temporal_rows), ("filename_lineage", filename_rows))
        for row in rows
        if len(row["splits"]) > 1
    ]
    if violations:
        first = violations[0]
        raise ValueError(
            "cross-split temporal/filename-lineage overlap: "
            f"{first['kind']} channel={first['channel']} lineage={first['lineage']} "
            f"sources={first['sources']} splits={first['splits']}"
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_cross_split_temporal_filename_lineage_audit",
        "status": "PASS",
        "cross_split_overlap_count": 0,
        "temporal_prefix_overlap_rows": temporal_rows,
        "filename_lineage_overlap_rows": filename_rows,
    }


def _split_sources_by_family(
    families: Sequence[Sequence[str]],
    *,
    val_source: str,
    test_source: str,
) -> dict[str, str]:
    split_for_source: dict[str, str] = {}
    for members in families:
        member_set = set(members)
        if val_source in member_set and test_source in member_set:
            raise ValueError(
                "named validation and test sources belong to one fork family: "
                f"{list(members)}"
            )
        split = "val" if val_source in member_set else "test" if test_source in member_set else "train"
        split_for_source.update({source: split for source in members})
    return split_for_source


def _retention_snapshot(
    source_counts: Sequence[Mapping[str, Any]],
    *,
    status: str,
    min_train_images: int,
    min_train_source_groups: int,
) -> dict[str, Any]:
    train_rows = [
        row for row in source_counts if row["split"] == "train" and int(row["images"]) > 0
    ]
    train_images = sum(int(row["images"]) for row in train_rows)
    train_groups = {str(row["family_id"]) for row in train_rows}
    meets_threshold = (
        train_images >= min_train_images and len(train_groups) >= min_train_source_groups
    )
    provisional = status.startswith("PROVISIONAL")
    verdict = (
        "PROVISIONAL_PENDING_HUMAN_REVIEW"
        if provisional and meets_threshold
        else "PASS"
        if meets_threshold
        else "PERSON_RF_POOL_TOO_THIN"
    )
    return {
        "status": status,
        "verdict": verdict,
        "meets_numeric_threshold_provisionally": meets_threshold if provisional else None,
        "permanently_closes_training_for_export": not meets_threshold,
        "p2_disposition": (
            "NO_ATTEMPT_PREREQ"
            if not meets_threshold
            else "PENDING_HUMAN_REVIEW"
            if provisional
            else "NUMERIC_RETENTION_PASS"
        ),
        "train_images": train_images,
        "train_source_count": len(train_rows),
        "train_source_group_count": len(train_groups),
        "minimum_train_images": int(min_train_images),
        "minimum_train_source_groups": int(min_train_source_groups),
    }


def _ingest_human_review(
    review_csv: Path | None,
    *,
    audit_selection: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    expected = {
        (source, str(sample["sample_id"])): len(sample.get("labels") or [])
        for source, samples in audit_selection.items()
        for sample in samples
    }
    base = {
        "box_precision_target": 0.98,
        "visible_on_court_person_recall_target": 0.95,
        "source_quarantine_below": 0.90,
        "self_certified": False,
        "expected_review_row_count": len(expected),
    }
    if review_csv is None:
        return {
            **base,
            "status": "PENDING_HUMAN_REVIEW",
            "review_csv": None,
            "reviewed_row_count": 0,
            "pending_row_count": len(expected),
            "all_frames_marked_complete": False,
            "quality_targets_pass": False,
            "source_metrics": [],
            "quarantined_sources": [],
            "post_quarantine_box_precision": None,
            "post_quarantine_visible_on_court_person_recall": None,
        }
    if not review_csv.is_file():
        raise FileNotFoundError(f"missing completed human-review CSV: {review_csv}")

    with review_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REVIEW_COLUMNS:
            raise ValueError(
                f"review CSV columns must exactly match {list(REVIEW_COLUMNS)}, got {reader.fieldnames}"
            )
        supplied: dict[tuple[str, str], dict[str, str]] = {}
        for line_number, raw in enumerate(reader, start=2):
            row = {key: str(value or "").strip() for key, value in raw.items()}
            key = (row["source"], row["sample_id"])
            if key in supplied:
                raise ValueError(f"duplicate review row at line {line_number}: {key}")
            if key not in expected:
                raise ValueError(f"unexpected review row at line {line_number}: {key}")
            if int(row["drawn_box_count"] or -1) != expected[key]:
                raise ValueError(
                    f"review drawn_box_count mismatch at line {line_number}: "
                    f"expected {expected[key]}, got {row['drawn_box_count']!r}"
                )
            supplied[key] = row

    metric_columns = REVIEW_COLUMNS[3:7]
    pending_keys = [
        key
        for key in expected
        if key not in supplied or any(not supplied[key][column] for column in metric_columns)
    ]
    if pending_keys:
        return {
            **base,
            "status": "PENDING_HUMAN_REVIEW",
            "review_csv": str(review_csv),
            "review_csv_sha256": file_sha256(review_csv),
            "reviewed_row_count": len(expected) - len(pending_keys),
            "pending_row_count": len(pending_keys),
            "pending_rows": [
                {"source": source, "sample_id": sample_id}
                for source, sample_id in sorted(pending_keys)
            ],
            "all_frames_marked_complete": False,
            "quality_targets_pass": False,
            "source_metrics": [],
            "quarantined_sources": [],
            "post_quarantine_box_precision": None,
            "post_quarantine_visible_on_court_person_recall": None,
        }

    totals_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    all_frames_complete = True
    for key, expected_drawn in expected.items():
        row = supplied[key]
        try:
            correct = int(row["correct_person_box_count"])
            visible = int(row["visible_on_court_person_count"])
            annotated = int(row["annotated_visible_on_court_person_count"])
        except ValueError as exc:
            raise ValueError(f"review counts must be integers for {key}") from exc
        if not 0 <= correct <= expected_drawn:
            raise ValueError(f"correct_person_box_count out of range for {key}")
        if not 0 <= annotated <= visible:
            raise ValueError(f"annotated visible-person count out of range for {key}")
        complete = row["frame_complete_yes_no"].lower()
        if complete not in {"yes", "no"}:
            raise ValueError(f"frame_complete_yes_no must be yes or no for {key}")
        all_frames_complete = all_frames_complete and complete == "yes"
        totals_by_source[key[0]].update(
            drawn=expected_drawn,
            correct=correct,
            visible=visible,
            annotated=annotated,
            reviewed_frames=1,
            complete_frames=int(complete == "yes"),
        )

    source_metrics: list[dict[str, Any]] = []
    quarantined: list[str] = []
    for source, totals in sorted(totals_by_source.items()):
        precision = totals["correct"] / totals["drawn"] if totals["drawn"] else None
        recall = totals["annotated"] / totals["visible"] if totals["visible"] else None
        below = precision is None or recall is None or precision < 0.90 or recall < 0.90
        if below:
            quarantined.append(source)
        source_metrics.append(
            {
                "source": source,
                "reviewed_frames": totals["reviewed_frames"],
                "complete_frames": totals["complete_frames"],
                "drawn_boxes": totals["drawn"],
                "correct_person_boxes": totals["correct"],
                "visible_on_court_persons": totals["visible"],
                "annotated_visible_on_court_persons": totals["annotated"],
                "box_precision": precision,
                "visible_on_court_person_recall": recall,
                "status": "QUARANTINED_BELOW_90_PERCENT" if below else "RETAINED",
            }
        )

    retained_metrics = [row for row in source_metrics if row["source"] not in quarantined]
    drawn = sum(int(row["drawn_boxes"]) for row in retained_metrics)
    correct = sum(int(row["correct_person_boxes"]) for row in retained_metrics)
    visible = sum(int(row["visible_on_court_persons"]) for row in retained_metrics)
    annotated = sum(int(row["annotated_visible_on_court_persons"]) for row in retained_metrics)
    precision = correct / drawn if drawn else None
    recall = annotated / visible if visible else None
    targets_pass = (
        precision is not None and recall is not None and precision >= 0.98 and recall >= 0.95
    )
    return {
        **base,
        "status": "COMPLETE",
        "review_csv": str(review_csv),
        "review_csv_sha256": file_sha256(review_csv),
        "reviewed_row_count": len(expected),
        "pending_row_count": 0,
        "all_frames_marked_complete": all_frames_complete,
        "quality_targets_pass": targets_pass,
        "source_metrics": source_metrics,
        "quarantined_sources": quarantined,
        "post_quarantine_box_precision": precision,
        "post_quarantine_visible_on_court_person_recall": recall,
    }


def _training_ready_gate(
    *,
    human_review: Mapping[str, Any],
    pre_quarantine_retention: Mapping[str, Any],
    post_quarantine_retention: Mapping[str, Any] | None,
    post_quarantine_split_counts: Sequence[Mapping[str, Any]] | None,
    cross_split_content_leak_count: int,
    collision_pair_count: int,
) -> dict[str, Any]:
    blockers: list[str] = []
    terminal_blockers: list[str] = []
    if pre_quarantine_retention["verdict"] == "PERSON_RF_POOL_TOO_THIN":
        blockers.append("PERSON_RF_POOL_TOO_THIN")
        terminal_blockers.append("PERSON_RF_POOL_TOO_THIN")
    if human_review["status"] != "COMPLETE":
        blockers.append("PENDING_HUMAN_REVIEW")
    if not human_review["all_frames_marked_complete"]:
        blockers.append("REVIEW_HAS_INCOMPLETE_FRAMES")
    if not human_review["quality_targets_pass"]:
        blockers.append("POST_QUARANTINE_QUALITY_TARGETS_UNMET")
    if post_quarantine_retention is None:
        blockers.append("POST_QUARANTINE_RETENTION_NOT_COMPUTED")
    elif post_quarantine_retention["verdict"] != "PASS":
        post_retention_blocker = str(post_quarantine_retention["verdict"])
        if post_retention_blocker not in blockers:
            blockers.append(post_retention_blocker)
        if post_retention_blocker == "PERSON_RF_POOL_TOO_THIN":
            terminal_blockers.append(post_retention_blocker)
    if post_quarantine_split_counts is None:
        blockers.append("POST_QUARANTINE_SPLITS_NOT_COMPUTED")
    else:
        empty_splits = sorted(
            str(row["split"])
            for row in post_quarantine_split_counts
            if int(row["images"]) == 0
        )
        if empty_splits:
            blockers.append(f"EMPTY_POST_QUARANTINE_SPLITS:{','.join(empty_splits)}")
    if collision_pair_count:
        blockers.append("PROTECTED_FRAME_COLLISION")
        terminal_blockers.append("PROTECTED_FRAME_COLLISION")
    if cross_split_content_leak_count:
        blockers.append("CROSS_SPLIT_CONTENT_LEAK")
        terminal_blockers.append("CROSS_SPLIT_CONTENT_LEAK")
    if not blockers:
        status = "PASS"
    elif terminal_blockers or human_review["status"] == "COMPLETE":
        status = "FAIL"
    else:
        status = "PENDING"
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_person_p2_training_ready_gate",
        "status": status,
        "permanently_closed_for_export": bool(terminal_blockers),
        "p2_disposition": (
            "READY"
            if status == "PASS"
            else "NO_ATTEMPT_PREREQ"
            if status == "FAIL"
            else "NO_ATTEMPT_PENDING_PREREQ"
        ),
        "p2_contract": (
            "The exact P2 command requires data.yaml. The exporter creates data.yaml only when this gate passes."
        ),
        "data_yaml_required": True,
        "blockers": blockers,
    }


def _sanitize_samples(
    samples: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    seen_ids: set[str] = set()
    retained: list[dict[str, Any]] = []
    invalid_annotations: list[dict[str, Any]] = []
    dropped_images: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id") or "")
        if not sample_id or sample_id in seen_ids:
            raise ValueError(f"missing or duplicate sample_id at index {index}: {sample_id!r}")
        seen_ids.add(sample_id)
        image_path = Path(str(sample.get("image_path") or ""))
        if not image_path.is_file():
            raise FileNotFoundError(f"missing indexed image for {sample_id}: {image_path}")
        width, height = int(sample.get("width") or 0), int(sample.get("height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid indexed dimensions for {sample_id}: {width}x{height}")
        labels = sample.get("labels")
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"PERSON subset row has no labels: {sample_id}")
        valid_labels: list[dict[str, Any]] = []
        for label in labels:
            try:
                _yolo_line(label.get("bbox_xywh"), width=width, height=height)
            except ValueError as exc:
                invalid_annotations.append(
                    {
                        "source": str(sample.get("source_slug")),
                        "sample_id": sample_id,
                        "annotation_id": label.get("annotation_id"),
                        "bbox_xywh": label.get("bbox_xywh"),
                        "reason": str(exc),
                    }
                )
                continue
            valid_labels.append(dict(label))
        if not valid_labels:
            dropped_images.append(
                {
                    "source": str(sample.get("source_slug")),
                    "sample_id": sample_id,
                    "reason": "no_valid_person_boxes_after_geometry_validation",
                }
            )
            continue
        normalized = dict(sample)
        normalized["labels"] = valid_labels
        retained.append(normalized)
    return retained, invalid_annotations, dropped_images


def _source_count_rows(
    samples: Sequence[Mapping[str, Any]],
    *,
    indexed_samples: Sequence[Mapping[str, Any]],
    invalid_annotations: Sequence[Mapping[str, Any]],
    split_for_source: Mapping[str, str],
    family_for_source: Mapping[str, str],
) -> list[dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        source = str(sample["source_slug"])
        counts[source].update(images=1, boxes=len(sample.get("labels") or []))
    indexed_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in indexed_samples:
        indexed_counts[str(sample["source_slug"])].update(images=1, boxes=len(sample.get("labels") or []))
    invalid_counts = Counter(str(row["source"]) for row in invalid_annotations)
    return [
        {
            "source": source,
            "family_id": family_for_source[source],
            "split": split_for_source[source],
            "images": counts[source]["images"],
            "boxes": counts[source]["boxes"],
            "indexed_images": indexed_counts[source]["images"],
            "indexed_boxes": indexed_counts[source]["boxes"],
            "invalid_boxes_excluded": invalid_counts[source],
            "images_dropped": indexed_counts[source]["images"] - counts[source]["images"],
        }
        for source in sorted(indexed_counts)
    ]


def _split_count_rows(source_counts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, Counter[str]] = {split: Counter() for split in ("train", "val", "test")}
    for row in source_counts:
        images = int(row["images"])
        counts[str(row["split"])].update(
            images=images,
            boxes=int(row["boxes"]),
            sources=int(images > 0),
        )
    return [
        {
            "split": split,
            "images": counts[split]["images"],
            "boxes": counts[split]["boxes"],
            "sources": counts[split]["sources"],
        }
        for split in ("train", "val", "test")
    ]


def _family_count_rows(
    source_counts: Sequence[Mapping[str, Any]],
    families: Sequence[Sequence[str]],
    *,
    family_for_source: Mapping[str, str],
) -> list[dict[str, Any]]:
    by_source = {str(row["source"]): row for row in source_counts}
    rows: list[dict[str, Any]] = []
    for members in families:
        member_rows = [by_source[source] for source in members]
        rows.append(
            {
                "family_id": family_for_source[members[0]],
                "members": list(members),
                "split": str(member_rows[0]["split"]),
                "images": sum(int(row["images"]) for row in member_rows),
                "boxes": sum(int(row["boxes"]) for row in member_rows),
            }
        )
    return rows


def _select_audit_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    per_source: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_source[str(sample["source_slug"])].append(dict(sample))
    selected: dict[str, list[dict[str, Any]]] = {}
    for source, rows in sorted(by_source.items()):
        ranked_derivatives = sorted(
            rows,
            key=lambda row: hashlib.sha256(f"{seed}:{source}:{row['sample_id']}".encode()).hexdigest(),
        )
        one_per_original: dict[str, dict[str, Any]] = {}
        for row in ranked_derivatives:
            one_per_original.setdefault(_original_frame_identity(row), row)
        ranked = sorted(
            one_per_original.values(),
            key=lambda row: hashlib.sha256(
                f"{seed}:{source}:{_original_frame_identity(row)}".encode()
            ).hexdigest(),
        )
        selected[source] = ranked[: min(per_source, len(ranked))]
    return selected


def _hash_export_candidates(samples: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, list[str]]:
    hashes = np.empty((len(samples), len(PHASH_SCALES)), dtype=np.uint64)
    refs: list[str] = []
    for index, sample in enumerate(samples):
        path = Path(str(sample["image_path"]))
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"cannot decode indexed image: {path}")
        indexed_wh = (int(sample["width"]), int(sample["height"]))
        decoded_wh = (int(frame.shape[1]), int(frame.shape[0]))
        if decoded_wh != indexed_wh:
            raise ValueError(
                f"indexed/decoded size mismatch for {sample['sample_id']}: {indexed_wh} vs {decoded_wh}"
            )
        hashes[index] = multiscale_phash(frame)
        refs.append(str(sample["sample_id"]))
    return hashes, refs


def _hash_protected_root(
    protected_root: Path,
    *,
    threshold: int,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    manifest_path = protected_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"protected root is missing required four-clip manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("artifact_type") != "pickleball_local_eval_clip_bundle":
        raise ValueError(f"unexpected protected manifest artifact_type: {manifest.get('artifact_type')}")
    clip_rows = manifest.get("clips")
    if not isinstance(clip_rows, list):
        raise ValueError("protected manifest clips must be a list")
    by_clip = {str(row.get("clip")): row for row in clip_rows if isinstance(row, Mapping)}
    if set(by_clip) != set(PROTECTED_CLIP_IDS) or len(clip_rows) != len(PROTECTED_CLIP_IDS):
        raise ValueError(
            f"protected manifest must contain exactly {list(PROTECTED_CLIP_IDS)}, got {sorted(by_clip)}"
        )

    root_resolved = protected_root.resolve()
    expected_videos: dict[Path, Mapping[str, Any]] = {}
    for clip_id in PROTECTED_CLIP_IDS:
        row = by_clip[clip_id]
        path = (protected_root / str(row.get("source_video") or "")).resolve()
        try:
            path.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError(f"protected source_video escapes root for {clip_id}: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"missing protected manifest video for {clip_id}: {path}")
        expected_videos[path] = row

    descriptors: list[np.ndarray] = []
    refs: list[str] = []
    image_files = sorted(
        path for path in protected_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    video_files = sorted(
        path for path in protected_root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    unexpected_videos = sorted(set(path.resolve() for path in video_files) - set(expected_videos))
    missing_videos = sorted(set(expected_videos) - set(path.resolve() for path in video_files))
    if unexpected_videos or missing_videos:
        raise ValueError(
            "protected video inventory differs from manifest: "
            f"unexpected={[str(path) for path in unexpected_videos]}, "
            f"missing={[str(path) for path in missing_videos]}"
        )
    for path in image_files:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"cannot decode protected image: {path}")
        descriptors.append(_protected_descriptor_variants(frame))
        refs.append(str(path))

    video_rows: list[dict[str, Any]] = []
    probe_counts: dict[str, Counter[str]] = {
        transform: Counter() for transform in ROBUSTNESS_PROBE_TRANSFORMS
    }
    for clip_id in PROTECTED_CLIP_IDS:
        row = by_clip[clip_id]
        path = (protected_root / str(row["source_video"])).resolve()
        actual_sha256 = file_sha256(path)
        expected_sha256 = str(row.get("source_sha256") or "")
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"protected video SHA-256 mismatch for {clip_id}: expected {expected_sha256}, got {actual_sha256}"
            )
        expected_frames = int(row.get("frame_count") or 0)
        if expected_frames <= 0:
            raise ValueError(f"protected manifest frame_count must be positive for {clip_id}")
        expected_width = int(row.get("width") or 0)
        expected_height = int(row.get("height") or 0)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"cannot open protected video: {path}")
        advertised = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if advertised > 0 and advertised != expected_frames:
            raise ValueError(
                f"protected manifest/container frame-count mismatch for {clip_id}: "
                f"manifest {expected_frames}, advertised {advertised}"
            )
        probe_indexes = {0, expected_frames // 2, expected_frames - 1}
        decoded = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                decoded_wh = (int(frame.shape[1]), int(frame.shape[0]))
                if decoded_wh != (expected_width, expected_height):
                    raise ValueError(
                        f"protected video dimensions mismatch for {clip_id} frame {decoded}: "
                        f"manifest {(expected_width, expected_height)}, decoded {decoded_wh}"
                    )
                descriptors.append(_protected_descriptor_variants(frame))
                refs.append(f"{path}#frame={decoded}")
                if decoded in probe_indexes:
                    detections = _probe_transform_detection(frame, threshold=threshold)
                    for transform, detected in detections.items():
                        probe_counts[transform].update(total=1, detected=int(detected))
                decoded += 1
        finally:
            capture.release()
        if decoded != expected_frames:
            raise ValueError(
                f"protected video decode was not exhaustive for {clip_id}: "
                f"manifest {expected_frames}, decoded {decoded}"
            )
        video_rows.append(
            {
                "clip": clip_id,
                "path": str(path),
                "source_sha256": actual_sha256,
                "manifest_frames": expected_frames,
                "advertised_frames": advertised,
                "decoded_frames": decoded,
            }
        )
    if not descriptors:
        raise ValueError(f"protected root contains no decodable images or video frames: {protected_root}")
    clip_stills = [
        path
        for path in image_files
        if path.relative_to(protected_root).parts
        and path.relative_to(protected_root).parts[0] in set(PROTECTED_CLIP_IDS)
    ]
    additional_stills = [path for path in image_files if path not in clip_stills]
    robustness_probe = {
        "protected_video_probe_frame_count": max(
            (counts["total"] for counts in probe_counts.values()), default=0
        ),
        "classes": {
            transform: {
                "detected": counts["detected"],
                "total": counts["total"],
                "detection_rate": (
                    counts["detected"] / counts["total"] if counts["total"] else None
                ),
            }
            for transform, counts in probe_counts.items()
        },
    }
    return np.stack(descriptors), refs, {
        "root": str(protected_root),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "protected_image_file_count": len(image_files),
        "four_clip_directory_still_count": len(clip_stills),
        "additional_conservative_still_count": len(additional_stills),
        "additional_conservative_stills": [str(path) for path in additional_stills],
        "protected_video_count": len(video_files),
        "four_clip_video_frame_count": sum(row["decoded_frames"] for row in video_rows),
        "protected_frame_descriptor_count": len(descriptors),
        "descriptor_variants_per_frame": len(PROTECTED_TRANSFORMS),
        "robustness_probe": robustness_probe,
        "videos": video_rows,
    }


def multiscale_phash(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim == 2:
        gray = frame_bgr
    else:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return _gray_variant_phash(gray, "identity")


def _protected_descriptor_variants(frame_bgr: np.ndarray) -> np.ndarray:
    gray = (
        frame_bgr
        if frame_bgr.ndim == 2
        else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    )
    return np.stack(
        [
            multiscale_phash(_apply_probe_transform(frame_bgr, name))
            if name.startswith("letterbox_")
            else _gray_variant_phash(gray, name)
            for name in PROTECTED_TRANSFORMS
        ]
    )


def _gray_variant_phash(gray: np.ndarray, transform: str) -> np.ndarray:
    if transform == "horizontal_flip":
        transformed = gray[:, ::-1]
    elif transform.startswith("center_crop_"):
        fraction = int(transform.removeprefix("center_crop_").removesuffix("pct")) / 100.0
        inset_y = max(1, round(gray.shape[0] * fraction))
        inset_x = max(1, round(gray.shape[1] * fraction))
        transformed = gray[inset_y : gray.shape[0] - inset_y, inset_x : gray.shape[1] - inset_x]
    else:
        transformed = gray
    if transformed.size == 0:
        raise ValueError(f"image is too small for protected transform {transform}: {gray.shape}")
    values: list[np.uint64] = []
    for scale in PHASH_SCALES:
        if transform.startswith("letterbox_"):
            fraction = int(transform.removeprefix("letterbox_").removesuffix("pct")) / 100.0
            inset = max(1, round(scale * fraction))
            content_size = scale - 2 * inset
            if content_size <= 0:
                raise ValueError(f"invalid letterbox transform {transform} at scale {scale}")
            resized = np.zeros((scale, scale), dtype=np.uint8)
            resized[inset : inset + content_size, inset : inset + content_size] = cv2.resize(
                gray, (content_size, content_size), interpolation=cv2.INTER_AREA
            )
        else:
            resized = cv2.resize(transformed, (scale, scale), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.asarray(resized, dtype=np.float32))[:8, :8].reshape(-1)
        median = float(np.median(dct[1:]))
        bits = dct > median
        bits[0] = False
        value = 0
        for bit_index, enabled in enumerate(bits):
            if bool(enabled):
                value |= 1 << bit_index
        values.append(np.uint64(value))
    return np.asarray(values, dtype=np.uint64)


def _apply_probe_transform(frame: np.ndarray, transform: str) -> np.ndarray:
    if transform == "horizontal_flip":
        return cv2.flip(frame, 1)
    fraction = int(transform.rsplit("_", 1)[1].removesuffix("pct")) / 100.0
    height, width = frame.shape[:2]
    inset_y = max(1, round(height * fraction))
    inset_x = max(1, round(width * fraction))
    if transform.startswith("center_crop_"):
        return frame[inset_y : height - inset_y, inset_x : width - inset_x].copy()
    if transform.startswith("letterbox_"):
        canvas = np.zeros_like(frame)
        resized = cv2.resize(
            frame,
            (width - 2 * inset_x, height - 2 * inset_y),
            interpolation=cv2.INTER_AREA,
        )
        canvas[inset_y : height - inset_y, inset_x : width - inset_x] = resized
        return canvas
    raise ValueError(f"unknown robustness-probe transform: {transform}")


def _probe_transform_detection(frame: np.ndarray, *, threshold: int) -> dict[str, bool]:
    protected = _protected_descriptor_variants(frame)
    results: dict[str, bool] = {}
    for transform in ROBUSTNESS_PROBE_TRANSFORMS:
        candidate = multiscale_phash(_apply_probe_transform(frame, transform))
        xor = np.bitwise_xor(candidate[None, :], protected)
        distances = _BYTE_POPCOUNT[xor.view(np.uint8).reshape(len(PROTECTED_TRANSFORMS), len(PHASH_SCALES), 8)].sum(
            axis=-1
        )
        results[transform] = bool(np.any(np.all(distances <= threshold, axis=-1)))
    return results


def _compare_multiscale_hashes(
    exported: np.ndarray,
    exported_refs: Sequence[str],
    protected: np.ndarray,
    protected_refs: Sequence[str],
    *,
    threshold: int,
    batch_size: int = 64,
) -> dict[str, Any]:
    if exported.shape != (len(exported_refs), len(PHASH_SCALES)):
        raise ValueError("exported descriptor shape mismatch")
    if protected.shape != (
        len(protected_refs),
        len(PROTECTED_TRANSFORMS),
        len(PHASH_SCALES),
    ):
        raise ValueError("protected descriptor shape mismatch")
    collisions: list[dict[str, Any]] = []
    collision_image_indexes: set[int] = set()
    for start in range(0, len(exported_refs), batch_size):
        stop = min(len(exported_refs), start + batch_size)
        batch_count = stop - start
        best_variant = np.full((batch_count, len(protected_refs)), -1, dtype=np.int16)
        best_score = np.full((batch_count, len(protected_refs)), 255, dtype=np.uint8)
        best_distances = np.zeros(
            (batch_count, len(protected_refs), len(PHASH_SCALES)), dtype=np.uint8
        )
        for variant_index in range(len(PROTECTED_TRANSFORMS)):
            xor = np.bitwise_xor(
                exported[start:stop, None, :], protected[None, :, variant_index, :]
            )
            byte_view = xor.view(np.uint8).reshape(
                batch_count, len(protected_refs), len(PHASH_SCALES), 8
            )
            distances = _BYTE_POPCOUNT[byte_view].sum(axis=-1).astype(np.uint8)
            matches = np.all(distances <= threshold, axis=-1)
            scores = np.max(distances, axis=-1)
            better = matches & (scores < best_score)
            best_variant[better] = variant_index
            best_score[better] = scores[better]
            best_distances[better] = distances[better]
        for local_index, protected_index in np.argwhere(best_variant >= 0):
            exported_index = start + int(local_index)
            collision_image_indexes.add(exported_index)
            variant_index = int(best_variant[local_index, protected_index])
            collisions.append(
                {
                    "sample_id": exported_refs[exported_index],
                    "protected_frame": protected_refs[int(protected_index)],
                    "matched_protected_transform": PROTECTED_TRANSFORMS[variant_index],
                    "hamming_distances": [
                        int(value) for value in best_distances[local_index, protected_index]
                    ],
                }
            )
    return {
        "exported_image_count": len(exported_refs),
        "protected_frame_count": len(protected_refs),
        "exhaustive_pair_count": len(exported_refs) * len(protected_refs),
        "descriptor_comparison_count": (
            len(exported_refs) * len(protected_refs) * len(PROTECTED_TRANSFORMS)
        ),
        "hamming_threshold_per_scale": int(threshold),
        "collision_image_count": len(collision_image_indexes),
        "collision_pair_count": len(collisions),
        "collisions": collisions,
    }


def _audit_cross_split_content(
    samples: Sequence[Mapping[str, Any]],
    *,
    all_hashes: np.ndarray,
    all_hash_refs: Sequence[str],
    split_for_source: Mapping[str, str],
    family_for_source: Mapping[str, str],
    threshold: int,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Exhaustively scan the final split for renamed shared footage.

    Multi-scale pHash is only the candidate generator. Every candidate is
    decoded and independently verified by SSIM and ORB homography evidence.
    """

    if all_hashes.shape != (len(all_hash_refs), len(PHASH_SCALES)):
        raise ValueError("content-audit descriptor shape mismatch")
    hash_index = {sample_id: index for index, sample_id in enumerate(all_hash_refs)}
    if len(hash_index) != len(all_hash_refs):
        raise ValueError("content-audit hash references must be unique")
    final_samples = sorted(
        (dict(sample) for sample in samples),
        key=lambda row: (str(row["source_slug"]), str(row["sample_id"])),
    )
    for sample in final_samples:
        sample_id = str(sample["sample_id"])
        if sample_id not in hash_index:
            raise ValueError(f"content-audit sample lacks pHash descriptor: {sample_id}")

    split_indexes: dict[str, list[int]] = {split: [] for split in ("train", "val", "test")}
    for sample_index, sample in enumerate(final_samples):
        source = str(sample["source_slug"])
        split_indexes[split_for_source[source]].append(sample_index)
    descriptors = np.stack(
        [all_hashes[hash_index[str(sample["sample_id"])]] for sample in final_samples]
    )

    candidate_index_pairs: list[tuple[int, int, list[int]]] = []
    split_pair_rows: list[dict[str, Any]] = []
    exhaustive_pair_count = 0
    for left_split, right_split in (("train", "val"), ("train", "test"), ("val", "test")):
        left_indexes = np.asarray(split_indexes[left_split], dtype=np.int64)
        right_indexes = np.asarray(split_indexes[right_split], dtype=np.int64)
        pair_count = int(len(left_indexes) * len(right_indexes))
        exhaustive_pair_count += pair_count
        candidate_count_before = len(candidate_index_pairs)
        for start in range(0, len(left_indexes), batch_size):
            left_batch = left_indexes[start : start + batch_size]
            xor = np.bitwise_xor(
                descriptors[left_batch, None, :], descriptors[right_indexes[None, :], :]
            )
            byte_view = xor.view(np.uint8).reshape(
                len(left_batch), len(right_indexes), len(PHASH_SCALES), 8
            )
            distances = _BYTE_POPCOUNT[byte_view].sum(axis=-1).astype(np.uint8)
            for left_local, right_local in np.argwhere(np.all(distances <= threshold, axis=-1)):
                candidate_index_pairs.append(
                    (
                        int(left_batch[int(left_local)]),
                        int(right_indexes[int(right_local)]),
                        [int(value) for value in distances[left_local, right_local]],
                    )
                )
        split_pair_rows.append(
            {
                "left_split": left_split,
                "right_split": right_split,
                "left_images": len(left_indexes),
                "right_images": len(right_indexes),
                "exhaustive_pair_count": pair_count,
                "phash_candidate_pair_count": len(candidate_index_pairs) - candidate_count_before,
            }
        )

    decoded_cache: dict[int, np.ndarray] = {}

    def decoded(index: int) -> np.ndarray:
        if index not in decoded_cache:
            path = Path(str(final_samples[index]["image_path"]))
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"cannot decode content-audit candidate: {path}")
            decoded_cache[index] = frame
        return decoded_cache[index]

    candidate_rows: list[dict[str, Any]] = []
    verified_leaks: list[dict[str, Any]] = []
    for left_index, right_index, distances in candidate_index_pairs:
        left = final_samples[left_index]
        right = final_samples[right_index]
        similarity = _verify_content_candidate(decoded(left_index), decoded(right_index))
        verified = bool(similarity["verified_shared_content"])
        row = {
            "left": _content_sample_ref(
                left,
                split_for_source=split_for_source,
                family_for_source=family_for_source,
            ),
            "right": _content_sample_ref(
                right,
                split_for_source=split_for_source,
                family_for_source=family_for_source,
            ),
            "phash_hamming_distances": distances,
            "verification": similarity,
            "verdict": "VERIFIED_CROSS_SPLIT_LEAK" if verified else "CANDIDATE_CLEARED",
        }
        candidate_rows.append(row)
        if verified:
            verified_leaks.append(row)

    return {
        "schema_version": 1,
        "artifact_type": "racketsport_cross_split_content_leak_audit",
        "status": "FAIL" if verified_leaks else "PASS",
        "mandatory_production_check": True,
        "final_split_image_counts": {
            split: len(split_indexes[split]) for split in ("train", "val", "test")
        },
        "method": {
            "candidate_generator": "multi-scale DCT pHash; every scale must be <= threshold",
            "phash_scales": list(PHASH_SCALES),
            "phash_hamming_threshold_per_scale": int(threshold),
            "candidate_verifier": "Gaussian-window SSIM plus ORB-RANSAC homography",
            "verified_when": (
                f"SSIM >= {CONTENT_SSIM_THRESHOLD:.2f} OR ORB homography inliers >= "
                f"{CONTENT_ORB_MIN_INLIERS} with inlier ratio >= {CONTENT_ORB_MIN_INLIER_RATIO:.2f}"
            ),
        },
        "split_pairs": split_pair_rows,
        "exhaustive_pair_count": exhaustive_pair_count,
        "phash_candidate_pair_count": len(candidate_rows),
        "cleared_candidate_pair_count": len(candidate_rows) - len(verified_leaks),
        "verified_leak_count": len(verified_leaks),
        "verified_leaks": verified_leaks,
        "candidate_pairs": candidate_rows,
    }


def _content_sample_ref(
    sample: Mapping[str, Any],
    *,
    split_for_source: Mapping[str, str],
    family_for_source: Mapping[str, str],
) -> dict[str, Any]:
    source = str(sample["source_slug"])
    return {
        "sample_id": str(sample["sample_id"]),
        "source": source,
        "split": split_for_source[source],
        "family_id": family_for_source[source],
        "image_path": str(sample["image_path"]),
    }


def _verify_content_candidate(left_bgr: np.ndarray, right_bgr: np.ndarray) -> dict[str, Any]:
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY) if left_bgr.ndim == 3 else left_bgr
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY) if right_bgr.ndim == 3 else right_bgr
    normalized_size = (256, 256)
    left_normalized = cv2.resize(left_gray, normalized_size, interpolation=cv2.INTER_AREA)
    right_normalized = cv2.resize(right_gray, normalized_size, interpolation=cv2.INTER_AREA)
    ssim = _ssim(left_normalized, right_normalized)
    orb = _orb_homography_evidence(left_gray, right_gray)
    verified = bool(
        ssim >= CONTENT_SSIM_THRESHOLD
        or (
            orb["homography_inliers"] >= CONTENT_ORB_MIN_INLIERS
            and orb["homography_inlier_ratio"] >= CONTENT_ORB_MIN_INLIER_RATIO
        )
    )
    return {
        "ssim": ssim,
        **orb,
        "verified_shared_content": verified,
    }


def _ssim(left_gray: np.ndarray, right_gray: np.ndarray) -> float:
    left = np.asarray(left_gray, dtype=np.float32)
    right = np.asarray(right_gray, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError(f"SSIM shape mismatch: {left.shape} vs {right.shape}")
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    mu_left = cv2.GaussianBlur(left, (11, 11), 1.5)
    mu_right = cv2.GaussianBlur(right, (11, 11), 1.5)
    mu_left_sq = mu_left * mu_left
    mu_right_sq = mu_right * mu_right
    mu_cross = mu_left * mu_right
    sigma_left = cv2.GaussianBlur(left * left, (11, 11), 1.5) - mu_left_sq
    sigma_right = cv2.GaussianBlur(right * right, (11, 11), 1.5) - mu_right_sq
    sigma_cross = cv2.GaussianBlur(left * right, (11, 11), 1.5) - mu_cross
    numerator = (2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_left_sq + mu_right_sq + c1) * (sigma_left + sigma_right + c2)
    score = float(np.mean(numerator / np.maximum(denominator, np.finfo(np.float32).eps)))
    return round(score, 9)


def _orb_homography_evidence(left_gray: np.ndarray, right_gray: np.ndarray) -> dict[str, Any]:
    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=7)
    left_keypoints, left_descriptors = orb.detectAndCompute(left_gray, None)
    right_keypoints, right_descriptors = orb.detectAndCompute(right_gray, None)
    good_matches: list[cv2.DMatch] = []
    if left_descriptors is not None and right_descriptors is not None:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        for candidates in matcher.knnMatch(left_descriptors, right_descriptors, k=2):
            if len(candidates) == 2 and candidates[0].distance < 0.80 * candidates[1].distance:
                good_matches.append(candidates[0])
    inliers = 0
    if len(good_matches) >= 4:
        left_points = np.float32(
            [left_keypoints[match.queryIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)
        right_points = np.float32(
            [right_keypoints[match.trainIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(left_points, right_points, cv2.RANSAC, 5.0)
        if mask is not None:
            inliers = int(mask.ravel().sum())
    return {
        "left_orb_keypoints": len(left_keypoints),
        "right_orb_keypoints": len(right_keypoints),
        "orb_ratio_matches": len(good_matches),
        "homography_inliers": inliers,
        "homography_inlier_ratio": round(inliers / len(good_matches), 9) if good_matches else 0.0,
    }


def _materialize_yolo(
    samples: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path,
    split_for_source: Mapping[str, str],
    family_for_source: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    materialization_counts: Counter[str] = Counter()
    for sample in sorted(samples, key=lambda row: (str(row["source_slug"]), str(row["sample_id"]))):
        source = str(sample["source_slug"])
        split = split_for_source[source]
        source_path = Path(str(sample["image_path"])).resolve()
        suffix = source_path.suffix.lower() or ".jpg"
        stem = f"{_safe_token(source)}__{hashlib.sha256(str(sample['sample_id']).encode()).hexdigest()[:16]}"
        image_path = out_dir / "images" / split / f"{stem}{suffix}"
        label_path = out_dir / "labels" / split / f"{stem}.txt"
        try:
            os.link(source_path, image_path)
            materialization_counts["hardlink"] += 1
        except OSError:
            shutil.copy2(source_path, image_path)
            materialization_counts["copy_fallback"] += 1
        lines = [
            _yolo_line(label.get("bbox_xywh"), width=int(sample["width"]), height=int(sample["height"]))
            for label in sample["labels"]
        ]
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        rows.append(
            {
                "sample_id": str(sample["sample_id"]),
                "source": source,
                "family_id": family_for_source[source],
                "split": split,
                "source_image": str(source_path),
                "image": image_path.relative_to(out_dir).as_posix(),
                "label": label_path.relative_to(out_dir).as_posix(),
                "box_count": len(lines),
            }
        )
    materialization = {
        "mode": "hardlink_with_copy_fallback",
        "reason": (
            "regular files keep the dataset movable; hardlinks avoid duplicate blocks on the source volume, "
            "and cross-device exports fall back to copies"
        ),
        "hardlink_count": materialization_counts["hardlink"],
        "copy_fallback_count": materialization_counts["copy_fallback"],
        "image_count": len(rows),
        "box_count": sum(int(row["box_count"]) for row in rows),
    }
    _write_json(
        out_dir / "dataset_manifest.json",
        {
            "schema_version": 2,
            "artifact_type": "racketsport_roboflow_person_yolo_rows",
            "materialization": materialization,
            "rows": rows,
        },
    )
    return rows, materialization


def _write_family_balanced_train_list(
    rows: Sequence[Mapping[str, Any]],
    *,
    out_dir: Path,
    seed: int,
    max_repetitions: int,
) -> dict[str, Any]:
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            by_family[str(row["family_id"])].append(row)
    counts = sorted(len(family_rows) for family_rows in by_family.values())
    target = counts[(len(counts) - 1) // 2] if counts else 0
    sampled_rows: list[Mapping[str, Any]] = []
    family_rows_out: list[dict[str, Any]] = []
    for family, family_rows in sorted(by_family.items()):
        ranked = sorted(
            family_rows,
            key=lambda row: hashlib.sha256(
                f"{seed}:{family}:{row['sample_id']}".encode()
            ).hexdigest(),
        )
        sampled_count = min(target, len(ranked) * max_repetitions)
        selected = [ranked[index % len(ranked)] for index in range(sampled_count)]
        sampled_rows.extend(selected)
        repetitions = Counter(str(row["sample_id"]) for row in selected)
        family_rows_out.append(
            {
                "family_id": family,
                "materialized_unique_images": len(ranked),
                "sampled_entries": len(selected),
                "sampled_unique_images": len(repetitions),
                "maximum_image_repetitions": max(repetitions.values(), default=0),
                "omitted_unique_images": len(ranked) - len(repetitions),
            }
        )
    lines = [f"./{row['image']}" for row in sampled_rows]
    train_list = out_dir / "train_family_balanced.txt"
    train_list.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {
        "enabled": True,
        "unit": "original_video_fork_family",
        "policy": (
            "deterministic lower-median family target; large families are hash-downsampled, small "
            "families are repeated only up to the declared per-image cap"
        ),
        "seed": int(seed),
        "target_entries_per_family": target,
        "maximum_repetitions_per_image": int(max_repetitions),
        "train_list": str(train_list),
        "train_list_entry_count": len(lines),
        "p2_consumption": "data.yaml train points directly at this list when training_ready_gate passes",
        "families": family_rows_out,
    }


def _write_data_yaml(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "train: train_family_balanced.txt",
                "val: images/val",
                "test: images/test",
                "nc: 1",
                "names:",
                "  0: person",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _materialize_audit_pack(
    selection: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    out_dir: Path,
    requested_per_source: int,
    seed: int,
    human_review: Mapping[str, Any],
    review_csv: Path | None,
) -> dict[str, Any]:
    audit_dir = out_dir / "audit"
    samples_dir = audit_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    html_sections: list[str] = []
    for source, samples in sorted(selection.items()):
        safe_source = _safe_token(source)
        source_dir = samples_dir / safe_source
        source_dir.mkdir(parents=True, exist_ok=True)
        cards: list[str] = []
        for position, sample in enumerate(samples, start=1):
            frame = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"cannot decode audit sample: {sample['image_path']}")
            for box_number, label in enumerate(sample["labels"], start=1):
                x1, y1, x2, y2 = _clipped_xyxy(
                    label.get("bbox_xywh"),
                    width=int(sample["width"]),
                    height=int(sample["height"]),
                )
                cv2.rectangle(frame, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    str(box_number),
                    (round(x1), max(14, round(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            output_name = f"{position:02d}__{hashlib.sha256(str(sample['sample_id']).encode()).hexdigest()[:12]}.jpg"
            output_path = source_dir / output_name
            if not cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                raise RuntimeError(f"failed to write audit image: {output_path}")
            relative = output_path.relative_to(audit_dir)
            row = {
                "source": source,
                "sample_id": str(sample["sample_id"]),
                "original_frame_identity": _original_frame_identity(sample),
                "audit_image": str(output_path),
                "source_image": str(sample["image_path"]),
                "drawn_box_count": len(sample["labels"]),
                "annotation_ids": [label.get("annotation_id") for label in sample["labels"]],
            }
            manifest_rows.append(row)
            cards.append(
                "<figure><img loading='lazy' src='{}' alt='{}'><figcaption>{} — {} boxes</figcaption></figure>".format(
                    html.escape(relative.as_posix(), quote=True),
                    html.escape(str(sample["sample_id"]), quote=True),
                    html.escape(str(sample["sample_id"])),
                    len(sample["labels"]),
                )
            )
        shortfall = max(0, requested_per_source - len(samples))
        html_sections.append(
            f"<section><h2>{html.escape(source)}</h2><p>{len(samples)} unique staged / {requested_per_source} requested"
            f"; shortfall {shortfall}.</p><div class='grid'>{''.join(cards)}</div></section>"
        )
    review_status = str(human_review["status"])
    completed_review_path: Path | None = None
    provided_incomplete_review_path: Path | None = None
    review_template_path: Path | None = None
    if review_status == "COMPLETE":
        if review_csv is None or not review_csv.is_file():
            raise RuntimeError("COMPLETE human review requires a readable review CSV")
        completed_review_path = audit_dir / "review_completed.csv"
        shutil.copy2(review_csv, completed_review_path)
    else:
        review_template_path = audit_dir / "review_template.csv"
        with review_template_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(REVIEW_COLUMNS)
            for row in manifest_rows:
                writer.writerow(
                    [row["source"], row["sample_id"], row["drawn_box_count"], "", "", "", "", ""]
                )
        if review_csv is not None and review_csv.is_file():
            provided_incomplete_review_path = audit_dir / "review_incomplete.csv"
            shutil.copy2(review_csv, provided_incomplete_review_path)

    audit_manifest = {
        "schema_version": 1,
        "artifact_type": "racketsport_roboflow_person_human_audit_pack",
        "status": review_status,
        "self_certified": False,
        "seed": int(seed),
        "requested_samples_per_source": int(requested_per_source),
        "reviewed_row_count": int(human_review["reviewed_row_count"]),
        "pending_row_count": int(human_review["pending_row_count"]),
        "completed_review_csv": (
            str(completed_review_path) if completed_review_path is not None else None
        ),
        "completed_review_csv_sha256": (
            file_sha256(completed_review_path) if completed_review_path is not None else None
        ),
        "provided_incomplete_review_csv": (
            str(provided_incomplete_review_path)
            if provided_incomplete_review_path is not None
            else None
        ),
        "rows": manifest_rows,
    }
    _write_json(audit_dir / "audit_manifest.json", audit_manifest)
    pending = review_status != "COMPLETE"
    heading = (
        "<h1>Roboflow PERSON audit — human review pending</h1>"
        if pending
        else "<h1>Roboflow PERSON audit — human review complete</h1>"
    )
    review_note = (
        "This page does not certify precision or recall.</p>"
        if pending
        else "The completed review CSV is bundled with this package.</p>"
    )
    (audit_dir / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Roboflow PERSON human audit</title>"
        "<style>body{font:15px system-ui;margin:24px;background:#f5f5f2;color:#181816}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}"
        "figure{margin:0;background:white;padding:10px;border:1px solid #ccc}img{width:100%;height:auto}"
        "figcaption{font:12px ui-monospace,monospace;overflow-wrap:anywhere;margin-top:6px}</style></head><body>"
        + heading
        + "<p>Green numbered rectangles are upstream boxes. For each frame, count correct person boxes, every "
        + "visible on-court person, and how many of those visible people have a box. "
        + review_note
        + "".join(html_sections)
        + "</body></html>\n",
        encoding="utf-8",
    )
    per_source = Counter(row["source"] for row in manifest_rows)
    return {
        "status": review_status,
        "self_certified": False,
        "path": str(audit_dir),
        "page": str(audit_dir / "index.html"),
        "review_template": (
            str(review_template_path) if review_template_path is not None else None
        ),
        "completed_review_csv": (
            str(completed_review_path) if completed_review_path is not None else None
        ),
        "completed_review_csv_sha256": (
            file_sha256(completed_review_path) if completed_review_path is not None else None
        ),
        "provided_incomplete_review_csv": (
            str(provided_incomplete_review_path)
            if provided_incomplete_review_path is not None
            else None
        ),
        "manifest": str(audit_dir / "audit_manifest.json"),
        "staged_unique_sample_count": len(manifest_rows),
        "per_source_sample_counts": dict(sorted(per_source.items())),
    }


def _assert_package_state_consistency(
    *,
    out_dir: Path,
    human_review: Mapping[str, Any],
    audit_summary: Mapping[str, Any],
    training_ready_gate: Mapping[str, Any],
) -> None:
    data_yaml_exists = (out_dir / "data.yaml").is_file()
    if training_ready_gate["status"] != "PASS":
        if data_yaml_exists:
            raise RuntimeError("non-PASS package must not contain data.yaml")
        return
    if human_review["status"] != "COMPLETE":
        raise RuntimeError("PASS package contradicts non-COMPLETE human review")
    if audit_summary["status"] != "COMPLETE":
        raise RuntimeError("PASS package contradicts non-COMPLETE bundled audit")
    completed_review = audit_summary.get("completed_review_csv")
    if not completed_review or not Path(str(completed_review)).is_file():
        raise RuntimeError("PASS package lacks bundled completed review CSV")
    manifest = _read_json(Path(str(audit_summary["manifest"])))
    if manifest.get("status") != "COMPLETE" or manifest.get("pending_row_count") != 0:
        raise RuntimeError("PASS package audit manifest is pending or incomplete")
    if manifest.get("completed_review_csv_sha256") != file_sha256(Path(str(completed_review))):
        raise RuntimeError("PASS package completed-review hash does not match audit manifest")
    if not data_yaml_exists:
        raise RuntimeError("PASS package is missing data.yaml")


def _yolo_line(bbox_xywh: Any, *, width: int, height: int) -> str:
    x1, y1, x2, y2 = _clipped_xyxy(bbox_xywh, width=width, height=height)
    box_w, box_h = x2 - x1, y2 - y1
    center_x, center_y = x1 + box_w / 2.0, y1 + box_h / 2.0
    return f"0 {center_x / width:.6f} {center_y / height:.6f} {box_w / width:.6f} {box_h / height:.6f}"


def _clipped_xyxy(bbox_xywh: Any, *, width: int, height: int) -> tuple[float, float, float, float]:
    if not isinstance(bbox_xywh, Sequence) or isinstance(bbox_xywh, (str, bytes)) or len(bbox_xywh) != 4:
        raise ValueError(f"bbox_xywh must contain four numbers: {bbox_xywh!r}")
    try:
        x, y, box_w, box_h = (float(value) for value in bbox_xywh)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bbox_xywh must contain finite numbers: {bbox_xywh!r}") from exc
    if not all(np.isfinite(value) for value in (x, y, box_w, box_h)) or box_w <= 0 or box_h <= 0:
        raise ValueError(f"invalid bbox_xywh: {bbox_xywh!r}")
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + box_w), min(float(height), y + box_h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox is outside image bounds: {bbox_xywh!r} in {width}x{height}")
    return x1, y1, x2, y2


def _metadata_paths(index_path: Path) -> tuple[Path, Path]:
    if index_path.parent.name != "subset_indexes" or index_path.parent.parent.name != "aggregated":
        raise ValueError("index must live under <roboflow-root>/aggregated/subset_indexes/")
    aggregated = index_path.parent.parent
    manifest = aggregated.parent / "manifest.json"
    card = aggregated / "corpus_card.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing Roboflow source manifest: {manifest}")
    if not card.is_file():
        raise FileNotFoundError(f"missing Roboflow corpus card: {card}")
    return manifest, card


def _write_attribution(path: Path, rights: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# Roboflow PERSON source attribution", "", "All retained sources are recorded as CC BY 4.0.", ""]
    for row in rights:
        lines.append(f"- {row['source']} — {row['license']} — {row['url']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not token:
        raise ValueError(f"cannot make safe token from {value!r}")
    return token


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compact_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "objective_result": summary["objective_result"],
        "verdict": summary["verdict"],
        "retained_source_count": summary["retained_source_count"],
        "retained_images": summary["retained_images"],
        "retained_boxes": summary["retained_boxes"],
        "split_counts": summary["split_counts"],
        "collision_count": summary["protected_collision_check"]["collision_pair_count"],
        "retention": summary["retention"],
        "training_ready_gate": summary["training_ready_gate"],
        "audit_pack": summary["audit_pack"]["path"],
        "summary": str(Path(summary["dataset_manifest"]).parent / "summary.json"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
