#!/usr/bin/env python3
"""Build the BALL B0 parent-source split and scratch-only evaluation judge.

The builder reconciles every reviewed label with its original package prelabel,
keeps accepted prelabels out of evaluation, holds out complete parent sources,
and exhaustively compares the staged scratch images with every frame of the
four protected BALL clips plus the two owner court-keypoint additions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEWED_ROOT = Path("runs/lanes/w7_ballingest4_20260709/reviewed_corpus")
DEFAULT_SCRATCH_PACKAGE = Path("cvat_upload/w7_audit_stratum_20260709/package_manifest.json")
DEFAULT_SCRATCH_EXPORT = Path(
    "cvat_upload/exports/w7_audit_stratum_20260709/"
    "w7_audit_stratum_uniform350_annotations.zip"
)
DEFAULT_NEGATIVE_ATTESTATION = Path(
    "cvat_upload/exports/w7_audit_stratum_20260709/owner_attestation.json"
)
DEFAULT_SCRATCH_IMPORT_LEDGER = Path(
    "runs/lanes/w7_auditstratum_20260709/import_report.json"
)
DEFAULT_OUT = Path("runs/lanes/ball_b0_split_20260721/split")
DEFAULT_SELECTION_MANIFESTS = (
    Path("runs/lanes/w5_labelpack_20260708/selection_manifest.json"),
    Path("runs/lanes/w6_labelpack_20260708/selection_manifest.json"),
)
DEFAULT_LEGACY_REVIEWED_ROOT = Path("cvat_upload/exports/harvest_review_20260707")
DEFAULT_LEGACY_PRELABEL_ROOT = Path("data/online_harvest_20260706/prelabels")
DEFAULT_PROTECTED_VIDEOS = (
    Path("eval_clips/ball/burlington_gold_0300_low_steep_corner/source.mp4"),
    Path("eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/source.mp4"),
    Path("eval_clips/ball/outdoor_webcam_iynbd_1500_long_high_baseline/source.mp4"),
    Path("eval_clips/ball/indoor_doubles_fwuks_0500_long_mid_baseline/source.mp4"),
)
DEFAULT_PROTECTED_ADDITIONS = (
    Path(
        "eval_clips/ball/owner_IMG_1605_8a193402780b/labels/"
        "court_keypoint_partial_frames/frame_000060.jpg"
    ),
    Path(
        "eval_clips/ball/owner_IMG_1605_8a193402780b/labels/"
        "court_keypoint_partial_frames/frame_000240.jpg"
    ),
)
DEFAULT_HOLDOUT_COUNTS = {"Ezz6HDNHlnk": 67, "HyUqT7zFiwk": 100}
EXPECTED_REVIEWED_REPORT_SHA256 = (
    "c4200032e86f912d68adfefdb27118e48b2ed0673fee6122a21df638c601968d"
)
EXPECTED_SCRATCH_PACKAGE_SHA256 = (
    "a04fd956ac56c16130643a79344c21298dec6a3e69f507e0788e996a927a9a55"
)
EXPECTED_SCRATCH_EXPORT_SHA256 = (
    "fea4b9529a4020ee577c5702478753fa4ea84f99c51a9e3ef16e87e96d7fc104"
)
EXPECTED_SCRATCH_IMPORT_LEDGER_SHA256 = (
    "46a306143bd1caa20ba599df5be41e72a1d23bc124ab23d313166b30c68f983f"
)
EXPECTED_SCRATCH_IMAGE_ZIP_SHA256 = (
    "f1b7ba88084c8664202bf19f73e4704599b46bd42e50e2a6c1a29265cff8b653"
)
EXPECTED_SCRATCH_TASK_ID = 87
EXPECTED_SCRATCH_JOB_ID = 87
EXPECTED_SCRATCH_TASK_NAME = "w7_audit_stratum_uniform350"
EXPECTED_SCRATCH_TASK_FINGERPRINT = (
    "fbe7f0db96c553ad4570f95260e592a91f7aaaa20ae0ebf7b56951105eefae8d"
)
# Aggregate SHA-256 over the pinned selection manifests, legacy CVAT review
# exports, and the legacy prelabel tracks those exports identify.  This value
# is populated from the canonical files below and deliberately has no
# path-mutable fallback.
EXPECTED_LINEAGE_INPUTS_SHA256 = (
    "56545ee6f099695d03f8f7ba0ae5085856b6c2996225185fed11f740a79701bd"
)
TASK_87_JOB_ID_RESIDUAL = (
    "local CVAT rendered the imported staged bytes for task 87; "
    "no independent job-id binding exists in the historical import ledger"
)
CANONICAL_PROTECTED_SHA256 = {
    DEFAULT_PROTECTED_VIDEOS[0]: "fc329b53a8d522046779a45fba4e695ee953421e1187070a4ce9a36239cb1aaa",
    DEFAULT_PROTECTED_VIDEOS[1]: "7f6c33b7cfd94a063405b68708d37d968cc1850e7435aa875f5b30f0afb6cb4b",
    DEFAULT_PROTECTED_VIDEOS[2]: "8b0265f5dc3bf3e3b5b5a1423bf7e58ac7972481dc163b8398dcb2f20bf070c9",
    DEFAULT_PROTECTED_VIDEOS[3]: "22955134f7bf9bdc9392bdde868173fbc6ec9afa4d7a8c58f3e7e0ed33d4e0f1",
    DEFAULT_PROTECTED_ADDITIONS[0]: "c8b05ee22213b8036de72f80b68590c1a985b4ff51653b041da0bb0ce9dfdf09",
    DEFAULT_PROTECTED_ADDITIONS[1]: "682b22d64ca0a1dce98f5eca80abfcc4247310426b0993e56aa6ab89b35c5265",
}
PROTECTED_ROW_TOKENS = (
    "burlington_gold",
    "wolverine_mixed",
    "outdoor_webcam_iynbd",
    "indoor_doubles_fwuks",
    "pwxnwffyqlq",
    "vqhtz8l6vqu",
    "iynbdrs1jdk",
    "fwuks",
)
IMAGE_NAME_RE = re.compile(
    r"^(?P<source>.+?)__(?P<clip>.+?)__abs_(?P<frame>\d+)\.(?:jpe?g|png)$",
    re.IGNORECASE,
)
VALID_VISIBILITY = {"clear", "partial", "full", "out_of_frame", "none"}


class BallNoCleanJudge(ValueError):
    """Raised whenever the B0 clean-judge contract cannot be proven."""


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    selection_manifests = args.selection_manifest or list(DEFAULT_SELECTION_MANIFESTS)
    protected_videos = args.protected_video or list(DEFAULT_PROTECTED_VIDEOS)
    protected_additions = args.protected_addition or list(DEFAULT_PROTECTED_ADDITIONS)
    try:
        expected_holdout_counts = _parse_expected_holdout_counts(
            args.expected_holdout_count,
            holdout_sources=args.holdout_source,
        )
        common = {
            "reviewed_root": args.reviewed_root,
            "scratch_package": args.scratch_package,
            "holdout_sources": args.holdout_source,
            "out": args.out,
            "selection_manifests": selection_manifests,
            "legacy_reviewed_root": args.legacy_reviewed_root,
            "legacy_prelabel_root": args.legacy_prelabel_root,
            "protected_videos": protected_videos,
            "protected_additions": protected_additions,
            "collision_hamming_threshold": args.collision_hamming_threshold,
            "expected_reviewed_count": args.expected_reviewed_count,
            "expected_scratch_count": args.expected_scratch_count,
            "expected_holdout_counts": expected_holdout_counts,
            "confirmed_prelabel_weight": args.confirmed_prelabel_weight,
            "expected_reviewed_report_sha256": args.expected_reviewed_report_sha256,
            "expected_scratch_package_sha256": args.expected_scratch_package_sha256,
            "expected_image_zip_sha256": args.expected_image_zip_sha256,
            "expected_lineage_inputs_sha256": args.expected_lineage_inputs_sha256,
            "production_mode": not args.fixture_mode,
        }
        if args.lineage_only:
            report = build_lineage_reconciliation(**common)
        else:
            report = build_ball_regroup_split(
                **common,
                scratch_export=args.scratch_export,
                negative_attestation=args.negative_attestation,
                scratch_import_ledger=args.scratch_import_ledger,
                expected_old_train_count=args.expected_old_train_count,
                expected_scratch_train_count=args.expected_scratch_train_count,
                expected_validation_count=args.expected_validation_count,
                expected_scratch_export_sha256=args.expected_scratch_export_sha256,
                expected_scratch_import_ledger_sha256=(
                    args.expected_scratch_import_ledger_sha256
                ),
                expected_task_fingerprint=args.expected_task_fingerprint,
            )
    except BallNoCleanJudge as exc:
        print(f"BALL_NO_CLEAN_JUDGE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-root", type=Path, default=DEFAULT_REVIEWED_ROOT)
    parser.add_argument("--scratch-package", type=Path, default=DEFAULT_SCRATCH_PACKAGE)
    parser.add_argument("--scratch-export", type=Path, default=DEFAULT_SCRATCH_EXPORT)
    parser.add_argument(
        "--negative-attestation",
        type=Path,
        default=None,
        help=(
            "Required for a full judge build. Owner JSON attesting that all export "
            "frames were inspected and boxless images mean no visible ball."
        ),
    )
    parser.add_argument(
        "--scratch-import-ledger", type=Path, default=DEFAULT_SCRATCH_IMPORT_LEDGER
    )
    parser.add_argument("--holdout-source", action="append", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--lineage-only",
        action="store_true",
        help="Reconcile the 3,026 reviewed rows and package collision guard without an export.",
    )
    parser.add_argument("--selection-manifest", action="append", type=Path, default=[])
    parser.add_argument(
        "--legacy-reviewed-root", type=Path, default=DEFAULT_LEGACY_REVIEWED_ROOT
    )
    parser.add_argument(
        "--legacy-prelabel-root", type=Path, default=DEFAULT_LEGACY_PRELABEL_ROOT
    )
    parser.add_argument("--protected-video", action="append", type=Path, default=[])
    parser.add_argument("--protected-addition", action="append", type=Path, default=[])
    parser.add_argument(
        "--fixture-mode",
        action="store_true",
        help="Allow synthetic protected inputs; can never emit BALL_CLEAN_JUDGE.",
    )
    parser.add_argument("--collision-hamming-threshold", type=int, default=3)
    parser.add_argument("--confirmed-prelabel-weight", type=float, default=0.25)
    parser.add_argument("--expected-reviewed-count", type=int, default=3026)
    parser.add_argument("--expected-scratch-count", type=int, default=350)
    parser.add_argument("--expected-old-train-count", type=int, default=2066)
    parser.add_argument("--expected-scratch-train-count", type=int, default=183)
    parser.add_argument("--expected-validation-count", type=int, default=167)
    parser.add_argument(
        "--expected-holdout-count",
        action="append",
        default=[],
        metavar="SOURCE=COUNT",
    )
    parser.add_argument(
        "--expected-reviewed-report-sha256",
        default=EXPECTED_REVIEWED_REPORT_SHA256,
    )
    parser.add_argument(
        "--expected-scratch-package-sha256",
        default=EXPECTED_SCRATCH_PACKAGE_SHA256,
    )
    parser.add_argument(
        "--expected-scratch-export-sha256",
        default=EXPECTED_SCRATCH_EXPORT_SHA256,
    )
    parser.add_argument(
        "--expected-scratch-import-ledger-sha256",
        default=EXPECTED_SCRATCH_IMPORT_LEDGER_SHA256,
    )
    parser.add_argument(
        "--expected-image-zip-sha256",
        default=EXPECTED_SCRATCH_IMAGE_ZIP_SHA256,
    )
    parser.add_argument(
        "--expected-task-fingerprint",
        default=EXPECTED_SCRATCH_TASK_FINGERPRINT,
    )
    parser.add_argument(
        "--expected-lineage-inputs-sha256",
        default=EXPECTED_LINEAGE_INPUTS_SHA256,
    )
    return parser


def build_ball_regroup_split(
    *,
    reviewed_root: Path,
    scratch_package: Path,
    scratch_export: Path,
    negative_attestation: Path | None,
    scratch_import_ledger: Path,
    holdout_sources: Sequence[str],
    out: Path,
    selection_manifests: Sequence[Path],
    legacy_reviewed_root: Path | None,
    legacy_prelabel_root: Path | None,
    protected_videos: Sequence[Path],
    protected_additions: Sequence[Path],
    collision_hamming_threshold: int,
    expected_reviewed_count: int,
    expected_scratch_count: int,
    expected_old_train_count: int,
    expected_scratch_train_count: int,
    expected_validation_count: int,
    expected_holdout_counts: Mapping[str, int],
    confirmed_prelabel_weight: float,
    expected_reviewed_report_sha256: str | None,
    expected_scratch_package_sha256: str | None,
    expected_scratch_export_sha256: str,
    expected_scratch_import_ledger_sha256: str,
    expected_image_zip_sha256: str,
    expected_task_fingerprint: str,
    expected_lineage_inputs_sha256: str,
    production_mode: bool,
) -> dict[str, Any]:
    """Build the deterministic B0 split, raising on any clean-judge violation."""

    _validate_production_inputs(
        production_mode=production_mode,
        reviewed_root=reviewed_root,
        scratch_package=scratch_package,
        holdout_sources=holdout_sources,
        selection_manifests=selection_manifests,
        legacy_reviewed_root=legacy_reviewed_root,
        legacy_prelabel_root=legacy_prelabel_root,
        expected_reviewed_count=expected_reviewed_count,
        expected_scratch_count=expected_scratch_count,
        expected_reviewed_report_sha256=expected_reviewed_report_sha256,
        expected_scratch_package_sha256=expected_scratch_package_sha256,
        expected_image_zip_sha256=expected_image_zip_sha256,
        expected_lineage_inputs_sha256=expected_lineage_inputs_sha256,
    )
    if production_mode:
        if scratch_export.resolve() != DEFAULT_SCRATCH_EXPORT.resolve():
            raise BallNoCleanJudge("production mode requires the canonical scratch export path")
        if scratch_import_ledger.resolve() != DEFAULT_SCRATCH_IMPORT_LEDGER.resolve():
            raise BallNoCleanJudge(
                "production mode requires the canonical task-87 import ledger path"
            )
        if expected_scratch_export_sha256 != EXPECTED_SCRATCH_EXPORT_SHA256:
            raise BallNoCleanJudge("production scratch export SHA-256 pin is immutable")
        if (
            expected_scratch_import_ledger_sha256
            != EXPECTED_SCRATCH_IMPORT_LEDGER_SHA256
        ):
            raise BallNoCleanJudge("production import-ledger SHA-256 pin is immutable")
        if expected_image_zip_sha256 != EXPECTED_SCRATCH_IMAGE_ZIP_SHA256:
            raise BallNoCleanJudge("production image-ZIP SHA-256 pin is immutable")
        if expected_task_fingerprint != EXPECTED_SCRATCH_TASK_FINGERPRINT:
            raise BallNoCleanJudge("production task fingerprint pin is immutable")
        if (
            expected_old_train_count != 2066
            or expected_scratch_train_count != 183
            or expected_validation_count != 167
            or dict(expected_holdout_counts) != DEFAULT_HOLDOUT_COUNTS
        ):
            raise BallNoCleanJudge("production split-count pins are immutable")
    _validate_common_arguments(
        holdout_sources=holdout_sources,
        collision_hamming_threshold=collision_hamming_threshold,
        confirmed_prelabel_weight=confirmed_prelabel_weight,
    )
    old_rows, input_contract = _reconcile_reviewed_rows(
        reviewed_root=reviewed_root,
        selection_manifests=selection_manifests,
        legacy_reviewed_root=legacy_reviewed_root,
        legacy_prelabel_root=legacy_prelabel_root,
        confirmed_prelabel_weight=confirmed_prelabel_weight,
        expected_reviewed_count=expected_reviewed_count,
        expected_reviewed_report_sha256=expected_reviewed_report_sha256,
        expected_lineage_inputs_sha256=expected_lineage_inputs_sha256,
    )
    scratch_package_rows, package_contract, candidate_hashes = _load_scratch_package(
        scratch_package,
        expected_scratch_count=expected_scratch_count,
        expected_scratch_package_sha256=expected_scratch_package_sha256,
        expected_image_zip_sha256=expected_image_zip_sha256,
    )
    protected_guard = _assert_no_protected_collisions(
        candidate_hashes,
        protected_videos=protected_videos,
        protected_additions=protected_additions,
        threshold=collision_hamming_threshold,
        production_mode=production_mode,
    )
    scratch_rows, export_contract = _parse_scratch_export(
        scratch_export,
        package_rows=scratch_package_rows,
        expected_scratch_count=expected_scratch_count,
        negative_attestation=negative_attestation,
        scratch_import_ledger=scratch_import_ledger,
        expected_export_sha256=expected_scratch_export_sha256,
        expected_import_ledger_sha256=expected_scratch_import_ledger_sha256,
        expected_task_fingerprint=expected_task_fingerprint,
        expected_task_name=str(package_contract["scratch_task_name"]),
        expected_image_zip=Path(str(package_contract["scratch_image_zip"])),
        expected_image_zip_sha256=expected_image_zip_sha256,
    )
    historical_scratch_overlap = _row_key_intersection(old_rows, scratch_rows)
    if historical_scratch_overlap:
        raise BallNoCleanJudge(
            "scratch package repeats historical reviewed rows: "
            f"{historical_scratch_overlap[:5]}"
        )

    holdout = set(holdout_sources)
    old_train = [row for row in old_rows if row["source_id"] not in holdout]
    scratch_train = [row for row in scratch_rows if row["source_id"] not in holdout]
    validation = [row for row in scratch_rows if row["source_id"] in holdout]
    for row in scratch_train:
        row["evaluation_eligible"] = False
    train = sorted([*old_train, *scratch_train], key=_row_sort_key)
    validation = sorted(validation, key=_row_sort_key)
    all_lineage = sorted([*old_rows, *scratch_rows], key=_row_sort_key)

    materialized_image_binding = _verify_materialized_scratch_image_bytes(
        [*scratch_train, *validation],
        expected_image_zip=Path(str(package_contract["scratch_image_zip"])),
        expected_image_zip_sha256=str(package_contract["image_zip_sha256"]),
        expected_entry_sha256=dict(package_contract["image_zip_entry_sha256"]),
    )

    _assert_count(len(old_train), expected_old_train_count, "old training row count")
    _assert_count(
        len(scratch_train), expected_scratch_train_count, "scratch training row count"
    )
    _assert_count(len(validation), expected_validation_count, "validation row count")
    actual_holdout_counts = Counter(str(row["source_id"]) for row in validation)
    if dict(sorted(actual_holdout_counts.items())) != dict(sorted(expected_holdout_counts.items())):
        raise BallNoCleanJudge(
            "holdout scratch counts mismatch: "
            f"expected={dict(sorted(expected_holdout_counts.items()))} "
            f"actual={dict(sorted(actual_holdout_counts.items()))}"
        )

    train_sources = {str(row["source_id"]) for row in train}
    validation_sources = {str(row["source_id"]) for row in validation}
    source_intersection = sorted(train_sources & validation_sources)
    if source_intersection:
        raise BallNoCleanJudge(
            f"parent-source leakage between train and validation: {source_intersection}"
        )
    non_scratch_eval = [row["row_key"] for row in validation if row["lineage_class"] != "scratch"]
    if non_scratch_eval:
        raise BallNoCleanJudge(
            f"evaluation contains non-scratch lineage rows: {non_scratch_eval[:5]}"
        )
    confirmed_eval = [
        row["row_key"] for row in validation if row["lineage_class"] == "confirmed_prelabel"
    ]
    if confirmed_eval:
        raise BallNoCleanJudge(
            f"confirmed_prelabel rows reached evaluation: {confirmed_eval[:5]}"
        )
    _assert_no_protected_row_tokens(all_lineage)

    metrics_by_source = {
        source_id: {
            "row_count": int(actual_holdout_counts[source_id]),
            "lineage_required": "scratch",
            "f1_at_20": None,
            "recall": None,
            "precision": None,
            "hidden_fp": None,
        }
        for source_id in sorted(holdout)
    }
    if set(metrics_by_source) != holdout:
        raise BallNoCleanJudge("evaluation metrics are not reported separately for every holdout")

    lineage_counts = _lineage_counts(all_lineage)
    checks = {
        "reviewed_lineage_reconciled": _pass_check(
            f"{len(old_rows)}/{expected_reviewed_count}"
        ),
        "scratch_package_reconciled": _pass_check(
            f"{len(scratch_rows)}/{expected_scratch_count}"
        ),
        "historical_scratch_row_intersection": {
            "verdict": "PASS",
            "count": 0,
            "row_keys": [],
        },
        "train_validation_source_intersection": {
            "verdict": "PASS",
            "count": 0,
            "sources": [],
        },
        "protected_collision_count": {
            "verdict": "PASS",
            "count": 0,
            "candidate_image_count": protected_guard["candidate_image_count"],
            "protected_frame_count": protected_guard["protected_frame_count"],
        },
        "evaluation_lineage": {
            "verdict": "PASS",
            "row_count": len(validation),
            "required": "scratch",
        },
        "confirmed_prelabel_policy": {
            "verdict": "PASS",
            "training_weight": confirmed_prelabel_weight,
            "evaluation_row_count": 0,
        },
        "metrics_separated_by_holdout_source": {
            "verdict": "PASS",
            "sources": sorted(metrics_by_source),
        },
        "scratch_materialized_image_bytes": materialized_image_binding,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_regroup_parent_source_split",
        "verdict": "BALL_CLEAN_JUDGE" if production_mode else "BALL_FIXTURE_CLEAN",
        "ball_verified": False,
        "production_eligible": production_mode,
        "split_semantics": "parent_source",
        "input_contract": {**input_contract, **package_contract, **export_contract},
        "residual_assumptions": list(export_contract["residual_assumptions"]),
        "checks": checks,
        "lineage_counts": lineage_counts,
        "split_counts": {
            "old_train": len(old_train),
            "scratch_train": len(scratch_train),
            "train": len(train),
            "validation": len(validation),
        },
        "train_sources": sorted(train_sources),
        "validation_sources": sorted(validation_sources),
        "evaluation_metrics_by_source": metrics_by_source,
        "protected_collision_guard": protected_guard,
        "weight_policy": {
            "scratch": 1.0,
            "corrected_prelabel": 1.0,
            "confirmed_prelabel": confirmed_prelabel_weight,
            "confirmed_prelabel_authority": "teacher_only_not_ground_truth",
        },
        "artifacts": {
            "lineage_rows": str(out / "lineage_rows.jsonl"),
            "train": str(out / "train.jsonl"),
            "validation": str(out / "validation.jsonl"),
            "report": str(out / "report.json"),
        },
    }
    _write_split_artifacts(
        out,
        lineage_rows=all_lineage,
        train_rows=train,
        validation_rows=validation,
        report=report,
    )
    return report


def build_lineage_reconciliation(
    *,
    reviewed_root: Path,
    scratch_package: Path,
    holdout_sources: Sequence[str],
    out: Path,
    selection_manifests: Sequence[Path],
    legacy_reviewed_root: Path | None,
    legacy_prelabel_root: Path | None,
    protected_videos: Sequence[Path],
    protected_additions: Sequence[Path],
    collision_hamming_threshold: int,
    expected_reviewed_count: int,
    expected_scratch_count: int,
    expected_holdout_counts: Mapping[str, int],
    confirmed_prelabel_weight: float,
    expected_reviewed_report_sha256: str | None,
    expected_scratch_package_sha256: str | None,
    expected_image_zip_sha256: str,
    expected_lineage_inputs_sha256: str,
    production_mode: bool,
) -> dict[str, Any]:
    """Reconcile real historical lineage while the scratch export is still pending."""

    _validate_production_inputs(
        production_mode=production_mode,
        reviewed_root=reviewed_root,
        scratch_package=scratch_package,
        holdout_sources=holdout_sources,
        selection_manifests=selection_manifests,
        legacy_reviewed_root=legacy_reviewed_root,
        legacy_prelabel_root=legacy_prelabel_root,
        expected_reviewed_count=expected_reviewed_count,
        expected_scratch_count=expected_scratch_count,
        expected_reviewed_report_sha256=expected_reviewed_report_sha256,
        expected_scratch_package_sha256=expected_scratch_package_sha256,
        expected_image_zip_sha256=expected_image_zip_sha256,
        expected_lineage_inputs_sha256=expected_lineage_inputs_sha256,
    )
    _validate_common_arguments(
        holdout_sources=holdout_sources,
        collision_hamming_threshold=collision_hamming_threshold,
        confirmed_prelabel_weight=confirmed_prelabel_weight,
    )
    old_rows, input_contract = _reconcile_reviewed_rows(
        reviewed_root=reviewed_root,
        selection_manifests=selection_manifests,
        legacy_reviewed_root=legacy_reviewed_root,
        legacy_prelabel_root=legacy_prelabel_root,
        confirmed_prelabel_weight=confirmed_prelabel_weight,
        expected_reviewed_count=expected_reviewed_count,
        expected_reviewed_report_sha256=expected_reviewed_report_sha256,
        expected_lineage_inputs_sha256=expected_lineage_inputs_sha256,
    )
    package_rows, package_contract, candidate_hashes = _load_scratch_package(
        scratch_package,
        expected_scratch_count=expected_scratch_count,
        expected_scratch_package_sha256=expected_scratch_package_sha256,
        expected_image_zip_sha256=expected_image_zip_sha256,
    )
    historical_scratch_overlap = _row_key_intersection(old_rows, package_rows)
    if historical_scratch_overlap:
        raise BallNoCleanJudge(
            "scratch package repeats historical reviewed rows: "
            f"{historical_scratch_overlap[:5]}"
        )
    package_holdout_counts = Counter(
        str(row["source_id"])
        for row in package_rows
        if str(row["source_id"]) in set(holdout_sources)
    )
    if dict(sorted(package_holdout_counts.items())) != dict(sorted(expected_holdout_counts.items())):
        raise BallNoCleanJudge(
            "scratch package holdout counts mismatch: "
            f"expected={dict(sorted(expected_holdout_counts.items()))} "
            f"actual={dict(sorted(package_holdout_counts.items()))}"
        )
    protected_guard = _assert_no_protected_collisions(
        candidate_hashes,
        protected_videos=protected_videos,
        protected_additions=protected_additions,
        threshold=collision_hamming_threshold,
        production_mode=production_mode,
    )
    _assert_no_protected_row_tokens(old_rows)
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_regroup_lineage_reconciliation",
        "verdict": "SPLITTER_READY_AWAITING_EXPORT",
        "ball_verified": False,
        "production_eligible": production_mode,
        "split_semantics": "parent_source",
        "input_contract": {**input_contract, **package_contract},
        "checks": {
            "reviewed_lineage_reconciled": _pass_check(
                f"{len(old_rows)}/{expected_reviewed_count}"
            ),
            "scratch_package_candidates": _pass_check(
                f"{len(package_rows)}/{expected_scratch_count}"
            ),
            "historical_scratch_row_intersection": {
                "verdict": "PASS",
                "count": 0,
                "row_keys": [],
            },
            "protected_collision_count": {
                "verdict": "PASS",
                "count": 0,
                "candidate_image_count": protected_guard["candidate_image_count"],
                "protected_frame_count": protected_guard["protected_frame_count"],
            },
            "real_export_ingested": {
                "verdict": "PENDING",
                "ingested": False,
            },
        },
        "lineage_counts": _lineage_counts(old_rows),
        "scratch_candidate_counts": {
            "total": len(package_rows),
            "per_source": dict(
                sorted(Counter(str(row["source_id"]) for row in package_rows).items())
            ),
        },
        "protected_collision_guard": protected_guard,
        "weight_policy": {
            "corrected_prelabel": 1.0,
            "confirmed_prelabel": confirmed_prelabel_weight,
            "confirmed_prelabel_authority": "teacher_only_not_ground_truth",
        },
        "artifacts": {
            "lineage_rows": str(out / "lineage_rows.jsonl"),
            "report": str(out / "report.json"),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "lineage_rows.jsonl", sorted(old_rows, key=_row_sort_key))
    _write_json(out / "report.json", report)
    return report


def _reconcile_reviewed_rows(
    *,
    reviewed_root: Path,
    selection_manifests: Sequence[Path],
    legacy_reviewed_root: Path | None,
    legacy_prelabel_root: Path | None,
    confirmed_prelabel_weight: float,
    expected_reviewed_count: int,
    expected_reviewed_report_sha256: str | None,
    expected_lineage_inputs_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not reviewed_root.is_dir():
        raise BallNoCleanJudge(f"missing reviewed corpus root: {reviewed_root}")
    ingest_report = reviewed_root.parent / "ingest_report.json"
    report_sha = _check_optional_sha256(
        ingest_report,
        expected_reviewed_report_sha256,
        label="reviewed ingest report",
    )
    reviewed_manifest_contract: dict[str, Any] = {}
    if report_sha is not None:
        reviewed_manifest_contract = _verify_reviewed_corpus_manifest(
            reviewed_root, _load_json(ingest_report)
        )
    lineage_input_contract = _verify_lineage_input_digest(
        selection_manifests=selection_manifests,
        legacy_reviewed_root=legacy_reviewed_root,
        legacy_prelabel_root=legacy_prelabel_root,
        expected_sha256=expected_lineage_inputs_sha256,
    )
    selection_rows = _load_selection_rows(selection_manifests)
    legacy_keys = _legacy_reviewed_keys(legacy_reviewed_root)
    prediction_cache: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    origins: Counter[str] = Counter()
    for payload_path in sorted(reviewed_root.glob("*/reviewed_boxes.json")):
        payload = _load_json(payload_path)
        clip_id = str(payload.get("clip_id") or payload_path.parent.name)
        source_id = _source_id_from_clip(clip_id)
        dimensions = _original_dimensions(payload)
        frames_by_index = {
            int(frame["frame_index"]): frame for frame in payload.get("frames") or []
        }
        for frame_index in sorted(
            {int(value) for value in payload.get("reviewed_frame_indices") or []}
        ):
            row_key = f"{clip_id}:{frame_index:06d}"
            if row_key in seen:
                raise BallNoCleanJudge(f"duplicate reviewed row key: {row_key}")
            seen.add(row_key)
            frame = frames_by_index.get(frame_index)
            if frame is None:
                raise BallNoCleanJudge(f"reviewed row is missing frame payload: {row_key}")
            final_label = _final_label(frame, row_key=row_key)
            if row_key in legacy_keys:
                if legacy_prelabel_root is None:
                    raise BallNoCleanJudge(
                        f"legacy prelabel root is required to reconcile {row_key}"
                    )
                lineage_origin = "legacy_wasb_prelabel"
                predictions = prediction_cache.get(clip_id)
                if predictions is None:
                    prediction_path = legacy_prelabel_root / clip_id / "ball_track.json"
                    prediction_payload = _load_json(prediction_path)
                    predictions = list(prediction_payload.get("frames") or [])
                    prediction_cache[clip_id] = predictions
                if frame_index >= len(predictions):
                    raise BallNoCleanJudge(
                        f"legacy prelabel missing frame {frame_index} for {clip_id}"
                    )
                original_prelabel = _legacy_prelabel(predictions[frame_index])
                confirmed = _legacy_label_confirmed(final_label, original_prelabel)
            else:
                selected = selection_rows.get(row_key)
                if selected is None:
                    raise BallNoCleanJudge(
                        f"reviewed row lacks original prelabel/package lineage: {row_key}"
                    )
                lineage_origin = str(selected["_lineage_origin"])
                original_prelabel = _selection_prelabel(selected, dimensions=dimensions)
                confirmed = _package_label_confirmed(final_label, original_prelabel)
            lineage_class = "confirmed_prelabel" if confirmed else "corrected_prelabel"
            origins[lineage_origin] += 1
            rows.append(
                {
                    "row_key": row_key,
                    "clip_id": clip_id,
                    "frame_index": frame_index,
                    "source_id": source_id,
                    "parent_source_id": source_id,
                    "source_class": None,
                    "split": None,
                    "lineage_class": lineage_class,
                    "lineage_origin": lineage_origin,
                    "original_prelabel": original_prelabel,
                    "final_label": final_label,
                    "training_weight": (
                        confirmed_prelabel_weight if confirmed else 1.0
                    ),
                    "ground_truth": not confirmed,
                    "teacher_derived": confirmed,
                    "evaluation_eligible": False,
                    "video_path": str(
                        ROOT
                        / "data/online_harvest_20260706/rallies"
                        / source_id
                        / f"{clip_id}.mp4"
                    ),
                }
            )
    _assert_count(len(rows), expected_reviewed_count, "reviewed lineage row count")
    source_counts = Counter(str(row["source_id"]) for row in rows)
    return sorted(rows, key=_row_sort_key), {
        "reviewed_root": str(reviewed_root),
        "reviewed_ingest_report": str(ingest_report),
        "reviewed_ingest_report_sha256": report_sha,
        "reviewed_row_count": len(rows),
        "reviewed_per_source": dict(sorted(source_counts.items())),
        "lineage_origin_counts": dict(sorted(origins.items())),
        "selection_manifests": [str(path) for path in selection_manifests],
        **reviewed_manifest_contract,
        **lineage_input_contract,
    }


def _load_scratch_package(
    package_path: Path,
    *,
    expected_scratch_count: int,
    expected_scratch_package_sha256: str | None,
    expected_image_zip_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    package_sha = _check_optional_sha256(
        package_path,
        expected_scratch_package_sha256,
        label="scratch package manifest",
    )
    package = _load_json(package_path)
    if str(package.get("labeling_mode")) != "scratch":
        raise BallNoCleanJudge("scratch package labeling_mode must be scratch")
    if package.get("prelabels_present") is not False:
        raise BallNoCleanJudge("scratch package must explicitly set prelabels_present=false")
    if package.get("prelabel_zip") is not None:
        raise BallNoCleanJudge("scratch package prelabel_zip must be null or absent")
    sessions = list(package.get("ball_sessions") or [])
    if len(sessions) != 1:
        raise BallNoCleanJudge(f"scratch package must contain one ball session, got {len(sessions)}")
    session = sessions[0]
    if session.get("prelabels_present") is not False:
        raise BallNoCleanJudge("scratch session must explicitly set prelabels_present=false")
    if session.get("prelabel_zip") is not None:
        raise BallNoCleanJudge("scratch session prelabel_zip must be null or absent")
    if str(session.get("provenance_class")) != "scratch":
        raise BallNoCleanJudge("scratch session provenance_class must be scratch")
    package_task_name = str(package.get("task_name") or "")
    session_task_name = str(session.get("task_name") or package_task_name)
    if not package_task_name or session_task_name != package_task_name:
        raise BallNoCleanJudge(
            "scratch package/session task_name must be present and identical"
        )

    sampling_path = _resolve_reference(str(package.get("sampling_manifest") or ""), package_path)
    sampling_md5 = _md5_file(sampling_path)
    expected_sampling_md5 = str(package.get("sampling_manifest_md5") or "")
    if sampling_md5 != expected_sampling_md5:
        raise BallNoCleanJudge(
            f"scratch sampling manifest MD5 mismatch: {sampling_md5} != {expected_sampling_md5}"
        )
    sampling = _load_json(sampling_path)
    if str(sampling.get("labeling_mode")) != "scratch":
        raise BallNoCleanJudge("sampling manifest labeling_mode must be scratch")
    if str(sampling.get("provenance_class")) != "scratch":
        raise BallNoCleanJudge("sampling manifest provenance_class must be scratch")
    raw_rows = list(sampling.get("frames") or [])
    _assert_count(len(raw_rows), expected_scratch_count, "scratch sampling row count")
    if int(session.get("frame_count") or 0) != expected_scratch_count:
        raise BallNoCleanJudge(
            f"scratch session frame_count {session.get('frame_count')} != {expected_scratch_count}"
        )

    image_zip = _resolve_reference(str(session.get("image_zip") or ""), package_path)
    image_zip_sha256 = _check_required_sha256(
        image_zip,
        expected_image_zip_sha256 or "",
        label="scratch image ZIP",
    )
    for owner_label, owner in (
        ("scratch package", package),
        ("scratch session", session),
        ("scratch sampling manifest", sampling),
    ):
        _validate_declared_sha256(
            owner,
            field="image_zip_sha256",
            expected=image_zip_sha256,
            label=owner_label,
        )
    package_rows: list[dict[str, Any]] = []
    candidate_hashes: list[dict[str, Any]] = []
    entry_sha256: dict[str, str] = {}
    seen_keys: set[str] = set()
    seen_names: set[str] = set()
    seen_members: set[str] = set()
    seen_ordinals: set[int] = set()
    with zipfile.ZipFile(image_zip) as archive:
        archive_names = {name for name in archive.namelist() if not name.endswith("/")}
        for raw in sorted(raw_rows, key=lambda item: int(item.get("sample_ordinal", 0))):
            source_id = str(raw.get("source_id") or "")
            clip_id = str(raw.get("rally_id") or raw.get("clip_id") or "")
            frame_index = int(raw.get("frame_index"))
            row_key = str(raw.get("row_key") or f"{clip_id}:{frame_index:06d}")
            image_name = str(raw.get("image_name") or "")
            image_member = str(raw.get("image_zip_member") or image_name)
            sample_ordinal = int(raw.get("sample_ordinal", 0))
            if not source_id or not clip_id:
                raise BallNoCleanJudge("scratch row lacks source_id or clip_id")
            if str(raw.get("provenance_class")) != "scratch":
                raise BallNoCleanJudge(
                    f"scratch row provenance_class must be scratch: {row_key}"
                )
            expected_row_key = f"{clip_id}:{frame_index:06d}"
            if row_key != expected_row_key:
                raise BallNoCleanJudge(
                    f"scratch row key lineage mismatch: {row_key} != {expected_row_key}"
                )
            parsed = _parse_scratch_image_name(image_name)
            if parsed != (source_id, clip_id, frame_index):
                raise BallNoCleanJudge(
                    f"scratch image name lineage mismatch for {image_name}: "
                    f"parsed={parsed} manifest={(source_id, clip_id, frame_index)}"
                )
            if (
                row_key in seen_keys
                or image_name in seen_names
                or image_member in seen_members
                or sample_ordinal in seen_ordinals
            ):
                raise BallNoCleanJudge(f"duplicate scratch row or image: {row_key} {image_name}")
            seen_keys.add(row_key)
            seen_names.add(image_name)
            seen_members.add(image_member)
            seen_ordinals.add(sample_ordinal)
            if image_member not in archive_names:
                raise BallNoCleanJudge(f"scratch image ZIP missing member: {image_member}")
            image_bytes = archive.read(image_member)
            actual_md5 = hashlib.md5(image_bytes).hexdigest()
            actual_sha256 = hashlib.sha256(image_bytes).hexdigest()
            if actual_md5 != str(raw.get("image_md5") or ""):
                raise BallNoCleanJudge(
                    f"scratch image MD5 mismatch for {image_name}: {actual_md5}"
                )
            image = _decode_image(image_bytes, label=image_name)
            entry_sha256[image_member] = actual_sha256
            image_height, image_width = image.shape[:2]
            for field, actual in (
                ("image_width", image_width),
                ("image_height", image_height),
                ("expected_width", image_width),
                ("expected_height", image_height),
            ):
                declared = raw.get(field)
                if declared is not None and int(declared) != actual:
                    raise BallNoCleanJudge(
                        f"scratch image dimension mismatch for {image_name}: "
                        f"{field}={declared} decoded={actual}"
                    )
            candidate_hashes.append(_image_hash_record(image, name=image_name, encoded=image_bytes))
            package_rows.append(
                {
                    "row_key": row_key,
                    "clip_id": clip_id,
                    "frame_index": frame_index,
                    "source_id": source_id,
                    "parent_source_id": source_id,
                    "source_class": raw.get("source_class"),
                    "image_name": image_name,
                    "image_zip": str(image_zip),
                    "image_zip_member": image_member,
                    "image_md5": actual_md5,
                    "image_sha256": actual_sha256,
                    "sample_ordinal": sample_ordinal,
                    "image_width": image_width,
                    "image_height": image_height,
                }
            )
        if archive_names != seen_members:
            raise BallNoCleanJudge(
                "scratch image ZIP members do not reconcile exactly: "
                f"expected={len(seen_members)} actual={len(archive_names)}"
            )
    _assert_count(len(package_rows), expected_scratch_count, "scratch package image count")
    if seen_ordinals != set(range(expected_scratch_count)):
        raise BallNoCleanJudge("scratch sample ordinals must be exactly 0..N-1")
    actual_source_counts = dict(
        sorted(Counter(str(row["source_id"]) for row in package_rows).items())
    )
    session_source_counts = {
        str(key): int(value) for key, value in dict(session.get("source_counts") or {}).items()
    }
    sampling_source_counts = {
        str(key): int(value)
        for key, value in dict(sampling.get("per_source_distribution") or {}).items()
    }
    if session_source_counts != actual_source_counts:
        raise BallNoCleanJudge(
            "scratch session source counts do not reconcile: "
            f"declared={session_source_counts} actual={actual_source_counts}"
        )
    if sampling_source_counts != actual_source_counts:
        raise BallNoCleanJudge(
            "scratch sampling source counts do not reconcile: "
            f"declared={sampling_source_counts} actual={actual_source_counts}"
        )
    return package_rows, {
        "scratch_package": str(package_path),
        "scratch_package_sha256": package_sha,
        "scratch_sampling_manifest": str(sampling_path),
        "scratch_sampling_manifest_md5": sampling_md5,
        "scratch_image_zip": str(image_zip),
        "image_zip_sha256": image_zip_sha256,
        "image_zip_entry_sha256": dict(sorted(entry_sha256.items())),
        "scratch_task_name": package_task_name,
        "scratch_package_image_count": len(package_rows),
    }, candidate_hashes


def _parse_scratch_export(
    export_path: Path,
    *,
    package_rows: Sequence[Mapping[str, Any]],
    expected_scratch_count: int,
    negative_attestation: Path | None,
    scratch_import_ledger: Path,
    expected_export_sha256: str,
    expected_import_ledger_sha256: str,
    expected_task_fingerprint: str,
    expected_task_name: str,
    expected_image_zip: Path,
    expected_image_zip_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    export_sha = _check_required_sha256(
        export_path, expected_export_sha256, label="scratch CVAT export"
    )
    attestation = _inspect_negative_attestation(
        negative_attestation,
        expected_task_id=EXPECTED_SCRATCH_TASK_ID,
        expected_export_sha256=export_sha,
    )
    with zipfile.ZipFile(export_path) as archive:
        try:
            raw_xml = archive.read("annotations.xml")
        except KeyError as exc:
            raise BallNoCleanJudge("scratch export is missing annotations.xml") from exc
    root = ElementTree.fromstring(raw_xml)
    if _text(root.find("version")) != "1.1":
        raise BallNoCleanJudge("scratch export must be CVAT images version 1.1")
    job = root.find("./meta/job")
    if job is None:
        raise BallNoCleanJudge("scratch export is missing meta/job provenance")
    raw_export_job_id = _text(job.find("id"))
    export_job_id = (
        None
        if not raw_export_job_id
        else _strict_positive_int(raw_export_job_id, label="scratch export job_id")
    )
    if export_job_id is not None and export_job_id != EXPECTED_SCRATCH_JOB_ID:
        raise BallNoCleanJudge(
            f"scratch export job_id {export_job_id} != {EXPECTED_SCRATCH_JOB_ID}"
        )
    if _text(job.find("mode")) != "annotation":
        raise BallNoCleanJudge("scratch export job mode must be annotation")
    provenance_contract = _validate_scratch_task_provenance(
        scratch_import_ledger,
        expected_sha256=expected_import_ledger_sha256,
        expected_task_id=EXPECTED_SCRATCH_TASK_ID,
        expected_task_name=expected_task_name,
        expected_task_fingerprint=expected_task_fingerprint,
        expected_frame_count=expected_scratch_count,
        expected_image_zip=expected_image_zip,
        expected_image_zip_sha256=expected_image_zip_sha256,
        export_job_id=export_job_id,
    )
    expected_by_name = {str(row["image_name"]): row for row in package_rows}
    export_names: set[str] = set()
    rows: list[dict[str, Any]] = []
    for image_node in root.findall("image"):
        raw_name = str(image_node.attrib.get("name") or "")
        image_name = PurePosixPath(raw_name).name
        if image_name in export_names:
            raise BallNoCleanJudge(f"duplicate image in scratch export: {image_name}")
        export_names.add(image_name)
        package_row = expected_by_name.get(image_name)
        if package_row is None:
            raise BallNoCleanJudge(f"scratch export image not in package: {image_name}")
        try:
            image_width = int(image_node.attrib["width"])
            image_height = int(image_node.attrib["height"])
        except (KeyError, ValueError) as exc:
            raise BallNoCleanJudge(
                f"scratch export image has invalid dimensions: {image_name}"
            ) from exc
        if (
            image_width != int(package_row["image_width"])
            or image_height != int(package_row["image_height"])
        ):
            raise BallNoCleanJudge(
                f"scratch export image dimensions disagree with package: {image_name}"
            )
        ball_boxes = [
            node
            for node in image_node.findall("box")
            if str(node.attrib.get("label") or "").strip().lower() == "ball"
        ]
        unexpected_boxes = [
            node
            for node in image_node.findall("box")
            if str(node.attrib.get("label") or "").strip().lower() != "ball"
        ]
        if unexpected_boxes:
            raise BallNoCleanJudge(f"scratch export contains non-ball boxes for {image_name}")
        if len(ball_boxes) > 1:
            raise BallNoCleanJudge(f"scratch export contains multiple ball boxes for {image_name}")
        if ball_boxes:
            node = ball_boxes[0]
            box_source = str(node.attrib.get("source") or "").strip().lower()
            if box_source != "manual":
                raise BallNoCleanJudge(
                    f"scratch export ball box source must be manual for {image_name}; "
                    f"got {box_source or '<missing>'}"
                )
            attrs = {
                str(attr.attrib.get("name") or ""): _text(attr)
                for attr in node.findall("attribute")
            }
            visibility = str(attrs.get("visibility_level") or "clear")
            if visibility not in VALID_VISIBILITY - {"none"}:
                raise BallNoCleanJudge(
                    f"invalid visibility_level {visibility!r} for {image_name}"
                )
            bbox = [float(node.attrib[name]) for name in ("xtl", "ytl", "xbr", "ybr")]
            if (
                not all(math.isfinite(value) for value in bbox)
                or not 0.0 <= bbox[0] < bbox[2] <= float(image_width)
                or not 0.0 <= bbox[1] < bbox[3] <= float(image_height)
            ):
                raise BallNoCleanJudge(
                    f"scratch export contains invalid ball box for {image_name}: {bbox}"
                )
            final_label = {
                "ball_present": True,
                "bbox_xyxy": bbox,
                "visibility_level": visibility,
            }
        else:
            final_label = {
                "ball_present": False,
                "bbox_xyxy": None,
                "visibility_level": "none",
            }
        rows.append(
            {
                **dict(package_row),
                "split": None,
                "lineage_class": "scratch",
                "lineage_origin": "scratch_no_prelabel_package",
                "original_prelabel": None,
                "final_label": final_label,
                "training_weight": 1.0,
                "ground_truth": bool(ball_boxes) or bool(attestation["valid"]),
                "teacher_derived": False,
                "evaluation_eligible": bool(ball_boxes) or bool(attestation["valid"]),
                "negative_attestation_status": (
                    "NOT_APPLICABLE_POSITIVE"
                    if ball_boxes
                    else "OWNER_ATTESTED_NEGATIVE"
                    if attestation["valid"]
                    else "UNATTESTED_NEGATIVE"
                ),
            }
        )
    missing = sorted(set(expected_by_name) - export_names)
    if missing:
        raise BallNoCleanJudge(
            f"scratch export is missing {len(missing)} package images: {missing[:5]}"
        )
    _assert_count(len(rows), expected_scratch_count, "scratch export image count")
    metadata_size = _cvat_images_metadata_size(root)
    if metadata_size is not None and metadata_size != expected_scratch_count:
        raise BallNoCleanJudge(
            f"scratch export metadata size {metadata_size} != {expected_scratch_count}"
        )
    if not attestation["valid"]:
        boxless_count = sum(not row["final_label"]["ball_present"] for row in rows)
        raise BallNoCleanJudge(
            "AWAITING_ATTESTATION: "
            f"{attestation['reason']}; {boxless_count} boxless rows are "
            "UNATTESTED_NEGATIVE and evaluation_eligible=false"
        )
    return sorted(rows, key=_row_sort_key), {
        "scratch_export": str(export_path),
        "scratch_export_sha256": export_sha,
        "scratch_export_image_count": len(rows),
        "negative_attestation": str(negative_attestation),
        "negative_attestation_sha256": attestation["sha256"],
        "negative_attested_by": attestation["attested_by"],
        "negative_attested_utc": attestation["attested_utc"],
        **provenance_contract,
    }


def _validate_scratch_task_provenance(
    ledger_path: Path,
    *,
    expected_sha256: str,
    expected_task_id: int,
    expected_task_name: str,
    expected_task_fingerprint: str,
    expected_frame_count: int,
    expected_image_zip: Path,
    expected_image_zip_sha256: str | None = None,
    export_job_id: int | None = None,
) -> dict[str, Any]:
    ledger_sha = _check_required_sha256(
        ledger_path, expected_sha256, label="scratch task import ledger"
    )
    ledger = _load_json(ledger_path)
    if str(ledger.get("status") or "") != "imported":
        raise BallNoCleanJudge("scratch task import ledger status must be imported")
    tasks = list(ledger.get("tasks") or [])
    if len(tasks) != 1 or not isinstance(tasks[0], Mapping):
        raise BallNoCleanJudge("scratch task import ledger must contain exactly one task")
    task = dict(tasks[0])
    image_zip_sha256 = _check_required_sha256(
        expected_image_zip,
        expected_image_zip_sha256 or "",
        label="scratch image ZIP",
    )
    for owner_label, owner in (
        ("scratch task import ledger", ledger),
        ("scratch task import ledger row", task),
    ):
        _validate_declared_sha256(
            owner,
            field="image_zip_sha256",
            expected=image_zip_sha256,
            label=owner_label,
        )
    if int(task.get("task_id") or -1) != expected_task_id:
        raise BallNoCleanJudge(
            f"scratch task import ledger task_id must be {expected_task_id}"
        )
    if str(task.get("task_name") or "") != expected_task_name:
        raise BallNoCleanJudge("scratch task import ledger task_name mismatch")
    if str(task.get("task_fingerprint") or "") != expected_task_fingerprint:
        raise BallNoCleanJudge("scratch task fingerprint mismatch")
    if int(task.get("frame_count") or -1) != expected_frame_count:
        raise BallNoCleanJudge("scratch task import ledger frame_count mismatch")
    if str(task.get("status") or "") != "imported":
        raise BallNoCleanJudge("scratch task ledger row status must be imported")
    if str(task.get("kind") or "") != "ball":
        raise BallNoCleanJudge("scratch task ledger kind must be ball")
    if task.get("prelabel_zip") is not None:
        raise BallNoCleanJudge("scratch task import ledger prelabel_zip must be null or absent")
    ledger_image_zip = Path(str(task.get("image_zip") or ""))
    if not ledger_image_zip.is_absolute():
        ledger_image_zip = ROOT / ledger_image_zip
    if ledger_image_zip.resolve() != expected_image_zip.resolve():
        raise BallNoCleanJudge("scratch task import ledger image_zip mismatch")
    ledger_job_ids: list[int] = []
    for owner_label, owner in (
        ("scratch task import ledger", ledger),
        ("scratch task import ledger row", task),
    ):
        if "job_id" not in owner:
            continue
        ledger_job_id = _strict_positive_int(owner["job_id"], label=f"{owner_label} job_id")
        if ledger_job_id != EXPECTED_SCRATCH_JOB_ID:
            raise BallNoCleanJudge(
                f"{owner_label} job_id {ledger_job_id} != {EXPECTED_SCRATCH_JOB_ID}"
            )
        ledger_job_ids.append(ledger_job_id)
    if len(set(ledger_job_ids)) > 1:
        raise BallNoCleanJudge("scratch task import ledger job_id values disagree")
    ledger_job_id = ledger_job_ids[0] if ledger_job_ids else None
    if export_job_id is not None and ledger_job_id is not None and export_job_id != ledger_job_id:
        raise BallNoCleanJudge(
            f"scratch export/import ledger job_id mismatch: {export_job_id} != {ledger_job_id}"
        )
    bound_job_id = export_job_id if export_job_id is not None else ledger_job_id
    if export_job_id is not None and ledger_job_id is not None:
        job_id_binding = "EXPORT_XML_AND_IMPORT_LEDGER"
        residual_assumptions: list[str] = []
    elif export_job_id is not None:
        job_id_binding = "EXPORT_XML_ONLY"
        residual_assumptions = [TASK_87_JOB_ID_RESIDUAL]
    elif ledger_job_id is not None:
        job_id_binding = "IMPORT_LEDGER_ONLY"
        residual_assumptions = [
            "the historical import ledger records job 87, but the CVAT export carries no "
            "independent job-id binding"
        ]
    else:
        job_id_binding = "UNAVAILABLE_IN_ARTIFACTS"
        residual_assumptions = [TASK_87_JOB_ID_RESIDUAL]
    return {
        "scratch_import_ledger": str(ledger_path),
        "scratch_import_ledger_sha256": ledger_sha,
        "scratch_task_id": expected_task_id,
        "scratch_task_name": expected_task_name,
        "scratch_task_fingerprint": expected_task_fingerprint,
        "image_zip_sha256": image_zip_sha256,
        "job_id": bound_job_id,
        "job_id_binding": job_id_binding,
        "residual_assumptions": residual_assumptions,
        "scratch_task_prelabel_zip": None,
        "scratch_task_box_source_required": "manual",
    }


def _verify_materialized_scratch_image_bytes(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_image_zip: Path,
    expected_image_zip_sha256: str,
    expected_entry_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Re-read every scratch train/judge image and enforce its staged entry digest."""

    image_zip_sha256 = _check_required_sha256(
        expected_image_zip,
        expected_image_zip_sha256,
        label="materialized scratch image ZIP",
    )
    expected_map = {str(member): str(digest) for member, digest in expected_entry_sha256.items()}
    if not expected_map:
        raise BallNoCleanJudge("scratch image ZIP entry SHA-256 map must be nonempty")
    for member, digest in expected_map.items():
        if not member or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise BallNoCleanJudge(
                f"scratch image ZIP entry SHA-256 map is invalid for {member or '<missing>'}"
            )

    seen_members: set[str] = set()
    with zipfile.ZipFile(expected_image_zip) as archive:
        archive_members = {name for name in archive.namelist() if not name.endswith("/")}
        if archive_members != set(expected_map):
            raise BallNoCleanJudge(
                "materialized scratch image ZIP members disagree with the entry digest map"
            )
        for row in rows:
            if str(row.get("lineage_class") or "") != "scratch":
                raise BallNoCleanJudge(
                    f"non-scratch row supplied to staged image verification: {row.get('row_key')}"
                )
            row_zip = Path(str(row.get("image_zip") or ""))
            if row_zip.resolve() != expected_image_zip.resolve():
                raise BallNoCleanJudge(
                    f"materialized scratch row image_zip mismatch: {row.get('row_key')}"
                )
            member = str(row.get("image_zip_member") or "")
            if not member or member in seen_members:
                raise BallNoCleanJudge(
                    f"materialized scratch row has missing/duplicate ZIP member: {member or '<missing>'}"
                )
            seen_members.add(member)
            expected_digest = expected_map.get(member)
            if expected_digest is None:
                raise BallNoCleanJudge(
                    f"materialized scratch row lacks staged ZIP entry digest: {member}"
                )
            row_digest = row.get("image_sha256")
            if row_digest != expected_digest:
                raise BallNoCleanJudge(
                    f"materialized scratch row SHA-256 mismatch for {member}: "
                    f"{row_digest} != {expected_digest}"
                )
            actual_digest = hashlib.sha256(archive.read(member)).hexdigest()
            if actual_digest != expected_digest:
                raise BallNoCleanJudge(
                    f"materialized scratch image byte SHA-256 mismatch for {member}: "
                    f"{actual_digest} != {expected_digest}"
                )
    if seen_members != set(expected_map):
        raise BallNoCleanJudge(
            "materialized scratch train/judge rows do not cover the staged ZIP entry digest map: "
            f"rows={len(seen_members)} entries={len(expected_map)}"
        )
    return {
        "verdict": "PASS",
        "digest": "sha256",
        "image_count": len(seen_members),
        "image_zip_sha256": image_zip_sha256,
    }


def _inspect_negative_attestation(
    path: Path | None,
    *,
    expected_task_id: int,
    expected_export_sha256: str,
) -> dict[str, Any]:
    invalid = {
        "valid": False,
        "reason": "--negative-attestation is required",
        "sha256": None,
        "attested_by": None,
        "attested_utc": None,
    }
    if path is None:
        return invalid
    if not path.is_file():
        return {**invalid, "reason": f"negative attestation file is missing: {path}"}
    try:
        payload = _load_json(path)
    except (BallNoCleanJudge, json.JSONDecodeError) as exc:
        return {**invalid, "reason": f"negative attestation is invalid JSON: {exc}"}
    sha256 = _sha256_file(path)
    if type(payload.get("task_id")) is not int or payload["task_id"] != expected_task_id:
        return {
            **invalid,
            "reason": f"negative attestation task_id must be {expected_task_id}",
            "sha256": sha256,
        }
    if payload.get("export_sha256") != expected_export_sha256:
        return {
            **invalid,
            "reason": "negative attestation export_sha256 mismatch",
            "sha256": sha256,
        }
    if not isinstance(payload.get("statement"), str) or not payload["statement"].strip():
        return {**invalid, "reason": "negative attestation statement is required", "sha256": sha256}
    if not isinstance(payload.get("attested_by"), str) or not payload["attested_by"].strip():
        return {**invalid, "reason": "negative attestation attested_by is required", "sha256": sha256}
    raw_utc = payload.get("attested_utc")
    if not isinstance(raw_utc, str) or not raw_utc.strip():
        return {**invalid, "reason": "negative attestation attested_utc is required", "sha256": sha256}
    try:
        parsed_utc = datetime.fromisoformat(raw_utc.replace("Z", "+00:00"))
    except ValueError:
        return {**invalid, "reason": "negative attestation attested_utc is invalid", "sha256": sha256}
    if parsed_utc.tzinfo is None or parsed_utc.utcoffset() != timezone.utc.utcoffset(parsed_utc):
        return {**invalid, "reason": "negative attestation attested_utc must be UTC", "sha256": sha256}
    if payload.get("all_frames_inspected") is not True:
        return {**invalid, "reason": "negative attestation all_frames_inspected must be true", "sha256": sha256}
    if payload.get("boxless_means_no_ball") is not True:
        return {**invalid, "reason": "negative attestation boxless_means_no_ball must be true", "sha256": sha256}
    return {
        "valid": True,
        "reason": None,
        "sha256": sha256,
        "attested_by": payload["attested_by"].strip(),
        "attested_utc": raw_utc,
    }


def _assert_no_protected_collisions(
    candidate_hashes: Sequence[Mapping[str, Any]],
    *,
    protected_videos: Sequence[Path],
    protected_additions: Sequence[Path],
    threshold: int,
    production_mode: bool = False,
) -> dict[str, Any]:
    canonical_verified = False
    if production_mode:
        supplied = [*protected_videos, *protected_additions]
        expected = [*DEFAULT_PROTECTED_VIDEOS, *DEFAULT_PROTECTED_ADDITIONS]
        supplied_resolved = [path.resolve() for path in supplied]
        expected_resolved = [path.resolve() for path in expected]
        if len(supplied_resolved) != len(expected_resolved) or set(supplied_resolved) != set(
            expected_resolved
        ):
            raise BallNoCleanJudge(
                "production mode requires the exact canonical four protected videos "
                "and two protected additions"
            )
        for path, expected_sha in CANONICAL_PROTECTED_SHA256.items():
            _check_required_sha256(path, expected_sha, label="canonical protected input")
        canonical_verified = True
    protected: list[dict[str, Any]] = []
    frame_counts: dict[str, int] = {}
    for video_path in protected_videos:
        video_hashes = _hash_video_all_frames(video_path)
        frame_counts[str(video_path)] = len(video_hashes)
        protected.extend(video_hashes)
    addition_counts: dict[str, int] = {}
    for image_path in protected_additions:
        if not image_path.is_file():
            raise BallNoCleanJudge(f"missing protected court-keypoint addition: {image_path}")
        encoded = image_path.read_bytes()
        image = _decode_image(encoded, label=str(image_path))
        protected.append(_image_hash_record(image, name=str(image_path), encoded=encoded))
        addition_counts[str(image_path)] = 1
    if not protected:
        raise BallNoCleanJudge("protected collision guard has zero protected frames")

    collisions: list[dict[str, Any]] = []
    for candidate in candidate_hashes:
        for guard in protected:
            distance = int(candidate["dhash"] ^ guard["dhash"]).bit_count()
            exact = candidate["raw_sha256"] == guard["raw_sha256"]
            if exact or distance <= threshold:
                collisions.append(
                    {
                        "candidate": candidate["name"],
                        "protected": guard["name"],
                        "exact_decoded_pixels": exact,
                        "dhash_hamming_distance": distance,
                    }
                )
    if collisions:
        first = collisions[0]
        raise BallNoCleanJudge(
            "protected image collision: "
            f"{first['candidate']} vs {first['protected']} "
            f"hamming={first['dhash_hamming_distance']}"
        )
    return {
        "hash_type": "decoded_pixel_sha256_plus_dhash_8x8_64bit",
        "collision_hamming_threshold": threshold,
        "candidate_image_count": len(candidate_hashes),
        "protected_video_frame_counts": frame_counts,
        "protected_addition_counts": addition_counts,
        "protected_video_count": len(protected_videos),
        "protected_addition_count": len(protected_additions),
        "protected_frame_count": len(protected),
        "collision_count": 0,
        "production_mode": production_mode,
        "canonical_protected_set_verified": canonical_verified,
    }


def _hash_video_all_frames(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BallNoCleanJudge(f"missing protected video: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise BallNoCleanJudge(f"cannot open protected video: {path}")
    expected = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    rows: list[dict[str, Any]] = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rows.append(_image_hash_record(frame, name=f"{path}#frame={frame_index}"))
        frame_index += 1
    cap.release()
    if expected > 0 and len(rows) != expected:
        raise BallNoCleanJudge(
            f"protected video decoded {len(rows)}/{expected} frames: {path}"
        )
    if not rows:
        raise BallNoCleanJudge(f"protected video decoded zero frames: {path}")
    return rows


def _verify_reviewed_corpus_manifest(
    reviewed_root: Path, ingest_report: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = ingest_report.get("md5_manifest")
    if not isinstance(manifest, Mapping):
        raise BallNoCleanJudge("reviewed ingest report lacks embedded md5_manifest")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise BallNoCleanJudge("reviewed ingest md5_manifest contains no files")
    expected: dict[Path, tuple[str, int]] = {}
    for raw in raw_files:
        if not isinstance(raw, Mapping):
            raise BallNoCleanJudge("reviewed ingest md5_manifest has invalid file row")
        relative = Path(str(raw.get("relative_path") or ""))
        if not relative.parts or relative.parts[0] != reviewed_root.name:
            raise BallNoCleanJudge(
                f"reviewed manifest path is outside {reviewed_root.name}: {relative}"
            )
        path = reviewed_root.parent / relative
        if path in expected:
            raise BallNoCleanJudge(f"duplicate reviewed manifest path: {relative}")
        expected[path] = (str(raw.get("md5") or ""), int(raw.get("bytes") or -1))
    actual = set(reviewed_root.glob("*/reviewed_boxes.json"))
    if set(expected) != actual:
        missing = sorted(str(path) for path in set(expected) - actual)
        extra = sorted(str(path) for path in actual - set(expected))
        raise BallNoCleanJudge(
            f"reviewed corpus file inventory mismatch: missing={missing[:3]} extra={extra[:3]}"
        )
    for path, (expected_md5, expected_bytes) in expected.items():
        actual_bytes = path.stat().st_size
        actual_md5 = _md5_file(path)
        if actual_bytes != expected_bytes or actual_md5 != expected_md5:
            raise BallNoCleanJudge(
                f"reviewed corpus content mismatch for {path}: "
                f"bytes={actual_bytes}/{expected_bytes} md5={actual_md5}/{expected_md5}"
            )
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "reviewed_corpus_manifest_verified": True,
        "reviewed_corpus_verified_file_count": len(expected),
        "reviewed_corpus_manifest_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _lineage_input_records(
    *,
    selection_manifests: Sequence[Path],
    legacy_reviewed_root: Path | None,
    legacy_prelabel_root: Path | None,
) -> list[dict[str, Any]]:
    inputs: list[tuple[str, str, Path]] = []
    for path in selection_manifests:
        inputs.append(("selection_manifest", _lineage_logical_path(path, None), path))
    if legacy_reviewed_root is not None:
        if not legacy_reviewed_root.is_dir():
            raise BallNoCleanJudge(f"missing legacy reviewed root: {legacy_reviewed_root}")
        annotations = sorted(legacy_reviewed_root.glob("*/annotations.xml"))
        if not annotations:
            raise BallNoCleanJudge(
                f"legacy reviewed root contains no annotations.xml: {legacy_reviewed_root}"
            )
        if legacy_prelabel_root is None:
            raise BallNoCleanJudge("legacy prelabel root is required with legacy review exports")
        for path in annotations:
            clip_id = path.parent.name
            inputs.append(
                (
                    "legacy_review_export",
                    _lineage_logical_path(path, legacy_reviewed_root),
                    path,
                )
            )
            prelabel = legacy_prelabel_root / clip_id / "ball_track.json"
            inputs.append(
                (
                    "legacy_prelabel_track",
                    _lineage_logical_path(prelabel, legacy_prelabel_root),
                    prelabel,
                )
            )
    records: list[dict[str, Any]] = []
    for role, logical_path, path in inputs:
        if not path.is_file():
            raise BallNoCleanJudge(f"missing {role}: {path}")
        records.append(
            {
                "role": role,
                "logical_path": logical_path,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return sorted(
        records,
        key=lambda row: (str(row["role"]), str(row["logical_path"]), str(row["sha256"])),
    )


def _lineage_inputs_digest(
    *,
    selection_manifests: Sequence[Path],
    legacy_reviewed_root: Path | None,
    legacy_prelabel_root: Path | None,
) -> tuple[str, list[dict[str, Any]]]:
    records = _lineage_input_records(
        selection_manifests=selection_manifests,
        legacy_reviewed_root=legacy_reviewed_root,
        legacy_prelabel_root=legacy_prelabel_root,
    )
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), records


def _verify_lineage_input_digest(
    *,
    selection_manifests: Sequence[Path],
    legacy_reviewed_root: Path | None,
    legacy_prelabel_root: Path | None,
    expected_sha256: str,
) -> dict[str, Any]:
    actual, records = _lineage_inputs_digest(
        selection_manifests=selection_manifests,
        legacy_reviewed_root=legacy_reviewed_root,
        legacy_prelabel_root=legacy_prelabel_root,
    )
    if actual != expected_sha256:
        raise BallNoCleanJudge(
            f"lineage inputs SHA-256 mismatch: {actual} != {expected_sha256}"
        )
    return {
        "lineage_inputs_sha256": actual,
        "lineage_input_file_count": len(records),
        "lineage_input_files": records,
    }


def _lineage_logical_path(path: Path, base: Path | None) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        if base is not None:
            return path.resolve().relative_to(base.resolve()).as_posix()
        return f"{path.parent.name}/{path.name}"


def _load_selection_rows(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = _load_json(path)
        for session in payload.get("sessions") or []:
            session_id = str(session.get("session_id") or "unknown_session")
            for raw in session.get("frames") or []:
                row = dict(raw)
                clip_id = str(row.get("clip_id") or "")
                frame_index = int(row.get("frame_index"))
                key = f"{clip_id}:{frame_index:06d}"
                if key in result:
                    raise BallNoCleanJudge(
                        f"duplicate package prelabel lineage for {key}: {path}"
                    )
                row["_lineage_origin"] = f"{path}:{session_id}"
                result[key] = row
    return result


def _legacy_reviewed_keys(root: Path | None) -> set[str]:
    if root is None:
        return set()
    if not root.is_dir():
        raise BallNoCleanJudge(f"missing legacy reviewed root: {root}")
    result: set[str] = set()
    for annotations_path in sorted(root.glob("*/annotations.xml")):
        xml_root = ElementTree.fromstring(annotations_path.read_bytes())
        task = xml_root.find("./meta/task")
        if task is None:
            task = xml_root.find("./meta/project/tasks/task")
        if task is None:
            raise BallNoCleanJudge(
                f"legacy annotations missing meta/task or project task: {annotations_path}"
            )
        # Historical CVAT tasks used inconsistent display-name prefixes. The
        # reviewed-corpus importer bound identity to the export directory.
        clip_id = annotations_path.parent.name
        start = int(_text(task.find("start_frame")) or 0)
        stop = int(_text(task.find("stop_frame")) or 0)
        size = int(_text(task.find("size")) or 0)
        frame_filter = _text(task.find("frame_filter"))
        match = re.search(r"(?:^|[;&\s])step\s*=\s*(\d+)(?:$|[;&\s])", frame_filter)
        if match is None:
            raise BallNoCleanJudge(
                f"legacy annotations lack explicit frame_filter step: {annotations_path}"
            )
        step = int(match.group(1))
        reviewed = list(range(start, stop + 1, step))
        if size > 0:
            reviewed = reviewed[:size]
        result.update(f"{clip_id}:{frame_index:06d}" for frame_index in reviewed)
    if not result:
        raise BallNoCleanJudge(f"legacy reviewed root contains no review rows: {root}")
    return result


def _selection_prelabel(
    row: Mapping[str, Any], *, dimensions: tuple[int, int]
) -> dict[str, Any]:
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for name in ("teacher", "student"):
        prediction = row.get(name)
        if isinstance(prediction, Mapping) and prediction.get("visible"):
            candidates.append((name, prediction))
    if not candidates:
        raise BallNoCleanJudge(
            f"package selection row lacks visible prelabel: {row.get('clip_id')}:{row.get('frame_index')}"
        )
    source, prediction = max(
        candidates, key=lambda item: float(item[1].get("score", 0.0))
    )
    x, y = [float(value) for value in prediction["xy"]]
    width, height = dimensions
    bbox = [
        max(0.0, x - 8.0),
        max(0.0, y - 8.0),
        min(float(width), x + 8.0),
        min(float(height), y + 8.0),
    ]
    return {
        "ball_present": True,
        "bbox_xyxy": [round(value, 2) for value in bbox],
        "visibility_level": "clear",
        "proposal_source": source,
        "proposal_score": float(prediction.get("score", 0.0)),
        "disagreement_type": row.get("disagreement_type"),
    }


def _legacy_prelabel(prediction: Mapping[str, Any]) -> dict[str, Any]:
    visible = bool(prediction.get("visible"))
    xy = prediction.get("xy") if visible else None
    return {
        "ball_present": visible,
        "xy": None if xy is None else [float(xy[0]), float(xy[1])],
        "confidence": float(prediction.get("conf", 0.0)),
        "proposal_source": "wasb",
    }


def _package_label_confirmed(
    final_label: Mapping[str, Any], original: Mapping[str, Any]
) -> bool:
    if final_label["ball_present"] != original["ball_present"]:
        return False
    if not final_label["ball_present"]:
        return (
            final_label.get("visibility_level") == "none"
            and original.get("visibility_level") == "none"
        )
    final_bbox = final_label.get("bbox_xyxy")
    original_bbox = original.get("bbox_xyxy")
    if not isinstance(final_bbox, Sequence) or not isinstance(original_bbox, Sequence):
        return False
    geometry_equal = max(
        abs(float(left) - float(right))
        for left, right in zip(final_bbox, original_bbox, strict=True)
    ) <= 0.011
    return geometry_equal and final_label.get("visibility_level") == "clear"


def _legacy_label_confirmed(
    final_label: Mapping[str, Any], original: Mapping[str, Any]
) -> bool:
    if final_label["ball_present"] != original["ball_present"]:
        return False
    if not final_label["ball_present"]:
        # An absent model record is not evidence for the annotator's semantic
        # reason.  Only an explicit final `none` is an unchanged absence;
        # out_of_frame/full/partial are human corrections.
        return final_label.get("visibility_level") == "none"
    bbox = final_label.get("bbox_xyxy")
    xy = original.get("xy")
    if not isinstance(bbox, Sequence) or not isinstance(xy, Sequence):
        return False
    center = ((float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2)
    distance = math.hypot(center[0] - float(xy[0]), center[1] - float(xy[1]))
    return distance <= 0.011 and final_label.get("visibility_level") == "clear"


def _final_label(frame: Mapping[str, Any], *, row_key: str) -> dict[str, Any]:
    boxes = [
        box
        for box in frame.get("boxes") or []
        if str(box.get("label") or "").strip().lower() == "ball"
    ]
    if len(boxes) > 1:
        raise BallNoCleanJudge(f"reviewed row has multiple final ball boxes: {row_key}")
    frame_visibility = dict(frame.get("visibility_levels_by_label") or {})
    if not boxes:
        return {
            "ball_present": False,
            "bbox_xyxy": None,
            "visibility_level": str(frame_visibility.get("ball") or "none"),
        }
    box = boxes[0]
    return {
        "ball_present": True,
        "bbox_xyxy": [float(value) for value in box["bbox_xyxy"]],
        "visibility_level": str(
            box.get("visibility_level") or frame_visibility.get("ball") or "clear"
        ),
    }


def _lineage_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    classes = ("scratch", "corrected_prelabel", "confirmed_prelabel")
    totals = Counter(str(row["lineage_class"]) for row in rows)
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_source[str(row["source_id"])][str(row["lineage_class"])] += 1
    return {
        "totals": {name: int(totals[name]) for name in classes},
        "per_source": {
            source: {
                "total": sum(counts.values()),
                **{name: int(counts[name]) for name in classes},
            }
            for source, counts in sorted(by_source.items())
        },
    }


def _write_split_artifacts(
    out: Path,
    *,
    lineage_rows: Sequence[Mapping[str, Any]],
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    train_payload = []
    for row in train_rows:
        payload = dict(row)
        payload["split"] = "train"
        train_payload.append(payload)
    validation_payload = []
    for row in validation_rows:
        payload = dict(row)
        payload["split"] = "validation"
        validation_payload.append(payload)
    lineage_payload = []
    train_keys = {str(row["row_key"]) for row in train_rows}
    validation_keys = {str(row["row_key"]) for row in validation_rows}
    for row in lineage_rows:
        payload = dict(row)
        key = str(row["row_key"])
        payload["split"] = (
            "train" if key in train_keys else "validation" if key in validation_keys else "excluded_holdout_old"
        )
        lineage_payload.append(payload)
    _write_jsonl(out / "lineage_rows.jsonl", sorted(lineage_payload, key=_row_sort_key))
    _write_jsonl(out / "train.jsonl", sorted(train_payload, key=_row_sort_key))
    _write_jsonl(out / "validation.jsonl", sorted(validation_payload, key=_row_sort_key))
    _write_json(out / "report.json", report)


def _parse_expected_holdout_counts(
    values: Sequence[str], *, holdout_sources: Sequence[str]
) -> dict[str, int]:
    if not values:
        if set(holdout_sources) == set(DEFAULT_HOLDOUT_COUNTS):
            return dict(DEFAULT_HOLDOUT_COUNTS)
        raise BallNoCleanJudge(
            "non-default holdout sources require --expected-holdout-count SOURCE=COUNT"
        )
    result: dict[str, int] = {}
    for value in values:
        source, separator, raw_count = value.partition("=")
        if not separator or not source:
            raise BallNoCleanJudge(
                f"--expected-holdout-count must be SOURCE=COUNT, got {value!r}"
            )
        result[source] = int(raw_count)
    if set(result) != set(holdout_sources):
        raise BallNoCleanJudge(
            f"expected holdout count sources {sorted(result)} != holdouts {sorted(holdout_sources)}"
        )
    return result


def _validate_common_arguments(
    *,
    holdout_sources: Sequence[str],
    collision_hamming_threshold: int,
    confirmed_prelabel_weight: float,
) -> None:
    if not holdout_sources or len(set(holdout_sources)) != len(holdout_sources):
        raise BallNoCleanJudge("holdout sources must be nonempty and unique")
    if collision_hamming_threshold < 0 or collision_hamming_threshold > 64:
        raise BallNoCleanJudge("collision Hamming threshold must be in [0, 64]")
    if not 0.0 < confirmed_prelabel_weight < 1.0:
        raise BallNoCleanJudge("confirmed_prelabel weight must be explicitly low in (0, 1)")


def _validate_production_inputs(
    *,
    production_mode: bool,
    reviewed_root: Path,
    scratch_package: Path,
    holdout_sources: Sequence[str],
    selection_manifests: Sequence[Path],
    legacy_reviewed_root: Path | None,
    legacy_prelabel_root: Path | None,
    expected_reviewed_count: int,
    expected_scratch_count: int,
    expected_reviewed_report_sha256: str | None,
    expected_scratch_package_sha256: str | None,
    expected_image_zip_sha256: str,
    expected_lineage_inputs_sha256: str,
) -> None:
    if not production_mode:
        return
    if reviewed_root.resolve() != DEFAULT_REVIEWED_ROOT.resolve():
        raise BallNoCleanJudge("production mode requires the canonical reviewed corpus")
    if scratch_package.resolve() != DEFAULT_SCRATCH_PACKAGE.resolve():
        raise BallNoCleanJudge("production mode requires the canonical scratch package")
    if set(holdout_sources) != set(DEFAULT_HOLDOUT_COUNTS):
        raise BallNoCleanJudge("production mode requires the canonical two holdout sources")
    if expected_reviewed_count != 3026 or expected_scratch_count != 350:
        raise BallNoCleanJudge("production reviewed/scratch counts are immutable")
    if expected_image_zip_sha256 != EXPECTED_SCRATCH_IMAGE_ZIP_SHA256:
        raise BallNoCleanJudge("production image-ZIP SHA-256 pin is immutable")
    supplied_selection = {path.resolve() for path in selection_manifests}
    expected_selection = {path.resolve() for path in DEFAULT_SELECTION_MANIFESTS}
    if supplied_selection != expected_selection or len(selection_manifests) != len(
        DEFAULT_SELECTION_MANIFESTS
    ):
        raise BallNoCleanJudge("production mode requires the canonical selection manifests")
    if (
        legacy_reviewed_root is None
        or legacy_reviewed_root.resolve() != DEFAULT_LEGACY_REVIEWED_ROOT.resolve()
        or legacy_prelabel_root is None
        or legacy_prelabel_root.resolve() != DEFAULT_LEGACY_PRELABEL_ROOT.resolve()
    ):
        raise BallNoCleanJudge("production mode requires canonical legacy lineage roots")
    if expected_reviewed_report_sha256 != EXPECTED_REVIEWED_REPORT_SHA256:
        raise BallNoCleanJudge("production reviewed-report SHA-256 pin is immutable")
    if expected_scratch_package_sha256 != EXPECTED_SCRATCH_PACKAGE_SHA256:
        raise BallNoCleanJudge("production scratch-package SHA-256 pin is immutable")
    if expected_lineage_inputs_sha256 != EXPECTED_LINEAGE_INPUTS_SHA256:
        raise BallNoCleanJudge("production lineage-input SHA-256 pin is immutable")


def _assert_no_protected_row_tokens(rows: Sequence[Mapping[str, Any]]) -> None:
    collisions = []
    for row in rows:
        identity = " ".join(
            str(row.get(field) or "")
            for field in ("row_key", "clip_id", "source_id", "image_name", "video_path")
        ).lower()
        if any(token in identity for token in PROTECTED_ROW_TOKENS):
            collisions.append(str(row.get("row_key")))
    if collisions:
        raise BallNoCleanJudge(
            f"protected eval identities reached split rows: {collisions[:5]}"
        )


def _row_key_intersection(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> list[str]:
    left_keys = {str(row.get("row_key") or "") for row in left}
    right_keys = {str(row.get("row_key") or "") for row in right}
    return sorted(left_keys & right_keys)


def _parse_scratch_image_name(name: str) -> tuple[str, str, int]:
    match = IMAGE_NAME_RE.match(PurePosixPath(name).name)
    if match is None:
        raise BallNoCleanJudge(f"unrecognized scratch image name: {name}")
    return match.group("source"), match.group("clip"), int(match.group("frame"))


def _image_hash_record(
    image: np.ndarray, *, name: str, encoded: bytes | None = None
) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(image)
    height, width = contiguous.shape[:2]
    raw = (
        int(width).to_bytes(4, "big")
        + int(height).to_bytes(4, "big")
        + contiguous.tobytes()
    )
    return {
        "name": name,
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "encoded_sha256": None if encoded is None else hashlib.sha256(encoded).hexdigest(),
        "dhash": _dhash(contiguous),
    }


def _dhash(image: np.ndarray, *, hash_size: int = 8) -> int:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    value = 0
    bit = 0
    for row in range(hash_size):
        for col in range(hash_size):
            if int(resized[row, col]) > int(resized[row, col + 1]):
                value |= 1 << bit
            bit += 1
    return value


def _decode_image(payload: bytes, *, label: str) -> np.ndarray:
    array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise BallNoCleanJudge(f"cannot decode image: {label}")
    return image


def _original_dimensions(payload: Mapping[str, Any]) -> tuple[int, int]:
    task = dict(payload.get("task") or {})
    size = task.get("original_size")
    if not isinstance(size, Sequence) or isinstance(size, (str, bytes)) or len(size) != 2:
        raise BallNoCleanJudge(f"reviewed payload lacks original_size: {payload.get('clip_id')}")
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise BallNoCleanJudge(f"invalid original_size for {payload.get('clip_id')}: {size}")
    return width, height


def _source_id_from_clip(clip_id: str) -> str:
    if "_rally_" not in clip_id:
        raise BallNoCleanJudge(f"cannot derive parent source from clip id: {clip_id}")
    return clip_id.rsplit("_rally_", 1)[0]


def _resolve_reference(raw: str, owner_path: Path) -> Path:
    if not raw:
        raise BallNoCleanJudge(f"missing path reference in {owner_path}")
    path = Path(raw)
    if path.is_absolute():
        return path
    root_candidate = ROOT / path
    if root_candidate.exists():
        return root_candidate
    return owner_path.parent / path


def _check_optional_sha256(path: Path, expected: str | None, *, label: str) -> str | None:
    if expected is None:
        return _sha256_file(path) if path.is_file() else None
    if not path.is_file():
        raise BallNoCleanJudge(f"missing {label}: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise BallNoCleanJudge(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return actual


def _check_required_sha256(path: Path, expected: str, *, label: str) -> str:
    if not expected or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise BallNoCleanJudge(f"{label} expected SHA-256 must be a pinned lowercase digest")
    if not path.is_file():
        raise BallNoCleanJudge(f"missing {label}: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise BallNoCleanJudge(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return actual


def _validate_declared_sha256(
    payload: Mapping[str, Any],
    *,
    field: str,
    expected: str,
    label: str,
) -> None:
    """Reject any supplied digest field unless it exactly matches computed bytes."""

    if field not in payload:
        return
    declared = payload[field]
    if type(declared) is not str or declared != expected:
        raise BallNoCleanJudge(
            f"{label} {field} mismatch: {declared!r} != {expected}"
        )


def _strict_positive_int(value: Any, *, label: str) -> int:
    if type(value) is int:
        parsed = value
    elif type(value) is str and re.fullmatch(r"[1-9][0-9]*", value):
        parsed = int(value)
    else:
        raise BallNoCleanJudge(f"{label} must be a positive integer")
    if parsed <= 0:
        raise BallNoCleanJudge(f"{label} must be a positive integer")
    return parsed


def _cvat_images_metadata_size(root: ElementTree.Element) -> int | None:
    node = root.find("./meta/job/size")
    if node is None:
        node = root.find("./meta/task/size")
    value = _text(node)
    return int(value) if value else None


def _assert_count(actual: int, expected: int, label: str) -> None:
    if actual != expected:
        raise BallNoCleanJudge(f"{label} {actual} != expected {expected}")


def _pass_check(after: str) -> dict[str, Any]:
    return {"verdict": "PASS", "after": after}


def _row_sort_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("source_id") or ""), str(row.get("row_key") or "")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BallNoCleanJudge(f"missing JSON input: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BallNoCleanJudge(f"JSON input must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(node: ElementTree.Element | None) -> str:
    return "" if node is None or node.text is None else node.text.strip()


if __name__ == "__main__":
    raise SystemExit(main())
