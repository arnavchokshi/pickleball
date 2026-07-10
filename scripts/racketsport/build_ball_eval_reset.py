#!/usr/bin/env python3
"""Build the NS-02.3/02.4 BALL evaluation-reset artifacts.

This is a CPU-only dataset/fold builder. It reads only the reviewed internal
corpus and non-label provenance/selection manifests. It never opens strict
held-out Outdoor/Indoor labels or the reserved HARVEST source directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEWED_ROOT = Path("runs/lanes/w7_ballingest2_20260709/reviewed_corpus")
DEFAULT_CORPUS_MD5_MANIFEST = Path("runs/lanes/w7_ballingest2_20260709/corpus_md5_manifest.json")
DEFAULT_SOURCE_METADATA = Path("cvat_upload/w6_labelpack_20260708/package_manifest.json")
DEFAULT_PROVENANCE_ROOT = Path("data/online_harvest_20260706/rallies")
DEFAULT_W5_SELECTION = Path("runs/lanes/w5_labelpack_20260708/selection_manifest.json")
DEFAULT_W6_SELECTION = Path("runs/lanes/w6_labelpack_20260708/selection_manifest.json")
DEFAULT_LEGACY_REVIEW_ROOT = Path("cvat_upload/exports/harvest_review_20260707")
DEFAULT_PRE_W7_REVIEWED_ROOT = Path("runs/lanes/w6_labelingest_20260708/reviewed_corpus")
DEFAULT_OUT_ROOT = Path("runs/lanes/ns02_evalreset_20260709")
PROTECTED_TOKENS = (
    "outdoor_webcam_iynbd",
    "indoor_doubles_fwuks",
    "pwxNwFfYQlQ",
    "vQhtz8l6VqU",
)
ROW_FIELDS = (
    "row_key",
    "clip_id",
    "frame_index",
    "source_id",
    "source_group_id",
    "source_class",
    "game_id",
    "session_id",
    "court_id",
    "device_id",
    "ball_present",
    "visibility_level",
    "selection_origin",
    "disagreement_type",
    "hard_disagreement",
    "visibility_slice_allowed",
    "occluded_slice",
    "official_tennis_control_seen_status",
    "stage1_official_seen_status",
    "seed_official_seen_status",
)


class EvalResetError(ValueError):
    """Raised when source identity or fold integrity cannot be proven."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-root", type=Path, default=DEFAULT_REVIEWED_ROOT)
    parser.add_argument("--corpus-md5-manifest", type=Path, default=DEFAULT_CORPUS_MD5_MANIFEST)
    parser.add_argument("--source-metadata-manifest", type=Path, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--provenance-root", type=Path, default=DEFAULT_PROVENANCE_ROOT)
    parser.add_argument("--w5-selection-manifest", type=Path, default=DEFAULT_W5_SELECTION)
    parser.add_argument("--w6-selection-manifest", type=Path, default=DEFAULT_W6_SELECTION)
    parser.add_argument("--legacy-review-root", type=Path, default=DEFAULT_LEGACY_REVIEW_ROOT)
    parser.add_argument("--pre-w7-reviewed-root", type=Path, default=DEFAULT_PRE_W7_REVIEWED_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--reviewed-sample-size", type=int, default=240)
    parser.add_argument("--unlabeled-sample-size", type=int, default=240)
    parser.add_argument("--expected-row-count", type=int, default=1750)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Execute the checked-in source-leakage test against an existing output directory.",
    )
    args = parser.parse_args(argv)

    try:
        if args.verify_only:
            report = verify_written_artifacts(args.out_root)
        else:
            report = build_artifacts(
                reviewed_root=args.reviewed_root,
                corpus_md5_manifest=args.corpus_md5_manifest,
                source_metadata_manifest=args.source_metadata_manifest,
                provenance_root=args.provenance_root,
                w5_selection_manifest=args.w5_selection_manifest,
                w6_selection_manifest=args.w6_selection_manifest,
                legacy_review_root=args.legacy_review_root,
                pre_w7_reviewed_root=args.pre_w7_reviewed_root,
                out_root=args.out_root,
                seed=args.seed,
                reviewed_sample_size=args.reviewed_sample_size,
                unlabeled_sample_size=args.unlabeled_sample_size,
                expected_row_count=args.expected_row_count,
            )
    except Exception as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_artifacts(
    *,
    reviewed_root: Path,
    corpus_md5_manifest: Path,
    source_metadata_manifest: Path,
    provenance_root: Path,
    w5_selection_manifest: Path,
    w6_selection_manifest: Path,
    legacy_review_root: Path,
    pre_w7_reviewed_root: Path,
    out_root: Path,
    seed: int,
    reviewed_sample_size: int,
    unlabeled_sample_size: int,
    expected_row_count: int,
) -> dict[str, Any]:
    _assert_internal_paths(
        reviewed_root,
        corpus_md5_manifest,
        source_metadata_manifest,
        provenance_root,
        w5_selection_manifest,
        w6_selection_manifest,
        legacy_review_root,
        pre_w7_reviewed_root,
    )
    _positive(reviewed_sample_size, "reviewed_sample_size")
    _positive(unlabeled_sample_size, "unlabeled_sample_size")

    source_metadata_payload = _load_json(source_metadata_manifest)
    source_metadata = dict(source_metadata_payload.get("source_metadata") or {})
    w5_current = _selection_rows(w5_selection_manifest, first_session_only=True)
    w6_current = _selection_rows(w6_selection_manifest, first_session_only=True)
    legacy_reviewed = _legacy_reviewed_row_keys(legacy_review_root)
    pre_w7_reviewed = _reviewed_row_keys(pre_w7_reviewed_root)
    disagreement_universe = {
        **_selection_rows(w5_selection_manifest, first_session_only=False),
        **_selection_rows(w6_selection_manifest, first_session_only=False),
    }

    rows, clips, groups = _load_reviewed_rows(
        reviewed_root=reviewed_root,
        provenance_root=provenance_root,
        source_metadata=source_metadata,
        w5_current=w5_current,
        w6_current=w6_current,
        legacy_reviewed=legacy_reviewed,
        pre_w7_reviewed=pre_w7_reviewed,
    )
    if len(rows) != expected_row_count:
        raise EvalResetError(f"reviewed row count {len(rows)} != expected {expected_row_count}")
    if len({row["row_key"] for row in rows}) != len(rows):
        raise EvalResetError("duplicate reviewed row_key values")

    out_root.mkdir(parents=True, exist_ok=True)
    _write_source_table(out_root / "source_group_table.csv", rows)
    source_summary = _source_summary(rows, groups)
    _write_json(out_root / "source_groups.json", source_summary)

    per_clip_manifest = _build_per_clip_manifest(rows)
    _write_json(out_root / "per_clip_fold_manifest.json", per_clip_manifest)
    grouped_manifest = _build_source_grouped_manifest(rows)
    _write_json(out_root / "source_grouped_fold_manifest.json", grouped_manifest)
    leakage_report = verify_grouped_fold_manifest(grouped_manifest)
    _write_json(out_root / "source_group_leakage_report.json", leakage_report)

    reviewed_sample = _sample_reviewed(rows, seed=seed, sample_size=reviewed_sample_size)
    _write_json(out_root / "uniform_random_reviewed_frames.json", reviewed_sample)
    hard_rows = _stratum_row_list(
        rows,
        name="hard_disagreement",
        predicate=lambda row: bool(row["hard_disagreement"]),
    )
    _write_json(out_root / "hard_disagreement_reviewed_frames.json", hard_rows)
    occluded_rows = _stratum_row_list(
        rows,
        name="occluded_visibility_eligible",
        predicate=lambda row: bool(row["occluded_slice"]),
    )
    _write_json(out_root / "occluded_reviewed_frames.json", occluded_rows)
    unlabeled_sample = _sample_unlabeled(
        rows=rows,
        clips=clips,
        groups=groups,
        disagreement_universe=disagreement_universe,
        seed=seed + 1,
        sample_size=unlabeled_sample_size,
    )
    _write_json(out_root / "uniform_random_unlabeled_owner_queue.json", unlabeled_sample)

    corpus_md5 = _md5_file(corpus_md5_manifest)
    card = _render_dataset_card(
        rows=rows,
        groups=groups,
        corpus_md5=corpus_md5,
        corpus_md5_manifest=corpus_md5_manifest,
        reviewed_root=reviewed_root,
        reviewed_sample=reviewed_sample,
        unlabeled_sample=unlabeled_sample,
        leakage_report=leakage_report,
        seed=seed,
    )
    (out_root / "dataset_card.md").write_text(card, encoding="utf-8")

    artifacts = [
        "source_group_table.csv",
        "source_groups.json",
        "per_clip_fold_manifest.json",
        "source_grouped_fold_manifest.json",
        "source_group_leakage_report.json",
        "uniform_random_reviewed_frames.json",
        "hard_disagreement_reviewed_frames.json",
        "occluded_reviewed_frames.json",
        "uniform_random_unlabeled_owner_queue.json",
        "dataset_card.md",
    ]
    artifact_manifest = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_eval_reset_artifact_manifest",
        "protected_labels_read": False,
        "cpu_only": True,
        "files": [
            {"path": name, "sha256": _sha256_file(out_root / name), "bytes": (out_root / name).stat().st_size}
            for name in artifacts
        ],
    }
    _write_json(out_root / "artifact_manifest.json", artifact_manifest)
    return {
        "objective_result": "PASS",
        "row_count": len(rows),
        "clip_count": len(clips),
        "source_group_count": len(groups),
        "unknown_source_row_count": source_summary["unknown_source_row_count"],
        "corpus_md5": corpus_md5,
        "leakage_test": leakage_report["objective_result"],
        "reviewed_sample_count": reviewed_sample["sample_count"],
        "unlabeled_sample_count": unlabeled_sample["sample_count"],
        "out_root": str(out_root),
    }


def verify_written_artifacts(out_root: Path) -> dict[str, Any]:
    manifest_path = out_root / "source_grouped_fold_manifest.json"
    report = verify_grouped_fold_manifest(_load_json(manifest_path))
    table_path = out_root / "source_group_table.csv"
    with table_path.open(newline="", encoding="utf-8") as handle:
        table_rows = list(csv.DictReader(handle))
    expected = int(_load_json(manifest_path)["row_count"])
    unique = len({row["row_key"] for row in table_rows})
    if len(table_rows) != expected or unique != expected:
        raise EvalResetError(
            f"source table coverage failed: rows={len(table_rows)} unique={unique} expected={expected}"
        )
    return {
        **report,
        "source_table_row_count": len(table_rows),
        "source_table_unique_row_count": unique,
    }


def verify_grouped_fold_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    expected_rows = int(manifest.get("row_count") or 0)
    for fold in manifest.get("folds") or []:
        fold_id = str(fold.get("fold_id"))
        role_groups = {
            role: set(fold.get(f"{role}_source_group_ids") or [])
            for role in ("train", "selection", "test")
        }
        role_rows = {
            role: set(fold.get(f"{role}_row_keys") or [])
            for role in ("train", "selection", "test")
        }
        if role_groups["train"] & role_groups["selection"]:
            failures.append(f"{fold_id}: train/selection source overlap")
        if role_groups["train"] & role_groups["test"]:
            failures.append(f"{fold_id}: train/test source overlap")
        if role_groups["selection"] & role_groups["test"]:
            failures.append(f"{fold_id}: selection/test source overlap")
        if role_rows["train"] & role_rows["selection"]:
            failures.append(f"{fold_id}: train/selection row overlap")
        if role_rows["train"] & role_rows["test"]:
            failures.append(f"{fold_id}: train/test row overlap")
        if role_rows["selection"] & role_rows["test"]:
            failures.append(f"{fold_id}: selection/test row overlap")
        union = role_rows["train"] | role_rows["selection"] | role_rows["test"]
        if len(union) != expected_rows:
            failures.append(f"{fold_id}: row coverage {len(union)} != {expected_rows}")
    if not manifest.get("folds"):
        failures.append("no folds")
    report = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_source_group_leakage_test",
        "objective_result": "PASS" if not failures else "FAIL",
        "fold_count": len(manifest.get("folds") or []),
        "row_count": expected_rows,
        "cross_fold_source_overlap_count": len(failures),
        "all_disjoint": not failures,
        "failures": failures,
    }
    if failures:
        raise EvalResetError("; ".join(failures))
    return report


def _load_reviewed_rows(
    *,
    reviewed_root: Path,
    provenance_root: Path,
    source_metadata: Mapping[str, Any],
    w5_current: Mapping[str, str],
    w6_current: Mapping[str, str],
    legacy_reviewed: set[str],
    pre_w7_reviewed: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    clips: dict[str, int] = {}
    groups: dict[str, dict[str, Any]] = {}
    paths = sorted(reviewed_root.glob("*/reviewed_boxes.json"))
    if not paths:
        raise EvalResetError(f"no reviewed_boxes.json files under {reviewed_root}")
    for path in paths:
        payload = _load_json(path)
        clip_id = str(payload.get("clip_id") or path.parent.name)
        _assert_no_protected_token(clip_id)
        source_id = _source_id_from_clip(clip_id)
        provenance_path = provenance_root / source_id / f"{clip_id}.provenance.json"
        provenance = _load_json(provenance_path)
        provenance_source = dict(provenance.get("source") or {})
        provenance_source_id = str(provenance_source.get("source_id") or "")
        if provenance_source_id != source_id:
            raise EvalResetError(
                f"source id mismatch for {clip_id}: derived={source_id!r} provenance={provenance_source_id!r}"
            )
        source_sha = str(provenance.get("source_sha256") or "")
        if len(source_sha) != 64:
            raise EvalResetError(f"missing source_sha256 for {clip_id}")
        group_id = f"recording:{source_id}:{source_sha[:12]}"
        metadata = dict(source_metadata.get(source_id) or {})
        group = groups.setdefault(
            group_id,
            {
                "source_group_id": group_id,
                "source_id": source_id,
                "source_sha256": source_sha,
                "source_class": str(metadata.get("source_class") or "UNKNOWN"),
                "title": str(metadata.get("title") or provenance_source.get("title") or "UNKNOWN"),
                "channel": str(metadata.get("channel") or provenance_source.get("channel") or "UNKNOWN"),
                "venue_summary": str(metadata.get("venue_summary") or "UNKNOWN"),
                "game_id": f"UNKNOWN_GAME::{source_id}",
                "session_id": f"recording_session::{source_id}",
                "court_id": f"UNKNOWN_COURT::{source_id}",
                "device_id": f"UNKNOWN_DEVICE::{source_id}",
                "unknown_dimensions": ["game_id", "court_id", "device_id"],
                "mapping_method": (
                    "clip prefix -> provenance source.source_id; group bound to immutable raw "
                    "source_sha256; all clips from one recording remain together"
                ),
                "clips": [],
            },
        )
        if group["source_sha256"] != source_sha:
            raise EvalResetError(f"inconsistent raw-source sha for {source_id}")
        group["clips"].append(clip_id)
        frame_count = len(payload.get("frames") or [])
        clips[clip_id] = frame_count
        reviewed_indices = sorted({int(value) for value in payload.get("reviewed_frame_indices") or []})
        frames_by_index = {
            int(frame["frame_index"]): frame for frame in payload.get("frames") or []
        }
        for frame_index in reviewed_indices:
            frame = frames_by_index.get(frame_index)
            if frame is None:
                raise EvalResetError(f"{clip_id}:{frame_index} is reviewed but missing from frames")
            row_key = f"{clip_id}:{frame_index:06d}"
            boxes = [
                box for box in frame.get("boxes") or [] if str(box.get("label") or "").lower() == "ball"
            ]
            frame_visibility = dict(frame.get("visibility_levels_by_label") or {})
            visibility_level = (
                str(boxes[0].get("visibility_level") or frame_visibility.get("ball") or "clear")
                if boxes
                else str(frame_visibility.get("ball") or "none")
            )
            if row_key in w6_current and row_key not in pre_w7_reviewed:
                origin = "disagreement_selected_w6_session_01_box_position_only"
                disagreement_type = w6_current[row_key]
                visibility_allowed = False
            elif row_key in w5_current and row_key not in legacy_reviewed:
                origin = "disagreement_selected_w5_session_01"
                disagreement_type = w5_current[row_key]
                visibility_allowed = True
            else:
                origin = "legacy_reviewed_selection_unknown"
                disagreement_type = "none"
                visibility_allowed = True
            rows.append(
                {
                    "row_key": row_key,
                    "clip_id": clip_id,
                    "frame_index": frame_index,
                    "source_id": source_id,
                    "source_group_id": group_id,
                    "source_class": group["source_class"],
                    "game_id": group["game_id"],
                    "session_id": group["session_id"],
                    "court_id": group["court_id"],
                    "device_id": group["device_id"],
                    "ball_present": bool(boxes),
                    "visibility_level": visibility_level,
                    "selection_origin": origin,
                    "disagreement_type": disagreement_type,
                    "hard_disagreement": origin.startswith("disagreement_selected"),
                    "visibility_slice_allowed": visibility_allowed,
                    "occluded_slice": visibility_allowed and visibility_level in {"partial", "out_of_frame"},
                    "official_tennis_control_seen_status": "UNKNOWN_UPSTREAM_TRAINING_SOURCES",
                    "stage1_official_seen_status": "UNSEEN_BY_SOURCE_ID",
                    "seed_official_seen_status": "SEEN_IN_486_ROW_FINE_TUNE_SOURCE",
                }
            )
    for group in groups.values():
        group["clips"] = sorted(set(group["clips"]))
    return sorted(rows, key=lambda row: row["row_key"]), clips, groups


def _build_per_clip_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_clip: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_clip[str(row["clip_id"])].append(str(row["row_key"]))
    all_keys = {str(row["row_key"]) for row in rows}
    folds = []
    for clip_id in sorted(by_clip):
        test = sorted(by_clip[clip_id])
        folds.append(
            {
                "fold_id": f"clip::{clip_id}",
                "test_clip_id": clip_id,
                "test_row_count": len(test),
                "test_row_keys": test,
                "train_row_count": len(all_keys) - len(test),
                "train_row_keys": sorted(all_keys - set(test)),
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_per_clip_fold_manifest",
        "protocol": "historical_clip_as_source_compatibility",
        "warning": "Not source-LoSO when multiple clips share one recording source.",
        "row_count": len(rows),
        "fold_count": len(folds),
        "folds": folds,
    }


def _build_source_grouped_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_group[str(row["source_group_id"])].append(str(row["row_key"]))
    groups = sorted(by_group)
    if len(groups) < 3:
        raise EvalResetError("source-grouped train/selection/test folds require at least three groups")
    folds = []
    for index, test_group in enumerate(groups):
        selection_group = groups[(index + 1) % len(groups)]
        train_groups = [group for group in groups if group not in {test_group, selection_group}]
        role_groups = {
            "train": train_groups,
            "selection": [selection_group],
            "test": [test_group],
        }
        fold: dict[str, Any] = {
            "fold_id": f"source_loso::{index + 1:02d}",
            "test_source_group_id": test_group,
        }
        for role, role_group_ids in role_groups.items():
            role_keys = sorted(key for group in role_group_ids for key in by_group[group])
            fold[f"{role}_source_group_ids"] = role_group_ids
            fold[f"{role}_row_count"] = len(role_keys)
            fold[f"{role}_row_keys"] = role_keys
        folds.append(fold)
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_true_source_grouped_fold_manifest",
        "protocol": "rotating_source_disjoint_train_selection_test",
        "row_count": len(rows),
        "source_group_count": len(groups),
        "source_group_ids": groups,
        "fold_count": len(folds),
        "folds": folds,
    }


def _sample_reviewed(rows: Sequence[Mapping[str, Any]], *, seed: int, sample_size: int) -> dict[str, Any]:
    if sample_size > len(rows):
        raise EvalResetError(f"reviewed sample size {sample_size} > row count {len(rows)}")
    sampled = random.Random(seed).sample(list(rows), sample_size)
    output_rows = []
    for ordinal, row in enumerate(sampled, start=1):
        output_rows.append({"sample_ordinal": ordinal, **dict(row)})
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_uniform_random_reviewed_sample",
        "seed": seed,
        "sampling_frame": "simple_random_sample_without_replacement_over_all_1750_reviewed_rows",
        "representativeness_warning": (
            "Uniform within the reviewed corpus, not within raw video. Existing corpus selection bias remains."
        ),
        "sample_count": sample_size,
        "rows": output_rows,
    }


def _stratum_row_list(
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
    predicate: Any,
) -> dict[str, Any]:
    selected = [dict(row) for row in rows if predicate(row)]
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_reviewed_stratum_row_list",
        "stratum": name,
        "sample_count": len(selected),
        "rows": selected,
    }


def _sample_unlabeled(
    *,
    rows: Sequence[Mapping[str, Any]],
    clips: Mapping[str, int],
    groups: Mapping[str, Mapping[str, Any]],
    disagreement_universe: Mapping[str, str],
    seed: int,
    sample_size: int,
) -> dict[str, Any]:
    reviewed = {str(row["row_key"]) for row in rows}
    group_by_clip = {
        clip: group_id for group_id, group in groups.items() for clip in group["clips"]
    }
    eligible: list[tuple[str, int]] = []
    for clip_id, frame_count in sorted(clips.items()):
        eligible.extend(
            (clip_id, frame_index)
            for frame_index in range(frame_count)
            if f"{clip_id}:{frame_index:06d}" not in reviewed
        )
    if sample_size > len(eligible):
        raise EvalResetError(f"unlabeled sample size {sample_size} > eligible count {len(eligible)}")
    sampled = random.Random(seed).sample(eligible, sample_size)
    output_rows = []
    for ordinal, (clip_id, frame_index) in enumerate(sampled, start=1):
        row_key = f"{clip_id}:{frame_index:06d}"
        group_id = group_by_clip[clip_id]
        group = groups[group_id]
        output_rows.append(
            {
                "sample_ordinal": ordinal,
                "row_key": row_key,
                "clip_id": clip_id,
                "frame_index": frame_index,
                "source_id": group["source_id"],
                "source_group_id": group_id,
                "selection_origin": "uniform_random_unlabeled_audit",
                "owner_label_status": "UNLABELED_OWNER_BLOCKED",
                "score_allowed": False,
                "overlaps_disagreement_queue": row_key in disagreement_universe,
                "disagreement_type_if_overlap": disagreement_universe.get(row_key),
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_uniform_random_unlabeled_owner_queue",
        "seed": seed,
        "sampling_frame": (
            "simple_random_sample_without_replacement_over_every unreviewed frame in the 38 internal clips"
        ),
        "eligible_unreviewed_frame_count": len(eligible),
        "sample_count": sample_size,
        "owner_blocked": True,
        "do_not_score_until_labeled": True,
        "rows": output_rows,
    }


def _source_summary(rows: Sequence[Mapping[str, Any]], groups: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["source_group_id"]) for row in rows)
    group_rows = []
    for group_id, group in sorted(groups.items()):
        group_rows.append({**dict(group), "reviewed_row_count": counts[group_id]})
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_true_source_group_table_summary",
        "row_count": len(rows),
        "mapped_row_count": len(rows),
        "unknown_source_row_count": sum(
            1 for row in rows if str(row["source_group_id"]).startswith("UNKNOWN_SOURCE::")
        ),
        "source_group_count": len(groups),
        "unknown_component_policy": (
            "Unknown game/court/device dimensions retain source-specific UNKNOWN::<source_id> values; "
            "they are never merged across recording sources."
        ),
        "groups": group_rows,
    }


def _render_dataset_card(
    *,
    rows: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Mapping[str, Any]],
    corpus_md5: str,
    corpus_md5_manifest: Path,
    reviewed_root: Path,
    reviewed_sample: Mapping[str, Any],
    unlabeled_sample: Mapping[str, Any],
    leakage_report: Mapping[str, Any],
    seed: int,
) -> str:
    origins = Counter(str(row["selection_origin"]) for row in rows)
    sources = Counter(str(row["source_id"]) for row in rows)
    classes = Counter("positive" if row["ball_present"] else "negative" for row in rows)
    visibility_eligible = sum(bool(row["visibility_slice_allowed"]) for row in rows)
    hard_count = sum(bool(row["hard_disagreement"]) for row in rows)
    occluded_count = sum(bool(row["occluded_slice"]) for row in rows)
    unlabeled_rows = list(unlabeled_sample["rows"])
    unlabeled_sources = Counter(str(row["source_id"]) for row in unlabeled_rows)
    overlap_count = sum(bool(row["overlaps_disagreement_queue"]) for row in unlabeled_rows)
    lines = [
        "# NS-02.3-02.5 BALL dataset card",
        "",
        "Status: evaluation infrastructure only; `BALL VERIFIED=0`; no inference or protected-label scoring was run.",
        "",
        "## Corpus and provenance",
        "",
        f"- Reviewed corpus: `{reviewed_root}`",
        f"- Reviewed rows: {len(rows)} ({classes['positive']} positive, {classes['negative']} reviewed-absent).",
        f"- Corpus manifest: `{corpus_md5_manifest}`; actual file md5 `{corpus_md5}`.",
        "- Protected Outdoor/Indoor labels and reserved HARVEST-1/HARVEST-2 sources were not read.",
        "- Source mapping: clip prefix -> provenance `source.source_id`, bound to raw `source_sha256`.",
        "- Exact game/court/device identifiers are unavailable. They remain source-specific UNKNOWN values; six recordings are never merged.",
        "",
        "## Source groups",
        "",
        "| Source recording | Rows | Class | Known identity | Unknown dimensions |",
        "|---|---:|---|---|---|",
    ]
    for group_id, group in sorted(groups.items()):
        lines.append(
            f"| `{group_id}` | {sources[group['source_id']]} | {group['source_class']} | "
            f"recording ID + SHA-256 | game, court, device |"
        )
    lines.extend(
        [
            "",
            "## Reviewed-corpus strata",
            "",
            "| Stratum | Rows | Scoring rule |",
            "|---|---:|---|",
        ]
    )
    for name, count in sorted(origins.items()):
        rule = "box-position only; no visibility slices" if "w6_session" in name else "visibility metadata usable"
        lines.append(f"| `{name}` | {count} | {rule} |")
    lines.extend(
        [
            f"| `hard_disagreement` | {hard_count} | report separately; do not average away source gaps |",
            f"| `visibility_slice_eligible` | {visibility_eligible} | excludes all w6-session rows |",
            f"| `occluded` | {occluded_count} | only eligible `partial`/`out_of_frame`; never infer from w6 `clear` |",
            "",
            "## Uniform-random audit",
            "",
            f"- Seed `{seed}`: {reviewed_sample['sample_count']} rows sampled uniformly without replacement from the 1,750 reviewed rows.",
            "  This is reproducible but still inherits the reviewed corpus selection bias.",
            f"- Seed `{seed + 1}`: {unlabeled_sample['sample_count']} NEW unreviewed frames sampled uniformly from "
            f"{unlabeled_sample['eligible_unreviewed_frame_count']} eligible internal-video frames.",
            f"- Natural overlap with a disagreement queue: {overlap_count}/{unlabeled_sample['sample_count']} (recorded, not filtered, so uniformity is preserved).",
            "- Owner must label the new list before any score. Until then it is a sample list, not ground truth.",
            "",
            "New unlabeled sample by source:",
            "",
            "| Source | Frames |",
            "|---|---:|",
        ]
    )
    for source_id, count in sorted(unlabeled_sources.items()):
        lines.append(f"| `{source_id}` | {count} |")
    lines.extend(
        [
            "",
            "## Seen/unseen contract (candidate-specific)",
            "",
            "| Candidate | Current six groups | Promotion meaning |",
            "|---|---|---|",
            "| `official_tennis_control` | UNKNOWN: upstream checkpoint training-source registry absent | score separately; not license-cleared |",
            "| `stage1_official` | unseen by source ID; public-corpus hash guard reported no protected collisions | diagnostic only; noncommercial training contamination blocks promotion |",
            "| `seed_official` | SEEN: its 486-row fine-tune used all six recording sources | seen-source diagnostic only; not source-disjoint promotion evidence |",
            "",
            "Random, hard/occluded, seen, and unseen metrics must be separate. A missing/unknown unseen stratum is reported as missing, never averaged into a global number.",
            "",
            "## Fold protocols and leakage",
            "",
            "- Historical compatibility: `per_clip_fold_manifest.json` (38 clip folds). Leave-one-clip is not source-LoSO.",
            "- True protocol: `source_grouped_fold_manifest.json` (six recording folds, rotating source-disjoint train/selection/test roles).",
            "- Executable reviewed strata: `uniform_random_reviewed_frames.json`, `hard_disagreement_reviewed_frames.json`, and `occluded_reviewed_frames.json`; pass any one via `ball_loso_validation.py --reviewed-row-list`.",
            f"- Executable proof: `.venv/bin/python scripts/racketsport/build_ball_eval_reset.py --out-root {DEFAULT_OUT_ROOT} --verify-only`.",
            f"- Latest proof: `{leakage_report['objective_result']}`, all_disjoint={str(leakage_report['all_disjoint']).lower()}, fold_count={leakage_report['fold_count']}.",
            "- Existing scoring harness: `scripts/racketsport/ball_loso_validation.py`; its default is clip-as-source. The new GPU commands pass explicit `--source-group` mappings and retain both reports.",
            "- Prior artifacts remain unchanged: `runs/lanes/w7_ballingest2_20260709/loso_fold_manifest.json` and `runs/lanes/w7_ballingest2_20260709/GPU_RESCORE_COMMANDS.sh`.",
            "",
            "## Owner/external blockers",
            "",
            "- Label `uniform_random_unlabeled_owner_queue.json`; do not score it beforehand.",
            "- Capture-device/court/game identities need authoritative metadata if finer grouping than recording ID is ever required.",
            "- Fresh untouched owner/HARVEST promotion sources remain reserved and unexposed; this lane does not spend them.",
            "- Candidate license clearance is incomplete/noncommercial-contaminated; see `candidate_license_card.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def _selection_rows(path: Path, *, first_session_only: bool) -> dict[str, str]:
    payload = _load_json(path)
    sessions = list(payload.get("sessions") or [])
    if first_session_only:
        sessions = sessions[:1]
    selected: dict[str, str] = {}
    for session in sessions:
        for frame in session.get("frames") or []:
            clip_id = str(frame["clip_id"])
            frame_index = int(frame["frame_index"])
            selected[f"{clip_id}:{frame_index:06d}"] = str(frame.get("disagreement_type") or "unknown")
    return selected


def _reviewed_row_keys(root: Path) -> set[str]:
    keys: set[str] = set()
    for path in sorted(root.glob("*/reviewed_boxes.json")):
        payload = _load_json(path)
        clip_id = str(payload.get("clip_id") or path.parent.name)
        keys.update(
            f"{clip_id}:{int(frame_index):06d}"
            for frame_index in payload.get("reviewed_frame_indices") or []
        )
    return keys


def _legacy_reviewed_row_keys(root: Path) -> set[str]:
    keys: set[str] = set()
    for path in sorted(root.glob("*/annotations.xml")):
        tree = ElementTree.parse(path)
        task = tree.getroot().find("./meta/task")
        if task is None:
            task = tree.getroot().find("./meta/project/tasks/task")
        if task is None:
            raise EvalResetError(f"legacy CVAT XML missing meta/task: {path}")
        source = (task.findtext("source") or f"{path.parent.name}.mp4").strip()
        clip_id = Path(source).stem
        start = int(task.findtext("start_frame") or 0)
        stop = int(task.findtext("stop_frame") or 0)
        size = int(task.findtext("size") or 0)
        frame_filter = (task.findtext("frame_filter") or "").strip()
        if not frame_filter.startswith("step="):
            raise EvalResetError(f"legacy CVAT XML lacks supported step filter: {path}")
        step = int(frame_filter.split("=", 1)[1])
        reviewed = list(range(start, stop + 1, step))
        if size > 0:
            reviewed = reviewed[:size]
        keys.update(f"{clip_id}:{frame_index:06d}" for frame_index in reviewed)
    return keys


def _write_source_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in ROW_FIELDS})


def _source_id_from_clip(clip_id: str) -> str:
    marker = "_rally_"
    if marker not in clip_id:
        return f"UNKNOWN_SOURCE::{clip_id}"
    source_id = clip_id.split(marker, 1)[0]
    return source_id or f"UNKNOWN_SOURCE::{clip_id}"


def _assert_internal_paths(*paths: Path) -> None:
    for path in paths:
        _assert_no_protected_token(str(path))


def _assert_no_protected_token(value: str) -> None:
    lowered = value.lower()
    for token in PROTECTED_TOKENS:
        if token.lower() in lowered:
            raise EvalResetError(f"refusing protected source token {token!r} in {value!r}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EvalResetError(f"missing required JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvalResetError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _positive(value: int, name: str) -> None:
    if value <= 0:
        raise EvalResetError(f"{name} must be positive")


def _md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
