#!/usr/bin/env python3
"""Leave-one-source-out (LoSO) validation harness for the BALL detector.

Why this exists (see ``runs/manager/heldout_eval_ledger.md`` rows 4/14/16/19-20/22-23):
BALL has had 5+ internal-val-wins that then LOSE on the strict held-out Outdoor clip.
Root cause, corroborated against a March-2026 ETH-Zurich badminton paper
(arXiv 2603.06691: random/background-split CV F1 0.864 -> leave-one-location-out CV
F1 0.703; recall collapses 0.789->0.552 under domain shift while precision stays
~0.95): our internal-val metric is a single POOLED/MIXED micro-average across
Burlington+Wolverine frames. Pooling lets whichever clip has more frames dominate the
number, and hides how a candidate's score varies clip-to-clip -- exactly the axis that
predicts (or fails to predict) generalization to a genuinely new capture context.

This script partitions the *existing*, already-labeled BALL corpus by CAPTURE SOURCE
(clip = one continuous capture session/venue/camera setup, never a random frame split
and never a sub-clip-within-a-source split) and, for each candidate detector, computes:

  * per-source metrics (F1@20, recall@20, precision@20, hidden-FP rate) via the
    existing ``benchmark_cvat_ball_track_candidate`` scorer -- reused unmodified;
  * the POOLED/MIXED metric exactly as today's internal-val process computes it
    (micro-average across all scored sources under one candidate name);
  * the LoSO-mean metric (unweighted mean of the per-source scores -- i.e. "if you'd
    only ever gotten to develop against N-1 of these sources, what would the held-out
    Nth source's own score have been, averaged over which one is held out");
  * the LoSO-worst-fold metric (the single worst per-source score -- a conservative
    generalization estimate);
  * the GENERALIZATION GAP = pooled/mixed metric - LoSO-mean metric.

This is SCORING only. It never runs inference, never trains anything, and never reads
or scores against a strict-holdout clip's labels (Outdoor/Indoor) -- that is a hard,
code-enforced refusal in this script (see ``_assert_scoreable_source``), independent of
``threed/racketsport/eval_guard.py``'s own guard (imported read-only here for the
registry, not modified). Already-published held-out numbers (e.g. ledger rows 4/23) may
be supplied via ``--heldout-metric`` as plain literals for comparison; this script never
computes them itself.

Today only two capture sources have real human-reviewed CVAT ball labels and are legal
LoSO folds: Burlington and Wolverine (both indoor). That yields a degenerate 2-fold
LoSO, not the N-way design this harness is built to support -- see DESIGN_NOTES.md in
this lane's directory for why a 2-fold LoSO among two mutually-similar indoor sources
cannot fully triangulate the indoor/outdoor domain shift that actually bit BALL, and
what widening the source pool (e.g. the online-harvest corpus, once it carries reviewed
labels) would add.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_cvat_benchmark import (  # noqa: E402
    CvatBallCandidate,
    benchmark_cvat_ball_tracker_candidates,
)
from threed.racketsport import ball_cvat_benchmark as cvat_benchmark  # noqa: E402
from threed.racketsport.ball_overlay import load_ball_track  # noqa: E402
from threed.racketsport.eval_guard import (  # noqa: E402
    INTERNAL_VAL_ONLY_CLIP_IDS,
    STRICT_HOLDOUT_CLIP_IDS,
)


ARTIFACT_TYPE = "racketsport_ball_loso_validation"
DEFAULT_CVAT_ROOT = Path("runs/cvat_imports/2026_06_30")
REVIEWED_BOXES_FILENAME = "reviewed_boxes.json"

# This is the accepted B0 judge named by EXACT_PLAN B0/B2 and the fix-round
# instruction.  Its identity is deliberately frozen in code: copying the same
# JSONL to a caller-selected directory, editing a row, or substituting a video
# with the same basename is not the accepted judge.
FROZEN_B0_SPLIT_DIR = ROOT / "runs/lanes/ball_b0_split_20260721/split"
FROZEN_B0_ARTIFACT_SHA256 = {
    "report.json": "122e65913d54df6be6c3e5c6ca91229fc17d207674edc7a31ea705bafd6eb3a3",
    "train.jsonl": "b92218d47816e01893a687c6414bdaa5220f02be6d3b1c25b684128d12ee9c20",
    "validation.jsonl": "39a07ed6d5211cbdc2ccc8a3f1f73b298a1ed262a6cae1f8a6190e5aa1533429",
    "lineage_rows.jsonl": "289a46c4bca3bb08d4df28058836abc3d1b29081f82a9e2c079e1a0f99dd7969",
}
FROZEN_B0_SOURCE_VIDEO_SHA256 = {
    "Ezz6HDNHlnk_rally_0001": "582ffbb02098bb8e59afff74ffa25fb178be8ecc7782889b06a0e0aa64bef844",
    "Ezz6HDNHlnk_rally_0002": "6ed769d5464d89fee54e29603605cdcac53c3ade29aa1b306f4b1eca50228650",
    "Ezz6HDNHlnk_rally_0004": "05fe6312108825bdd51a812dac6dd5a2450f2ee1e20aca62197569b233e906a2",
    "Ezz6HDNHlnk_rally_0005": "e622b1646920f43639d4c076e5e7cc5a10484d83526d5478c18c3836b221819d",
    "Ezz6HDNHlnk_rally_0006": "4f30bc394415f9e7d4de6cf0cc20e493646c266eb83c998f562a0b830e35a355",
    "Ezz6HDNHlnk_rally_0007": "77d02862ffd890e1f2935f56ed66c64bf39eb984f8b7c125723b83db8883b0f8",
    "Ezz6HDNHlnk_rally_0008": "88e095ca7260226f65d9b24ba2257590ba558a28c777d380daba94390e163c9a",
    "HyUqT7zFiwk_rally_0001": "056f1710d864bf9f5847c896cab8842d34b94da661fa4fd62b59d9ae1219eae3",
}
FROZEN_B0_LEGACY_PREDICTION_VIDEO_PATH = {
    "Ezz6HDNHlnk_rally_0001": "/home/arnavchokshi/w3_gpustage_20260707/clips/rallies/Ezz6HDNHlnk/Ezz6HDNHlnk_rally_0001.mp4",
    "Ezz6HDNHlnk_rally_0002": "/home/arnavchokshi/w3_gpustage_20260707/clips/rallies/Ezz6HDNHlnk/Ezz6HDNHlnk_rally_0002.mp4",
    "Ezz6HDNHlnk_rally_0004": "/home/arnavchokshi/prelabel_20260707/data/online_harvest_20260706/rallies/Ezz6HDNHlnk/Ezz6HDNHlnk_rally_0004.mp4",
    "Ezz6HDNHlnk_rally_0005": "/home/arnavchokshi/w3_gpustage_20260707/clips/rallies/Ezz6HDNHlnk/Ezz6HDNHlnk_rally_0005.mp4",
    "Ezz6HDNHlnk_rally_0006": "/home/arnavchokshi/w3_gpustage_20260707/clips/rallies/Ezz6HDNHlnk/Ezz6HDNHlnk_rally_0006.mp4",
    "Ezz6HDNHlnk_rally_0007": "/home/arnavchokshi/w3_gpustage_20260707/clips/rallies/Ezz6HDNHlnk/Ezz6HDNHlnk_rally_0007.mp4",
    "Ezz6HDNHlnk_rally_0008": "/home/arnavchokshi/w3_gpustage_20260707/clips/rallies/Ezz6HDNHlnk/Ezz6HDNHlnk_rally_0008.mp4",
    "HyUqT7zFiwk_rally_0001": "/home/arnavchokshi/prelabel_20260707/data/online_harvest_20260706/rallies/HyUqT7zFiwk/HyUqT7zFiwk_rally_0001.mp4",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MD5_RE = re.compile(r"[0-9a-f]{32}")

# metric_field (per-source / benchmark_cvat_ball_track_candidate label_metrics key)
# -> (pooled/aggregate field name, display name, direction)
METRIC_SPECS: dict[str, dict[str, str]] = {
    "label_f1_at_20px": {
        "pooled_field": "micro_label_f1_at_20px",
        "display": "F1@20",
        "direction": "higher_is_better",
    },
    "visible_recall_at_20px": {
        "pooled_field": "micro_visible_recall_at_20px",
        "display": "Recall@20",
        "direction": "higher_is_better",
    },
    "precision_at_20px": {
        "pooled_field": "micro_precision_at_20px",
        "display": "Precision@20",
        "direction": "higher_is_better",
    },
    "hidden_false_positive_rate": {
        "pooled_field": "micro_hidden_false_positive_rate",
        "display": "HiddenFP",
        "direction": "lower_is_better",
    },
}
METRIC_ORDER: tuple[str, ...] = (
    "label_f1_at_20px",
    "visible_recall_at_20px",
    "precision_at_20px",
    "hidden_false_positive_rate",
)


class LoSOValidationError(ValueError):
    """Raised for invalid CLI input or an attempted strict-holdout-clip scoring touch."""


@dataclass(frozen=True)
class SourceTrack:
    candidate: str
    clip: str
    path: Path
    source_group: str | None = None


@dataclass(frozen=True)
class HeldoutReference:
    candidate: str
    clip: str
    metric: str
    value: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Leave-one-source-out (LoSO) validation harness: partitions existing labeled BALL "
            "predictions by capture source, scores each source independently, and reports a "
            "generalization gap between the pooled/mixed internal-val metric and the LoSO-mean."
        )
    )
    parser.add_argument(
        "--candidate-track",
        action="append",
        default=[],
        metavar="CANDIDATE=CLIP=PATH",
        help=(
            "One (candidate, capture-source clip, existing ball_track.json prediction) triple. "
            "May be repeated; repeat the same CANDIDATE with different CLIP values to add folds. "
            "CLIP must not be a strict-holdout clip (Outdoor/Indoor) -- this script refuses those."
        ),
    )
    parser.add_argument(
        "--heldout-metric",
        action="append",
        default=[],
        metavar="CANDIDATE=CLIP=METRIC=VALUE",
        help=(
            "An already-published held-out metric value (e.g. from "
            "runs/manager/heldout_eval_ledger.md) supplied as a literal for comparison only. "
            "This script never computes or reads these itself. METRIC must be one of: "
            + ", ".join(METRIC_ORDER)
        ),
    )
    parser.add_argument(
        "--source-group",
        action="append",
        default=[],
        metavar="CLIP=SOURCE_GROUP",
        help=(
            "Map a clip to its true recording/game/session/court/device source group. May be "
            "repeated. Clips without a mapping retain the historical clip-as-source behavior."
        ),
    )
    parser.add_argument(
        "--parent-source-split",
        type=Path,
        default=None,
        help=(
            "B0 regroup split directory or validation.jsonl. Uses each validation row's "
            "parent_source_id as the scoring group and its clip_id/frame_index as the frozen row set."
        ),
    )
    parser.add_argument("--cvat-root", type=Path, default=DEFAULT_CVAT_ROOT)
    parser.add_argument(
        "--reviewed-row-list",
        type=Path,
        default=None,
        help=(
            "Optional JSON sample/stratum list containing rows with clip_id+frame_index or row_key. "
            "Only those already-reviewed internal rows are scored."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--hit-radius-px", type=float, default=36.0)
    parser.add_argument("--f1-radius-px", type=float, default=20.0)
    parser.add_argument("--teleport-px-per-frame", type=float, default=160.0)
    parser.add_argument("--max-jump-gap-frames", type=int, default=3)
    args = parser.parse_args(argv)

    try:
        tracks = _parse_candidate_track_specs(args.candidate_track)
        source_groups = _parse_source_group_specs(args.source_group)
        parent_source_contract = None
        cvat_root = args.cvat_root
        if args.parent_source_split is not None:
            if args.reviewed_row_list is not None:
                raise LoSOValidationError(
                    "--parent-source-split already freezes reviewed rows; do not also pass --reviewed-row-list"
                )
            parent_source_contract = _load_parent_source_split(args.parent_source_split)
            source_groups = _merge_source_groups(
                source_groups,
                parent_source_contract["source_group_by_clip"],
            )
            reviewed_row_filter = parent_source_contract["reviewed_frame_indices_by_clip"]
            _validate_parent_source_candidate_coverage(
                tracks,
                expected_clips=set(parent_source_contract["clips"]),
            )
        else:
            reviewed_row_filter = (
                _load_reviewed_row_filter(args.reviewed_row_list)
                if args.reviewed_row_list is not None
                else None
            )
        tracks = _apply_source_groups(tracks, source_groups)
        heldout = _parse_heldout_metric_specs(args.heldout_metric)
        if not tracks:
            raise LoSOValidationError("at least one --candidate-track is required")
        report = build_loso_report(
            tracks=tracks,
            heldout=heldout,
            cvat_root=cvat_root,
            hit_radius_px=args.hit_radius_px,
            f1_radius_px=args.f1_radius_px,
            teleport_px_per_frame=args.teleport_px_per_frame,
            max_jump_gap_frames=args.max_jump_gap_frames,
            reviewed_frame_indices_by_clip=reviewed_row_filter,
            parent_source_contract=parent_source_contract,
        )
    except Exception as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 2

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_json if args.out_json is not None else out_dir / "loso_report.json"
    out_md = args.out_md if args.out_md is not None else out_dir / "loso_report.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "objective_result": report["objective_result"],
                "out_json": str(out_json),
                "out_md": str(out_md),
                "candidate_count": len(report["candidates"]),
                "heldout_comparison_count": len(report["heldout_comparisons"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_candidate_track_specs(specs: Sequence[str]) -> list[SourceTrack]:
    tracks: list[SourceTrack] = []
    for spec in specs:
        parts = spec.split("=", 2)
        if len(parts) != 3:
            raise LoSOValidationError(f"--candidate-track must be CANDIDATE=CLIP=PATH: {spec!r}")
        candidate, clip, path_str = (part.strip() for part in parts)
        if not candidate or not clip or not path_str:
            raise LoSOValidationError(f"--candidate-track has an empty field: {spec!r}")
        tracks.append(SourceTrack(candidate=candidate, clip=clip, path=Path(path_str)))
    return tracks


def _parse_source_group_specs(specs: Sequence[str]) -> dict[str, str]:
    mappings: dict[str, str] = {}
    for spec in specs:
        parts = spec.split("=", 1)
        if len(parts) != 2:
            raise LoSOValidationError(f"--source-group must be CLIP=SOURCE_GROUP: {spec!r}")
        clip, source_group = (part.strip() for part in parts)
        if not clip or not source_group:
            raise LoSOValidationError(f"--source-group has an empty field: {spec!r}")
        previous = mappings.get(clip)
        if previous is not None and previous != source_group:
            raise LoSOValidationError(
                f"conflicting --source-group mappings for {clip!r}: {previous!r} vs {source_group!r}"
            )
        mappings[clip] = source_group
    return mappings


def _merge_source_groups(
    explicit: Mapping[str, str],
    parent_source: Mapping[str, str],
) -> dict[str, str]:
    merged = dict(parent_source)
    for clip, source_group in explicit.items():
        automatic = merged.get(clip)
        if automatic is not None and automatic != source_group:
            raise LoSOValidationError(
                f"--source-group for {clip!r} conflicts with B0 parent_source_id: "
                f"{source_group!r} vs {automatic!r}"
            )
        merged[clip] = source_group
    return merged


def _load_parent_source_split(path: Path) -> dict[str, Any]:
    requested = path / "validation.jsonl" if path.is_dir() else path
    validation_path = _canonical_regular_file(requested, "B0 validation JSONL")
    if validation_path.name != "validation.jsonl":
        raise LoSOValidationError(
            f"--parent-source-split must name a split directory or validation.jsonl: {validation_path}"
        )
    split_dir = validation_path.parent
    report_path = _canonical_regular_file(split_dir / "report.json", "B0 split report")
    report = _read_json_object(report_path, "B0 split report")
    artifacts = _validate_b0_split_report(report, report_path=report_path, validation_path=validation_path)

    rows = _read_jsonl_objects(validation_path, "B0 validation row")
    if not rows:
        raise LoSOValidationError(f"B0 validation split is empty: {validation_path}")

    lineage_path = artifacts["lineage_rows"]
    lineage_rows = _read_jsonl_objects(lineage_path, "B0 lineage row")
    lineage_by_key: dict[str, Mapping[str, Any]] = {}
    for lineage in lineage_rows:
        key = str(lineage.get("row_key") or "")
        if not key or key in lineage_by_key:
            raise LoSOValidationError(f"B0 lineage has missing/duplicate row_key: {key!r}")
        lineage_by_key[key] = lineage

    input_contract = report.get("input_contract")
    if not isinstance(input_contract, Mapping):
        raise LoSOValidationError("B0 split report requires input_contract")
    sampling_value = input_contract.get("scratch_sampling_manifest")
    if not isinstance(sampling_value, str) or not sampling_value.strip():
        raise LoSOValidationError("B0 split report requires input_contract.scratch_sampling_manifest")
    sampling_path = _canonical_regular_file(
        _declared_path(sampling_value), "B0 scratch sampling manifest"
    )
    expected_sampling_md5 = str(input_contract.get("scratch_sampling_manifest_md5") or "")
    if not _MD5_RE.fullmatch(expected_sampling_md5):
        raise LoSOValidationError("B0 split report requires lowercase scratch_sampling_manifest_md5")
    actual_sampling_md5 = _file_digest(sampling_path, "md5")
    if actual_sampling_md5 != expected_sampling_md5:
        raise LoSOValidationError(
            "B0 scratch sampling manifest MD5 mismatch: "
            f"expected={expected_sampling_md5} actual={actual_sampling_md5}"
        )
    sampling = _read_json_object(sampling_path, "B0 scratch sampling manifest")
    samples_by_key = _sampling_rows_by_key(sampling)
    universe_by_clip = _sampling_universe_by_clip(sampling)

    frozen_path = validation_path == (FROZEN_B0_SPLIT_DIR / "validation.jsonl").resolve()
    if frozen_path:
        for name, expected_sha in FROZEN_B0_ARTIFACT_SHA256.items():
            artifact_path = _canonical_regular_file(split_dir / name, f"frozen B0 {name}")
            actual_sha = _file_digest(artifact_path, "sha256")
            if actual_sha != expected_sha:
                raise LoSOValidationError(
                    f"frozen B0 {name} SHA-256 mismatch: expected={expected_sha} actual={actual_sha}"
                )

    source_group_by_clip: dict[str, str] = {}
    reviewed: dict[str, set[int]] = {}
    rows_by_clip: dict[str, list[Mapping[str, Any]]] = {}
    source_video_by_clip: dict[str, str] = {}
    source_video_sha256_by_clip: dict[str, str] = {}
    source_video_contract_by_clip: dict[str, Mapping[str, Any]] = {}
    seen_rows: set[tuple[str, int]] = set()
    media_sha_cache: dict[Path, str] = {}

    for row in rows:
        clip, parent, frame_index = _validate_b0_validation_row(row, validation_path)
        row_key_text = str(row["row_key"])
        lineage = lineage_by_key.get(row_key_text)
        if lineage is None:
            raise LoSOValidationError(f"B0 validation row is absent from lineage_rows.jsonl: {row_key_text}")
        _validate_b0_lineage_match(row, lineage)
        sample = samples_by_key.get(row_key_text)
        if sample is None:
            raise LoSOValidationError(f"B0 validation row is absent from scratch sampling: {row_key_text}")
        _validate_b0_sampling_match(row, sample)

        universe = universe_by_clip.get(clip)
        if universe is None:
            raise LoSOValidationError(f"B0 validation clip is absent from sampling universe: {clip}")
        _validate_source_alias_fields(lineage, expected_clip=clip, expected_parent=parent)
        _validate_source_alias_fields(sample, expected_clip=clip, expected_parent=parent)
        _validate_source_alias_fields(universe, expected_clip=clip, expected_parent=parent)
        _validate_source_alias_fields(row, expected_clip=clip, expected_parent=parent)
        media_paths = _source_media_paths(row, lineage, sample, universe)
        if not media_paths:
            raise LoSOValidationError(f"B0 row has no canonical source-video path: {row_key_text}")
        canonical_media = {
            _canonical_regular_file(_declared_path(value), f"B0 source video for {clip}")
            for value in media_paths
        }
        if len(canonical_media) != 1:
            raise LoSOValidationError(
                f"conflicting video/source_video aliases for {clip}: "
                f"{sorted(str(value) for value in canonical_media)}"
            )
        media_path = next(iter(canonical_media))
        _assert_canonical_clip_media_path(media_path, clip=clip, parent=parent)

        expected_sha = FROZEN_B0_SOURCE_VIDEO_SHA256.get(clip)
        if expected_sha is None:
            expected_sha = _declared_source_video_sha256(row, lineage, sample, universe)
        elif _declared_source_video_sha256(row, lineage, sample, universe, required=False) not in {
            None,
            expected_sha,
        }:
            raise LoSOValidationError(f"declared source-video SHA conflicts with frozen SHA for {clip}")
        if expected_sha is None or not _SHA256_RE.fullmatch(expected_sha):
            raise LoSOValidationError(f"B0 source video lacks an expected SHA-256 for {clip}")
        if media_path not in media_sha_cache:
            media_sha_cache[media_path] = _file_digest(media_path, "sha256")
        actual_sha = media_sha_cache[media_path]
        if actual_sha != expected_sha:
            raise LoSOValidationError(
                f"B0 source-video SHA-256 mismatch for {clip}: expected={expected_sha} actual={actual_sha}"
            )

        previous_parent = source_group_by_clip.get(clip)
        if previous_parent is not None and previous_parent != parent:
            raise LoSOValidationError(
                f"clip {clip!r} maps to conflicting parent_source_id values: "
                f"{previous_parent!r} vs {parent!r}"
            )
        source_group_by_clip[clip] = parent
        pair = (clip, frame_index)
        if pair in seen_rows:
            raise LoSOValidationError(f"duplicate B0 validation row: {clip}:{frame_index}")
        seen_rows.add(pair)
        reviewed.setdefault(clip, set()).add(frame_index)
        rows_by_clip.setdefault(clip, []).append(row)
        source_video_by_clip[clip] = str(media_path)
        source_video_sha256_by_clip[clip] = actual_sha
        source_video_contract_by_clip[clip] = dict(universe)

    clips = set(source_group_by_clip)
    frozen_clips = set(FROZEN_B0_SOURCE_VIDEO_SHA256)
    if clips == frozen_clips and not frozen_path:
        raise LoSOValidationError(
            "the accepted B0 validation identity must be consumed from its canonical split path; "
            f"expected={FROZEN_B0_SPLIT_DIR / 'validation.jsonl'} actual={validation_path}"
        )
    if frozen_path and clips != frozen_clips:
        raise LoSOValidationError(
            f"frozen B0 clip set mismatch: missing={sorted(frozen_clips - clips)} "
            f"unexpected={sorted(clips - frozen_clips)}"
        )
    expected_count = report.get("split_counts", {}).get("validation")
    if type(expected_count) is not int or expected_count != len(rows):
        raise LoSOValidationError(
            f"B0 report validation count mismatch: report={expected_count!r} rows={len(rows)}"
        )
    report_sources = report.get("validation_sources")
    if not isinstance(report_sources, list) or sorted(report_sources) != sorted(set(source_group_by_clip.values())):
        raise LoSOValidationError("B0 report validation_sources do not match derived parent sources")

    return {
        "mode": "b0_parent_source_split",
        "label_source": "validation_jsonl.final_label",
        "identity_mode": "frozen_b0_20260721" if frozen_path else "structurally_bound_fixture",
        "validation_path": str(validation_path),
        "validation_sha256": _file_digest(validation_path, "sha256"),
        "report_path": str(report_path),
        "report_sha256": _file_digest(report_path, "sha256"),
        "lineage_path": str(lineage_path),
        "lineage_sha256": _file_digest(lineage_path, "sha256"),
        "sampling_path": str(sampling_path),
        "sampling_md5": actual_sampling_md5,
        "row_count": len(rows),
        "clip_count": len(source_group_by_clip),
        "parent_source_count": len(set(source_group_by_clip.values())),
        "clips": sorted(source_group_by_clip),
        "parent_sources": sorted(set(source_group_by_clip.values())),
        "source_group_by_clip": dict(sorted(source_group_by_clip.items())),
        "reviewed_frame_indices_by_clip": {
            clip: sorted(indices) for clip, indices in sorted(reviewed.items())
        },
        "validation_rows_by_clip": {
            clip: sorted(clip_rows, key=lambda item: int(item["frame_index"]))
            for clip, clip_rows in sorted(rows_by_clip.items())
        },
        "source_video_by_clip": dict(sorted(source_video_by_clip.items())),
        "source_video_sha256_by_clip": dict(sorted(source_video_sha256_by_clip.items())),
        "source_video_contract_by_clip": dict(sorted(source_video_contract_by_clip.items())),
    }


def _validate_b0_split_report(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    validation_path: Path,
) -> dict[str, Path]:
    if report.get("artifact_type") != "racketsport_ball_regroup_parent_source_split":
        raise LoSOValidationError("unexpected B0 split report artifact_type")
    if report.get("split_semantics") != "parent_source" or report.get("verdict") != "BALL_CLEAN_JUDGE":
        raise LoSOValidationError("B0 split report is not the accepted parent-source clean judge")
    checks = report.get("checks")
    required_checks = (
        "evaluation_lineage",
        "protected_collision_count",
        "scratch_package_reconciled",
        "train_validation_source_intersection",
    )
    if not isinstance(checks, Mapping) or any(
        not isinstance(checks.get(name), Mapping) or checks[name].get("verdict") != "PASS"
        for name in required_checks
    ):
        raise LoSOValidationError("B0 split report required gates are not all PASS")
    declarations = report.get("artifacts")
    if not isinstance(declarations, Mapping):
        raise LoSOValidationError("B0 split report requires artifacts")
    resolved: dict[str, Path] = {}
    for key, filename in {
        "report": "report.json",
        "train": "train.jsonl",
        "validation": "validation.jsonl",
        "lineage_rows": "lineage_rows.jsonl",
    }.items():
        value = declarations.get(key)
        if not isinstance(value, str) or not value.strip():
            raise LoSOValidationError(f"B0 split report artifacts.{key} is required")
        declared = _canonical_regular_file(_declared_path(value), f"B0 artifacts.{key}")
        expected = _canonical_regular_file(report_path.parent / filename, f"B0 {filename}")
        if declared != expected:
            raise LoSOValidationError(
                f"B0 artifacts.{key} canonical path conflict: declared={declared} expected={expected}"
            )
        resolved[key] = declared
    if resolved["report"] != report_path or resolved["validation"] != validation_path:
        raise LoSOValidationError("B0 report/validation canonical identity mismatch")
    return resolved


def _validate_b0_validation_row(row: Mapping[str, Any], path: Path) -> tuple[str, str, int]:
    clip = str(row.get("clip_id") or "").strip()
    parent = str(row.get("parent_source_id") or "").strip()
    source_id = str(row.get("source_id") or "").strip()
    frame_value = row.get("frame_index")
    if not clip or not parent or type(frame_value) is not int or frame_value < 0:
        raise LoSOValidationError(
            f"B0 validation row requires clip_id, parent_source_id, and nonnegative integer frame_index: {row!r}"
        )
    derived_parent = _parent_from_clip(clip)
    if parent != derived_parent or source_id != derived_parent:
        raise LoSOValidationError(
            f"B0 caller-swapped parent/source for {clip}: derived={derived_parent!r} "
            f"parent_source_id={parent!r} source_id={source_id!r}"
        )
    _assert_scoreable_source(clip, context=f"--parent-source-split {path}")
    _assert_scoreable_source(parent, context=f"--parent-source-split {path}")
    if row.get("split") != "validation":
        raise LoSOValidationError(f"B0 parent-source mode accepts validation rows only: {row!r}")
    if row.get("evaluation_eligible") is not True or row.get("ground_truth") is not True:
        raise LoSOValidationError(f"B0 validation row is not independent evaluation truth: {clip}:{frame_value}")
    if row.get("lineage_class") != "scratch" or row.get("teacher_derived") is not False:
        raise LoSOValidationError(f"B0 validation row must be scratch and non-teacher-derived: {clip}:{frame_value}")
    if row.get("original_prelabel") is not None or float(row.get("training_weight", -1.0)) != 1.0:
        raise LoSOValidationError(f"B0 scratch validation authority fields are invalid: {clip}:{frame_value}")

    expected_row_key = f"{clip}:{frame_value:06d}"
    if row.get("row_key") != expected_row_key:
        raise LoSOValidationError(
            f"B0 row_key identity mismatch: expected={expected_row_key!r} actual={row.get('row_key')!r}"
        )
    expected_image = f"{parent}__{clip}__abs_{frame_value:06d}.png"
    if row.get("image_name") != expected_image or row.get("image_zip_member") != expected_image:
        raise LoSOValidationError(
            f"B0 image identity mismatch for {expected_row_key}: expected={expected_image!r}"
        )
    image_md5 = str(row.get("image_md5") or "")
    if not _MD5_RE.fullmatch(image_md5):
        raise LoSOValidationError(f"B0 image_md5 is invalid for {expected_row_key}")
    width, height = row.get("image_width"), row.get("image_height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise LoSOValidationError(f"B0 image dimensions are invalid for {expected_row_key}")
    ordinal = row.get("sample_ordinal")
    if type(ordinal) is not int or ordinal < 0:
        raise LoSOValidationError(f"B0 sample_ordinal must be nonnegative for {expected_row_key}")
    _validate_final_label(row.get("final_label"), width=width, height=height, row_key=expected_row_key)
    return clip, parent, frame_value


def _validate_final_label(label: Any, *, width: int, height: int, row_key: str) -> None:
    if not isinstance(label, Mapping) or type(label.get("ball_present")) is not bool:
        raise LoSOValidationError(f"B0 final_label.ball_present must be boolean: {row_key}")
    bbox = label.get("bbox_xyxy")
    if label["ball_present"] is False:
        if bbox is not None:
            raise LoSOValidationError(f"B0 negative final_label must have bbox_xyxy=null: {row_key}")
        return
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise LoSOValidationError(f"B0 positive final_label requires four-value bbox_xyxy: {row_key}")
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise LoSOValidationError(f"B0 bbox is not numeric: {row_key}") from exc
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise LoSOValidationError(f"B0 bbox is not finite: {row_key}")
    if x1 < 0.0 or y1 < 0.0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
        raise LoSOValidationError(f"B0 bbox is out of image bounds: {row_key}")


def _validate_b0_lineage_match(row: Mapping[str, Any], lineage: Mapping[str, Any]) -> None:
    for key in (
        "clip_id",
        "evaluation_eligible",
        "final_label",
        "frame_index",
        "ground_truth",
        "lineage_class",
        "lineage_origin",
        "original_prelabel",
        "parent_source_id",
        "row_key",
        "source_class",
        "source_id",
        "split",
        "teacher_derived",
        "training_weight",
    ):
        if lineage.get(key) != row.get(key):
            raise LoSOValidationError(f"B0 lineage mismatch for {row['row_key']} field {key}")


def _validate_b0_sampling_match(row: Mapping[str, Any], sample: Mapping[str, Any]) -> None:
    expected = {
        "rally_id": row["clip_id"],
        "frame_index": row["frame_index"],
        "row_key": row["row_key"],
        "source_id": row["source_id"],
        "source_class": row["source_class"],
        "sample_ordinal": row["sample_ordinal"],
        "image_name": row["image_name"],
        "image_zip_member": row["image_zip_member"],
        "image_md5": row["image_md5"],
        "image_width": row["image_width"],
        "image_height": row["image_height"],
    }
    for key, value in expected.items():
        if sample.get(key) != value:
            raise LoSOValidationError(f"B0 sampling mismatch for {row['row_key']} field {key}")


def _sampling_rows_by_key(sampling: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    frames = sampling.get("frames")
    if not isinstance(frames, list):
        raise LoSOValidationError("B0 scratch sampling manifest requires frames")
    result: dict[str, Mapping[str, Any]] = {}
    for row in frames:
        if not isinstance(row, Mapping):
            raise LoSOValidationError("B0 scratch sampling frames must be objects")
        key = str(row.get("row_key") or "")
        if not key or key in result:
            raise LoSOValidationError(f"B0 scratch sampling has missing/duplicate row_key: {key!r}")
        result[key] = row
    return result


def _sampling_universe_by_clip(sampling: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    videos = sampling.get("universe_videos")
    if not isinstance(videos, list):
        raise LoSOValidationError("B0 scratch sampling manifest requires universe_videos")
    result: dict[str, Mapping[str, Any]] = {}
    for video in videos:
        if not isinstance(video, Mapping):
            raise LoSOValidationError("B0 sampling universe_videos must contain objects")
        clip = str(video.get("rally_id") or "")
        if not clip or clip in result:
            raise LoSOValidationError(f"B0 sampling universe has missing/duplicate rally_id: {clip!r}")
        result[clip] = video
    return result


def _parent_from_clip(clip: str) -> str:
    parent, marker, ordinal = clip.rpartition("_rally_")
    if not marker or not parent or len(ordinal) != 4 or not ordinal.isdigit():
        raise LoSOValidationError(f"B0 clip_id does not encode canonical parent source: {clip!r}")
    return parent


def _source_media_paths(*objects: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for obj in objects:
        for key in ("video_path", "video", "source_video"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value)
    return values


def _validate_source_alias_fields(
    obj: Mapping[str, Any],
    *,
    expected_clip: str,
    expected_parent: str,
) -> None:
    values = [
        value
        for key in ("video_path", "video", "source_video")
        if isinstance((value := obj.get(key)), str) and value.strip()
    ]
    identities = {_structured_media_identity(Path(value)) for value in values}
    if len(identities) > 1:
        raise LoSOValidationError(
            f"conflicting video/source_video aliases for {expected_clip}: {sorted(identities)}"
        )
    if identities and next(iter(identities)) != ("rallies", expected_parent, f"{expected_clip}.mp4"):
        raise LoSOValidationError(
            f"source-video alias is not canonically bound to {expected_clip}: {sorted(identities)}"
        )


def _structured_media_identity(path: Path) -> tuple[str, str, str]:
    parts = path.parts
    if len(parts) < 3 or parts[-3] != "rallies" or path.suffix.lower() != ".mp4":
        raise LoSOValidationError(f"source-video path lacks canonical rallies/source/clip structure: {path}")
    return parts[-3], parts[-2], parts[-1]


def _assert_canonical_clip_media_path(path: Path, *, clip: str, parent: str) -> None:
    identity = _structured_media_identity(path)
    expected = ("rallies", parent, f"{clip}.mp4")
    if identity != expected:
        raise LoSOValidationError(f"canonical source-video identity mismatch: expected={expected} actual={identity}")


def _declared_source_video_sha256(
    *objects: Mapping[str, Any],
    required: bool = True,
) -> str | None:
    values = {
        str(value).lower()
        for obj in objects
        for key in ("source_video_sha256", "video_sha256", "media_sha256")
        if isinstance((value := obj.get(key)), str) and value.strip()
    }
    if len(values) > 1:
        raise LoSOValidationError(f"conflicting declared source-video SHA-256 values: {sorted(values)}")
    value = next(iter(values), None)
    if value is not None and not _SHA256_RE.fullmatch(value):
        raise LoSOValidationError(f"invalid declared source-video SHA-256: {value!r}")
    if required and value is None:
        return None
    return value


def _declared_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _canonical_regular_file(path: Path, description: str) -> Path:
    lexical = _declared_path(path)
    if ".." in lexical.parts:
        raise LoSOValidationError(f"{description} contains parent traversal: {path}")
    lexical = lexical.absolute()
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if current.is_symlink():
            raise LoSOValidationError(f"{description} may not use symlinks: {current}")
    try:
        resolved = lexical.resolve(strict=True)
    except FileNotFoundError as exc:
        raise LoSOValidationError(f"missing {description}: {lexical}") from exc
    if resolved != lexical or not resolved.is_file():
        raise LoSOValidationError(
            f"{description} canonical path conflict or non-file: lexical={lexical} resolved={resolved}"
        )
    return resolved


def _read_json_object(path: Path, description: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LoSOValidationError(f"invalid JSON in {description} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise LoSOValidationError(f"{description} must contain a JSON object: {path}")
    return payload


def _read_jsonl_objects(path: Path, description: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LoSOValidationError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(row, Mapping):
            raise LoSOValidationError(f"{description} must be an object: {path}:{line_number}")
        rows.append(row)
    return rows


def _file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_parent_source_candidate_coverage(
    tracks: Sequence[SourceTrack],
    *,
    expected_clips: set[str],
) -> None:
    clips_by_candidate: dict[str, list[str]] = {}
    for track in tracks:
        clips_by_candidate.setdefault(track.candidate, []).append(track.clip)
    for candidate, clip_list in clips_by_candidate.items():
        duplicates = sorted({clip for clip in clip_list if clip_list.count(clip) > 1})
        if duplicates:
            raise LoSOValidationError(
                f"candidate {candidate!r} repeats B0 validation clips: {duplicates}"
            )
        actual = set(clip_list)
        if actual != expected_clips:
            raise LoSOValidationError(
                f"candidate {candidate!r} does not cover the frozen B0 validation clips: "
                f"missing={sorted(expected_clips - actual)} unexpected={sorted(actual - expected_clips)}"
            )


def _validate_parent_source_prediction_bindings(
    tracks: Sequence[SourceTrack],
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    bindings: dict[str, dict[str, Any]] = {}
    for track in tracks:
        track_path = _canonical_regular_file(
            track.path, f"candidate prediction for {track.candidate}/{track.clip}"
        )
        if track_path.name != "ball_track.json":
            raise LoSOValidationError(f"candidate prediction must be named ball_track.json: {track_path}")
        path_clip = _clip_from_prediction_path(track_path)
        if path_clip != track.clip:
            raise LoSOValidationError(
                f"candidate prediction canonical path is bound to {path_clip!r}, not caller clip {track.clip!r}"
            )

        metadata_path = _canonical_regular_file(
            track_path.with_name("ball_track_metadata.json"),
            f"candidate metadata for {track.candidate}/{track.clip}",
        )
        metadata = _read_json_object(metadata_path, "candidate ball-track metadata")
        if metadata.get("artifact_type") != "racketsport_wasb_ball_run":
            raise LoSOValidationError(f"unexpected candidate metadata artifact_type: {metadata_path}")
        runtime = metadata.get("runtime")
        if not isinstance(runtime, Mapping):
            raise LoSOValidationError(f"candidate metadata lacks runtime object: {metadata_path}")

        output_values = [
            value
            for key in ("out", "ball_track_path", "prediction_path")
            if isinstance((value := metadata.get(key)), str) and value.strip()
        ]
        if not output_values:
            raise LoSOValidationError(f"candidate metadata does not declare its output path: {metadata_path}")
        output_clips = {_clip_from_prediction_path(Path(value)) for value in output_values}
        if output_clips != {track.clip}:
            raise LoSOValidationError(
                f"candidate metadata output aliases conflict with clip {track.clip}: {sorted(output_clips)}"
            )

        source_video = _canonical_regular_file(
            Path(str(contract["source_video_by_clip"][track.clip])),
            f"canonical source video for prediction {track.clip}",
        )
        metadata_video_binding = _validate_prediction_video_aliases(
            metadata,
            runtime,
            clip=track.clip,
            parent=str(contract["source_group_by_clip"][track.clip]),
            canonical_source_video=source_video,
            identity_mode=str(contract.get("identity_mode") or ""),
        )
        expected_video_sha = str(contract["source_video_sha256_by_clip"][track.clip])
        actual_video_sha = _file_digest(source_video, "sha256")
        if actual_video_sha != expected_video_sha:
            raise LoSOValidationError(
                f"prediction source-video SHA-256 mismatch for {track.clip}: "
                f"expected={expected_video_sha} actual={actual_video_sha}"
            )
        declared_video_sha = _declared_source_video_sha256(metadata, runtime, required=False)
        if declared_video_sha is not None and declared_video_sha != expected_video_sha:
            raise LoSOValidationError(
                f"prediction metadata source-video SHA-256 mismatch for {track.clip}: "
                f"expected={expected_video_sha} actual={declared_video_sha}"
            )

        video_contract = contract["source_video_contract_by_clip"][track.clip]
        expected_frames = video_contract.get("frame_count")
        expected_width = video_contract.get("width")
        expected_height = video_contract.get("height")
        if runtime.get("source_video_frame_count") != expected_frames:
            raise LoSOValidationError(
                f"prediction metadata frame count mismatch for {track.clip}: "
                f"expected={expected_frames!r} actual={runtime.get('source_video_frame_count')!r}"
            )
        if runtime.get("source_video_size") != [expected_width, expected_height]:
            raise LoSOValidationError(
                f"prediction metadata source-video size mismatch for {track.clip}: "
                f"expected={[expected_width, expected_height]!r} actual={runtime.get('source_video_size')!r}"
            )
        parsed_track = load_ball_track(track_path)
        parsed_frame_count = cvat_benchmark._track_frame_count(parsed_track)
        if parsed_frame_count != expected_frames or metadata.get("frame_count") != expected_frames:
            raise LoSOValidationError(
                f"prediction/metadata horizon mismatch for {track.clip}: "
                f"prediction={parsed_frame_count} metadata={metadata.get('frame_count')!r} "
                f"source={expected_frames!r}"
            )

        candidate_bindings = bindings.setdefault(track.candidate, {})
        candidate_bindings[track.clip] = {
            "clip_id": track.clip,
            "parent_source_id": contract["source_group_by_clip"][track.clip],
            "prediction_path": str(track_path),
            "prediction_sha256": _file_digest(track_path, "sha256"),
            "metadata_path": str(metadata_path),
            "metadata_sha256": _file_digest(metadata_path, "sha256"),
            "canonical_source_video_path": str(source_video),
            "canonical_source_video_sha256": actual_video_sha,
            "metadata_source_video_path": metadata_video_binding["path"],
            "metadata_source_video_path_mode": metadata_video_binding["mode"],
            "metadata_source_video_identity": metadata_video_binding["identity"],
            "frame_count": parsed_frame_count,
        }
    return {candidate: dict(sorted(rows.items())) for candidate, rows in sorted(bindings.items())}


def _clip_from_prediction_path(path: Path) -> str:
    if path.name != "ball_track.json":
        raise LoSOValidationError(f"prediction output alias must end in ball_track.json: {path}")
    parent = path.parent
    if parent.name == "wasb":
        parent = parent.parent
    clip = parent.name
    if not clip:
        raise LoSOValidationError(f"cannot derive clip identity from prediction path: {path}")
    return clip


def _validate_prediction_video_aliases(
    metadata: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    clip: str,
    parent: str,
    canonical_source_video: Path,
    identity_mode: str,
) -> dict[str, Any]:
    aliases = [
        (f"{scope}.{key}", value.strip())
        for scope, obj in (("metadata", metadata), ("runtime", runtime))
        for key in ("video", "source_video", "video_path")
        if isinstance((value := obj.get(key)), str) and value.strip()
    ]
    if not aliases:
        raise LoSOValidationError("candidate metadata lacks runtime.video")
    normalized_paths: dict[str, list[str]] = {}
    for field, value in aliases:
        path = Path(value)
        if ".." in path.parts:
            raise LoSOValidationError(
                f"prediction metadata source-video alias contains parent traversal for {clip}: "
                f"{field}={value!r}"
            )
        normalized_paths.setdefault(str(path), []).append(field)
    if len(normalized_paths) != 1:
        raise LoSOValidationError(
            f"conflicting video/source_video aliases in prediction metadata for {clip}; "
            f"aliases disagree on canonical path identity: {normalized_paths}"
        )

    declared_text = next(iter(normalized_paths))
    declared_path = Path(declared_text)
    expected_identity = ("rallies", parent, f"{clip}.mp4")
    declared_identity = _structured_media_identity(declared_path)
    if declared_identity != expected_identity:
        raise LoSOValidationError(
            f"prediction metadata source-video identity mismatch for {clip}: "
            f"expected={expected_identity} actual={declared_identity}"
        )

    local_candidate = _declared_path(declared_path)
    if local_candidate.exists() or local_candidate.is_symlink():
        resolved = _canonical_regular_file(
            local_candidate, f"prediction metadata source-video alias for {clip}"
        )
        if resolved != canonical_source_video:
            raise LoSOValidationError(
                f"prediction metadata source-video canonical path conflict for {clip}: "
                f"declared={resolved} expected={canonical_source_video}"
            )
        mode = "canonical_local"
    else:
        frozen_legacy = FROZEN_B0_LEGACY_PREDICTION_VIDEO_PATH.get(clip)
        if identity_mode != "frozen_b0_20260721" or declared_text != frozen_legacy:
            raise LoSOValidationError(
                f"prediction metadata source-video path is neither canonical-local nor the frozen "
                f"legacy path for {clip}: {declared_text}"
            )
        mode = "frozen_legacy_remote"
    return {
        "path": declared_text,
        "mode": mode,
        "identity": list(expected_identity),
    }


def _apply_source_groups(tracks: Sequence[SourceTrack], mappings: Mapping[str, str]) -> list[SourceTrack]:
    tracked_clips = {track.clip for track in tracks}
    unknown = sorted(set(mappings) - tracked_clips)
    if unknown:
        raise LoSOValidationError(f"--source-group names clips without candidate tracks: {unknown}")
    return [
        SourceTrack(
            candidate=track.candidate,
            clip=track.clip,
            path=track.path,
            source_group=mappings.get(track.clip, track.clip),
        )
        for track in tracks
    ]


def _parse_heldout_metric_specs(specs: Sequence[str]) -> list[HeldoutReference]:
    references: list[HeldoutReference] = []
    for spec in specs:
        parts = spec.split("=", 3)
        if len(parts) != 4:
            raise LoSOValidationError(f"--heldout-metric must be CANDIDATE=CLIP=METRIC=VALUE: {spec!r}")
        candidate, clip, metric, value_str = (part.strip() for part in parts)
        if not candidate or not clip or not metric:
            raise LoSOValidationError(f"--heldout-metric has an empty field: {spec!r}")
        if metric not in METRIC_SPECS:
            raise LoSOValidationError(
                f"--heldout-metric metric must be one of {sorted(METRIC_SPECS)}: {spec!r}"
            )
        try:
            value = float(value_str)
        except ValueError as exc:
            raise LoSOValidationError(f"--heldout-metric value must be a float: {spec!r}") from exc
        references.append(HeldoutReference(candidate=candidate, clip=clip, metric=metric, value=value))
    return references


def _load_reviewed_row_filter(path: Path) -> dict[str, list[int]]:
    if not path.is_file():
        raise LoSOValidationError(f"missing --reviewed-row-list JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise LoSOValidationError("--reviewed-row-list must contain a non-empty rows array")
    by_clip: dict[str, set[int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise LoSOValidationError("--reviewed-row-list rows must be objects")
        clip = str(row.get("clip_id") or "").strip()
        frame_value = row.get("frame_index")
        row_key = str(row.get("row_key") or "")
        if (not clip or frame_value is None) and ":" in row_key:
            clip, frame_token = row_key.rsplit(":", 1)
            frame_value = frame_token
        if not clip or frame_value is None:
            raise LoSOValidationError(f"reviewed row lacks clip_id/frame_index: {row!r}")
        _assert_scoreable_source(clip, context=f"--reviewed-row-list {path}")
        by_clip.setdefault(clip, set()).add(int(frame_value))
    return {clip: sorted(indices) for clip, indices in sorted(by_clip.items())}


def _assert_scoreable_source(clip_id: str, *, context: str) -> None:
    """Fail closed if ``clip_id`` names (or embeds) a strict-holdout clip.

    This lane never scores against Outdoor/Indoor labels, independent of and in
    addition to ``threed/racketsport/eval_guard.py``'s own guard (that guard protects
    training/validation-during-fitting inputs; this check protects this script's own
    read path against reviewed_boxes.json).
    """

    lowered = clip_id.lower()
    for strict_id in STRICT_HOLDOUT_CLIP_IDS:
        if strict_id in lowered:
            raise LoSOValidationError(
                f"{context}: refusing to score strict held-out clip {strict_id!r} (matched in "
                f"{clip_id!r}). This lane never reads or scores against protected Outdoor/Indoor "
                "labels. Supply its already-published number via --heldout-metric instead."
            )


def build_loso_report(
    *,
    tracks: Sequence[SourceTrack],
    heldout: Sequence[HeldoutReference],
    cvat_root: Path,
    hit_radius_px: float,
    f1_radius_px: float,
    teleport_px_per_frame: float,
    max_jump_gap_frames: int,
    reviewed_frame_indices_by_clip: Mapping[str, Sequence[int]] | None = None,
    parent_source_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if hit_radius_px < 0.0 or f1_radius_px <= 0.0 or teleport_px_per_frame <= 0.0:
        raise LoSOValidationError("scoring radii and teleport threshold must be finite and positive")
    if not all(math.isfinite(value) for value in (hit_radius_px, f1_radius_px, teleport_px_per_frame)):
        raise LoSOValidationError("scoring radii and teleport threshold must be finite")
    if max_jump_gap_frames < 1:
        raise LoSOValidationError("max_jump_gap_frames must be >= 1")

    by_candidate: dict[str, list[SourceTrack]] = {}
    for track in tracks:
        _assert_scoreable_source(track.clip, context=f"--candidate-track {track.candidate}={track.clip}")
        _canonical_regular_file(track.path, f"ball_track.json for {track.candidate}/{track.clip}")
        by_candidate.setdefault(track.candidate, []).append(track)

    prediction_bindings: dict[str, Any] = {}
    if parent_source_contract is not None:
        expected_clips = set(parent_source_contract["clips"])
        _validate_parent_source_candidate_coverage(tracks, expected_clips=expected_clips)
        prediction_bindings = _validate_parent_source_prediction_bindings(
            tracks,
            contract=parent_source_contract,
        )

    candidates_report: dict[str, Any] = {}
    for name, candidate_tracks in sorted(by_candidate.items()):
        if reviewed_frame_indices_by_clip is not None:
            candidate_tracks = [
                track for track in candidate_tracks if track.clip in reviewed_frame_indices_by_clip
            ]
            if not candidate_tracks:
                raise LoSOValidationError(f"candidate {name!r} has no clips in --reviewed-row-list")
        candidates_report[name] = _score_candidate(
            name,
            candidate_tracks,
            cvat_root=cvat_root,
            hit_radius_px=hit_radius_px,
            f1_radius_px=f1_radius_px,
            teleport_px_per_frame=teleport_px_per_frame,
            max_jump_gap_frames=max_jump_gap_frames,
            reviewed_frame_indices_by_clip=reviewed_frame_indices_by_clip,
            parent_source_contract=parent_source_contract,
        )

    heldout_by_key: dict[tuple[str, str, str], float] = {}
    for reference in heldout:
        heldout_by_key[(reference.candidate, reference.clip, reference.metric)] = reference.value

    comparisons = _build_heldout_comparisons(candidates_report, heldout)

    fold_counts = {name: row["fold_count"] for name, row in candidates_report.items()}
    objective_result = (
        "PASS"
        if candidates_report and any(count >= 2 for count in fold_counts.values())
        else "PARTIAL"
    )

    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": "TESTED-ON-REAL-DATA",
        "ball_verified": False,
        "objective_result": objective_result,
        "cvat_root": str(cvat_root) if parent_source_contract is None else None,
        "label_source": (
            "cvat_reviewed_boxes" if parent_source_contract is None else "b0_validation_final_label"
        ),
        "internal_val_only_clip_ids": list(INTERNAL_VAL_ONLY_CLIP_IDS),
        "strict_holdout_clip_ids_never_scored": list(STRICT_HOLDOUT_CLIP_IDS),
        "metric_specs": METRIC_SPECS,
        "candidates": candidates_report,
        "heldout_reference_count": len(heldout),
        "heldout_references": [
            {"candidate": key[0], "clip": key[1], "metric": key[2], "value": value}
            for key, value in sorted(heldout_by_key.items())
        ],
        "heldout_comparisons": comparisons,
        "reviewed_row_filter": (
            {
                "clip_count": len(reviewed_frame_indices_by_clip),
                "row_count": sum(len(indices) for indices in reviewed_frame_indices_by_clip.values()),
            }
            if reviewed_frame_indices_by_clip is not None
            else None
        ),
        "source_grouping_mode": (
            "b0_parent_source_split" if parent_source_contract is not None else "explicit_or_clip_compatibility"
        ),
        "parent_source_split": (
            {
                key: value
                for key, value in parent_source_contract.items()
                if key
                not in {
                    "reviewed_frame_indices_by_clip",
                    "source_group_by_clip",
                    "validation_rows_by_clip",
                    "source_video_by_clip",
                    "source_video_contract_by_clip",
                }
            }
            if parent_source_contract is not None
            else None
        ),
        "prediction_artifacts": prediction_bindings,
        "methodology": {
            "capture_source_definition": (
                "An explicit --source-group is the conservative recording/game/session/court/device "
                "identity shared by every clip from that source. Without mappings, the historical "
                "compatibility behavior treats each CVAT clip id as a source. Never a random frame split."
            ),
            "pooled_mixed_metric": (
                "Micro-average across every scored source under one candidate name -- reproduces "
                "today's internal-val process exactly (see benchmark_cvat_ball_tracker_candidates' "
                "'aggregate')."
            ),
            "loso_mean_metric": (
                "Unweighted mean of each source's own independently-scored metric. Candidates here "
                "are frozen/zero-shot detectors (not retrained per fold), so 'leave-one-source-out' "
                "means: score each source on its own, then average -- simulating what the held-out "
                "score would have looked like had only the other source(s) been available during "
                "development."
            ),
            "loso_worst_fold_metric": "The single worst per-source score across folds (a conservative estimate).",
            "generalization_gap": "pooled_mixed_metric - loso_mean_metric, per metric, signed as specified.",
            "not_a_retraining_harness": (
                "This script only scores already-materialized ball_track.json predictions against "
                "already-reviewed labels. B0 parent-source mode consumes validation.jsonl final_label "
                "directly; it never redirects those scratch rows to the historical reviewed corpus. "
                "It never runs inference and never trains a model."
            ),
        },
        "limitations": [
            "Only Burlington and Wolverine currently carry human-reviewed CVAT ball labels usable as "
            "LoSO folds; both are indoor sources. A 2-fold LoSO among two similar indoor sources "
            "cannot fully triangulate the indoor/outdoor shift that has actually caused BALL "
            "inversions -- see this lane's DESIGN_NOTES.md.",
            "heldout_metric values are literals supplied by the caller from already-published "
            "scoring runs (e.g. the ledger); this script does not verify they were computed "
            "correctly and does not itself compute anything against a strict-holdout clip.",
        ],
    }


def _score_candidate(
    name: str,
    candidate_tracks: Sequence[SourceTrack],
    *,
    cvat_root: Path,
    hit_radius_px: float,
    f1_radius_px: float,
    teleport_px_per_frame: float,
    max_jump_gap_frames: int,
    reviewed_frame_indices_by_clip: Mapping[str, Sequence[int]] | None,
    parent_source_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if parent_source_contract is None:
        cvat_candidates = [
            CvatBallCandidate(clip=track.clip, name=name, path=track.path)
            for track in candidate_tracks
        ]
        summary = benchmark_cvat_ball_tracker_candidates(
            candidates=cvat_candidates,
            cvat_root=cvat_root,
            hit_radius_px=hit_radius_px,
            f1_radius_px=f1_radius_px,
            teleport_px_per_frame=teleport_px_per_frame,
            max_jump_gap_frames=max_jump_gap_frames,
            reviewed_frame_indices_by_clip=reviewed_frame_indices_by_clip,
        )
        benchmark_rows = list(summary["results"])
        pooled_metrics = dict(summary["aggregate"].get(name, {}))
    else:
        rows_by_clip = parent_source_contract["validation_rows_by_clip"]
        benchmark_rows = [
            _benchmark_b0_final_labels(
                name=name,
                track=track,
                rows=rows_by_clip[track.clip],
                source_video_contract=parent_source_contract["source_video_contract_by_clip"][track.clip],
                hit_radius_px=hit_radius_px,
                f1_radius_px=f1_radius_px,
                teleport_px_per_frame=teleport_px_per_frame,
                max_jump_gap_frames=max_jump_gap_frames,
            )
            for track in candidate_tracks
        ]
        pooled_metrics = dict(cvat_benchmark._aggregate(benchmark_rows).get(name, {}))
    per_clip: dict[str, dict[str, Any]] = {}
    benchmark_by_clip: dict[str, Mapping[str, Any]] = {}
    for row in benchmark_rows:
        benchmark_by_clip[str(row["clip"])] = row
        per_clip[str(row["clip"])] = {
            "label_metrics": dict(row["label_metrics"]),
            "jitter_metrics": dict(row["jitter_metrics"]),
        }

    tracks_by_source: dict[str, list[SourceTrack]] = {}
    for track in candidate_tracks:
        source_group = track.source_group or track.clip
        tracks_by_source.setdefault(source_group, []).append(track)

    per_source: dict[str, dict[str, Any]] = {}
    clips_by_source: dict[str, list[str]] = {}
    for source_group, source_tracks in sorted(tracks_by_source.items()):
        if parent_source_contract is None:
            grouped_summary = benchmark_cvat_ball_tracker_candidates(
                candidates=[
                    CvatBallCandidate(clip=track.clip, name=name, path=track.path)
                    for track in source_tracks
                ],
                cvat_root=cvat_root,
                hit_radius_px=hit_radius_px,
                f1_radius_px=f1_radius_px,
                teleport_px_per_frame=teleport_px_per_frame,
                max_jump_gap_frames=max_jump_gap_frames,
                reviewed_frame_indices_by_clip=reviewed_frame_indices_by_clip,
            )
            aggregate = dict(grouped_summary["aggregate"].get(name, {}))
            grouped_rows = list(grouped_summary["results"])
        else:
            grouped_rows = [benchmark_by_clip[track.clip] for track in source_tracks]
            aggregate = dict(cvat_benchmark._aggregate(grouped_rows).get(name, {}))
        metrics: dict[str, Any] = {
            metric: aggregate.get(spec["pooled_field"])
            for metric, spec in METRIC_SPECS.items()
        }
        p95_values = [
            float(row["label_metrics"]["p95_error_px"])
            for row in grouped_rows
            if row["label_metrics"].get("p95_error_px") is not None
        ]
        p99_values = [
            float(row["label_metrics"]["p99_error_px"])
            for row in grouped_rows
            if row["label_metrics"].get("p99_error_px") is not None
        ]
        metrics.update(
            {
                "clip_count": len(source_tracks),
                "p95_error_px_worst_clip": max(p95_values) if p95_values else None,
                "p99_error_px_worst_clip": max(p99_values) if p99_values else None,
                "teleport_count_total": sum(
                    int(row["jitter_metrics"].get("teleport_count") or 0)
                    for row in grouped_rows
                ),
            }
        )
        per_source[source_group] = metrics
        clips_by_source[source_group] = sorted(track.clip for track in source_tracks)

    fold_count = len(per_source)
    sufficient = fold_count >= 2

    loso_mean: dict[str, float] = {}
    loso_worst: dict[str, float] = {}
    loso_fold_spread: dict[str, float] = {}
    generalization_gap: dict[str, float] = {}
    if sufficient:
        # LoSO-mean/worst/spread/gap are only meaningful with >=2 independently-scored
        # sources; with a single source they would just restate that source's own
        # number under a misleading "cross-validated" label, so they are left empty.
        for metric, spec in METRIC_SPECS.items():
            values = [
                per_source[clip][metric]
                for clip in per_source
                if per_source[clip].get(metric) is not None
            ]
            if not values:
                continue
            mean_value = statistics.fmean(values)
            loso_mean[metric] = mean_value
            loso_worst[metric] = min(values) if spec["direction"] == "higher_is_better" else max(values)
            loso_fold_spread[metric] = max(values) - min(values)
            pooled_value = pooled_metrics.get(spec["pooled_field"])
            if pooled_value is not None:
                generalization_gap[metric] = float(pooled_value) - mean_value

    return {
        "sources_scored": sorted(per_source),
        "clips_scored": sorted(per_clip),
        "clips_by_source_group": clips_by_source,
        "source_grouping_applied": any((track.source_group or track.clip) != track.clip for track in candidate_tracks),
        "fold_count": fold_count,
        "sufficient_for_loso": sufficient,
        "per_source_metrics": per_source,
        "per_clip_metrics": per_clip,
        "pooled_mixed_metrics": pooled_metrics,
        "pooled_parent_source_metrics": {
            metric: pooled_metrics.get(spec["pooled_field"])
            for metric, spec in METRIC_SPECS.items()
        },
        "loso_mean_metrics": loso_mean,
        "loso_worst_fold_metrics": loso_worst,
        "loso_fold_spread": loso_fold_spread,
        "generalization_gap_pooled_minus_losomean": generalization_gap,
    }


def _benchmark_b0_final_labels(
    *,
    name: str,
    track: SourceTrack,
    rows: Sequence[Mapping[str, Any]],
    source_video_contract: Mapping[str, Any],
    hit_radius_px: float,
    f1_radius_px: float,
    teleport_px_per_frame: float,
    max_jump_gap_frames: int,
) -> dict[str, Any]:
    ball_track = load_ball_track(track.path)
    samples_by_index = cvat_benchmark._track_samples_by_frame_index(ball_track)
    track_frame_count = cvat_benchmark._track_frame_count(ball_track)
    expected_frame_count = source_video_contract.get("frame_count")
    if type(expected_frame_count) is not int or expected_frame_count <= 0:
        raise LoSOValidationError(f"B0 sampling universe frame_count is invalid for {track.clip}")
    if track_frame_count != expected_frame_count:
        raise LoSOValidationError(
            f"prediction horizon does not match canonical source video for {track.clip}: "
            f"track={track_frame_count} source={expected_frame_count}"
        )
    frame_indices = sorted(int(row["frame_index"]) for row in rows)
    if not frame_indices or frame_indices[-1] >= track_frame_count:
        raise LoSOValidationError(f"B0 validation rows exceed prediction horizon for {track.clip}")
    centers_by_frame: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        label = row["final_label"]
        if label["ball_present"]:
            x1, y1, x2, y2 = (float(value) for value in label["bbox_xyxy"])
            centers_by_frame[int(row["frame_index"])] = [((x1 + x2) / 2.0, (y1 + y2) / 2.0)]
    label_metrics = cvat_benchmark._label_metrics(
        samples_by_index=samples_by_index,
        centers_by_frame=centers_by_frame,
        evaluated_reviewed_frame_indices=frame_indices,
        hit_radius_px=hit_radius_px,
        f1_radius_px=f1_radius_px,
        fps=float(ball_track.fps),
    )
    jitter_metrics = cvat_benchmark._jitter_metrics(
        samples_by_index=samples_by_index,
        evaluated_frame_count=track_frame_count,
        teleport_px_per_frame=teleport_px_per_frame,
        max_jump_gap_frames=max_jump_gap_frames,
    )
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_b0_final_label_ball_tracker_benchmark_candidate",
        "status": "TESTED-ON-REAL-DATA",
        "ball_verified": False,
        "clip": track.clip,
        "candidate": name,
        "category": "generalizable",
        "ball_track_path": str(_canonical_regular_file(track.path, f"prediction for {track.clip}")),
        "label_source": "b0_validation_jsonl.final_label",
        "track_frame_count": track_frame_count,
        "cvat_frame_count": expected_frame_count,
        "evaluated_frame_count": expected_frame_count,
        "reviewed_frame_count": len(frame_indices),
        "evaluated_reviewed_frame_count": len(frame_indices),
        "excluded_reviewed_frame_count": 0,
        "excluded_cvat_frame_count": 0,
        "excluded_cvat_visible_label_count": 0,
        "excluded_cvat_hidden_frame_count": 0,
        "cvat_visible_label_count": len(centers_by_frame),
        "evaluated_cvat_visible_label_count": len(centers_by_frame),
        "label_metrics": label_metrics,
        "jitter_metrics": jitter_metrics,
        "quality_score": cvat_benchmark._quality_score(label_metrics, jitter_metrics),
    }


def _build_heldout_comparisons(
    candidates_report: Mapping[str, Any],
    heldout: Sequence[HeldoutReference],
) -> list[dict[str, Any]]:
    by_clip_metric: dict[tuple[str, str], list[HeldoutReference]] = {}
    for reference in heldout:
        by_clip_metric.setdefault((reference.clip, reference.metric), []).append(reference)

    comparisons: list[dict[str, Any]] = []
    for (clip, metric), references in sorted(by_clip_metric.items()):
        if len(references) < 2:
            continue
        spec = METRIC_SPECS[metric]
        direction = spec["direction"]
        rows = []
        for reference in references:
            candidate_row = candidates_report.get(reference.candidate)
            if candidate_row is None:
                continue
            pooled_value = candidate_row["pooled_mixed_metrics"].get(spec["pooled_field"])
            loso_mean_value = candidate_row["loso_mean_metrics"].get(metric)
            loso_worst_value = candidate_row["loso_worst_fold_metrics"].get(metric)
            rows.append(
                {
                    "candidate": reference.candidate,
                    "heldout_value": reference.value,
                    "pooled_mixed_value": pooled_value,
                    "loso_mean_value": loso_mean_value,
                    "loso_worst_fold_value": loso_worst_value,
                    "pooled_abs_error": (
                        abs(pooled_value - reference.value) if pooled_value is not None else None
                    ),
                    "loso_mean_abs_error": (
                        abs(loso_mean_value - reference.value) if loso_mean_value is not None else None
                    ),
                }
            )
        if len(rows) < 2:
            continue

        heldout_winner = _winner(rows, key="heldout_value", direction=direction)
        pooled_winner = _winner(rows, key="pooled_mixed_value", direction=direction)
        loso_mean_winner = _winner(rows, key="loso_mean_value", direction=direction)
        loso_worst_winner = _winner(rows, key="loso_worst_fold_value", direction=direction)

        loso_mean_errors = [row["loso_mean_abs_error"] for row in rows if row["loso_mean_abs_error"] is not None]
        pooled_errors = [row["pooled_abs_error"] for row in rows if row["pooled_abs_error"] is not None]
        loso_mean_lower_error_count = sum(
            1
            for row in rows
            if row["loso_mean_abs_error"] is not None
            and row["pooled_abs_error"] is not None
            and row["loso_mean_abs_error"] < row["pooled_abs_error"]
        )

        comparisons.append(
            {
                "clip": clip,
                "metric": metric,
                "direction": direction,
                "candidates": [row["candidate"] for row in rows],
                "rows": rows,
                "heldout_winner": heldout_winner,
                "pooled_mixed_predicted_winner": pooled_winner,
                "loso_mean_predicted_winner": loso_mean_winner,
                "loso_worst_fold_predicted_winner": loso_worst_winner,
                "pooled_mixed_correctly_predicted_winner": pooled_winner == heldout_winner,
                "loso_mean_correctly_predicted_winner": loso_mean_winner == heldout_winner,
                "loso_worst_fold_correctly_predicted_winner": loso_worst_winner == heldout_winner,
                "loso_mean_had_lower_prediction_error_than_pooled_for_n_of_m_candidates": (
                    f"{loso_mean_lower_error_count}/{len(rows)}"
                ),
                "mean_pooled_abs_error": statistics.fmean(pooled_errors) if pooled_errors else None,
                "mean_loso_mean_abs_error": statistics.fmean(loso_mean_errors) if loso_mean_errors else None,
            }
        )
    return comparisons


def _winner(rows: Sequence[Mapping[str, Any]], *, key: str, direction: str) -> str | None:
    candidates_with_value = [row for row in rows if row.get(key) is not None]
    if not candidates_with_value:
        return None
    if direction == "higher_is_better":
        best = max(candidates_with_value, key=lambda row: row[key])
    else:
        best = min(candidates_with_value, key=lambda row: row[key])
    return str(best["candidate"])


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# BALL LoSO (Leave-One-Source-Out) Validation Report",
        "",
        f"Status: `{report.get('status')}` | objective_result: `{report.get('objective_result')}`",
        "",
        "BALL is not verified by this report. This is a scoring/analysis artifact over "
        "already-materialized predictions and already-reviewed labels; it runs no inference "
        "and trains nothing.",
        "",
        f"- CVAT root: `{report.get('cvat_root')}`",
        f"- Internal-val-only (legal LoSO fold) clip ids: `{report.get('internal_val_only_clip_ids')}`",
        f"- Strict-holdout clip ids never scored by this script: `{report.get('strict_holdout_clip_ids_never_scored')}`",
        "",
        "## Candidates",
        "",
        "| Candidate | Folds | Metric | Pooled/Mixed | LoSO-mean | LoSO-worst | Gap (pooled-mean) |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in sorted(report.get("candidates", {}).items()):
        for metric in METRIC_ORDER:
            display = METRIC_SPECS[metric]["display"]
            pooled = row["pooled_mixed_metrics"].get(METRIC_SPECS[metric]["pooled_field"])
            mean_value = row["loso_mean_metrics"].get(metric)
            worst_value = row["loso_worst_fold_metrics"].get(metric)
            gap = row["generalization_gap_pooled_minus_losomean"].get(metric)
            lines.append(
                "| {name} | {folds} | {metric} | {pooled} | {mean} | {worst} | {gap} |".format(
                    name=name,
                    folds=row["fold_count"],
                    metric=display,
                    pooled=_fmt(pooled),
                    mean=_fmt(mean_value),
                    worst=_fmt(worst_value),
                    gap=_fmt(gap),
                )
            )
    lines.extend(["", "## Held-out comparisons (literals supplied via --heldout-metric)", ""])
    if not report.get("heldout_comparisons"):
        lines.append("_No held-out comparisons: fewer than 2 candidates supplied --heldout-metric for the same clip+metric._")
    else:
        lines.extend(
            [
                "| Clip | Metric | Heldout winner | Pooled predicted | LoSO-mean predicted | LoSO-worst predicted | Pooled correct | LoSO-mean correct |",
                "| --- | --- | --- | --- | --- | --- | :---: | :---: |",
            ]
        )
        for comparison in report["heldout_comparisons"]:
            lines.append(
                "| {clip} | {metric} | {heldout} | {pooled} | {mean} | {worst} | {pooled_ok} | {mean_ok} |".format(
                    clip=comparison["clip"],
                    metric=METRIC_SPECS[comparison["metric"]]["display"],
                    heldout=comparison["heldout_winner"],
                    pooled=comparison["pooled_mixed_predicted_winner"],
                    mean=comparison["loso_mean_predicted_winner"],
                    worst=comparison["loso_worst_fold_predicted_winner"],
                    pooled_ok="yes" if comparison["pooled_mixed_correctly_predicted_winner"] else "NO",
                    mean_ok="yes" if comparison["loso_mean_correctly_predicted_winner"] else "NO",
                )
            )
        lines.append("")
        for comparison in report["heldout_comparisons"]:
            lines.append(
                f"- `{comparison['clip']}` / `{METRIC_SPECS[comparison['metric']]['display']}`: LoSO-mean had lower "
                f"absolute prediction error than the pooled/mixed metric for "
                f"{comparison['loso_mean_had_lower_prediction_error_than_pooled_for_n_of_m_candidates']} candidates "
                f"(mean abs error pooled={_fmt(comparison['mean_pooled_abs_error'])}, "
                f"loso_mean={_fmt(comparison['mean_loso_mean_abs_error'])})."
            )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
