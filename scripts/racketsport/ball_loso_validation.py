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
import json
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
from threed.racketsport.eval_guard import (  # noqa: E402
    INTERNAL_VAL_ONLY_CLIP_IDS,
    STRICT_HOLDOUT_CLIP_IDS,
)


ARTIFACT_TYPE = "racketsport_ball_loso_validation"
DEFAULT_CVAT_ROOT = Path("runs/cvat_imports/2026_06_30")
REVIEWED_BOXES_FILENAME = "reviewed_boxes.json"

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
        tracks = _apply_source_groups(tracks, _parse_source_group_specs(args.source_group))
        reviewed_row_filter = (
            _load_reviewed_row_filter(args.reviewed_row_list)
            if args.reviewed_row_list is not None
            else None
        )
        heldout = _parse_heldout_metric_specs(args.heldout_metric)
        if not tracks:
            raise LoSOValidationError("at least one --candidate-track is required")
        report = build_loso_report(
            tracks=tracks,
            heldout=heldout,
            cvat_root=args.cvat_root,
            hit_radius_px=args.hit_radius_px,
            f1_radius_px=args.f1_radius_px,
            teleport_px_per_frame=args.teleport_px_per_frame,
            max_jump_gap_frames=args.max_jump_gap_frames,
            reviewed_frame_indices_by_clip=reviewed_row_filter,
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
) -> dict[str, Any]:
    by_candidate: dict[str, list[SourceTrack]] = {}
    for track in tracks:
        _assert_scoreable_source(track.clip, context=f"--candidate-track {track.candidate}={track.clip}")
        if not track.path.is_file():
            raise LoSOValidationError(f"missing ball_track.json for {track.candidate}/{track.clip}: {track.path}")
        by_candidate.setdefault(track.candidate, []).append(track)

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
        "cvat_root": str(cvat_root),
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
                "already-reviewed CVAT labels. It never runs inference and never trains a model."
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
) -> dict[str, Any]:
    cvat_candidates = [
        CvatBallCandidate(clip=track.clip, name=name, path=track.path) for track in candidate_tracks
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
    per_clip: dict[str, dict[str, Any]] = {}
    for row in summary["results"]:
        per_clip[str(row["clip"])] = {
            "label_metrics": dict(row["label_metrics"]),
            "jitter_metrics": dict(row["jitter_metrics"]),
        }
    pooled_metrics = dict(summary["aggregate"].get(name, {}))

    tracks_by_source: dict[str, list[SourceTrack]] = {}
    for track in candidate_tracks:
        source_group = track.source_group or track.clip
        tracks_by_source.setdefault(source_group, []).append(track)

    per_source: dict[str, dict[str, Any]] = {}
    clips_by_source: dict[str, list[str]] = {}
    for source_group, source_tracks in sorted(tracks_by_source.items()):
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
        "loso_mean_metrics": loso_mean,
        "loso_worst_fold_metrics": loso_worst,
        "loso_fold_spread": loso_fold_spread,
        "generalization_gap_pooled_minus_losomean": generalization_gap,
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
