#!/usr/bin/env python3
"""Benchmark the soft ball player-proximity prior on Burlington/Wolverine only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_player_proximity_prior import (  # noqa: E402
    BallPlayerProximityPriorConfig,
    apply_ball_player_proximity_prior_from_files,
)
from threed.racketsport.ball_threshold_sweep import sweep_ball_track_cvat_thresholds  # noqa: E402
from threed.racketsport.eval_guard import INTERNAL_VAL_ONLY_CLIP_IDS, STRICT_HOLDOUT_CLIP_IDS  # noqa: E402


DEFAULT_CLIPS = (
    "burlington_gold_0300_low_steep_corner",
    "wolverine_mixed_0200_mid_steep_corner",
)
DEFAULT_SOURCE_ROOT = Path("runs/ball_goal_m1_wasb_finetune_20260701T210431Z/step0_zero_shot_sweep/scored/tennis")
DEFAULT_SOURCE_CANDIDATE_DIR = "wasb_tennis_zeroshot_thr_0_050"
DEFAULT_TRACKS_ROOT = Path("runs/synergy_wirings_20260702T043529Z/w3_ball_fp_proximity")
DEFAULT_CVAT_ROOT = Path("runs/cvat_imports/2026_06_30")
DEFAULT_STRENGTHS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
DEFAULT_OPERATING_THRESHOLD = 0.5
DEFAULT_MATERIAL_HIDDEN_FP_ABS_DROP = 0.05
DEFAULT_MAX_RECALL_LOSS = 0.02


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply/sweep the soft ball player-proximity prior and score only Burlington/Wolverine "
            "internal-val CVAT ball labels. Strict held-out Outdoor/Indoor clips are refused."
        )
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--clip", action="append", default=[], help="Internal clip id. Defaults to Burlington/Wolverine.")
    parser.add_argument("--track", action="append", default=[], help="Optional clip=/path/to/source_ball_track.json override.")
    parser.add_argument(
        "--player-track",
        action="append",
        default=[],
        help="Optional clip=/path/to/player_tracks.json override.",
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-candidate-dir", default=DEFAULT_SOURCE_CANDIDATE_DIR)
    parser.add_argument("--tracks-root", type=Path, default=DEFAULT_TRACKS_ROOT)
    parser.add_argument("--cvat-root", type=Path, default=DEFAULT_CVAT_ROOT)
    parser.add_argument("--strength", action="append", type=float, default=[])
    parser.add_argument(
        "--influence-diag-fraction",
        type=float,
        default=BallPlayerProximityPriorConfig().influence_diag_fraction,
    )
    parser.add_argument("--operating-threshold", type=float, default=DEFAULT_OPERATING_THRESHOLD)
    parser.add_argument("--material-hidden-fp-abs-drop", type=float, default=DEFAULT_MATERIAL_HIDDEN_FP_ABS_DROP)
    parser.add_argument("--max-recall-loss", type=float, default=DEFAULT_MAX_RECALL_LOSS)
    args = parser.parse_args(argv)

    try:
        clips = tuple(args.clip or DEFAULT_CLIPS)
        _validate_internal_only(clips)
        strengths = _normalize_strengths(args.strength or DEFAULT_STRENGTHS)
        operating_threshold = _unit_interval(args.operating_threshold, "operating_threshold")
        material_hidden_fp_abs_drop = _nonnegative(args.material_hidden_fp_abs_drop, "material_hidden_fp_abs_drop")
        max_recall_loss = _nonnegative(args.max_recall_loss, "max_recall_loss")
        track_overrides = _parse_path_overrides(args.track, "track")
        player_track_overrides = _parse_path_overrides(args.player_track, "player-track")
        summary = run_internal_prior_benchmark(
            clips=clips,
            out_root=args.out_root,
            source_root=args.source_root,
            source_candidate_dir=args.source_candidate_dir,
            tracks_root=args.tracks_root,
            cvat_root=args.cvat_root,
            track_overrides=track_overrides,
            player_track_overrides=player_track_overrides,
            strengths=strengths,
            influence_diag_fraction=args.influence_diag_fraction,
            operating_threshold=operating_threshold,
            material_hidden_fp_abs_drop=material_hidden_fp_abs_drop,
            max_recall_loss=max_recall_loss,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "out_root": str(args.out_root),
                "best_clearing_setting": summary["best_clearing_setting"],
                "baseline": summary["baseline"],
                "curve_count": len(summary["curves"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_internal_prior_benchmark(
    *,
    clips: tuple[str, ...],
    out_root: Path,
    source_root: Path,
    source_candidate_dir: str,
    tracks_root: Path,
    cvat_root: Path,
    track_overrides: Mapping[str, Path],
    player_track_overrides: Mapping[str, Path],
    strengths: list[float],
    influence_diag_fraction: float,
    operating_threshold: float,
    material_hidden_fp_abs_drop: float,
    max_recall_loss: float,
) -> dict[str, Any]:
    config_probe = BallPlayerProximityPriorConfig(strength=0.0, influence_diag_fraction=influence_diag_fraction)
    out_root.mkdir(parents=True, exist_ok=True)
    curves: list[dict[str, Any]] = []
    for strength in strengths:
        config = BallPlayerProximityPriorConfig(
            strength=strength,
            influence_diag_fraction=config_probe.influence_diag_fraction,
        )
        strength_token = _value_token(strength)
        strength_root = out_root / f"strength_{strength_token}"
        prior_tracks_by_clip: dict[str, Path] = {}
        prior_reports_by_clip: dict[str, Path] = {}
        for clip in clips:
            source_track = track_overrides.get(clip) or source_root / clip / source_candidate_dir / "ball_track.json"
            player_track = player_track_overrides.get(clip) or tracks_root / f"{clip}_reviewed_player_tracks.json"
            out_ball_track = strength_root / clip / "ball_track_proximity_prior_prethreshold.json"
            out_report = strength_root / clip / "player_proximity_prior_report.json"
            apply_ball_player_proximity_prior_from_files(
                ball_track_path=source_track,
                tracks_path=player_track,
                out_ball_track_path=out_ball_track,
                out_report_path=out_report,
                config=config,
            )
            prior_tracks_by_clip[clip] = out_ball_track
            prior_reports_by_clip[clip] = out_report

        prefix = f"wasb_tennis_proxprior_s{strength_token}"
        threshold_summary = sweep_ball_track_cvat_thresholds(
            tracks_by_clip=prior_tracks_by_clip,
            cvat_root=cvat_root,
            out_root=strength_root / "thresholded",
            candidate_name_prefix=prefix,
            thresholds=[operating_threshold],
            category="player_proximity_prior_internal",
        )
        candidate_name = f"{prefix}_thr_{_value_token(operating_threshold)}"
        aggregate_row = threshold_summary["benchmark"]["aggregate"][candidate_name]
        curves.append(
            {
                "strength": strength,
                "candidate": candidate_name,
                "source_track_paths": {
                    clip: str(track_overrides.get(clip) or source_root / clip / source_candidate_dir / "ball_track.json")
                    for clip in clips
                },
                "prior_track_paths": {clip: str(path) for clip, path in prior_tracks_by_clip.items()},
                "prior_report_paths": {clip: str(path) for clip, path in prior_reports_by_clip.items()},
                "threshold_summary_path": str(strength_root / "thresholded" / "threshold_sweep_summary.json"),
                "benchmark_path": str(strength_root / "thresholded" / "benchmark.json"),
                "metrics": _metrics_from_aggregate(aggregate_row),
            }
        )

    baseline = _baseline_row(curves)
    for row in curves:
        row["deltas_vs_baseline"] = _deltas(row["metrics"], baseline)
        row["clears_internal_bar"] = (
            row["deltas_vs_baseline"]["hidden_fp_rate_abs_drop"] >= material_hidden_fp_abs_drop
            and row["deltas_vs_baseline"]["recall_at_20_loss"] <= max_recall_loss
        )
    best = _best_clearing_setting(curves)
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_player_proximity_prior_internal_benchmark",
        "status": "internal_val_scored",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strict_holdout_touched": False,
        "out_root": str(out_root),
        "clips": list(clips),
        "clip_policy": "Burlington/Wolverine internal-val only; Outdoor/Indoor refused by this script.",
        "source_checkpoint": "WASB-SBDT tennis zero-shot",
        "source_candidate_dir": source_candidate_dir,
        "source_root": str(source_root),
        "tracks_root": str(tracks_root),
        "cvat_root": str(cvat_root),
        "operating_threshold": operating_threshold,
        "prior_parameters": {
            "strengths": strengths,
            "influence_diag_fraction": float(influence_diag_fraction),
        },
        "internal_bar": {
            "material_hidden_fp_abs_drop": material_hidden_fp_abs_drop,
            "max_recall_at_20_loss": max_recall_loss,
        },
        "baseline": baseline,
        "curves": curves,
        "best_clearing_setting": best,
        "promotion_claimed": False,
        "notes": [
            "Confidence-only prior artifacts preserve every candidate and visible flag; the existing downstream threshold filter is applied afterward.",
            "This benchmark is an internal-val selection signal only. It is not held-out BALL gate evidence.",
        ],
    }
    _write_json(out_root / "internal_benchmark_summary.json", summary)
    (out_root / "REPORT.md").write_text(_render_report(summary), encoding="utf-8")
    if best is not None:
        (out_root / "PREREGISTRATION.md").write_text(_render_preregistration(summary, best), encoding="utf-8")
    return summary


def _validate_internal_only(clips: tuple[str, ...]) -> None:
    if not clips:
        raise ValueError("at least one clip is required")
    strict = set(STRICT_HOLDOUT_CLIP_IDS)
    internal = set(INTERNAL_VAL_ONLY_CLIP_IDS)
    for clip in clips:
        if clip in strict:
            raise ValueError(
                f"refusing strict held-out clip {clip!r}; write heldout_eval_ledger.md preregistration and stop "
                "for manager approval before any Outdoor/Indoor scoring"
            )
        if clip not in internal:
            raise ValueError(f"this internal benchmark only accepts Burlington/Wolverine internal-val clips, got {clip!r}")


def _parse_path_overrides(specs: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--{label} must be clip=/path, got: {spec}")
        clip, path = spec.split("=", 1)
        if not clip or not path:
            raise ValueError(f"--{label} missing clip or path: {spec}")
        result[clip] = Path(path)
    return result


def _normalize_strengths(values: tuple[float, ...] | list[float]) -> list[float]:
    normalized = {0.0}
    for value in values:
        number = _unit_interval_exclusive_one(value, "strength")
        normalized.add(number)
    return sorted(normalized)


def _metrics_from_aggregate(row: Mapping[str, Any]) -> dict[str, Any]:
    visible_label_count = int(row.get("total_visible_label_count") or 0)
    recall_at_20 = float(row.get("micro_visible_recall_at_20px") or 0.0)
    return {
        "f1_at_20": float(row.get("micro_label_f1_at_20px") or 0.0),
        "recall_at_20": recall_at_20,
        "precision_at_20": float(row.get("micro_precision_at_20px") or 0.0),
        "hidden_fp_rate": float(row.get("micro_hidden_false_positive_rate") or 0.0),
        "hidden_fp_count": int(row.get("total_hidden_false_positive_count") or 0),
        "hidden_label_count": int(row.get("total_hidden_label_count") or 0),
        "teleport_count": int(row.get("total_teleport_count") or 0),
        "visible_label_count": visible_label_count,
        "true_positive_count_at_20": int(round(recall_at_20 * visible_label_count)),
    }


def _baseline_row(curves: list[dict[str, Any]]) -> dict[str, Any]:
    for row in curves:
        if row["strength"] == 0.0:
            return dict(row["metrics"])
    raise ValueError("strength 0 baseline missing")


def _deltas(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float | int]:
    return {
        "f1_at_20_delta": float(metrics["f1_at_20"]) - float(baseline["f1_at_20"]),
        "recall_at_20_delta": float(metrics["recall_at_20"]) - float(baseline["recall_at_20"]),
        "recall_at_20_loss": max(0.0, float(baseline["recall_at_20"]) - float(metrics["recall_at_20"])),
        "precision_at_20_delta": float(metrics["precision_at_20"]) - float(baseline["precision_at_20"]),
        "hidden_fp_rate_abs_drop": float(baseline["hidden_fp_rate"]) - float(metrics["hidden_fp_rate"]),
        "hidden_fp_count_drop": int(baseline["hidden_fp_count"]) - int(metrics["hidden_fp_count"]),
    }


def _best_clearing_setting(curves: list[dict[str, Any]]) -> dict[str, Any] | None:
    clearing = [row for row in curves if row["clears_internal_bar"]]
    if not clearing:
        return None

    def key(row: Mapping[str, Any]) -> tuple[float, float, float]:
        deltas = row["deltas_vs_baseline"]
        metrics = row["metrics"]
        return (
            float(deltas["hidden_fp_rate_abs_drop"]),
            float(metrics["f1_at_20"]),
            -float(row["strength"]),
        )

    best = max(clearing, key=key)
    return {
        "strength": best["strength"],
        "candidate": best["candidate"],
        "metrics": best["metrics"],
        "deltas_vs_baseline": best["deltas_vs_baseline"],
    }


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Ball Player-Proximity Prior Internal Benchmark",
        "",
        "Strict held-out Outdoor/Indoor clips were not scored.",
        "",
        f"Operating threshold: `{summary['operating_threshold']}`",
        f"Influence radius: `{summary['prior_parameters']['influence_diag_fraction']}` player-box diagonals",
        "",
        "| Strength | F1@20 | Recall@20 | Precision@20 | Hidden FP | Hidden FP count | Recall loss | Clears bar |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["curves"]:
        metrics = row["metrics"]
        deltas = row["deltas_vs_baseline"]
        lines.append(
            "| {strength:.3f} | {f1:.4f} | {recall:.4f} | {precision:.4f} | {hidden:.4f} | {hidden_count} | {loss:.4f} | {clear} |".format(
                strength=float(row["strength"]),
                f1=float(metrics["f1_at_20"]),
                recall=float(metrics["recall_at_20"]),
                precision=float(metrics["precision_at_20"]),
                hidden=float(metrics["hidden_fp_rate"]),
                hidden_count=int(metrics["hidden_fp_count"]),
                loss=float(deltas["recall_at_20_loss"]),
                clear="yes" if row["clears_internal_bar"] else "no",
            )
        )
    lines.extend(["", "## Verdict", ""])
    best = summary.get("best_clearing_setting")
    if best is None:
        lines.append("No internal setting cleared the hidden-FP drop plus <2 point recall-loss bar. No held-out preregistration should be filed.")
    else:
        lines.append(
            "A single frozen setting cleared the internal bar. Write/keep the held-out preregistration and stop before Outdoor/Indoor scoring."
        )
        lines.append("")
        lines.append(f"Frozen strength: `{best['strength']}`")
    lines.append("")
    return "\n".join(lines)


def _render_preregistration(summary: Mapping[str, Any], best: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# BALL Player-Proximity Prior Held-Out Preregistration",
            "",
            "Do not run this against Outdoor/Indoor until the manager approves the held-out eval.",
            "",
            f"Candidate: WASB-SBDT tennis zero-shot plus soft player-proximity prior strength `{best['strength']}`.",
            f"Influence radius: `{summary['prior_parameters']['influence_diag_fraction']}` player-box diagonals.",
            f"Downstream operating threshold: `{summary['operating_threshold']}`.",
            "Selection basis: Burlington+Wolverine internal-val benchmark only.",
            f"Internal benchmark path: `{summary['out_root']}`.",
            "",
            "Held-out plan: exactly one Outdoor BALL detector scoring run at the frozen setting above; no threshold sweep.",
            "",
        ]
    )


def _value_token(value: float) -> str:
    return f"{float(value):0.3f}".replace(".", "_")


def _unit_interval(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return number


def _unit_interval_exclusive_one(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number < 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1)")
    return number


def _nonnegative(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative value")
    return number


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
