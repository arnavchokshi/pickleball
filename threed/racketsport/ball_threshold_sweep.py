"""Threshold sweeps for raw ball prediction artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ball_benchmark import BallCandidate, write_ball_tracker_benchmark
from .pbmat_adapter import write_ball_track_from_pbmat_predictions
from .totnet_adapter import write_ball_track_from_totnet_predictions


PredictionFamily = str
MIN_BEST_VISIBLE_HIT_RECALL = 0.10
MAX_BEST_HIDDEN_FALSE_POSITIVE_RATE = 0.50
MAX_BEST_TELEPORT_COUNT = 100


def sweep_prediction_thresholds(
    *,
    predictions_by_clip: Mapping[str, str | Path],
    review_root: str | Path,
    out_root: str | Path,
    family: PredictionFamily,
    candidate_name_prefix: str,
    thresholds: Sequence[float],
    category: str = "threshold_sweep",
    hit_radius_px: float = 36.0,
    teleport_px_per_frame: float = 160.0,
    max_jump_gap_frames: int = 3,
    out_json: str | Path | None = None,
    out_markdown: str | Path | None = None,
) -> dict[str, Any]:
    """Convert raw prediction JSONs at several thresholds and benchmark them."""

    normalized_thresholds = _normalize_thresholds(thresholds)
    normalized_family = _normalize_family(family)
    if not predictions_by_clip:
        raise ValueError("at least one prediction clip is required")
    prefix = _safe_candidate_prefix(candidate_name_prefix)
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)

    candidates: list[BallCandidate] = []
    generated: dict[str, dict[str, str]] = {}
    for clip, predictions_path in sorted(predictions_by_clip.items()):
        clip_name = str(clip)
        prediction_file = Path(predictions_path)
        if not prediction_file.is_file():
            raise FileNotFoundError(f"missing prediction JSON for {clip_name}: {prediction_file}")
        generated[clip_name] = {}
        for threshold in normalized_thresholds:
            candidate_name = f"{prefix}_thr_{_threshold_token(threshold)}"
            candidate_dir = root / clip_name / candidate_name
            ball_track_path = candidate_dir / "ball_track.json"
            metadata_path = candidate_dir / "run.json"
            _write_threshold_track(
                family=normalized_family,
                predictions_path=prediction_file,
                threshold=threshold,
                out=ball_track_path,
                metadata_out=metadata_path,
            )
            candidates.append(
                BallCandidate(
                    clip=clip_name,
                    name=candidate_name,
                    category=category,
                    path=ball_track_path,
                )
            )
            generated[clip_name][candidate_name] = str(ball_track_path)

    benchmark = write_ball_tracker_benchmark(
        candidates=candidates,
        review_root=review_root,
        out_json=root / "benchmark.json",
        out_markdown=root / "benchmark.md",
        hit_radius_px=hit_radius_px,
        teleport_px_per_frame=teleport_px_per_frame,
        max_jump_gap_frames=max_jump_gap_frames,
    )
    best_candidate, best_threshold = _best_threshold_candidate(benchmark["aggregate"], normalized_thresholds, prefix)
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_prediction_threshold_sweep",
        "family": normalized_family,
        "candidate_name_prefix": prefix,
        "category": category,
        "thresholds": normalized_thresholds,
        "selection_gates": {
            "min_micro_visible_hit_recall": MIN_BEST_VISIBLE_HIT_RECALL,
            "max_micro_hidden_false_positive_rate": MAX_BEST_HIDDEN_FALSE_POSITIVE_RATE,
            "max_total_teleport_count": MAX_BEST_TELEPORT_COUNT,
        },
        "best_candidate": best_candidate,
        "best_threshold": best_threshold,
        "generated_candidates": generated,
        "benchmark": benchmark,
        "not_ground_truth": True,
    }
    summary_path = Path(out_json) if out_json is not None else root / "threshold_sweep_summary.json"
    _write_json(summary_path, summary)
    if out_markdown is not None:
        Path(out_markdown).write_text(_render_markdown(summary), encoding="utf-8")
    return summary


def _write_threshold_track(
    *,
    family: str,
    predictions_path: Path,
    threshold: float,
    out: Path,
    metadata_out: Path,
) -> None:
    if family == "totnet":
        write_ball_track_from_totnet_predictions(
            predictions_path,
            out=out,
            metadata_out=metadata_out,
            confidence_threshold=threshold,
        )
        return
    if family == "pbmat":
        write_ball_track_from_pbmat_predictions(
            predictions_path,
            out=out,
            metadata_out=metadata_out,
            visibility_threshold=threshold,
        )
        return
    raise ValueError(f"unsupported prediction family: {family}")


def _best_threshold_candidate(
    aggregate: Mapping[str, Mapping[str, Any]],
    thresholds: Sequence[float],
    prefix: str,
) -> tuple[str | None, float | None]:
    if not aggregate:
        return None, None
    threshold_by_candidate = {f"{prefix}_thr_{_threshold_token(threshold)}": threshold for threshold in thresholds}

    def key(item: tuple[str, Mapping[str, Any]]) -> tuple[float, float, float, float, float]:
        name, row = item
        quality = _numeric(row.get("mean_quality_score"), default=-math.inf)
        recall = _numeric(row.get("micro_visible_hit_recall"), default=-math.inf)
        hidden_fp = _numeric(row.get("micro_hidden_false_positive_rate"), default=math.inf)
        teleports = _numeric(row.get("total_teleport_count"), default=math.inf)
        threshold = threshold_by_candidate.get(name, -math.inf)
        return (quality, recall, -hidden_fp, -teleports, threshold)

    eligible_items = [
        item
        for item in aggregate.items()
        if _passes_best_candidate_gates(item[1])
    ]
    if not eligible_items:
        return None, None
    best_name, _best_row = max(eligible_items, key=key)
    return best_name, threshold_by_candidate.get(best_name)


def _passes_best_candidate_gates(row: Mapping[str, Any]) -> bool:
    return (
        _numeric(row.get("micro_visible_hit_recall"), default=0.0) >= MIN_BEST_VISIBLE_HIT_RECALL
        and _numeric(row.get("micro_hidden_false_positive_rate"), default=math.inf) <= MAX_BEST_HIDDEN_FALSE_POSITIVE_RATE
        and _numeric(row.get("total_teleport_count"), default=math.inf) <= MAX_BEST_TELEPORT_COUNT
    )


def _normalize_thresholds(thresholds: Sequence[float]) -> list[float]:
    if not thresholds:
        raise ValueError("at least one threshold is required")
    normalized = []
    for threshold in thresholds:
        value = _numeric(threshold, default=math.nan)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("thresholds must be finite values in [0, 1]")
        normalized.append(float(value))
    return sorted(set(normalized))


def _normalize_family(family: str) -> str:
    normalized = str(family).strip().lower()
    if normalized not in {"totnet", "pbmat"}:
        raise ValueError("family must be one of: totnet, pbmat")
    return normalized


def _safe_candidate_prefix(prefix: str) -> str:
    value = str(prefix).strip()
    if not value:
        raise ValueError("candidate_name_prefix is required")
    if any(char in value for char in "/\\:= "):
        raise ValueError("candidate_name_prefix must not contain path or candidate separators")
    return value


def _threshold_token(threshold: float) -> str:
    return f"{float(threshold):0.3f}".replace(".", "_")


def _numeric(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _render_markdown(summary: Mapping[str, Any]) -> str:
    aggregate = summary["benchmark"]["aggregate"]
    lines = [
        "# Ball Prediction Threshold Sweep",
        "",
        f"Family: `{summary['family']}`",
        f"Best candidate: `{summary['best_candidate']}`",
        f"Best threshold: `{summary['best_threshold']}`",
        "",
        "| Candidate | Quality | Hit recall | Hidden FP | Teleports |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in sorted(aggregate.items()):
        lines.append(
            "| {name} | {quality:.6f} | {recall:.3f} | {hidden:.3f} | {teleports:.0f} |".format(
                name=name,
                quality=float(row.get("mean_quality_score") or 0.0),
                recall=float(row.get("micro_visible_hit_recall") or 0.0),
                hidden=float(row.get("micro_hidden_false_positive_rate") or 0.0),
                teleports=float(row.get("total_teleport_count") or 0.0),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["sweep_prediction_thresholds"]
