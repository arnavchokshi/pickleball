#!/usr/bin/env python3
"""Evaluate an unpromoted event-head checkpoint on public, owner-val, or protected data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import cv2
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.event_head.datasets import (
    BOUNCE, HIT, build_public_manifest, decode_video_frames,
    manifest_event_centered_windows, preprocess_rgb, validate_current_manifest,
)
from threed.racketsport.event_head.matcher import Event, event_metrics, greedy_match, peak_pick
from threed.racketsport.event_head.model import load_checkpoint

PROTECTED_LABELS = ROOT / "runs/lanes/event_bootstrap_20260713/spot_check_tier_a_50.json"
PROTECTED_ANSWERS = ROOT / "runs/lanes/event_bootstrap_20260713/owner_spot_check_results_20260715.json"
DEFAULT_OWNER_MANIFEST = (
    ROOT / "runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json"
)
EXPECTED_OWNER_SELECTION_ROWS = 41
EXPECTED_OWNER_NEGATIVE_ROWS = 22


def _predict(
    model: torch.nn.Module, frames: torch.Tensor, *, threshold: float,
) -> tuple[list[Event], dict[str, Any]]:
    with torch.no_grad():
        device = next(model.parameters()).device
        logits = model(frames.unsqueeze(0).to(device))[0]
    probabilities = logits.softmax(-1)
    finite = torch.isfinite(probabilities)
    max_probability_by_class: dict[str, float | None] = {}
    for class_id, name in enumerate(("background", "HIT", "BOUNCE")):
        finite_values = probabilities[:, class_id][finite[:, class_id]]
        max_probability_by_class[name] = (
            float(finite_values.max().cpu()) if finite_values.numel() else None
        )
    positive_maxima = [
        value for name, value in max_probability_by_class.items()
        if name != "background" and value is not None
    ]
    diagnostics = {
        "max_probability_by_class": max_probability_by_class,
        "max_positive_class_probability": max(positive_maxima) if positive_maxima else None,
        "nonfinite_probability_count": int((~finite).sum().cpu()),
    }
    return peak_pick(logits, threshold=threshold), diagnostics


def _checkpoint_window_frames(payload: dict[str, Any]) -> int:
    """Read the training context stamped into the checkpoint/train manifest."""

    config = payload.get("config") or {}
    configured = config.get("window_frames")
    top_level = payload.get("window_frames")
    if configured is None and top_level is None:
        raise ValueError("checkpoint has no config.window_frames training provenance")
    if configured is not None and top_level is not None and int(configured) != int(top_level):
        raise ValueError(
            "checkpoint window provenance disagrees: "
            f"config.window_frames={configured} vs window_frames={top_level}"
        )
    value = int(configured if configured is not None else top_level)
    if value < 1:
        raise ValueError(f"checkpoint window_frames must be positive, got {value}")
    return value


def _resolve_window_frames(payload: dict[str, Any], requested: int | None) -> int:
    trained = _checkpoint_window_frames(payload)
    if requested is not None and requested != trained:
        raise ValueError(
            f"eval window mismatch: requested {requested}, checkpoint trained with {trained}"
        )
    return trained


def _aggregate_clip_metrics(clips: list[dict[str, Any]], tolerance: int) -> dict[str, Any]:
    totals = {name: {"tp": 0, "fp": 0, "fn": 0, "errors_ms": []} for name in ("HIT", "BOUNCE")}
    for clip in clips:
        for class_id, name in ((HIT, "HIT"), (BOUNCE, "BOUNCE")):
            matched = greedy_match(
                [e for e in clip["predictions"] if e.class_id == class_id],
                [e for e in clip["ground_truth"] if e.class_id == class_id],
                tolerance_frames=tolerance,
            )
            totals[name]["tp"] += matched["tp"]
            totals[name]["fp"] += matched["fp"]
            totals[name]["fn"] += matched["fn"]
            totals[name]["errors_ms"].extend(
                (pred.frame - gt.frame) * 1000.0 / clip["fps"] for pred, gt in matched["pairs"]
            )
    output: dict[str, Any] = {}
    for name, value in totals.items():
        tp, fp, fn = value["tp"], value["fp"], value["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        output[name] = {
            "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "mean_absolute_timing_error_ms": mean(map(abs, value["errors_ms"])) if value["errors_ms"] else None,
        }
    return {"tolerance_frames": tolerance, "per_class": output}


def eval_public(
    model: torch.nn.Module, *, image_size: int, threshold: float,
    window_frames: int, max_clips: int, manifest: dict[str, Any],
) -> dict[str, Any]:
    candidates = (
        manifest_event_centered_windows(
            manifest, split="val", limit=4000, window_frames=window_frames,
        )
        + manifest_event_centered_windows(
            manifest, split="test", limit=4000, window_frames=window_frames,
        )
    )
    selected = [item for item in candidates if item.events][:max_clips]
    if not selected:
        raise RuntimeError("public eval requires at least one media-present val/test event window")
    clips: list[dict[str, Any]] = []
    for spec in selected:
        frames = decode_video_frames(
            spec.video_path, list(range(spec.start_frame, spec.start_frame + spec.num_frames)),
            image_size=image_size,
        )
        predictions, diagnostics = _predict(model, frames, threshold=threshold)
        ground_truth = [Event(frame, class_id) for frame, class_id in spec.events]
        clips.append({"source": spec.source, "video": str(spec.video_path), "fps": spec.fps,
                      "predictions": predictions, "ground_truth": ground_truth,
                      "diagnostics": diagnostics})
    return {
        "mode": "public_heldout_slice", "clip_count": len(clips),
        "window_frames": window_frames,
        "window_policy": "first_event_centered_frozen_eval_protocol",
        "sources": sorted({clip["source"] for clip in clips}),
        "tolerance_sweep": [_aggregate_clip_metrics(clips, tolerance) for tolerance in (1, 2, 5)],
        "protocol_reference": "third_party/spot@edec4201 util/eval.py; type-aware extension",
        "clip_summaries": [{
            "source": clip["source"], "video": clip["video"], "fps": clip["fps"],
            "prediction_count": len(clip["predictions"]), "gt_count": len(clip["ground_truth"]),
            "tolerance_ms": {"1": 1000.0 / clip["fps"], "2": 2000.0 / clip["fps"]},
            **clip["diagnostics"],
        } for clip in clips],
    }


def eval_protected(model: torch.nn.Module, *, image_size: int, threshold: float) -> dict[str, Any]:
    labels = json.loads(PROTECTED_LABELS.read_text())["labels"]
    answers = json.loads(PROTECTED_ANSWERS.read_text())["answers"]
    typed_clips: list[dict[str, Any]] = []
    negative_fp = 0
    negatives = 0
    other_rows = 0
    row_summaries: list[dict[str, Any]] = []
    for index, label in enumerate(labels, 1):
        answer = answers[str(index)]
        video = Path(label["source"]["video_path"])
        capture = cv2.VideoCapture(str(video))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        anchor_s = float(label["anchor"]["pts_s"])
        start = max(0, round((anchor_s - 1.0) * fps))
        end = min(total, round((anchor_s + 1.0) * fps) + 1)
        frames = decode_video_frames(video, list(range(start, end)), image_size=image_size)
        predictions, diagnostics = _predict(model, frames, threshold=threshold)
        decision = answer["decision"]
        summary = {"label_id": label["label_id"], "decision": decision,
                   "prediction_count": len(predictions), **diagnostics}
        if decision in {"paddle", "ground"}:
            corrected_s = anchor_s + float(answer.get("dt", 0.0))
            gt = Event(round(corrected_s * fps) - start, HIT if decision == "paddle" else BOUNCE)
            typed_clips.append({"fps": fps, "predictions": predictions, "ground_truth": [gt]})
        elif decision == "other":
            other_rows += 1
        else:
            negatives += 1
            anchor_frame = round(anchor_s * fps) - start
            has_fp = any(abs(event.frame - anchor_frame) <= round(0.3 * fps) for event in predictions)
            negative_fp += int(has_fp)
            summary["false_positive_within_0_3s"] = has_fp
        row_summaries.append(summary)
    return {
        "mode": "protected_owner_seed", "eval_only": True, "review_only": True,
        "never_training": True, "seed_rows": len(labels), "typed_rows": len(typed_clips),
        "other_rows_reported_separately": other_rows, "negative_rows": negatives,
        "negative_false_positives": negative_fp,
        "negative_false_positive_rate": negative_fp / negatives if negatives else 0.0,
        "tolerance_sweep": [_aggregate_clip_metrics(typed_clips, tolerance) for tolerance in (1, 2)],
        "row_summaries": row_summaries,
    }


def _nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0,1]")
    ordered = sorted(float(value) for value in values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def owner_val_metrics_from_predictions(
    manifest: dict[str, Any],
    predictions_by_row: Mapping[str, Sequence[Event]],
    *,
    arm: str,
    seed: int,
    completed_steps: int,
    target_steps: int,
    full_video_event_count: int,
    full_video_duration_s: float,
) -> dict[str, Any]:
    """Aggregate the frozen owner-41 gate fields from deterministic predictions."""

    validate_current_manifest(manifest)
    rows = [row for row in manifest["rows"] if row["split"] == "val"]
    if len(rows) != EXPECTED_OWNER_SELECTION_ROWS:
        raise ValueError(
            f"owner-val requires exactly {EXPECTED_OWNER_SELECTION_ROWS} val rows, "
            f"got {len(rows)}"
        )
    if str(arm).upper() not in {"A", "B", "C"}:
        raise ValueError("owner-val arm must be A, B, or C")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("owner-val seed must be a nonnegative integer")
    if completed_steps < 0 or target_steps < 1:
        raise ValueError("owner-val step counts are invalid")
    if full_video_event_count < 0 or not math.isfinite(full_video_duration_s) or full_video_duration_s <= 0.0:
        raise ValueError("owner-val full-video rate inputs are invalid")

    row_ids = [str(row.get("label_id", row.get("video"))) for row in rows]
    if len(set(row_ids)) != len(row_ids) or set(predictions_by_row) != set(row_ids):
        raise ValueError("owner-val predictions must cover each val row exactly once")

    clips: list[dict[str, Any]] = []
    row_summaries: list[dict[str, Any]] = []
    negative_false_positives = 0
    negative_rows = 0
    timing_errors_frames: list[float] = []
    for row_id, row in zip(row_ids, rows, strict=True):
        predictions = list(predictions_by_row[row_id])
        ground_truth = [
            Event(int(item["frame"]), HIT if item["class"] == "HIT" else BOUNCE)
            for item in row["events"]
        ]
        clip = {
            "fps": float(row["fps"]),
            "predictions": predictions,
            "ground_truth": ground_truth,
        }
        clips.append(clip)
        is_negative = not ground_truth
        if is_negative:
            negative_rows += 1
            negative_false_positives += int(bool(predictions))
        for class_id in (HIT, BOUNCE):
            matched = greedy_match(
                [event for event in predictions if event.class_id == class_id],
                [event for event in ground_truth if event.class_id == class_id],
                tolerance_frames=2,
            )
            timing_errors_frames.extend(
                abs(pred.frame - gt.frame) for pred, gt in matched["pairs"]
            )
        row_summaries.append({
            "row_id": row_id,
            "negative": is_negative,
            "prediction_count": len(predictions),
            "ground_truth_count": len(ground_truth),
        })
    if negative_rows != EXPECTED_OWNER_NEGATIVE_ROWS:
        raise ValueError(
            f"owner-val requires exactly {EXPECTED_OWNER_NEGATIVE_ROWS} negative rows, "
            f"got {negative_rows}"
        )

    tolerance_sweep = [
        _aggregate_clip_metrics(clips, tolerance) for tolerance in (1, 2)
    ]
    tolerance_two = tolerance_sweep[-1]["per_class"]
    timing_p90 = (
        _nearest_rank_percentile(timing_errors_frames, 0.90)
        if timing_errors_frames
        else float(manifest.get("config", {}).get("window_frames", 64))
    )
    rate = full_video_event_count / full_video_duration_s
    return {
        "artifact_type": "event_head_abc_arm_eval",
        "mode": "owner_validation_41",
        "selection_scope": "owner_validation_41",
        "selection_rows": len(rows),
        "protected_50_touched": False,
        "arm": str(arm).upper(),
        "seed": int(seed),
        "completed_steps": int(completed_steps),
        "target_steps": int(target_steps),
        "negative_rows": negative_rows,
        "negative_false_positives": negative_false_positives,
        "timing_error_p90_frames": timing_p90,
        "timing_error_p90_policy": "nearest_rank_absolute_error_at_tolerance_2; no_matches=window_frames",
        "full_video_events_per_second": rate,
        "full_video_event_count": full_video_event_count,
        "full_video_duration_s": full_video_duration_s,
        "macro_f1_at_2": mean(
            float(tolerance_two[name]["f1"]) for name in ("HIT", "BOUNCE")
        ),
        "tolerance_sweep": tolerance_sweep,
        "row_summaries": row_summaries,
    }


def _predict_full_video_rate(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, Any]],
    *,
    image_size: int,
    threshold: float,
    window_frames: int,
) -> tuple[int, float, list[dict[str, Any]]]:
    """Cover each distinct owner-val source video once in fixed non-overlapping windows."""

    sources: dict[str, Path] = {}
    for row in rows:
        source_video = str(row["source_video"])
        video_path = Path(str(row["video_path"]))
        previous = sources.setdefault(source_video, video_path)
        if previous != video_path:
            raise ValueError(f"owner-val source {source_video} maps to multiple media paths")

    total_events = 0
    total_duration_s = 0.0
    summaries: list[dict[str, Any]] = []
    for source_video, video_path in sorted(sources.items()):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"cannot open owner-val full video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0.0:
            capture.release()
            raise ValueError(f"owner-val full video has invalid FPS: {video_path}")
        processed = 0
        event_count = 0
        batch: list[torch.Tensor] = []
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                batch.append(preprocess_rgb(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), image_size))
                processed += 1
                if len(batch) == window_frames:
                    predictions, _ = _predict(model, torch.stack(batch), threshold=threshold)
                    event_count += len(predictions)
                    batch.clear()
            if batch:
                predictions, _ = _predict(model, torch.stack(batch), threshold=threshold)
                event_count += len(predictions)
        finally:
            capture.release()
        if processed == 0:
            raise ValueError(f"owner-val full video decoded zero frames: {video_path}")
        duration_s = processed / fps
        total_events += event_count
        total_duration_s += duration_s
        summaries.append({
            "source_video": source_video,
            "video_path": str(video_path),
            "frames": processed,
            "fps": fps,
            "duration_s": duration_s,
            "event_count": event_count,
            "events_per_second": event_count / duration_s,
            "window_policy": "non_overlapping_checkpoint_windows_covering_every_frame_once",
        })
    return total_events, total_duration_s, summaries


def eval_owner_val(
    model: torch.nn.Module,
    *,
    image_size: int,
    threshold: float,
    window_frames: int,
    manifest: dict[str, Any],
    arm: str,
    seed: int,
    completed_steps: int,
    target_steps: int,
) -> dict[str, Any]:
    validate_current_manifest(manifest)
    rows = [row for row in manifest["rows"] if row["split"] == "val"]
    predictions_by_row: dict[str, list[Event]] = {}
    diagnostics_by_row: dict[str, dict[str, Any]] = {}
    for row in rows:
        if int(row["num_frames"]) != window_frames:
            raise ValueError("owner-val row context must match checkpoint window_frames")
        row_id = str(row.get("label_id", row.get("video")))
        absolute_frames = range(
            int(row["source_start_frame"]),
            int(row["source_start_frame"]) + int(row["num_frames"]),
        )
        frames = decode_video_frames(
            Path(row["video_path"]), list(absolute_frames), image_size=image_size
        )
        predictions, diagnostics = _predict(model, frames, threshold=threshold)
        predictions_by_row[row_id] = predictions
        diagnostics_by_row[row_id] = diagnostics
    full_count, full_duration_s, full_summaries = _predict_full_video_rate(
        model,
        rows,
        image_size=image_size,
        threshold=threshold,
        window_frames=window_frames,
    )
    metrics = owner_val_metrics_from_predictions(
        manifest,
        predictions_by_row,
        arm=arm,
        seed=seed,
        completed_steps=completed_steps,
        target_steps=target_steps,
        full_video_event_count=full_count,
        full_video_duration_s=full_duration_s,
    )
    metrics["full_video_summaries"] = full_summaries
    for summary in metrics["row_summaries"]:
        summary.update(diagnostics_by_row[summary["row_id"]])
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("public", "owner-val", "protected-seed"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--window-frames", type=int)
    parser.add_argument("--max-clips", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--arm", choices=("A", "B", "C"))
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--manifest", type=Path,
        help="Frozen public manifest; defaults to rebuilding the current public corpus",
    )
    args = parser.parse_args()
    if args.out.suffix != ".json":
        parser.error("evaluation output must be a .json metrics artifact; training artifacts are forbidden")
    try:
        model, payload = load_checkpoint(args.checkpoint, device=args.device)
        image_size = int(payload.get("image_size", 224))
        window_frames = _resolve_window_frames(payload, args.window_frames)
        if args.max_clips < 1:
            raise ValueError("--max-clips must be positive")
        if args.mode == "public":
            manifest = (
                json.loads(args.manifest.read_text()) if args.manifest is not None
                else build_public_manifest(ROOT / "data/event_public_20260713")
            )
            metrics = eval_public(
                model, image_size=image_size, threshold=args.threshold,
                window_frames=window_frames, max_clips=args.max_clips,
                manifest=manifest,
            )
        elif args.mode == "owner-val":
            if args.arm is None or args.seed is None:
                raise ValueError("owner-val requires --arm and --seed")
            manifest_path = args.manifest or DEFAULT_OWNER_MANIFEST
            manifest = json.loads(manifest_path.read_text())
            config = payload.get("config")
            if not isinstance(config, Mapping):
                raise ValueError("owner-val checkpoint lacks fine-tune config")
            completed_steps = int(payload.get("completed_steps", -1))
            target_steps = int(config.get("steps", -1))
            checkpoint_seed = int(config.get("seed", -1))
            if checkpoint_seed != args.seed:
                raise ValueError(
                    f"owner-val --seed={args.seed} disagrees with checkpoint seed={checkpoint_seed}"
                )
            metrics = eval_owner_val(
                model,
                image_size=image_size,
                threshold=args.threshold,
                window_frames=window_frames,
                manifest=manifest,
                arm=args.arm,
                seed=args.seed,
                completed_steps=completed_steps,
                target_steps=target_steps,
            )
        else:
            metrics = eval_protected(
                model, image_size=image_size, threshold=args.threshold
            )
        result = {
            "schema_version": 1, "artifact_type": "event_head_eval_metrics",
            "verified": False, "checkpoint": str(args.checkpoint), "threshold": args.threshold,
            "checkpoint_train_window_frames": window_frames,
            **metrics,
        }
        if args.mode == "protected-seed":
            result.update({"eval_only": True, "review_only": True, "never_training": True})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.exit(3, f"event-head evaluation failed: {exc}\n")
    print(json.dumps({"out": str(args.out), "mode": args.mode, "verified": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
