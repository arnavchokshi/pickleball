#!/usr/bin/env python3
"""Evaluate an unpromoted event-head checkpoint on public or protected data."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.event_head.datasets import (
    BOUNCE, HIT, build_public_manifest, decode_video_frames,
    manifest_event_centered_windows,
)
from threed.racketsport.event_head.matcher import Event, event_metrics, greedy_match, peak_pick
from threed.racketsport.event_head.model import load_checkpoint

PROTECTED_LABELS = ROOT / "runs/lanes/event_bootstrap_20260713/spot_check_tier_a_50.json"
PROTECTED_ANSWERS = ROOT / "runs/lanes/event_bootstrap_20260713/owner_spot_check_results_20260715.json"


def _predict(model: torch.nn.Module, frames: torch.Tensor, *, threshold: float) -> list[Event]:
    with torch.no_grad():
        logits = model(frames.unsqueeze(0))[0]
    return peak_pick(logits, threshold=threshold)


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
        predictions = _predict(model, frames, threshold=threshold)
        ground_truth = [Event(frame, class_id) for frame, class_id in spec.events]
        clips.append({"source": spec.source, "video": str(spec.video_path), "fps": spec.fps,
                      "predictions": predictions, "ground_truth": ground_truth})
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
        predictions = _predict(model, frames, threshold=threshold)
        decision = answer["decision"]
        summary = {"label_id": label["label_id"], "decision": decision,
                   "prediction_count": len(predictions)}
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("public", "protected-seed"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--window-frames", type=int)
    parser.add_argument("--max-clips", type=int, default=50)
    parser.add_argument(
        "--manifest", type=Path,
        help="Frozen public manifest; defaults to rebuilding the current public corpus",
    )
    args = parser.parse_args()
    if args.out.suffix != ".json":
        parser.error("evaluation output must be a .json metrics artifact; training artifacts are forbidden")
    try:
        model, payload = load_checkpoint(args.checkpoint)
        image_size = int(payload.get("image_size", 224))
        window_frames = _resolve_window_frames(payload, args.window_frames)
        if args.max_clips < 1:
            raise ValueError("--max-clips must be positive")
        manifest = (
            json.loads(args.manifest.read_text()) if args.manifest is not None
            else build_public_manifest(ROOT / "data/event_public_20260713")
        )
        metrics = (eval_public(
                       model, image_size=image_size, threshold=args.threshold,
                       window_frames=window_frames, max_clips=args.max_clips,
                       manifest=manifest,
                   )
                   if args.mode == "public" else
                   eval_protected(model, image_size=image_size, threshold=args.threshold))
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
