#!/usr/bin/env python3
"""Diagnostic: public eval with TRAIN-MATCHED window length + threshold sweep.

Why this exists (coordinator directive 2026-07-17): the committed
scripts/racketsport/eval_event_head.py hardcodes ``window_frames=15``
(lines 68-69) and scores only 3 clips, while the GPU pretrain trained on
64-frame windows. Scoring a 64-frame-context temporal head on 15-frame crops
is a methodological mismatch, so its "0 TP" is a broken MEASUREMENT, not a
model verdict. This script re-runs the same protocol at matched window length
over a bounded clip sample, and sweeps thresholds so we can tell three cases
apart:

  (a) model is SILENT      -> no probs above threshold at any threshold
  (b) model has WEAK signal -> TPs appear at lower thresholds
  (c) model is MISTIMED     -> peaks exist but land outside tolerance

The model is a GRU over per-frame features, so it is sequence-length agnostic;
running it at 64 frames is valid.

Lane-local diagnostic (runs/lanes/event_head_pretrain_20260716/) — NOT a
registered repo CLI, produces lane evidence only. VERIFIED=0; eval-only; no
promotion. Never trains, never touches protected/owner media.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from threed.racketsport.event_head.datasets import (  # noqa: E402
    BOUNCE,
    HIT,
    build_public_manifest,
    decode_video_frames,
    manifest_windows,
)
from threed.racketsport.event_head.matcher import Event, greedy_match, peak_pick  # noqa: E402
from threed.racketsport.event_head.model import EventHead  # noqa: E402


def _load(checkpoint: Path) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    config = payload.get("config", {}) or {}
    model = EventHead(weights="none")
    model.load_state_dict(state)
    model.eval()
    return model, config


def _clip_metrics(clips: list[dict], tolerance: int) -> dict:
    totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in ("HIT", "BOUNCE")}
    for clip in clips:
        for class_id, name in ((HIT, "HIT"), (BOUNCE, "BOUNCE")):
            matched = greedy_match(
                [e for e in clip["predictions"] if e.class_id == class_id],
                [e for e in clip["ground_truth"] if e.class_id == class_id],
                tolerance_frames=tolerance,
            )
            for key in ("tp", "fp", "fn"):
                totals[name][key] += matched[key]
    out = {}
    for name, value in totals.items():
        tp, fp, fn = value["tp"], value["fp"], value["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out[name] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        }
    return {"tolerance_frames": tolerance, "per_class": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--window-frames", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--max-clips", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, config = _load(Path(args.checkpoint))
    device = torch.device(args.device)
    model.to(device)

    manifest = build_public_manifest(ROOT / "data/event_public_20260713")
    specs = (
        manifest_windows(manifest, split="val", limit=4000, window_frames=args.window_frames)
        + manifest_windows(manifest, split="test", limit=4000, window_frames=args.window_frames)
    )
    specs = [s for s in specs if s.events][: args.max_clips]
    if not specs:
        raise SystemExit("no eval windows available")

    clips: list[dict] = []
    all_scores: list[float] = []
    for spec in specs:
        frames = decode_video_frames(
            spec.video_path,
            list(range(spec.start_frame, spec.start_frame + spec.num_frames)),
            image_size=args.image_size,
        )
        with torch.no_grad():
            logits = model(frames.unsqueeze(0).to(device))[0].cpu()
        probs = logits.softmax(-1)
        # peak scores for the two event classes, for the distribution diagnostic
        all_scores.extend(probs[:, 1].tolist())
        all_scores.extend(probs[:, 2].tolist())
        clips.append({
            "source": spec.source,
            "video": str(spec.video_path),
            "logits": logits,
            "ground_truth": [Event(frame, class_id) for frame, class_id in spec.events],
        })

    sweep: list[dict] = []
    for threshold in (0.5, 0.3, 0.2, 0.1, 0.05):
        scored = []
        for clip in clips:
            scored.append({
                "predictions": peak_pick(clip["logits"], threshold=threshold),
                "ground_truth": clip["ground_truth"],
            })
        entry = {
            "threshold": threshold,
            "prediction_count": sum(len(c["predictions"]) for c in scored),
            "tolerances": [_clip_metrics(scored, t) for t in (1, 2, 5)],
        }
        sweep.append(entry)

    ordered = sorted(all_scores, reverse=True)
    diagnostic = {
        "artifact_type": "event_head_matched_window_eval_diagnostic",
        "verified": False,
        "eval_only": True,
        "never_training": True,
        "checkpoint": str(args.checkpoint),
        "window_frames": args.window_frames,
        "committed_cli_window_frames": 15,
        "why": "committed eval_event_head.py hardcodes 15-frame windows; training used 64 — this re-scores at matched length",
        "clips_scored": len(clips),
        "gt_event_count": sum(len(c["ground_truth"]) for c in clips),
        "sources": sorted({c["source"] for c in clips}),
        "class_score_distribution": {
            "max": round(max(all_scores), 6),
            "p99": round(ordered[len(ordered) // 100], 6),
            "median": round(statistics.median(all_scores), 6),
            "frames_scored": len(all_scores) // 2,
        },
        "threshold_sweep": sweep,
        "model_config": config,
    }
    Path(args.out).write_text(json.dumps(diagnostic, indent=2) + "\n")
    print(json.dumps({
        "clips": len(clips),
        "gt_events": diagnostic["gt_event_count"],
        "max_class_prob": diagnostic["class_score_distribution"]["max"],
        "best": max(
            (
                (e["threshold"], t["tolerance_frames"], t["per_class"]["HIT"]["tp"] + t["per_class"]["BOUNCE"]["tp"])
                for e in sweep for t in e["tolerances"]
            ),
            key=lambda x: x[2],
        ),
        "out": args.out,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
