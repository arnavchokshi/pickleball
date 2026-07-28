#!/usr/bin/env python3
"""Ad hoc firing-rate measurement on a single full clip, reusing the eval harness's
own prediction/decode primitives (never a bespoke inference path). Lane-owned
evidence generator for ev2_train_20260728; not a new pipeline entrypoint.

Directly comparable to the North Star's 7.16 HIT/s zero-shot-transfer figure, which
was measured on this same 697s pb.vision demo clip (sha256 272a2132...).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.event_head.datasets import preprocess_rgb  # noqa: E402
from threed.racketsport.event_head.model import load_checkpoint  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "racketsport"))
from eval_event_head import _predict, _resolve_window_frames  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model, payload = load_checkpoint(args.checkpoint, device=args.device)
    image_size = int(payload.get("image_size", 224))
    window_frames = _resolve_window_frames(payload, None)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"invalid fps for {args.video}")

    processed = 0
    hit_count = 0
    bounce_count = 0
    batch: list[torch.Tensor] = []
    window_summaries: list[dict[str, object]] = []
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            batch.append(preprocess_rgb(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), image_size))
            processed += 1
            if len(batch) == window_frames:
                predictions, diagnostics = _predict(model, torch.stack(batch), threshold=args.threshold)
                hits = sum(1 for event in predictions if event.class_id == 1)
                bounces = sum(1 for event in predictions if event.class_id == 2)
                hit_count += hits
                bounce_count += bounces
                window_summaries.append({
                    "window_start_frame": processed - window_frames,
                    "hit": hits, "bounce": bounces,
                    "max_positive_class_probability": diagnostics["max_positive_class_probability"],
                })
                batch.clear()
        if batch:
            predictions, diagnostics = _predict(model, torch.stack(batch), threshold=args.threshold)
            hits = sum(1 for event in predictions if event.class_id == 1)
            bounces = sum(1 for event in predictions if event.class_id == 2)
            hit_count += hits
            bounce_count += bounces
    finally:
        capture.release()

    duration_s = processed / fps
    total_events = hit_count + bounce_count
    result = {
        "artifact_type": "ev2_train_20260728_firing_rate_measurement",
        "checkpoint": str(args.checkpoint),
        "video": str(args.video),
        "threshold": args.threshold,
        "window_frames": window_frames,
        "image_size": image_size,
        "fps": fps,
        "processed_frames": processed,
        "duration_s": duration_s,
        "hit_count": hit_count,
        "bounce_count": bounce_count,
        "total_event_count": total_events,
        "events_per_second": total_events / duration_s if duration_s > 0 else None,
        "hit_per_second": hit_count / duration_s if duration_s > 0 else None,
        "bounce_per_second": bounce_count / duration_s if duration_s > 0 else None,
        "window_policy": "non_overlapping_windows_covering_every_frame_once",
        "plausible_band_events_per_s": [0.3, 1.0],
        "north_star_zero_shot_reference_hit_per_s": 7.16,
        "verified": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "events_per_second": result["events_per_second"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
