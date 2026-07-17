#!/usr/bin/env python3
"""Arm 0b feeder — trk_detbench_20260716 (lane-dir only; NOT pipeline code).

Fresh YOLO26m detections at the PRODUCTION operating point (conf=0.05, imgsz=1536,
classes=[0], per threed/racketsport/orchestrator.py RealYOLO26BoTSORTReIDTrackingRunner
defaults) fed through the ultralytics BOTSORT tracker constructed from
configs/racketsport/botsort_no_reid_loose.yaml via its per-frame update() API
(NOT model.track()/stream persistence) -> pool JSON in the tracked_detections.json
schema + a metrics.json counts block with source/calibration dims and bbox_scale,
so the frozen association step (run_raw_pool_person_authority.py) can consume it
exactly like a production pool.

This is the arm-0b "confound check" feeder: same detector+tracker as production,
but driven through the raw per-frame update() API instead of model.track(stream=True),
to isolate any behavioral delta introduced by the streaming/track() convenience path
before candidate detectors are ever trusted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def build_tracker(tracker_yaml: Path, device: str):
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml
    from ultralytics.trackers.bot_sort import BOTSORT

    resolved = check_yaml(str(tracker_yaml))
    cfg = IterableSimpleNamespace(**YAML.load(resolved))
    cfg.device = device
    assert cfg.tracker_type == "botsort", f"expected botsort, got {cfg.tracker_type}"
    assert cfg.with_reid is False, "arm 0b must use the no-reid loose config (with_reid: False)"
    return BOTSORT(args=cfg)


def main() -> int:
    ap = argparse.ArgumentParser(description="Arm 0b feeder: YOLO26m -> BOTSORT per-frame update() pool")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True, help="models/checkpoints/yolo26m.pt")
    ap.add_argument("--tracker-yaml", type=Path, required=True, help="configs/racketsport/botsort_no_reid_loose.yaml")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--device", default="0")
    ap.add_argument("--calibration-width", type=int, default=1920)
    ap.add_argument("--calibration-height", type=int, default=1080)
    ap.add_argument("--max-frames", type=int, default=None, help="Smoke-test cap: stop after N frames.")
    args = ap.parse_args()

    from ultralytics import YOLO

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"could not open video {args.video}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    model = YOLO(str(args.weights))
    tracker = build_tracker(args.tracker_yaml, device=args.device)

    frames_payload = []
    t0 = time.perf_counter()
    n_frames = 0
    n_dets_pre_track = 0
    n_dets_post_track = 0

    stream = model.predict(
        source=str(args.video),
        conf=args.conf,
        imgsz=args.imgsz,
        iou=args.iou,
        classes=[0],
        device=args.device,
        stream=True,
        verbose=False,
    )

    for frame_idx, result in enumerate(stream):
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break
        det = result.boxes.cpu().numpy()
        n_dets_pre_track += len(det)
        tracks = tracker.update(det, result.orig_img)
        detections = []
        if len(tracks) > 0:
            for row in tracks:
                x1, y1, x2, y2, track_id, score, cls_id, _idx = row.tolist()
                detections.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "class": "person",
                        "conf": float(score),
                        "track_id": int(track_id),
                    }
                )
            n_dets_post_track += len(detections)
        frames_payload.append({"frame": frame_idx, "detections": detections})
        n_frames += 1

    wall_s = time.perf_counter() - t0
    ms_per_frame = (wall_s * 1000.0 / n_frames) if n_frames else None

    pool = {"fps": fps, "frames": frames_payload}
    pool_path = args.out_dir / "tracked_detections.json"
    pool_path.write_text(json.dumps(pool), encoding="utf-8")
    # raw_tracked_detections.json mirrors tracked_detections.json here (no separate
    # pre-role-lock raw stage in this feeder; the BOTSORT-updated pool IS the raw pool
    # that run_raw_pool_person_authority.py consumes).
    (args.out_dir / "raw_tracked_detections.json").write_text(json.dumps(pool), encoding="utf-8")

    metrics = {
        "artifact_type": "racketsport_person_tracker_candidate",
        "clip": args.clip_id,
        "arm": "arm0b_yolo26m_botsort_update_feeder",
        "batch_size": 1,
        "counts": {
            "source_width": src_w,
            "source_height": src_h,
            "calibration_width": args.calibration_width,
            "calibration_height": args.calibration_height,
            "bbox_scale_x": src_w / args.calibration_width if args.calibration_width else 1.0,
            "bbox_scale_y": src_h / args.calibration_height if args.calibration_height else 1.0,
            "total_frames": n_frames,
            "detections_pre_track": n_dets_pre_track,
            "detections_post_track": n_dets_post_track,
        },
        "detector": {
            "model": str(args.weights),
            "conf": args.conf,
            "imgsz": args.imgsz,
            "iou": args.iou,
            "classes": [0],
        },
        "tracker": {
            "config": str(args.tracker_yaml),
            "api": "per_frame_update",
        },
        "timing": {
            "wall_seconds": wall_s,
            "n_frames": n_frames,
            "ms_per_frame_batch1": ms_per_frame,
        },
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps({"pool_path": str(pool_path), "metrics_path": str(args.out_dir / "metrics.json"), "n_frames": n_frames, "ms_per_frame_batch1": ms_per_frame}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
