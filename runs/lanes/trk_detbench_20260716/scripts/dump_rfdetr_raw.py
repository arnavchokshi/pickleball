#!/usr/bin/env python3
"""RF-DETR raw-detection dumper — runs in the ISOLATED rfdetr venv (lane-dir script).

Writes per-frame raw person detections (conf >= floor) as JSON for the pipeline-venv
BOTSORT feeder (--detector raw-json), so rfdetr's dependency tree never touches the
pipeline venv. For the seg model, also archives per-detection masks (COCO RLE if
pycocotools is available, else polygons).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def rle_encode(mask: np.ndarray):
    try:
        from pycocotools import mask as mask_util

        rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        return {"format": "coco_rle", "size": rle["size"], "counts": rle["counts"]}
    except Exception:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return {"format": "polygon", "polygons": [c.reshape(-1).tolist() for c in contours if len(c) >= 3]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("large", "seg-large"))
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--mask-archive", type=Path, default=None)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--person-class-id", type=int, default=None, help="override; else resolved from class map")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    import rfdetr

    if args.model == "large":
        from rfdetr import RFDETRLarge as Model
    else:
        try:
            from rfdetr import RFDETRSegLarge as Model
        except ImportError:
            from rfdetr import RFDETRSegL as Model  # naming fallback across releases

    model = Model()
    version = getattr(rfdetr, "__version__", "unknown")

    # Resolve person class id from the model's class map and RECORD it.
    class_map = None
    for attr in ("class_names", "classes", "id2label"):
        cm = getattr(model, attr, None)
        if cm:
            class_map = cm
            break
    if class_map is None:
        try:
            from rfdetr.util.coco_classes import COCO_CLASSES  # type: ignore

            class_map = COCO_CLASSES
        except Exception:
            pass
    person_id = args.person_class_id
    if person_id is None and class_map is not None:
        if isinstance(class_map, dict):
            for k, v in class_map.items():
                if str(v).lower() == "person":
                    person_id = int(k)
                    break
        else:
            for i, v in enumerate(class_map):
                if str(v).lower() == "person":
                    person_id = int(i)
                    break
    if person_id is None:
        print("FATAL: could not resolve person class id; pass --person-class-id", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_out = []
    masks_out = []
    n = 0
    det_wall = 0.0
    while True:
        if args.max_frames is not None and n >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        det = model.predict(Image.fromarray(rgb), threshold=args.conf)
        dt = time.perf_counter() - t0
        det_wall += dt
        boxes = np.asarray(det.xyxy, dtype=np.float32).reshape(-1, 4)
        confs = np.asarray(det.confidence, dtype=np.float32).reshape(-1)
        clss = np.asarray(det.class_id, dtype=np.int64).reshape(-1)
        keep = clss == person_id
        dets = [
            {"bbox": [float(a), float(b), float(c), float(d)], "conf": float(cf), "class_id": int(ci)}
            for (a, b, c, d), cf, ci in zip(boxes[keep], confs[keep], clss[keep])
        ]
        frames_out.append({"frame": n, "detections": dets, "det_wall_s": dt})
        mask_attr = getattr(det, "mask", None)
        if args.mask_archive is not None and mask_attr is not None and len(mask_attr):
            marr = np.asarray(mask_attr)
            kept = marr[keep]
            masks_out.append(
                {
                    "frame": n,
                    "masks": [
                        {"conf": float(cf), "class_id": int(ci), "rle": rle_encode(m)}
                        for m, cf, ci in zip(kept, confs[keep], clss[keep])
                    ],
                }
            )
        n += 1
    cap.release()

    payload = {
        "fps": fps,
        "source_width": src_w,
        "source_height": src_h,
        "clip_id": args.clip_id,
        "detector": f"rfdetr-{args.model}",
        "rfdetr_version": version,
        "person_class_id": person_id,
        "class_map_sample": (list(class_map.items())[:5] if isinstance(class_map, dict) else list(class_map[:5])) if class_map is not None else None,
        "conf_floor": args.conf,
        "timing": {"detector_wall_seconds": det_wall, "n_frames": n, "detector_ms_per_frame_batch1": det_wall * 1000.0 / n if n else None},
        "frames": frames_out,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload), encoding="utf-8")
    if args.mask_archive is not None and masks_out:
        args.mask_archive.parent.mkdir(parents=True, exist_ok=True)
        args.mask_archive.write_text(json.dumps({"clip_id": args.clip_id, "detector": f"rfdetr-{args.model}", "frames": masks_out}), encoding="utf-8")
    print(json.dumps({"n_frames": n, "person_class_id": person_id, "rfdetr_version": version, "detector_ms_per_frame_batch1": payload["timing"]["detector_ms_per_frame_batch1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
