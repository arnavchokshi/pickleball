#!/usr/bin/env python3
"""Candidate-detector feeder — trk_detbench_20260716 (lane-dir only; NOT pipeline code).

Runs a candidate detector (rfdetr / rfdetr-seg / dfine / deimv2) over a clip, keeps ALL
person detections with conf >= 0.05 (production pool floor), feeds them through the SAME
ultralytics BOTSORT (configs/racketsport/botsort_no_reid_loose.yaml) per-frame update()
API used by arm 0b, and writes the pool JSON schema + metrics.json counts block that
run_raw_pool_person_authority.py consumes. Detector-swap discipline: ONLY the detection
stage differs from arm 0b; the tracker construction and pool schema are byte-identical
code paths.

For rfdetr-seg, per-detection masks are archived (RLE via pycocotools if available,
else polygon fallback) to --mask-archive for the future mask-cue lane; boxes are scored.

The detector runs inside its own venv; the BOTSORT update step needs ultralytics, so run
this script with the PIPELINE venv python and give --detector-venv-python for the
detector subprocess... NO -- simpler contract used here: this script is run with a python
that has BOTH the candidate detector package AND ultralytics importable. If dependency
isolation forces separate venvs, run the detector with --dump-raw-only in the detector
venv (writes raw per-frame detections JSON), then re-run with --from-raw in the pipeline
venv to do the BOTSORT pass.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def build_tracker(tracker_yaml: Path, device: str):
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml
    from ultralytics.trackers.bot_sort import BOTSORT

    resolved = check_yaml(str(tracker_yaml))
    cfg = IterableSimpleNamespace(**YAML.load(resolved))
    cfg.device = device
    assert cfg.tracker_type == "botsort", f"expected botsort, got {cfg.tracker_type}"
    assert cfg.with_reid is False, "candidate arms must use the no-reid loose config"
    return BOTSORT(args=cfg)


class _DetBoxes:
    """Minimal Boxes-like shim for BYTETracker.update(): needs .conf, .xywh (or parse via
    bot_sort init_track path which uses parse_bboxes on the results object). We mirror the
    ultralytics Boxes numpy interface subset BOTSORT actually touches: conf, xywh, cls,
    boolean indexing, len()."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self._xyxy = xyxy.astype(np.float32).reshape(-1, 4)
        self.conf = conf.astype(np.float32).reshape(-1)
        self.cls = cls.astype(np.float32).reshape(-1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, mask):
        return _DetBoxes(self._xyxy[mask], self.conf[mask], self.cls[mask])

    @property
    def xyxy(self):
        return self._xyxy

    @property
    def xywh(self):
        x1, y1, x2, y2 = self._xyxy.T
        return np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], axis=1)


def iter_frames(video_path: Path, max_frames: int | None):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def gen():
        i = 0
        while True:
            if max_frames is not None and i >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
        cap.release()

    return fps, src_w, src_h, gen()


def detect_rfdetr(args, frames, seg: bool):
    """Yield (frame_idx, frame_bgr, xyxy, conf, cls_ids, masks_or_None, det_wall_s)."""
    if seg:
        from rfdetr import RFDETRSegLarge as Model  # noqa
    else:
        from rfdetr import RFDETRLarge as Model  # noqa
    kwargs = {}
    if args.weights:
        kwargs["pretrain_weights"] = str(args.weights)
    model = Model(**kwargs)
    # person class id: record from the model's class map (COCO: person is usually 1 in
    # rfdetr's map -- VERIFY at runtime and print it)
    class_map = getattr(model, "class_names", None) or getattr(model, "classes", None)
    print(f"RFDETR class map (verify person id): {class_map}", file=sys.stderr)
    for idx, frame in frames:
        t0 = time.perf_counter()
        from PIL import Image
        import cv2 as _cv2

        rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
        det = model.predict(Image.fromarray(rgb), threshold=args.conf)
        dt = time.perf_counter() - t0
        boxes = np.asarray(det.xyxy, dtype=np.float32) if len(det.xyxy) else np.zeros((0, 4), np.float32)
        confs = np.asarray(det.confidence, dtype=np.float32) if len(det.xyxy) else np.zeros((0,), np.float32)
        clss = np.asarray(det.class_id, dtype=np.int64) if len(det.xyxy) else np.zeros((0,), np.int64)
        keep = clss == args.person_class_id
        masks = None
        if seg and getattr(det, "mask", None) is not None and len(det.mask):
            masks = np.asarray(det.mask)[keep]
        yield idx, frame, boxes[keep], confs[keep], clss[keep], masks, dt


def rle_encode(mask: np.ndarray):
    try:
        from pycocotools import mask as mask_util

        rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        return {"format": "coco_rle", **rle}
    except Exception:
        # polygon fallback
        import cv2

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return {"format": "polygon", "polygons": [c.reshape(-1).tolist() for c in contours if len(c) >= 3]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Candidate detector -> BOTSORT update() -> pool JSON")
    ap.add_argument("--detector", required=True, choices=("rfdetr", "rfdetr-seg", "raw-json"))
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tracker-yaml", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--person-class-id", type=int, required=True, help="VERIFY from the model class map first")
    ap.add_argument("--device", default="0")
    ap.add_argument("--mask-archive", type=Path, default=None)
    ap.add_argument("--raw-json", type=Path, default=None, help="raw-json detector: per-frame detections JSON produced in the detector venv")
    ap.add_argument("--calibration-width", type=int, default=1920)
    ap.add_argument("--calibration-height", type=int, default=1080)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fps, src_w, src_h, frames = iter_frames(args.video, args.max_frames)
    tracker = build_tracker(args.tracker_yaml, device=args.device)

    if args.detector == "raw-json":
        raw = json.loads(Path(args.raw_json).read_text())
        raw_by_frame = {f["frame"]: f for f in raw["frames"]}

        def det_iter():
            for idx, frame in frames:
                entry = raw_by_frame.get(idx, {"detections": []})
                dets = entry["detections"]
                boxes = np.asarray([d["bbox"] for d in dets], np.float32).reshape(-1, 4)
                confs = np.asarray([d["conf"] for d in dets], np.float32)
                clss = np.asarray([0] * len(dets), np.int64)
                yield idx, frame, boxes, confs, clss, None, entry.get("det_wall_s", 0.0)

        gen = det_iter()
        det_name = f"raw-json:{args.raw_json}"
    else:
        gen = detect_rfdetr(args, frames, seg=(args.detector == "rfdetr-seg"))
        det_name = args.detector

    frames_payload = []
    mask_archive = []
    det_wall_total = 0.0
    n_frames = 0
    for idx, frame, boxes, confs, clss, masks, dt in gen:
        det_wall_total += float(dt)
        n = len(confs)
        det = _DetBoxes(boxes, confs, np.zeros(n))  # tracker sees class 0 (person) only
        tracks = tracker.update(det, frame)
        detections = []
        for row in tracks:
            x1, y1, x2, y2, track_id, score, cls_id, _i = row.tolist()
            detections.append({"bbox": [float(x1), float(y1), float(x2), float(y2)], "class": "person", "conf": float(score), "track_id": int(track_id)})
        frames_payload.append({"frame": idx, "detections": detections})
        if masks is not None and args.mask_archive is not None:
            mask_archive.append({"frame": idx, "masks": [{"conf": float(c), "class_id": int(k), "rle": rle_encode(m)} for m, c, k in zip(masks, confs, clss)]})
        n_frames += 1

    pool = {"fps": fps, "frames": frames_payload}
    (args.out_dir / "tracked_detections.json").write_text(json.dumps(pool), encoding="utf-8")
    (args.out_dir / "raw_tracked_detections.json").write_text(json.dumps(pool), encoding="utf-8")
    metrics = {
        "artifact_type": "racketsport_person_tracker_candidate",
        "clip": args.clip_id,
        "arm": det_name,
        "counts": {
            "source_width": src_w, "source_height": src_h,
            "calibration_width": args.calibration_width, "calibration_height": args.calibration_height,
            "bbox_scale_x": src_w / args.calibration_width, "bbox_scale_y": src_h / args.calibration_height,
            "total_frames": n_frames,
        },
        "detector": {"name": det_name, "conf_floor": args.conf, "person_class_id": args.person_class_id, "weights": str(args.weights) if args.weights else None},
        "tracker": {"config": str(args.tracker_yaml), "api": "per_frame_update"},
        "timing": {"detector_wall_seconds": det_wall_total, "n_frames": n_frames,
                   "detector_ms_per_frame_batch1": (det_wall_total * 1000.0 / n_frames) if n_frames else None},
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.mask_archive is not None and mask_archive:
        args.mask_archive.parent.mkdir(parents=True, exist_ok=True)
        args.mask_archive.write_text(json.dumps({"clip_id": args.clip_id, "frames": mask_archive}), encoding="utf-8")
    print(json.dumps({"n_frames": n_frames, "detector_ms_per_frame_batch1": metrics["timing"]["detector_ms_per_frame_batch1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
