#!/usr/bin/env python3
"""D-FINE raw-detection dumper — trk_detbench_20260716 lane script.

Mirrors the repo's tools/inference/torch_inf.py flow: YAMLConfig(resume=ckpt),
deploy() model + postprocessor, 640x640 resize + ToTensor, orig_target_sizes to
rescale boxes. Keeps person detections conf >= floor. Emits the same raw JSON
contract consumed by candidate_detector_feeder.py --detector raw-json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--person-class-id", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--probe", action="store_true", help="print top raw labels/boxes on frame 0 and exit")
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo))
    from src.core import YAMLConfig  # noqa: E402

    cfg = YAMLConfig(str(args.config), resume=str(args.checkpoint))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    state = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    if "ema" in state:
        weights = state["ema"]["module"]
    else:
        weights = state["model"]
    cfg.model.load_state_dict(weights)

    class Deploy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    device = torch.device(args.device)
    model = Deploy().to(device).eval()
    tf = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_out = []
    n = 0
    det_wall = 0.0
    with torch.no_grad():
        while True:
            if args.max_frames is not None and n >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            orig_size = torch.tensor([[src_w, src_h]], device=device)
            img = tf(pil).unsqueeze(0).to(device)
            t0 = time.perf_counter()
            labels, boxes, scores = model(img, orig_size)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            det_wall += dt
            labels = labels[0].cpu().numpy()
            boxes = boxes[0].cpu().numpy()
            scores = scores[0].cpu().numpy()
            if args.probe and n == 0:
                import collections

                order = np.argsort(-scores)
                print("label distribution (score>=0.3):", collections.Counter(labels[scores >= 0.3].tolist()))
                for i in order[:10]:
                    print(f"  label={labels[i]} score={scores[i]:.3f} box={boxes[i].round(0).tolist()}")
                return 0
            keep = (labels == args.person_class_id) & (scores >= args.conf)
            dets = [
                {"bbox": [float(a), float(b), float(c), float(d)], "conf": float(s), "class_id": int(l)}
                for (a, b, c, d), s, l in zip(boxes[keep], scores[keep], labels[keep])
            ]
            frames_out.append({"frame": n, "detections": dets, "det_wall_s": dt})
            n += 1
    cap.release()

    payload = {
        "fps": fps,
        "source_width": src_w,
        "source_height": src_h,
        "clip_id": args.clip_id,
        "detector": "dfine_l_obj2coco_e25",
        "person_class_id": args.person_class_id,
        "conf_floor": args.conf,
        "timing": {"detector_wall_seconds": det_wall, "n_frames": n, "detector_ms_per_frame_batch1": det_wall * 1000.0 / n if n else None},
        "frames": frames_out,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps({"n_frames": n, "detector_ms_per_frame_batch1": payload["timing"]["detector_ms_per_frame_batch1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
