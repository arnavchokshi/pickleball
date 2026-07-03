#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.racket_detector_mask_gate import (
    build_detector_mask_report,
    load_yolo_rows,
    run_grounding_dino_sam2_rows,
    run_yolo_sam2_rows,
    write_json,
    write_records,
)


GROUNDING_DINO_DEFAULT = "models/checkpoints/racket/grounding-dino-tiny"
YOLO_DEFAULT = "models/checkpoints/yolo11n.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real paddle detector + SAM2 mask gate.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("runs/cvat_imports/2026_06_30/yolo_datasets/paddle/manifest.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--records-out", type=Path)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--clip", action="append", default=[])
    parser.add_argument("--max-frames-per-clip", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--detector-kind", choices=("groundingdino", "yolo"), default="groundingdino")
    parser.add_argument("--detector-model")
    parser.add_argument("--sam2-model", default="facebook/sam2-hiera-tiny")
    parser.add_argument("--prompt", default="pickleball paddle.")
    parser.add_argument("--detector-device", default="auto")
    parser.add_argument("--sam2-device", default="cpu")
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.10)
    parser.add_argument("--yolo-imgsz", type=int, default=960)
    parser.add_argument("--yolo-conf", type=float, default=0.01)
    parser.add_argument("--yolo-iou", type=float, default=0.7)
    parser.add_argument("--yolo-class-id", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=0, help="Enable sliced inference with this tile size in px.")
    parser.add_argument("--tile-overlap", type=int, default=0, help="Tile overlap in px when --tile-size is enabled.")
    parser.add_argument("--tile-nms-iou", type=float, default=0.6, help="NMS IoU for merging tiled detections.")
    parser.add_argument("--min-box-area", type=float)
    parser.add_argument("--max-box-area", type=float)
    parser.add_argument("--min-box-aspect", type=float)
    parser.add_argument("--max-box-aspect", type=float)
    parser.add_argument("--sam2-box-batch-size", type=int)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--recall-gate", type=float, default=0.90)
    args = parser.parse_args()

    records_out = args.records_out or args.out.with_name(args.out.stem + "_records.json")
    mask_dir = args.mask_dir or args.out.with_name(args.out.stem + "_masks")
    rows = load_yolo_rows(
        args.manifest,
        max_frames_per_clip=args.max_frames_per_clip,
        frame_stride=args.frame_stride,
        clip_ids=set(args.clip) if args.clip else None,
    )
    detector_model = args.detector_model or (
        YOLO_DEFAULT if args.detector_kind == "yolo" else GROUNDING_DINO_DEFAULT
    )
    if args.detector_kind == "yolo":
        records = run_yolo_sam2_rows(
            rows,
            detector_model=str(detector_model),
            sam2_model=str(args.sam2_model),
            detector_device=args.detector_device,
            sam2_device=args.sam2_device,
            imgsz=args.yolo_imgsz,
            conf=args.yolo_conf,
            nms_iou=args.yolo_iou,
            class_id=args.yolo_class_id,
            mask_dir=mask_dir,
            tile_size=args.tile_size if args.tile_size > 0 else None,
            tile_overlap=args.tile_overlap,
            tile_nms_iou=args.tile_nms_iou,
            min_box_area=args.min_box_area,
            max_box_area=args.max_box_area,
            min_box_aspect=args.min_box_aspect,
            max_box_aspect=args.max_box_aspect,
            sam2_box_batch_size=args.sam2_box_batch_size,
        )
    else:
        records = run_grounding_dino_sam2_rows(
            rows,
            detector_model=str(detector_model),
            sam2_model=str(args.sam2_model),
            prompt=args.prompt,
            detector_device=args.detector_device,
            sam2_device=args.sam2_device,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            mask_dir=mask_dir,
            tile_size=args.tile_size if args.tile_size > 0 else None,
            tile_overlap=args.tile_overlap,
            tile_nms_iou=args.tile_nms_iou,
            min_box_area=args.min_box_area,
            max_box_area=args.max_box_area,
            min_box_aspect=args.min_box_aspect,
            max_box_aspect=args.max_box_aspect,
            sam2_box_batch_size=args.sam2_box_batch_size,
        )
    write_records(records_out, records)
    report = build_detector_mask_report(
        records,
        model_sources={
            "detector_kind": args.detector_kind,
            "detector": str(detector_model),
            "mask": str(args.sam2_model),
            "prompt": args.prompt,
        },
        iou_threshold=args.iou_threshold,
        recall_gate=args.recall_gate,
    )
    report["inputs"] = {
        "manifest": str(args.manifest),
        "records": str(records_out),
        "mask_dir": str(mask_dir),
        "selected_frame_count": len(rows),
        "max_frames_per_clip": args.max_frames_per_clip,
        "frame_stride": args.frame_stride,
        "clips": args.clip,
        "detector_kind": args.detector_kind,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "yolo_imgsz": args.yolo_imgsz,
        "yolo_conf": args.yolo_conf,
        "yolo_iou": args.yolo_iou,
        "yolo_class_id": args.yolo_class_id,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "tile_nms_iou": args.tile_nms_iou,
        "min_box_area": args.min_box_area,
        "max_box_area": args.max_box_area,
        "min_box_aspect": args.min_box_aspect,
        "max_box_aspect": args.max_box_aspect,
        "sam2_box_batch_size": args.sam2_box_batch_size,
        "detector_device": args.detector_device,
        "sam2_device": args.sam2_device,
    }
    write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
