#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.person_reid_diagnostics import (  # noqa: E402
    ReIDEmbeddingExportConfig,
    build_source_reid_embedding_export,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export learned per-detection ReID embeddings from tracked detections and source video crops."
    )
    parser.add_argument("--video", type=Path, required=True, help="Source video used to crop player detections.")
    parser.add_argument("--detections", type=Path, required=True, help="tracked_detections.json or raw_tracked_detections.json.")
    parser.add_argument("--model", type=Path, required=True, help="YOLO/ReID model path used to produce learned embeddings.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path for per-detection embeddings.")
    parser.add_argument("--max-detections", type=int, default=None, help="Optional hard cap for bounded exports.")
    parser.add_argument("--sample-stride-frames", type=int, default=1)
    parser.add_argument("--crop-padding-px", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--embed-layer", type=int, default=21)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0, cuda:0, mps, or cpu.")
    parser.add_argument("--half", action="store_true", help="Request FP16 inference when supported by the selected device.")
    parser.add_argument("--no-l2-normalize", action="store_true", help="Persist raw model features without L2 normalization.")
    args = parser.parse_args()

    try:
        detections_payload = _read_json_object(args.detections)
        report = build_source_reid_embedding_export(
            video_path=args.video,
            detections_payload=detections_payload,
            output_path=args.output,
            model_path=args.model,
            command_metadata={
                "argv": sys.argv,
                "cwd": os.getcwd(),
                "detections_path": str(args.detections),
            },
            config=ReIDEmbeddingExportConfig(
                max_detections=args.max_detections,
                sample_stride_frames=args.sample_stride_frames,
                crop_padding_px=args.crop_padding_px,
                batch_size=args.batch_size,
                imgsz=args.imgsz,
                embed_layer=args.embed_layer,
                device=args.device,
                half=True if args.half else None,
                l2_normalize=not args.no_l2_normalize,
            ),
        )
    except Exception as exc:
        print(f"person ReID embedding export failed: {exc}", file=sys.stderr)
        return 1

    print(args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "detection_count": report["detection_count"],
                "feature_dim": report["feature_dim"],
                "model_sha256": report["model_sha256"],
                "promote_trk": report["promote_trk"],
            },
            sort_keys=True,
        )
    )
    return 0


def _read_json_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
