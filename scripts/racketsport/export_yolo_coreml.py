#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a YOLO person model to Core ML for iPhone FastTier testing.")
    parser.add_argument(
        "--weights",
        default=str(ROOT / "models" / "checkpoints" / "yolo26n.pt"),
        help="YOLO weights path or model name.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("models_coreml"), help="Directory for exported Core ML assets.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input image size.")
    parser.add_argument("--quantize", type=int, default=8, choices=(8, 16), help="Core ML quantization mode.")
    parser.add_argument("--batch", type=int, default=1, help="Core ML export batch size.")
    parser.add_argument("--nms", action="store_true", help="Enable export-time NMS; intended only for YOLO11 fallback.")
    parser.add_argument("--dry-run", action="store_true", help="Print the exact export arguments without running export.")
    args = parser.parse_args()

    export_kwargs = {
        "format": "coreml",
        "imgsz": args.imgsz,
        "batch": args.batch,
    }
    export_kwargs["quantize"] = args.quantize
    if args.nms:
        export_kwargs["nms"] = True

    printable = " ".join([f"model={args.weights}"] + [f"{key}={value}" for key, value in export_kwargs.items()])
    if args.dry_run:
        print(printable)
        return 0

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        print(f"ultralytics is required for Core ML export: {exc}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model = YOLO(args.weights)
        exported = Path(model.export(**export_kwargs))
    except Exception as exc:
        print(f"Core ML export failed: {exc}", file=sys.stderr)
        return 1

    destination = args.out_dir / exported.name
    if exported.resolve() != destination.resolve():
        if exported.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(exported, destination)
        else:
            shutil.copy2(exported, destination)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
