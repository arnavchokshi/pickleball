#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_tracknet_cvat_dataset import (  # noqa: E402
    MANIFEST_JSON,
    MANIFEST_MD,
    build_ball_tracknet_cvat_dataset,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build dense TrackNetV3 label CSVs from reviewed CVAT ball boxes.",
    )
    parser.add_argument("--cvat-root", type=Path, required=True, help="Root containing <clip>/reviewed_boxes.json files.")
    parser.add_argument("--yolo-manifest", type=Path, required=True, help="Ball YOLO manifest whose by-clip split is reused.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for TrackNet CSVs and manifest artifacts.")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--clip", action="append", default=[], help="Optional clip id to include. May be repeated.")
    parser.add_argument("--materialize-frames", action="store_true", help="Extract TrackNet frame PNGs and median.npz files.")
    parser.add_argument("--video", action="append", default=[], help="Video mapping clip_id=/path/to/video.mp4. May be repeated.")
    parser.add_argument("--hard-negative-plan", type=Path, help="BALL hard-negative iteration plan JSON to consume.")
    parser.add_argument("--hard-negative-context-frames", type=int, default=0, help="Context frames to include around hard-negative ranges.")
    parser.add_argument("--hard-negative-repeat", type=int, default=1, help="Deterministic oversampling repeat count for hard-negative windows.")
    args = parser.parse_args(argv)

    try:
        video_paths = _parse_video_args(args.video)
        manifest = build_ball_tracknet_cvat_dataset(
            cvat_root=args.cvat_root,
            yolo_manifest=args.yolo_manifest,
            out_dir=args.out_dir,
            fps=args.fps,
            clips=tuple(args.clip) if args.clip else None,
            materialize_frames=args.materialize_frames,
            video_paths=video_paths,
            hard_negative_plan=args.hard_negative_plan,
            hard_negative_context_frames=args.hard_negative_context_frames,
            hard_negative_repeat=args.hard_negative_repeat,
        )
    except Exception as exc:
        print(f"BALL TrackNet CVAT dataset build failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest_json": str(args.out_dir / MANIFEST_JSON),
                "manifest_md": str(args.out_dir / MANIFEST_MD),
                "label_counts": manifest["label_counts"],
                "splits": {
                    split: [row["clip"] for row in rows]
                    for split, rows in manifest["splits"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_video_args(items: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--video must be clip_id=/path/to/video.mp4: {item}")
        clip, path = item.split("=", 1)
        if not clip or not path:
            raise ValueError(f"--video must include both clip id and path: {item}")
        parsed[clip] = Path(path)
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
