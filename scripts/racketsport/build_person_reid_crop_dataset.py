#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.person_reid_dataset import (  # noqa: E402
    PersonReIDDatasetConfig,
    clip_specs_from_import_manifest,
    export_person_reid_crop_dataset,
    parse_reid_clip_specs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a labeled person ReID crop dataset from reviewed CVAT player boxes.")
    parser.add_argument("--manifest", type=Path, default=Path("runs/cvat_imports/2026_06_30/manifest.json"))
    parser.add_argument("--clip", action="append", default=[], help="clip_id=video.mp4=person_ground_truth.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--val-clip", action="append", required=True, help="Clip held out as query/gallery identities.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--query-every", type=int, default=10)
    parser.add_argument("--crop-padding-px", type=int, default=12)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-samples-per-identity", type=int, default=None)
    args = parser.parse_args()

    try:
        clips = parse_reid_clip_specs(args.clip) if args.clip else clip_specs_from_import_manifest(args.manifest)
        summary = export_person_reid_crop_dataset(
            clips=clips,
            out_dir=args.out_dir,
            config=PersonReIDDatasetConfig(
                split_mode="by_clip",
                val_clips=tuple(args.val_clip),
                frame_stride=args.frame_stride,
                query_every=args.query_every,
                crop_padding_px=args.crop_padding_px,
                jpeg_quality=args.jpeg_quality,
                max_samples_per_identity=args.max_samples_per_identity,
            ),
        )
    except Exception as exc:
        print(f"person ReID crop dataset export failed: {exc}", file=sys.stderr)
        return 1

    compact = {key: value for key, value in summary.items() if key != "rows"}
    print(json.dumps(compact, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
