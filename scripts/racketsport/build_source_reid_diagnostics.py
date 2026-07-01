#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.person_reid_diagnostics import (  # noqa: E402
    AppearanceDiagnosticConfig,
    build_source_appearance_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build source-only appearance/ReID diagnostics from tracked detections and source video."
    )
    parser.add_argument("--video", type=Path, required=True, help="Source video used to crop player detections.")
    parser.add_argument("--detections", type=Path, required=True, help="tracked_detections.json or raw_tracked_detections.json.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory under runs/phase2.")
    parser.add_argument("--max-samples-per-track", type=int, default=24)
    parser.add_argument("--sample-stride-frames", type=int, default=15)
    parser.add_argument("--crop-padding-px", type=int, default=8)
    parser.add_argument("--histogram-bins", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=96)
    args = parser.parse_args()

    try:
        detections_payload = _read_json_object(args.detections)
        report = build_source_appearance_diagnostic(
            video_path=args.video,
            detections_payload=detections_payload,
            out_dir=args.out_dir,
            config=AppearanceDiagnosticConfig(
                max_samples_per_track=args.max_samples_per_track,
                sample_stride_frames=args.sample_stride_frames,
                crop_padding_px=args.crop_padding_px,
                histogram_bins=args.histogram_bins,
                tile_size=args.tile_size,
            ),
        )
    except Exception as exc:
        print(f"source appearance/ReID diagnostic failed: {exc}", file=sys.stderr)
        return 1

    print(args.out_dir / "source_appearance_diagnostics.json")
    print(args.out_dir / "track_appearance_features.json")
    print(json.dumps({"status": report["status"], "sample_count": report["sample_count"], "promote_trk": report["promote_trk"]}, sort_keys=True))
    return 0


def _read_json_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
