#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.tracknet_adapter import run_tracknet_or_convert


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or convert TrackNetV3 predictions into ball_track.json.")
    parser.add_argument("--predictions-csv", type=Path, default=None, help="Existing official TrackNet CSV to convert.")
    parser.add_argument("--video", type=Path, default=None, help="Video input for official TrackNetV3 predict.py.")
    parser.add_argument("--tracknet-file", type=Path, default=None, help="TrackNet_best.pt checkpoint.")
    parser.add_argument("--inpaintnet-file", type=Path, default=None, help="InpaintNet_best.pt checkpoint.")
    parser.add_argument("--tracknet-repo", type=Path, default=None, help="Official TrackNetV3 repo containing predict.py.")
    parser.add_argument("--prediction-dir", type=Path, default=None, help="Keep official TrackNet CSV outputs in this directory.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--video-range",
        type=int,
        nargs=2,
        metavar=("START_S", "END_S"),
        default=None,
        help="Pass official TrackNetV3 --video_range START,END for background median sampling; it does not trim prediction frames.",
    )
    parser.add_argument("--large-video", action="store_true", help="Pass --large_video to official TrackNetV3 predict.py.")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, default=None)
    args = parser.parse_args()

    try:
        summary = run_tracknet_or_convert(
            out=args.out,
            fps=args.fps,
            metadata_out=args.metadata_out,
            predictions_csv=args.predictions_csv,
            video=args.video,
            tracknet_file=args.tracknet_file,
            inpaintnet_file=args.inpaintnet_file,
            tracknet_repo=args.tracknet_repo,
            prediction_dir=args.prediction_dir,
            batch_size=args.batch_size,
            video_range=args.video_range,
            large_video=args.large_video,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
