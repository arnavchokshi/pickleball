#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.wasb_adapter import run_wasb_or_convert


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or convert WASB-SBDT predictions into ball_track.json.")
    parser.add_argument("--predictions-csv", type=Path, default=None, help="Existing WASB CSV to convert.")
    parser.add_argument("--video", type=Path, default=None, help="Video input for official WASB-SBDT inference.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Official WASB .pth.tar checkpoint.")
    parser.add_argument("--wasb-repo", type=Path, default=None, help="Official WASB-SBDT repo root.")
    parser.add_argument("--prediction-csv-out", type=Path, default=None, help="Keep raw WASB predictions CSV here.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--visible-threshold",
        type=float,
        default=0.5,
        help="BallTrack visibility cutoff for real WASB heatmap peak confidence. The primary threshold is 0.50.",
    )
    parser.add_argument(
        "--video-range",
        type=int,
        nargs=2,
        metavar=("START_S", "END_S"),
        default=None,
        help="Optional video trim range in seconds for WASB inference.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum frames for smoke runs.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--input-preprocessing",
        choices=("official", "harness_v0"),
        default="official",
        help=(
            "Input preprocessing for checkpoint inference. official keeps the WASB affine+ImageNet path; "
            "harness_v0 is resize+/255 and is a non-promotable measurement mode."
        ),
    )
    parser.add_argument(
        "--emit-candidates",
        action="store_true",
        help="Write top-K WASB blob candidates to ball_candidates.json next to --out during official inference.",
    )
    parser.add_argument("--candidate-top-k", type=int, default=5, help="Maximum ball candidates to keep per frame.")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--frame-times", type=Path, help="Optional frame_times.json for VFR-correct output timestamps.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path, default=None)
    args = parser.parse_args()

    try:
        summary = run_wasb_or_convert(
            out=args.out,
            fps=args.fps,
            frame_times=args.frame_times,
            metadata_out=args.metadata_out,
            predictions_csv=args.predictions_csv,
            video=args.video,
            checkpoint=args.checkpoint,
            wasb_repo=args.wasb_repo,
            prediction_csv_out=args.prediction_csv_out,
            batch_size=args.batch_size,
            visible_threshold=args.visible_threshold,
            video_range=args.video_range,
            max_frames=args.max_frames,
            device=args.device,
            emit_candidates=args.emit_candidates,
            candidate_top_k=args.candidate_top_k,
            input_preprocessing=args.input_preprocessing,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
