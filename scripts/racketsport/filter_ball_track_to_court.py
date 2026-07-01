#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_court_filter import write_filtered_ball_track  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter TrackNet ball detections to the calibrated target court.")
    parser.add_argument("--ball-track", type=Path, required=True, help="Input ball_track.json path.")
    parser.add_argument("--calibration", type=Path, required=True, help="Input court_calibration.json path.")
    parser.add_argument("--target-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None)
    parser.add_argument("--margin-px", type=float, default=80.0, help="Pixel margin around the target court polygon.")
    parser.add_argument(
        "--margin-m",
        type=float,
        default=None,
        help="Metric margin around regulation court in meters. Overrides --margin-px when set; M2 spec default is 0.5.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output filtered ball_track.json path.")
    parser.add_argument("--summary-out", type=Path, required=True, help="Output filter summary JSON path.")
    args = parser.parse_args()

    try:
        summary = write_filtered_ball_track(
            ball_track_path=args.ball_track,
            calibration_path=args.calibration,
            out_path=args.out,
            summary_path=args.summary_out,
            target_size=tuple(args.target_size) if args.target_size is not None else None,
            margin_px=args.margin_px,
            margin_m=args.margin_m,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
