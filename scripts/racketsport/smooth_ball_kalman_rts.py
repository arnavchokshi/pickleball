#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_kalman_rts_smoother import write_kalman_rts_smoothed_ball_track


def main() -> int:
    parser = argparse.ArgumentParser(description="Smooth a BALL track with constant-acceleration Kalman + RTS.")
    parser.add_argument("--ball-track", type=Path, required=True)
    parser.add_argument("--max-gap-fill-frames", type=int, default=6)
    parser.add_argument("--measurement-variance-px", type=float, default=4.0)
    parser.add_argument("--process-variance", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    try:
        summary = write_kalman_rts_smoothed_ball_track(
            ball_track_path=args.ball_track,
            out_path=args.out,
            summary_path=args.summary_out,
            max_gap_fill_frames=args.max_gap_fill_frames,
            measurement_variance_px=args.measurement_variance_px,
            process_variance=args.process_variance,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
