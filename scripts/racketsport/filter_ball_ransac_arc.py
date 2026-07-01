#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_ransac_arc_gate import write_ransac_arc_filtered_ball_track


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter BALL track samples by quadratic-arc RANSAC residual.")
    parser.add_argument("--ball-track", type=Path, required=True)
    parser.add_argument("--max-residual-px", type=float, default=5.0)
    parser.add_argument("--min-fit-points", type=int, default=5)
    parser.add_argument("--max-gap-frames", type=int, default=6)
    parser.add_argument("--max-trials", type=int, default=2000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    try:
        summary = write_ransac_arc_filtered_ball_track(
            ball_track_path=args.ball_track,
            out_path=args.out,
            summary_path=args.summary_out,
            max_residual_px=args.max_residual_px,
            min_fit_points=args.min_fit_points,
            max_gap_frames=args.max_gap_frames,
            max_trials=args.max_trials,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
