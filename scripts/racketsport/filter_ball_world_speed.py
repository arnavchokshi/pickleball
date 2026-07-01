#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_world_speed_gate import write_world_speed_filtered_ball_track


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter BALL track links by calibrated world speed and pixel jump.")
    parser.add_argument("--ball-track", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--target-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None)
    parser.add_argument("--max-world-speed-mps", type=float, default=30.0)
    parser.add_argument("--base-jump-px", type=float, default=60.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    try:
        summary = write_world_speed_filtered_ball_track(
            ball_track_path=args.ball_track,
            calibration_path=args.calibration,
            target_size=tuple(args.target_size) if args.target_size is not None else None,
            max_world_speed_mps=args.max_world_speed_mps,
            base_jump_px=args.base_jump_px,
            out_path=args.out,
            summary_path=args.summary_out,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
