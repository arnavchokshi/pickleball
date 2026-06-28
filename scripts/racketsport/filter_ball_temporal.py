#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_temporal_filter import write_temporal_filtered_ball_track  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter a ball track by target-ball temporal consistency.")
    parser.add_argument("--ball-track", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--mode", choices=("path", "outlier", "local_trajectory"), default="path")
    parser.add_argument("--max-speed-px-per-second", type=float, default=7200.0)
    parser.add_argument("--base-jump-px", type=float, default=60.0)
    parser.add_argument("--max-link-gap-frames", type=int, default=10)
    parser.add_argument("--max-interpolate-gap-frames", type=int, default=3)
    parser.add_argument("--min-chain-visible-frames", type=int, default=3)
    parser.add_argument("--max-neighbor-gap-frames", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--local-trajectory-window-frames", type=int, default=20)
    parser.add_argument("--local-trajectory-max-error-px", type=float, default=80.0)
    parser.add_argument("--local-trajectory-min-pair-predictions", type=int, default=4)
    args = parser.parse_args()

    try:
        summary = write_temporal_filtered_ball_track(
            ball_track_path=args.ball_track,
            out_path=args.out,
            summary_path=args.summary_out,
            mode=args.mode,
            max_speed_px_per_second=args.max_speed_px_per_second,
            base_jump_px=args.base_jump_px,
            max_link_gap_frames=args.max_link_gap_frames,
            max_interpolate_gap_frames=args.max_interpolate_gap_frames,
            min_chain_visible_frames=args.min_chain_visible_frames,
            max_neighbor_gap_frames=args.max_neighbor_gap_frames,
            max_iterations=args.max_iterations,
            local_trajectory_window_frames=args.local_trajectory_window_frames,
            local_trajectory_max_error_px=args.local_trajectory_max_error_px,
            local_trajectory_min_pair_predictions=args.local_trajectory_min_pair_predictions,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
