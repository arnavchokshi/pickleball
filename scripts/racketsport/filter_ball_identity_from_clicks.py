#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_identity_filter import write_identity_filtered_ball_track  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter a ball track using sparse human ball-click anchors.")
    parser.add_argument("--ball-track", type=Path, required=True, help="Input court-gated ball_track.json.")
    parser.add_argument("--clicks", type=Path, required=True, help="Reviewed ball_points.json.")
    parser.add_argument("--out", type=Path, required=True, help="Output identity-filtered ball_track.json.")
    parser.add_argument("--summary-out", type=Path, required=True, help="Output identity-filter summary JSON.")
    parser.add_argument("--max-identity-error-px", type=float, default=80.0)
    parser.add_argument("--interpolate-max-gap-frames", type=int, default=45)
    args = parser.parse_args()

    try:
        summary = write_identity_filtered_ball_track(
            ball_track_path=args.ball_track,
            clicks_path=args.clicks,
            out_path=args.out,
            summary_path=args.summary_out,
            max_identity_error_px=args.max_identity_error_px,
            interpolate_max_gap_frames=args.interpolate_max_gap_frames,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
