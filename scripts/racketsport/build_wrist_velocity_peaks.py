#!/usr/bin/env python3
"""Build review-only wrist_velocity_peaks.json artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.wrist_velocity_peaks import (  # noqa: E402
    build_blocked_wrist_velocity_peaks,
    build_wrist_velocity_peaks_from_file,
    write_wrist_velocity_peaks,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build review-only wrist velocity cue peaks from skeleton3d.json.")
    parser.add_argument("--skeleton3d", type=Path, required=True, help="Input skeleton3d.json path.")
    parser.add_argument("--out", type=Path, required=True, help="Output wrist_velocity_peaks.json path.")
    parser.add_argument("--min-speed-mps", type=float, default=4.0, help="Minimum central-difference wrist speed.")
    parser.add_argument("--min-confidence", type=float, default=0.25, help="Minimum joint confidence.")
    parser.add_argument("--min-separation-s", type=float, default=0.10, help="Minimum spacing between kept peaks.")
    parser.add_argument("--left-wrist-index", type=int, help="Explicit left wrist joint index when joint names are generic.")
    parser.add_argument("--right-wrist-index", type=int, help="Explicit right wrist joint index when joint names are generic.")
    parser.add_argument("--allow-missing", action="store_true", help="Write a blocked artifact if skeleton3d.json is absent.")
    args = parser.parse_args()

    if args.allow_missing and not args.skeleton3d.is_file():
        payload = build_blocked_wrist_velocity_peaks(
            source_path=args.skeleton3d,
            min_speed_mps=args.min_speed_mps,
            min_confidence=args.min_confidence,
            min_separation_s=args.min_separation_s,
        )
    else:
        payload = build_wrist_velocity_peaks_from_file(
            args.skeleton3d,
            min_speed_mps=args.min_speed_mps,
            min_confidence=args.min_confidence,
            min_separation_s=args.min_separation_s,
            left_wrist_index=args.left_wrist_index,
            right_wrist_index=args.right_wrist_index,
        )
    write_wrist_velocity_peaks(args.out, payload)
    print(f"wrote {args.out} ({payload['status']}, peaks={payload['summary']['peak_count']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
