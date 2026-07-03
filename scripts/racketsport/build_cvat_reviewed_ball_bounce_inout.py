#!/usr/bin/env python3
"""Build reviewed BALL bounce/in-out artifacts from reviewed CVAT boxes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_cvat_bounce_labels import write_cvat_reviewed_bounce_inout_labels  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cvat-labels", required=True, help="Path to reviewed CVAT video boxes JSON.")
    parser.add_argument("--court-corners", required=True, help="Path to manual court corner labels JSON.")
    parser.add_argument("--fps", required=True, type=float, help="Clip frame rate.")
    parser.add_argument("--out-bounces", required=True, help="Output reviewed bounce label JSON.")
    parser.add_argument("--out-inout", required=True, help="Output reviewed in/out label JSON.")
    parser.add_argument("--sport", default="pickleball", choices=("pickleball", "tennis"))
    parser.add_argument("--min-vertical-delta-px", type=float, default=2.0)
    parser.add_argument("--min-separation-s", type=float, default=0.10)
    parser.add_argument("--max-frame-gap", type=int, default=2)
    parser.add_argument("--uncertainty-m", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bounces, inout = write_cvat_reviewed_bounce_inout_labels(
        cvat_labels_path=args.cvat_labels,
        court_corners_path=args.court_corners,
        fps=args.fps,
        out_bounces=args.out_bounces,
        out_inout=args.out_inout,
        sport=args.sport,
        min_vertical_delta_px=args.min_vertical_delta_px,
        min_separation_s=args.min_separation_s,
        max_frame_gap=args.max_frame_gap,
        uncertainty_m=args.uncertainty_m,
    )
    print(
        json.dumps(
            {
                "clip": bounces["clip"],
                "reviewed_bounce_count": len(bounces["bounces"]),
                "reviewed_inout_count": len(inout["calls"]),
                "candidate_count": bounces["candidate_count"],
                "out_bounces": args.out_bounces,
                "out_inout": args.out_inout,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
