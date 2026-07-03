#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_manual_court_inout import write_manual_court_inout_ball_track


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply manual-corner court-plane in/out fields to BALL bounces.")
    parser.add_argument("--ball-track", type=Path, required=True)
    parser.add_argument("--court-corners", type=Path, required=True)
    parser.add_argument("--sport", choices=("pickleball", "tennis"), default="pickleball")
    parser.add_argument(
        "--target-width",
        type=int,
        required=True,
        help=(
            "Pixel width of the space the ball track's contact_xy_img points live in (almost "
            "always the native source-video width). Required: a 2026-07-02 audit found manual "
            "court corners tapped against downscaled preview frames being silently consumed as "
            "native-resolution coordinates, so this is no longer inferred -- state it explicitly, "
            "e.g. from `ffprobe -select_streams v:0 -show_entries stream=width,height <source.mp4>`."
        ),
    )
    parser.add_argument(
        "--target-height",
        type=int,
        required=True,
        help="Pixel height of the space the ball track's contact_xy_img points live in. See --target-width.",
    )
    parser.add_argument(
        "--uncertainty-m",
        type=float,
        default=None,
        help=(
            "Explicit fixed uncertainty radius override, in meters. When omitted (default), "
            "the per-bounce uncertainty is derived from the clip's own camera geometry "
            "(see threed/racketsport/ball_inout_uncertainty.py)."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()

    try:
        summary = write_manual_court_inout_ball_track(
            ball_track_path=args.ball_track,
            court_corners_path=args.court_corners,
            out=args.out,
            target_image_size=(args.target_width, args.target_height),
            summary_out=args.summary_out,
            sport=args.sport,
            uncertainty_m=args.uncertainty_m,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
