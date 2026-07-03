#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_bounce_2d import write_2d_bounce_ball_track


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect BALL bounces from image/court-plane 2D trajectory inflections.")
    parser.add_argument("--ball-track", type=Path, required=True)
    parser.add_argument("--court-corners", type=Path, required=True)
    parser.add_argument("--sport", choices=("pickleball", "tennis"), default="pickleball")
    parser.add_argument(
        "--target-width",
        type=int,
        required=True,
        help=(
            "Pixel width of the space the ball track's xy points live in (almost always the "
            "native source-video width). Required so court corners can be rescaled onto the "
            "ball track's own pixel space instead of being silently assumed to match."
        ),
    )
    parser.add_argument(
        "--target-height",
        type=int,
        required=True,
        help="Pixel height of the space the ball track's xy points live in. See --target-width.",
    )
    parser.add_argument("--min-p-bounce", type=float, default=0.5)
    parser.add_argument("--min-separation-s", type=float, default=0.10)
    parser.add_argument("--min-vertical-delta-px", type=float, default=4.0)
    parser.add_argument("--min-candidate-t-s", type=float, default=0.20)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--detector-out", type=Path, required=True)
    args = parser.parse_args(argv)

    command = " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv])
    try:
        summary = write_2d_bounce_ball_track(
            ball_track_path=args.ball_track,
            court_corners_path=args.court_corners,
            out=args.out,
            detector_out=args.detector_out,
            target_image_size=(args.target_width, args.target_height),
            sport=args.sport,
            min_p_bounce=args.min_p_bounce,
            min_separation_s=args.min_separation_s,
            min_vertical_delta_px=args.min_vertical_delta_px,
            min_candidate_t_s=args.min_candidate_t_s,
            command=command,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
