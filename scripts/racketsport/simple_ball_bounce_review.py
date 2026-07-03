#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_bounce_simple_review_server import run_simple_review_server  # noqa: E402


DEFAULT_CLIPS = [
    "burlington_gold_0300_low_steep_corner",
    "indoor_doubles_fwuks_0500_long_mid_baseline",
    "outdoor_webcam_iynbd_1500_long_high_baseline",
    "wolverine_mixed_0200_mid_steep_corner",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a one-page BALL bounce review UI.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("runs/ball_bounce_inout_review_packets_ground_contact_only_20260701T200001Z"),
    )
    parser.add_argument("--clip", action="append", dest="clips", help="Clip folder to include. Repeatable.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    clips = args.clips or DEFAULT_CLIPS
    url = f"http://{args.host}:{args.port}/"
    if not args.no_open:
        webbrowser.open(url)
    run_simple_review_server(root=args.root, clips=clips, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
