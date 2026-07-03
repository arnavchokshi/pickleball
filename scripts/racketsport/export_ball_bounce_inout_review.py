#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_bounce_inout_review import export_ball_bounce_inout_review_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export BALL bounce/in-out candidate frames for human review.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--ball-track", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clip", default=None)
    parser.add_argument("--context-frames", type=int, default=2)
    args = parser.parse_args()

    try:
        summary = export_ball_bounce_inout_review_bundle(
            video_path=args.video,
            ball_track_path=args.ball_track,
            out_dir=args.out,
            clip=args.clip,
            context_frames=args.context_frames,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
