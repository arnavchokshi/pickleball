#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_overlay import render_ball_track_overlay  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a CPU OpenCV ball-track overlay video.")
    parser.add_argument("--video", type=Path, required=True, help="Source video path.")
    parser.add_argument("--ball-track", type=Path, required=True, help="Schema-valid ball_track.json path.")
    parser.add_argument("--out", type=Path, required=True, help="Output overlay MP4 path.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of output frames to render.")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth source frame.")
    parser.add_argument("--tail", type=int, default=12, help="Number of recent visible ball samples to draw.")
    parser.add_argument("--fps-out", type=float, default=None, help="Optional output video FPS.")
    args = parser.parse_args()

    try:
        summary = render_ball_track_overlay(
            video_path=args.video,
            ball_track_path=args.ball_track,
            out_path=args.out,
            max_frames=args.max_frames,
            stride=args.stride,
            tail=args.tail,
            fps_out=args.fps_out,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
