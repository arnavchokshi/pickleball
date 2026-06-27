#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_click_review import export_ball_click_review_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a local 30-frame human ball-click review bundle.")
    parser.add_argument("--video", type=Path, required=True, help="Source video path.")
    parser.add_argument("--out", type=Path, required=True, help="Output review bundle directory.")
    parser.add_argument("--clip", default=None, help="Clip id; defaults to the video stem.")
    parser.add_argument("--sample-count", type=int, default=30, help="Number of frames to export.")
    args = parser.parse_args()

    try:
        summary = export_ball_click_review_bundle(
            video_path=args.video,
            out_dir=args.out,
            clip=args.clip,
            sample_count=args.sample_count,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
