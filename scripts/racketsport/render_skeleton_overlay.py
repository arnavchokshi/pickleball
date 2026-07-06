#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.skeleton_video_overlay import render_skeleton_overlay  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render BODY skeleton inference overlays on video pixels.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing skeleton3d.json and court_calibration.json.")
    parser.add_argument("--video", type=Path, required=True, help="Source video path.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output packet directory.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of video frames to render.")
    parser.add_argument("--contact-sheet-frame-count", type=int, default=24, help="Target number of contact-sheet frames.")
    args = parser.parse_args(argv)

    try:
        summary = render_skeleton_overlay(
            run_dir=args.run_dir,
            video_path=args.video,
            out_dir=args.out_dir,
            max_frames=args.max_frames,
            contact_sheet_frame_count=args.contact_sheet_frame_count,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: skeleton overlay render failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
