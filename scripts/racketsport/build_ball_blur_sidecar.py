#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_blur_sidecar import write_ball_blur_sidecar  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate blur sidecar attributes around ball-track candidates.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--ball-track", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--crop-radius-px", type=int, default=24)
    parser.add_argument("--min-abs-delta", type=int, default=12)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    args = parser.parse_args(argv)

    try:
        payload = write_ball_blur_sidecar(
            video_path=args.video,
            ball_track_path=args.ball_track,
            out_json=args.out_json,
            max_frames=args.max_frames,
            crop_radius_px=args.crop_radius_px,
            min_abs_delta=args.min_abs_delta,
            threshold_percentile=args.threshold_percentile,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
