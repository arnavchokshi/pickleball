#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_local_search import write_local_search_ball_track  # noqa: E402


def main(argv: list[str] | None = None, *, cv2_module: Any | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recover/suppress ball-track samples with CPU local pixel search.")
    parser.add_argument("--video", type=Path, required=True, help="Source video for image evidence.")
    parser.add_argument("--ball-track", type=Path, required=True, help="Input schema-valid ball_track.json.")
    parser.add_argument("--court-calibration", type=Path, default=None, help="Optional court_calibration.json.")
    parser.add_argument("--out", type=Path, required=True, help="Output local-search ball_track.json.")
    parser.add_argument("--summary-out", type=Path, required=True, help="Output local-search summary JSON.")
    parser.add_argument("--search-radius-px", type=int, default=12)
    parser.add_argument("--min-contrast", type=float, default=35.0)
    parser.add_argument("--max-speed-px-per-second", type=float, default=1800.0)
    parser.add_argument("--base-jump-px", type=float, default=20.0)
    parser.add_argument("--max-prediction-gap-frames", type=int, default=6)
    parser.add_argument("--suppress-conf-threshold", type=float, default=0.35)
    parser.add_argument("--court-margin-px", type=float, default=20.0)
    args = parser.parse_args(argv)

    try:
        summary = write_local_search_ball_track(
            video_path=args.video,
            ball_track_path=args.ball_track,
            court_calibration_path=args.court_calibration,
            out_path=args.out,
            summary_path=args.summary_out,
            search_radius_px=args.search_radius_px,
            min_contrast=args.min_contrast,
            max_speed_px_per_second=args.max_speed_px_per_second,
            base_jump_px=args.base_jump_px,
            max_prediction_gap_frames=args.max_prediction_gap_frames,
            suppress_conf_threshold=args.suppress_conf_threshold,
            court_margin_px=args.court_margin_px,
            cv2_module=cv2_module,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
