#!/usr/bin/env python3
"""Propose label-free ball bounce candidates from 2D track geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_bounce_candidates import BounceCandidateConfig, write_bounce_candidate_payload  # noqa: E402


def main() -> int:
    args = _parse_args()
    payload = write_bounce_candidate_payload(
        ball_track_path=args.ball_track,
        calibration_path=args.court_calibration,
        out_path=args.out,
        clip_id=args.clip or "",
        config=BounceCandidateConfig(
            smoothing_window=args.smoothing_window,
            min_run_len=args.min_run_len,
            max_visible_gap_frames=args.max_visible_gap_frames,
            min_candidate_separation_frames=args.min_candidate_separation_frames,
            court_margin_m=args.court_margin_m,
            sharpness_relative_floor=args.sharpness_relative_floor,
            min_gap_hidden_frames=args.min_gap_hidden_frames,
            max_gap_hidden_frames=args.max_gap_hidden_frames,
            gap_window_frames=args.gap_window_frames,
            gap_min_image_speed_px_s=args.gap_min_image_speed_px_s,
            gap_max_ray_rms_m=args.gap_max_ray_rms_m,
            gap_min_vertical_speed_mps=args.gap_min_vertical_speed_mps,
            gap_max_speed_mps=args.gap_max_speed_mps,
        ),
    )
    print(json.dumps({"out": str(args.out), "summary": payload["summary"]}, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", help="Clip id to record in the output payload.")
    parser.add_argument("--ball-track", type=Path, required=True, help="Input ball_track.json path.")
    parser.add_argument("--court-calibration", type=Path, required=True, help="Input court_calibration.json path.")
    parser.add_argument("--out", type=Path, required=True, help="Output auto-bounce candidates JSON path.")
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--min-run-len", type=int, default=7)
    parser.add_argument("--max-visible-gap-frames", type=int, default=3)
    parser.add_argument("--min-candidate-separation-frames", type=int, default=6)
    parser.add_argument("--court-margin-m", type=float, default=2.0)
    parser.add_argument("--sharpness-relative-floor", type=float, default=0.15)
    parser.add_argument("--min-gap-hidden-frames", type=int, default=2)
    parser.add_argument("--max-gap-hidden-frames", type=int, default=24)
    parser.add_argument("--gap-window-frames", type=int, default=4)
    parser.add_argument("--gap-min-image-speed-px-s", type=float, default=40.0)
    parser.add_argument("--gap-max-ray-rms-m", type=float, default=2.0)
    parser.add_argument("--gap-min-vertical-speed-mps", type=float, default=0.5)
    parser.add_argument("--gap-max-speed-mps", type=float, default=35.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
