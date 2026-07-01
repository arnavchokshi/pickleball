#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.sat_hmr_body_fallback import build_sat_hmr_body_fallback  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert raw SAT-HMR predictions into BODY world artifacts.")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--court-calibration", type=Path, required=True)
    parser.add_argument("--body-compute-execution", type=Path, required=True)
    parser.add_argument("--frame-compute-plan", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--smoothing-alpha", type=float, default=1.0)
    parser.add_argument("--max-root-speed-mps", type=float, default=8.0)
    parser.add_argument("--max-track-anchor-smoothing-residual-m", type=float, default=0.75)
    parser.add_argument("--min-assignment-iou", type=float, default=0.05)
    args = parser.parse_args()

    try:
        report = build_sat_hmr_body_fallback(
            clip=args.clip,
            predictions_dir=args.predictions_dir,
            tracks_path=args.tracks,
            calibration_path=args.court_calibration,
            body_compute_execution_path=args.body_compute_execution,
            frame_compute_plan_path=args.frame_compute_plan,
            out_dir=args.out_dir,
            smoothing_alpha=args.smoothing_alpha,
            max_root_speed_mps=args.max_root_speed_mps,
            max_track_anchor_smoothing_residual_m=args.max_track_anchor_smoothing_residual_m,
            min_assignment_iou=args.min_assignment_iou,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
