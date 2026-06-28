#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_model_fusion import write_fused_ball_track  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Fuse primary/stable/verifier ball tracks without click labels.")
    parser.add_argument("--primary-ball-track", type=Path, required=True)
    parser.add_argument("--stable-ball-track", type=Path, required=True)
    parser.add_argument("--verifier-ball-track", type=Path, action="append", default=[])
    parser.add_argument("--outlier-distance-px", type=float, default=100.0)
    parser.add_argument("--require-stable-verifier-support", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    try:
        summary = write_fused_ball_track(
            primary_ball_track_path=args.primary_ball_track,
            stable_ball_track_path=args.stable_ball_track,
            verifier_ball_track_paths=args.verifier_ball_track,
            outlier_distance_px=args.outlier_distance_px,
            require_stable_verifier_support=args.require_stable_verifier_support,
            out_path=args.out,
            summary_path=args.summary_out,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
