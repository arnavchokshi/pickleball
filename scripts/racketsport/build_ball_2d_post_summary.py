#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_2d_post_summary import write_ball_2d_postprocess_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a truthful BALL M2 2D postprocess summary artifact.")
    parser.add_argument("--model-consensus-summary", action="append", type=Path, default=[])
    parser.add_argument("--court-gating-summary", type=Path, default=None)
    parser.add_argument("--max-speed-summary", type=Path, default=None)
    parser.add_argument("--ransac-summary", type=Path, default=None)
    parser.add_argument("--local-search-summary", type=Path, default=None)
    parser.add_argument("--kalman-rts-summary", type=Path, default=None)
    parser.add_argument("--primary-model", default=None)
    parser.add_argument("--verifier-model", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        summary = write_ball_2d_postprocess_summary(
            out=args.out,
            model_consensus_summary_paths=args.model_consensus_summary,
            court_gating_summary_path=args.court_gating_summary,
            max_speed_summary_path=args.max_speed_summary,
            ransac_summary_path=args.ransac_summary,
            local_search_summary_path=args.local_search_summary,
            kalman_rts_summary_path=args.kalman_rts_summary,
            primary_model=args.primary_model,
            verifier_model=args.verifier_model,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
