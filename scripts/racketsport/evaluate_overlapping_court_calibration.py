#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.overlapping_court_calibration import (  # noqa: E402
    build_lm_homography_reviewed_label_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate overlapping-court calibration helpers against reviewed court labels."
    )
    parser.add_argument("--eval-root", type=Path, default=Path("eval_clips/ball"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-mean-residual-ft", type=float, default=0.2)
    args = parser.parse_args(argv)

    try:
        report = build_lm_homography_reviewed_label_report(
            eval_root=args.eval_root,
            out_path=args.out,
            target_mean_residual_ft=args.target_mean_residual_ft,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"ERROR: overlapping-court calibration eval failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
