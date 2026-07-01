#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_2d_post_gate import write_ball_2d_post_gate_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BALL M2 2D post-processing gate artifacts.")
    parser.add_argument("--m1-detector-report", type=Path, default=None)
    parser.add_argument("--postprocess-summary", type=Path, default=None)
    parser.add_argument("--benchmark", action="append", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = write_ball_2d_post_gate_report(
            m1_detector_report_path=args.m1_detector_report,
            postprocess_summary_path=args.postprocess_summary,
            benchmark_paths=args.benchmark,
            out=args.out,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
