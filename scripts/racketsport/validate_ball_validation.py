#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_validation_gate import write_ball_validation_gate_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BALL M8 verifier/full-suite gate artifacts.")
    parser.add_argument(
        "--milestone-report",
        action="append",
        default=[],
        help="Milestone report in KEY=PATH form, for example M4=runs/.../m4_report.json.",
    )
    parser.add_argument("--tracknet-benchmark", type=Path, default=None)
    parser.add_argument("--wasb-track", type=Path, default=None)
    parser.add_argument("--wasb-metadata", type=Path, default=None)
    parser.add_argument("--wasb-benchmark", type=Path, default=None)
    parser.add_argument("--eval-suite", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = write_ball_validation_gate_report(
            milestone_report_paths=_parse_milestone_reports(args.milestone_report),
            tracknet_benchmark_path=args.tracknet_benchmark,
            wasb_track_path=args.wasb_track,
            wasb_metadata_path=args.wasb_metadata,
            wasb_benchmark_path=args.wasb_benchmark,
            eval_suite_path=args.eval_suite,
            out=args.out,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_result"] == "pass" else 1


def _parse_milestone_reports(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--milestone-report must use KEY=PATH form: {value}")
        key, raw_path = value.split("=", 1)
        key = key.strip().upper()
        if not key:
            raise ValueError(f"empty milestone key in --milestone-report {value}")
        parsed[key] = Path(raw_path)
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
