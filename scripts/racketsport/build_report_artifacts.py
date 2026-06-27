#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.report_model import build_report_artifacts, write_report_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build CPU-only RPT-1 habit_report.json and coach_report.json placeholders."
    )
    parser.add_argument("--metrics", type=Path, required=True, help="Path to racket_sport_metrics.json.")
    parser.add_argument("--corrections", type=Path, help="Optional corrections_queue.json with report exclusions.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory where report artifacts are written.")
    args = parser.parse_args(argv)

    try:
        artifacts = build_report_artifacts(args.metrics, corrections_path=args.corrections)
        summary = write_report_artifacts(args.out_dir, artifacts)
    except ValueError as exc:
        print("ERROR: report artifact build failed:", file=sys.stderr)
        for line in str(exc).splitlines():
            print(f"- {line}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
