#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_finding_technology_benchmark import (  # noqa: E402
    DEFAULT_TECHNOLOGIES,
    build_court_finding_technology_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark court-finding technologies against reviewed court labels without promoting calibration."
    )
    parser.add_argument("--eval-root", type=Path, default=Path("eval_clips/ball"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--technology",
        action="append",
        dest="technologies",
        default=[],
        help="Technology adapter to run. Repeat to run multiple adapters. Defaults to the built-in set.",
    )
    args = parser.parse_args(argv)

    try:
        report = build_court_finding_technology_report(
            eval_root=args.eval_root,
            technologies=args.technologies or list(DEFAULT_TECHNOLOGIES),
            out_dir=args.out_dir,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"ERROR: court-finding technology benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
