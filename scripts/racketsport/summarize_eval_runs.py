#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.eval.summary import build_eval_run_summary, write_eval_run_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize CPU-only racket-sport eval metrics artifacts.")
    parser.add_argument(
        "--phase-dir",
        type=Path,
        action="append",
        dest="phase_dirs",
        required=True,
        help="Phase directory containing a metrics.json file. Repeat for multiple phases.",
    )
    parser.add_argument("--out", type=Path, help="Optional path to write the summary JSON.")
    args = parser.parse_args()

    try:
        payload = build_eval_run_summary(args.phase_dirs)
    except ValueError as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    if args.out is not None:
        write_eval_run_summary(args.out, payload)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
