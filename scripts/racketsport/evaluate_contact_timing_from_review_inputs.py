#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.eval.contact_timing_review_eval import (  # noqa: E402
    DEFAULT_MAX_MATCH_DELTA_FRAMES,
    evaluate_review_alignment,
    write_review_alignment_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare promoted contact_windows.json files to saved review UI contact timestamps."
    )
    parser.add_argument("--review-input", type=Path, required=True, help="Saved review_input_server.py JSON.")
    parser.add_argument("--run-root", type=Path, required=True, help="Root containing <clip>/contact_windows.json files.")
    parser.add_argument("--clip", dest="clips", action="append", help="Clip id to evaluate. Repeat for multiple clips.")
    parser.add_argument("--fps", type=float, default=60.0, help="Frame rate used for +/- frame timing metrics.")
    parser.add_argument(
        "--max-match-delta-frames",
        type=float,
        default=DEFAULT_MAX_MATCH_DELTA_FRAMES,
        help="Maximum absolute frame delta for one-to-one matching.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output contact timing review-alignment report.")
    args = parser.parse_args(argv)

    try:
        report = evaluate_review_alignment(
            review_input_path=args.review_input,
            run_root=args.run_root,
            clips=args.clips,
            fps=args.fps,
            max_match_delta_frames=args.max_match_delta_frames,
        )
        write_review_alignment_report(args.out, report)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: contact timing review evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
