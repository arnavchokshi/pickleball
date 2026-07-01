#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_keypoint_labelspace_diagnostic import (  # noqa: E402
    DEFAULT_EVAL_REPORT,
    DEFAULT_MARKDOWN_OUTPUT,
    DEFAULT_OUTPUT,
    DEFAULT_THRESHOLD,
    build_court_keypoint_labelspace_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a diagnostic-only CAL court-keypoint label-space replay/confusion report.",
    )
    parser.add_argument("--eval-report", type=Path, default=DEFAULT_EVAL_REPORT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--max-frames-per-clip", type=int, default=None)
    args = parser.parse_args(argv)

    report = build_court_keypoint_labelspace_report(
        eval_report_path=args.eval_report,
        out=args.out,
        markdown_out=args.markdown_out,
        threshold=args.threshold,
        max_frames_per_clip=args.max_frames_per_clip,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
