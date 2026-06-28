#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.racket_pose_preview import (  # noqa: E402
    build_racket_pose_preview_from_files,
    write_racket_pose_preview,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a preview-only paddle pose artifact from explicit corners.")
    parser.add_argument("--court-calibration", type=Path, required=True, help="court_calibration.json artifact.")
    parser.add_argument("--racket-candidates", type=Path, required=True, help="racket_candidates.json artifact.")
    parser.add_argument("--out", type=Path, required=True, help="Output racket_pose_preview.json path.")
    parser.add_argument(
        "--max-reprojection-error-px",
        type=float,
        default=6.0,
        help="Drop preview frames above this reprojection error. Default: 6.0.",
    )
    parser.add_argument(
        "--ambiguity-margin-threshold-px",
        type=float,
        default=1.0,
        help="IPPE solution margin below which a frame is marked ambiguous. Default: 1.0.",
    )
    args = parser.parse_args(argv)

    try:
        payload, summary = build_racket_pose_preview_from_files(
            court_calibration_path=args.court_calibration,
            racket_candidates_path=args.racket_candidates,
            max_reprojection_error_px=args.max_reprojection_error_px,
            ambiguity_margin_threshold_px=args.ambiguity_margin_threshold_px,
        )
        write_racket_pose_preview(args.out, payload)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: racket-pose preview build failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"out": str(args.out), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
