#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_bounce_inout_review import write_reviewed_bounce_inout_labels  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build reviewed BALL bounce and in/out labels from review decisions.")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out-bounces", type=Path, required=True)
    parser.add_argument("--out-inout", type=Path, required=True)
    args = parser.parse_args()

    try:
        bounces, inout = write_reviewed_bounce_inout_labels(
            review_path=args.review,
            out_bounces=args.out_bounces,
            out_inout=args.out_inout,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps({"reviewed_bounces": bounces, "reviewed_inout": inout}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
