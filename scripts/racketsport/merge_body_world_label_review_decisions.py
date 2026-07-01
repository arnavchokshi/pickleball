#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.body_world_label_review_corrections import (  # noqa: E402
    merge_body_world_label_review_decisions_into_corrections,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge saved BODY overlay-review decisions into a pending BODY corrections manifest."
    )
    parser.add_argument("--corrections", type=Path, required=True, help="Input BODY corrections manifest JSON.")
    parser.add_argument("--review-input", type=Path, required=True, help="Saved review-input JSON.")
    parser.add_argument("--run-id", required=True, help="BODY run id key inside body_world_label_review.")
    parser.add_argument("--out", type=Path, required=True, help="Output merged corrections manifest JSON.")
    args = parser.parse_args(argv)

    try:
        summary = merge_body_world_label_review_decisions_into_corrections(
            corrections_path=args.corrections,
            review_input_path=args.review_input,
            run_id=args.run_id,
            out_path=args.out,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: BODY review-decision merge failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "written" else 1


if __name__ == "__main__":
    raise SystemExit(main())
