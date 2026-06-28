#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.racket_promotion_audit import (
    build_racket_promotion_audit_from_files,
    write_racket_promotion_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit whether preview paddle pose leaked into racket_pose.json.")
    parser.add_argument("--clip", required=True, help="Clip identifier.")
    parser.add_argument("--racket-candidates", type=Path, required=True, help="Input racket_candidates.json.")
    parser.add_argument("--racket-pose-preview", type=Path, help="Optional preview-only racket_pose_preview.json.")
    parser.add_argument("--racket-pose", type=Path, help="Optional canonical racket_pose.json.")
    parser.add_argument("--out", type=Path, required=True, help="Output racket_promotion_audit.json.")
    args = parser.parse_args(argv)

    try:
        payload = build_racket_promotion_audit_from_files(
            clip=args.clip,
            racket_candidates_path=args.racket_candidates,
            racket_pose_preview_path=args.racket_pose_preview,
            racket_pose_path=args.racket_pose,
        )
        write_racket_promotion_audit(args.out, payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: racket promotion audit failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "out": str(args.out),
                "status": payload["status"],
                "trusted_for_rkt_promotion": payload["trusted_for_rkt_promotion"],
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
