#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.racket_pose_readiness import (
    build_racket_pose_readiness_from_files,
    write_racket_pose_readiness,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build paddle/racket pose readiness diagnostics.")
    parser.add_argument("--clip", required=True, help="Clip identifier.")
    parser.add_argument("--racket-candidates", type=Path, required=True, help="Input racket_candidates.json.")
    parser.add_argument("--racket-pose-preview", type=Path, help="Optional preview-only racket_pose_preview.json.")
    parser.add_argument("--racket-pose", type=Path, help="Optional promoted racket_pose.json.")
    parser.add_argument("--out", type=Path, required=True, help="Output racket_pose_readiness.json.")
    args = parser.parse_args(argv)

    try:
        payload = build_racket_pose_readiness_from_files(
            clip=args.clip,
            racket_candidates_path=args.racket_candidates,
            racket_pose_preview_path=args.racket_pose_preview,
            racket_pose_path=args.racket_pose,
        )
        write_racket_pose_readiness(args.out, payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: racket-pose readiness failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "out": str(args.out),
                "status": payload["status"],
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
