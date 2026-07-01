#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.rkt_shot_replay_lane_audit import (  # noqa: E402
    build_rkt_shot_replay_lane_audit,
    write_rkt_shot_replay_lane_audit,
    write_rkt_shot_replay_lane_audit_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a fail-closed RKT/SHOT/RPL lane audit from local artifacts.")
    parser.add_argument("--cvat-import-root", type=Path, required=True, help="CVAT import root containing manifest.json.")
    parser.add_argument("--replay-readiness", type=Path, required=True, help="Replay readiness JSON report.")
    parser.add_argument("--shot-review-root", type=Path, help="Optional shot classification review root.")
    parser.add_argument(
        "--shot-external-eval",
        action="append",
        type=Path,
        default=[],
        help="Optional external-domain shot eval summary JSON. May be provided more than once.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--md-out", type=Path, help="Optional Markdown output path.")
    args = parser.parse_args(argv)

    try:
        payload = build_rkt_shot_replay_lane_audit(
            cvat_import_root=args.cvat_import_root,
            replay_readiness_path=args.replay_readiness,
            shot_review_root=args.shot_review_root,
            shot_external_eval_paths=args.shot_external_eval,
        )
        write_rkt_shot_replay_lane_audit(args.out, payload)
        if args.md_out is not None:
            write_rkt_shot_replay_lane_audit_markdown(args.md_out, payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: RKT/SHOT/RPL lane audit failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "status": payload["status"],
                "out": str(args.out),
                "md_out": str(args.md_out) if args.md_out else None,
                "next_best_action": payload["next_best_action"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
