#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.eval.body_gate_report import (  # noqa: E402
    DEFAULT_WORLD_MPJPE_THRESHOLD_M,
    build_body_gate_report,
    write_clip_body_gate_reports,
    write_body_gate_markdown,
    write_body_gate_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a fail-closed BODY accuracy/full-clip gate report.")
    parser.add_argument("--root", type=Path, default=Path("runs/eval0/prototype_gate_h100_v2"))
    parser.add_argument("--labels-root", type=Path, help="Root containing per-clip BODY world-joint labels.")
    parser.add_argument("--clip", action="append", dest="clips", help="Clip id to include. Repeatable.")
    parser.add_argument("--out", type=Path, default=Path("runs/eval0/prototype_gate_h100_v2/body_gate_report.json"))
    parser.add_argument("--markdown-out", type=Path, help="Optional Markdown report path.")
    parser.add_argument("--write-clip-reports", action="store_true", help="Write body_gate_report JSON/Markdown in each clip directory.")
    parser.add_argument("--world-mpjpe-threshold-m", type=float, default=DEFAULT_WORLD_MPJPE_THRESHOLD_M)
    parser.add_argument(
        "--allow-not-verified",
        action="store_true",
        help="Return exit 0 after writing reports even when BODY remains unverified.",
    )
    args = parser.parse_args(argv)

    try:
        payload = build_body_gate_report(
            root=args.root,
            clips=args.clips,
            labels_root=args.labels_root,
            world_mpjpe_threshold_m=args.world_mpjpe_threshold_m,
        )
        write_body_gate_report(args.out, payload)
        if args.markdown_out:
            write_body_gate_markdown(args.markdown_out, payload)
        if args.write_clip_reports:
            write_clip_body_gate_reports(args.root, payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: BODY gate report failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "out": str(args.out),
                "markdown_out": str(args.markdown_out or ""),
                "status": payload["status"],
                "clip_count": payload["summary"]["clip_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if payload["status"] == "pass" or args.allow_not_verified:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
