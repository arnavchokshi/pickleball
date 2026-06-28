#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.review_packet import build_review_packet, write_review_packet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a human review packet for racket-sport run artifacts.")
    parser.add_argument("--run-root", type=Path, required=True, help="Root to scan for pipeline and review artifacts.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory where review packet files are written.")
    parser.add_argument("--packet-id", help="Stable packet/manifest id. Defaults to run-root name.")
    parser.add_argument(
        "--corrections-root",
        type=Path,
        default=Path("corrections/inbox"),
        help="Directory where the optional corrections template should live.",
    )
    parser.add_argument(
        "--write-corrections-template",
        action="store_true",
        help="Also write an editable corrections manifest template.",
    )
    parser.add_argument(
        "--clip",
        action="append",
        default=[],
        help="Only include this clip id. Repeat for multiple accepted clips.",
    )
    parser.add_argument(
        "--exclude-clip",
        action="append",
        default=[],
        help="Exclude this clip id from the packet. Repeat for multiple rejected clips.",
    )
    args = parser.parse_args(argv)

    packet = build_review_packet(
        args.run_root,
        packet_id=args.packet_id,
        corrections_root=args.corrections_root,
        include_clips=args.clip,
        exclude_clips=args.exclude_clip,
    )
    summary = write_review_packet(
        packet,
        out_dir=args.out_dir,
        write_corrections_template=args.write_corrections_template,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
