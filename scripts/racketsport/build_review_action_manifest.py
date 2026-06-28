#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.review_action_manifest import (
    build_review_action_manifest,
    review_action_manifest_html,
    write_review_action_manifest,
    write_review_action_manifest_html,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an actionable human-review checklist from a review packet.")
    parser.add_argument("--packet", type=Path, required=True, help="Input human review packet JSON.")
    parser.add_argument("--out-json", type=Path, required=True, help="Output review_actions.json.")
    parser.add_argument("--out-html", type=Path, required=True, help="Output review_actions.html.")
    args = parser.parse_args(argv)

    try:
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
        manifest = build_review_action_manifest(packet, packet_path=args.packet)
        write_review_action_manifest(args.out_json, manifest)
        html = review_action_manifest_html(manifest, base_dir=args.out_html.parent)
        write_review_action_manifest_html(args.out_html, html)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: review action manifest failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "out_json": str(args.out_json),
                "out_html": str(args.out_html),
                "action_count": manifest["summary"]["action_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
