#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.virtual_world_review import (  # noqa: E402
    DEFAULT_THREE_MODULE_URL,
    build_virtual_world_review_from_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a browser-reviewable Three.js virtual-world page.")
    parser.add_argument("--virtual-world", type=Path, required=True, help="virtual_world*.json artifact.")
    parser.add_argument("--out-html", type=Path, required=True, help="Output HTML review page.")
    parser.add_argument("--index-out", type=Path, required=True, help="Output virtual_world_review_index.json.")
    parser.add_argument("--title", help="Optional page title.")
    parser.add_argument("--clip", help="Optional clip id override for the review packet.")
    parser.add_argument(
        "--three-module-url",
        default=DEFAULT_THREE_MODULE_URL,
        help="Three.js ESM import URL/path relative to the output HTML.",
    )
    args = parser.parse_args(argv)

    try:
        index = build_virtual_world_review_from_file(
            virtual_world_path=args.virtual_world,
            out_html_path=args.out_html,
            index_out_path=args.index_out,
            title=args.title,
            clip=args.clip,
            three_module_url=args.three_module_url,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: virtual-world review build failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
