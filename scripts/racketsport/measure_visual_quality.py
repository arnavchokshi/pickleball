#!/usr/bin/env python3
"""Measure visual smoothness for a completed racket-sport clip directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.visual_quality import write_visual_quality  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Completed clip directory containing replay artifacts.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for visual_quality.json and visual_quality.md.")
    parser.add_argument("--json", action="store_true", help="Print the metrics JSON after writing outputs.")
    args = parser.parse_args()

    json_path, md_path, metrics = write_visual_quality(args.run_dir, out_dir=args.out_dir)
    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        print(f"wrote {json_path}")
        print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
