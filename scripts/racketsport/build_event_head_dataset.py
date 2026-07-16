#!/usr/bin/env python3
"""Build the deterministic event-head public-data manifest (labels only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.event_head.datasets import DatasetFormatError, build_public_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, default=ROOT / "data/event_public_20260713")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    try:
        manifest = build_public_manifest(args.public_root, seed=args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    except (DatasetFormatError, FileNotFoundError) as exc:
        parser.exit(2, f"dataset manifest rejected: {exc}\n")
    print(json.dumps({"out": str(args.out), "totals": manifest["totals"], "verified": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
