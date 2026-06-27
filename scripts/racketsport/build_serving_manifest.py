#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.serving_manifest import build_serving_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a CPU-only serving/runtime readiness manifest from models/MANIFEST.json."
    )
    parser.add_argument("--manifest", type=Path, default=Path("models/MANIFEST.json"))
    parser.add_argument("--out", type=Path, help="Write JSON report to this path. Defaults to stdout.")
    args = parser.parse_args()

    report = build_serving_manifest(args.manifest)
    body = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body, encoding="utf-8")
    else:
        print(body, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
