#!/usr/bin/env python3
"""Build the permanently-preview one_world_v1 artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.one_world_v1 import build_one_world, canonical_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic preview-only one_world_v1 fusion.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Explicit same-run artifact directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output one_world_v1.json path.")
    args = parser.parse_args()
    try:
        artifact = build_one_world(args.run_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(canonical_json(artifact), encoding="utf-8")
    except Exception as exc:
        print(f"one_world_v1 build failed: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.out} (preview_only, VERIFIED=0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
