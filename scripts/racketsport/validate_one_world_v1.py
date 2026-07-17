#!/usr/bin/env python3
"""Validate one_world_v1 schema, hashes, abstentions, and trust policy."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.one_world_v1 import canonical_json, validate_one_world  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a preview-only one_world_v1 artifact.")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        result = validate_one_world(args.artifact, args.run_dir)
        serialized = canonical_json(result)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0 if result.valid else 1
    except Exception as exc:
        print(f"one_world_v1 validation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
