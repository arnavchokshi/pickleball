#!/usr/bin/env python3
"""Report frozen baseline-versus-fused one_world_v1 metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.one_world_v1 import build_metrics, canonical_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Score one_world_v1 with the frozen design procedure.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_metrics(args.run_dir, args.fused)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(canonical_json(report), encoding="utf-8")
    except Exception as exc:
        print(f"one_world_v1 metrics failed: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.out} (preview_only, VERIFIED=0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
