#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.io_decode import measure_decode_throughput


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Phase 0 ffmpeg video decode throughput.")
    parser.add_argument("clip", type=Path)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    result = measure_decode_throughput(args.clip, backend=args.backend)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
