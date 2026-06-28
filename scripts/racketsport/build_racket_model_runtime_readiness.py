#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.racket_model_runtime_readiness import (
    build_racket_model_runtime_readiness,
    write_racket_model_runtime_readiness,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build CPU-only paddle model/runtime readiness report.")
    parser.add_argument("--manifest", type=Path, default=Path("models/MANIFEST.json"))
    parser.add_argument("--out", type=Path, required=True, help="Output racket_model_runtime_readiness.json.")
    parser.add_argument("--check-files", action="store_true", help="Verify declared checkpoint file existence and sha256.")
    parser.add_argument("--fail-on-blocked", action="store_true", help="Exit 2 after writing the report if status is blocked.")
    args = parser.parse_args(argv)

    try:
        payload = build_racket_model_runtime_readiness(args.manifest, check_files=args.check_files)
        write_racket_model_runtime_readiness(args.out, payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: racket model/runtime readiness failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "out": str(args.out),
                "status": payload["status"],
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_blocked and payload["status"] == "blocked":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
