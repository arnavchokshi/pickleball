#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.owner_capture_intake import (  # noqa: E402
    DEFAULT_OWNER_DATA_MANIFEST,
    prelabel_owner_capture,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create candidate-only prelabel job specs for registered owner captures.")
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OWNER_DATA_MANIFEST)
    parser.add_argument("--owner-data-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate wiring and write job specs without model inference.")
    args = parser.parse_args()

    try:
        result = prelabel_owner_capture(
            args.capture_id,
            manifest_path=args.manifest,
            owner_data_root=args.owner_data_root,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"owner capture prelabel failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
