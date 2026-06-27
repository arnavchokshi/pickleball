#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.testclips import build_testclip_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local racket-sport test-clip labels.")
    parser.add_argument("--root", type=Path, default=Path("data/testclips"))
    args = parser.parse_args()

    manifest = build_testclip_manifest(args.root)
    print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if manifest.is_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
