#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.label_review import export_cvat_tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="Export prototype review frames into CVAT-style task folders.")
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = export_cvat_tasks(review_manifest=args.review_manifest, out=args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "ready_for_cvat_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
