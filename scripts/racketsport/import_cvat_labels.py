#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.label_review import import_corrected_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Import human corrections into prototype draft labels.")
    parser.add_argument("--drafts-root", type=Path, required=True)
    parser.add_argument("--corrections-root", type=Path, required=True)
    args = parser.parse_args()
    summary = import_corrected_labels(drafts_root=args.drafts_root, corrections_root=args.corrections_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
