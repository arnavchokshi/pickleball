#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_corner_review import build_calibration_from_corrections


def main() -> int:
    parser = argparse.ArgumentParser(description="Build calibration artifacts from reviewed court-corner corrections.")
    parser.add_argument("--drafts-root", type=Path, required=True)
    parser.add_argument("--corrections-root", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--sport", choices=["pickleball", "tennis"], default="pickleball")
    args = parser.parse_args()

    summary = build_calibration_from_corrections(
        drafts_root=args.drafts_root,
        corrections_root=args.corrections_root,
        frames_root=args.frames_root,
        out_root=args.out_root,
        sport=args.sport,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["calibrated_clip_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
