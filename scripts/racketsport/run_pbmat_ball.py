#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.pbmat_adapter import write_ball_track_from_pbmat_predictions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert PB-MAT predictions into ball_track.json.")
    parser.add_argument("--predictions-json", type=Path, required=True, help="PB-MAT prediction artifact JSON.")
    parser.add_argument("--out", type=Path, required=True, help="Output ball_track.json.")
    parser.add_argument("--metadata-out", type=Path, default=None, help="Optional PB-MAT run metadata JSON.")
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    args = parser.parse_args()

    try:
        summary = write_ball_track_from_pbmat_predictions(
            args.predictions_json,
            out=args.out,
            metadata_out=args.metadata_out,
            visibility_threshold=args.visibility_threshold,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
