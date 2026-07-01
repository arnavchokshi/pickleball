#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_detector_gate import write_ball_detector_gate_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BALL M1 offline detector gate artifacts.")
    parser.add_argument("--model-manifest", type=Path, default=None)
    parser.add_argument("--ball-track", type=Path, default=None)
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = write_ball_detector_gate_report(
            model_manifest_path=args.model_manifest,
            ball_track_path=args.ball_track,
            benchmark_path=args.benchmark,
            metadata_path=args.metadata,
            out=args.out,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
