#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_on_device_gate import write_ball_on_device_gate_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BALL M7 on-device CoreML/runtime artifacts.")
    parser.add_argument("--offline-ball-track", type=Path, required=True)
    parser.add_argument("--coreml-manifest", type=Path, default=None)
    parser.add_argument("--device-metrics", type=Path, default=None)
    parser.add_argument("--on-device-ball-track", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = write_ball_on_device_gate_report(
            offline_ball_track_path=args.offline_ball_track,
            coreml_manifest_path=args.coreml_manifest,
            device_metrics_path=args.device_metrics,
            on_device_ball_track_path=args.on_device_ball_track,
            out=args.out,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
