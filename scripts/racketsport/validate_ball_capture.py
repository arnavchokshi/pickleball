#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_capture_protocol import write_ball_capture_protocol_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BALL M0 capture protocol for a real clip.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = write_ball_capture_protocol_report(video_path=args.video, sidecar_path=args.sidecar, out=args.out)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
