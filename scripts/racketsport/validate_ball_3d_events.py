#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_3d_events_gate import write_ball_3d_events_gate_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BALL M6 3D/spin/speed/events artifacts.")
    parser.add_argument("--ball-track", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--m4-bounce-report", type=Path, default=None)
    parser.add_argument("--m5-inout-report", type=Path, default=None)
    parser.add_argument("--physics-segments", type=Path, default=None)
    parser.add_argument("--contact-windows", type=Path, default=None)
    parser.add_argument("--reviewed-contacts", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = write_ball_3d_events_gate_report(
            ball_track_path=args.ball_track,
            video_path=args.video,
            m4_bounce_report_path=args.m4_bounce_report,
            m5_inout_report_path=args.m5_inout_report,
            physics_segments_path=args.physics_segments,
            contact_windows_path=args.contact_windows,
            reviewed_contacts_path=args.reviewed_contacts,
            out=args.out,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
