#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_auto_evidence import (  # noqa: E402
    build_auto_court_line_evidence_from_frame,
    build_auto_court_line_evidence_from_video,
    write_auto_court_line_evidence,
)
from threed.racketsport.schemas import CourtCalibration, NetPlane, validate_artifact_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build court_line_evidence.json from a frame or video.")
    parser.add_argument("--calibration", type=Path, required=True, help="court_calibration.json")
    parser.add_argument("--net-plane", type=Path, required=True, help="net_plane.json")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frame", type=Path, help="Single calibration frame image.")
    source.add_argument("--video", type=Path, help="Video to sample for court evidence.")
    parser.add_argument("--sample-count", type=int, default=7, help="Number of video frames to sample.")
    parser.add_argument("--out", type=Path, required=True, help="Output court_line_evidence.json")
    args = parser.parse_args()

    try:
        calibration = validate_artifact_file("court_calibration", args.calibration)
        net_plane = validate_artifact_file("net_plane", args.net_plane)
        if not isinstance(calibration, CourtCalibration):
            raise ValueError("calibration artifact did not parse as CourtCalibration")
        if not isinstance(net_plane, NetPlane):
            raise ValueError("net-plane artifact did not parse as NetPlane")
        if args.frame is not None:
            evidence = build_auto_court_line_evidence_from_frame(args.frame, calibration, net_plane=net_plane)
        else:
            evidence = build_auto_court_line_evidence_from_video(
                args.video,
                calibration,
                net_plane=net_plane,
                sample_count=args.sample_count,
            )
        write_auto_court_line_evidence(args.out, evidence)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
