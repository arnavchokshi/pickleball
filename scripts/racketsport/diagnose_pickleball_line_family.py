#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.pickleball_line_family import run_pickleball_line_family_diagnostic  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose pickleball line color-family and centerline topology for evidence-only CAL research."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Input video to sample deterministically.")
    source.add_argument("--frame", type=Path, help="Single input frame image.")
    parser.add_argument("--calibration", type=Path, required=True, help="court_calibration.json with court->image homography.")
    parser.add_argument("--keypoints", type=Path, help="Optional reviewed court_keypoints.json for residual diagnostics.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for JSON, Markdown, and overlays.")
    parser.add_argument("--frames", type=int, default=5, help="Number of spread video frames to sample; ignored for --frame.")
    args = parser.parse_args()

    payload = run_pickleball_line_family_diagnostic(
        video=args.video,
        frame=args.frame,
        calibration_path=args.calibration,
        keypoints_path=args.keypoints,
        out_dir=args.out,
        frame_count=args.frames,
    )
    print(args.out / "line_family_diagnostic.json")
    print(f"auto_centerline_evidence_ready={str(payload['auto_centerline_evidence_ready']).lower()}")
    print("verified=false")
    print("not_cal3_verified=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
