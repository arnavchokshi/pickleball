#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_calibration import (
    calibration_from_manual_tap_frames,
    calibration_from_manual_taps,
    metric_calibration_from_sidecar_and_keypoints,
)
from threed.racketsport.court_templates import Sport
from threed.racketsport.court_zones import build_court_zones
from threed.racketsport.net_plane import build_net_plane


def write_artifact(path: Path, artifact: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(artifact, "model_dump"):
        payload = artifact.model_dump(mode="json")
    else:
        payload = artifact
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build court calibration artifacts from a capture sidecar.")
    parser.add_argument("--sidecar", type=Path, action="append", required=True, help="Path to capture_sidecar.json; repeat for static multi-frame averaging.")
    parser.add_argument("--court-keypoints", type=Path, help="Path to aggregated court_keypoints.json for no-tap ARKit floor-plane metric calibration.")
    parser.add_argument("--sport", choices=["pickleball", "tennis"], default="pickleball")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for court_calibration.json, court_zones.json, and net_plane.json.")
    args = parser.parse_args()

    sport: Sport = args.sport
    if args.court_keypoints is not None:
        if len(args.sidecar) != 1:
            parser.error("--court-keypoints requires exactly one --sidecar")
        calibration = metric_calibration_from_sidecar_and_keypoints(args.sidecar[0], args.court_keypoints, sport=sport)
    else:
        calibration = (
            calibration_from_manual_tap_frames(args.sidecar, sport=sport)
            if len(args.sidecar) > 1
            else calibration_from_manual_taps(args.sidecar[0], sport=sport)
        )

    write_artifact(args.out / "court_calibration.json", calibration)
    write_artifact(args.out / "court_zones.json", build_court_zones(sport))
    write_artifact(args.out / "net_plane.json", build_net_plane(sport))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
