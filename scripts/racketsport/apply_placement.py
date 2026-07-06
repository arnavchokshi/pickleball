#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.placement import PlacementConfig, rewrite_tracks_with_placement  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rewrite tracks.json world_xy using foot-keypoint placement fusion.")
    parser.add_argument("--tracks", type=Path, required=True, help="tracks.json to rewrite in place.")
    parser.add_argument("--court-calibration", type=Path, required=True, help="court_calibration.json with homography/intrinsics.")
    parser.add_argument("--placement-out", type=Path, required=True, help="placement.json output path.")
    parser.add_argument("--keypoints-2d", type=Path, default=None, help="Optional native/body keypoints_2d.json.")
    parser.add_argument("--sam3d-keypoints-2d", type=Path, default=None, help="Optional sam3d_keypoints_2d.json sidecar.")
    parser.add_argument("--stance-phases", type=Path, default=None, help="Optional foot_contact_phases.json or foot_pin_audit.json stance phase artifact.")
    parser.add_argument("--refine-from-sam3d", action="store_true", help="Run the post-BODY SAM3D refinement pass.")
    parser.add_argument("--no-placement-undistort", action="store_true", help="Disable pixel undistortion before homography projection.")
    parser.add_argument("--keypoint-conf-min", type=float, default=PlacementConfig().keypoint_conf_min)
    parser.add_argument("--bbox-base-sigma-px", type=float, default=PlacementConfig().bbox_base_sigma_px)
    args = parser.parse_args(argv)

    try:
        result = rewrite_tracks_with_placement(
            tracks_path=args.tracks,
            calibration_path=args.court_calibration,
            placement_path=args.placement_out,
            native2d_keypoints_path=args.keypoints_2d,
            sam3d_keypoints_path=args.sam3d_keypoints_2d,
            stance_phases_path=args.stance_phases,
            refine_from_sam3d=args.refine_from_sam3d,
            config=PlacementConfig(
                keypoint_conf_min=args.keypoint_conf_min,
                bbox_base_sigma_px=args.bbox_base_sigma_px,
                undistort=not args.no_placement_undistort,
            ),
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: apply_placement failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_apply_placement_summary",
                "placement": str(result.placement_path),
                "backup_tracks": str(result.backup_tracks_path),
                "coverage_unchanged": result.coverage_unchanged,
                "source_counts": result.source_counts,
                "court_bounds_violations": result.court_bounds_violations,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
