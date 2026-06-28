#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.virtual_world import build_virtual_world_state_from_files, write_virtual_world  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an inspectable court_Z0 virtual-world artifact.")
    parser.add_argument("--court-calibration", type=Path, required=True, help="court_calibration.json artifact.")
    parser.add_argument("--tracks", type=Path, help="Optional tracks.json artifact.")
    parser.add_argument("--smpl-motion", type=Path, help="Optional smpl_motion.json artifact.")
    parser.add_argument("--skeleton3d", type=Path, help="Optional skeleton3d.json artifact.")
    parser.add_argument("--ball-track", type=Path, help="Optional ball_track.json artifact.")
    parser.add_argument("--racket-pose", type=Path, help="Optional racket_pose.json artifact.")
    parser.add_argument("--out", type=Path, required=True, help="Output virtual_world.json path.")
    args = parser.parse_args(argv)

    try:
        payload = build_virtual_world_state_from_files(
            court_calibration_path=args.court_calibration,
            tracks_path=args.tracks,
            smpl_motion_path=args.smpl_motion,
            skeleton3d_path=args.skeleton3d,
            ball_track_path=args.ball_track,
            racket_pose_path=args.racket_pose,
        )
        write_virtual_world(args.out, payload)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: virtual-world build failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "out": str(args.out),
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
