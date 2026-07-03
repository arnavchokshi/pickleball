#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.virtual_world import (  # noqa: E402
    BALL_WORLD_POLICIES,
    DEFAULT_BALL_WORLD_POLICY,
    build_virtual_world_state_from_files,
    build_virtual_world_state_from_run_dir,
    write_virtual_world,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an inspectable court_Z0 virtual-world artifact.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        help=(
            "Optional run directory. When supplied, the builder auto-consumes best available calibration "
            "(metric15 first), tracks/skeleton/body/ball/racket artifacts, and optional PHYS artifacts."
        ),
    )
    parser.add_argument(
        "--allow-internal-val-run-dir",
        action="store_true",
        help="Allow Burlington/Wolverine internal-val run directories; strict Outdoor/Indoor held-out clips still fail closed.",
    )
    parser.add_argument("--court-calibration", type=Path, help="court_calibration.json artifact.")
    parser.add_argument("--tracks", type=Path, help="Optional tracks.json artifact.")
    parser.add_argument("--smpl-motion", type=Path, help="Optional smpl_motion.json artifact.")
    parser.add_argument("--skeleton3d", type=Path, help="Optional skeleton3d.json artifact.")
    parser.add_argument("--ball-track", type=Path, help="Optional ball_track.json artifact.")
    parser.add_argument(
        "--ball-track-arc-solved",
        type=Path,
        help=(
            "Optional ball_track_arc_solved.json (BALL-ARC-SOLVER output). When supplied it is "
            "authoritative for ball world_xyz: frames it bounded between two confident events are "
            "overlaid with its analytic position, and frames it could not bound are forced hidden."
        ),
    )
    parser.add_argument("--racket-pose", type=Path, help="Optional racket_pose.json artifact.")
    parser.add_argument(
        "--ball-world-policy",
        choices=sorted(BALL_WORLD_POLICIES),
        default=DEFAULT_BALL_WORLD_POLICY,
        help=(
            "Controls whether 2D-only ball detections can be lifted to the court plane. "
            "Default replay/world mode requires trusted world_xyz or arc/physics world output; "
            "court_plane_approx_for_review_only preserves the old approximate review overlay."
        ),
    )
    parser.add_argument("--out", type=Path, required=True, help="Output virtual_world.json path.")
    args = parser.parse_args(argv)

    try:
        if args.run_dir is not None:
            payload = build_virtual_world_state_from_run_dir(
                args.run_dir,
                court_calibration_path=args.court_calibration,
                allow_internal_val=args.allow_internal_val_run_dir,
                ball_world_policy=args.ball_world_policy,
            )
        else:
            if args.court_calibration is None:
                raise ValueError("--court-calibration is required unless --run-dir is supplied")
            run_dir = args.out.parent
            payload = build_virtual_world_state_from_files(
                court_calibration_path=args.court_calibration,
                tracks_path=args.tracks,
                smpl_motion_path=args.smpl_motion,
                skeleton3d_path=args.skeleton3d,
                ball_track_path=args.ball_track,
                racket_pose_path=args.racket_pose,
                physics_footlock_path=_existing(run_dir / "physics_footlock.json"),
                ball_track_physics_filled_path=_existing(run_dir / "ball_track_physics_filled.json"),
                ball_track_arc_solved_path=args.ball_track_arc_solved or _existing(run_dir / "ball_track_arc_solved.json"),
                racket_pose_estimate_path=_existing(run_dir / "racket_pose_estimate.json"),
                ball_world_policy=args.ball_world_policy,
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


def _existing(path: Path) -> Path | None:
    return path if path.is_file() else None


if __name__ == "__main__":
    raise SystemExit(main())
