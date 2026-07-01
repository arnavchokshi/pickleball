#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.body_joint_quality import (  # noqa: E402
    build_body_joint_quality_from_paths,
    write_body_joint_quality,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a CPU-only BODY world-joint quality audit.")
    parser.add_argument("--clip", required=True, help="Clip id.")
    parser.add_argument("--smpl-motion", type=Path, help="Optional smpl_motion.json path.")
    parser.add_argument("--skeleton3d", type=Path, help="Optional skeleton3d.json path.")
    parser.add_argument("--body-compute-execution", type=Path, help="Optional body_compute_execution.json path.")
    parser.add_argument("--body-world-label-packet", type=Path, help="Optional compact body_world_label_packet.json path.")
    parser.add_argument("--out", type=Path, required=True, help="Output body_joint_quality.json path.")
    parser.add_argument("--min-joint-count", type=int, default=17, help="Minimum acceptable world-joint count per frame.")
    parser.add_argument(
        "--floor-z-tolerance-m",
        type=float,
        default=0.15,
        help="Allowed negative world-z tolerance before a joint is treated as below the court floor.",
    )
    parser.add_argument(
        "--warn-track-anchor-residual-m",
        type=float,
        default=1.5,
        help="Warn when smoothed BODY root drifts this far from the tracked court anchor.",
    )
    parser.add_argument(
        "--max-track-anchor-residual-for-review-m",
        type=float,
        default=3.0,
        help="Block review when smoothed BODY root drifts this far from the tracked court anchor.",
    )
    args = parser.parse_args(argv)

    try:
        payload = build_body_joint_quality_from_paths(
            clip=args.clip,
            smpl_motion_path=args.smpl_motion,
            skeleton3d_path=args.skeleton3d,
            body_compute_execution_path=args.body_compute_execution,
            body_world_label_packet_path=args.body_world_label_packet,
            min_joint_count=args.min_joint_count,
            floor_z_tolerance_m=args.floor_z_tolerance_m,
            warn_track_anchor_residual_m=args.warn_track_anchor_residual_m,
            max_track_anchor_residual_for_review_m=args.max_track_anchor_residual_for_review_m,
        )
        write_body_joint_quality(args.out, payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: BODY joint quality audit failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
