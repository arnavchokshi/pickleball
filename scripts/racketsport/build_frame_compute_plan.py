#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.frame_rating import (
    DEFAULT_MESH_COVERAGE_MODE,
    DEFAULT_TARGET_MESH_FRAME_BUDGET,
    MESH_COVERAGE_MODES,
    build_frame_compute_plan_from_files,
    write_frame_compute_plan,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a CPU-only adaptive frame compute plan.")
    parser.add_argument("--tracks", type=Path, required=True, help="tracks.json artifact.")
    parser.add_argument("--ball-track", type=Path, help="Optional ball_track.json artifact.")
    parser.add_argument("--contact-windows", type=Path, help="Optional contact_windows.json artifact.")
    parser.add_argument("--expected-players", type=int, default=4, help="Expected on-court player count.")
    parser.add_argument(
        "--mesh-coverage-mode",
        choices=MESH_COVERAGE_MODES,
        default=DEFAULT_MESH_COVERAGE_MODE,
        help="Mesh scheduling policy: contact_only preserves the old scoring behavior; uniform spreads the budget across rally spans; hybrid does both.",
    )
    parser.add_argument(
        "--target-mesh-frame-budget",
        type=int,
        default=DEFAULT_TARGET_MESH_FRAME_BUDGET,
        help="Target deep-mesh frame budget for uniform/hybrid scheduling.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output frame_compute_plan.json path.")
    args = parser.parse_args(argv)

    try:
        plan = build_frame_compute_plan_from_files(
            tracks_path=args.tracks,
            ball_track_path=args.ball_track,
            contact_windows_path=args.contact_windows,
            expected_players=args.expected_players,
            mesh_coverage_mode=args.mesh_coverage_mode,
            target_mesh_frame_budget=args.target_mesh_frame_budget,
        )
        write_frame_compute_plan(args.out, plan)
    except ValueError as exc:
        print(f"ERROR: frame compute plan failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "out": str(args.out),
                "frame_count": plan["frame_count"],
                "mesh_coverage_policy": plan["mesh_coverage_policy"],
                "summary": plan["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
