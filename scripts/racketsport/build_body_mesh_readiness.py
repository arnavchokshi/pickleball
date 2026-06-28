#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.body_mesh_readiness import (  # noqa: E402
    build_body_mesh_readiness_from_paths,
    write_body_mesh_readiness,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a CPU-only BODY mesh readiness audit.")
    parser.add_argument("--clip", required=True, help="Clip id.")
    parser.add_argument("--smpl-motion", type=Path, help="Optional smpl_motion.json path.")
    parser.add_argument("--skeleton3d", type=Path, help="Optional skeleton3d.json path.")
    parser.add_argument("--frame-compute-plan", type=Path, help="Optional frame_compute_plan.json path.")
    parser.add_argument("--body-compute-execution", type=Path, help="Optional body_compute_execution.json path.")
    parser.add_argument("--out", type=Path, required=True, help="Output body_mesh_readiness.json path.")
    args = parser.parse_args(argv)

    try:
        payload = build_body_mesh_readiness_from_paths(
            clip=args.clip,
            smpl_motion_path=args.smpl_motion,
            skeleton3d_path=args.skeleton3d,
            frame_compute_plan_path=args.frame_compute_plan,
            body_compute_execution_path=args.body_compute_execution,
        )
        write_body_mesh_readiness(args.out, payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: BODY mesh readiness audit failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"schema_version": 1, "out": str(args.out), "status": payload["status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
