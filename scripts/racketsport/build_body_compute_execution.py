#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.body_compute import build_body_compute_execution, write_body_compute_execution
from threed.racketsport.schemas import Tracks, validate_artifact_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a BODY compute execution manifest from tracks and an optional frame plan.")
    parser.add_argument("--tracks", type=Path, required=True, help="tracks.json artifact.")
    parser.add_argument("--frame-compute-plan", type=Path, help="Optional frame_compute_plan.json artifact.")
    parser.add_argument("--max-frames", type=int, help="Optional cap on scheduled BODY frames.")
    parser.add_argument("--body-skeleton-stride", type=int, help="Base BODY skeleton cadence in source frames.")
    parser.add_argument("--out", type=Path, required=True, help="Output body_compute_execution.json path.")
    args = parser.parse_args(argv)

    try:
        tracks = validate_artifact_file("tracks", args.tracks)
        if not isinstance(tracks, Tracks):
            raise ValueError("tracks artifact did not parse as Tracks")
        kwargs = {}
        if args.body_skeleton_stride is not None:
            kwargs["skeleton_stride"] = args.body_skeleton_stride
        execution = build_body_compute_execution(
            tracks,
            frame_plan_path=args.frame_compute_plan,
            max_frames=args.max_frames,
            **kwargs,
        )
        write_body_compute_execution(args.out, execution)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: BODY compute execution failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "out": str(args.out),
                "summary": execution["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
