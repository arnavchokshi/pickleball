#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.physics_world_refinement import (  # noqa: E402
    build_physics_refinement_from_file,
    write_physics_refinement,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a CPU physics-refinement artifact from virtual_world.json.")
    parser.add_argument("--clip", required=True, help="Clip id.")
    parser.add_argument("--virtual-world", type=Path, required=True, help="virtual_world.json path.")
    parser.add_argument("--requested-mode", default="auto", help="Physics mode request for the existing scaffold.")
    parser.add_argument("--mjx-available", action="store_true", help="Record MJX availability in the execution plan.")
    parser.add_argument("--pad-frames", type=int, default=3, help="Contact-window padding.")
    parser.add_argument("--out", type=Path, required=True, help="Output physics_refinement.json path.")
    args = parser.parse_args(argv)

    try:
        artifact = build_physics_refinement_from_file(
            clip_id=args.clip,
            virtual_world_path=args.virtual_world,
            requested_mode=args.requested_mode,
            mjx_available=args.mjx_available,
            pad_frames=args.pad_frames,
        )
        write_physics_refinement(args.out, artifact)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: physics refinement build failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "out": str(args.out),
                "clip": args.clip,
                "physics": artifact["physics"],
                "foot2_done": artifact["foot2_done"],
                "constraint_summary": artifact["constraint_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
