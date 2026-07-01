#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.body_world_label_packet import (  # noqa: E402
    build_body_world_label_packet_from_paths,
    write_body_world_label_packet,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a prediction packet for reviewed BODY world-joint labels.")
    parser.add_argument("--clip", required=True, help="Clip id.")
    parser.add_argument("--smpl-motion", type=Path, help="smpl_motion.json path.")
    parser.add_argument("--skeleton3d", type=Path, help="skeleton3d.json path.")
    parser.add_argument("--body-compute-execution", type=Path, help="body_compute_execution.json path.")
    parser.add_argument("--source-video", default="", help="Source video path or URI for reviewers.")
    parser.add_argument("--suggested-label-path", default="labels/body_world_joints.json")
    parser.add_argument("--out", type=Path, required=True, help="Output body_world_label_packet.json path.")
    args = parser.parse_args(argv)

    try:
        payload = build_body_world_label_packet_from_paths(
            clip=args.clip,
            smpl_motion_path=args.smpl_motion,
            skeleton3d_path=args.skeleton3d,
            body_compute_execution_path=args.body_compute_execution,
            source_video=args.source_video,
            suggested_label_path=args.suggested_label_path,
        )
        write_body_world_label_packet(args.out, payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: BODY world-label packet failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
