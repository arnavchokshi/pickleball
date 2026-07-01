#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.body_world_label_review_bundle import (  # noqa: E402
    build_body_world_label_review_bundle_from_paths,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a selected-sample BODY world-joint review bundle.")
    parser.add_argument("--packet", type=Path, required=True, help="body_world_label_packet.json path.")
    parser.add_argument("--body-frames-dir", type=Path, required=True, help="Directory containing frame_*.jpg images.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output review bundle directory.")
    args = parser.parse_args(argv)

    try:
        manifest = build_body_world_label_review_bundle_from_paths(
            packet_path=args.packet,
            body_frames_dir=args.body_frames_dir,
            out_dir=args.out_dir,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: BODY world-label review bundle failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "ready_for_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
