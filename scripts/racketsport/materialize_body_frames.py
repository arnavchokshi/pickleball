#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.body_frame_materialization import materialize_body_frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract exact scheduled BODY frames from a source video.")
    parser.add_argument("--video", type=Path, required=True, help="Source video containing the scheduled frames.")
    parser.add_argument(
        "--body-compute-execution",
        type=Path,
        required=True,
        help="body_compute_execution.json with scheduled BODY frames.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for frame_XXXXXX.jpg files.")
    parser.add_argument("--no-overwrite", action="store_true", help="Keep existing frame files.")
    args = parser.parse_args(argv)

    try:
        summary = materialize_body_frames(
            video_path=args.video,
            execution_path=args.body_compute_execution,
            out_dir=args.out_dir,
            overwrite=not args.no_overwrite,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: BODY frame materialization failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
