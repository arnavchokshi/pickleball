#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.body_world_label_review_overlay import (  # noqa: E402
    build_body_world_label_review_overlays,
    build_body_world_label_review_overlays_from_run,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build BODY world-joint selected-sample review overlays.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Run directory containing BODY review bundle, tracks, and calibration.")
    source.add_argument("--queue", type=Path, help="body_world_label_review_queue.json path.")
    parser.add_argument("--tracks", type=Path, help="tracks.json path when --queue is used.")
    parser.add_argument("--calibration", type=Path, help="court_calibration.json path when --queue is used.")
    parser.add_argument("--out-dir", type=Path, help="Output overlay directory.")
    args = parser.parse_args(argv)

    try:
        if args.run_dir is not None:
            manifest = build_body_world_label_review_overlays_from_run(
                run_dir=args.run_dir,
                out_dir=args.out_dir,
            )
        else:
            if args.tracks is None or args.calibration is None or args.out_dir is None:
                parser.error("--queue requires --tracks, --calibration, and --out-dir")
            manifest = build_body_world_label_review_overlays(
                queue_path=args.queue,
                tracks_path=args.tracks,
                calibration_path=args.calibration,
                out_dir=args.out_dir,
            )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: BODY world-label review overlay failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "ready_for_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
