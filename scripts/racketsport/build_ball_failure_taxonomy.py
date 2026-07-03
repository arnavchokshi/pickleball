#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_failure_taxonomy import write_ball_failure_taxonomy  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a frame-level BALL failure taxonomy from CVAT labels.")
    parser.add_argument("--ball-track", type=Path, required=True, help="Candidate ball_track.json.")
    parser.add_argument("--cvat-labels", type=Path, required=True, help="Reviewed CVAT reviewed_boxes.json.")
    parser.add_argument("--candidate", required=True, help="Candidate name for report provenance.")
    parser.add_argument("--out-json", type=Path, required=True, help="Output taxonomy JSON path.")
    parser.add_argument("--out-md", type=Path, default=None, help="Optional Markdown summary path.")
    parser.add_argument("--f1-radius-px", type=float, default=20.0)
    parser.add_argument("--teleport-px-per-frame", type=float, default=160.0)
    parser.add_argument("--max-jump-gap-frames", type=int, default=3)
    args = parser.parse_args(argv)

    try:
        taxonomy = write_ball_failure_taxonomy(
            ball_track_path=args.ball_track,
            cvat_labels_path=args.cvat_labels,
            candidate_name=args.candidate,
            out_json=args.out_json,
            out_markdown=args.out_md,
            f1_radius_px=args.f1_radius_px,
            teleport_px_per_frame=args.teleport_px_per_frame,
            max_jump_gap_frames=args.max_jump_gap_frames,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(taxonomy["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
