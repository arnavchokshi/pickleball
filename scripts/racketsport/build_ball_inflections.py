#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_inflections import (  # noqa: E402
    DEFAULT_IMAGE_MIN_CANDIDATE_SEPARATION_S,
    DEFAULT_IMAGE_MIN_TURN_DEGREES,
    DEFAULT_MIN_CANDIDATE_SEPARATION_S,
    DEFAULT_MIN_TURN_DEGREES,
    build_ball_inflections_from_ball_track_file,
    build_ball_inflections_from_file,
    write_ball_inflections,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build review-only ball_inflections.json from virtual_world.json or ball_track.json.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--virtual-world", type=Path, help="Input virtual_world.json.")
    source.add_argument("--ball-track", type=Path, help="Input ball_track.json for image-space motion inflections.")
    parser.add_argument("--out", type=Path, required=True, help="Output ball_inflections.json.")
    parser.add_argument(
        "--min-turn-degrees",
        type=float,
        default=None,
        help="Minimum turn angle. Defaults to 45 degrees for virtual-world input and 60 degrees for ball-track input.",
    )
    parser.add_argument("--min-speed-mps", type=float, default=0.75)
    parser.add_argument("--min-speed-px-s", type=float, default=75.0)
    parser.add_argument("--max-neighbor-gap-s", type=float, default=0.2)
    parser.add_argument(
        "--min-candidate-separation-s",
        type=float,
        default=None,
        help="Minimum time between kept cues. Defaults to 0.15s for virtual-world input and 0.05s for ball-track input.",
    )
    args = parser.parse_args()

    try:
        if args.ball_track is not None:
            payload = build_ball_inflections_from_ball_track_file(
                args.ball_track,
                min_turn_degrees=args.min_turn_degrees if args.min_turn_degrees is not None else DEFAULT_IMAGE_MIN_TURN_DEGREES,
                min_speed_px_per_s=args.min_speed_px_s,
                max_neighbor_gap_s=args.max_neighbor_gap_s,
                min_candidate_separation_s=args.min_candidate_separation_s
                if args.min_candidate_separation_s is not None
                else DEFAULT_IMAGE_MIN_CANDIDATE_SEPARATION_S,
            )
        else:
            payload = build_ball_inflections_from_file(
                args.virtual_world,
                min_turn_degrees=args.min_turn_degrees if args.min_turn_degrees is not None else DEFAULT_MIN_TURN_DEGREES,
                min_speed_mps=args.min_speed_mps,
                max_neighbor_gap_s=args.max_neighbor_gap_s,
                min_candidate_separation_s=args.min_candidate_separation_s
                if args.min_candidate_separation_s is not None
                else DEFAULT_MIN_CANDIDATE_SEPARATION_S,
            )
        write_ball_inflections(args.out, payload)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "schema_version": payload["schema_version"],
                "artifact_type": payload["artifact_type"],
                "candidate_count": payload["summary"]["candidate_count"],
                "out": str(args.out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
