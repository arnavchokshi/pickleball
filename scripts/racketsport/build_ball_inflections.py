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
    build_ball_inflections_from_file,
    write_ball_inflections,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build review-only ball_inflections.json from virtual_world.json.")
    parser.add_argument("--virtual-world", type=Path, required=True, help="Input virtual_world.json.")
    parser.add_argument("--out", type=Path, required=True, help="Output ball_inflections.json.")
    parser.add_argument("--min-turn-degrees", type=float, default=45.0)
    parser.add_argument("--min-speed-mps", type=float, default=0.75)
    parser.add_argument("--max-neighbor-gap-s", type=float, default=0.2)
    parser.add_argument("--min-candidate-separation-s", type=float, default=0.15)
    args = parser.parse_args()

    try:
        payload = build_ball_inflections_from_file(
            args.virtual_world,
            min_turn_degrees=args.min_turn_degrees,
            min_speed_mps=args.min_speed_mps,
            max_neighbor_gap_s=args.max_neighbor_gap_s,
            min_candidate_separation_s=args.min_candidate_separation_s,
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
