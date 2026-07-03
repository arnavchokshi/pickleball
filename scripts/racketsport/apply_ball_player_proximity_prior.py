#!/usr/bin/env python3
"""Apply the soft player-proximity confidence prior to one ball_track artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_player_proximity_prior import (  # noqa: E402
    BallPlayerProximityPriorConfig,
    apply_ball_player_proximity_prior_from_files,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ball-track", type=Path, required=True, help="Input ball_track.json artifact.")
    parser.add_argument("--tracks", type=Path, required=True, help="Input player tracks.json artifact.")
    parser.add_argument("--out-ball-track", type=Path, required=True, help="Output adjusted ball_track.json.")
    parser.add_argument("--out-report", type=Path, required=True, help="Output proximity-prior provenance report JSON.")
    parser.add_argument(
        "--strength",
        type=float,
        default=BallPlayerProximityPriorConfig().strength,
        help="Maximum fractional confidence reduction at/on player boxes. Must be in [0, 1).",
    )
    parser.add_argument(
        "--influence-diag-fraction",
        type=float,
        default=BallPlayerProximityPriorConfig().influence_diag_fraction,
        help="Prior taper radius in units of nearest player-box diagonal.",
    )
    args = parser.parse_args(argv)

    try:
        config = BallPlayerProximityPriorConfig(
            strength=args.strength,
            influence_diag_fraction=args.influence_diag_fraction,
        )
        report = apply_ball_player_proximity_prior_from_files(
            ball_track_path=args.ball_track,
            tracks_path=args.tracks,
            out_ball_track_path=args.out_ball_track,
            out_report_path=args.out_report,
            config=config,
        )
    except Exception as exc:
        print(f"ERROR: failed to apply ball player-proximity prior: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "out_ball_track": str(args.out_ball_track),
                "out_report": str(args.out_report),
                "adjusted_frame_count": report["adjusted_frame_count"],
                "min_factor": report["min_factor"],
                "additive_safe": report["additive_safe"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
