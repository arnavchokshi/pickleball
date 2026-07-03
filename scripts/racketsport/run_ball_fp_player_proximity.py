#!/usr/bin/env python3
"""Bucket ball-detector false positives by distance to the nearest player box.

Diagnostic-only (source-only inputs, no gate). Reads an already-scored ball
track, already-reviewed CVAT ball labels, and an already-computed player
``tracks.json`` per clip; writes a JSON report bucketing every hidden false
positive as on_player / near_player / far_field / no_players_in_frame. See
``threed/racketsport/ball_fp_player_proximity.py`` for the bucket
definitions and the unsafe-flow note on why any resulting player-proximity
signal must stay a soft prior, never a hard veto.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_fp_player_proximity import (  # noqa: E402
    bucket_ball_fps_by_player_proximity_from_files,
    write_ball_fp_player_proximity,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip",
        action="append",
        default=[],
        required=True,
        metavar="CLIP:BALL_TRACK:REVIEWED_BOXES:TRACKS",
        help=(
            "Repeatable. Colon-separated clip id, ball_track.json path, "
            "reviewed_boxes.json path, and tracks.json path for one clip."
        ),
    )
    parser.add_argument(
        "--on-player-diag-fraction",
        type=float,
        default=0.25,
        help="Distance (in nearest player box diagonals) at or below which a FP counts as on_player.",
    )
    parser.add_argument(
        "--near-player-diag-fraction",
        type=float,
        default=1.5,
        help="Distance (in nearest player box diagonals) at or below which a FP counts as near_player.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        clips = [_parse_clip_arg(raw) for raw in args.clip]
        report = bucket_ball_fps_by_player_proximity_from_files(
            clips=clips,
            on_player_diag_fraction=args.on_player_diag_fraction,
            near_player_diag_fraction=args.near_player_diag_fraction,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: ball FP player-proximity bucketing failed: {exc}", file=sys.stderr)
        return 1

    write_ball_fp_player_proximity(args.out, report)
    print(json.dumps(report["combined"], indent=2, sort_keys=True))
    return 0


def _parse_clip_arg(raw: str) -> dict[str, str]:
    parts = raw.split(":")
    if len(parts) != 4:
        raise ValueError(f"--clip must be CLIP:BALL_TRACK:REVIEWED_BOXES:TRACKS, got: {raw!r}")
    clip, ball_track, reviewed_boxes, tracks = parts
    return {"clip": clip, "ball_track": ball_track, "reviewed_boxes": reviewed_boxes, "tracks": tracks}


if __name__ == "__main__":
    raise SystemExit(main())
