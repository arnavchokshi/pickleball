#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_detector_error_mining import (  # noqa: E402
    DetectorTrackInput,
    mine_ball_detector_errors,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mine TrackNet/WASB detector errors against reviewed CVAT labels for BALL fine-tuning.",
    )
    parser.add_argument("--cvat-root", type=Path, required=True, help="Root containing <clip>/reviewed_boxes.json files.")
    parser.add_argument(
        "--track",
        action="append",
        default=[],
        help="Track spec clip:split:candidate=/path/to/ball_track.json. Split is train, val, test, or eval.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--radius-px", type=float, default=20.0)
    args = parser.parse_args(argv)

    try:
        tracks = [_parse_track_spec(item) for item in args.track]
        plan = mine_ball_detector_errors(
            cvat_root=args.cvat_root,
            tracks=tracks,
            out_json=args.out_json,
            radius_px=args.radius_px,
        )
    except Exception as exc:
        print(f"BALL detector error mining failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": plan["status"],
                "out_json": str(args.out_json),
                "summary": plan["summary"],
                "train_clips": plan["train_clips"],
                "validation_clips": plan["validation_clips"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_track_spec(spec: str) -> DetectorTrackInput:
    if "=" not in spec:
        raise ValueError(f"--track must contain '=': {spec}")
    left, path = spec.split("=", 1)
    parts = left.split(":")
    if len(parts) != 3:
        raise ValueError(f"--track must be clip:split:candidate=/path/to/ball_track.json: {spec}")
    clip, split, candidate = parts
    if not clip or not split or not candidate or not path:
        raise ValueError(f"--track requires non-empty clip, split, candidate, and path: {spec}")
    return DetectorTrackInput(clip=clip, split=split, candidate=candidate, path=Path(path))


if __name__ == "__main__":
    raise SystemExit(main())
