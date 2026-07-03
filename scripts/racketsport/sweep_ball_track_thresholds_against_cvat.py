#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_threshold_sweep import sweep_ball_track_cvat_thresholds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep existing BallTrack confidence thresholds and benchmark against reviewed CVAT ball boxes.",
    )
    parser.add_argument("--candidate-prefix", required=True)
    parser.add_argument("--cvat-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--track",
        action="append",
        default=[],
        help="Clip track spec in the form clip_id=/path/to/ball_track.json. May be repeated.",
    )
    parser.add_argument("--threshold", action="append", type=float, default=[])
    parser.add_argument("--category", default="detector_threshold_sweep")
    parser.add_argument("--hit-radius-px", type=float, default=36.0)
    parser.add_argument("--f1-radius-px", type=float, default=20.0)
    parser.add_argument("--teleport-px-per-frame", type=float, default=160.0)
    parser.add_argument("--max-jump-gap-frames", type=int, default=3)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    try:
        tracks_by_clip = _parse_tracks(args.track)
        summary = sweep_ball_track_cvat_thresholds(
            tracks_by_clip=tracks_by_clip,
            cvat_root=args.cvat_root,
            out_root=args.out_root,
            candidate_name_prefix=args.candidate_prefix,
            thresholds=args.threshold,
            category=args.category,
            hit_radius_px=args.hit_radius_px,
            f1_radius_px=args.f1_radius_px,
            teleport_px_per_frame=args.teleport_px_per_frame,
            max_jump_gap_frames=args.max_jump_gap_frames,
            out_json=args.out_json,
            out_markdown=args.out_md,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "out_root": str(args.out_root),
                "best_candidate": summary["best_candidate"],
                "best_threshold": summary["best_threshold"],
                "aggregate": summary["benchmark"]["aggregate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_tracks(specs: list[str]) -> dict[str, Path]:
    if not specs:
        raise ValueError("at least one --track is required")
    result: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--track must contain '=': {spec}")
        clip, path = spec.split("=", 1)
        if not clip or not path:
            raise ValueError(f"--track missing clip or path: {spec}")
        result[clip] = Path(path)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
