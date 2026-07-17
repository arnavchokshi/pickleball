#!/usr/bin/env python3
"""Score BODY placement slide on producer-rebuilt and frozen phase windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.placement_trajectory_refine import (
    PlacementTrajectoryError,
    read_json_object,
    score_placement_slide,
    sha256_file,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skeleton", required=True, type=Path)
    parser.add_argument("--phases", required=True, type=Path, help="Explicit frozen phase-window source.")
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--keypoints-2d", type=Path)
    parser.add_argument("--body-reference-skeleton", type=Path)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        payload = score_placement_slide(
            read_json_object(args.skeleton),
            frozen_phases_payload=read_json_object(args.phases),
            calibration_payload=read_json_object(args.calibration),
            tracks_payload=read_json_object(args.tracks),
            keypoints_2d_payload=read_json_object(args.keypoints_2d) if args.keypoints_2d else None,
            body_reference_payload=(
                read_json_object(args.body_reference_skeleton) if args.body_reference_skeleton else None
            ),
            clip=args.clip,
        )
        payload["inputs"] = {
            "skeleton": {"path": str(args.skeleton.resolve()), "sha256": sha256_file(args.skeleton)},
            "phases": {"path": str(args.phases.resolve()), "sha256": sha256_file(args.phases)},
            "calibration": {"path": str(args.calibration.resolve()), "sha256": sha256_file(args.calibration)},
            "tracks": {"path": str(args.tracks.resolve()), "sha256": sha256_file(args.tracks)},
            "keypoints_2d": (
                {"path": str(args.keypoints_2d.resolve()), "sha256": sha256_file(args.keypoints_2d)}
                if args.keypoints_2d else None
            ),
            "body_reference_skeleton": (
                {
                    "path": str(args.body_reference_skeleton.resolve()),
                    "sha256": sha256_file(args.body_reference_skeleton),
                }
                if args.body_reference_skeleton else None
            ),
        }
    except (OSError, PlacementTrajectoryError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "accepted_phase": payload["accepted_phase"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
