#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.cvat_player_review_proxy import (  # noqa: E402
    PROXY_TRACKS_FILENAME,
    build_cvat_player_review_proxy,
    write_cvat_player_review_proxy,
)
from threed.racketsport.schemas import (  # noqa: E402
    CourtCalibration,
    CvatVideoAnnotations,
    PersonGroundTruth,
    validate_artifact_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a review-only, noncanonical CVAT-derived player track proxy."
    )
    parser.add_argument("--person-ground-truth", type=Path, required=True)
    parser.add_argument("--court-calibration", type=Path, required=True)
    parser.add_argument(
        "--reviewed-boxes",
        type=Path,
        default=None,
        help="Optional reviewed_boxes.json; uses task.original_size for CVAT-to-calibration coordinate scaling.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-players", type=int, default=None)
    args = parser.parse_args()

    try:
        ground_truth = validate_artifact_file("person_ground_truth", args.person_ground_truth)
        if not isinstance(ground_truth, PersonGroundTruth):
            raise ValueError(f"{args.person_ground_truth} did not parse as PersonGroundTruth")
        calibration = validate_artifact_file("court_calibration", args.court_calibration)
        if not isinstance(calibration, CourtCalibration):
            raise ValueError(f"{args.court_calibration} did not parse as CourtCalibration")
        source_image_size = None
        if args.reviewed_boxes is not None:
            reviewed = validate_artifact_file("cvat_video_annotations", args.reviewed_boxes)
            if not isinstance(reviewed, CvatVideoAnnotations):
                raise ValueError(f"{args.reviewed_boxes} did not parse as CvatVideoAnnotations")
            source_image_size = tuple(float(value) for value in reviewed.task.original_size)

        output_tracks_path = args.out_dir / PROXY_TRACKS_FILENAME
        result = build_cvat_player_review_proxy(
            ground_truth=ground_truth,
            calibration=calibration,
            source_ground_truth_path=args.person_ground_truth,
            source_calibration_path=args.court_calibration,
            output_tracks_path=output_tracks_path,
            expected_players=args.expected_players,
            source_image_size=source_image_size,
        )
        paths = write_cvat_player_review_proxy(out_dir=args.out_dir, result=result)
    except Exception as exc:
        print(f"CVAT player review proxy failed: {exc}", file=sys.stderr)
        return 1

    print(paths["tracks"])
    print(paths["report"])
    print(paths["markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
