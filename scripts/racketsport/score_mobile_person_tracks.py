#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.mobile_person_eval import score_mobile_person_tracks, write_mobile_person_metrics  # noqa: E402
from threed.racketsport.schemas import OnDevicePersonTracks, PersonGroundTruth, validate_artifact_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Score iPhone on-device person tracks against person_ground_truth.json.")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Input person_ground_truth.json.")
    parser.add_argument("--predictions", type=Path, required=True, help="Input on_device_person_tracks.json.")
    parser.add_argument("--out", type=Path, required=True, help="Output mobile_person_tracking_metrics.json.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for frame-level matching.")
    parser.add_argument("--expected-players", type=int, default=None, help="Override expected on-court player count.")
    args = parser.parse_args()

    try:
        gt = validate_artifact_file("person_ground_truth", args.ground_truth)
        predictions = validate_artifact_file("on_device_person_tracks", args.predictions)
        if not isinstance(gt, PersonGroundTruth):
            raise ValueError("ground truth did not parse as PersonGroundTruth")
        if not isinstance(predictions, OnDevicePersonTracks):
            raise ValueError("predictions did not parse as OnDevicePersonTracks")
        metrics = score_mobile_person_tracks(
            gt,
            predictions,
            iou_threshold=args.iou_threshold,
            expected_players=args.expected_players,
        )
        write_mobile_person_metrics(args.out, metrics)
    except Exception as exc:
        print(f"mobile person scoring failed: {exc}", file=sys.stderr)
        return 1

    print(
        "mobile person scoring: "
        f"clip_id={metrics.clip_id} "
        f"candidate={metrics.candidate} "
        f"idf1={metrics.idf1:.4f} "
        f"mota={metrics.mota:.4f} "
        f"id_switches={metrics.id_switches}",
        file=sys.stderr,
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
