#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.mobile_person_yolo_replay import (  # noqa: E402
    BEST_PERSON_TRACKING_BBOX_EXPAND,
    BEST_PERSON_TRACKING_CANDIDATE,
    BEST_PERSON_TRACKING_CONF,
    BEST_PERSON_TRACKING_DETECTOR_OUTPUT_LIMIT,
    BEST_PERSON_TRACKING_IMGSZ,
    BEST_PERSON_TRACKING_IOU,
    BEST_PERSON_TRACKING_MODEL,
    BEST_PERSON_TRACKING_TRACKER,
    ReplayYoloCandidate,
    run_replay_yolo_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a YOLO replay candidate into the mobile person tracking schema.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument(
        "--model",
        default=BEST_PERSON_TRACKING_MODEL,
        help="YOLO model path/name, including .pt or .mlpackage.",
    )
    parser.add_argument("--candidate", default=BEST_PERSON_TRACKING_CANDIDATE)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=BEST_PERSON_TRACKING_IMGSZ)
    parser.add_argument("--conf", type=float, default=BEST_PERSON_TRACKING_CONF)
    parser.add_argument("--iou", type=float, default=BEST_PERSON_TRACKING_IOU)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-players", type=int, choices=(2, 4), default=4)
    parser.add_argument("--tracker", default=BEST_PERSON_TRACKING_TRACKER)
    parser.add_argument("--tracker-config", default=None)
    parser.add_argument("--link-iou-threshold", type=float, default=None)
    parser.add_argument("--max-age-frames", type=int, default=None)
    parser.add_argument("--prune-mode", choices=("confidence", "court"), default="confidence")
    parser.add_argument("--court-calibration", type=Path, default=None)
    parser.add_argument("--court-margin-m", type=float, default=1.25)
    parser.add_argument("--bbox-expand", type=float, default=BEST_PERSON_TRACKING_BBOX_EXPAND)
    parser.add_argument("--detector-output-limit", type=int, default=BEST_PERSON_TRACKING_DETECTOR_OUTPUT_LIMIT)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()

    try:
        summary = run_replay_yolo_candidate(
            video_path=args.video,
            ground_truth_path=args.ground_truth,
            candidate=ReplayYoloCandidate(
                name=args.candidate,
                model=args.model,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                max_players=args.max_players,
                tracker=args.tracker,
                tracker_config=args.tracker_config,
                link_iou_threshold=args.link_iou_threshold,
                max_age_frames=args.max_age_frames,
                prune_mode=args.prune_mode,
                court_calibration=str(args.court_calibration) if args.court_calibration is not None else None,
                court_margin_m=args.court_margin_m,
                bbox_expand=args.bbox_expand,
                detector_output_limit=args.detector_output_limit,
            ),
            out_dir=args.out_dir,
            max_frames=args.max_frames,
            render_overlay=not args.no_overlay,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
