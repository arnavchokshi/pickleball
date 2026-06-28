#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.mobile_person_yolo_replay import ReplayYoloCandidate, run_replay_yolo_candidate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a YOLO replay candidate into the mobile person tracking schema.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--model", required=True, help="YOLO model path/name, including .pt or .mlpackage.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-players", type=int, choices=(2, 4), default=4)
    parser.add_argument("--tracker", default="predict_iou")
    parser.add_argument("--tracker-config", default=None)
    parser.add_argument("--link-iou-threshold", type=float, default=None)
    parser.add_argument("--max-age-frames", type=int, default=None)
    parser.add_argument("--prune-mode", choices=("confidence", "court"), default="confidence")
    parser.add_argument("--court-calibration", type=Path, default=None)
    parser.add_argument("--court-margin-m", type=float, default=1.25)
    parser.add_argument("--bbox-expand", type=float, default=1.0)
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
