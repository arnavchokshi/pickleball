#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.person_tracking_benchmark import (  # noqa: E402
    build_person_tracking_report,
    parse_candidate_spec,
    run_person_tracking_candidate,
    write_person_tracking_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark person detector/tracker variants and render track overlays.")
    parser.add_argument("--clip", required=True, help="Clip identifier used in output paths and reports.")
    parser.add_argument("--video", type=Path, required=True, help="Source video for real tracking.")
    parser.add_argument("--calibration", type=Path, required=True, help="court_calibration.json for the clip.")
    parser.add_argument("--out-root", type=Path, required=True, help="Output root for per-variant runs and report.")
    parser.add_argument("--ball-track", type=Path, help="Optional ball_track.json for candidate-local adaptive frame rating.")
    parser.add_argument(
        "--contact-windows",
        type=Path,
        help="Optional contact_windows.json for candidate-local adaptive BODY scheduling.",
    )
    parser.add_argument(
        "--expected-players",
        type=int,
        default=None,
        help="Expected player count for adaptive frame rating; defaults to --max-players.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate spec name=model,tracker. Example: yolo26n_botsort=yolo26n.pt,configs/racketsport/botsort_reid.yaml",
    )
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0, cuda:0, mps, or cpu.")
    parser.add_argument("--max-players", type=int, choices=(2, 4), default=4)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=32, help="YOLO predict batch size for tiled candidates.")
    parser.add_argument("--half", action="store_true", help="Use FP16 YOLO predict for tiled candidates on CUDA devices.")
    parser.add_argument(
        "--adaptive-min-detections",
        type=int,
        default=None,
        help="Override the adaptive crop fallback trigger count for adaptive crop presets.",
    )
    parser.add_argument(
        "--crop-regions",
        default="default4",
        help=(
            "Tiled crop preset or explicit semicolon-separated x0,y0,x1,y1 regions. "
            "Presets: default4, full_lr3, full_tb3, full, adaptive_full_tb3."
        ),
    )
    parser.add_argument("--max-step-m", type=float, default=2.0)
    parser.add_argument(
        "--court-margin-m",
        type=float,
        default=0.0,
        help="Runoff margin around the regulation court footprint for accepting player footpoints.",
    )
    parser.add_argument(
        "--id-strategy",
        choices=("auto", "raw_track", "role_lock"),
        default="auto",
        help=(
            "auto role-locks prototype player-label detections without tracker IDs and otherwise keeps raw tracker IDs; "
            "raw_track keeps tracker IDs; role_lock assigns stable logical near/far left/right player IDs per frame."
        ),
    )
    args = parser.parse_args()

    try:
        if not args.candidate:
            raise ValueError("at least one --candidate is required")
        candidates = [parse_candidate_spec(spec) for spec in args.candidate]
        rows = []
        for candidate in candidates:
            candidate_dir = args.out_root / args.clip / candidate.name
            rows.append(
                run_person_tracking_candidate(
                    candidate=candidate,
                    clip=args.clip,
                    video_path=args.video,
                    calibration_path=args.calibration,
                    out_dir=candidate_dir,
                    max_players=args.max_players,
                    max_frames=args.max_frames,
                    device=args.device,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    max_step_m=args.max_step_m,
                    court_margin_m=args.court_margin_m,
                    id_strategy=args.id_strategy,
                    batch_size=args.batch_size,
                    half=True if args.half else None,
                    crop_regions=args.crop_regions,
                    adaptive_min_detections=args.adaptive_min_detections,
                    ball_track_path=args.ball_track,
                    contact_windows_path=args.contact_windows,
                    expected_players=args.expected_players,
                )
            )
        summary = build_person_tracking_report(rows, device=args.device, max_frames=args.max_frames)
        write_person_tracking_report(summary, out_dir=args.out_root)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
