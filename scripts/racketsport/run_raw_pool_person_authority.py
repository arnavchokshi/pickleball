#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.raw_pool_person_authority import (  # noqa: E402
    RawPoolAuthorityConfig,
    run_raw_pool_authority_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run offline person ReID/global association directly over a raw, pre-role-lock detection pool."
    )
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--raw-pool-dir",
        type=Path,
        required=True,
        help="Directory containing tracked_detections.json/raw_tracked_detections.json and metrics.json for the raw pool.",
    )
    parser.add_argument("--calibration", type=Path, required=True, help="court_calibration.json artifact for this clip.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reid-model", type=Path, required=True)
    parser.add_argument("--embedding-export", type=Path, default=None, help="Reuse an existing source-only ReID embedding export JSON.")
    parser.add_argument("--reid-backend", choices=("osnet", "ultralytics_yolo"), default="osnet")
    parser.add_argument("--reid-model-name", default="osnet_x1_0")
    parser.add_argument("--reid-device", default=None)
    parser.add_argument("--reid-batch-size", type=int, default=32)
    parser.add_argument("--reid-half", action="store_true")
    parser.add_argument("--ground-truth", type=Path, default=None, help="Optional person_ground_truth.json for immediate scoring.")
    parser.add_argument("--expected-players", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--min-conf", type=float, default=0.0, help="Drop raw-pool detections below this confidence before association.")
    parser.add_argument("--max-embedding-bbox-delta-px", type=float, default=2.5)
    parser.add_argument("--split-gap-frames", type=int, default=24)
    parser.add_argument("--max-fragment-speed-m-s", type=float, default=12.0)
    parser.add_argument("--local-switch-split-distance-m", type=float, default=float("inf"))
    parser.add_argument("--local-switch-split-embedding-distance", type=float, default=float("inf"))
    parser.add_argument("--local-switch-split-max-gap-frames", type=int, default=0)
    parser.add_argument("--embedding-split-eps", type=float, default=0.35)
    parser.add_argument("--embedding-split-min-samples", type=int, default=1)
    parser.add_argument("--max-gap-fill-frames", type=int, default=48)
    parser.add_argument("--max-gap-fill-speed-m-s", type=float, default=7.0)
    parser.add_argument("--max-merge-gap-frames", type=int, default=240)
    parser.add_argument("--max-merge-speed-m-s", type=float, default=9.0)
    parser.add_argument("--max-merge-cost", type=float, default=2.0)
    parser.add_argument("--appearance-weight", type=float, default=1.0)
    parser.add_argument("--motion-weight", type=float, default=1.0)
    parser.add_argument("--side-prior-weight", type=float, default=0.25)
    parser.add_argument("--max-fragments-for-global", type=int, default=400)
    parser.add_argument("--enable-cardinality-backfill", action="store_true")
    parser.add_argument("--backfill-max-cost", type=float, default=2.5)
    parser.add_argument("--backfill-iou-threshold", type=float, default=0.25)
    parser.add_argument(
        "--court-margin-m",
        type=float,
        default=2.0,
        help="Apron margin around the court template before outside-court detections are rejected.",
    )
    parser.add_argument("--keep-outside-court", action="store_true", help="Diagnostic only: do not drop world points outside the court template.")
    parser.add_argument(
        "--post-association-court-margin-m",
        type=float,
        default=None,
        help=(
            "Optional second, typically tighter court-polygon margin applied to the "
            "final selected tracks after association (not just candidate detections "
            "before fragment building). Frames outside this margin are dropped from "
            "the output track without changing which fragment/identity was selected. "
            "Use 0.0 to mirror the strict court-only definition used by "
            "off_court_false_positive_frames scoring."
        ),
    )
    args = parser.parse_args()

    try:
        report = run_raw_pool_authority_candidate(
            clip_id=args.clip_id,
            candidate=args.candidate,
            video_path=args.video,
            raw_pool_dir=args.raw_pool_dir,
            calibration_path=args.calibration,
            out_dir=args.out_dir,
            reid_model_path=args.reid_model,
            embedding_export_path=args.embedding_export,
            ground_truth_path=args.ground_truth,
            expected_players=args.expected_players,
            config=RawPoolAuthorityConfig(
                expected_players=args.expected_players,
                reid_backend=args.reid_backend,
                reid_model_name=args.reid_model_name,
                reid_batch_size=args.reid_batch_size,
                reid_device=args.reid_device,
                reid_half=True if args.reid_half else None,
                min_conf=args.min_conf,
                max_embedding_bbox_delta_px=args.max_embedding_bbox_delta_px,
                split_gap_frames=args.split_gap_frames,
                max_fragment_speed_m_s=args.max_fragment_speed_m_s,
                local_switch_split_distance_m=args.local_switch_split_distance_m,
                local_switch_split_embedding_distance=args.local_switch_split_embedding_distance,
                local_switch_split_max_gap_frames=args.local_switch_split_max_gap_frames,
                embedding_split_eps=args.embedding_split_eps,
                embedding_split_min_samples=args.embedding_split_min_samples,
                max_gap_fill_frames=args.max_gap_fill_frames,
                max_gap_fill_speed_m_s=args.max_gap_fill_speed_m_s,
                max_merge_gap_frames=args.max_merge_gap_frames,
                max_merge_speed_m_s=args.max_merge_speed_m_s,
                max_merge_cost=args.max_merge_cost,
                appearance_weight=args.appearance_weight,
                motion_weight=args.motion_weight,
                side_prior_weight=args.side_prior_weight,
                max_fragments_for_global=args.max_fragments_for_global,
                cardinality_backfill=args.enable_cardinality_backfill,
                backfill_max_cost=args.backfill_max_cost,
                backfill_iou_threshold=args.backfill_iou_threshold,
                drop_outside_court=not args.keep_outside_court,
                court_margin_m=args.court_margin_m,
                post_association_court_margin_m=args.post_association_court_margin_m,
            ),
        )
    except Exception as exc:
        print(f"raw pool person authority failed: {exc}", file=sys.stderr)
        return 1

    print(report["summary_path"])
    print(report["tracks_path"])
    print(json.dumps({"status": report["status"], "score_path": report["score_path"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
