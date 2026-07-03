#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.offline_person_authority import OfflineAuthorityConfig, run_offline_authority_candidate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline authoritative person ReID/global association for one tracked candidate.")
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--source-run-dir", type=Path, required=True, help="Directory containing tracks.json.")
    parser.add_argument(
        "--detections-run-dir",
        type=Path,
        default=None,
        help="Optional directory containing raw_tracked_detections.json or tracked_detections.json when tracks are repaired elsewhere.",
    )
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
    parser.add_argument("--embedding-bbox-scale", type=float, default=1.0)
    parser.add_argument("--max-embedding-bbox-delta-px", type=float, default=2.5)
    parser.add_argument("--max-gap-fill-frames", type=int, default=24)
    parser.add_argument("--max-merge-gap-frames", type=int, default=240)
    parser.add_argument("--max-merge-speed-m-s", type=float, default=9.0)
    parser.add_argument("--appearance-weight", type=float, default=1.0)
    parser.add_argument("--motion-weight", type=float, default=1.0)
    parser.add_argument(
        "--court-margin-m",
        type=float,
        default=3.0,
        help="Apron margin around the court template before outside-court detections are rejected.",
    )
    parser.add_argument(
        "--enable-cardinality-backfill",
        action="store_true",
        help="Experimental: fill exact-N frame gaps with unused source detections; source-only, no synthetic boxes.",
    )
    parser.add_argument("--backfill-max-cost", type=float, default=2.5)
    parser.add_argument("--backfill-iou-threshold", type=float, default=0.25)
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
        report = run_offline_authority_candidate(
            clip_id=args.clip_id,
            candidate=args.candidate,
            video_path=args.video,
            source_run_dir=args.source_run_dir,
            detections_run_dir=args.detections_run_dir,
            out_dir=args.out_dir,
            reid_model_path=args.reid_model,
            embedding_export_path=args.embedding_export,
            ground_truth_path=args.ground_truth,
            expected_players=args.expected_players,
            config=OfflineAuthorityConfig(
                expected_players=args.expected_players,
                reid_backend=args.reid_backend,
                reid_model_name=args.reid_model_name,
                reid_batch_size=args.reid_batch_size,
                reid_device=args.reid_device,
                reid_half=True if args.reid_half else None,
                embedding_bbox_scale=args.embedding_bbox_scale,
                max_embedding_bbox_delta_px=args.max_embedding_bbox_delta_px,
                max_gap_fill_frames=args.max_gap_fill_frames,
                max_merge_gap_frames=args.max_merge_gap_frames,
                max_merge_speed_m_s=args.max_merge_speed_m_s,
                appearance_weight=args.appearance_weight,
                motion_weight=args.motion_weight,
                drop_outside_court=not args.keep_outside_court,
                court_margin_m=args.court_margin_m,
                post_association_court_margin_m=args.post_association_court_margin_m,
                cardinality_backfill=args.enable_cardinality_backfill,
                backfill_max_cost=args.backfill_max_cost,
                backfill_iou_threshold=args.backfill_iou_threshold,
            ),
        )
    except Exception as exc:
        print(f"offline person authority failed: {exc}", file=sys.stderr)
        return 1

    print(report["summary_path"])
    print(report["tracks_path"])
    print(json.dumps({"status": report["status"], "score_path": report["score_path"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
