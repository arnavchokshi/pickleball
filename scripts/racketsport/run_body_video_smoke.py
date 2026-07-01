#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.body_video_smoke import run_body_video_smoke  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a fail-closed BODY video-to-world-joints smoke.")
    parser.add_argument("--clip", required=True, help="Clip id.")
    parser.add_argument("--inputs", type=Path, required=True, help="Input bundle with sidecar/detections/frame plan.")
    parser.add_argument("--video", type=Path, required=True, help="Submitted source video.")
    parser.add_argument("--out", type=Path, required=True, help="Output run directory.")
    parser.add_argument("--tracking-mode", choices=["real", "precomputed", "precomputed_tracks"], default="precomputed")
    parser.add_argument("--sport", choices=["pickleball", "tennis"], default="pickleball")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=Path("models/MANIFEST.json"))
    parser.add_argument("--tracker-config", type=Path, default=Path("configs/racketsport/botsort_reid.yaml"))
    parser.add_argument("--max-players", type=int, default=4)
    parser.add_argument("--court-margin-m", type=float, default=0.0)
    parser.add_argument("--id-strategy", choices=["auto", "detection", "track"], default="auto")
    parser.add_argument("--ball-source", type=Path, default=None)
    parser.add_argument("--fast-sam-repo", type=Path, default=None, help="FastSAM-3D-Body repository for BODY runtime.")
    parser.add_argument(
        "--body-detector-name",
        default=None,
        help='Detector backend passed to FastSAM-3D-Body. Pass "" to disable when tracks provide bboxes.',
    )
    parser.add_argument(
        "--body-fov-name",
        default=None,
        help='FOV backend passed to FastSAM-3D-Body. Pass "" to use the runtime default FOV.',
    )
    parser.add_argument("--min-joint-count", type=int, default=17)
    parser.add_argument("--no-overwrite-frames", action="store_true", help="Keep existing materialized BODY frames.")
    parser.add_argument(
        "--full-track-body",
        action="store_true",
        help="Ignore any frame_compute_plan.json and schedule BODY for every tracked player-frame.",
    )
    args = parser.parse_args(argv)

    try:
        report = run_body_video_smoke(
            clip=args.clip,
            inputs_dir=args.inputs,
            video_path=args.video,
            run_dir=args.out,
            tracking_mode=args.tracking_mode,
            sport=args.sport,
            device=args.device,
            max_frames=args.max_frames,
            manifest_path=args.manifest,
            tracker_config_path=args.tracker_config,
            max_players=args.max_players,
            court_margin_m=args.court_margin_m,
            id_strategy=args.id_strategy,
            ball_source_path=args.ball_source,
            fast_sam_repo=args.fast_sam_repo,
            body_detector_name=args.body_detector_name,
            body_fov_name=args.body_fov_name,
            min_joint_count=args.min_joint_count,
            overwrite_frames=not args.no_overwrite_frames,
            full_track_body=args.full_track_body,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: BODY video smoke failed before report write: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
