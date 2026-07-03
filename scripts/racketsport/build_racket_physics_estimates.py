#!/usr/bin/env python3
"""Build preview-only racket physics estimate artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.racket_physics_estimate import (  # noqa: E402
    build_racket_physics_estimate_from_files,
    render_racket_physics_estimate_overlays,
    write_racket_physics_estimate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build physics-derived, preview-only racket_pose_estimate.json artifacts."
    )
    parser.add_argument("--clip", help="Clip id for an explicit single-clip run.")
    parser.add_argument("--contact-windows", type=Path, help="Explicit contact_windows.json path.")
    parser.add_argument("--ball-track", type=Path, help="Explicit ball_track/filled/virtual_world JSON path.")
    parser.add_argument("--skeleton3d", type=Path, help="Explicit skeleton3d.json path.")
    parser.add_argument("--wrist-peaks", type=Path, help="Optional wrist_velocity_peaks.json path.")
    parser.add_argument("--clip-dir", type=Path, action="append", help="Clip run directory to process.")
    parser.add_argument("--prototype-root", type=Path, help="Root with one subdirectory per clip.")
    parser.add_argument("--clips", nargs="*", help="Clip ids to process under --prototype-root.")
    parser.add_argument(
        "--ball-source",
        choices=("ball_track", "virtual_world"),
        default="ball_track",
        help="For clip-dir/prototype-root runs, choose raw ball_track.json or virtual_world.json ball frames.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output run directory.")
    parser.add_argument("--restitution-range", nargs=2, type=float, default=[0.55, 0.90])
    parser.add_argument("--velocity-window-s", type=float, default=0.14)
    parser.add_argument("--max-sample-gap-s", type=float, default=0.12)
    parser.add_argument("--min-velocity-vectors-per-side", type=int, default=2)
    parser.add_argument("--min-delta-v-mps", type=float, default=0.35)
    parser.add_argument("--ball-position-noise-m", type=float, default=0.05)
    parser.add_argument("--wrist-to-paddle-center-m", type=float, default=0.15)
    parser.add_argument("--max-contact-to-wrist-m", type=float, default=0.75)
    parser.add_argument("--max-wrist-time-gap-s", type=float, default=0.18)
    parser.add_argument("--render-overlays", action="store_true", help="Render qualitative PNG overlays when video/calibration exist.")
    parser.add_argument("--video", type=Path, help="Explicit source video for a single-clip overlay render.")
    parser.add_argument("--court-calibration", type=Path, help="Explicit court_calibration.json for a single-clip overlay render.")
    parser.add_argument("--eval-clips-root", type=Path, default=Path("eval_clips/ball"), help="Fallback source.mp4 root.")
    parser.add_argument("--max-overlays", type=int, default=10)
    args = parser.parse_args()

    tasks = _tasks(args)
    if not tasks:
        parser.error("provide explicit --clip/--contact-windows/--ball-track or --clip-dir/--prototype-root inputs")

    artifacts: list[dict[str, Any]] = []
    for task in tasks:
        clip_out_dir = args.out_dir / task["clip"]
        artifact = build_racket_physics_estimate_from_files(
            clip_id=task["clip"],
            contact_windows_path=task["contact_windows"],
            ball_track_path=task["ball_track"],
            skeleton3d_path=task.get("skeleton3d"),
            wrist_peaks_path=task.get("wrist_peaks"),
            restitution_range=(float(args.restitution_range[0]), float(args.restitution_range[1])),
            velocity_window_s=args.velocity_window_s,
            max_sample_gap_s=args.max_sample_gap_s,
            min_velocity_vectors_per_side=args.min_velocity_vectors_per_side,
            min_delta_v_mps=args.min_delta_v_mps,
            ball_position_noise_m=args.ball_position_noise_m,
            wrist_to_paddle_center_m=args.wrist_to_paddle_center_m,
            max_contact_to_wrist_m=args.max_contact_to_wrist_m,
            max_wrist_time_gap_s=args.max_wrist_time_gap_s,
        )
        estimate_path = clip_out_dir / "racket_pose_estimate.json"
        write_racket_physics_estimate(estimate_path, artifact)
        item: dict[str, Any] = {
            "clip": task["clip"],
            "status": artifact["status"],
            "racket_pose_estimate": str(estimate_path),
            "reviewed_contact_count": artifact["summary"]["reviewed_contact_count"],
            "estimate_count": artifact["summary"]["estimate_count"],
            "skipped_contact_count": artifact["summary"]["skipped_contact_count"],
        }
        if args.render_overlays:
            overlay = _render_overlay_if_possible(
                args=args,
                task=task,
                artifact=artifact,
                clip_out_dir=clip_out_dir,
            )
            item["overlay_index"] = overlay.get("index_path")
            item["overlay_status"] = overlay.get("status")
            item["rendered_overlay_count"] = overlay.get("rendered_overlay_count")
        artifacts.append(item)

    summary = _summary(artifacts)
    payload = {"schema_version": 1, "artifact_type": "racketsport_racket_physics_estimate_run", "summary": summary, "artifacts": artifacts}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "racket_physics_estimate_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


def _tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.contact_windows is not None or args.ball_track is not None:
        if args.clip is None or args.contact_windows is None or args.ball_track is None:
            raise SystemExit("--clip, --contact-windows, and --ball-track are required for explicit single-clip runs")
        return [
            {
                "clip": args.clip,
                "contact_windows": args.contact_windows,
                "ball_track": args.ball_track,
                "skeleton3d": args.skeleton3d,
                "wrist_peaks": args.wrist_peaks,
                "video": args.video,
                "court_calibration": args.court_calibration,
            }
        ]

    clip_dirs = list(args.clip_dir or [])
    if args.prototype_root is not None:
        selected = args.clips or [
            path.name
            for path in sorted(args.prototype_root.iterdir())
            if path.is_dir() and (path / "contact_windows.json").is_file()
        ]
        clip_dirs.extend(args.prototype_root / clip for clip in selected)

    tasks: list[dict[str, Any]] = []
    for clip_dir in clip_dirs:
        clip = clip_dir.name
        ball_path = clip_dir / ("virtual_world.json" if args.ball_source == "virtual_world" else "ball_track.json")
        task = {
            "clip": clip,
            "contact_windows": clip_dir / "contact_windows.json",
            "ball_track": ball_path,
            "skeleton3d": clip_dir / "skeleton3d.json" if (clip_dir / "skeleton3d.json").is_file() else None,
            "wrist_peaks": clip_dir / "wrist_velocity_peaks.json" if (clip_dir / "wrist_velocity_peaks.json").is_file() else None,
            "video": args.eval_clips_root / clip / "source.mp4",
            "court_calibration": clip_dir / "court_calibration.json",
        }
        missing = [name for name in ("contact_windows", "ball_track") if not Path(task[name]).is_file()]
        if missing:
            raise SystemExit(f"{clip}: missing required inputs: {', '.join(missing)}")
        tasks.append(task)
    return tasks


def _render_overlay_if_possible(
    *,
    args: argparse.Namespace,
    task: Mapping[str, Any],
    artifact: Mapping[str, Any],
    clip_out_dir: Path,
) -> dict[str, Any]:
    video = Path(task["video"]) if task.get("video") is not None else None
    calibration = Path(task["court_calibration"]) if task.get("court_calibration") is not None else None
    overlay_dir = clip_out_dir / "overlays"
    if video is None or calibration is None or not video.is_file() or not calibration.is_file():
        overlay_dir.mkdir(parents=True, exist_ok=True)
        index = {
            "schema_version": 1,
            "artifact_type": "racketsport_racket_pose_estimate_overlays",
            "status": "blocked",
            "blockers": ["missing_video_or_court_calibration"],
            "rendered_overlay_count": 0,
        }
        index_path = overlay_dir / "racket_pose_estimate_overlay_index.json"
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"status": "blocked", "rendered_overlay_count": 0, "index_path": str(index_path)}
    calibration_payload = json.loads(calibration.read_text(encoding="utf-8"))
    overlay = render_racket_physics_estimate_overlays(
        video_path=video,
        court_calibration=calibration_payload,
        estimate_artifact=artifact,
        output_dir=overlay_dir,
        max_overlays=args.max_overlays,
    )
    return {
        "status": overlay.get("status"),
        "rendered_overlay_count": overlay.get("rendered_overlay_count", 0),
        "index_path": str(overlay_dir / "racket_pose_estimate_overlay_index.json"),
    }


def _summary(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "clip_count": len(artifacts),
        "total_reviewed_contacts": sum(int(item["reviewed_contact_count"]) for item in artifacts),
        "total_estimates": sum(int(item["estimate_count"]) for item in artifacts),
        "total_skipped_contacts": sum(int(item["skipped_contact_count"]) for item in artifacts),
        "preview_only_not_gate_verified": True,
        "never_canonical_racket_pose": True,
    }


if __name__ == "__main__":
    raise SystemExit(main())
