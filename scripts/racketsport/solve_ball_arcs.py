#!/usr/bin/env python3
"""Solve event-anchored render-only 3D ball arcs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_arc_solver import (  # noqa: E402
    ARTIFACT_TYPE,
    BallArcSolverConfig,
    PhysicsParameters,
    _project_world_point,
    solve_ball_arc_track,
)
from threed.racketsport.schemas import load_ball_candidates_file  # noqa: E402


def main() -> int:
    args = _parse_args()
    if args.measure_size_video is not None:
        if args.ball_track is None or args.ball_sizes_out is None:
            raise SystemExit("--measure-size-video requires --ball-track and --ball-sizes-out")
        ball_track = _read_json_object(args.ball_track, "ball_track")
        payload = _measure_ball_sizes_from_video(
            args.measure_size_video,
            ball_track,
            crop_radius_px=args.size_crop_radius_px,
        )
        _write_json(args.ball_sizes_out, payload)
        print(json.dumps({"ball_sizes": str(args.ball_sizes_out), "summary": payload["summary"]}, sort_keys=True))
        return 0
    if args.out_dir is None:
        raise SystemExit("--out-dir is required for solve mode")
    tasks = _tasks(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    generated_at = _utc_stamp()

    for task in tasks:
        clip = str(task["clip"])
        clip_out_dir = args.out_dir / clip
        clip_out_dir.mkdir(parents=True, exist_ok=True)
        ball_track = _read_json_object(task["ball_track"], "ball_track")
        calibration = _read_json_object(task["court_calibration"], "court_calibration")
        ball_sizes = _read_optional_json(task.get("ball_sizes"))
        ball_candidate_sidecars = [_read_ball_candidates(path) for path in task.get("ball_candidates", [])]
        candidate_extra_tracks = {
            name: _read_json_object(path, f"candidate_extra_track:{name}")
            for name, path in task.get("candidate_extra_tracks", {}).items()
        }
        contact_windows = _read_optional_json(task.get("contact_windows"))
        skeleton3d = _read_optional_json(task.get("skeleton3d"))
        frame_times = _read_optional_json(task.get("frame_times"))
        reviewed_bounces = _read_optional_json(task.get("reviewed_bounces"))
        auto_bounce_candidates = _read_optional_json(task.get("auto_bounce_candidates"))
        rally_spans = _read_optional_json(task.get("rally_spans"))
        net_plane = _read_optional_json(task.get("net_plane"))
        physics = PhysicsParameters.for_ball_type(args.ball_type)
        config = BallArcSolverConfig(
            robust_pixel_sigma=args.robust_pixel_sigma,
            max_reprojection_inlier_px=args.max_reprojection_inlier_px,
            contact_anchor_sigma_m=args.contact_anchor_sigma_m,
            contact_reach_offset_m=args.contact_reach_offset_m,
            reviewed_bounce_base_sigma_m=args.reviewed_bounce_base_sigma_m,
            proposed_bounce_sigma_m=args.proposed_bounce_sigma_m,
            enable_event_subset_selection=not args.no_event_subset_selection,
            selection_max_speed_mps=args.selection_max_speed_mps,
            selection_min_residual_reduction=args.selection_min_residual_reduction,
            selection_split_penalty=args.selection_split_penalty,
            selection_max_nfev=args.selection_max_nfev,
            loo_max_nfev=args.loo_max_nfev,
            rally_endpoint_sigma_m=args.rally_endpoint_sigma_m,
            enable_event_discovery=not args.no_event_discovery,
            discovery_reprojection_px=args.discovery_reprojection_px,
            baseline_loo_median_m=args.baseline_loo_median_m,
            max_physical_violation_fraction=args.max_physical_violation_fraction,
            enable_size_depth_residual=not args.no_size_depth_residual,
            size_depth_sigma_m=args.size_depth_sigma_m,
            weak_size_depth_sigma_m=args.weak_size_depth_sigma_m,
            enable_weak_segments=not args.no_weak_segments,
            weak_segment_min_observations=args.weak_segment_min_observations,
            candidate_selection_max_iterations=args.candidate_selection_max_iterations,
            max_candidates_per_frame=args.max_candidates_per_frame,
            candidate_association_mode=args.candidate_association_mode,
            candidate_score_floors=_candidate_score_floor_specs(args.candidate_score_floor),
        )
        artifact = solve_ball_arc_track(
            ball_track=ball_track,
            calibration=calibration,
            ball_sizes=ball_sizes,
            ball_candidate_sidecars=ball_candidate_sidecars,
            candidate_extra_tracks=candidate_extra_tracks,
            contact_windows=contact_windows,
            skeleton3d=skeleton3d,
            reviewed_bounces=reviewed_bounces,
            auto_bounce_candidates=auto_bounce_candidates,
            rally_spans=rally_spans,
            net_plane=net_plane,
            frame_times=frame_times,
            physics=physics,
            config=config,
            clip_id=clip,
        )
        artifact["generated_at"] = generated_at
        artifact["inputs"] = {
            "ball_track": str(task["ball_track"]),
            "court_calibration": str(task["court_calibration"]),
            "ball_sizes": str(task.get("ball_sizes") or ""),
            "ball_candidates": [str(path) for path in task.get("ball_candidates", [])],
            "candidate_extra_tracks": {name: str(path) for name, path in task.get("candidate_extra_tracks", {}).items()},
            "contact_windows": str(task.get("contact_windows") or ""),
            "skeleton3d": str(task.get("skeleton3d") or ""),
            "frame_times": str(task.get("frame_times") or ""),
            "reviewed_bounces": str(task.get("reviewed_bounces") or ""),
            "auto_bounce_candidates": str(task.get("auto_bounce_candidates") or ""),
            "rally_spans": str(task.get("rally_spans") or ""),
            "net_plane": str(task.get("net_plane") or ""),
        }

        artifact_path = clip_out_dir / "ball_track_arc_solved.json"
        events_selected_path = clip_out_dir / "events_selected.json"
        report_json_path = clip_out_dir / "ball_arc_solver_report.json"
        report_md_path = clip_out_dir / "REPORT.md"
        commands_path = clip_out_dir / "COMMANDS.sh"
        _write_json(artifact_path, artifact)
        _write_json(events_selected_path, artifact["event_selection"])
        _write_json(report_json_path, _report_payload(clip, artifact, artifact_path, events_selected_path))
        _write_text(report_md_path, _markdown_report(clip, artifact, artifact_path, report_json_path))
        _write_text(commands_path, _commands(args, task, clip_out_dir))
        commands_path.chmod(0o755)
        artifacts.append(
            {
                "clip": clip,
                "status": artifact["status"],
                "ball_track_arc_solved": str(artifact_path),
                "events_selected": str(events_selected_path),
                "report": str(report_json_path),
                "report_md": str(report_md_path),
                "coverage_world_xyz_count": artifact["summary"]["coverage_world_xyz_count"],
                "segment_count": artifact["summary"]["segment_count"],
                "fp_sightings_pruned_count": artifact["summary"]["fp_sightings_pruned_count"],
                "auto_bounce_candidate_count": artifact["summary"].get("auto_bounce_candidate_count", 0),
                "candidate_selection_source_counts": artifact["summary"].get("candidate_selection_source_counts", {}),
                "discovered_bounce_count": artifact["summary"]["discovered_bounce_count"],
            }
        )

    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_arc_solver_run",
        "generated_at": generated_at,
        "summary": {
            "clip_count": len(artifacts),
            "experimental_off_count": sum(1 for item in artifacts if item["status"] == "experimental_off"),
            "total_coverage_world_xyz_count": sum(int(item["coverage_world_xyz_count"]) for item in artifacts),
            "total_fp_sightings_pruned_count": sum(int(item["fp_sightings_pruned_count"]) for item in artifacts),
            "render_only": True,
            "not_for_detection_metrics": True,
        },
        "artifacts": artifacts,
    }
    _write_json(args.out_dir / "ball_arc_solver_run_summary.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", help="Clip id for explicit single-clip run.")
    parser.add_argument("--ball-track", type=Path, help="Explicit ball_track.json path.")
    parser.add_argument("--court-calibration", type=Path, help="Explicit court_calibration.json path.")
    parser.add_argument("--ball-sizes", type=Path, help="Optional ball_size_observations.json path.")
    parser.add_argument("--ball-candidates", type=Path, action="append", default=[], help="Optional racketsport_ball_candidates sidecar. Repeatable.")
    parser.add_argument(
        "--candidate-extra-track",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Inject visible samples from another ball_track.json as candidates, e.g. blurball=path/to/ball_track.json.",
    )
    parser.add_argument("--max-candidates-per-frame", type=int, default=12, help="Cap combined primary/extra/sidecar candidates per frame.")
    parser.add_argument("--candidate-selection-max-iterations", type=int, default=5, help="Maximum arc_irls_v1 candidate association iterations per segment.")
    parser.add_argument(
        "--candidate-association-mode",
        choices=("free", "rescue_only"),
        default="free",
        help="Candidate association policy. Default preserves W3 free association; rescue_only preserves primary inliers.",
    )
    parser.add_argument(
        "--candidate-score-floor",
        action="append",
        default=[],
        metavar="SOURCE=FLOAT",
        help="Minimum score required for candidates from SOURCE. Repeatable; SOURCE can be a prefix like tracknet or wasb.",
    )
    parser.add_argument("--measure-size-video", type=Path, help="Helper mode: measure apparent ball sizes from a video and ball_track.json.")
    parser.add_argument("--ball-sizes-out", type=Path, help="Output path for --measure-size-video helper mode.")
    parser.add_argument("--size-crop-radius-px", type=int, default=24, help="Crop radius around each ball xy for size measurement helper.")
    parser.add_argument("--contact-windows", type=Path, help="Optional contact_windows.json path.")
    parser.add_argument("--skeleton3d", type=Path, help="Optional skeleton3d.json path.")
    parser.add_argument("--frame-times", type=Path, help="Optional frame_times.json for VFR-correct fallback timestamps.")
    parser.add_argument("--reviewed-bounces", type=Path, help="Optional reviewed_ball_bounces.json path.")
    parser.add_argument("--auto-bounce-candidates", type=Path, help="Optional label-free auto-bounce candidates JSON path.")
    parser.add_argument("--rally-spans", type=Path, help="Optional rally_spans.json path.")
    parser.add_argument("--net-plane", type=Path, help="Optional net_plane.json path.")
    parser.add_argument("--clip-dir", type=Path, action="append", help="Run directory with standard artifact names.")
    parser.add_argument("--prototype-root", type=Path, help="Root containing one run directory per clip.")
    parser.add_argument("--clips", nargs="*", help="Clip ids under --prototype-root.")
    parser.add_argument("--out-dir", type=Path, help="Output run directory.")
    parser.add_argument("--ball-type", choices=("outdoor", "indoor", "no_drag_test"), default="outdoor")
    parser.add_argument("--robust-pixel-sigma", type=float, default=6.0)
    parser.add_argument("--max-reprojection-inlier-px", type=float, default=18.0)
    parser.add_argument("--contact-anchor-sigma-m", type=float, default=0.35)
    parser.add_argument("--contact-reach-offset-m", type=float, default=0.15)
    parser.add_argument("--reviewed-bounce-base-sigma-m", type=float, default=0.05)
    parser.add_argument("--proposed-bounce-sigma-m", type=float, default=0.18)
    parser.add_argument("--no-event-subset-selection", action="store_true")
    parser.add_argument("--selection-max-speed-mps", type=float, default=35.0)
    parser.add_argument("--selection-min-residual-reduction", type=float, default=0.05)
    parser.add_argument("--selection-split-penalty", type=float, default=0.25)
    parser.add_argument("--selection-max-nfev", type=int, default=350)
    parser.add_argument("--loo-max-nfev", type=int, default=600)
    parser.add_argument("--rally-endpoint-sigma-m", type=float, default=2.0)
    parser.add_argument("--no-event-discovery", action="store_true")
    parser.add_argument("--discovery-reprojection-px", type=float, default=60.0)
    parser.add_argument("--baseline-loo-median-m", type=float, default=0.1012)
    parser.add_argument("--max-physical-violation-fraction", type=float, default=0.20)
    parser.add_argument("--no-size-depth-residual", action="store_true")
    parser.add_argument("--size-depth-sigma-m", type=float, default=200.0)
    parser.add_argument("--weak-size-depth-sigma-m", type=float, default=6.0)
    parser.add_argument("--no-weak-segments", action="store_true")
    parser.add_argument("--weak-segment-min-observations", type=int, default=4)
    return parser.parse_args()


def _tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.ball_track is not None or args.court_calibration is not None:
        if args.clip is None or args.ball_track is None or args.court_calibration is None:
            raise SystemExit("--clip, --ball-track, and --court-calibration are required for explicit runs")
        return [
            {
                "clip": args.clip,
                "ball_track": args.ball_track,
                "court_calibration": args.court_calibration,
                "ball_sizes": args.ball_sizes,
                "ball_candidates": list(getattr(args, "ball_candidates", None) or []),
                "candidate_extra_tracks": _candidate_extra_track_specs(getattr(args, "candidate_extra_track", None) or []),
                "contact_windows": args.contact_windows,
                "skeleton3d": args.skeleton3d,
                "frame_times": args.frame_times,
                "reviewed_bounces": args.reviewed_bounces,
                "auto_bounce_candidates": args.auto_bounce_candidates,
                "rally_spans": args.rally_spans,
                "net_plane": args.net_plane,
            }
        ]

    clip_dirs = list(args.clip_dir or [])
    if args.prototype_root is not None:
        selected = args.clips or [
            path.name
            for path in sorted(args.prototype_root.iterdir())
            if path.is_dir() and (path / "ball_track.json").is_file()
        ]
        clip_dirs.extend(args.prototype_root / clip for clip in selected)

    tasks: list[dict[str, Any]] = []
    for clip_dir in clip_dirs:
        clip = _clip_name_for_dir(clip_dir)
        task = {
            "clip": clip,
            "ball_track": clip_dir / "ball_track.json",
            "court_calibration": clip_dir / "court_calibration.json",
            "ball_sizes": clip_dir / "ball_size_observations.json" if (clip_dir / "ball_size_observations.json").is_file() else None,
            "ball_candidates": [clip_dir / "ball_candidates.json"] if (clip_dir / "ball_candidates.json").is_file() else list(getattr(args, "ball_candidates", None) or []),
            "candidate_extra_tracks": _candidate_extra_track_specs(getattr(args, "candidate_extra_track", None) or []),
            "contact_windows": clip_dir / "contact_windows.json" if (clip_dir / "contact_windows.json").is_file() else None,
            "skeleton3d": clip_dir / "skeleton3d.json" if (clip_dir / "skeleton3d.json").is_file() else None,
            "frame_times": clip_dir / "frame_times.json" if (clip_dir / "frame_times.json").is_file() else None,
            "reviewed_bounces": _default_reviewed_bounces_path(clip_dir, clip),
            "auto_bounce_candidates": clip_dir / "auto_bounce_candidates.json" if (clip_dir / "auto_bounce_candidates.json").is_file() else None,
            "rally_spans": clip_dir / "rally_spans.json" if (clip_dir / "rally_spans.json").is_file() else None,
            "net_plane": clip_dir / "net_plane.json" if (clip_dir / "net_plane.json").is_file() else None,
        }
        missing = [name for name in ("ball_track", "court_calibration") if not Path(task[name]).is_file()]
        if missing:
            raise SystemExit(f"{clip_dir}: missing required inputs: {', '.join(missing)}")
        tasks.append(task)
    if not tasks:
        raise SystemExit("provide explicit --ball-track/--court-calibration, --clip-dir, or --prototype-root")
    return tasks


def _clip_name_for_dir(clip_dir: Path) -> str:
    for name in ("replay_viewer_manifest.json", "pipeline_run.json"):
        path = clip_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping) and isinstance(payload.get("clip"), str) and payload["clip"]:
            return str(payload["clip"])
    return clip_dir.name


def _default_reviewed_bounces_path(clip_dir: Path, clip: str) -> Path | None:
    direct = clip_dir / "reviewed_ball_bounces.json"
    if direct.is_file():
        return direct
    review_root = Path("runs/ball_bounce_inout_review_packets_ground_contact_only_20260701T200001Z")
    candidate = review_root / clip / "reviewed_ball_bounces.json"
    return candidate if candidate.is_file() else None


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return payload


def _read_optional_json(path: Any) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _read_ball_candidates(path: Path) -> dict[str, Any]:
    return load_ball_candidates_file(path).model_dump()


def build_product_ball_track_view(
    *,
    arc_solved: Mapping[str, Any],
    fused_track: Mapping[str, Any],
    calibration: Mapping[str, Any],
    measured_bands: set[str] | None = None,
    veto_px: float | None = None,
    weak_support_required: bool = True,
    weak_confidence_threshold: float = 0.6,
    fusion_decisions: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the measured-plus-fused product view, optionally vetoing weak fallbacks.

    This is deliberately a view-builder helper, not solver behavior. Measured
    frames are always the selected arc-solver measured bands; the veto only
    suppresses fused fallback frames that disagree with an available solved arc.
    """

    solver_status = str(arc_solved.get("status") or "ran")
    kill_reasons = _kill_reasons(arc_solved)
    if solver_status != "ran":
        return _fused_only_product_view(
            fused_track,
            solver_status=solver_status,
            kill_reasons=kill_reasons,
            weak_support_required=weak_support_required,
            weak_confidence_threshold=weak_confidence_threshold,
        )

    bands = measured_bands or {"anchored_measured"}
    arc_frames = list(arc_solved.get("frames") or [])
    fused_frames = list(fused_track.get("frames") or [])
    if len(arc_frames) != len(fused_frames):
        raise ValueError(f"arc/fused frame count mismatch: {len(arc_frames)} != {len(fused_frames)}")
    if veto_px is not None and (not math.isfinite(float(veto_px)) or float(veto_px) <= 0.0):
        raise ValueError("veto_px must be positive when provided")
    if weak_confidence_threshold < 0.0 or weak_confidence_threshold > 1.0:
        raise ValueError("weak_confidence_threshold must be in [0, 1]")

    fps = _float_or_none(fused_track.get("fps")) or _float_or_none(arc_solved.get("fps")) or 30.0
    decisions_by_frame = _fusion_decisions_by_frame(fusion_decisions)
    output_frames: list[dict[str, Any]] = []
    dropped_frames: list[int] = []
    veto_details: list[dict[str, Any]] = []
    kept_strong_support_count = 0
    distance_checked_count = 0
    fallback_visible_count = 0
    measured_visible_count = 0

    for frame_index, (arc_frame, fused_frame) in enumerate(zip(arc_frames, fused_frames, strict=True)):
        t = _frame_t(arc_frame, fused_frame, frame_index=frame_index, fps=fps)
        arc_projection = _arc_projection_frame(arc_frame, calibration=calibration, t=t)
        if arc_projection is not None and str(arc_frame.get("band") or "") in bands:
            output_frames.append(arc_projection)
            measured_visible_count += 1
            continue

        fallback = _fallback_frame(fused_frame, t=t)
        if not fallback["visible"]:
            output_frames.append(fallback)
            continue

        fallback_visible_count += 1
        should_drop = False
        detail: dict[str, Any] | None = None
        if veto_px is not None and arc_projection is not None:
            distance_checked_count += 1
            distance_px = _distance2(fallback["xy"], arc_projection["xy"])
            if distance_px > float(veto_px):
                decision = decisions_by_frame.get(frame_index)
                weak_reasons = _weak_fusion_support_reasons(
                    fused_frame,
                    decision,
                    confidence_threshold=weak_confidence_threshold,
                )
                weak_enough = bool(weak_reasons) or not weak_support_required
                if weak_enough:
                    should_drop = True
                    detail = {
                        "frame": frame_index,
                        "t": round(t, 9),
                        "distance_px": round(distance_px, 6),
                        "veto_px": float(veto_px),
                        "weak_support_required": bool(weak_support_required),
                        "weak_support_reasons": weak_reasons,
                        "fusion_decision": dict(decision or {}),
                        "fallback_conf": _optional_round(_float_or_none(fused_frame.get("conf")), 6),
                    }
                elif weak_support_required:
                    kept_strong_support_count += 1
        if should_drop:
            output_frames.append(_hidden_frame_for_view(t))
            dropped_frames.append(frame_index)
            if detail is not None:
                veto_details.append(detail)
        else:
            output_frames.append(fallback)

    payload = {
        "schema_version": 1,
        "fps": fps,
        "source": "physics_filled",
        "frames": output_frames,
        "bounces": [],
    }
    report = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_product_view_report",
        "product_view_mode": "arc_composed",
        "solver_status": solver_status,
        "solver_killed": False,
        "kill_reasons": kill_reasons,
        "frame_count": len(output_frames),
        "measured_bands": sorted(bands),
        "visible_count": sum(1 for frame in output_frames if frame.get("visible") is True),
        "measured_visible_count": measured_visible_count,
        "fallback_visible_count": fallback_visible_count,
        "veto": {
            "enabled": veto_px is not None,
            "veto_px": None if veto_px is None else float(veto_px),
            "weak_support_required": bool(weak_support_required),
            "weak_confidence_threshold": float(weak_confidence_threshold),
            "distance_checked_count": distance_checked_count,
            "dropped_count": len(dropped_frames),
            "dropped_frames": dropped_frames,
            "kept_strong_support_count": kept_strong_support_count,
            "details": veto_details,
        },
    }
    return payload, report


def _fused_only_product_view(
    fused_track: Mapping[str, Any],
    *,
    solver_status: str,
    kill_reasons: Sequence[str],
    weak_support_required: bool,
    weak_confidence_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fps = _float_or_none(fused_track.get("fps")) or 30.0
    fused_frames = list(fused_track.get("frames") or [])
    output_frames = []
    for index, frame in enumerate(fused_frames):
        fused_frame = frame if isinstance(frame, Mapping) else {}
        output_frames.append(
            _fallback_frame(
                fused_frame,
                t=_frame_t({}, fused_frame, frame_index=index, fps=fps),
            )
        )
    visible_count = sum(1 for frame in output_frames if frame.get("visible") is True)
    payload = {
        "schema_version": 1,
        "fps": fps,
        "source": "fused",
        "product_view_mode": "fused_only_solver_killed",
        "solver_status": solver_status,
        "kill_reasons": list(kill_reasons),
        "frames": output_frames,
        "bounces": list(fused_track.get("bounces") or []),
    }
    report = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_product_view_report",
        "product_view_mode": "fused_only_solver_killed",
        "solver_status": solver_status,
        "solver_killed": True,
        "kill_reasons": list(kill_reasons),
        "frame_count": len(output_frames),
        "measured_bands": [],
        "visible_count": visible_count,
        "measured_visible_count": 0,
        "fallback_visible_count": visible_count,
        "veto": {
            "enabled": False,
            "veto_px": None,
            "weak_support_required": bool(weak_support_required),
            "weak_confidence_threshold": float(weak_confidence_threshold),
            "distance_checked_count": 0,
            "dropped_count": 0,
            "dropped_frames": [],
            "kept_strong_support_count": 0,
            "details": [],
        },
    }
    return payload, report


def _kill_reasons(arc_solved: Mapping[str, Any]) -> list[str]:
    raw = arc_solved.get("kill_reasons")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    return [str(item) for item in raw]


def _frame_t(
    arc_frame: Mapping[str, Any],
    fused_frame: Mapping[str, Any],
    *,
    frame_index: int,
    fps: float,
) -> float:
    return (
        _float_or_none(arc_frame.get("t"))
        or _float_or_none(fused_frame.get("t"))
        or frame_index / max(float(fps), 1e-9)
    )


def _arc_projection_frame(arc_frame: Mapping[str, Any], *, calibration: Mapping[str, Any], t: float) -> dict[str, Any] | None:
    world_xyz = arc_frame.get("world_xyz")
    if not isinstance(world_xyz, Sequence) or isinstance(world_xyz, (str, bytes)) or len(world_xyz) != 3:
        return None
    try:
        world = (float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in world):
        return None
    xy = _project_world_point(calibration, world)
    return {
        "t": t,
        "xy": [round(float(xy[0]), 6), round(float(xy[1]), 6)],
        "conf": 1.0,
        "visible": True,
        "world_xyz": None,
        "spin_rpm": None,
        "speed_mps": None,
        "approx": str(arc_frame.get("band") or "") != "anchored_measured",
    }


def _fallback_frame(fused_frame: Mapping[str, Any], *, t: float) -> dict[str, Any]:
    if fused_frame.get("visible") is not True:
        return _hidden_frame_for_view(t)
    xy = _xy_tuple(fused_frame.get("xy")) or (0.0, 0.0)
    return {
        "t": t,
        "xy": [float(xy[0]), float(xy[1])],
        "conf": float(_float_or_none(fused_frame.get("conf")) or 0.0),
        "visible": True,
        "world_xyz": None,
        "spin_rpm": None,
        "speed_mps": None,
        "approx": bool(fused_frame.get("approx", False)),
    }


def _hidden_frame_for_view(t: float) -> dict[str, Any]:
    return {
        "t": t,
        "xy": [0.0, 0.0],
        "conf": 0.0,
        "visible": False,
        "world_xyz": None,
        "spin_rpm": None,
        "speed_mps": None,
        "approx": False,
    }


def _fusion_decisions_by_frame(fusion_decisions: Mapping[str, Any] | None) -> dict[int, Mapping[str, Any]]:
    if not isinstance(fusion_decisions, Mapping):
        return {}
    raw_frames = fusion_decisions.get("frames")
    if isinstance(raw_frames, Mapping):
        return {
            int(frame): decision
            for frame, decision in raw_frames.items()
            if str(frame).lstrip("-").isdigit() and isinstance(decision, Mapping)
        }
    if isinstance(raw_frames, Sequence) and not isinstance(raw_frames, (str, bytes)):
        output: dict[int, Mapping[str, Any]] = {}
        for index, item in enumerate(raw_frames):
            if not isinstance(item, Mapping):
                continue
            frame = _frame_from_view_mapping(item)
            output[index if frame is None else frame] = item
        return output
    return {}


def _weak_fusion_support_reasons(
    fused_frame: Mapping[str, Any],
    decision: Mapping[str, Any] | None,
    *,
    confidence_threshold: float,
) -> list[str]:
    reasons: list[str] = []
    conf = _float_or_none(fused_frame.get("conf"))
    if conf is not None and conf < confidence_threshold:
        reasons.append("conf_below_threshold")
    if _is_lone_detector_fusion(decision) or _is_lone_detector_fusion(fused_frame):
        reasons.append("lone_detector_accept")
    return reasons


def _is_lone_detector_fusion(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    decision_type = str(
        payload.get("type")
        or payload.get("decision")
        or payload.get("fusion_type")
        or payload.get("support")
        or ""
    ).lower()
    if "lone" in decision_type:
        return True
    detectors = payload.get("dets") or payload.get("detectors") or payload.get("sources")
    if isinstance(detectors, Sequence) and not isinstance(detectors, (str, bytes)):
        return len(detectors) == 1 and "consensus" not in decision_type
    provenance = payload.get("provenance")
    return _is_lone_detector_fusion(provenance) if isinstance(provenance, Mapping) else False


def _frame_from_view_mapping(item: Mapping[str, Any]) -> int | None:
    for key in ("frame", "frame_index", "idx"):
        value = item.get(key)
        try:
            frame = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        return frame
    return None


def _distance2(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _optional_round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(float(value), digits)


def _candidate_extra_track_specs(items: Sequence[str]) -> dict[str, Path]:
    specs: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit("--candidate-extra-track must be NAME=PATH")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit("--candidate-extra-track name must be non-empty")
        specs[name] = Path(raw_path)
    return specs


def _candidate_score_floor_specs(items: Sequence[str]) -> dict[str, float]:
    floors: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit("--candidate-score-floor must be SOURCE=FLOAT")
        source, raw_value = item.split("=", 1)
        source = source.strip()
        if not source:
            raise SystemExit("--candidate-score-floor source must be non-empty")
        value = _float_or_none(raw_value)
        if value is None or value < 0.0 or value > 1.0:
            raise SystemExit("--candidate-score-floor value must be in [0, 1]")
        floors[source] = value
    return floors


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _measure_ball_sizes_from_video(
    video_path: Path,
    ball_track: Mapping[str, Any],
    *,
    crop_radius_px: int,
) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise SystemExit(f"size measurement helper requires cv2 and numpy: {exc}") from exc
    frames = ball_track.get("frames")
    if not isinstance(frames, list):
        raise ValueError("ball_track.frames must be a list")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    measured: list[dict[str, Any]] = []
    fps = _float_or_none(ball_track.get("fps")) or _float_or_none(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    crop_radius_px = max(4, int(crop_radius_px))
    frame_index = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if frame_index >= len(frames):
            break
        source_frame = frames[frame_index]
        if isinstance(source_frame, Mapping) and source_frame.get("visible") is True:
            xy = _xy_tuple(source_frame.get("xy"))
            if xy is not None:
                item = _measure_one_ball_size(
                    image,
                    xy,
                    frame_index=frame_index,
                    t=_float_or_none(source_frame.get("t")) or frame_index / max(fps, 1e-9),
                    crop_radius_px=crop_radius_px,
                    cv2=cv2,
                    np=np,
                )
                if item is not None:
                    measured.append(item)
        frame_index += 1
    cap.release()
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_size_observations",
        "source": "video_crop_color_component_heuristic",
        "video": str(video_path),
        "known_ball_diameter_m": 0.0742,
        "policy": {
            "render_only_depth_cue": True,
            "weak_measurement": True,
            "not_ground_truth": True,
            "notes": [
                "Measured from local crops around existing ball_track xy; no detector labels or held-out annotations are read.",
                "Confidence is heuristic and must be used only as a weak residual weight.",
            ],
        },
        "summary": {
            "input_frame_count": len(frames),
            "video_frame_count_read": frame_index,
            "measured_count": len(measured),
            "crop_radius_px": crop_radius_px,
        },
        "frames": measured,
    }


def _measure_one_ball_size(
    image: Any,
    xy: tuple[float, float],
    *,
    frame_index: int,
    t: float,
    crop_radius_px: int,
    cv2: Any,
    np: Any,
) -> dict[str, Any] | None:
    height, width = image.shape[:2]
    x, y = xy
    x0 = max(0, int(round(x - crop_radius_px)))
    x1 = min(width, int(round(x + crop_radius_px + 1)))
    y0 = max(0, int(round(y - crop_radius_px)))
    y1 = min(height, int(round(y + crop_radius_px + 1)))
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    bright_floor = float(np.percentile(val, 80))
    mask = (((val >= bright_floor) & (sat > 45)) | ((hue >= 15) & (hue <= 45) & (val > 80) & (sat > 35))).astype("uint8") * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    local_x = x - x0
    local_y = y - y0
    candidates: list[tuple[float, int, float, float, float, float, tuple[float, float]]] = []
    for component in range(1, count):
        area = int(stats[component, cv2.CC_STAT_AREA])
        if area < 4 or area > max(80, crop_radius_px * crop_radius_px):
            continue
        component_width = float(stats[component, cv2.CC_STAT_WIDTH])
        component_height = float(stats[component, cv2.CC_STAT_HEIGHT])
        diameter_px = (component_width + component_height) * 0.5
        equivalent_diameter_px = 2.0 * (area / 3.141592653589793) ** 0.5
        center = (float(centroids[component][0]), float(centroids[component][1]))
        center_distance = ((center[0] - local_x) ** 2 + (center[1] - local_y) ** 2) ** 0.5
        circularity_penalty = abs(diameter_px - equivalent_diameter_px)
        score = center_distance + 0.25 * circularity_penalty - 0.01 * min(area, 120)
        candidates.append((score, area, center_distance, diameter_px, equivalent_diameter_px, circularity_penalty, center))
    if not candidates:
        return None
    _, area, center_distance, diameter_px, equivalent_diameter_px, circularity_penalty, center = sorted(candidates, key=lambda item: item[0])[0]
    center_score = 1.0 / (1.0 + center_distance / 4.0)
    area_score = min(1.0, area / 80.0)
    shape_score = 1.0 / (1.0 + circularity_penalty / 4.0)
    confidence = max(0.0, min(1.0, center_score * area_score * shape_score))
    return {
        "frame": int(frame_index),
        "t": round(float(t), 9),
        "xy": [round(float(x), 6), round(float(y), 6)],
        "diameter_px": round(float(diameter_px), 6),
        "radius_px": round(float(diameter_px) / 2.0, 6),
        "equivalent_diameter_px": round(float(equivalent_diameter_px), 6),
        "confidence": round(float(confidence), 6),
        "source": "video_crop_color_component_heuristic",
        "details": {
            "component_area_px": int(area),
            "component_center_xy": [round(center[0] + x0, 6), round(center[1] + y0, 6)],
            "center_distance_px": round(float(center_distance), 6),
        },
    }


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _report_payload(clip: str, artifact: Mapping[str, Any], artifact_path: Path, events_selected_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_arc_solver_report",
        "clip": clip,
        "ball_track_arc_solved": str(artifact_path),
        "events_selected": str(events_selected_path),
        "status": artifact["status"],
        "kill_reasons": artifact["kill_reasons"],
        "summary": artifact["summary"],
        "leave_one_out": artifact["validation"]["leave_one_out"],
        "physical_sanity": artifact["validation"]["physical_sanity"],
        "candidate_association": artifact["validation"].get("candidate_association", {}),
        "event_selection": artifact["event_selection"],
        "segments": artifact["segments"],
        "policy": artifact["policy"],
    }


def _markdown_report(
    clip: str,
    artifact: Mapping[str, Any],
    artifact_path: Path,
    report_json_path: Path,
) -> str:
    summary = artifact["summary"]
    loo = artifact["validation"]["leave_one_out"]
    physical = artifact["validation"]["physical_sanity"]
    event_selection = artifact["event_selection"]
    median_loo = loo["ray_distance_m"]["median"]
    process_video_note = (
        "process_video.py integration is intentionally not applied; PIPELINE-GUARDS should add one line "
        "after ball_track.json is produced to run scripts/racketsport/solve_ball_arcs.py with the clip run directory."
    )
    lines = [
        f"# BALL-ARC-SOLVER Report - {clip}",
        "",
        f"- Artifact: `{artifact_path}`",
        f"- JSON report: `{report_json_path}`",
        f"- Status: `{artifact['status']}`",
        f"- Render only: `{artifact['render_only']}`",
        f"- Not for detection metrics: `{artifact['not_for_detection_metrics']}`",
        f"- Segments: `{summary['segment_count']}`",
        f"- World coverage: `{summary['coverage_world_xyz_count']}/{summary['input_frame_count']}`",
        f"- FP sightings pruned: `{summary['fp_sightings_pruned_count']}`",
        f"- Human-reviewed bounces: `{summary['human_reviewed_bounce_count']}`",
        f"- Auto-bounce candidates: `{summary.get('auto_bounce_candidate_count', 0)}`",
        f"- Solver-proposed bounces: `{summary['discovered_bounce_count']}`",
        f"- Selected optional events: `{event_selection['selected_optional_count']}`",
        f"- Rejected optional events: `{event_selection['rejected_optional_count']}`",
        f"- LOO median ray distance m: `{median_loo}`",
        f"- Physical violations: `{physical['violation_count']}/{physical['segment_count']}`",
        "",
        "## Integration Diff",
        "",
        process_video_note,
        "",
        "## Policy",
        "",
        "This artifact feeds the WORLD only. It must not be used for detector gates, training, or BALL promotion.",
    ]
    candidate_association = artifact["validation"].get("candidate_association", {})
    if candidate_association.get("enabled") is True:
        counts = dict(candidate_association.get("selection_counts_by_source") or {})
        rescue_counts = dict(candidate_association.get("rescue_counts_by_source") or {})
        lines[13:13] = [
            f"- Candidate association: `{candidate_association.get('candidate_selection')}`",
            f"- Candidate association mode: `{candidate_association.get('mode')}`",
            f"- Candidate association converged segments: `{candidate_association.get('converged_segment_count')}/{candidate_association.get('segment_count')}`",
        ]
        for source, count in sorted(counts.items()):
            lines.insert(15, f"- Candidate selections from {source}: `{count}`")
        for source, count in sorted(rescue_counts.items()):
            lines.insert(15, f"- Candidate rescues from {source}: `{count}`")
    if artifact["kill_reasons"]:
        lines.extend(["", "## Kill Reasons", ""])
        lines.extend(f"- {reason}" for reason in artifact["kill_reasons"])
    return "\n".join(lines) + "\n"


def _commands(args: argparse.Namespace, task: Mapping[str, Any], clip_out_dir: Path) -> str:
    parts = [
        "python",
        "scripts/racketsport/solve_ball_arcs.py",
        "--clip",
        str(task["clip"]),
        "--ball-track",
        str(task["ball_track"]),
        "--court-calibration",
        str(task["court_calibration"]),
        "--out-dir",
        str(clip_out_dir.parent),
        "--ball-type",
        args.ball_type,
    ]
    for flag, key in (
        ("--contact-windows", "contact_windows"),
        ("--ball-sizes", "ball_sizes"),
        ("--skeleton3d", "skeleton3d"),
        ("--reviewed-bounces", "reviewed_bounces"),
        ("--auto-bounce-candidates", "auto_bounce_candidates"),
        ("--rally-spans", "rally_spans"),
        ("--net-plane", "net_plane"),
    ):
        if task.get(key):
            parts.extend([flag, str(task[key])])
    for path in task.get("ball_candidates", []) or []:
        parts.extend(["--ball-candidates", str(path)])
    for name, path in sorted((task.get("candidate_extra_tracks") or {}).items()):
        parts.extend(["--candidate-extra-track", f"{name}={path}"])
    parts.extend(["--candidate-association-mode", args.candidate_association_mode])
    for source, floor in sorted(_candidate_score_floor_specs(args.candidate_score_floor).items()):
        parts.extend(["--candidate-score-floor", f"{source}={floor:g}"])
    parts.extend(["--max-candidates-per-frame", str(args.max_candidates_per_frame)])
    parts.extend(["--candidate-selection-max-iterations", str(args.candidate_selection_max_iterations)])
    return "#!/usr/bin/env bash\nset -euo pipefail\n" + " ".join(parts) + "\n"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _xy_tuple(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    try:
        xy = (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None
    if not all(component == component and component not in (float("inf"), float("-inf")) for component in xy):
        return None
    return xy


if __name__ == "__main__":
    raise SystemExit(main())
