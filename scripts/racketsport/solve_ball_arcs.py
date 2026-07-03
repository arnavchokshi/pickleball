#!/usr/bin/env python3
"""Solve event-anchored render-only 3D ball arcs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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
    solve_ball_arc_track,
)


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
        contact_windows = _read_optional_json(task.get("contact_windows"))
        skeleton3d = _read_optional_json(task.get("skeleton3d"))
        reviewed_bounces = _read_optional_json(task.get("reviewed_bounces"))
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
        )
        artifact = solve_ball_arc_track(
            ball_track=ball_track,
            calibration=calibration,
            ball_sizes=ball_sizes,
            contact_windows=contact_windows,
            skeleton3d=skeleton3d,
            reviewed_bounces=reviewed_bounces,
            rally_spans=rally_spans,
            net_plane=net_plane,
            physics=physics,
            config=config,
            clip_id=clip,
        )
        artifact["generated_at"] = generated_at
        artifact["inputs"] = {
            "ball_track": str(task["ball_track"]),
            "court_calibration": str(task["court_calibration"]),
            "ball_sizes": str(task.get("ball_sizes") or ""),
            "contact_windows": str(task.get("contact_windows") or ""),
            "skeleton3d": str(task.get("skeleton3d") or ""),
            "reviewed_bounces": str(task.get("reviewed_bounces") or ""),
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
    parser.add_argument("--measure-size-video", type=Path, help="Helper mode: measure apparent ball sizes from a video and ball_track.json.")
    parser.add_argument("--ball-sizes-out", type=Path, help="Output path for --measure-size-video helper mode.")
    parser.add_argument("--size-crop-radius-px", type=int, default=24, help="Crop radius around each ball xy for size measurement helper.")
    parser.add_argument("--contact-windows", type=Path, help="Optional contact_windows.json path.")
    parser.add_argument("--skeleton3d", type=Path, help="Optional skeleton3d.json path.")
    parser.add_argument("--reviewed-bounces", type=Path, help="Optional reviewed_ball_bounces.json path.")
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
                "contact_windows": args.contact_windows,
                "skeleton3d": args.skeleton3d,
                "reviewed_bounces": args.reviewed_bounces,
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
            "contact_windows": clip_dir / "contact_windows.json" if (clip_dir / "contact_windows.json").is_file() else None,
            "skeleton3d": clip_dir / "skeleton3d.json" if (clip_dir / "skeleton3d.json").is_file() else None,
            "reviewed_bounces": _default_reviewed_bounces_path(clip_dir, clip),
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
        ("--rally-spans", "rally_spans"),
        ("--net-plane", "net_plane"),
    ):
        if task.get(key):
            parts.extend([flag, str(task[key])])
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
