#!/usr/bin/env python3
"""Apply render-only ball physics fill to a world artifact directory."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_physics_fill import (
    ARTIFACT_TYPE,
    LANE,
    PhysicsFillConfig,
    fill_ball_track_physics,
    validate_physics_fill,
)
from threed.racketsport.ball_physics3d import reconstruct_bounce_arcs_from_image_track


DEFAULT_WORLD_DIR = Path(
    "runs/v1_polish_20260702T113800Z/metric15/wolverine_mixed_0200_mid_steep_corner"
)


def main() -> int:
    args = _parse_args()
    world_dir = args.world_dir
    generated_at = _utc_stamp()
    out_dir = args.out_dir or Path(f"runs/phys_ballfill_20260702T{generated_at[11:13]}{generated_at[14:16]}{generated_at[17:19]}Z")
    out_dir.mkdir(parents=True, exist_ok=True)

    ball_payload, source_info = _load_ball_payload(world_dir)
    calibration = _load_optional_json(world_dir / "court_calibration.json")
    config = PhysicsFillConfig(
        min_confidence=args.min_confidence,
        min_segment_samples=args.min_segment_samples,
        max_local_segment_samples=args.max_local_segment_samples,
        max_anchor_gap_frames=args.max_anchor_gap_frames,
        max_anchor_speed_mps=args.max_anchor_speed_mps,
        max_fit_rms_m=args.max_fit_rms_m,
        max_fit_max_residual_m=args.max_fit_max_residual_m,
        max_reprojection_error_px=args.max_reprojection_error_px,
        max_extrapolate_frames=args.max_extrapolate_frames,
        drag_per_s=args.drag_per_s,
        max_xy_interpolate_gap_frames=args.max_xy_interpolate_gap_frames,
        max_unreviewed_inflection_speed_px_s=args.max_unreviewed_inflection_speed_px_s,
        inflection_wrist_tolerance_frames=args.inflection_wrist_tolerance_frames,
    )
    reviewed_bounces = _load_optional_json(args.reviewed_bounces) if args.reviewed_bounces else None
    ball_inflections = _load_optional_json(args.ball_inflections) if args.ball_inflections else _load_optional_json(world_dir / "ball_inflections.json")
    wrist_velocity_peaks = _load_optional_json(args.wrist_velocity_peaks) if args.wrist_velocity_peaks else _load_optional_json(world_dir / "wrist_velocity_peaks.json")
    physics3d_reconstruction = None
    physics3d_summary: dict[str, Any] | None = None
    if calibration is not None and not args.no_physics3d:
        physics3d_reconstruction = reconstruct_bounce_arcs_from_image_track(
            ball_payload,
            calibration,
            max_reprojection_rmse_px=args.physics3d_max_reprojection_rmse_px,
            max_fit_samples=args.physics3d_max_fit_samples,
        )
        physics3d_summary = physics3d_reconstruction.summary()
        if physics3d_reconstruction.status != "ran":
            physics3d_reconstruction = None

    filled = fill_ball_track_physics(
        ball_payload,
        calibration=calibration,
        config=config,
        evidence_path=str(source_info["source_path"]),
        reviewed_bounces=reviewed_bounces,
        ball_inflections=ball_inflections,
        wrist_velocity_peaks=wrist_velocity_peaks,
        physics3d_reconstruction=physics3d_reconstruction,
    )
    validation = validate_physics_fill(
        ball_payload,
        calibration=calibration,
        config=config,
        reviewed_bounces=reviewed_bounces,
        ball_inflections=ball_inflections,
        wrist_velocity_peaks=wrist_velocity_peaks,
        seed=args.validation_seed,
        max_samples=args.validation_max_samples,
    )

    filled["physics_fill"]["generated_at"] = generated_at
    filled["physics_fill"]["input"] = source_info
    filled["physics_fill"]["validation"] = validation
    filled["physics_fill"]["physics3d_reconstruction"] = physics3d_summary
    filled["physics_fill"]["reviewed_bounces_input"] = str(args.reviewed_bounces) if args.reviewed_bounces else None
    filled["physics_fill"]["ball_inflections_input"] = str(args.ball_inflections) if args.ball_inflections else (str(world_dir / "ball_inflections.json") if (world_dir / "ball_inflections.json").is_file() else None)
    filled["physics_fill"]["wrist_velocity_peaks_input"] = str(args.wrist_velocity_peaks) if args.wrist_velocity_peaks else (str(world_dir / "wrist_velocity_peaks.json") if (world_dir / "wrist_velocity_peaks.json").is_file() else None)

    filled_path = out_dir / "ball_track_physics_filled.json"
    report_path = out_dir / "physics_fill_report.json"
    commands_path = out_dir / "COMMANDS.sh"
    markdown_path = out_dir / "REPORT.md"

    _write_json(filled_path, filled)
    _write_json(
        report_path,
        {
            "artifact_type": ARTIFACT_TYPE,
            "lane": LANE,
            "generated_at": generated_at,
            "world_dir": str(world_dir),
            "output": str(filled_path),
            "input": source_info,
            "coverage": filled["physics_fill"]["coverage"],
            "segments": filled["physics_fill"]["segments"],
            "bounce_boundaries": filled["physics_fill"]["bounce_boundaries"],
            "physics3d_reconstruction": physics3d_summary,
            "validation": validation,
            "policy": {
                "render_only": True,
                "not_for_detection_metrics": True,
                "protected_eval_labels_used": False,
                "notes": [
                    "No Outdoor/Indoor labels are read.",
                    "The filled JSON is additive render continuity data only.",
                    "Confident world samples are preserved and generated samples carry source=physics_interpolated.",
                ],
            },
        },
    )
    _write_text(markdown_path, _markdown_report(report_path, filled_path, filled, validation, source_info))
    _write_text(commands_path, _commands(world_dir=world_dir, out_dir=out_dir, args=args))
    commands_path.chmod(0o755)

    print(json.dumps({"output": str(filled_path), "report": str(report_path)}, indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-dir", type=Path, default=DEFAULT_WORLD_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--min-segment-samples", type=int, default=4)
    parser.add_argument("--max-local-segment-samples", type=int, default=8)
    parser.add_argument("--max-anchor-gap-frames", type=int, default=12)
    parser.add_argument("--max-anchor-speed-mps", type=float, default=120.0)
    parser.add_argument("--max-fit-rms-m", type=float, default=0.35)
    parser.add_argument("--max-fit-max-residual-m", type=float, default=0.75)
    parser.add_argument("--max-reprojection-error-px", type=float, default=18.0)
    parser.add_argument("--max-extrapolate-frames", type=int, default=2)
    parser.add_argument("--drag-per-s", type=float, default=0.0)
    parser.add_argument("--max-xy-interpolate-gap-frames", type=int, default=8)
    parser.add_argument("--max-unreviewed-inflection-speed-px-s", type=float, default=5000.0)
    parser.add_argument("--inflection-wrist-tolerance-frames", type=int, default=3)
    parser.add_argument("--reviewed-bounces", type=Path, default=None)
    parser.add_argument("--ball-inflections", type=Path, default=None)
    parser.add_argument("--wrist-velocity-peaks", type=Path, default=None)
    parser.add_argument("--no-physics3d", action="store_true", help="Disable calibrated ball_physics3d z reconstruction.")
    parser.add_argument("--physics3d-max-reprojection-rmse-px", type=float, default=12.0)
    parser.add_argument("--physics3d-max-fit-samples", type=int, default=13)
    parser.add_argument("--validation-seed", type=int, default=20260702)
    parser.add_argument("--validation-max-samples", type=int, default=None)
    return parser.parse_args()


def _load_ball_payload(world_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    ball_track_path = world_dir / "ball_track.json"
    virtual_world_path = world_dir / "virtual_world.json"
    ball_track = _load_optional_json(ball_track_path) or {}
    virtual_world = _load_optional_json(virtual_world_path) or {}
    ball_track_frames = _frames(ball_track)
    virtual_ball = virtual_world.get("ball") if isinstance(virtual_world.get("ball"), Mapping) else {}
    virtual_frames = _frames(virtual_ball)
    ball_track_world_count = _world_count(ball_track_frames)
    virtual_world_count = _world_count(virtual_frames)

    if virtual_frames and virtual_world_count > ball_track_world_count:
        payload = {
            "schema_version": int(ball_track.get("schema_version", 1)),
            "fps": float(ball_track.get("fps") or virtual_world.get("fps") or 30.0),
            "source": virtual_ball.get("source") or ball_track.get("source") or "wasb",
            "frames": virtual_frames,
            "bounces": ball_track.get("bounces", []),
        }
        return payload, {
            "selected_source": "virtual_world.ball",
            "source_path": str(virtual_world_path),
            "ball_track_path": str(ball_track_path),
            "ball_track_frame_count": len(ball_track_frames),
            "ball_track_world_xyz_count": ball_track_world_count,
            "virtual_world_ball_frame_count": len(virtual_frames),
            "virtual_world_world_xyz_count": virtual_world_count,
            "source_note": (
                "Selected virtual_world.ball because it contains more world_xyz samples than ball_track.json. "
                "In the Wolverine v1 input these are court-plane approximate world samples, not BALL gate evidence."
            ),
        }

    if not ball_track_frames:
        raise FileNotFoundError(f"no ball frames found under {world_dir}")
    return dict(ball_track), {
        "selected_source": "ball_track.json",
        "source_path": str(ball_track_path),
        "ball_track_frame_count": len(ball_track_frames),
        "ball_track_world_xyz_count": ball_track_world_count,
        "virtual_world_ball_frame_count": len(virtual_frames),
        "virtual_world_world_xyz_count": virtual_world_count,
    }


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _frames(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    frames = payload.get("frames")
    if not isinstance(frames, list):
        return []
    return [dict(frame) for frame in frames if isinstance(frame, Mapping)]


def _world_count(frames: list[Mapping[str, Any]]) -> int:
    count = 0
    for frame in frames:
        world_xyz = frame.get("world_xyz")
        if isinstance(world_xyz, list) and len(world_xyz) == 3:
            count += 1
    return count


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _markdown_report(
    report_path: Path,
    filled_path: Path,
    filled: Mapping[str, Any],
    validation: Mapping[str, Any],
    source_info: Mapping[str, Any],
) -> str:
    coverage = filled["physics_fill"]["coverage"]
    loo = validation["leave_one_out"]
    error_3d = loo["error_3d_m"]
    reproj = loo["reprojection_error_px"]
    return "\n".join(
        [
            "# PHYS-BALLFILL Report",
            "",
            "Render-only physics interpolation for ball continuity. This output is not for BALL gates, metrics, training, or promotion.",
            "",
            f"- Filled track: `{filled_path}`",
            f"- Machine report: `{report_path}`",
            f"- Selected input: `{source_info.get('selected_source')}` from `{source_info.get('source_path')}`",
            f"- Source note: {source_info.get('source_note', 'n/a')}",
            f"- Input frames/world_xyz: {coverage['input_frame_count']} / {coverage['input_world_xyz_count']}",
            f"- Output world_xyz: {coverage['output_world_xyz_count']}",
            f"- Filled frames: {coverage['filled_frame_count']}",
            f"- Physics3D reconstructed frames: {coverage.get('physics3d_reconstructed_frame_count', 0)}",
            f"- Clamped-to-court-plane filled frames: {coverage.get('clamped_to_court_plane_frame_count', 0)}",
            f"- 2D trail additive frames (`xy_interpolated`): {coverage.get('xy_interpolated_frame_count', 0)}",
            f"- Lifted 2D frames: {coverage['lifted_2d_frame_count']}",
            f"- Reprojection rejects: {coverage['reprojection_rejected_frame_count']}",
            f"- Segments fitted: {len(filled['physics_fill']['segments'])}",
            f"- Bounce boundaries detected: {len(filled['physics_fill']['bounce_boundaries'])}",
            f"- Leave-one-out samples: {loo['sample_count']} / {loo['candidate_count']}",
            f"- LOO 3D error max m: {error_3d['max']}",
            f"- LOO reprojection error max px: {reproj['max']}",
            "",
            "Policy notes:",
            "- No Outdoor/Indoor labels were read.",
            "- Confident world samples are preserved.",
            "- Generated samples carry `source=\"physics_interpolated\"` and a low-confidence trust band.",
            "- Calibrated z samples carry `source=\"physics3d_reconstructed\"`, `render_uncertainty_m`, and `physics_fill.uncertainty_m`.",
            "- 2D-only trail continuity is additive under `xy_interpolated`; measured `xy` and `visible` are never overwritten.",
            "- Thresholded render views should be allowed to hide these generated samples.",
            "",
        ]
    )


def _commands(*, world_dir: Path, out_dir: Path, args: argparse.Namespace) -> str:
    option_lines = [
        f"  --world-dir {world_dir}",
        f"  --out-dir {out_dir}",
        f"  --min-confidence {args.min_confidence}",
        f"  --min-segment-samples {args.min_segment_samples}",
        f"  --max-local-segment-samples {args.max_local_segment_samples}",
        f"  --max-anchor-gap-frames {args.max_anchor_gap_frames}",
        f"  --max-anchor-speed-mps {args.max_anchor_speed_mps}",
        f"  --max-fit-rms-m {args.max_fit_rms_m}",
        f"  --max-fit-max-residual-m {args.max_fit_max_residual_m}",
        f"  --max-reprojection-error-px {args.max_reprojection_error_px}",
        f"  --max-extrapolate-frames {args.max_extrapolate_frames}",
        f"  --drag-per-s {args.drag_per_s}",
        f"  --max-xy-interpolate-gap-frames {args.max_xy_interpolate_gap_frames}",
        f"  --max-unreviewed-inflection-speed-px-s {args.max_unreviewed_inflection_speed_px_s}",
        f"  --inflection-wrist-tolerance-frames {args.inflection_wrist_tolerance_frames}",
        f"  --physics3d-max-reprojection-rmse-px {args.physics3d_max_reprojection_rmse_px}",
        f"  --physics3d-max-fit-samples {args.physics3d_max_fit_samples}",
    ]
    if args.reviewed_bounces:
        option_lines.append(f"  --reviewed-bounces {args.reviewed_bounces}")
    if args.ball_inflections:
        option_lines.append(f"  --ball-inflections {args.ball_inflections}")
    if args.wrist_velocity_peaks:
        option_lines.append(f"  --wrist-velocity-peaks {args.wrist_velocity_peaks}")
    if args.no_physics3d:
        option_lines.append("  --no-physics3d")
    option_lines.append(f"  --validation-seed {args.validation_seed}")
    rendered_options = [
        f"{line} \\" if index < len(option_lines) - 1 else line
        for index, line in enumerate(option_lines)
    ]
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "python scripts/racketsport/apply_ball_physics_fill.py \\",
            *rendered_options,
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
