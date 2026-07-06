#!/usr/bin/env python3
"""Build a fused wrist+palm+grip-transform render-only racket_pose_estimate.json.

Interface is parallel to build_paddle_proxy.py: --skeleton is required, all evidence-channel
inputs are optional (best-effort), and the output is the SAME racket_pose_estimate.json
artifact contract paddle_proxy.py produces, so virtual_world.py (unmodified) consumes it
identically. See threed/racketsport/paddle_pose_fused.py for the solver design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.paddle_pose_fused import (  # noqa: E402
    DEFAULT_CONTACT_LOCK_WINDOW_S,
    DEFAULT_DETECTOR_BOX_MAX_CORRECTION_M,
    DEFAULT_DETECTOR_BOX_ROLL_SEARCH_DEG,
    DEFAULT_DETECTOR_BOX_WRIST_GATE_RADIUS_PX,
    DEFAULT_HAND_SWITCH_MAJORITY,
    DEFAULT_HAND_SWITCH_MIN_HOLD_S,
    DEFAULT_HAND_SWITCH_MIN_VOTES,
    DEFAULT_MAX_POSITION_JUMP_M_PER_FRAME,
    DEFAULT_PER_FRAME_DEVIATION_DECAY,
    DEFAULT_PER_FRAME_DEVIATION_MAX_M,
    DEFAULT_PER_FRAME_DEVIATION_SLEW_M,
    DEFAULT_POSITION_ONE_EURO_BETA,
    DEFAULT_POSITION_ONE_EURO_MIN_CUTOFF,
    DEFAULT_GRIP_OFFSET_M,
    DEFAULT_MIN_JOINT_CONFIDENCE,
    DEFAULT_MIN_SEGMENT_DURATION_S,
    DEFAULT_ONE_EURO_BETA,
    DEFAULT_ONE_EURO_D_CUTOFF,
    DEFAULT_ONE_EURO_MIN_CUTOFF,
    DEFAULT_PADDLE_DIMS_IN,
    DEFAULT_PRIOR_ROTATION_WEIGHT,
    DEFAULT_PRIOR_TRANSLATION_WEIGHT,
    DEFAULT_REFLECTION_WEIGHT_SCALE,
    DEFAULT_SEGMENT_BREAK_ANGLE_DEG,
    build_paddle_pose_fused_from_file,
    write_paddle_pose_fused,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a fused wrist+palm+grip-transform render-only paddle 6-DOF estimate."
    )
    parser.add_argument("--skeleton", type=Path, required=True, help="Input skeleton3d.json (raw 70-joint MHR array).")
    parser.add_argument("--out", type=Path, required=True, help="Output racket_pose_estimate.json path.")
    parser.add_argument("--clip", required=True, help="Clip/run id for artifact provenance.")
    parser.add_argument("--dominant-hand", choices=("right", "left", "auto"), default="auto")
    parser.add_argument(
        "--player-hand",
        action="append",
        default=[],
        metavar="PLAYER_ID:HAND",
        help="Override a specific player's hand, e.g. --player-hand 7:right. May be repeated.",
    )
    parser.add_argument("--grip-offset-m", type=float, default=DEFAULT_GRIP_OFFSET_M)
    parser.add_argument("--min-joint-confidence", type=float, default=DEFAULT_MIN_JOINT_CONFIDENCE)
    parser.add_argument("--paddle-length-in", type=float, default=DEFAULT_PADDLE_DIMS_IN["length"])
    parser.add_argument("--paddle-width-in", type=float, default=DEFAULT_PADDLE_DIMS_IN["width"])
    parser.add_argument("--ball-track", type=Path, help="Optional ball_track.json (interface symmetry; not required directly).")
    parser.add_argument("--contact-windows", type=Path, help="Optional contact_windows.json (interface symmetry).")
    parser.add_argument(
        "--physics-estimate",
        type=Path,
        help="Optional racket_physics_estimate.json; wires the ball-reflection cone factor when present.",
    )
    parser.add_argument("--detector-boxes", type=Path, help="Optional detector-box predictions for the optional reprojection factor.")
    parser.add_argument("--calibration", type=Path, help="Optional court_calibration.json, required alongside --detector-boxes.")
    parser.add_argument("--membership", type=Path, help="Optional player-court-membership artifact; excluded players get no paddle.")
    parser.add_argument("--disable-reflection", action="store_true", help="Disable the ball-reflection cone factor (ablation).")
    parser.add_argument(
        "--enable-detector-boxes",
        action="store_true",
        help="Enable the optional wrist-gated detector-box reprojection factor (ablation; requires --detector-boxes and --calibration).",
    )
    parser.add_argument(
        "--disable-detector-box-handedness",
        action="store_true",
        help="Disable detector-box handedness verification (on by default when --detector-boxes and --calibration are given).",
    )
    parser.add_argument("--min-segment-duration-s", type=float, default=DEFAULT_MIN_SEGMENT_DURATION_S)
    parser.add_argument("--segment-break-angle-deg", type=float, default=DEFAULT_SEGMENT_BREAK_ANGLE_DEG)
    parser.add_argument("--prior-rotation-weight", type=float, default=DEFAULT_PRIOR_ROTATION_WEIGHT)
    parser.add_argument("--prior-translation-weight", type=float, default=DEFAULT_PRIOR_TRANSLATION_WEIGHT)
    parser.add_argument("--reflection-weight-scale", type=float, default=DEFAULT_REFLECTION_WEIGHT_SCALE)
    parser.add_argument("--detector-box-wrist-gate-radius-px", type=float, default=DEFAULT_DETECTOR_BOX_WRIST_GATE_RADIUS_PX)
    parser.add_argument("--detector-box-max-correction-m", type=float, default=DEFAULT_DETECTOR_BOX_MAX_CORRECTION_M)
    parser.add_argument("--detector-box-roll-search-deg", type=float, default=DEFAULT_DETECTOR_BOX_ROLL_SEARCH_DEG)
    parser.add_argument("--per-frame-deviation-max-m", type=float, default=DEFAULT_PER_FRAME_DEVIATION_MAX_M)
    parser.add_argument("--per-frame-deviation-decay", type=float, default=DEFAULT_PER_FRAME_DEVIATION_DECAY)
    parser.add_argument("--per-frame-deviation-slew-m", type=float, default=DEFAULT_PER_FRAME_DEVIATION_SLEW_M)
    parser.add_argument("--hand-switch-min-hold-s", type=float, default=DEFAULT_HAND_SWITCH_MIN_HOLD_S)
    parser.add_argument("--hand-switch-min-votes", type=int, default=DEFAULT_HAND_SWITCH_MIN_VOTES)
    parser.add_argument("--hand-switch-majority", type=float, default=DEFAULT_HAND_SWITCH_MAJORITY)
    parser.add_argument("--max-position-jump-m-per-frame", type=float, default=DEFAULT_MAX_POSITION_JUMP_M_PER_FRAME)
    parser.add_argument("--position-one-euro-min-cutoff", type=float, default=DEFAULT_POSITION_ONE_EURO_MIN_CUTOFF)
    parser.add_argument("--position-one-euro-beta", type=float, default=DEFAULT_POSITION_ONE_EURO_BETA)
    parser.add_argument("--one-euro-min-cutoff", type=float, default=DEFAULT_ONE_EURO_MIN_CUTOFF)
    parser.add_argument("--one-euro-beta", type=float, default=DEFAULT_ONE_EURO_BETA)
    parser.add_argument("--one-euro-d-cutoff", type=float, default=DEFAULT_ONE_EURO_D_CUTOFF)
    parser.add_argument("--contact-lock-window-s", type=float, default=DEFAULT_CONTACT_LOCK_WINDOW_S)
    args = parser.parse_args(argv)

    try:
        payload = build_paddle_pose_fused_from_file(
            args.skeleton,
            clip_id=args.clip,
            dominant_hand=args.dominant_hand,
            dominant_hand_by_player=_parse_player_hand_overrides(args.player_hand),
            paddle_dims_in={"length": args.paddle_length_in, "width": args.paddle_width_in},
            grip_offset_m=args.grip_offset_m,
            min_joint_confidence=args.min_joint_confidence,
            ball_track=_read_optional_json(args.ball_track),
            contact_windows=_read_optional_json(args.contact_windows),
            physics_estimate=_read_optional_json(args.physics_estimate),
            detector_boxes=_read_optional_json(args.detector_boxes),
            calibration=_read_optional_json(args.calibration),
            membership=_read_optional_json(args.membership),
            use_reflection=not args.disable_reflection,
            use_detector_boxes=args.enable_detector_boxes,
            use_detector_box_handedness=not args.disable_detector_box_handedness,
            min_segment_duration_s=args.min_segment_duration_s,
            segment_break_angle_deg=args.segment_break_angle_deg,
            prior_rotation_weight=args.prior_rotation_weight,
            prior_translation_weight=args.prior_translation_weight,
            reflection_weight_scale=args.reflection_weight_scale,
            detector_box_wrist_gate_radius_px=args.detector_box_wrist_gate_radius_px,
            detector_box_max_correction_m=args.detector_box_max_correction_m,
            detector_box_roll_search_deg=args.detector_box_roll_search_deg,
            per_frame_deviation_max_m=args.per_frame_deviation_max_m,
            per_frame_deviation_decay=args.per_frame_deviation_decay,
            per_frame_deviation_slew_m=args.per_frame_deviation_slew_m,
            hand_switch_min_hold_s=args.hand_switch_min_hold_s,
            hand_switch_min_votes=args.hand_switch_min_votes,
            hand_switch_majority=args.hand_switch_majority,
            max_position_jump_m_per_frame=args.max_position_jump_m_per_frame,
            position_one_euro_min_cutoff=args.position_one_euro_min_cutoff,
            position_one_euro_beta=args.position_one_euro_beta,
            one_euro_min_cutoff=args.one_euro_min_cutoff,
            one_euro_beta=args.one_euro_beta,
            one_euro_d_cutoff=args.one_euro_d_cutoff,
            contact_lock_window_s=args.contact_lock_window_s,
        )
        write_paddle_pose_fused(args.out, payload)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: fused paddle pose build failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "out": str(args.out),
                "summary": payload["summary"],
                "warnings": payload["warnings"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _read_optional_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _parse_player_hand_overrides(values: list[str]) -> dict[int, str]:
    overrides: dict[int, str] = {}
    for value in values:
        if ":" not in value:
            raise ValueError("--player-hand must use PLAYER_ID:HAND")
        player_id_raw, hand = value.split(":", 1)
        try:
            player_id = int(player_id_raw)
        except ValueError as exc:
            raise ValueError("--player-hand PLAYER_ID must be an integer") from exc
        if hand not in {"right", "left"}:
            raise ValueError("--player-hand HAND must be right or left")
        overrides[player_id] = hand
    return overrides


if __name__ == "__main__":
    raise SystemExit(main())
