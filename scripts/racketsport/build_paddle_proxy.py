#!/usr/bin/env python3
"""Build render-only wrist-proxy paddle estimates from skeleton3d.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.paddle_proxy import (  # noqa: E402
    DEFAULT_GRIP_OFFSET_M,
    DEFAULT_MIN_JOINT_CONFIDENCE,
    DEFAULT_MOTION_HINT_MAX_BALL_TIME_DELTA_S,
    DEFAULT_PADDLE_DIMS_IN,
    DEFAULT_SMOOTHING_ALPHA,
    DEFAULT_WRIST_OFFSET_M,
    build_paddle_proxy_from_file,
    write_paddle_proxy,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a wrist-anchored, render-only racket_pose_estimate.json paddle proxy."
    )
    parser.add_argument("--skeleton3d", type=Path, required=True, help="Input skeleton3d.json.")
    parser.add_argument("--out", type=Path, required=True, help="Output racket_pose_estimate.json path.")
    parser.add_argument("--clip", required=True, help="Clip/run id for artifact provenance.")
    parser.add_argument("--dominant-hand", choices=("right", "left", "auto"), default="right")
    parser.add_argument(
        "--player-hand",
        action="append",
        default=[],
        metavar="PLAYER_ID:HAND",
        help="Override a specific player's hand, e.g. --player-hand 7:right. May be repeated.",
    )
    parser.add_argument(
        "--wrist-offset-m",
        type=float,
        default=None,
        help=(
            "Legacy explicit wrist-to-face-center offset in meters. "
            f"Omit to place the grip at the hand and derive face center from paddle dimensions. Historical default was {DEFAULT_WRIST_OFFSET_M}."
        ),
    )
    parser.add_argument("--grip-offset-m", type=float, default=DEFAULT_GRIP_OFFSET_M)
    parser.add_argument("--min-joint-confidence", type=float, default=DEFAULT_MIN_JOINT_CONFIDENCE)
    parser.add_argument("--smoothing-alpha", type=float, default=DEFAULT_SMOOTHING_ALPHA)
    parser.add_argument("--ball-track", type=Path, help="Optional ball_track.json for estimated orientation hints.")
    parser.add_argument("--motion-hint-max-ball-time-delta-s", type=float, default=DEFAULT_MOTION_HINT_MAX_BALL_TIME_DELTA_S)
    parser.add_argument("--paddle-length-in", type=float, default=DEFAULT_PADDLE_DIMS_IN["length"])
    parser.add_argument("--paddle-width-in", type=float, default=DEFAULT_PADDLE_DIMS_IN["width"])
    args = parser.parse_args(argv)

    try:
        payload = build_paddle_proxy_from_file(
            args.skeleton3d,
            clip_id=args.clip,
            dominant_hand=args.dominant_hand,
            dominant_hand_by_player=_parse_player_hand_overrides(args.player_hand),
            paddle_dims_in={"length": args.paddle_length_in, "width": args.paddle_width_in},
            wrist_offset_m=args.wrist_offset_m,
            grip_offset_m=args.grip_offset_m,
            min_joint_confidence=args.min_joint_confidence,
            smoothing_alpha=args.smoothing_alpha,
            ball_track=_read_optional_json(args.ball_track),
            motion_hint_max_ball_time_delta_s=args.motion_hint_max_ball_time_delta_s,
        )
        write_paddle_proxy(args.out, payload)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: paddle proxy build failed: {exc}", file=sys.stderr)
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
