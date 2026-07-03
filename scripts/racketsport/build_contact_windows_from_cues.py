#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.event_fusion import (
    DEFAULT_MAX_TIME_DELTA_S,
    DEFAULT_POST_WINDOW_S,
    DEFAULT_PRE_WINDOW_S,
    DEFAULT_WRIST_ONLY_CONFIDENCE_CAP,
    DEFAULT_WRIST_ONLY_MIN_SEPARATION_S,
    DEFAULT_WRIST_ONLY_POST_WINDOW_S,
    DEFAULT_WRIST_ONLY_PRE_WINDOW_S,
    fuse_contact_windows_from_cue_files,
)
from threed.racketsport.schemas import ContactWindows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fuse contact cue artifacts into contact_windows.json."
    )
    parser.add_argument("--audio-onsets", type=Path, help="audio_onsets.json artifact.")
    parser.add_argument("--wrist-velocity-peaks", type=Path, required=True, help="wrist_velocity_peaks.json artifact.")
    parser.add_argument("--ball-inflections", type=Path, help="ball_inflections.json artifact.")
    parser.add_argument("--tracks", type=Path, help="Optional tracks.json used to infer FPS.")
    parser.add_argument("--ball-track", type=Path, help="Optional ball_track.json used to infer FPS.")
    parser.add_argument("--fps", type=float, help="Explicit frame rate override.")
    parser.add_argument(
        "--contact-fusion-mode",
        choices=("audio_wrist_ball", "wrist_ball"),
        default="audio_wrist_ball",
        help="Cue families required for a contact window; wrist_ball is an explicit degraded two-cue mode.",
    )
    parser.add_argument(
        "--allow-wrist-only-contact-hints",
        action="store_true",
        help="Emit low-confidence wrist-cue-only contact hints when full cue fusion cannot derive contacts.",
    )
    parser.add_argument("--max-time-delta-s", type=float, default=DEFAULT_MAX_TIME_DELTA_S)
    parser.add_argument("--pre-s", type=float, default=DEFAULT_PRE_WINDOW_S)
    parser.add_argument("--post-s", type=float, default=DEFAULT_POST_WINDOW_S)
    parser.add_argument("--wrist-only-pre-s", type=float, default=DEFAULT_WRIST_ONLY_PRE_WINDOW_S)
    parser.add_argument("--wrist-only-post-s", type=float, default=DEFAULT_WRIST_ONLY_POST_WINDOW_S)
    parser.add_argument("--wrist-only-min-separation-s", type=float, default=DEFAULT_WRIST_ONLY_MIN_SEPARATION_S)
    parser.add_argument("--wrist-only-confidence-cap", type=float, default=DEFAULT_WRIST_ONLY_CONFIDENCE_CAP)
    parser.add_argument("--out", type=Path, required=True, help="Output contact_windows.json.")
    args = parser.parse_args(argv)

    try:
        fps = _resolve_fps(explicit_fps=args.fps, tracks_path=args.tracks, ball_track_path=args.ball_track)
        require_audio = args.contact_fusion_mode == "audio_wrist_ball"
        if args.ball_inflections is None and not args.allow_wrist_only_contact_hints:
            raise ValueError("--ball-inflections is required unless --allow-wrist-only-contact-hints is set")
        payload = fuse_contact_windows_from_cue_files(
            fps=fps,
            audio_onsets_path=args.audio_onsets,
            wrist_velocity_peaks_path=args.wrist_velocity_peaks,
            ball_inflections_path=args.ball_inflections,
            require_audio=require_audio,
            max_time_delta_s=args.max_time_delta_s,
            pre_s=args.pre_s,
            post_s=args.post_s,
            allow_wrist_only_contact_hints=args.allow_wrist_only_contact_hints,
            wrist_only_pre_s=args.wrist_only_pre_s,
            wrist_only_post_s=args.wrist_only_post_s,
            wrist_only_min_separation_s=args.wrist_only_min_separation_s,
            wrist_only_confidence_cap=args.wrist_only_confidence_cap,
        )
        ContactWindows.model_validate(payload)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"ERROR: contact-window cue fusion failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "out": str(args.out),
                "fps": fps,
                "event_count": len(payload.get("events", [])),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _resolve_fps(
    *,
    explicit_fps: float | None,
    tracks_path: Path | None,
    ball_track_path: Path | None,
) -> float:
    if explicit_fps is not None:
        return _positive_fps(explicit_fps, "--fps")
    for path, label in ((tracks_path, "--tracks"), (ball_track_path, "--ball-track")):
        if path is None:
            continue
        payload = _read_json_object(path, label)
        fps = payload.get("fps")
        if fps is not None:
            return _positive_fps(fps, f"{label}.fps")
    raise ValueError("FPS is required; pass --fps or a tracks/ball-track artifact with fps")


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _positive_fps(value: Any, name: str) -> float:
    fps = float(value)
    if fps <= 0:
        raise ValueError(f"{name} must be positive")
    return fps


if __name__ == "__main__":
    raise SystemExit(main())
