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

from threed.racketsport.shot_transfer_baseline import (
    DEFAULT_MAX_BALL_DT_S,
    classify_shots_from_payloads,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate provisional shot labels from contact_windows.json and ball_inflections.json."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing cue artifacts.")
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--contact-windows", type=Path, help="Override contact_windows.json path.")
    parser.add_argument("--ball-inflections", type=Path, help="Override ball_inflections.json path.")
    parser.add_argument("--skeleton3d", type=Path, help="Override skeleton3d.json path.")
    parser.add_argument("--smpl-motion", type=Path, help="Override smpl_motion.json path.")
    parser.add_argument("--tracks", type=Path, help="Override tracks.json path.")
    parser.add_argument("--ball-track", type=Path, help="Override ball_track.json path.")
    parser.add_argument("--max-ball-dt-s", type=float, default=DEFAULT_MAX_BALL_DT_S)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        contact_path = args.contact_windows or args.run_dir / "contact_windows.json"
        ball_path = args.ball_inflections or args.run_dir / "ball_inflections.json"
        skeleton_path = args.skeleton3d or args.run_dir / "skeleton3d.json"
        smpl_path = args.smpl_motion or args.run_dir / "smpl_motion.json"
        tracks_path = args.tracks or args.run_dir / "tracks.json"
        ball_track_path = args.ball_track or args.run_dir / "ball_track.json"
        payload = classify_shots_from_payloads(
            clip_id=args.clip_id,
            contact_windows_payload=_read_json_object(contact_path, "contact windows"),
            ball_inflections_payload=_read_json_object(ball_path, "ball inflections") if ball_path.exists() else {},
            skeleton3d_payload=_read_json_object(skeleton_path, "skeleton3d") if skeleton_path.exists() else None,
            smpl_motion_payload=_read_json_object(smpl_path, "smpl motion") if smpl_path.exists() else None,
            tracks_payload=_read_json_object(tracks_path, "tracks") if tracks_path.exists() else None,
            ball_track_payload=_read_json_object(ball_track_path, "ball track") if ball_track_path.exists() else None,
            max_ball_dt_s=args.max_ball_dt_s,
        )
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"ERROR: shot classification generation failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "out": str(args.out_json),
                "shot_count": payload["summary"]["shot_count"],
                "known_count": payload["summary"]["known_count"],
                "unknown_count": payload["summary"]["unknown_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
