#!/usr/bin/env python3
"""CLI: derive rally_spans.json from cheap already-computed per-clip signals.

Standalone tool -- does not touch the orchestrator/pipeline_cli. Any stage
(BALL, BODY, TRK/association, world build) can call this once per clip and
then filter its own frame loop against the emitted spans (see
``threed.racketsport.rally_gating.frame_schedule`` / ``in_rally_span``).

Example:

    python3 scripts/racketsport/build_rally_spans.py \\
        --clip-id burlington_gold_0300_low_steep_corner \\
        --ball-track runs/.../ball_track.json \\
        --tracks runs/.../tracks.json \\
        --audio-onsets runs/.../audio_onsets.json \\
        --duration-s 10.01 \\
        --out runs/.../rally_spans.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.rally_gating import (
    DEFAULT_PAD_SECONDS,
    build_rally_spans_artifact_from_paths,
    load_json,
)


def _infer_duration_s(*, ball_track_path: Path | None, tracks_path: Path | None) -> float | None:
    """Best-effort duration inference from the max timestamp of any provided signal."""

    candidates: list[float] = []
    if ball_track_path and ball_track_path.exists():
        payload = load_json(ball_track_path)
        frames = payload.get("frames", [])
        if frames:
            candidates.append(max(float(f["t"]) for f in frames))
    if tracks_path and tracks_path.exists():
        payload = load_json(tracks_path)
        for player in payload.get("players", []):
            frames = player.get("frames", [])
            if frames:
                candidates.append(max(float(f["t"]) for f in frames))
    return max(candidates) if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Derive rally_spans.json from ball/player/audio signals.")
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--ball-track", type=Path, default=None, help="ball_track.json artifact.")
    parser.add_argument("--tracks", type=Path, default=None, help="tracks.json artifact.")
    parser.add_argument("--audio-onsets", type=Path, default=None, help="audio_onsets.json artifact.")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Clip duration in seconds; inferred from the max signal timestamp if omitted.",
    )
    parser.add_argument("--pad-seconds", type=float, default=DEFAULT_PAD_SECONDS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    duration_s = args.duration_s
    if duration_s is None:
        duration_s = _infer_duration_s(ball_track_path=args.ball_track, tracks_path=args.tracks)
    if duration_s is None:
        parser.error("--duration-s must be provided when it cannot be inferred from --ball-track/--tracks")

    artifact = build_rally_spans_artifact_from_paths(
        clip_id=args.clip_id,
        duration_s=duration_s,
        ball_track_path=args.ball_track,
        tracks_path=args.tracks,
        audio_onsets_path=args.audio_onsets,
        pad_seconds=args.pad_seconds,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
