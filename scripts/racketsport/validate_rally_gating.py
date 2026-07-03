#!/usr/bin/env python3
"""Validate rally-span gating: dead-time fraction + zero-missed-event safety check.

Standalone CLI over the 4 accepted eval clips (Burlington/Wolverine/Outdoor/
Indoor). For each clip it:

1. Derives rally spans from the already-computed ``ball_track.json`` /
   ``tracks.json`` / ``audio_onsets.json`` artifacts (via
   ``threed.racketsport.rally_gating``).
2. Loads the clip's human-reviewed ``contact_windows.json`` events
   (``sources.human_review > 0``) as the reviewed-event ground truth and
   checks that none of them fall outside a span.
3. Reports the dead-time fraction over the reviewed excerpt's own duration
   (derived from ``ball_track.json``'s own frame span, which matches the
   reviewed-event time range for all 4 clips).

Because the 4 reviewed excerpts are themselves single pre-trimmed rallies
(no between-point dead time by construction -- see RUNTIME_BUDGET.md /
CUTS_SPEC.md for the full explanation), step 3 is expected to show ~0% dead
time on the real clips. To demonstrate the mechanism at the scale it is
actually meant for (a full capture with multiple points and real between-point
pauses), this script also builds a clearly-labeled **synthetic multi-rally
composition**: the real per-clip signal is repeated N times with a fixed
silent gap between reps (representing a serve-prep + changeover pause), and
the same gating code (unmodified) is run over that composed timeline. This is
explicitly a demonstration, not a claim about real match statistics.

Usage:

    python3 scripts/racketsport/validate_rally_gating.py --out-dir runs/<run_dir>/rally_gating_validation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.rally_gating import (
    build_rally_spans_artifact,
    dead_time_fraction,
    derive_rally_spans,
    missed_events,
    onsets_from_audio_onsets,
)

EVAL_CLIPS_DIR = ROOT / "runs/eval0/prototype_gate_h100_v2"

CLIP_IDS = [
    "burlington_gold_0300_low_steep_corner",
    "wolverine_mixed_0200_mid_steep_corner",
    "outdoor_webcam_iynbd_1500_long_high_baseline",
    "indoor_doubles_fwuks_0500_long_mid_baseline",
]


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def reviewed_contact_event_times(contact_windows: dict[str, Any]) -> list[float]:
    """Reviewed (``sources.human_review > 0``) contact-event timestamps, seconds."""

    events = contact_windows.get("events", [])
    return [
        float(event["t"])
        for event in events
        if float(event.get("sources", {}).get("human_review", 0.0) or 0.0) > 0.0
    ]


def validate_real_clip(clip_id: str) -> dict[str, Any]:
    clip_dir = EVAL_CLIPS_DIR / clip_id
    ball_track = _load(clip_dir / "ball_track.json")
    tracks_path = clip_dir / "tracks.json"
    audio_path = clip_dir / "audio_onsets.json"
    contact_windows = _load(clip_dir / "contact_windows.json")

    frames = ball_track["frames"]
    duration_s = max(float(f["t"]) for f in frames)  # authoritative: matches ball_track's own frame span

    artifact = build_rally_spans_artifact(
        clip_id=clip_id,
        duration_s=duration_s,
        ball_track=ball_track,
        tracks=_load(tracks_path) if tracks_path.exists() else None,
        audio_onsets=_load(audio_path) if audio_path.exists() else None,
        ball_track_path=str(clip_dir / "ball_track.json"),
        tracks_path=str(tracks_path) if tracks_path.exists() else None,
        audio_onsets_path=str(audio_path) if audio_path.exists() else None,
    )

    reviewed_events = reviewed_contact_event_times(contact_windows)
    missed = missed_events(reviewed_events, artifact["spans"])

    return {
        "clip_id": clip_id,
        "duration_s": duration_s,
        "spans": artifact["spans"],
        "dead_time_fraction": artifact["dead_time_fraction"],
        "signals_used": artifact["signals_used"],
        "reviewed_event_count": len(reviewed_events),
        "reviewed_event_times_s": reviewed_events,
        "missed_event_count": len(missed),
        "missed_event_times_s": missed,
        "zero_missed_events": len(missed) == 0,
    }


def build_synthetic_multi_rally(
    clip_id: str,
    *,
    repeats: int = 3,
    gap_seconds: float = 15.0,
) -> dict[str, Any]:
    """Tile one real clip's cheap signals + reviewed events `repeats` times with
    `gap_seconds` of silence between reps, then re-run the exact same gating
    code. Clearly a synthetic composition of real per-rally signal, built only
    to demonstrate dead-time cutting at multi-point scale; not a real-match
    statistic.
    """

    clip_dir = EVAL_CLIPS_DIR / clip_id
    ball_track = _load(clip_dir / "ball_track.json")
    tracks_path = clip_dir / "tracks.json"
    audio_path = clip_dir / "audio_onsets.json"
    contact_windows = _load(clip_dir / "contact_windows.json")

    frames = ball_track["frames"]
    rally_duration_s = max(float(f["t"]) for f in frames)
    tracks = _load(tracks_path) if tracks_path.exists() else {"players": []}
    audio_onsets = onsets_from_audio_onsets(_load(audio_path)) if audio_path.exists() else []
    reviewed_events = reviewed_contact_event_times(contact_windows)

    period = rally_duration_s + gap_seconds
    tiled_ball_frames: list[dict[str, Any]] = []
    tiled_players: list[dict[str, Any]] = [
        {**player, "frames": []} for player in tracks.get("players", [])
    ]
    tiled_audio_onsets: list[float] = []
    tiled_reviewed_events: list[float] = []

    for rep in range(repeats):
        offset = rep * period
        for frame in frames:
            if float(frame["t"]) > rally_duration_s:
                continue
            tiled_ball_frames.append({**frame, "t": float(frame["t"]) + offset})
        for src_player, dst_player in zip(tracks.get("players", []), tiled_players):
            for frame in src_player.get("frames", []):
                if float(frame["t"]) > rally_duration_s:
                    continue
                dst_player["frames"].append({**frame, "t": float(frame["t"]) + offset})
        for onset in audio_onsets:
            if onset <= rally_duration_s:
                tiled_audio_onsets.append(onset + offset)
        for event_t in reviewed_events:
            tiled_reviewed_events.append(event_t + offset)

    total_duration_s = repeats * rally_duration_s + (repeats - 1) * gap_seconds

    spans = derive_rally_spans(
        ball_frames=tiled_ball_frames,
        players=tiled_players,
        audio_onsets=tiled_audio_onsets,
        duration_s=total_duration_s,
    )
    dtf = dead_time_fraction(spans, total_duration_s)
    missed = missed_events(tiled_reviewed_events, spans)

    return {
        "clip_id": clip_id,
        "kind": "synthetic_multi_rally_demo",
        "not_a_real_match_statistic": True,
        "construction": (
            f"real single-rally signal ({rally_duration_s:.2f}s) repeated {repeats}x "
            f"with {gap_seconds:.1f}s silent gaps between reps"
        ),
        "total_duration_s": total_duration_s,
        "span_count": len(spans),
        "spans": spans,
        "dead_time_fraction": dtf,
        "reviewed_event_count": len(tiled_reviewed_events),
        "missed_event_count": len(missed),
        "missed_event_times_s": missed,
        "zero_missed_events": len(missed) == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate rally-span gating against reviewed contact events.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--synthetic-repeats", type=int, default=3)
    parser.add_argument("--synthetic-gap-seconds", type=float, default=15.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    real_results = [validate_real_clip(clip_id) for clip_id in CLIP_IDS]
    synthetic_results = [
        build_synthetic_multi_rally(
            clip_id, repeats=args.synthetic_repeats, gap_seconds=args.synthetic_gap_seconds
        )
        for clip_id in CLIP_IDS
    ]

    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_rally_gating_validation",
        "real_clips": real_results,
        "synthetic_multi_rally_demo": synthetic_results,
        "all_real_clips_zero_missed_events": all(r["zero_missed_events"] for r in real_results),
        "all_synthetic_zero_missed_events": all(r["zero_missed_events"] for r in synthetic_results),
        "total_reviewed_events": sum(r["reviewed_event_count"] for r in real_results),
        "total_missed_events": sum(r["missed_event_count"] for r in real_results),
    }

    out_path = args.out_dir / "validation_summary.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_real_clips_zero_missed_events"] and summary["all_synthetic_zero_missed_events"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
