#!/usr/bin/env python3
"""Convert a VNDetectTrajectoriesRequest spike harness JSON into ball_track.json.

W0-BALL-SPIKE (see NORTH_STAR_ROADMAP.md): rung-1 spike of Apple's native
`VNDetectTrajectoriesRequest` as an on-device ball-tracking candidate. The
Swift harness at `spikes/vn_trajectories/` writes RAW per-frame candidate
trajectory emissions (see `spikes/vn_trajectories/Sources/VNTrajectorySpike/JSONModels.swift`
for the schema and the reasoning behind keeping it raw rather than
pre-collapsed). This script reduces that raw multi-candidate-per-frame
output down to the single-point-per-frame `threed.racketsport.schemas.BallTrack`
shape used everywhere else in the repo (`ball_track.json`), so it can be
scored with the same `scripts/racketsport/benchmark_ball_tracks_against_cvat.py`
/ `scripts/racketsport/sweep_ball_track_thresholds_against_cvat.py` tooling
used for every other BALL candidate.

Two real quirks of VNDetectTrajectoriesRequest observed while building this,
both handled below (see `_dedupe_first_emission_per_uuid` docstring for why):

1. `request.results` is NOT "only new detections this frame" -- Vision keeps
   already-found trajectory observations (same `uuid`, byte-identical
   `detectedPoints`/`timeRange`/`confidence`) in the returned array for
   several subsequent `perform()` calls after they were first found. Naively
   treating every emission as a fresh window double- (or 5x-, 6x-) counts
   the same physical trajectory and, worse, would compute wrong frame
   indices for the stale copies (their `emitted_at_frame_index` keeps
   advancing even though the trajectory itself stopped growing).
2. Multiple simultaneous candidate trajectories are extremely common with a
   wide radius filter (moving paddles/limbs also fit short parabolic
   segments) -- this script resolves one point per output frame by taking
   the highest-confidence candidate covering that frame, and separately
   reports how much candidate contention there was so that isn't hidden.
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

from threed.racketsport.schemas import BallTrack  # noqa: E402


ARTIFACT_TYPE = "racketsport_vn_trajectories_conversion"


def convert_vn_trajectories_to_ball_track(
    *,
    harness_json: dict[str, Any],
    confidence_threshold: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (ball_track_payload, conversion_metadata).

    `confidence_threshold` gates the `visible` flag only (matching the
    convention used by `threed/racketsport/ball_threshold_sweep.py`'s
    `_write_ball_track_threshold_track`: confidence is preserved unthresholded
    in `conf`, and `visible` is a downstream-tunable flag on top of it, so
    `sweep_ball_track_thresholds_against_cvat.py` can sweep further
    thresholds against this track directly without re-running the harness).
    """

    if harness_json.get("status") == "BLOCKED":
        raise ValueError(
            "harness JSON is a blocker report, not a completed run: "
            f"{harness_json.get('blocked_reason')}: {harness_json.get('detail')}"
        )
    if harness_json.get("artifact_type") != "vn_trajectories_spike_raw":
        raise ValueError(f"unexpected artifact_type: {harness_json.get('artifact_type')!r}")

    video = harness_json["video"]
    width = float(video["width"])
    height = float(video["height"])
    fps = float(video["fps"])
    frame_count = int(video["frame_count"])
    frame_pts_s: list[float] = [float(t) for t in harness_json.get("frame_pts_s", [])]

    raw_trajectories = harness_json.get("trajectories", [])
    deduped = _dedupe_first_emission_per_uuid(raw_trajectories)

    candidates_by_frame: dict[int, list[dict[str, Any]]] = {}
    for traj in deduped:
        confidence = float(traj["confidence"])
        uuid = traj["observation_uuid"]
        for point in traj["detected_points"]:
            frame_index = point.get("frame_index")
            if frame_index is None or not (0 <= frame_index < frame_count):
                continue
            x_px, y_px = _to_pixel_xy(point["x_norm"], point["y_norm"], width=width, height=height)
            candidates_by_frame.setdefault(frame_index, []).append(
                {
                    "uuid": uuid,
                    "confidence": confidence,
                    "xy": (x_px, y_px),
                }
            )

    frames: list[dict[str, Any]] = []
    visible_count = 0
    contested_frame_count = 0
    max_candidates_at_a_frame = 0
    for frame_index in range(frame_count):
        t = frame_pts_s[frame_index] if frame_index < len(frame_pts_s) else (frame_index / fps if fps > 0 else 0.0)
        candidates = candidates_by_frame.get(frame_index, [])
        max_candidates_at_a_frame = max(max_candidates_at_a_frame, len(candidates))
        if len(candidates) > 1:
            contested_frame_count += 1
        if not candidates:
            frames.append({"t": t, "xy": [0.0, 0.0], "conf": 0.0, "visible": False, "approx": False})
            continue
        best = max(candidates, key=lambda c: (c["confidence"], c["uuid"]))
        visible = best["confidence"] >= confidence_threshold
        if visible:
            visible_count += 1
        frames.append(
            {
                "t": t,
                "xy": [best["xy"][0], best["xy"][1]],
                "conf": best["confidence"],
                "visible": visible,
                "approx": False,
            }
        )

    ball_track_payload = {
        "schema_version": 1,
        "fps": fps,
        "source": "vn_trajectories",
        "frames": frames,
        "bounces": [],
    }
    BallTrack.model_validate(ball_track_payload)

    metadata = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": "TESTED-ON-REAL-DATA",
        "source_video": harness_json.get("source_video"),
        "request_config": harness_json.get("request_config"),
        "confidence_threshold": confidence_threshold,
        "frame_count": frame_count,
        "raw_emission_count": len(raw_trajectories),
        "deduped_trajectory_count": len(deduped),
        "candidate_frame_count": len(candidates_by_frame),
        "contested_frame_count": contested_frame_count,
        "max_simultaneous_candidates_at_a_frame": max_candidates_at_a_frame,
        "visible_frame_count": visible_count,
        "visible_frame_rate": _ratio(visible_count, frame_count),
        "not_ground_truth": True,
        "notes": [
            "candidates_by_frame is resolved to one point per frame by max(confidence), tie-broken by uuid string.",
            "Vision's bottom-left-origin normalized coordinates are converted to top-left-origin pixel coordinates here.",
            "VNDetectTrajectoriesRequest re-emits already-found trajectories (same uuid) on subsequent perform() calls; "
            "only the first emission per uuid is used (see module docstring).",
        ],
    }
    return ball_track_payload, metadata


def _dedupe_first_emission_per_uuid(raw_trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the first (lowest emitted_at_frame_index) emission per uuid.

    VNDetectTrajectoriesRequest keeps already-found trajectories in
    `request.results` for multiple subsequent `perform()` calls. Empirically
    (see spike harness runs), every re-emission of a given `uuid` carries
    byte-identical `confidence`/`time_range_*`/`detected_points` -- it is the
    same completed observation being reported again, not trajectory growth.
    Only the first emission's `detected_points[i].frame_index` values are
    correct (they are computed in the Swift harness as
    `emitted_at_frame_index - len(points) + 1 + i`, which is only valid at
    the moment the observation is newly completed; re-emissions keep the
    same points but an advancing `emitted_at_frame_index`, which would
    silently compute wrong frame indices if used).
    """

    first_by_uuid: dict[str, dict[str, Any]] = {}
    for traj in raw_trajectories:
        uuid = traj["observation_uuid"]
        current = first_by_uuid.get(uuid)
        if current is None or traj["emitted_at_frame_index"] < current["emitted_at_frame_index"]:
            first_by_uuid[uuid] = traj
    return list(first_by_uuid.values())


def _to_pixel_xy(x_norm: float, y_norm: float, *, width: float, height: float) -> tuple[float, float]:
    x_px = float(x_norm) * width
    y_px = (1.0 - float(y_norm)) * height
    return x_px, y_px


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / float(denominator)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness-json", type=Path, required=True, help="Raw harness output from the Swift spike CLI.")
    parser.add_argument("--out-ball-track", type=Path, required=True)
    parser.add_argument("--out-metadata", type=Path, default=None)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum Vision confidence for a frame's visible flag (default 0.0: keep raw recall, gate later via sweep).",
    )
    args = parser.parse_args()

    try:
        harness_json = json.loads(args.harness_json.read_text(encoding="utf-8"))
        ball_track_payload, metadata = convert_vn_trajectories_to_ball_track(
            harness_json=harness_json,
            confidence_threshold=args.confidence_threshold,
        )
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 2

    args.out_ball_track.parent.mkdir(parents=True, exist_ok=True)
    args.out_ball_track.write_text(json.dumps(ball_track_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_metadata_path = args.out_metadata or args.out_ball_track.with_name(
        args.out_ball_track.stem + "_conversion_metadata.json"
    )
    out_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    out_metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
