#!/usr/bin/env python3
"""Read-only feasibility probe for one_world_v1 baseline metrics.

The probe never mutates its run directory.  It computes metrics only where an
exact source frame exists; it does not interpolate BODY or ball observations.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


WRIST_INDICES = (9, 10)
# Repository-wide solver constant (threed/racketsport/ball_arc_solver.py:30).
BALL_RADIUS_M = 0.0371
WRIST_VOLUME_RADIUS_M = 0.12


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _frame_id(frame: dict[str, Any], fps: float, fallback: int | None = None) -> int:
    if frame.get("frame_idx") is not None:
        return int(frame["frame_idx"])
    if frame.get("frame") is not None:
        return int(frame["frame"])
    if frame.get("t") is not None:
        return int(round(float(frame["t"]) * fps))
    if fallback is not None:
        return fallback
    raise ValueError("frame has no frame_idx, frame, or t")


def _nearest_rank(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty list")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _player_frame_map(payload: dict[str, Any], fps: float) -> dict[int, dict[int, dict[str, Any]]]:
    return {
        int(player["id"]): {
            _frame_id(frame, fps): frame
            for frame in player.get("frames", [])
            if isinstance(frame, dict)
        }
        for player in payload.get("players", [])
        if isinstance(player, dict)
    }


def _ball_frame_map(payload: dict[str, Any], fps: float) -> dict[int, dict[str, Any]]:
    return {
        _frame_id(frame, fps, index): frame
        for index, frame in enumerate(payload.get("frames", []))
        if isinstance(frame, dict)
    }


def _contact_metric(run_dir: Path, fps: float) -> dict[str, Any]:
    events = _load(run_dir / "contact_windows.json").get("events", [])
    smpl = _load(run_dir / "smpl_motion.json")
    arc_path = run_dir / "ball_track_arc_solved.json"
    ball_path = arc_path if arc_path.exists() else run_dir / "ball_track.json"
    ball = _load(ball_path)
    body_by_player = _player_frame_map(smpl, fps)
    ball_by_frame = _ball_frame_map(ball, fps)

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("type") != "contact":
            continue
        frame_id = int(event["frame"])
        player_id = event.get("player_id")
        body_frame = body_by_player.get(int(player_id), {}).get(frame_id) if player_id is not None else None
        ball_frame = ball_by_frame.get(frame_id)
        reasons: list[str] = []
        if player_id is None:
            reasons.append("missing_player_id")
        if body_frame is None:
            reasons.append("missing_exact_body_frame")
        if ball_frame is None or ball_frame.get("world_xyz") is None:
            reasons.append("missing_exact_ball_world_xyz")
        joints = body_frame.get("joints_world", []) if body_frame else []
        joint_conf = body_frame.get("joint_conf", []) if body_frame else []
        if len(joints) <= max(WRIST_INDICES) or len(joint_conf) <= max(WRIST_INDICES):
            reasons.append("missing_body17_wrists_or_confidence")
        if reasons:
            missing.append({"event_index": event_index, "frame": frame_id, "reasons": reasons})
            continue

        ball_xyz = [float(value) for value in ball_frame["world_xyz"]]
        distances = [math.dist(ball_xyz, [float(value) for value in joints[index]]) for index in WRIST_INDICES]
        winning_local = min(range(len(distances)), key=distances.__getitem__)
        wrist_index = WRIST_INDICES[winning_local]
        center_distance = distances[winning_local]
        rows.append(
            {
                "event_index": event_index,
                "frame": frame_id,
                "t": float(event["t"]),
                "player_id": int(player_id),
                "wrist_index": wrist_index,
                "ball_confidence": float(ball_frame.get("conf", 0.0)),
                "wrist_joint_confidence": float(joint_conf[wrist_index]),
                "event_confidence": float(event["confidence"]),
                "center_distance_m": center_distance,
                "wrist_volume_residual_m": max(
                    0.0,
                    center_distance - WRIST_VOLUME_RADIUS_M - BALL_RADIUS_M,
                ),
            }
        )

    center = [row["center_distance_m"] for row in rows]
    volume = [row["wrist_volume_residual_m"] for row in rows]
    return {
        "definition": (
            "Exact contact_windows.frame join; nearest BODY_17 wrist (indices 9/10) for the "
            "declared hitter; center distance and distance outside a 0.12m wrist uncertainty "
            "sphere expanded by the repository's 0.0371m ball radius; no interpolation."
        ),
        "ball_world_source": ball_path.name,
        "event_count": sum(isinstance(event, dict) and event.get("type") == "contact" for event in events),
        "computable_event_count": len(rows),
        "missing_event_count": len(missing),
        "center_distance_m": {
            "median": statistics.median(center) if center else None,
            "p90_nearest_rank": _nearest_rank(center, 0.90) if center else None,
        },
        "wrist_volume_residual_m": {
            "median": statistics.median(volume) if volume else None,
            "p90_nearest_rank": _nearest_rank(volume, 0.90) if volume else None,
        },
        "rows": rows,
        "missing": missing,
    }


def _rally_frame_ids(run_dir: Path, tracks: dict[str, Any], fps: float) -> list[int]:
    path = run_dir / "rally_spans.json"
    spans = _load(path).get("spans", []) if path.exists() else tracks.get("rally_spans", [])
    frames: set[int] = set()
    for span in spans:
        start = int(math.ceil(float(span["t0"]) * fps - 1e-9))
        end_exclusive = int(math.ceil(float(span["t1"]) * fps - 1e-9))
        frames.update(range(start, end_exclusive))
    return sorted(frames)


def _coverage_metric(run_dir: Path, fps: float, threshold: float) -> dict[str, Any]:
    tracks = _load(run_dir / "tracks.json")
    placement_path = run_dir / "placement.json"
    placement = _load(placement_path) if placement_path.exists() else None
    arc_path = run_dir / "ball_track_arc_solved.json"
    ball_path = arc_path if arc_path.exists() else run_dir / "ball_track.json"
    ball = _load(ball_path)

    track_by_player = _player_frame_map(tracks, fps)
    placement_by_player = _player_frame_map(placement, fps) if placement is not None else track_by_player
    ball_by_frame = _ball_frame_map(ball, fps)
    expected_players = sorted(track_by_player)
    rally_frames = _rally_frame_ids(run_dir, tracks, fps)
    complete: list[int] = []
    player_failures = 0
    ball_failures = 0
    for frame_id in rally_frames:
        players_ok = all(
            frame_id in placement_by_player.get(player_id, {})
            and float(track_by_player.get(player_id, {}).get(frame_id, {}).get("conf", 0.0)) >= threshold
            for player_id in expected_players
        )
        ball_frame = ball_by_frame.get(frame_id, {})
        ball_ok = ball_frame.get("world_xyz") is not None and float(ball_frame.get("conf", 0.0)) >= threshold
        if not players_ok:
            player_failures += 1
        if not ball_ok:
            ball_failures += 1
        if players_ok and ball_ok:
            complete.append(frame_id)
    return {
        "definition": (
            "Rally frames are integer frames in each half-open [t0,t1) span. A player is "
            "world-placed when placement (fallback tracks) has the frame and tracks.conf >= threshold. "
            "Ball is world-placed when arc-solved (fallback ball_track) has world_xyz and conf >= threshold."
        ),
        "confidence_threshold": threshold,
        "player_coordinate_source": placement_path.name if placement is not None else "tracks.json",
        "player_confidence_source": "tracks.json players[].frames[].conf",
        "ball_world_source": ball_path.name,
        "ball_confidence_source": f"{ball_path.name} frames[].conf",
        "expected_player_ids": expected_players,
        "rally_frame_count": len(rally_frames),
        "simultaneously_world_placed_frame_count": len(complete),
        "coverage_fraction": len(complete) / len(rally_frames) if rally_frames else None,
        "frames_failing_player_requirement": player_failures,
        "frames_failing_ball_requirement": ball_failures,
        "complete_frame_ids": complete,
    }


def build_report(run_dir: Path, threshold: float) -> dict[str, Any]:
    tracks = _load(run_dir / "tracks.json")
    fps = float(tracks["fps"])
    return {
        "schema_version": 1,
        "artifact_type": "one_world_v1_baseline_feasibility_probe",
        "run_dir": str(run_dir.resolve()),
        "fps": fps,
        "preview_band": True,
        "verified": 0,
        "raw_inputs_mutated": False,
        "metrics": {
            "ball_at_contact_to_hitter_wrist": _contact_metric(run_dir, fps),
            "world_coverage": _coverage_metric(run_dir, fps, threshold),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.confidence_threshold <= 1.0:
        parser.error("--confidence-threshold must be in [0,1]")
    report = build_report(args.run_dir, args.confidence_threshold)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
