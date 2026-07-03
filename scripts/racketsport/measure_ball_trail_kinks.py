#!/usr/bin/env python3
"""Measure horizontal-velocity direction reversals ("kinks") in a ball stream.

The owner's law for the rendered 3D ball trail: between two confident ball
positions there is exactly one parabola, and direction changes only happen at
selected events (`events_selected.json`, BALL-ARC-SOLVER output). A true
parabola segment never reverses its horizontal (x/y) velocity direction --
only gravity acts vertically. Any within-segment sign change in horizontal
velocity is therefore a kink: a non-analytic (raw/interpolated) sighting that
leaked into what should be a pure arc-evaluated trail.

This tool is read-only measurement/reporting. It does not mutate any
artifact and is not a promotion gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_ball_trail_kink_report"


def extract_ball_frames(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the per-frame ball list from a world- or ball-track-shaped payload.

    Accepts `virtual_world.json`/`confidence_gated_world.json` (`ball.frames`)
    or `ball_track_arc_solved.json`/`ball_track_physics_filled.json`/
    `ball_track.json` (top-level `frames`).
    """

    ball = payload.get("ball")
    if isinstance(ball, Mapping) and isinstance(ball.get("frames"), list):
        return [frame for frame in ball["frames"] if isinstance(frame, Mapping)]
    frames = payload.get("frames")
    if isinstance(frames, list):
        return [frame for frame in frames if isinstance(frame, Mapping)]
    raise ValueError("payload has neither ball.frames nor a top-level frames list")


def segment_time_bounds(events_selected: Mapping[str, Any]) -> list[tuple[float, float]]:
    """Consecutive (t0, t1) pairs between selected events, sorted by time.

    Matches how the arc solver itself bounds one free-flight segment between
    each pair of consecutive selected anchors (`ball_arc_solver._fit_segments_from_anchors`).
    """

    selected = events_selected.get("selected")
    if not isinstance(selected, list):
        return []
    times = sorted(
        float(item["t"])
        for item in selected
        if isinstance(item, Mapping) and item.get("t") is not None
    )
    return list(zip(times, times[1:]))


def horizontal_velocity_sign_changes(points: Sequence[Sequence[float]]) -> int:
    """Count direction reversals of horizontal (x, y) velocity across points.

    Near-zero displacements (< 1mm) are ignored as noise rather than treated
    as a sign change, so a momentarily near-stationary ball does not produce
    spurious kinks.
    """

    deltas = [(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])) for a, b in zip(points, points[1:])]
    changes = 0
    for axis in (0, 1):
        previous_sign = 0
        for delta in deltas:
            value = delta[axis]
            if abs(value) < 1e-3:
                continue
            sign = 1 if value > 0 else -1
            if previous_sign != 0 and sign != previous_sign:
                changes += 1
            previous_sign = sign
    return changes


def _assign_segment_index(t: float, segments: Sequence[tuple[float, float]], *, eps_s: float) -> int | None:
    """Assign one frame time to exactly one segment, mirroring `ball_arc_solver._segment_for_time`.

    A frame exactly at a shared anchor between two segments (a bounce or
    contact -- a legitimate place for velocity to change direction) must
    belong to only one segment's point list. Counting it in both would
    manufacture a false "kink" out of an owner-sanctioned event boundary
    instead of a real within-segment reversal. Ties break toward the segment
    whose midpoint is nearest `t`, exactly like the production solver.
    """

    candidates = [
        index for index, (t0, t1) in enumerate(segments) if t0 - eps_s <= t <= t1 + eps_s
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda index: abs((segments[index][0] + segments[index][1]) / 2.0 - t))


def kink_report(
    frames: Sequence[Mapping[str, Any]],
    segments: Sequence[tuple[float, float]],
    *,
    eps_s: float = 1e-6,
) -> dict[str, Any]:
    ordered_frames = sorted(
        (frame for frame in frames if frame.get("world_xyz") is not None and frame.get("t") is not None),
        key=lambda frame: float(frame["t"]),
    )
    points_by_segment: list[list[Sequence[float]]] = [[] for _ in segments]
    for frame in ordered_frames:
        index = _assign_segment_index(float(frame["t"]), segments, eps_s=eps_s)
        if index is not None:
            points_by_segment[index].append(frame["world_xyz"])

    segment_reports = []
    total_kinks = 0
    total_points = 0
    for (t0, t1), points in zip(segments, points_by_segment):
        kinks = horizontal_velocity_sign_changes(points)
        total_kinks += kinks
        total_points += len(points)
        segment_reports.append(
            {
                "t0": round(t0, 6),
                "t1": round(t1, 6),
                "point_count": len(points),
                "kink_count": kinks,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "segment_count": len(segments),
        "total_point_count": total_points,
        "total_kink_count": total_kinks,
        "segments": segment_reports,
    }


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} ({path}) must contain a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ball-stream",
        type=Path,
        required=True,
        help="World artifact (ball.frames) or ball-track artifact (frames) to measure.",
    )
    parser.add_argument(
        "--events-selected",
        type=Path,
        required=True,
        help="events_selected.json (BALL-ARC-SOLVER) providing segment boundaries.",
    )
    parser.add_argument("--out", type=Path, help="Optional path to write the JSON report.")
    args = parser.parse_args(argv)

    try:
        ball_payload = _read_json_object(args.ball_stream, "ball-stream")
        events_selected = _read_json_object(args.events_selected, "events-selected")
        frames = extract_ball_frames(ball_payload)
        segments = segment_time_bounds(events_selected)
        report = kink_report(frames, segments)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: kink measurement failed: {exc}", file=sys.stderr)
        return 1

    report["inputs"] = {"ball_stream": str(args.ball_stream), "events_selected": str(args.events_selected)}
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
