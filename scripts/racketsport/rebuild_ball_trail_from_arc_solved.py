#!/usr/bin/env python3
"""Rebuild ONLY the ball portion of staged world artifacts from an arc solution.

Some run directories were staged (hand-composed) before the BALL-ARC-SOLVER
overlay existed in `threed.racketsport.virtual_world`
(`apply_ball_track_arc_solved_overlay`), so their `ball_track_physics_filled.json`
and derived `confidence_gated_world.json` can still carry a raw/interpolated
`world_xyz` for frames the arc solver never confidently bounded between two
events. This script re-applies the overlay to those two artifacts in place --
and *only* the ball stream inside them. Player/skeleton/mesh/paddle data,
court placement, and confidence-band bookkeeping for already-correct frames
are left untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.confidence_gate import BAND_HIDDEN_NO_PREDICTION  # noqa: E402
from threed.racketsport.virtual_world import apply_ball_track_arc_solved_overlay  # noqa: E402


def _time_key(t: float) -> str:
    return f"{float(t):.6f}"


def rebuild_physics_filled(
    physics_filled: Mapping[str, Any],
    arc_solved: Mapping[str, Any],
) -> dict[str, Any]:
    merged = apply_ball_track_arc_solved_overlay(physics_filled, arc_solved)
    if merged is None:
        raise ValueError("physics_filled overlay unexpectedly produced no result")
    return dict(merged)


def rebuild_world_ball_field(
    world: Mapping[str, Any],
    rebuilt_physics_filled: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Patch only `world["ball"]["frames"][*]["world_xyz"]` (+ provenance for newly-hidden frames).

    Every other key of `world` -- including `players`, `paddles`, `court`,
    top-level `confidence_gate` counts, and every other ball-frame field
    (`xy`, `conf`, `visible`, `trust_band`, `render_only`, ...) -- passes
    through unchanged.
    """

    by_key: dict[str, Mapping[str, Any]] = {}
    for frame in rebuilt_physics_filled.get("frames", []):
        if not isinstance(frame, Mapping) or frame.get("t") is None:
            continue
        by_key[_time_key(float(frame["t"]))] = frame

    patched = dict(world)
    ball = patched.get("ball")
    if not isinstance(ball, Mapping):
        return patched, {"changed_frame_count": 0, "newly_hidden_frame_count": 0}
    ball = dict(ball)
    frames = list(ball.get("frames", []))
    changed_count = 0
    newly_hidden_count = 0
    new_frames = []
    for frame in frames:
        if not isinstance(frame, Mapping) or frame.get("t") is None:
            new_frames.append(frame)
            continue
        source = by_key.get(_time_key(float(frame["t"])))
        if source is None:
            new_frames.append(frame)
            continue
        new_world_xyz = source.get("world_xyz")
        if new_world_xyz == frame.get("world_xyz"):
            new_frames.append(frame)
            continue
        changed_count += 1
        updated = dict(frame)
        updated["world_xyz"] = list(new_world_xyz) if new_world_xyz is not None else None
        if new_world_xyz is None and frame.get("world_xyz") is not None:
            newly_hidden_count += 1
            provenance = updated.get("confidence_provenance")
            if isinstance(provenance, Mapping):
                updated["confidence_provenance"] = {
                    **provenance,
                    "band": BAND_HIDDEN_NO_PREDICTION,
                    "display_band": BAND_HIDDEN_NO_PREDICTION,
                    "predictor": "ball_arc_solver",
                    "predicted_sigma_m": None,
                }
        new_frames.append(updated)
    ball["frames"] = new_frames
    patched["ball"] = ball
    return patched, {"changed_frame_count": changed_count, "newly_hidden_frame_count": newly_hidden_count}


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} ({path}) must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-filled", type=Path, required=True, help="ball_track_physics_filled.json to rebuild.")
    parser.add_argument("--arc-solved", type=Path, required=True, help="ball_track_arc_solved.json (BALL-ARC-SOLVER output).")
    parser.add_argument(
        "--out-physics-filled",
        type=Path,
        help="Output path for the rebuilt physics-filled artifact. Defaults to overwriting --physics-filled.",
    )
    parser.add_argument("--world", type=Path, help="Optional confidence_gated_world.json / virtual_world.json to patch.")
    parser.add_argument(
        "--out-world",
        type=Path,
        help="Output path for the patched world artifact. Defaults to overwriting --world.",
    )
    args = parser.parse_args(argv)

    try:
        physics_filled = _read_json_object(args.physics_filled, "physics-filled")
        arc_solved = _read_json_object(args.arc_solved, "arc-solved")
        rebuilt_physics_filled = rebuild_physics_filled(physics_filled, arc_solved)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: ball-trail rebuild failed: {exc}", file=sys.stderr)
        return 1

    out_physics_filled = args.out_physics_filled or args.physics_filled
    _write_json(out_physics_filled, rebuilt_physics_filled)

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_trail_rebuild_report",
        "physics_filled_out": str(out_physics_filled),
        "arc_solved_overlay": rebuilt_physics_filled.get("arc_solved_overlay"),
    }

    if args.world is not None:
        try:
            world = _read_json_object(args.world, "world")
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: ball-trail rebuild failed: {exc}", file=sys.stderr)
            return 1
        patched_world, world_counts = rebuild_world_ball_field(world, rebuilt_physics_filled)
        out_world = args.out_world or args.world
        _write_json(out_world, patched_world)
        report["world_out"] = str(out_world)
        report["world_ball_field_changes"] = world_counts

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
