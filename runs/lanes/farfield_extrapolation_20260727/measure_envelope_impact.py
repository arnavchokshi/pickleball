#!/usr/bin/env python3
"""Count emitted 3D ball positions that fall outside their clip's calibrated envelope.

Read-only. Walks a tree for run directories that hold both a
``court_calibration.json`` and a solved ball track, builds each clip's
calibrated image envelope from the calibration's own correspondences, and
counts how many frames that emitted a ``world_xyz`` did so from a pixel the
camera model was never fit at.

    python3 runs/lanes/farfield_extrapolation_20260727/measure_envelope_impact.py \
        --root /path/to/repo --out runs/lanes/farfield_extrapolation_20260727/impact.json

Nothing is written into any scanned run directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from threed.racketsport.ball_arc_solver import _project_world_point  # noqa: E402
from threed.racketsport.calibration_extrapolation import (  # noqa: E402
    calibrated_image_envelope,
    evaluate_ball_track_extrapolation,
)

TRACK_NAMES = (
    "ball_track_arc_solved.json",
    "ball_track_physics_filled.json",
    "ball_track.json",
)


def _load(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _frames(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        frames = payload.get("frames")
        if isinstance(frames, list):
            return [frame for frame in frames if isinstance(frame, dict)]
    return []


def _reprojected_frames(
    frames: list[dict[str, Any]], calibration: dict[str, Any]
) -> list[dict[str, Any]]:
    """Re-key each emitted position to the pixel it actually projects to.

    The stored ``xy`` is the detection, and bridged/physics-filled frames carry
    the sentinel ``[0.0, 0.0]`` (or no ``xy`` at all) while still emitting a
    ``world_xyz``. ``[0.0, 0.0]`` sits exactly on the frame corner, so trusting
    it would invent far-field positions that were never observed there. The
    question this sweep asks is "where in the image does this EMITTED position
    live", so every frame is re-keyed to the projection of its own
    ``world_xyz``.
    """

    out: list[dict[str, Any]] = []
    for frame in frames:
        world = frame.get("world_xyz")
        if not isinstance(world, (list, tuple)) or len(world) != 3:
            out.append({})
            continue
        try:
            pixel = _project_world_point(
                calibration, (float(world[0]), float(world[1]), float(world[2]))
            )
        except (TypeError, ValueError, IndexError, KeyError, ZeroDivisionError):
            out.append({})
            continue
        out.append({"world_xyz": list(world), "xy": [float(pixel[0]), float(pixel[1])]})
    return out


def scan(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for calibration_path in sorted(root.rglob("court_calibration.json")):
        run_dir = calibration_path.parent
        calibration = _load(calibration_path)
        if not isinstance(calibration, dict):
            continue
        envelope = calibrated_image_envelope(calibration)
        for track_name in TRACK_NAMES:
            track_path = run_dir / track_name
            if not track_path.exists():
                continue
            payload = _load(track_path)
            frames = _frames(payload)
            if not frames:
                continue
            clip_id = payload.get("clip_id") if isinstance(payload, dict) else None
            report = evaluate_ball_track_extrapolation(
                _reprojected_frames(frames, calibration), calibration, envelope=envelope
            )
            summary = report["summary"]
            if summary["emitted_position_count"] == 0:
                continue
            radii = [entry["radius_pct_of_half_diagonal"] for entry in report["frames"]]
            rows.append(
                {
                    "run_dir": str(run_dir.relative_to(root)),
                    "clip_id": str(clip_id) if clip_id else run_dir.name,
                    "track": track_name,
                    "calibrated_radius_pct": (
                        None
                        if envelope is None
                        else round(envelope.radius_pct(envelope.calibrated_radius_px), 2)
                    ),
                    "correspondence_count": (
                        None if envelope is None else envelope.correspondence_count
                    ),
                    "emitted_position_count": summary["emitted_position_count"],
                    "evaluated_frame_count": summary["evaluated_frame_count"],
                    "extrapolated_frame_count": summary["extrapolated_frame_count"],
                    "far_extrapolated_frame_count": summary["far_extrapolated_frame_count"],
                    "max_extrapolated_radius_pct": round(max(radii), 2) if radii else None,
                }
            )

    by_clip: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "run_count": 0,
            "emitted_position_count": 0,
            "evaluated_frame_count": 0,
            "extrapolated_frame_count": 0,
            "far_extrapolated_frame_count": 0,
            "calibrated_radius_pct": [],
            "max_extrapolated_radius_pct": None,
        }
    )
    for row in rows:
        entry = by_clip[row["clip_id"]]
        entry["run_count"] += 1
        for key in (
            "emitted_position_count",
            "evaluated_frame_count",
            "extrapolated_frame_count",
            "far_extrapolated_frame_count",
        ):
            entry[key] += row[key]
        if row["calibrated_radius_pct"] is not None:
            entry["calibrated_radius_pct"].append(row["calibrated_radius_pct"])
        if row["max_extrapolated_radius_pct"] is not None:
            current = entry["max_extrapolated_radius_pct"]
            entry["max_extrapolated_radius_pct"] = (
                row["max_extrapolated_radius_pct"]
                if current is None
                else max(current, row["max_extrapolated_radius_pct"])
            )
    for entry in by_clip.values():
        radii = entry.pop("calibrated_radius_pct")
        entry["calibrated_radius_pct_min"] = round(min(radii), 2) if radii else None
        entry["calibrated_radius_pct_max"] = round(max(radii), 2) if radii else None

    totals = {
        "run_track_count": len(rows),
        "clip_count": len(by_clip),
        "emitted_position_count": sum(row["emitted_position_count"] for row in rows),
        "evaluated_frame_count": sum(row["evaluated_frame_count"] for row in rows),
        "extrapolated_frame_count": sum(row["extrapolated_frame_count"] for row in rows),
        "far_extrapolated_frame_count": sum(
            row["far_extrapolated_frame_count"] for row in rows
        ),
    }
    evaluated = totals["evaluated_frame_count"] or 1
    totals["extrapolated_fraction_of_evaluated"] = round(
        totals["extrapolated_frame_count"] / evaluated, 6
    )
    return {
        "policy": "calibration_extrapolation_v1",
        "root": str(root),
        "totals": totals,
        "by_clip": {key: by_clip[key] for key in sorted(by_clip)},
        "rows": sorted(rows, key=lambda row: row["run_dir"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    report = scan(Path(args.root).resolve())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
    totals = report["totals"]
    print(
        f"{totals['clip_count']} clips / {totals['run_track_count']} tracks: "
        f"{totals['extrapolated_frame_count']} of {totals['evaluated_frame_count']} "
        f"emitted 3D ball positions are outside the calibrated envelope "
        f"({100.0 * totals['extrapolated_fraction_of_evaluated']:.1f}%), "
        f"{totals['far_extrapolated_frame_count']} of them far outside."
    )
    for clip, entry in report["by_clip"].items():
        if entry["extrapolated_frame_count"] == 0:
            continue
        print(
            f"  {clip}: {entry['extrapolated_frame_count']}/{entry['evaluated_frame_count']} "
            f"extrapolated ({entry['far_extrapolated_frame_count']} far), "
            f"calibrated to {entry['calibrated_radius_pct_max']}%, "
            f"worst emitted at {entry['max_extrapolated_radius_pct']}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
