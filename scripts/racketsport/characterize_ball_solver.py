#!/usr/bin/env python3
"""Characterize the CURRENT fail-closed ball 3D arc solver over solved clips.

Measurement-only harness (VERIFIED=0 stays binding; this is not a promotion
instrument). Default mode reads existing ``ball_track_arc_solved.json``
artifacts; ``--solve`` opt-in re-runs the default ball arc chain first (slow).
Outputs: ``manifest.json`` (pinned inputs: path + sha256 + solver config
echo), ``report.json`` (deterministic bytes given the same manifest), and
``REPORT.md`` (human summary).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_solver_characterization import (  # noqa: E402
    discover_clip_inputs,
    run_characterization,
    write_characterization_outputs,
)


def _parse_named_paths(entries: list[str], *, flag: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for entry in entries:
        name, separator, value = entry.partition("=")
        if not separator or not name or not value:
            raise ValueError(f"{flag} expects NAME=DIR (or NAME=PATH), got: {entry!r}")
        parsed[name] = Path(value)
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the deterministic characterization report for the current "
            "fail-closed ball 3D arc solver (measurement only, VERIFIED=0)."
        )
    )
    parser.add_argument(
        "--clip",
        action="append",
        default=[],
        metavar="NAME=DIR",
        required=True,
        help="Clip name and artifact directory holding ball_track_arc_solved.json et al. Repeatable.",
    )
    parser.add_argument(
        "--calibration",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Optional per-clip court_calibration.json override (e.g. a sha-matched copy of the "
            "calibration the solve consumed). Residual recompute only runs when the file's sha256 "
            "matches the clip's ball_chain_manifest.json record."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for manifest.json / report.json / REPORT.md.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Root for relativizing manifest paths (default: repo root).")
    parser.add_argument("--label", default="ball_solver_characterization", help="Deterministic label recorded in manifest and report.")
    parser.add_argument(
        "--solve",
        action="store_true",
        help="Opt-in: re-run the default ball arc chain per clip before characterizing (slow).",
    )
    parser.add_argument(
        "--ball-track",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="--solve only: per-clip ball_track.json input.",
    )
    parser.add_argument(
        "--net-plane",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="--solve only: optional per-clip net_plane.json input.",
    )
    parser.add_argument(
        "--frame-times",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="--solve only: optional per-clip frame_times.json input.",
    )
    parser.add_argument("--ball-type", default="outdoor", help="--solve only: physics ball type (default: outdoor).")
    parser.add_argument(
        "--enable-joint-anchor-search",
        action="store_true",
        help=(
            "--solve only: enable the dormant joint-anchor search evidence source for the re-solve "
            "(candidate-only, default off; measurement only, never a promotion)."
        ),
    )
    parser.add_argument(
        "--enable-ukf-fallback",
        action="store_true",
        help=(
            "--solve only: enable the dormant conservative UKF fallback sidecar for the re-solve "
            "(physics_interpolated trust-band sidecar, default off; measurement only, never counted as accepted)."
        ),
    )
    args = parser.parse_args(argv)

    try:
        if (args.enable_joint_anchor_search or args.enable_ukf_fallback) and not args.solve:
            raise ValueError(
                "--enable-joint-anchor-search/--enable-ukf-fallback require --solve "
                "(read mode never re-solves; experiment flags cannot alter read-mode reports)"
            )

        clips = _parse_named_paths(args.clip, flag="--clip")
        calibrations = _parse_named_paths(args.calibration, flag="--calibration")
        ball_tracks = _parse_named_paths(args.ball_track, flag="--ball-track")
        net_planes = _parse_named_paths(args.net_plane, flag="--net-plane")
        frame_times = _parse_named_paths(args.frame_times, flag="--frame-times")

        if args.solve:
            clips = _solve_clips(
                clips,
                calibrations=calibrations,
                ball_tracks=ball_tracks,
                net_planes=net_planes,
                frame_times=frame_times,
                ball_type=args.ball_type,
                out_dir=args.out_dir,
                enable_joint_anchor_search=args.enable_joint_anchor_search,
                enable_ukf_fallback=args.enable_ukf_fallback,
            )

        clip_inputs = [
            discover_clip_inputs(name, directory, calibration_override=calibrations.get(name))
            for name, directory in sorted(clips.items())
        ]
        result = run_characterization(clip_inputs, root=args.root, label=args.label)
        paths = write_characterization_outputs(
            out_dir=args.out_dir, manifest=result["manifest"], report=result["report"]
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    pooled = result["report"]["pooled"]
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "report": paths["report"].name,
                "manifest_sha256": result["report"]["manifest_sha256"],
                "pooled": {
                    "clip_count": pooled["clip_count"],
                    "skipped_clip_count": pooled["skipped_clip_count"],
                    "accepted_3d_coverage_fraction": pooled["coverage"]["accepted_3d_coverage_fraction"],
                    "segment_verdict_counts": pooled["segment_verdict_counts"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _solve_clips(
    clips: dict[str, Path],
    *,
    calibrations: dict[str, Path],
    ball_tracks: dict[str, Path],
    net_planes: dict[str, Path],
    frame_times: dict[str, Path],
    ball_type: str,
    out_dir: Path,
    enable_joint_anchor_search: bool = False,
    enable_ukf_fallback: bool = False,
) -> dict[str, Path]:
    """Re-run the default chain per clip; returns the fresh solve directories."""

    from threed.racketsport.ball_arc_chain import run_default_ball_arc_chain

    solved: dict[str, Path] = {}
    for name in sorted(clips):
        ball_track = ball_tracks.get(name)
        calibration = calibrations.get(name)
        if ball_track is None or calibration is None:
            raise ValueError(
                f"--solve requires --ball-track {name}=PATH and --calibration {name}=PATH"
            )
        solve_dir = out_dir / "solve" / name
        run_default_ball_arc_chain(
            clip=name,
            ball_track_path=ball_track,
            court_calibration_path=calibration,
            out_dir=solve_dir,
            net_plane_path=net_planes.get(name),
            frame_times_path=frame_times.get(name),
            ball_type=ball_type,
            enable_joint_anchor_search=enable_joint_anchor_search,
            enable_ukf_fallback=enable_ukf_fallback,
        )
        solved[name] = solve_dir
    return solved


if __name__ == "__main__":
    raise SystemExit(main())
