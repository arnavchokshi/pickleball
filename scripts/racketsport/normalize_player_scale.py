#!/usr/bin/env python3
"""Normalize skeleton3d per-player scale from track bboxes and court geometry."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.player_scale import (  # noqa: E402
    DEFAULT_ANTHROPOMETRIC_BAND_M,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_WINDOW_SPREAD_MAX_M,
    PlayerScaleError,
    estimate_player_metric_heights,
    normalize_skeleton_scale_payload,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Run directory containing skeleton3d.json, tracks.json, and court_calibration.json.")
    parser.add_argument("--skeleton", type=Path, help="Explicit skeleton3d.json path; defaults to --run-dir/skeleton3d.json.")
    parser.add_argument("--tracks", type=Path, help="Explicit tracks.json path; defaults to --run-dir/tracks.json.")
    parser.add_argument(
        "--court-calibration",
        type=Path,
        help="Explicit court_calibration.json path; defaults to --run-dir/court_calibration.json.",
    )
    parser.add_argument("--output-skeleton", type=Path, help="Output skeleton path; defaults to replacing --skeleton.")
    parser.add_argument("--estimates-out", type=Path, help="player_scale_estimates.json output path.")
    parser.add_argument("--report-out", type=Path, help="player_scale_normalization_report.json output path.")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--min-bbox-confidence", type=float, default=0.20)
    parser.add_argument("--min-valid-samples", type=int, default=20)
    parser.add_argument("--samples-per-window", type=int, default=30)
    parser.add_argument("--estimator-percentile", type=float, default=80.0)
    parser.add_argument("--window-spread-max-m", type=float, default=DEFAULT_WINDOW_SPREAD_MAX_M)
    parser.add_argument("--anthropometric-min-m", type=float, default=DEFAULT_ANTHROPOMETRIC_BAND_M[0])
    parser.add_argument("--anthropometric-max-m", type=float, default=DEFAULT_ANTHROPOMETRIC_BAND_M[1])
    parser.add_argument("--allow-unstable", action="store_true", help="Write normalization even if window spread exceeds the instability gate.")
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing pre-scale backup or output skeleton.")
    args = parser.parse_args(argv)

    try:
        paths = _resolve_paths(args)
        skeleton = _read_json(paths["skeleton"])
        tracks = _read_json(paths["tracks"])
        calibration = _read_json(paths["calibration"])
        estimates = estimate_player_metric_heights(
            tracks,
            calibration,
            skeleton_payload=skeleton,
            min_bbox_confidence=args.min_bbox_confidence,
            min_valid_samples=args.min_valid_samples,
            samples_per_window=args.samples_per_window,
            estimator_percentile=args.estimator_percentile,
            window_spread_max_m=args.window_spread_max_m,
        )
        _write_json(paths["estimates"], estimates)
        if estimates.get("unstable") and not args.allow_unstable:
            unstable_players = ", ".join(str(player) for player in estimates.get("unstable_players", []))
            raise PlayerScaleError(f"unstable height estimates exceed window spread gate for players: {unstable_players}")

        backup_path = paths["output_skeleton"].with_name("skeleton3d.pre_player_scale.json")
        if backup_path.exists() and not args.force:
            raise PlayerScaleError(f"pre-scale backup already exists; pass --force to replace it: {backup_path}")
        if paths["output_skeleton"].exists() and paths["output_skeleton"] != paths["skeleton"] and not args.force:
            raise PlayerScaleError(f"output skeleton already exists; pass --force to replace it: {paths['output_skeleton']}")

        normalized, report = normalize_skeleton_scale_payload(
            skeleton,
            estimates,
            estimate_path=str(paths["estimates"]),
            pre_scale_backup_path=str(backup_path),
            min_confidence=args.min_confidence,
            anthropometric_band_m=(args.anthropometric_min_m, args.anthropometric_max_m),
            require_stable=not args.allow_unstable,
        )
        if backup_path.exists():
            backup_path.unlink()
        shutil.copy2(paths["skeleton"], backup_path)
        _write_json(paths["output_skeleton"], normalized)
        _write_json(paths["report"], report)
    except (OSError, ValueError, TypeError, json.JSONDecodeError, PlayerScaleError) as exc:
        print(f"ERROR: player scale normalization failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "racketsport_player_scale_normalization_cli_summary",
                "status": "normalized",
                "skeleton": str(paths["output_skeleton"]),
                "pre_scale_backup": str(backup_path),
                "estimates": str(paths["estimates"]),
                "report": str(paths["report"]),
                "players": report["players"],
                "unstable": estimates["unstable"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    skeleton = args.skeleton or (args.run_dir / "skeleton3d.json" if args.run_dir is not None else None)
    tracks = args.tracks or (args.run_dir / "tracks.json" if args.run_dir is not None else None)
    calibration = args.court_calibration or (args.run_dir / "court_calibration.json" if args.run_dir is not None else None)
    if skeleton is None:
        raise ValueError("--skeleton is required unless --run-dir is supplied")
    if tracks is None:
        raise ValueError("--tracks is required unless --run-dir is supplied")
    if calibration is None:
        raise ValueError("--court-calibration is required unless --run-dir is supplied")
    for path, label in ((skeleton, "skeleton3d.json"), (tracks, "tracks.json"), (calibration, "court_calibration.json")):
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")
    output_skeleton = args.output_skeleton or skeleton
    out_dir = output_skeleton.parent
    return {
        "skeleton": skeleton,
        "tracks": tracks,
        "calibration": calibration,
        "output_skeleton": output_skeleton,
        "estimates": args.estimates_out or out_dir / "player_scale_estimates.json",
        "report": args.report_out or out_dir / "player_scale_normalization_report.json",
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
