#!/usr/bin/env python3
"""Build lane-B skeleton3d_v2.json from a keypoints_2d artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.skeleton_lift_2d import Lift2DConfig, lift_skeleton_from_2d, load_player_bone_lengths  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keypoints-2d", type=Path, required=True, help="Path to keypoints_2d.json")
    parser.add_argument("--tracks", type=Path, required=True, help="Path to tracks.json with per-player world_xy")
    parser.add_argument("--court-calibration", type=Path, required=True, help="Path to court_calibration.json")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output run directory")
    parser.add_argument("--output-name", default="skeleton3d_v2.json")
    parser.add_argument("--report-name", default="skeleton_lift_2d_report.json")
    parser.add_argument("--min-joint-confidence", type=float, default=0.2)
    parser.add_argument("--root-smoothing-radius", type=int, default=2)
    parser.add_argument("--default-player-height-m", type=float, default=1.72)
    parser.add_argument(
        "--player-bone-lengths",
        type=Path,
        default=None,
        help=(
            "Optional path to a bone_calib player_bone_lengths.json (e.g. "
            "runs/bone_calib_20260703T0102Z/player_bone_lengths.json). Supplies HARD "
            "per-player leg bone lengths for the kinematic-chain solve; players absent "
            "from this file fall back to a self-measured per-clip median."
        ),
    )
    args = parser.parse_args(argv)

    try:
        keypoints = _read_json(args.keypoints_2d)
        tracks = _read_json(args.tracks)
        calibration = _read_json(args.court_calibration)
        player_bone_lengths = (
            load_player_bone_lengths(_read_json(args.player_bone_lengths))
            if args.player_bone_lengths is not None
            else None
        )
        skeleton, report = lift_skeleton_from_2d(
            keypoints,
            tracks_payload=tracks,
            calibration_payload=calibration,
            config=Lift2DConfig(
                min_joint_confidence=args.min_joint_confidence,
                root_smoothing_radius=args.root_smoothing_radius,
                default_player_height_m=args.default_player_height_m,
                player_bone_lengths=player_bone_lengths,
            ),
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        skeleton_path = args.out_dir / args.output_name
        report_path = args.out_dir / args.report_name
        _write_json(skeleton_path, skeleton)
        _write_json(report_path, report)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: build_skeleton_from_2d failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "skeleton_lift_2d_cli_summary",
                "skeleton": str(skeleton_path),
                "report": str(report_path),
                "summary": report["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any] | dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
