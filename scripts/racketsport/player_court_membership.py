#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.player_court_membership import (  # noqa: E402
    compute_player_court_membership,
    write_membership_evidence,
    write_membership_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build preview-only target-court membership verdicts from player tracks, "
            "court calibration, and optional camera-motion compensation."
        )
    )
    parser.add_argument("tracks", type=Path, help="tracks.json artifact.")
    parser.add_argument("--calibration", type=Path, required=True, help="court_calibration.json artifact.")
    parser.add_argument("--camera-motion", type=Path, help="Optional camera_motion.json artifact.")
    parser.add_argument("--out", type=Path, required=True, help="Output membership.json path.")
    parser.add_argument("--evidence-dir", type=Path, help="Optional directory for annotated preview evidence images.")
    parser.add_argument("--video", type=Path, help="Source video for evidence crops/overlay.")
    args = parser.parse_args(argv)

    try:
        tracks_payload = _read_json(args.tracks)
        calibration_payload = _read_json(args.calibration)
        camera_motion_payload = _read_json(args.camera_motion) if args.camera_motion is not None else None
        payload = compute_player_court_membership(
            tracks_payload,
            calibration_payload,
            camera_motion_payload,
        )
        if args.evidence_dir is not None or args.video is not None:
            if args.evidence_dir is None or args.video is None:
                raise ValueError("--evidence-dir and --video must be supplied together")
            payload["evidence"] = write_membership_evidence(
                membership_payload=payload,
                tracks_payload=tracks_payload,
                calibration_payload=calibration_payload,
                camera_motion_payload=camera_motion_payload,
                video_path=args.video,
                evidence_dir=args.evidence_dir,
            )
        write_membership_json(payload, args.out)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: player-court membership failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "out": args.out.as_posix(),
                "camera_motion_used": payload["camera_motion_used"],
                "n_compensated_frames_used": payload["n_compensated_frames_used"],
                "per_player": {
                    player_id: {
                        "verdict": metrics["verdict"],
                        "inside_strict_frac": metrics["inside_strict_frac"],
                        "inside_asym_frac": metrics["inside_asym_frac"],
                        "median_x_m": metrics["median_x_m"],
                        "median_y_m": metrics["median_y_m"],
                        "abs_y_p10_m": metrics["abs_y_p10_m"],
                        "min_abs_y_m": metrics["min_abs_y_m"],
                    }
                    for player_id, metrics in payload["per_player"].items()
                },
                "verified": False,
                "not_gate_verified": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _read_json(path: Path | None) -> dict:
    if path is None:
        raise FileNotFoundError("missing JSON path")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
