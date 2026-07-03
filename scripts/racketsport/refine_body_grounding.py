#!/usr/bin/env python3
"""Apply post-hoc rigid BODY grounding refinement to skeleton/world artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.body_grounding_refine import GroundingRefineConfig, refine_body_grounding  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skeleton", type=Path, required=True, help="Path to skeleton3d.json or smpl_motion.json")
    parser.add_argument("--tracks", type=Path, required=True, help="Path to tracks.json with world_xy per player frame")
    parser.add_argument(
        "--foot-contact-phases",
        type=Path,
        required=True,
        help="Path to foot_contact_phases.json produced by run_physics_footlock.py",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output payload filename; defaults to the input skeleton filename",
    )
    parser.add_argument("--court-z-m", type=float, default=0.0)
    parser.add_argument(
        "--root-joint-names",
        nargs="+",
        default=("left_hip", "right_hip"),
        help="Joint names averaged as the root proxy when transl_world is absent.",
    )
    parser.add_argument("--smoothness-weight", type=float, default=0.15)
    parser.add_argument("--max-correction-warn-m", type=float, default=0.15)
    parser.add_argument(
        "--fail-if-residual-worse",
        action="store_true",
        help="Return exit code 2 after writing artifacts when foot-plane or track residuals increase.",
    )
    args = parser.parse_args(argv)

    try:
        skeleton = _read_json(args.skeleton)
        tracks = _read_json(args.tracks)
        phases = _read_json(args.foot_contact_phases)
        refined, report = refine_body_grounding(
            skeleton,
            foot_contact_phases=phases,
            tracks=tracks,
            config=GroundingRefineConfig(
                court_z_m=args.court_z_m,
                root_joint_names=tuple(args.root_joint_names),
                smoothness_weight=args.smoothness_weight,
                max_correction_warn_m=args.max_correction_warn_m,
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or args.skeleton.name
    payload_path = args.out_dir / output_name
    report_path = args.out_dir / "body_grounding_refinement.json"
    _write_json(payload_path, refined)
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                "refined_payload": str(payload_path),
                "report": str(report_path),
                "summary": report["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_if_residual_worse and report["summary"]["kill_recommended"]:
        return 2
    return 0


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
