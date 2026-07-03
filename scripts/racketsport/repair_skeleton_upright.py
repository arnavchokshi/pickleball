#!/usr/bin/env python3
"""Repair skeleton3d camera-frame offset grounding with court-frame rotation."""

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

from threed.racketsport.skeleton_upright import repair_skeleton_upright_payload  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Run directory containing skeleton3d.json and court_calibration.json.")
    parser.add_argument("--skeleton", type=Path, help="Explicit skeleton3d.json path; defaults to --run-dir/skeleton3d.json.")
    parser.add_argument(
        "--court-calibration",
        type=Path,
        help="Explicit court_calibration.json path; defaults to --run-dir/court_calibration.json.",
    )
    parser.add_argument(
        "--foot-contact-phases",
        type=Path,
        help="Optional foot_contact_phases.json; contact frames use raw foot-min-to-floor z grounding.",
    )
    parser.add_argument("--z-smoothing-radius", type=int, default=2)
    parser.add_argument("--stature-min-m", type=float, default=1.4)
    parser.add_argument("--stature-max-m", type=float, default=1.8)
    parser.add_argument(
        "--overlay-scale-suspect-caption",
        default=None,
        help="Caption burned by render_skeleton_overlay.py when stature_check scale_suspect is true.",
    )
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing skeleton3d.pre_upright.json backup.")
    args = parser.parse_args(argv)

    try:
        skeleton_path, calibration_path = _resolve_paths(args.run_dir, args.skeleton, args.court_calibration)
        skeleton = _read_json(skeleton_path)
        calibration = _read_json(calibration_path)
        phases = _read_json(args.foot_contact_phases) if args.foot_contact_phases else None
        rotation = calibration["extrinsics"]["R"]
        repaired, report = repair_skeleton_upright_payload(
            skeleton,
            calibration_rotation=rotation,
            calibration_path=str(calibration_path),
            z_smoothing_radius=args.z_smoothing_radius,
            foot_contact_phases=phases,
            stature_band_m=(args.stature_min_m, args.stature_max_m),
            overlay_scale_suspect_caption=args.overlay_scale_suspect_caption,
        )
        backup_path = skeleton_path.with_name("skeleton3d.pre_upright.json")
        if backup_path.exists() and not args.force:
            raise ValueError(f"backup already exists; pass --force to replace it: {backup_path}")
        if backup_path.exists():
            backup_path.unlink()
        shutil.copy2(skeleton_path, backup_path)
        report["pre_upright_backup"] = str(backup_path)
        repaired.setdefault("provenance", {}).setdefault("skeleton_upright_repair", {})["pre_upright_backup"] = str(backup_path)
        _write_json(skeleton_path, repaired)
        report_path = skeleton_path.with_name("skeleton_upright_repair.json")
        _write_json(report_path, report)
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: skeleton upright repair failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "skeleton_upright_repair_cli_summary",
                "skeleton": str(skeleton_path),
                "pre_upright_backup": str(backup_path),
                "report": str(report_path),
                "selected_convention": report["selected_convention"],
                "metrics_after": report["metrics_after"],
                "stature_check": report["stature_check"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _resolve_paths(run_dir: Path | None, skeleton: Path | None, calibration: Path | None) -> tuple[Path, Path]:
    skeleton_path = skeleton or (run_dir / "skeleton3d.json" if run_dir is not None else None)
    calibration_path = calibration or (run_dir / "court_calibration.json" if run_dir is not None else None)
    if skeleton_path is None:
        raise ValueError("--skeleton is required unless --run-dir is supplied")
    if calibration_path is None:
        raise ValueError("--court-calibration is required unless --run-dir is supplied")
    if not skeleton_path.is_file():
        raise FileNotFoundError(f"missing skeleton3d.json: {skeleton_path}")
    if not calibration_path.is_file():
        raise FileNotFoundError(f"missing court_calibration.json: {calibration_path}")
    return skeleton_path, calibration_path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
