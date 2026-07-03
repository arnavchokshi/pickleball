#!/usr/bin/env python3
"""Apply confidence-gated physics provenance to a chain-style virtual world."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.confidence_gate import ConfidenceGateConfig, apply_confidence_gate_to_world, summarize_bands  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Chain-style run directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--calibration-curves", type=Path, help="calibration_curves.json. Defaults to nearby run root.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--short-gap-max-frames", type=int, default=12)
    parser.add_argument("--hysteresis-frames", type=int, default=3)
    parser.add_argument("--max-non-target-displacement-m", type=float, default=0.15)
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    if not run_dir.is_dir():
        print(f"ERROR: run dir does not exist: {run_dir}", file=sys.stderr)
        return 1
    world_path = run_dir / "virtual_world.json"
    if not world_path.is_file():
        print(f"ERROR: missing virtual_world.json under {run_dir}", file=sys.stderr)
        return 1

    curves_path = args.calibration_curves or _discover_curves_path(args.out)
    calibration_curves = _load_optional_json(curves_path) if curves_path else None
    config = ConfidenceGateConfig(
        confidence_threshold=args.confidence_threshold,
        short_gap_max_frames=args.short_gap_max_frames,
        hysteresis_frames=args.hysteresis_frames,
        max_non_target_displacement_m=args.max_non_target_displacement_m,
    )
    gated = apply_confidence_gate_to_world(
        _load_json(world_path),
        ball_track_physics_filled=_load_optional_json(run_dir / "ball_track_physics_filled.json"),
        physics_footlock=_load_optional_json(run_dir / "physics_footlock.json"),
        racket_pose_estimate=_load_optional_json(run_dir / "racket_pose_estimate.json"),
        contact_windows=_load_optional_json(run_dir / "contact_windows.json"),
        calibration_curves=calibration_curves,
        config=config,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    out_world = args.out / "confidence_gated_world.json"
    out_summary = args.out / "confidence_gate_summary.json"
    out_world.write_text(json.dumps(gated, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    schema_validation = _validate_current_virtual_world_schema(out_world)
    summary = {
        "schema_version": 1,
        "out": str(out_world),
        "run_dir": str(run_dir),
        "calibration_curves": str(curves_path) if curves_path else None,
        "counts_by_entity_band": summarize_bands(gated),
        "schema_validation": schema_validation,
        "policy": {
            "additive_only": True,
            "protected_eval_labels_used": False,
            "outdoor_indoor_labels_read": False,
        },
    }
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _discover_curves_path(out_dir: Path) -> Path | None:
    candidates = [
        out_dir / "calibration_curves.json",
        out_dir.parent / "calibration_curves.json",
        out_dir.parent.parent / "calibration_curves.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _validate_current_virtual_world_schema(path: Path) -> dict[str, Any]:
    try:
        from threed.racketsport.schemas import validate_artifact_file

        validate_artifact_file("virtual_world", path)
    except Exception as exc:  # noqa: BLE001 - report schema conflict without hiding output.
        lines = str(exc).splitlines()
        return {
            "status": "fail",
            "reason": lines[0] if lines else type(exc).__name__,
            "sample_errors": lines[1:7],
            "expected_current_conflict": (
                "Current VirtualWorld schema forbids confidence_provenance/render_only fields; "
                "schemas are read-only for this lane."
            ),
        }
    return {"status": "pass"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return _load_json(path)


if __name__ == "__main__":
    raise SystemExit(main())
