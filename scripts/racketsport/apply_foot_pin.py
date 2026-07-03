#!/usr/bin/env python3
"""Apply stance-phase foot pinning to skeleton/world artifacts without mutating inputs by default."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.foot_pin import FootPinSettings, apply_foot_pin_to_payload  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.skeleton3d is None and args.world is None:
        raise SystemExit("at least one of --skeleton3d or --world is required")

    settings = FootPinSettings(
        enter_height_m=args.enter_height_m,
        exit_height_m=args.exit_height_m,
        enter_speed_mps=args.enter_speed_mps,
        exit_speed_mps=args.exit_speed_mps,
        min_phase_confidence=args.min_phase_confidence,
        min_phase_frames=args.min_phase_frames,
        low_foot_band_m=args.low_foot_band_m,
        taper_frames=args.taper_frames,
        max_correction_m=args.max_correction_m,
        max_smoothing_correction_m=args.max_smoothing_correction_m,
        interpolate_between_stances=not args.no_interpolate_between_stances,
    )
    out_dir = args.out_dir.expanduser()
    if not args.in_place:
        out_dir.mkdir(parents=True, exist_ok=True)
    audit_relpath = "foot_pin_audit.json"
    written: dict[str, str] = {}
    audit_payload: dict[str, Any] | None = None

    for label, input_path in (("skeleton3d", args.skeleton3d), ("world", args.world)):
        if input_path is None:
            continue
        source_path = input_path.expanduser()
        payload = _load_json(source_path)
        result = apply_foot_pin_to_payload(payload, settings=settings, audit_path=audit_relpath)
        output_path = source_path if args.in_place else out_dir / source_path.name
        _write_json(output_path, result.payload)
        written[label] = str(output_path)
        if label == "world" or audit_payload is None:
            audit_payload = result.audit

    audit_path = (args.world or args.skeleton3d).expanduser().parent / audit_relpath if args.in_place else out_dir / audit_relpath
    _write_json(audit_path, audit_payload or {})
    summary = {
        "out_dir": str(out_dir),
        "in_place": bool(args.in_place),
        "audit": str(audit_path),
        "written": written,
        "summary": (audit_payload or {}).get("summary", {}),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skeleton3d", type=Path, help="Input skeleton3d.json artifact.")
    parser.add_argument("--world", type=Path, help="Input confidence_gated_world.json or virtual_world.json artifact.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for corrected copies and audit JSON.")
    parser.add_argument("--in-place", action="store_true", help="Overwrite input artifacts instead of writing corrected copies.")
    parser.add_argument("--enter-height-m", type=float, default=0.060)
    parser.add_argument("--exit-height-m", type=float, default=0.100)
    parser.add_argument("--enter-speed-mps", type=float, default=0.75)
    parser.add_argument("--exit-speed-mps", type=float, default=1.25)
    parser.add_argument("--min-phase-confidence", type=float, default=0.20)
    parser.add_argument("--min-phase-frames", type=int, default=2)
    parser.add_argument("--low-foot-band-m", type=float, default=0.025)
    parser.add_argument("--taper-frames", type=int, default=0)
    parser.add_argument("--max-correction-m", type=float, default=0.15)
    parser.add_argument("--max-smoothing-correction-m", type=float, default=0.049)
    parser.add_argument(
        "--no-interpolate-between-stances",
        action="store_true",
        help="Disable linear root-correction interpolation between confident stance knots.",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
