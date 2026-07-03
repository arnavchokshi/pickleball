#!/usr/bin/env python3
"""Apply the SAM-3D lower-arm wrist bone lock to an existing skeleton3d artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.pose_temporal import apply_sam3d_wrist_bone_lock  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Project SAM-3D wrist joints to canonical per-player lower-arm lengths.",
    )
    parser.add_argument("--skeleton3d", required=True, type=Path, help="Input skeleton3d.json artifact.")
    parser.add_argument(
        "--player-bone-lengths",
        type=Path,
        default=None,
        help="player_bone_lengths.json canonical bone calibration artifact.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output locked skeleton3d.json path.")
    parser.add_argument(
        "--confidence-floor",
        type=float,
        default=0.25,
        help="Minimum elbow/wrist joint confidence required to lock a frame.",
    )
    parser.add_argument(
        "--degenerate-epsilon-m",
        type=float,
        default=1e-6,
        help="Leave a frame unlocked when elbow-to-wrist length is at or below this value.",
    )
    parser.add_argument(
        "--disable",
        action="store_true",
        help="Copy the input payload without applying the lock.",
    )
    args = parser.parse_args(argv)

    skeleton_path = args.skeleton3d.expanduser()
    out_path = args.out.expanduser()
    canonical = _load_json(args.player_bone_lengths.expanduser()) if args.player_bone_lengths else None
    locked = apply_sam3d_wrist_bone_lock(
        _load_json(skeleton_path),
        canonical_bone_lengths=canonical,
        enabled=not args.disable,
        confidence_floor=args.confidence_floor,
        degenerate_epsilon_m=args.degenerate_epsilon_m,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(locked, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = locked.get("provenance", {}) if isinstance(locked, dict) else {}
    summary = {
        "skeleton3d": str(skeleton_path),
        "out": str(out_path),
        "sam3d_wrist_bone_lock": provenance.get("sam3d_wrist_bone_lock", {}),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
