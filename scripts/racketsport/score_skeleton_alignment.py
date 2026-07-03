#!/usr/bin/env python3
"""Score skeleton3d alignment against authoritative 2D keypoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.skeleton_alignment_metrics import score_skeleton_alignment  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skeleton", type=Path, required=True, help="Path to skeleton3d.json or skeleton3d_v2.json")
    parser.add_argument("--keypoints-2d", type=Path, required=True, help="Path to keypoints_2d.json")
    parser.add_argument("--court-calibration", type=Path, required=True, help="Path to court_calibration.json")
    parser.add_argument("--out", type=Path, required=True, help="Output skeleton_alignment_metrics.json")
    parser.add_argument("--min-keypoint-confidence", type=float, default=0.05)
    args = parser.parse_args(argv)

    try:
        skeleton = _read_json(args.skeleton)
        keypoints = _read_json(args.keypoints_2d)
        calibration = _read_json(args.court_calibration)
        report = score_skeleton_alignment(
            skeleton,
            keypoints,
            calibration,
            min_keypoint_confidence=args.min_keypoint_confidence,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.out, report)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: score_skeleton_alignment failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "skeleton_alignment_metrics_cli_summary",
                "metrics": str(args.out),
                "projection_error_px": report["projection_error_px"]["overall"],
                "comparison_ready": report["comparison_ready"],
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
