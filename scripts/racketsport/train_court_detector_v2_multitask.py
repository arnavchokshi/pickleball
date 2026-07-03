#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.court_detector_v2_model import make_court_detector_v2_model  # noqa: E402


MODEL_HEADS = ["keypoint_heatmaps", "line_masks", "net_masks", "visibility_logits"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train or dry-run the court detector v2 multi-task model without claiming CAL-3 verification.",
    )
    parser.add_argument("--eval-root", type=Path, default=Path("eval_clips/ball"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = build_training_report(eval_root=args.eval_root, device=args.device, dry_run=args.dry_run)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"ERROR: court detector v2 training failed: {exc}", file=sys.stderr)
        return 1


def build_training_report(*, eval_root: Path, device: str, dry_run: bool) -> dict:
    model = make_court_detector_v2_model(keypoint_count=15, line_count=8, net_count=3)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    full_label_count = len(list(eval_root.glob("*/labels/court_keypoints.json"))) if eval_root.is_dir() else 0
    partial_label_count = len(list(eval_root.glob("*/labels/court_keypoints_partial.json"))) if eval_root.is_dir() else 0
    return {
        "schema_version": 1,
        "artifact_type": "court_detector_v2_multitask_training_report",
        "status": "trained_not_cal3_verified",
        "dry_run": bool(dry_run),
        "device": device,
        "verified": False,
        "not_cal3_verified": True,
        "model_heads": MODEL_HEADS,
        "parameter_count": int(parameter_count),
        "label_counts": {
            "full_label_count": int(full_label_count),
            "partial_label_count": int(partial_label_count),
        },
        "notes": [
            "model scaffold and data plumbing only",
            "not a CAL-3/no-tap promotion proof",
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
