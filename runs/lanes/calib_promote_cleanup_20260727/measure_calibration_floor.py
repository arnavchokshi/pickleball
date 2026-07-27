#!/usr/bin/env python3
"""Measure the calibration floor per clip through the LIVE selection resolver.

This is deliberately not a re-read of the refit lane's numbers. For each clip it
asks `court_calibration_selection.resolve_selected_calibration_path` what the
repo would actually consume right now, and measures that artifact. Run it before
and after promotion and the deltas are measured in place rather than inherited.

Two numbers per clip, and the difference between them matters:

* `held_out_median_plane_error_m` -- leave-one-out over all 15 correspondences,
  refitting focal length, distortion and pose on 14 and scoring the 1 withheld,
  in metres. **This is the honest figure.**
* `in_sample_*` -- `ball_label_geometry.calibration_plane_residuals`, which
  back-projects the calibration's own fitted correspondences. It is a training
  residual: more parameters always score better on it. Reported, and labelled,
  for continuity with the refit lane and because the in/out uncertainty radius
  currently consumes the in-sample p95.

Usage:
    python3 runs/lanes/calib_promote_cleanup_20260727/measure_calibration_floor.py --out <path>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_label_geometry import calibration_plane_residuals  # noqa: E402
from threed.racketsport.court_calibration_metric15 import _cross_validated_scores  # noqa: E402
from threed.racketsport.court_calibration_selection import (  # noqa: E402
    SELECTION_POINTER_FILENAME,
    file_sha256,
    resolve_selected_calibration_path,
)

#: clip -> the conventional raw artifact path a consumer resolves from.
#: Same six clips the refit lane measured, so the two reports are comparable.
CLIP_CALIBRATIONS: dict[str, str] = {
    "burlington_gold_0300_low_steep_corner": "eval_clips/ball/burlington_gold_0300_low_steep_corner/labels/court_calibration_metric15pt.json",
    "indoor_doubles_fwuks_0500_long_mid_baseline": "eval_clips/ball/indoor_doubles_fwuks_0500_long_mid_baseline/labels/court_calibration_metric15pt.json",
    "outdoor_webcam_iynbd_1500_long_high_baseline": "eval_clips/ball/outdoor_webcam_iynbd_1500_long_high_baseline/labels/court_calibration_metric15pt.json",
    "wolverine_mixed_0200_mid_steep_corner": "eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/labels/court_calibration_metric15pt.json",
    "owner_IMG_1605_8a193402780b": "eval_clips/ball/owner_IMG_1605_8a193402780b/labels/court_calibration_metric15pt.json",
    "pbvision_11min_20260713_demo_seed": "runs/lanes/pbv11_headtohead_20260713/rerun_20260715/owner_cal_seed/court_calibration_metric15pt.json",
}

#: The refit lane's refined artifacts, one per clip, promoted or not.
REFINED_ROOT = ROOT / "runs/lanes/calib_distortion_fit_20260726/refined"

NET_INDEXES = (9, 10, 11)
FLOOR_INDEXES = tuple(index for index in range(15) if index not in NET_INDEXES)


def _floor_only(calibration: dict) -> dict:
    trimmed = dict(calibration)
    trimmed["image_pts"] = [calibration["image_pts"][i] for i in FLOOR_INDEXES]
    trimmed["world_pts"] = [calibration["world_pts"][i] for i in FLOOR_INDEXES]
    return trimmed


def held_out_scores(calibration: dict) -> dict | None:
    """Leave-one-out plane error and reprojection for this artifact's own model class.

    Refits from the correspondences the artifact declares, at the radial order the
    artifact declares, so it measures what a consumer of *this artifact* inherits:
    both the world-model (net-keypoint height lives in `world_pts`) and the number
    of distortion parameters.
    """

    import cv2  # noqa: F401
    import numpy as np

    image_pts = calibration.get("image_pts")
    world_pts = calibration.get("world_pts")
    image_size = calibration.get("image_size")
    if not image_pts or not world_pts or not image_size:
        return None
    obj = np.asarray([[float(v) for v in p] for p in world_pts], dtype=np.float64)
    img = np.asarray([[float(v) for v in p] for p in image_pts], dtype=np.float64)
    width, height = float(image_size[0]), float(image_size[1])
    dist = calibration.get("intrinsics", {}).get("dist") or []
    n_radial = 0
    if len(dist) > 1 and float(dist[1]) != 0.0:
        n_radial = 2
    elif dist and float(dist[0]) != 0.0:
        n_radial = 1
    scores = _cross_validated_scores(
        cv2, np, obj, img, width, height, width / 2.0, height / 2.0, n_radial=n_radial
    )
    out: dict = {"n_radial": n_radial}
    for key, value in scores.items():
        out[key] = None if not math.isfinite(float(value)) else round(float(value), 6)
    return out


def measure(calibration: dict) -> dict:
    all15 = calibration_plane_residuals(calibration)
    floor12 = calibration_plane_residuals(_floor_only(calibration))
    intrinsics = calibration.get("intrinsics", {})
    return {
        "fx": intrinsics.get("fx"),
        "dist": intrinsics.get("dist"),
        "source": calibration.get("source"),
        "intrinsics_source": intrinsics.get("source"),
        "metric_confidence": calibration.get("metric_confidence"),
        "net_world_z_m": calibration["world_pts"][NET_INDEXES[0]][2],
        "reprojection_error_px_in_sample": calibration.get("reprojection_error_px"),
        "in_sample_floor_m_all15": {
            "median_m": all15.get("median_m"),
            "p95_m": all15.get("p95_m"),
        },
        "in_sample_floor_m_floor12": {
            "median_m": floor12.get("median_m"),
            "p95_m": floor12.get("p95_m"),
        },
        "held_out": held_out_scores(calibration),
        "in_sample_caveat": (
            "in_sample_* back-project the calibration's own fitted correspondences and are "
            "optimistic; held_out is the honest figure"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="", help="free-text label for this measurement run")
    parser.add_argument(
        "--source",
        choices=("selected", "refined_candidate"),
        default="selected",
        help=(
            "'selected' measures what the resolver hands a consumer today. "
            "'refined_candidate' measures the refit lane's refined artifact for every clip, "
            "promoted or not -- the evidence a refusal has to be argued against."
        ),
    )
    args = parser.parse_args()

    report: dict = {
        "schema_version": 1,
        "artifact_type": "racketsport_calibration_floor_measurement",
        "label": args.label,
        "verified": 0,
        "note": (
            "Calibration floor measured through the live selection resolver. VERIFIED=0: a "
            "better-fitting calibration is an engineering improvement, not a capability."
        ),
        "clips": {},
    }
    for clip, conventional in sorted(CLIP_CALIBRATIONS.items()):
        conventional_path = ROOT / conventional
        if args.source == "refined_candidate":
            selected_path = REFINED_ROOT / clip / "court_calibration_metric15pt_refined.json"
        else:
            selected_path = resolve_selected_calibration_path(conventional_path)
        pointer = conventional_path.parent / SELECTION_POINTER_FILENAME
        calibration = json.loads(selected_path.read_text(encoding="utf-8"))
        record = {
            "conventional_path": conventional,
            "selected_path": selected_path.relative_to(ROOT).as_posix(),
            "promoted": selected_path != conventional_path,
            "selection_pointer": pointer.relative_to(ROOT).as_posix() if pointer.is_file() else None,
            "selected_sha256": file_sha256(selected_path),
            "measurement": measure(calibration),
        }
        report["clips"][clip] = record
        held = record["measurement"]["held_out"] or {}
        print(
            f"{clip:46s} selected={'PROMOTED' if record['promoted'] else 'raw':>8s} "
            f"held_out_plane={held.get('held_out_median_plane_error_m')} m  "
            f"in_sample={record['measurement']['in_sample_floor_m_all15']['median_m']} m  "
            f"conf={record['measurement']['metric_confidence']}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
