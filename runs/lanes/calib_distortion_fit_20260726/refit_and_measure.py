#!/usr/bin/env python3
"""Evidence driver for the CAL distortion/net-height refit (2026-07-26).

For every clip with reviewed 15-point court keypoints:

1. read the SHIPPED raw calibration artifact (never modified);
2. refit with the corrected metric-15pt path;
3. write a refined artifact WITH PROVENANCE beside this script (the raw solve
   stays immutable; promotion into `eval_clips/` is an owner decision, not this
   script's);
4. measure the downstream calibration-residual FLOOR in metres before and after
   with `ball_label_geometry.calibration_plane_residuals`, both over all 15
   correspondences and over the 12 floor-only ones (so the improvement cannot be
   attributed solely to the 3 net points whose declared height changed);
5. re-derive the bounce-label uncertainty floor for the owner's reviewed bounce
   labels under both calibrations.

Run from the repo root. Writes report.json beside itself.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.ball_label_geometry import (  # noqa: E402
    bounce_depth_sigma_m,
    calibration_plane_residuals,
)
from threed.racketsport.court_calibration_metric15 import (  # noqa: E402
    _cross_validated_scores,
    metric_calibration_from_reviewed_keypoints_15pt,
)

LANE = Path(__file__).resolve().parent

# clip -> (reviewed keypoints, shipped raw calibration, native size override)
CLIPS: dict[str, tuple[str, str, tuple[float, float] | None]] = {
    "burlington_gold_0300_low_steep_corner": (
        "eval_clips/ball/burlington_gold_0300_low_steep_corner/labels/court_keypoints.json",
        "eval_clips/ball/burlington_gold_0300_low_steep_corner/labels/court_calibration_metric15pt.json",
        None,
    ),
    "indoor_doubles_fwuks_0500_long_mid_baseline": (
        "eval_clips/ball/indoor_doubles_fwuks_0500_long_mid_baseline/labels/court_keypoints.json",
        "eval_clips/ball/indoor_doubles_fwuks_0500_long_mid_baseline/labels/court_calibration_metric15pt.json",
        None,
    ),
    "outdoor_webcam_iynbd_1500_long_high_baseline": (
        "eval_clips/ball/outdoor_webcam_iynbd_1500_long_high_baseline/labels/court_keypoints.json",
        "eval_clips/ball/outdoor_webcam_iynbd_1500_long_high_baseline/labels/court_calibration_metric15pt.json",
        None,
    ),
    "wolverine_mixed_0200_mid_steep_corner": (
        "eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/labels/court_keypoints.json",
        "eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/labels/court_calibration_metric15pt.json",
        None,
    ),
    "owner_IMG_1605_8a193402780b": (
        "eval_clips/ball/owner_IMG_1605_8a193402780b/labels/court_keypoints.json",
        "eval_clips/ball/owner_IMG_1605_8a193402780b/labels/court_calibration_metric15pt.json",
        (1080.0, 1920.0),
    ),
    "pbvision_11min_20260713_demo_seed": (
        "runs/lanes/pbv11_headtohead_20260713/rerun_20260715/owner_cal_seed/court_keypoints_reviewed.json",
        "runs/lanes/pbv11_headtohead_20260713/rerun_20260715/owner_cal_seed/court_calibration_metric15pt.json",
        None,
    ),
}

NET_INDEXES = (9, 10, 11)
FLOOR_INDEXES = tuple(index for index in range(15) if index not in NET_INDEXES)


def floor_only(calibration: dict) -> dict:
    """Same calibration, restricted to the 12 court-plane correspondences."""

    trimmed = dict(calibration)
    trimmed["image_pts"] = [calibration["image_pts"][index] for index in FLOOR_INDEXES]
    trimmed["world_pts"] = [calibration["world_pts"][index] for index in FLOOR_INDEXES]
    return trimmed


def pinhole_only(calibration: dict) -> dict:
    """The same calibration as a distortion-blind consumer would have read it.

    Reproduces the f29145a behaviour of `pixel_ray_world`, which ignored
    `intrinsics.dist` entirely, so the seam fix and the refit can be attributed
    separately instead of being reported as one lump.
    """

    blinded = dict(calibration)
    intrinsics = dict(calibration.get("intrinsics", {}))
    intrinsics["dist"] = [0.0, 0.0, 0.0, 0.0]
    blinded["intrinsics"] = intrinsics
    return blinded


def held_out_floor_m(calibration: dict) -> float | None:
    """Leave-one-out plane error in metres for this calibration's own points.

    `calibration_plane_residuals` is in-sample: it back-projects the very
    correspondences the calibration was fit to, so more parameters always score
    better on it. This refits the camera 15 times, each time withholding one
    correspondence, and reports the median error on the withheld ones.
    """

    try:
        import cv2  # noqa: F401
        import numpy as np
    except ImportError:  # pragma: no cover
        return None
    image_pts = calibration.get("image_pts")
    world_pts = calibration.get("world_pts")
    image_size = calibration.get("image_size")
    if not image_pts or not world_pts or not image_size:
        return None
    obj = np.asarray([[float(v) for v in point] for point in world_pts], dtype=np.float64)
    img = np.asarray([[float(v) for v in point] for point in image_pts], dtype=np.float64)
    width, height = float(image_size[0]), float(image_size[1])
    n_radial = 0
    dist = calibration.get("intrinsics", {}).get("dist") or []
    if len(dist) > 1 and float(dist[1]) != 0.0:
        n_radial = 2
    elif dist and float(dist[0]) != 0.0:
        n_radial = 1
    scores = _cross_validated_scores(
        cv2, np, obj, img, width, height, width / 2.0, height / 2.0, n_radial=n_radial
    )
    value = scores["held_out_median_plane_error_m"]
    return None if not math.isfinite(value) else round(value, 6)


def summarise(calibration: dict) -> dict:
    intrinsics = calibration.get("intrinsics", {})
    all_points = calibration_plane_residuals(calibration)
    floor_points = calibration_plane_residuals(floor_only(calibration))
    return {
        "fx": intrinsics.get("fx"),
        "dist": intrinsics.get("dist"),
        "metric_confidence": calibration.get("metric_confidence"),
        "reprojection_error_px": calibration.get("reprojection_error_px"),
        "net_world_z_m": calibration["world_pts"][NET_INDEXES[0]][2],
        "calibration_floor_m_all15": {
            "median_m": all_points.get("median_m"),
            "p95_m": all_points.get("p95_m"),
            "max_m": all_points.get("max_m"),
        },
        "calibration_floor_m_floor12": {
            "median_m": floor_points.get("median_m"),
            "p95_m": floor_points.get("p95_m"),
            "max_m": floor_points.get("max_m"),
        },
        "held_out_floor_m": held_out_floor_m(calibration),
        "inout_uncertainty_radius_m": inout_uncertainty_radius_m(calibration),
    }


def bounce_label_floors(calibration: dict, labels_path: Path) -> dict | None:
    if not labels_path.is_file():
        return None
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    residuals = calibration_plane_residuals(calibration)
    plane_residual_m = residuals.get("median_m") if residuals.get("available") else None
    sigmas: list[float] = []
    failures = 0
    for label in payload.get("labels", []):
        if label.get("kind") != "bounce":
            continue
        try:
            sigma, _basis = bounce_depth_sigma_m(
                calibration, label["pixel_xy"], plane_residual_m=plane_residual_m
            )
        except Exception:
            failures += 1
            continue
        sigmas.append(float(sigma))
    if not sigmas:
        return {"bounce_label_count": 0, "unsolvable": failures}
    sigmas.sort()
    return {
        "bounce_label_count": len(sigmas),
        "unsolvable": failures,
        "plane_residual_floor_m": plane_residual_m,
        "sigma_along_ray_median_m": round(sigmas[len(sigmas) // 2], 6),
        "sigma_along_ray_min_m": round(sigmas[0], 6),
        "sigma_along_ray_max_m": round(sigmas[-1], 6),
    }


def inout_uncertainty_radius_m(calibration: dict) -> dict | None:
    """In/out uncertainty radius at four canonical near-line bounce spots.

    `ball_line_calls` returns "unknown" -- abstains -- whenever the bounce's
    distance to the nearest boundary is inside this radius, so it is the number
    that decides abstention. `ball_inout_uncertainty` builds it from
    `reprojection_error_px.p95 * GSD` plus elevation parallax, so a better fit
    shrinks it directly.
    """

    from threed.racketsport.ball_inout_uncertainty import (
        CameraPose,
        bounce_geometric_uncertainty_m,
    )
    from threed.racketsport.ball_label_geometry import project_world_to_pixel

    intrinsics = calibration.get("intrinsics") or {}
    extrinsics = calibration.get("extrinsics") or {}
    rotation = extrinsics.get("R")
    translation = extrinsics.get("t")
    if not rotation or not translation:
        return None
    centre = [
        -sum(rotation[k][i] * float(translation[k]) for k in range(3)) for i in range(3)
    ]
    pose = CameraPose(
        fx=float(intrinsics["fx"]),
        fy=float(intrinsics["fy"]),
        cx=float(intrinsics["cx"]),
        cy=float(intrinsics["cy"]),
        R=[[float(value) for value in row] for row in rotation],
        t=[float(value) for value in translation],
        camera_center_world=centre,
        camera_height_m=abs(centre[2]),
        reprojection_error_px_median=float(calibration["reprojection_error_px"]["median"]),
        reprojection_error_px_p95=float(calibration["reprojection_error_px"]["p95"]),
        source=str(calibration.get("source") or ""),
    )
    # Just inside each baseline/sideline: the calls that actually get made.
    spots = {
        "near_baseline_centre": (0.0, -6.60),
        "far_baseline_centre": (0.0, 6.60),
        "left_sideline_mid": (-2.95, -3.0),
        "right_sideline_mid": (2.95, 3.0),
    }
    out: dict = {}
    for name, world_xy in spots.items():
        try:
            pixel = project_world_to_pixel(calibration, (world_xy[0], world_xy[1], 0.0))
            record = bounce_geometric_uncertainty_m(
                contact_xy_img=pixel,
                world_xy=world_xy,
                pose=pose,
                fps=30.0,
                reprojection_error_px_p95=pose.reprojection_error_px_p95,
            )
        except Exception as exc:  # pragma: no cover - diagnostic driver
            out[name] = {"error": str(exc)}
            continue
        out[name] = {
            "uncertainty_m": round(float(record["uncertainty_m"]), 5),
            "sigma_reproj_m": round(float(record["breakdown"]["sigma_reproj_m"]), 5),
            "dominant_term": record["dominant_uncertainty_term"],
        }
    return out


BOUNCE_LABEL_SOURCES = {
    "wolverine_mixed_0200_mid_steep_corner": Path(
        "/private/tmp/pickleball-balllabel-20260726/runs/lanes/ball_label_tool_20260726/labels/wolverine/ball_human_labels.json"
    ),
    "outdoor_webcam_iynbd_1500_long_high_baseline": Path(
        "/private/tmp/pickleball-balllabel-20260726/runs/lanes/ball_label_tool_20260726/labels/outdoor_webcam/ball_human_labels.json"
    ),
}


def main() -> int:
    refined_root = LANE / "refined"
    refined_root.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "generated_by": "runs/lanes/calib_distortion_fit_20260726/refit_and_measure.py",
        "verified": 0,
        "note": (
            "Engineering improvement to the metric-15pt calibration fit. Promotes no "
            "capability; the ball is not verified. Raw solves are read-only here."
        ),
        "clips": {},
    }
    for clip, (keypoints, raw_calibration, native) in CLIPS.items():
        started = time.time()
        before = json.loads((ROOT / raw_calibration).read_text(encoding="utf-8"))
        refit = metric_calibration_from_reviewed_keypoints_15pt(
            ROOT / keypoints, native_image_size=native
        )
        after = refit.model_dump(mode="json")
        after["provenance"] = {
            "refinement_of": raw_calibration,
            "reviewed_keypoints": keypoints,
            "refit_lane": "runs/lanes/calib_distortion_fit_20260726",
            "refit_reason": (
                "raw solve was fit with the 3 net keypoints declared at net-top height and a "
                "distortion gate scored on training residual; both corrected"
            ),
            "authority_unchanged": (
                "source stays metric_15pt_reviewed -- a better fit of the same owner-reviewed "
                "correspondences is not a new authority class"
            ),
            "not_gate_verified": True,
        }
        target = refined_root / clip
        target.mkdir(parents=True, exist_ok=True)
        (target / "court_calibration_metric15pt_refined.json").write_text(
            json.dumps(after, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        record: dict = {
            "reviewed_keypoints": keypoints,
            "raw_calibration": raw_calibration,
            "refined_calibration": str(
                (target / "court_calibration_metric15pt_refined.json").relative_to(ROOT)
            ),
            # A: exactly what f29145a produced -- shipped artifact read by a
            #    distortion-blind consumer.
            "state_a_baseline_f29145a": summarise(pinhole_only(before)),
            # B: shipped artifact, distortion-aware consumer (seam fix only).
            "state_b_seam_fix_only": summarise(before),
            # C: refined artifact, distortion-aware consumer (full change).
            "state_c_refit": summarise(after),
            "before": summarise(pinhole_only(before)),
            "after": summarise(after),
            "refit_seconds": round(time.time() - started, 1),
        }
        for reason in after.get("capture_quality", {}).get("reasons", []):
            if reason.startswith("model_selected_on_") or reason.startswith("net_keypoint_label_height"):
                record.setdefault("selection_reasons", []).append(reason)
        labels_path = BOUNCE_LABEL_SOURCES.get(clip)
        if labels_path is not None:
            record["owner_bounce_labels"] = {
                "labels_path": str(labels_path),
                "before": bounce_label_floors(pinhole_only(before), labels_path),
                "after": bounce_label_floors(after, labels_path),
            }
        before_median = record["before"]["calibration_floor_m_all15"]["median_m"]
        after_median = record["after"]["calibration_floor_m_all15"]["median_m"]
        held_before = record["before"]["held_out_floor_m"]
        held_after = record["after"]["held_out_floor_m"]
        record["floor_change_all15_in_sample"] = {
            "before_m": before_median,
            "after_m": after_median,
            "delta_m": round((after_median or 0.0) - (before_median or 0.0), 6),
            "improved": (after_median is not None and before_median is not None and after_median < before_median),
            "caveat": "in-sample: back-projects the calibration's own fitted correspondences",
        }
        record["floor_change_held_out"] = {
            "before_m": held_before,
            "after_m": held_after,
            "delta_m": (None if held_before is None or held_after is None else round(held_after - held_before, 6)),
            "improved": (held_before is not None and held_after is not None and held_after < held_before),
        }
        report["clips"][clip] = record
        print(
            f"{clip:46s} in-sample {before_median:.4f} -> {after_median:.4f} m | "
            f"held-out {held_before} -> {held_after} m | "
            f"conf {record['before']['metric_confidence']} -> {record['after']['metric_confidence']} | "
            f"dist {record['after']['dist']} ({record['refit_seconds']}s)",
            flush=True,
        )

    improved = [
        clip for clip, record in report["clips"].items() if record["floor_change_held_out"]["improved"]
    ]
    report["summary"] = {
        "clip_count": len(report["clips"]),
        "held_out_floor_improved_clip_count": len(improved),
        "held_out_floor_improved_clips": improved,
        "in_sample_floor_improved_clips": [
            clip
            for clip, record in report["clips"].items()
            if record["floor_change_all15_in_sample"]["improved"]
        ],
    }
    (LANE / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {LANE / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
