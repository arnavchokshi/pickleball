#!/usr/bin/env python3
"""Reproduce the far-field diagnosis on outdoor_webcam_20s. Read-only, CPU only.

Four measurements, in the order they were made:

1. The radial distribution of the calibration's correspondences and of the six
   owner bounce labels, plus the court x each label back-projects to.
2. What ANY radial distortion could do to those positions: the camera is held
   fixed and ``intrinsics.dist`` is swept. This is the discriminating test
   against the "uncorrected barrel distortion" hypothesis -- it bounds the
   mechanism's magnitude regardless of which sign or size is true.
3. Leave-one-out over the 15 correspondences, scored in metres on the plane,
   against the held-out point's own radius. Shows whether the validation had
   any leverage where the extrapolation happens.
4. Resampled refits (15 jackknife + 40 leave-3-out), each re-fitting focal AND
   pose, back-projecting the six bounce pixels. The spread is how much this
   MODEL FAMILY disagrees with itself at each radius. It is NOT an accuracy
   bound: every refit shares the training set's central blind spot.

    python3 runs/lanes/farfield_extrapolation_20260727/diagnose_far_field.py \
        --run <dir with court_calibration.json> --labels <ball_human_labels.json> \
        --out runs/lanes/farfield_extrapolation_20260727/diagnosis.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from threed.racketsport.ball_arc_solver import (  # noqa: E402
    BALL_RADIUS_M,
    intersect_ray_z,
    pixel_ray_world,
)
from threed.racketsport.calibration_extrapolation import (  # noqa: E402
    calibrated_image_envelope,
    evaluate_pixel,
)

SIDELINE_M = 3.048


def radius_pct(pixel: Any, width: float, height: float) -> float:
    half_diagonal = math.hypot(width / 2.0, height / 2.0)
    return 100.0 * math.hypot(pixel[0] - width / 2.0, pixel[1] - height / 2.0) / half_diagonal


def solve_bounce(calibration: dict[str, Any], pixel: Any) -> tuple[float, float, float]:
    origin, direction = pixel_ray_world(calibration, pixel)
    return intersect_ray_z(origin, direction, BALL_RADIUS_M)


# --- a local re-fit of the same model class (zero distortion, searched focal) ---
# Reimplemented here rather than imported so the metric-15pt fit path, which
# another lane owns, is neither touched nor depended on. Reproduces the shipped
# focal to 0.06 px.


def _solve_pose(focal: float, image: np.ndarray, world: np.ndarray, k1: float):
    matrix = np.array([[focal, 0.0, 960.0], [0.0, focal, 540.0], [0.0, 0.0, 1.0]])
    distortion = np.array([k1, 0.0, 0.0, 0.0])
    ok, rvec, tvec = cv2.solvePnP(
        world.astype(np.float64),
        image.astype(np.float64),
        matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    return rotation, tvec.reshape(3), matrix, distortion


def _rms(fit, image: np.ndarray, world: np.ndarray) -> float:
    rotation, translation, matrix, distortion = fit
    projected, _ = cv2.projectPoints(
        world.astype(np.float64),
        cv2.Rodrigues(rotation)[0],
        translation.reshape(3, 1),
        matrix,
        distortion,
    )
    return float(np.sqrt(((projected.reshape(-1, 2) - image) ** 2).sum(1).mean()))


def fit_focal(image: np.ndarray, world: np.ndarray, k1: float = 0.0):
    best = None
    for focal in np.arange(1200.0, 5000.0, 20.0):
        fit = _solve_pose(focal, image, world, k1)
        if fit is None:
            continue
        score = _rms(fit, image, world)
        if best is None or score < best[0]:
            best = (score, focal)
    focal = best[1]
    for span, step in ((20.0, 2.0), (2.0, 0.2), (0.2, 0.02)):
        candidate = None
        for value in np.arange(focal - span, focal + span + 1e-9, step):
            fit = _solve_pose(value, image, world, k1)
            if fit is None:
                continue
            score = _rms(fit, image, world)
            if candidate is None or score < candidate[0]:
                candidate = (score, value)
        focal = candidate[1]
    return candidate[0], focal, _solve_pose(focal, image, world, k1)


def back_project(fit, pixel: Any, z: float):
    rotation, translation, matrix, distortion = fit
    point = np.array([[float(pixel[0]), float(pixel[1])]], np.float64).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(point, matrix, distortion).reshape(2)
    origin = -rotation.T @ translation
    direction = rotation.T @ np.array([normalized[0], normalized[1], 1.0])
    direction = direction / np.linalg.norm(direction)
    if abs(direction[2]) < 1e-12:
        return None
    scale = (z - origin[2]) / direction[2]
    if scale <= 0.0:
        return None
    return origin + scale * direction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    calibration = json.loads(
        (Path(args.run) / "court_calibration.json").read_text(encoding="utf-8")
    )
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    width, height = calibration["image_size"]
    envelope = calibrated_image_envelope(calibration)
    image = np.array(calibration["image_pts"], float)
    world = np.array(calibration["world_pts"], float)

    # --- 1. radial distribution -------------------------------------------
    correspondence_radii = sorted(radius_pct(point, width, height) for point in image)
    bounces = [
        (label["frame"], label["pixel_xy"])
        for label in labels["labels"]
        if label["kind"] == "bounce"
    ]
    bounce_rows = []
    for frame, pixel in bounces:
        solved = solve_bounce(calibration, pixel)
        verdict = evaluate_pixel(pixel, envelope)
        bounce_rows.append(
            {
                "frame": frame,
                "pixel_xy": pixel,
                "radius_pct_of_half_diagonal": round(radius_pct(pixel, width, height), 2),
                "court_x_m": round(solved[0], 4),
                "court_y_m": round(solved[1], 4),
                "metres_outside_sideline": round(abs(solved[0]) - SIDELINE_M, 4),
                "envelope_verdict": verdict["verdict"],
            }
        )

    # --- 2. what any radial distortion could do ---------------------------
    distortion_rows = []
    for frame, pixel in bounces:
        row = {"frame": frame, "k1": {}}
        base_x = solve_bounce(calibration, pixel)[0]
        row["k1_zero_court_x_m"] = round(base_x, 4)
        for k1 in (-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30):
            variant = copy.deepcopy(calibration)
            variant["intrinsics"]["dist"] = [k1, 0.0, 0.0, 0.0]
            row["k1"][f"{k1:+.2f}"] = round(solve_bounce(variant, pixel)[0] - base_x, 4)
        distortion_rows.append(row)

    # --- 3. leave-one-out, scored against the held-out point's radius ------
    loo_rows = []
    for index in range(len(image)):
        mask = np.ones(len(image), bool)
        mask[index] = False
        _, focal, fit = fit_focal(image[mask], world[mask])
        point = back_project(fit, image[index], world[index][2])
        error = (
            float(math.hypot(point[0] - world[index][0], point[1] - world[index][1]))
            if point is not None
            else float("nan")
        )
        loo_rows.append(
            {
                "radius_pct_of_half_diagonal": round(radius_pct(image[index], width, height), 2),
                "held_out_plane_error_m": round(error, 4),
                "refit_focal_px": round(float(focal), 2),
                "world_xyz_m": [round(float(v), 4) for v in world[index]],
            }
        )

    # --- 4. resampled refits at the six bounce pixels ---------------------
    rng = np.random.default_rng(args.seed)
    fits = []
    for index in range(len(image)):
        mask = np.ones(len(image), bool)
        mask[index] = False
        fits.append(fit_focal(image[mask], world[mask])[2])
    seen: set[tuple[int, ...]] = set()
    while len(fits) < len(image) + 40:
        drop = tuple(sorted(rng.choice(len(image), 3, replace=False).tolist()))
        if drop in seen:
            continue
        seen.add(drop)
        mask = np.ones(len(image), bool)
        mask[list(drop)] = False
        fits.append(fit_focal(image[mask], world[mask])[2])

    spread_rows = []
    for frame, pixel in bounces:
        points = [back_project(fit, pixel, BALL_RADIUS_M) for fit in fits]
        stack = np.array([point for point in points if point is not None])
        spread_rows.append(
            {
                "frame": frame,
                "radius_pct_of_half_diagonal": round(radius_pct(pixel, width, height), 2),
                "refit_count": int(len(stack)),
                "court_x_mean_m": round(float(stack[:, 0].mean()), 4),
                "court_x_sd_m": round(float(stack[:, 0].std()), 4),
                "court_x_min_m": round(float(stack[:, 0].min()), 4),
                "court_x_max_m": round(float(stack[:, 0].max()), 4),
            }
        )

    # --- 5. the distortion question, with a full refit at each candidate ----
    # Not the frozen-camera sweep of step 2 but the honest version: focal AND
    # pose are re-fit under each candidate k1, then the six bounces are
    # re-solved and leave-one-out is re-scored. This is the comparison the
    # calibration lane's refusal rests on, extended to the four far bounces the
    # refusal never looked at.
    refit_rows = []
    for k1 in (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.16, 0.20):
        errors = []
        for index in range(len(image)):
            mask = np.ones(len(image), bool)
            mask[index] = False
            _, _, fold = fit_focal(image[mask], world[mask], k1)
            point = back_project(fold, image[index], world[index][2])
            errors.append(
                float(math.hypot(point[0] - world[index][0], point[1] - world[index][1]))
                if point is not None
                else float("nan")
            )
        _, focal, fit = fit_focal(image, world, k1)
        court_x = {}
        for frame, pixel in bounces:
            point = back_project(fit, pixel, BALL_RADIUS_M)
            court_x[str(frame)] = None if point is None else round(float(point[0]), 4)
        refit_rows.append(
            {
                "brown_k1": k1,
                "refit_focal_px": round(float(focal), 2),
                "leave_one_out_median_plane_error_m": round(float(np.median(errors)), 4),
                "leave_one_out_p95_plane_error_m": round(float(np.percentile(errors, 95)), 4),
                "bounce_court_x_m": court_x,
            }
        )

    report = {
        "policy": "calibration_extrapolation_v1",
        "distortion_refit_sweep": refit_rows,
        "clip_id": labels.get("clip_id"),
        "calibration": {
            "intrinsics_source": calibration["intrinsics"]["source"],
            "dist": calibration["intrinsics"]["dist"],
            "fx": calibration["intrinsics"]["fx"],
            "reprojection_error_px": calibration["reprojection_error_px"],
            "correspondence_radius_pct": [round(value, 2) for value in correspondence_radii],
            "envelope": None if envelope is None else envelope.to_json(),
        },
        "bounces": bounce_rows,
        "distortion_sensitivity_court_x_delta_m": distortion_rows,
        "leave_one_out": loo_rows,
        "resampled_refit_spread": spread_rows,
        "resampled_refit_caveat": (
            "Every refit is the same model class fit to the same central "
            "correspondences. Their agreement bounds sampling noise, never "
            "model-class error, and cannot see past 50% radius any better than "
            "the original fit could."
        ),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )

    print("correspondence radii %%: %s" % [round(v, 1) for v in correspondence_radii])
    for row in bounce_rows:
        print(
            "frame %-4s r=%5.1f%%  court_x %+7.3f  %+6.3f m outside sideline  %s"
            % (
                row["frame"],
                row["radius_pct_of_half_diagonal"],
                row["court_x_m"],
                row["metres_outside_sideline"],
                row["envelope_verdict"],
            )
        )
    print("\nlargest |court_x| shift from any swept distortion, metres:")
    for row in distortion_rows:
        worst = max(abs(value) for value in row["k1"].values())
        print("  frame %-4s %.4f" % (row["frame"], worst))
    print("\nfull refit (focal+pose) under each candidate distortion:")
    print("  brown_k1  focal_px  LOO_median_m  court_x@414_m")
    for row in refit_rows:
        print(
            "  %+8.2f  %8.2f  %12.4f  %13s"
            % (
                row["brown_k1"],
                row["refit_focal_px"],
                row["leave_one_out_median_plane_error_m"],
                row["bounce_court_x_m"].get("414"),
            )
        )
    print("\nresampled refit spread of court_x, metres:")
    for row in spread_rows:
        print(
            "  frame %-4s r=%5.1f%%  sd %.4f  range %.4f"
            % (
                row["frame"],
                row["radius_pct_of_half_diagonal"],
                row["court_x_sd_m"],
                row["court_x_max_m"] - row["court_x_min_m"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
