"""Per-axis calibration of the bounce-anchor uncertainty against measured error.

Two models are scored side by side on every TT3D view so the before/after is
apples to apples in one run:

LEGACY  -- one isotropic scalar. The solver used to consume it that way
           (`_scaled_vec(position - anchor.world_xyz, anchor_sigma)` divided all
           three world components by the SAME number), so an honest scalar
           requires EVERY Cartesian component of the anchor error to behave like
           N(0, sigma).
RAY     -- `anchor_uncertainty_for_bounce`: sigma_along_ray / sigma_perp plus an
           explicit, signed bias along the ray. The depth axis is scored against
           sigma_along and the two image-plane axes against sigma_perp.

Both are measured in the camera-aligned frame (depth-along-ray and the two
image-plane axes). Two coverage criteria are reported because they cannot both
hold for a non-Gaussian error:

    rms_ratio    -- RMS(component) / sigma. Correct target for a least-squares
                    WEIGHT, which is what the solver actually uses sigma for.
    frac_within  -- P(|component| <= sigma). 0.683 for a Gaussian, but only
                    0.577 for a uniform error, and the dominant bounce-anchor
                    term (sub-frame timing) is uniform, not Gaussian.

Run:  PYTHONPATH=. python3 scripts/tt3d/sigma_calibration.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tt3d"))

from tt3d_adapter import TT_BALL_RADIUS_M, TT3DView, iter_trajectories  # noqa: E402
from run_tt3d_validation import (  # noqa: E402
    DATA_ROOT,
    VIEWS,
    bounce_observations,
    decompose,
    estimate_contact,
)
from threed.racketsport import ball_arc_solver as solver  # noqa: E402
from threed.racketsport.ball_arc_solver import (  # noqa: E402
    BALL_RADIUS_M,
    anchor_uncertainty_for_bounce,
    build_bounce_anchor,
    intersect_ray_z,
    pixel_ray_world,
    refine_bounce_contact_time,
)


def pre_fix_sigma_f29145a(calibration, xy, *, base_sigma_m: float) -> float:
    """Verbatim `anchor_sigma_for_bounce` as of base commit f29145a.

    Kept here so BEFORE and AFTER are produced by the same run rather than by
    comparing against an archived file. It reproduces the original bugs on
    purpose, including the finite difference being taken at the module constant
    BALL_RADIUS_M (pickleball, 0.0371 m) instead of the caller's ball radius --
    which on table-tennis data is the wrong plane by 17 mm.
    """

    def _ray_plane_pixel_sigma(calib, pixel, z):
        try:
            base_origin, base_dir = pixel_ray_world(calib, pixel)
            base = intersect_ray_z(base_origin, base_dir, z)
            x_origin, x_dir = pixel_ray_world(calib, (float(pixel[0]) + 1.0, float(pixel[1])))
            y_origin, y_dir = pixel_ray_world(calib, (float(pixel[0]), float(pixel[1]) + 1.0))
            dx = math.dist(base, intersect_ray_z(x_origin, x_dir, z))
            dy = math.dist(base, intersect_ray_z(y_origin, y_dir, z))
            return math.sqrt(dx * dx + dy * dy)
        except Exception:
            return None

    gsd_sigma = solver._gsd_sigma(calibration, xy)
    finite_diff = _ray_plane_pixel_sigma(calibration, xy, BALL_RADIUS_M)
    components = [base_sigma_m]
    if gsd_sigma is not None:
        components.append(gsd_sigma)
    if finite_diff is not None:
        components.append(min(0.15, finite_diff))
    reprojection = calibration.get("reprojection_error_px")
    reproj_p95 = reprojection.get("p95") if isinstance(reprojection, dict) else None
    if reproj_p95 is not None:
        components.append(min(0.18, 0.004 * float(reproj_p95)))
    sigma = math.sqrt(sum(value * value for value in components))
    return max(base_sigma_m, min(0.35, sigma))

# Physical ball speeds at contact, MEASURED from the TT3D ground truth itself
# (scripts/tt3d/sigma_calibration.py --print-physics). They are properties of
# table tennis, not of our sigma model: no term below is fitted to the error it
# predicts. A pickleball deployment supplies its own numbers the same way; the
# library defaults (4.0 / 8.0 m/s) are pickleball priors.
TT_BOUNCE_VERTICAL_SPEED_MPS = 1.94
TT_BOUNCE_HORIZONTAL_SPEED_MPS = 5.10
TT_BOUNCE_SPEED_CV = 0.3


def _axis_stats(values: np.ndarray, sigma: np.ndarray) -> dict:
    rms = float(np.sqrt(np.mean(values**2)))
    return {
        "bias_m": float(np.mean(values)),
        "rms_m": rms,
        "sigma_m_median": float(np.median(sigma)),
        "implied_sigma_over_reported": float(rms / np.median(sigma)),
        "frac_within_1sigma": float(np.mean(np.abs(values) <= sigma)),
        "p95_abs_m": float(np.percentile(np.abs(values), 95)),
    }


def analyse(view: TT3DView, sub: str) -> dict:
    calib = view.calibration()
    cam_c = view.camera_center_world
    cam_right, _, _ = view.camera_basis()
    depth, img1, img2 = [], [], []
    legacy_sigma = []
    ray: dict[str, dict[str, list]] = {
        "measured_physics": {"along": [], "perp": [], "bias": []},
        "library_defaults": {"along": [], "perp": [], "bias": []},
    }
    sf_depth, sf_img1, sf_img2 = [], [], []
    sf_along, sf_perp, sf_bias = [], [], []
    sf_refined = 0

    for traj in iter_trajectories(DATA_ROOT, sub):
        contact = estimate_contact(traj)
        if contact is None:
            continue
        t_c = contact["t_contact"]
        gt = contact["xyz_contact"]
        j = int(np.argmin(np.abs(traj.t - t_c)))
        uv = traj.uv[j]
        anchor = build_bounce_anchor(
            {"frame": j, "t": float(t_c), "xy": [float(uv[0]), float(uv[1])]},
            calib,
            ball_radius_m=TT_BALL_RADIUS_M,
            status="human_reviewed",
        )
        e = np.array(anchor.world_xyz, dtype=float) - gt
        a, b, c = decompose(e, gt, cam_c, cam_right)
        depth.append(a)
        img1.append(b)
        img2.append(c)
        legacy_sigma.append(
            pre_fix_sigma_f29145a(calib, (float(uv[0]), float(uv[1])), base_sigma_m=0.05)
        )
        for key, speeds in (
            ("measured_physics", (TT_BOUNCE_VERTICAL_SPEED_MPS, TT_BOUNCE_HORIZONTAL_SPEED_MPS)),
            ("library_defaults", (None, None)),
        ):
            kwargs = {}
            if speeds[0] is not None:
                kwargs = {
                    "vertical_speed_mps": speeds[0],
                    "horizontal_speed_mps": speeds[1],
                    "speed_cv": TT_BOUNCE_SPEED_CV,
                }
            u = anchor_uncertainty_for_bounce(
                calib,
                (float(uv[0]), float(uv[1])),
                base_sigma_m=0.05,
                ball_radius_m=TT_BALL_RADIUS_M,
                fps=traj.fps,
                **kwargs,
            )
            ray[key]["along"].append(u.sigma_along_ray_m)
            ray[key]["perp"].append(u.sigma_perp_m)
            ray[key]["bias"].append(u.bias_along_ray_m)

        # --- sub-frame variant, same bounce, same pass ---------------------
        timing = refine_bounce_contact_time(bounce_observations(traj), j, fps=traj.fps)
        sf_anchor = build_bounce_anchor(
            {"frame": j, "t": float(t_c), "xy": [float(uv[0]), float(uv[1])]},
            calib,
            ball_radius_m=TT_BALL_RADIUS_M,
            status="human_reviewed",
            subframe_timing=timing,
        )
        e_sf = np.array(sf_anchor.world_xyz, dtype=float) - gt
        a2, b2, c2 = decompose(e_sf, gt, cam_c, cam_right)
        sf_depth.append(a2)
        sf_img1.append(b2)
        sf_img2.append(c2)
        applied = timing is not None and timing.refined
        sf_refined += int(applied)
        u_sf = anchor_uncertainty_for_bounce(
            calib,
            sf_anchor.details["pixel_xy"],
            base_sigma_m=0.05,
            ball_radius_m=TT_BALL_RADIUS_M,
            fps=traj.fps,
            vertical_speed_mps=TT_BOUNCE_VERTICAL_SPEED_MPS,
            horizontal_speed_mps=TT_BOUNCE_HORIZONTAL_SPEED_MPS,
            speed_cv=TT_BOUNCE_SPEED_CV,
            subframe_timing_sd_s=timing.timing_sd_s if applied else None,
        )
        sf_along.append(u_sf.sigma_along_ray_m)
        sf_perp.append(u_sf.sigma_perp_m)
        sf_bias.append(u_sf.bias_along_ray_m)

    depth = np.asarray(depth)
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    img = np.concatenate([img1, img2])
    legacy_sigma = np.asarray(legacy_sigma)

    out = {
        "n": int(depth.size),
        "pre_fix_f29145a_isotropic": {
            "reported_sigma_m_median": float(np.median(legacy_sigma)),
            "depth": _axis_stats(depth, legacy_sigma),
            "img1": _axis_stats(img1, legacy_sigma),
            "img2": _axis_stats(img2, legacy_sigma),
        },
    }
    # Counterfactual: what a single scalar does even when it is sized CORRECTLY.
    # The honest scalar is the worst axis, so score all three axes against it.
    m_along = np.asarray(ray["measured_physics"]["along"])
    m_perp = np.asarray(ray["measured_physics"]["perp"])
    m_bias = np.asarray(ray["measured_physics"]["bias"])
    worst = np.maximum(m_along, m_perp)
    out["corrected_but_still_isotropic"] = {
        "reported_sigma_m_median": float(np.median(worst)),
        "note": (
            "The corrected sigma consumed as ONE scalar. Depth becomes honest and the image "
            "plane becomes over-conservative: this is why two numbers are required."
        ),
        "depth_bias_corrected": _axis_stats(depth - m_bias, worst),
        "image_plane_pooled": _axis_stats(img, np.concatenate([worst, worst])),
    }

    for key, arrays in ray.items():
        along = np.asarray(arrays["along"])
        perp = np.asarray(arrays["perp"])
        bias = np.asarray(arrays["bias"])
        out[f"ray_aligned__{key}"] = {
            "sigma_along_ray_m_median": float(np.median(along)),
            "sigma_perp_m_median": float(np.median(perp)),
            "bias_along_ray_m_median": float(np.median(bias)),
            "anisotropy_along_over_perp": float(np.median(along) / np.median(perp)),
            # Uncorrected: the bias is reported but NOT removed from the anchor.
            "depth_bias_uncorrected": _axis_stats(depth, along),
            # Corrected: the consumer subtracted the reported bias.
            "depth_bias_corrected": _axis_stats(depth - bias, along),
            "img1": _axis_stats(img1, perp),
            "img2": _axis_stats(img2, perp),
            "image_plane_pooled": _axis_stats(img, np.concatenate([perp, perp])),
        }

    # SUB-FRAME: the anchor is placed at the estimated contact instant instead
    # of at the marked frame, and its uncertainty carries the estimator's own
    # timing sigma. Scored against the same ground-truth contacts.
    sfd = np.asarray(sf_depth)
    sfi = np.concatenate([np.asarray(sf_img1), np.asarray(sf_img2)])
    sfa = np.asarray(sf_along)
    sfp = np.asarray(sf_perp)
    sfb = np.asarray(sf_bias)
    out["ray_aligned__subframe_measured_physics"] = {
        "n_refined": int(sf_refined),
        "n": int(sfd.size),
        "sigma_along_ray_m_median": float(np.median(sfa)),
        "sigma_perp_m_median": float(np.median(sfp)),
        "bias_along_ray_m_median": float(np.median(sfb)),
        "anisotropy_along_over_perp": float(np.median(sfa) / np.median(sfp)),
        "depth_bias_uncorrected": _axis_stats(sfd, sfa),
        "depth_bias_corrected": _axis_stats(sfd - sfb, sfa),
        "img1": _axis_stats(np.asarray(sf_img1), sfp),
        "img2": _axis_stats(np.asarray(sf_img2), sfp),
        "image_plane_pooled": _axis_stats(sfi, np.concatenate([sfp, sfp])),
    }
    return out


def _print_table(results: dict) -> None:
    print("\nBEFORE -- isotropic scalar as of f29145a (the defect)")
    print(f"{'view/cond':<18}{'sigma':>8}{'d_rms':>9}{'d_ratio':>9}{'d_in1s':>8}"
          f"{'i_rms':>9}{'i_ratio':>9}{'i_in1s':>8}{'d_bias':>9}")
    for k, r in results.items():
        legacy = r["pre_fix_f29145a_isotropic"]
        sigma = legacy["reported_sigma_m_median"]
        i_rms = float(np.sqrt((legacy["img1"]["rms_m"] ** 2 + legacy["img2"]["rms_m"] ** 2) / 2))
        print(f"{k:<18}{sigma:>8.4f}{legacy['depth']['rms_m']:>9.4f}"
              f"{legacy['depth']['implied_sigma_over_reported']:>9.2f}"
              f"{legacy['depth']['frac_within_1sigma']:>8.2f}{i_rms:>9.4f}"
              f"{i_rms / sigma:>9.2f}"
              f"{(legacy['img1']['frac_within_1sigma'] + legacy['img2']['frac_within_1sigma']) / 2:>8.2f}"
              f"{legacy['depth']['bias_m']:>9.4f}")

    for key, title in (
        ("measured_physics", "AFTER -- ray-aligned, TT3D's own measured bounce speeds"),
        ("library_defaults", "AFTER -- ray-aligned, library defaults (pickleball priors: 4.0 / 8.0 m/s)"),
    ):
        print(f"\n{title}; bias REMOVED by the consumer")
        print(f"{'view/cond':<18}{'sig_along':>10}{'sig_perp':>9}{'aniso':>7}{'bias':>8}"
              f"{'d_ratio':>9}{'d_in1s':>8}{'i_ratio':>9}{'i_in1s':>8}{'d_resid':>9}")
        for k, r in results.items():
            ray = r[f"ray_aligned__{key}"]
            dep = ray["depth_bias_corrected"]
            img = ray["image_plane_pooled"]
            print(f"{k:<18}{ray['sigma_along_ray_m_median']:>10.4f}"
                  f"{ray['sigma_perp_m_median']:>9.4f}{ray['anisotropy_along_over_perp']:>7.1f}"
                  f"{ray['bias_along_ray_m_median']:>8.4f}"
                  f"{dep['implied_sigma_over_reported']:>9.2f}{dep['frac_within_1sigma']:>8.2f}"
                  f"{img['implied_sigma_over_reported']:>9.2f}{img['frac_within_1sigma']:>8.2f}"
                  f"{dep['bias_m']:>9.4f}")

    print("\nCOUNTERFACTUAL -- the corrected sigma squeezed back into ONE scalar (worst axis)")
    print(f"{'view/cond':<18}{'sigma':>8}{'d_ratio':>9}{'d_in1s':>8}{'i_ratio':>9}{'i_in1s':>8}")
    for k, r in results.items():
        c = r["corrected_but_still_isotropic"]
        print(f"{k:<18}{c['reported_sigma_m_median']:>8.4f}"
              f"{c['depth_bias_corrected']['implied_sigma_over_reported']:>9.2f}"
              f"{c['depth_bias_corrected']['frac_within_1sigma']:>8.2f}"
              f"{c['image_plane_pooled']['implied_sigma_over_reported']:>9.2f}"
              f"{c['image_plane_pooled']['frac_within_1sigma']:>8.2f}")

    print("\nAFTER (measured physics), bias REPORTED but NOT removed -- the default posture")
    print(f"{'view/cond':<18}{'d_ratio':>9}{'d_in1s':>8}{'d_bias':>9}")
    for k, r in results.items():
        dep = r["ray_aligned__measured_physics"]["depth_bias_uncorrected"]
        print(f"{k:<18}{dep['implied_sigma_over_reported']:>9.2f}"
              f"{dep['frac_within_1sigma']:>8.2f}{dep['bias_m']:>9.4f}")

    print("\nSUB-FRAME -- anchor placed at the estimated contact instant, bias NOT removed")
    print(f"{'view/cond':<18}{'refined':>8}{'sig_along':>10}{'sig_perp':>9}{'bias':>8}"
          f"{'d_rms':>8}{'d_ratio':>9}{'d_in1s':>8}{'i_ratio':>9}{'i_in1s':>8}{'d_bias':>9}")
    for k, r in results.items():
        s = r["ray_aligned__subframe_measured_physics"]
        dep = s["depth_bias_uncorrected"]
        img = s["image_plane_pooled"]
        print(f"{k:<18}{s['n_refined']:>4}/{s['n']:<3}{s['sigma_along_ray_m_median']:>10.4f}"
              f"{s['sigma_perp_m_median']:>9.4f}{s['bias_along_ray_m_median']:>8.4f}"
              f"{dep['rms_m']:>8.4f}{dep['implied_sigma_over_reported']:>9.2f}"
              f"{dep['frac_within_1sigma']:>8.2f}{img['implied_sigma_over_reported']:>9.2f}"
              f"{img['frac_within_1sigma']:>8.2f}{dep['bias_m']:>9.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="runs/lanes/tt3d_external_validation_20260726/sigma_calibration.json",
    )
    args = ap.parse_args()

    results = {}
    for name in VIEWS:
        v = TT3DView.from_yaml(DATA_ROOT / f"{name}.yaml", name)
        for cond in ("no_noise", "noise"):
            sub = f"{name}_no_noise" if cond == "no_noise" else name
            results[f"{name}/{cond}"] = analyse(v, sub)

    _print_table(results)

    payload = {
        "measured_tt3d_bounce_physics": {
            "vertical_speed_mps": TT_BOUNCE_VERTICAL_SPEED_MPS,
            "horizontal_speed_mps": TT_BOUNCE_HORIZONTAL_SPEED_MPS,
            "speed_cv": TT_BOUNCE_SPEED_CV,
            "note": (
                "Measured from TT3D ground truth, supplied as model INPUT. These are "
                "physical properties of table tennis, not parameters fitted to the "
                "error being predicted."
            ),
        },
        "coverage_note": (
            "rms_ratio ~ 1.0 and frac_within_1sigma ~ 0.683 cannot both hold unless the "
            "error is Gaussian. The dominant bounce-anchor term is sub-frame timing, which "
            "is uniform; a uniform error at rms_ratio 1.0 gives frac_within_1sigma 0.577, "
            "not 0.683."
        ),
        "views": results,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
