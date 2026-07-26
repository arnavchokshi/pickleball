"""Per-axis calibration of `anchor_sigma_for_bounce` against measured error.

The solver consumes the anchor sigma isotropically -- ball_arc_solver.py:2357
does `_scaled_vec(position - anchor.world_xyz, anchor_sigma)`, dividing all
three world components by the SAME scalar. An honest isotropic 1-sigma
therefore requires each Cartesian component of the anchor error to behave like
N(0, sigma): ~68.3% of |component| within 1 sigma, and RMS(component) ~ sigma.

This script measures that per component in the camera-aligned frame
(depth-along-ray, and the two image-plane axes).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tt3d"))

from tt3d_adapter import TT_BALL_RADIUS_M, TT3DView, iter_trajectories  # noqa: E402
from run_tt3d_validation import DATA_ROOT, VIEWS, decompose, estimate_contact  # noqa: E402
from threed.racketsport.ball_arc_solver import (  # noqa: E402
    anchor_sigma_for_bounce,
    build_bounce_anchor,
)


def analyse(view: TT3DView, sub: str) -> dict:
    calib = view.calibration()
    cam_c = view.camera_center_world
    cam_right, _, _ = view.camera_basis()
    comps = {"depth": [], "img1": [], "img2": []}
    sig = []
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
        comps["depth"].append(a)
        comps["img1"].append(b)
        comps["img2"].append(c)
        sig.append(anchor_sigma_for_bounce(calib, (float(uv[0]), float(uv[1])), base_sigma_m=0.05))

    sig = np.asarray(sig)
    out = {"n": int(sig.size), "reported_sigma_m_median": float(np.median(sig))}
    for key, vals in comps.items():
        v = np.asarray(vals)
        rms = float(np.sqrt(np.mean(v**2)))
        out[key] = {
            "bias_m": float(np.mean(v)),
            "rms_m": rms,
            "implied_sigma_over_reported": float(rms / np.median(sig)),
            "frac_within_1sigma": float(np.mean(np.abs(v) <= sig)),
            "p95_abs_m": float(np.percentile(np.abs(v), 95)),
        }
    return out


def main() -> int:
    results = {}
    for name in VIEWS:
        v = TT3DView.from_yaml(DATA_ROOT / f"{name}.yaml", name)
        for cond in ("no_noise", "noise"):
            sub = f"{name}_no_noise" if cond == "no_noise" else name
            results[f"{name}/{cond}"] = analyse(v, sub)

    print(f"{'view/cond':<18}{'sigma':>8}{'depth_rms':>11}{'d_ratio':>9}{'d_in1s':>8}"
          f"{'img_rms':>10}{'i_ratio':>9}{'i_in1s':>8}{'depth_bias':>11}")
    for k, r in results.items():
        img_rms = float(np.sqrt((r["img1"]["rms_m"] ** 2 + r["img2"]["rms_m"] ** 2) / 2))
        print(f"{k:<18}{r['reported_sigma_m_median']:>8.4f}{r['depth']['rms_m']:>11.4f}"
              f"{r['depth']['implied_sigma_over_reported']:>9.2f}{r['depth']['frac_within_1sigma']:>8.2f}"
              f"{img_rms:>10.4f}{img_rms / r['reported_sigma_m_median']:>9.2f}"
              f"{(r['img1']['frac_within_1sigma'] + r['img2']['frac_within_1sigma']) / 2:>8.2f}"
              f"{r['depth']['bias_m']:>11.4f}")

    out = REPO_ROOT / "runs/lanes/tt3d_external_validation_20260726/sigma_calibration.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    print("\nHonest isotropic 1-sigma would give ratio ~1.00 and frac_within_1sigma ~0.683 "
          "on EVERY axis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
