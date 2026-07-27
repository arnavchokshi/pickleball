"""Sub-frame bounce timing measured on the owner's real pickleball labels.

WHAT THIS CAN AND CANNOT SHOW
-----------------------------
The 8 wolverine labels are human clicks on the bounce pixel, turned into 3D by
the SAME ray-plane intersection the solver uses. A human click at a marked
frame therefore inherits the SAME sub-frame overshoot as the solver's anchor.
So this file is NOT independent 3D ground truth on the depth axis and cannot
prove the position got more accurate. What it can measure honestly:

  1. The magnitude of the sub-frame correction on real pickleball at 30 fps --
     the estimated |dt|, the pixel it moves, and the resulting 3D shift. That
     is the size of the effect on our own footage, which table tennis at
     25 fps cannot tell us.
  2. Whether the solver prefill and the human label move CLOSER together. Both
     are biased the same way, so agreement is weak evidence, but a large
     divergence would be a red flag.
  3. Whether the estimator abstains or fires on real, gappy detector tracks.

Run:  PYTHONPATH=. python3 scripts/tt3d/owner_bounce_label_check.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from threed.racketsport.ball_arc_solver import (  # noqa: E402
    BallObservation,
    build_bounce_anchor,
    refine_bounce_contact_time,
)

LABELS = REPO_ROOT / "runs/lanes/ball_label_tool_20260726/labels/wolverine/ball_human_labels.json"
CALIBRATION = REPO_ROOT / (
    "eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/labels/"
    "court_calibration_metric15pt.json"
)
ARC_SOLVED = REPO_ROOT / (
    "runs/lanes/ballarc_anchorfusion_20260716/wolverine_no_soft_current/"
    "ball_track_arc_solved.json"
)
BALL_RADIUS_M = 0.0371


def load_observations(payload: dict[str, Any], fps: float) -> list[BallObservation]:
    """The clip's 2D ball track, exactly as the solver consumed it."""

    observations: list[BallObservation] = []
    for index, frame in enumerate(payload.get("frames") or []):
        if not isinstance(frame, dict) or frame.get("visible") is not True:
            continue
        xy = frame.get("xy")
        if not isinstance(xy, list) or len(xy) != 2:
            continue
        t = frame.get("t")
        observations.append(
            BallObservation(
                frame=index,
                t=float(t) if t is not None else index / fps,
                xy=(float(xy[0]), float(xy[1])),
                confidence=float(frame.get("conf") or 1.0),
                visible=True,
            )
        )
    return observations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="runs/lanes/subframe_bounce_timing_20260727/owner_label_check.json",
    )
    ap.add_argument(
        "--labels",
        default=str(LABELS),
        help="Label set to score. Defaults to the committed wolverine set; point it at a "
             "newer one when the owner has labelled more bounces.",
    )
    args = ap.parse_args()

    label_path = Path(args.labels)
    labels = json.loads(label_path.read_text())
    calibration = json.loads(CALIBRATION.read_text())
    arc = json.loads(ARC_SOLVED.read_text())
    fps = float(labels.get("fps") or 30.0)
    observations = load_observations(arc, fps)

    rows: list[dict[str, Any]] = []
    for label in labels["labels"]:
        if label.get("kind") != "bounce":
            continue
        frame = int(label["frame"])
        clicked = [float(v) for v in label["pixel_xy"]]
        human_xyz = [float(v) for v in label["world_xyz_m"]]
        prefill = label.get("prefill") or {}

        baseline = build_bounce_anchor(
            {"frame": frame, "t": float(label["timestamp_s"]), "fps": fps, "xy": clicked},
            calibration,
            ball_radius_m=BALL_RADIUS_M,
            status="human_reviewed",
        )
        timing = refine_bounce_contact_time(observations, frame, fps=fps)
        refined = build_bounce_anchor(
            {"frame": frame, "t": float(label["timestamp_s"]), "fps": fps, "xy": clicked},
            calibration,
            ball_radius_m=BALL_RADIUS_M,
            status="human_reviewed",
            subframe_timing=timing,
        )
        applied = timing is not None and timing.refined

        row: dict[str, Any] = {
            "frame": frame,
            "label_id": label["label_id"],
            "timing_status": "no_observation_at_frame" if timing is None else timing.status,
            "timing_applied": bool(applied),
            "human_label_world_xyz_m": human_xyz,
            "baseline_world_xyz_m": [round(v, 6) for v in baseline.world_xyz],
            "refined_world_xyz_m": [round(v, 6) for v in refined.world_xyz],
            "anchor_shift_m": round(math.dist(baseline.world_xyz, refined.world_xyz), 6),
            "reported_sigma_baseline_m": round(baseline.sigma_m, 6),
            "reported_sigma_refined_m": round(refined.sigma_m, 6),
            "reported_bias_baseline_m": round(
                baseline.details["uncertainty"]["bias_along_ray_m"], 6
            ),
            "reported_bias_refined_m": round(
                refined.details["uncertainty"]["bias_along_ray_m"], 6
            ),
        }
        if applied:
            row.update(
                {
                    "dt_from_frame_s": round(timing.dt_from_frame_s, 6),
                    "dt_from_frame_frames": round(timing.dt_from_frame_s * fps, 4),
                    "pixel_shift_px": round(timing.displacement_px, 4),
                    "fit_rms_px": round(timing.fit_rms_px, 4),
                    "timing_sd_s": round(timing.timing_sd_s, 6),
                    "observations_before": timing.observations_before,
                    "observations_after": timing.observations_after,
                }
            )
        if prefill.get("band") == "anchored_measured":
            solver_xyz = [float(v) for v in prefill["world_xyz_m"]]
            row["solver_prefill_world_xyz_m"] = solver_xyz
            row["prefill_band"] = prefill["band"]
            row["delta_solver_to_human_baseline_m"] = round(
                math.dist(solver_xyz, human_xyz), 6
            )
            row["delta_solver_to_human_refined_m"] = round(
                math.dist(solver_xyz, refined.world_xyz), 6
            )
        rows.append(row)

    applied_rows = [r for r in rows if r["timing_applied"]]
    anchored = [r for r in rows if r.get("prefill_band") == "anchored_measured"]
    payload = {
        "clip_id": labels["clip_id"],
        "sport": "pickleball",
        "fps": fps,
        "label_count": len(rows),
        "timing_applied_count": len(applied_rows),
        "guard_statuses": {
            status: sum(1 for r in rows if r["timing_status"] == status)
            for status in sorted({r["timing_status"] for r in rows})
        },
        "median_abs_dt_frames": (
            round(
                sorted(abs(r["dt_from_frame_frames"]) for r in applied_rows)[
                    len(applied_rows) // 2
                ],
                4,
            )
            if applied_rows
            else None
        ),
        "median_anchor_shift_m": (
            round(sorted(r["anchor_shift_m"] for r in applied_rows)[len(applied_rows) // 2], 6)
            if applied_rows
            else None
        ),
        "anchored_measured_prefills": {
            "count": len(anchored),
            "delta_before_m": [r["delta_solver_to_human_baseline_m"] for r in anchored],
            "delta_after_m": [r["delta_solver_to_human_refined_m"] for r in anchored],
        },
        "not_independent_ground_truth": True,
        "interpretation_limit": (
            "The human label's 3D position is itself a ray-plane intersection at a marked "
            "frame, so it carries the SAME sub-frame overshoot as the solver anchor. These "
            "numbers measure the SIZE of the correction on real pickleball, not its accuracy. "
            "VERIFIED=0 for pickleball bounce position."
        ),
        "inputs": {
            "labels": (
                str(label_path.relative_to(REPO_ROOT))
                if label_path.is_relative_to(REPO_ROOT)
                else str(label_path)
            ),
            "calibration": str(CALIBRATION.relative_to(REPO_ROOT)),
            "ball_track": str(ARC_SOLVED.relative_to(REPO_ROOT)),
        },
        "rows": rows,
    }

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{'frame':>6}{'status':>26}{'dt_frames':>11}{'px':>8}{'shift_m':>10}"
          f"{'sigma_before':>14}{'sigma_after':>13}{'bias_before':>13}{'bias_after':>12}")
    for r in rows:
        print(f"{r['frame']:>6}{r['timing_status']:>26}"
              f"{r.get('dt_from_frame_frames', float('nan')):>11.3f}"
              f"{r.get('pixel_shift_px', float('nan')):>8.2f}{r['anchor_shift_m']:>10.4f}"
              f"{r['reported_sigma_baseline_m']:>14.4f}{r['reported_sigma_refined_m']:>13.4f}"
              f"{r['reported_bias_baseline_m']:>13.4f}{r['reported_bias_refined_m']:>12.4f}")
    print(f"\napplied {len(applied_rows)}/{len(rows)}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
