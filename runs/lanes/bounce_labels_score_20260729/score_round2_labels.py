#!/usr/bin/env python3
"""Score the owner's fresh (round-2, 2026-07-28/29) ball labels against the
production solver, per clip, following the 2026-07-26 pilot's methodology
(runs/lanes/ball_label_tool_20260726/REPORT.md).

VERIFIED=0. Human labels are review-only, not verified ground truth. Only
`bounce` labels have a solved depth (ray-plane intersection); `near_player`
and `free_flight` numbers below are estimate-vs-estimate comparisons, never
accuracy claims, and are kept in a clearly separate table.

Per NORTH_STAR_ROADMAP.md §2.3: reprojection error is never used to gate,
band, or promote any 3D quantity here. All comparisons below are 3D-to-3D
(human ray-plane world_xyz vs. solver world_xyz) or 2D-to-2D (pixel), never
reprojection. `depth_unvalidated` language from the source artifacts is
preserved in the output, not laundered into a stronger claim.

Read-only against everything outside this lane's own output directory.
"""
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LANE_OUT = Path(__file__).resolve().parent

LABEL_ROUND2_DIR = REPO_ROOT / "runs/lanes/ball_labels_round2_20260728"

# clip_key -> (label dir under LABEL_ROUND2_DIR, run dir holding
# ball_track_arc_solved.json, or None if no live solver artifact exists)
CLIPS = {
    "wolverine": {
        "label_dir": LABEL_ROUND2_DIR / "wolverine",
        "run_dir": REPO_ROOT
        / "runs/lanes/ball_f1_three_clip_runs_20260705/wolverine_mixed_0200_mid_steep_corner",
    },
    "burlington": {
        "label_dir": LABEL_ROUND2_DIR / "burlington",
        "run_dir": REPO_ROOT
        / "runs/lanes/ball_f1_three_clip_runs_20260705/burlington_gold_0300_low_steep_corner",
    },
    "outdoor_webcam": {
        "label_dir": LABEL_ROUND2_DIR / "outdoor_webcam",
        "run_dir": REPO_ROOT
        / "runs/full_mesh_examples_20260725/outdoor_mesh_final/outdoor_webcam_20s_fullmesh_final",
    },
    "pbv11": {
        "label_dir": LABEL_ROUND2_DIR / "pbv11",
        "run_dir": None,  # no ball_track_arc_solved.json exists for this clip (README, confirmed on disk)
    },
}

# Calibration floor, read live off each clip's own ball_human_labels.json
# calibration_evidence.plane_residual_check (matches the round README's
# headline numbers: wolverine 0.127 m / burlington 0.191 m / outdoor 0.101 m
# / pbv11 0.144 m median).
CALIBRATION_FLOOR_ROUNDED = {
    "wolverine": 0.127,
    "burlington": 0.191,
    "outdoor_webcam": 0.101,
    "pbv11": 0.144,
}

BOUNCE_BAND_PRIORITY = [
    "anchored_measured",
    "arc_interpolated",
    "arc_extrapolated",
    "arc_weak",
    "hidden",
]


def dist3(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def median(xs):
    return statistics.median(xs) if xs else None


def p90(xs):
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    # linear-interpolation percentile, matches numpy's default ('linear')
    k = 0.90 * (len(xs) - 1)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def load_solver_frames(run_dir):
    if run_dir is None:
        return None
    p = run_dir / "ball_track_arc_solved.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    frames = d["frames"]
    assert isinstance(frames, list)
    return frames


def solver_lookup(frames, frame_idx):
    if frames is None or frame_idx >= len(frames):
        return None
    f = frames[frame_idx]
    return {
        "band": f.get("band"),
        "visible": f.get("visible"),
        "world_xyz": f.get("world_xyz"),
        "xy": f.get("xy"),
        "sigma_m": f.get("sigma_m"),
        "not_for_detection_metrics": f.get("not_for_detection_metrics"),
    }


def score_clip(clip_key, cfg):
    label_path = cfg["label_dir"] / "ball_human_labels.json"
    labels_doc = json.loads(label_path.read_text())
    labels = labels_doc["labels"]
    cal = labels_doc["calibration_evidence"]["plane_residual_check"]
    cal_floor_m = cal["median_m"]

    solver_frames = load_solver_frames(cfg["run_dir"])
    live_solver_available = solver_frames is not None

    by_kind = {"bounce": [], "near_player": [], "free_flight": []}
    for lab in labels:
        by_kind.setdefault(lab["kind"], []).append(lab)

    n_total = len(labels)
    n_prefill_corrected = sum(1 for l in labels if l["origin"] == "prefill_corrected")
    n_prefill_confirmed = sum(1 for l in labels if l["origin"] == "prefill_confirmed")
    n_fresh = sum(1 for l in labels if l["origin"] == "fresh")

    click_sigma_along_by_kind = {
        k: [l["sigma_along_ray_m"] for l in v] for k, v in by_kind.items()
    }

    # ---- bounce: live-solver vs human-corrected-click table ----
    bounce_rows = []
    for lab in by_kind["bounce"]:
        frame_idx = lab["frame"]
        human_xyz = lab["world_xyz_m"]
        human_px = lab["pixel_xy"]
        row = {
            "frame": frame_idx,
            "label_id": lab["label_id"],
            "origin": lab["origin"],
            "human_world_xyz_m": human_xyz,
            "human_pixel_xy": human_px,
            "human_sigma_along_ray_m": lab["sigma_along_ray_m"],
            "prefill_used": lab["prefill"] is not None,
        }
        if lab["prefill"] is not None:
            row["prefill_band"] = lab["prefill"]["band"]
            row["prefill_delta_m_asrecorded"] = lab["prefill"]["delta_m"]
            row["prefill_delta_px_asrecorded"] = lab["prefill"]["delta_px"]

        live = solver_lookup(solver_frames, frame_idx) if live_solver_available else None
        row["live_solver_available_for_clip"] = live_solver_available
        if live is not None:
            row["live_band"] = live["band"]
            row["live_visible"] = live["visible"]
            if live["world_xyz"] is not None:
                row["live_world_xyz_m"] = live["world_xyz"]
                row["error_3d_m"] = dist3(human_xyz, live["world_xyz"])
            else:
                row["live_world_xyz_m"] = None
                row["error_3d_m"] = None  # band=hidden -> no position to compare
            if live["xy"] is not None:
                row["live_xy_px"] = live["xy"]
                row["error_2d_px"] = dist2(human_px, live["xy"])
            else:
                row["live_xy_px"] = None
                row["error_2d_px"] = None
        else:
            row["live_band"] = None
            row["error_3d_m"] = None
            row["error_2d_px"] = None
        bounce_rows.append(row)

    # aggregate bounce error by live band (only rows with a live solver position)
    by_band = {}
    for row in bounce_rows:
        band = row.get("live_band")
        if band is None or row.get("error_3d_m") is None:
            continue
        by_band.setdefault(band, {"error_3d_m": [], "error_2d_px": []})
        by_band[band]["error_3d_m"].append(row["error_3d_m"])
        by_band[band]["error_2d_px"].append(row["error_2d_px"])

    band_summary = {}
    for band, vals in by_band.items():
        band_summary[band] = {
            "n": len(vals["error_3d_m"]),
            "error_3d_m_median": median(vals["error_3d_m"]),
            "error_3d_m_p90": p90(vals["error_3d_m"]),
            "error_3d_m_max": max(vals["error_3d_m"]),
            "error_2d_px_median": median(vals["error_2d_px"]),
            "error_2d_px_p90": p90(vals["error_2d_px"]),
        }

    all_bounce_3d = [r["error_3d_m"] for r in bounce_rows if r.get("error_3d_m") is not None]
    all_bounce_2d = [r["error_2d_px"] for r in bounce_rows if r.get("error_2d_px") is not None]

    # ---- near_player / free_flight: review-only, estimate-vs-estimate ----
    review_only_rows = []
    for kind in ("near_player", "free_flight"):
        for lab in by_kind[kind]:
            frame_idx = lab["frame"]
            human_xyz = lab["world_xyz_m"]
            live = solver_lookup(solver_frames, frame_idx) if live_solver_available else None
            row = {
                "kind": kind,
                "frame": frame_idx,
                "label_id": lab["label_id"],
                "origin": lab["origin"],
                "human_sigma_along_ray_m": lab["sigma_along_ray_m"],
                "estimate_vs_estimate_only": True,
                "not_an_accuracy_claim": True,
            }
            if lab["prefill"] is not None:
                row["prefill_band"] = lab["prefill"]["band"]
                row["prefill_delta_m_asrecorded"] = lab["prefill"]["delta_m"]
            if live is not None and live["world_xyz"] is not None:
                row["live_band"] = live["band"]
                row["est_delta_3d_m"] = dist3(human_xyz, live["world_xyz"])
            review_only_rows.append(row)

    return {
        "clip_key": clip_key,
        "clip_id": labels_doc["clip_id"],
        "verified_ground_truth": labels_doc["verified_ground_truth"],
        "review_only": labels_doc["review_only"],
        "calibration_floor_median_m_measured": cal_floor_m,
        "calibration_floor_median_m_task_rounded": CALIBRATION_FLOOR_ROUNDED[clip_key],
        "calibration_floor_p95_m": cal.get("p95_m"),
        "calibration_floor_max_m": cal.get("max_m"),
        "live_solver_artifact_available": live_solver_available,
        "label_counts": {
            "total": n_total,
            "by_kind": {k: len(v) for k, v in by_kind.items()},
            "by_origin": {
                "fresh": n_fresh,
                "prefill_confirmed": n_prefill_confirmed,
                "prefill_corrected": n_prefill_corrected,
            },
            "prefill_corrected_fraction_of_total": (
                n_prefill_corrected / n_total if n_total else None
            ),
            "prefill_touched_fraction_of_total": (
                (n_prefill_corrected + n_prefill_confirmed) / n_total if n_total else None
            ),
        },
        "click_sigma_along_ray_m_median_by_kind": {
            k: median(v) for k, v in click_sigma_along_by_kind.items()
        },
        "bounce_vs_live_solver": {
            "n_bounce_labels": len(bounce_rows),
            "n_with_live_solver_position": len(all_bounce_3d),
            "pooled_error_3d_m_median": median(all_bounce_3d),
            "pooled_error_3d_m_p90": p90(all_bounce_3d),
            "pooled_error_2d_px_median": median(all_bounce_2d),
            "pooled_error_2d_px_p90": p90(all_bounce_2d),
            "by_band": band_summary,
            "rows": bounce_rows,
        },
        "near_player_and_free_flight_review_only": {
            "note": (
                "Depth is a human judgement (near_player) or an unreferenced "
                "human estimate (free_flight), never solved. These numbers are "
                "estimate-vs-estimate comparisons against the solver's own "
                "(also-unvalidated-in-depth) output, banded here as estimates, "
                "and must never be aggregated with the bounce table above or "
                "read as an accuracy measurement."
            ),
            "rows": review_only_rows,
        },
    }


def main():
    per_clip = {}
    for clip_key, cfg in CLIPS.items():
        per_clip[clip_key] = score_clip(clip_key, cfg)

    # ---- pooled bounce-vs-solver table across clips with live solver data ----
    pooled_by_band = {}
    pooled_3d = []
    pooled_2d = []
    for clip_key, result in per_clip.items():
        for band, vals in result["bounce_vs_live_solver"]["by_band"].items():
            pooled_by_band.setdefault(band, {"error_3d_m": [], "error_2d_px": []})
        for row in result["bounce_vs_live_solver"]["rows"]:
            if row.get("error_3d_m") is None:
                continue
            band = row["live_band"]
            pooled_by_band[band]["error_3d_m"].append(row["error_3d_m"])
            pooled_by_band[band]["error_2d_px"].append(row["error_2d_px"])
            pooled_3d.append(row["error_3d_m"])
            pooled_2d.append(row["error_2d_px"])

    pooled_band_summary = {}
    for band, vals in pooled_by_band.items():
        if not vals["error_3d_m"]:
            continue
        pooled_band_summary[band] = {
            "n": len(vals["error_3d_m"]),
            "error_3d_m_median": median(vals["error_3d_m"]),
            "error_3d_m_p90": p90(vals["error_3d_m"]),
            "error_3d_m_max": max(vals["error_3d_m"]),
            "error_2d_px_median": median(vals["error_2d_px"]),
            "error_2d_px_p90": p90(vals["error_2d_px"]),
        }

    # ---- arc_weak vs anchored_measured separation check (the pilot's headline) ----
    arc_weak_3d = pooled_by_band.get("arc_weak", {}).get("error_3d_m", [])
    anchored_3d = pooled_by_band.get("anchored_measured", {}).get("error_3d_m", [])
    separation_check = {
        "arc_weak_n": len(arc_weak_3d),
        "arc_weak_error_3d_m_median": median(arc_weak_3d),
        "arc_weak_error_3d_m_max": max(arc_weak_3d) if arc_weak_3d else None,
        "arc_weak_error_3d_m_min": min(arc_weak_3d) if arc_weak_3d else None,
        "anchored_measured_n": len(anchored_3d),
        "anchored_measured_error_3d_m_median": median(anchored_3d),
        "anchored_measured_error_3d_m_max": max(anchored_3d) if anchored_3d else None,
        "separation_holds_on_fresh_labels": (
            (median(arc_weak_3d) > median(anchored_3d))
            if (arc_weak_3d and anchored_3d)
            else None
        ),
        "pilot_reference_2026_07_26": {
            "arc_weak_range_m": [2.5, 24.8],
            "anchored_measured_approx_m": 0.3,
            "note": "wolverine-only, 19 pilot labels, per NORTH_STAR_ROADMAP.md",
        },
    }

    total_labels = sum(r["label_counts"]["total"] for r in per_clip.values())
    total_bounce = sum(r["label_counts"]["by_kind"]["bounce"] for r in per_clip.values())

    out = {
        "artifact_type": "bounce_labels_score_report",
        "lane": "bounce_labels_score_20260729",
        "verified_ground_truth": False,
        "review_only": True,
        "not_ground_truth": True,
        "methodology": (
            "Follows runs/lanes/ball_label_tool_20260726/REPORT.md (2026-07-26 pilot). "
            "Bounce depth is solved by ray-plane intersection at z=BALL_RADIUS_M=0.0371m "
            "(never a human depth judgement); error is compared 3D-to-3D and 2D-to-2D "
            "against the live ball_track_arc_solved.json frame at the labeled frame index, "
            "independent of whether the label's origin used the P-key prefill. "
            "near_player/free_flight are human depth estimates and are reported "
            "separately as review-only, never aggregated with bounce. "
            "No 3D quantity is gated, banded, or promoted on reprojection error "
            "(NORTH_STAR_ROADMAP.md section 2.3); reprojection is not used anywhere "
            "in this scoring. Calibration floor is carried per clip, not folded away."
        ),
        "totals": {
            "n_clips": len(per_clip),
            "n_labels_total": total_labels,
            "n_bounce_labels_total": total_bounce,
        },
        "pooled_bounce_vs_live_solver": {
            "note": (
                "Pooled across wolverine, burlington, outdoor_webcam only "
                "(pbv11 has no ball_track_arc_solved.json -- confirmed absent on disk, "
                "per the round-2 README's disclosed 'About clip 4' limitation)."
            ),
            "n_with_live_solver_position": len(pooled_3d),
            "error_3d_m_median": median(pooled_3d),
            "error_3d_m_p90": p90(pooled_3d),
            "error_2d_px_median": median(pooled_2d),
            "error_2d_px_p90": p90(pooled_2d),
            "by_band": pooled_band_summary,
        },
        "arc_weak_vs_anchored_measured_separation_check": separation_check,
        "per_clip": per_clip,
    }

    out_path = LANE_OUT / "bounce_labels_score_report.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=False))
    print(f"wrote {out_path}")

    # small human-readable summary too
    summary = {
        "totals": out["totals"],
        "pooled_bounce_vs_live_solver": out["pooled_bounce_vs_live_solver"],
        "arc_weak_vs_anchored_measured_separation_check": separation_check,
        "per_clip_headline": {
            ck: {
                "clip_id": r["clip_id"],
                "label_counts": r["label_counts"],
                "calibration_floor_median_m": r["calibration_floor_median_m_measured"],
                "live_solver_artifact_available": r["live_solver_artifact_available"],
                "bounce_pooled_error_3d_m_median": r["bounce_vs_live_solver"][
                    "pooled_error_3d_m_median"
                ],
                "bounce_pooled_error_3d_m_p90": r["bounce_vs_live_solver"][
                    "pooled_error_3d_m_p90"
                ],
                "bounce_by_band": {
                    b: {
                        "n": v["n"],
                        "error_3d_m_median": v["error_3d_m_median"],
                        "error_3d_m_p90": v["error_3d_m_p90"],
                    }
                    for b, v in r["bounce_vs_live_solver"]["by_band"].items()
                },
            }
            for ck, r in per_clip.items()
        },
    }
    summary_path = LANE_OUT / "bounce_labels_score_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=False))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
