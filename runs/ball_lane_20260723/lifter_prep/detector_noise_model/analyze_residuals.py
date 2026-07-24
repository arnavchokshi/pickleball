"""Empirical detector-noise + occlusion residual characterization (WS5, freeze-safe).

Measures — from EXISTING human labels vs EXISTING WASB detector outputs only —
the pixel-residual, miss/false-positive, and missing-detection gap statistics
that a future pose-conditioned synthetic generator (PLAN.md Phase B1) must
inject. No model is trained, no solver behavior changes. VERIFIED=0: the
labels themselves are human-reviewed clicks flagged `not_ground_truth` by
their own artifacts; every number here is a measurement of those artifacts,
not a verified accuracy claim.

Data sets (all read-only, main checkout):
  SET A  eval_clips/ball/{4 clips}/labels/ball_points.json   (sparse human clicks)
     vs  runs/lanes/ball_tracking_track_regen_20260704/tracks/{clip}/
         wasb_tennis_zeroshot_thr_0_500/source_wasb_predictions.csv (dense, per-frame)
  SET B  runs/lanes/w3_reviewimport_20260707/normalized_cvat/{6 clips}/annotations.xml
         (sparse human CVAT keyframes, stride ~18)
     vs  data/online_harvest_20260706/prelabels/{clip}/ball_track.json (dense raw WASB)
  BLUR   runs/lanes/ball_anchor_boost_20260712/{burlington,wolverine}_ball_blur_sidecar.json
         (per-frame blur measurements; computed on the ballcand_20260710 baseline track —
         association caveat recorded in the output)

Deterministic output: sorted keys, floats rounded to 6 decimals, no timestamps.
Self-contained: stdlib + numpy only. Re-run:
  /Users/arnavchokshi/Desktop/pickleball/.venv/bin/python analyze_residuals.py \
      [--repo-root /Users/arnavchokshi/Desktop/pickleball] [--out residual_analysis.json]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_ball_detector_noise_model"
FLOAT_DECIMALS = 6
ASSOC_RADIUS_PX = 20.0  # matches the repo's F1@20 convention
TIGHT_RADIUS_PX = 10.0
# A label whose center sits within this distance of the detector output is
# treated as a detector-seeded CVAT prelabel the annotator left unmoved
# (circular; uninformative for localization noise). Measured during WS5:
# 126/209 SET B pairs are <= 0.5 px from the WASB prelabel; SET A is 1/70.
COINCIDENCE_EPS_PX = 0.5
GAP_BUCKETS = [(1, 2), (3, 5), (6, 11), (12, 29), (30, 59), (60, 10**9)]

# ---------------------------------------------------------------------------
# Dataset table. `regime` provenance:
#   clip_metadata  = eval_clips clip_metadata.json `environment` field
#   visual_screen  = one screening thumbnail per harvest video, visually
#                    classified during WS5 (approximate, unverified)
# fps values transcribed from `frame_rate_fps` in each clip's registered
# eval_clips clip_metadata.json (verified against the ledger-registered files).
# ---------------------------------------------------------------------------
SET_A = [
    # clip, regime, regime_source, fps
    ("burlington_gold_0300_low_steep_corner", "indoor", "clip_metadata", 60.0),
    ("indoor_doubles_fwuks_0500_long_mid_baseline", "indoor", "clip_metadata", 30.0),
    ("outdoor_webcam_iynbd_1500_long_high_baseline", "outdoor_day", "clip_metadata+visual_screen", 60.0),
    ("wolverine_mixed_0200_mid_steep_corner", "outdoor_night", "clip_metadata+visual_screen", 30.0),
]
SET_B = [
    ("_L0HVmAlCQI_rally_0001", "outdoor_night", "visual_screen"),
    ("73VurrTKCZ8_rally_0002", "outdoor_day", "visual_screen"),
    ("Ezz6HDNHlnk_rally_0004", "outdoor_night", "visual_screen"),
    ("HyUqT7zFiwk_rally_0001", "indoor", "visual_screen"),
    ("wBu8bC4OfUY_rally_0001", "outdoor_night", "visual_screen"),
    ("zwCtH_i1_S4_rally_0001", "outdoor_day", "visual_screen"),
]
BLUR_SIDECARS = {
    "burlington_gold_0300_low_steep_corner": "runs/lanes/ball_anchor_boost_20260712/burlington_ball_blur_sidecar.json",
    "wolverine_mixed_0200_mid_steep_corner": "runs/lanes/ball_anchor_boost_20260712/wolverine_ball_blur_sidecar.json",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def round_floats(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (float, np.floating)):
        v = round(float(value), FLOAT_DECIMALS)
        return 0.0 if v == 0 else v
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {k: round_floats(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_labels_set_a(path: Path) -> dict[int, dict]:
    doc = json.loads(path.read_text())
    labels: dict[int, dict] = {}
    for item in doc["items"]:
        idx = int(item["frame_index"])
        labels[idx] = {
            "visible": bool(item.get("visible")),
            "xy": [float(item["xy_px"][0]), float(item["xy_px"][1])] if item.get("visible") else None,
        }
    return labels


def load_preds_set_a(path: Path) -> dict[int, dict]:
    preds: dict[int, dict] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            idx = int(row["Frame"])
            vis = int(row["Visibility"]) == 1
            preds[idx] = {
                "visible": vis,
                "xy": [float(row["X"]), float(row["Y"])] if vis else None,
                "conf": float(row["Confidence"]),
            }
    return preds


def load_labels_set_b(path: Path) -> dict[int, dict]:
    """CVAT keyframe boxes: outside=0 -> visible click (box center);
    outside=1 -> human 'not visible at this sampled frame' marker (box coords
    are CVAT carry-over and are ignored). Visible wins on frame collisions."""
    root = ET.parse(path).getroot()
    labels: dict[int, dict] = {}
    for track in sorted(root.findall(".//track"), key=lambda t: int(t.attrib.get("id", 0))):
        if track.attrib.get("label") != "ball":
            continue
        for box in track.findall("box"):
            if box.attrib.get("keyframe") != "1":
                continue
            frame = int(box.attrib["frame"])
            visible = box.attrib.get("outside") == "0"
            cx = (float(box.attrib["xtl"]) + float(box.attrib["xbr"])) / 2.0
            cy = (float(box.attrib["ytl"]) + float(box.attrib["ybr"])) / 2.0
            record = {"visible": visible, "xy": [cx, cy] if visible else None}
            if frame in labels and labels[frame]["visible"] and not visible:
                continue  # keep the visible assertion
            labels[frame] = record
    return labels


def load_preds_set_b(path: Path) -> tuple[dict[int, dict], float]:
    doc = json.loads(path.read_text())
    preds: dict[int, dict] = {}
    for idx, frame in enumerate(doc["frames"]):
        vis = bool(frame.get("visible"))
        preds[idx] = {
            "visible": vis,
            "xy": [float(frame["xy"][0]), float(frame["xy"][1])] if vis else None,
            "conf": float(frame.get("conf", 0.0)),
        }
    return preds, float(doc.get("fps") or 0.0)


def load_blur_sidecar(path: Path) -> dict[int, float]:
    doc = json.loads(path.read_text())
    return {
        int(f["frame_index"]): float(f["blur_length_px"])
        for f in doc.get("frames", [])
        if f.get("blur_length_px") is not None
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def quantiles(arr: np.ndarray, qs=(50, 90, 95, 99)) -> dict:
    if arr.size == 0:
        return {f"p{q}": None for q in qs}
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs}


def axis_stats(deltas: np.ndarray) -> dict:
    """deltas: (n,) signed per-axis residuals."""
    if deltas.size == 0:
        return {"n": 0}
    med = float(np.median(deltas))
    mad = float(np.median(np.abs(deltas - med)))
    m2 = float(np.mean((deltas - deltas.mean()) ** 2))
    kurt = float(np.mean((deltas - deltas.mean()) ** 4) / (m2**2) - 3.0) if m2 > 0 else None
    out = {
        "n": int(deltas.size),
        "mean_bias_px": float(deltas.mean()),
        "std_px": float(deltas.std(ddof=1)) if deltas.size > 1 else None,
        "median_px": med,
        "mad_px": mad,
        "mad_sigma_px": mad * 1.4826,
        "excess_kurtosis": kurt,
        "abs": quantiles(np.abs(deltas)),
    }
    return out


def radial_stats(r: np.ndarray) -> dict:
    if r.size == 0:
        return {"n": 0}
    stats = {"n": int(r.size), "median_px": float(np.median(r)), "mean_px": float(r.mean())}
    stats.update(quantiles(r))
    for thr in (10, 20, 50, 100):
        stats[f"frac_gt_{thr}px"] = float(np.mean(r > thr))
    return stats


def pair_clip(labels: dict[int, dict], preds: dict[int, dict]) -> dict:
    """Join sparse labels against dense predictions at labeled frames.

    Each pair is (frame, dx, dy, r, conf, coincident) where `coincident`
    means the label sits within COINCIDENCE_EPS_PX of the detector output —
    for CVAT-prelabel-seeded labels this indicates an unmoved seed, which is
    circular evidence and excluded from localization-noise estimates.
    """
    pairs = []
    misses, fp_hidden, correct_reject, no_pred_row = [], [], [], []
    for frame in sorted(labels):
        lab = labels[frame]
        pred = preds.get(frame)
        if pred is None:
            no_pred_row.append(frame)
            continue
        if lab["visible"]:
            if pred["visible"]:
                dx = pred["xy"][0] - lab["xy"][0]
                dy = pred["xy"][1] - lab["xy"][1]
                r = float(np.hypot(dx, dy))
                pairs.append((frame, dx, dy, r, pred["conf"], r <= COINCIDENCE_EPS_PX))
            else:
                misses.append(frame)
        else:
            if pred["visible"]:
                fp_hidden.append((frame, pred["conf"]))
            else:
                correct_reject.append(frame)
    return {
        "pairs": pairs,
        "misses": misses,
        "fp_hidden": fp_hidden,
        "correct_reject": correct_reject,
        "no_pred_row": no_pred_row,
    }


def independent(pairs: list) -> list:
    """Pairs usable for localization-noise estimation (non-coincident)."""
    return [p for p in pairs if not p[5]]


def gap_analysis(preds: dict[int, dict], fps: float) -> dict:
    """Runs of consecutive frames with no visible detection, interior to the
    first/last visible detection (boundary runs are censored and only counted)."""
    frames = sorted(preds)
    vis = np.array([preds[f]["visible"] for f in frames], dtype=bool)
    n = vis.size
    visible_idx = np.flatnonzero(vis)
    result = {
        "n_frames": int(n),
        "n_visible": int(visible_idx.size),
        "detection_coverage": float(visible_idx.size / n) if n else None,
    }
    if visible_idx.size < 2:
        result.update({"n_interior_gaps": 0, "censored_boundary_frames": int(n - visible_idx.size)})
        return result
    gaps = []
    prev = visible_idx[0]
    for idx in visible_idx[1:]:
        if idx - prev > 1:
            gaps.append(int(idx - prev - 1))
        prev = idx
    gaps_arr = np.array(gaps, dtype=float)
    buckets = {}
    for lo, hi in GAP_BUCKETS:
        key = f"{lo}-{hi}f" if hi < 10**9 else f"{lo}+f"
        buckets[key] = int(np.sum((gaps_arr >= lo) & (gaps_arr <= hi))) if gaps else 0
    result.update(
        {
            "n_interior_gaps": len(gaps),
            "gap_frames": quantiles(gaps_arr) | {"max": float(gaps_arr.max()) if gaps else None},
            "gap_ms": (
                {k: (None if v is None else v * 1000.0 / fps) for k, v in quantiles(gaps_arr).items()}
                | {"max": float(gaps_arr.max() * 1000.0 / fps) if gaps else None}
                if fps
                else None
            ),
            "gap_frames_histogram": buckets,
            "censored_boundary_frames": int(visible_idx[0] + (n - 1 - visible_idx[-1])),
            "fps_used_for_ms": fps or None,
        }
    )
    return result


def residual_block(pairs: list) -> dict:
    """Full residual characterization for a list of (frame,dx,dy,r,conf)."""
    if not pairs:
        return {"n_paired_detections": 0}
    dx = np.array([p[1] for p in pairs])
    dy = np.array([p[2] for p in pairs])
    r = np.array([p[3] for p in pairs])
    inl = r <= ASSOC_RADIUS_PX
    out = r > ASSOC_RADIUS_PX
    block = {
        "n_paired_detections": len(pairs),
        "all_pairs": {"dx": axis_stats(dx), "dy": axis_stats(dy), "radial": radial_stats(r)},
        "inliers_le_20px": {
            "n": int(inl.sum()),
            "rate": float(inl.mean()),
            "dx": axis_stats(dx[inl]),
            "dy": axis_stats(dy[inl]),
            "radial": radial_stats(r[inl]),
            "frac_le_10px": float(np.mean(r <= TIGHT_RADIUS_PX)),
        },
        "outliers_gt_20px": {
            "n": int(out.sum()),
            "rate": float(out.mean()),
            "displacement_px": quantiles(r[out]) if out.any() else None,
            "note": "detector fired far from the labeled ball at a labeled-visible frame; inject as a mis-association/false-peak mode, not as Gaussian jitter",
        },
    }
    return block


def rate_block(paired: dict) -> dict:
    n_vis = len(paired["pairs"]) + len(paired["misses"])
    n_hid = len(paired["fp_hidden"]) + len(paired["correct_reject"])
    fp_confs = np.array([c for _, c in paired["fp_hidden"]])
    return {
        "labeled_visible_frames": n_vis,
        "detector_fired_at_visible": len(paired["pairs"]),
        "miss_count": len(paired["misses"]),
        "miss_rate_at_labeled_visible": float(len(paired["misses"]) / n_vis) if n_vis else None,
        "labeled_not_visible_frames": n_hid,
        "fp_count_at_labeled_hidden": len(paired["fp_hidden"]),
        "fp_rate_at_labeled_hidden": float(len(paired["fp_hidden"]) / n_hid) if n_hid else None,
        "fp_confidence": (
            {"median": float(np.median(fp_confs)), "p90": float(np.percentile(fp_confs, 90))}
            if fp_confs.size
            else None
        ),
    }


def confidence_bins(pairs: list) -> list:
    bins = [(0.0, 0.7), (0.7, 0.9), (0.9, 1.01)]
    rows = []
    for lo, hi in bins:
        sel = [p for p in pairs if lo <= p[4] < hi]
        r = np.array([p[3] for p in sel])
        rows.append(
            {
                "conf_range": [lo, min(hi, 1.0)],
                "n": len(sel),
                "median_radial_px": float(np.median(r)) if sel else None,
                "inlier_le_20px_rate": float(np.mean(r <= ASSOC_RADIUS_PX)) if sel else None,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default="/Users/arnavchokshi/Desktop/pickleball")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "residual_analysis.json"))
    args = ap.parse_args()
    root = Path(args.repo_root)

    inputs = []  # provenance: root-relative path + sha256

    def register(rel: str) -> Path:
        path = root / rel
        inputs.append({"path": rel, "sha256": sha256_of(path)})
        return path

    clips = {}
    pooled = {"all": [], "indoor": [], "outdoor_day": [], "outdoor_night": []}
    pooled_paired = {
        "all": {"pairs": [], "misses": [], "fp_hidden": [], "correct_reject": [], "no_pred_row": []},
    }
    for reg in ("indoor", "outdoor_day", "outdoor_night"):
        pooled_paired[reg] = {"pairs": [], "misses": [], "fp_hidden": [], "correct_reject": [], "no_pred_row": []}

    blur_join_pairs = []  # (blur_length_px, radial_residual_px)

    per_set_pairs = {"A_eval_clips": [], "B_online_harvest_cvat": []}

    def accumulate(clip: str, set_name: str, regime: str, regime_source: str,
                   labels: dict, preds: dict, fps: float, blur_map: dict | None) -> None:
        paired = pair_clip(labels, preds)
        for key in pooled_paired["all"]:
            pooled_paired["all"][key].extend(paired[key])
            pooled_paired[regime][key].extend(paired[key])
        per_set_pairs[set_name].extend(paired["pairs"])
        if blur_map:
            for frame, _, _, r, _, _ in paired["pairs"]:
                if frame in blur_map:
                    blur_join_pairs.append((blur_map[frame], r))
        indep = independent(paired["pairs"])
        clips[clip] = {
            "set": set_name,
            "regime": regime,
            "regime_source": regime_source,
            "n_labeled_frames": len(labels),
            "n_labeled_visible": sum(1 for v in labels.values() if v["visible"]),
            "label_provenance": {
                "n_paired_detections": len(paired["pairs"]),
                "n_coincident_with_prediction_le_0p5px": len(paired["pairs"]) - len(indep),
                "n_independent": len(indep),
            },
            "rates": rate_block(paired),
            "residuals_independent_labels": residual_block(indep),
            "missing_detection_gaps": gap_analysis(preds, fps),
            "frames_without_prediction_row": len(paired["no_pred_row"]),
        }

    # SET A
    for clip, regime, regime_source, fps in SET_A:
        lab_path = register(f"eval_clips/ball/{clip}/labels/ball_points.json")
        pred_path = register(
            f"runs/lanes/ball_tracking_track_regen_20260704/tracks/{clip}/wasb_tennis_zeroshot_thr_0_500/source_wasb_predictions.csv"
        )
        register(f"eval_clips/ball/{clip}/clip_metadata.json")
        blur_map = None
        if clip in BLUR_SIDECARS:
            blur_map = load_blur_sidecar(register(BLUR_SIDECARS[clip]))
        accumulate(clip, "A_eval_clips", regime, regime_source,
                   load_labels_set_a(lab_path), load_preds_set_a(pred_path), fps, blur_map)

    # SET B
    for clip, regime, regime_source in SET_B:
        lab_path = register(f"runs/lanes/w3_reviewimport_20260707/normalized_cvat/{clip}/annotations.xml")
        pred_path = register(f"data/online_harvest_20260706/prelabels/{clip}/ball_track.json")
        preds, fps = load_preds_set_b(pred_path)
        accumulate(clip, "B_online_harvest_cvat", regime, regime_source,
                   load_labels_set_b(lab_path), preds, fps, None)

    # Pooled blocks — residual/confidence estimates use ONLY independent
    # (non-coincident) label pairs; rate estimates use all labeled frames.
    def pooled_block(paired: dict) -> dict:
        indep = independent(paired["pairs"])
        return {
            "rates": rate_block(paired),
            "label_provenance": {
                "n_paired_detections": len(paired["pairs"]),
                "n_coincident_with_prediction_le_0p5px": len(paired["pairs"]) - len(indep),
                "n_independent": len(indep),
            },
            "residuals_independent_labels": residual_block(indep),
            "confidence_bins_independent_pairs": confidence_bins(indep),
        }

    by_regime = {reg: pooled_block(pooled_paired[reg]) for reg in ("indoor", "outdoor_day", "outdoor_night")}
    by_set = {
        "A_eval_clips_independent_clicks": residual_block(independent(per_set_pairs["A_eval_clips"])),
        "B_cvat_human_moved_only": residual_block(independent(per_set_pairs["B_online_harvest_cvat"])),
        "B_cvat_coincident_seed_count": len(per_set_pairs["B_online_harvest_cvat"])
        - len(independent(per_set_pairs["B_online_harvest_cvat"])),
    }

    blur_block: dict
    if blur_join_pairs:
        blur_arr = np.array(blur_join_pairs)  # (n, 2): blur_length_px, radial residual
        med_blur = float(np.median(blur_arr[:, 0]))
        low = blur_arr[blur_arr[:, 0] <= med_blur]
        high = blur_arr[blur_arr[:, 0] > med_blur]
        blur_block = {
            "n_joined_label_frames": int(blur_arr.shape[0]),
            "clips_with_sidecar": sorted(BLUR_SIDECARS),
            "sidecar_track_caveat": "blur sidecars were computed along the ballcand_20260710 baseline track, not along the human labels; the join is by frame index only",
            "blur_length_px": quantiles(blur_arr[:, 0]) | {"median": med_blur},
            "split_at_median_blur": {
                "low_blur": {"n": int(low.shape[0]), "median_radial_residual_px": float(np.median(low[:, 1])) if low.size else None},
                "high_blur": {"n": int(high.shape[0]), "median_radial_residual_px": float(np.median(high[:, 1])) if high.size else None},
            },
        }
    else:
        blur_block = {"n_joined_label_frames": 0}

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "ball_verified": False,
        "not_ground_truth": True,
        "measurement_only": True,
        "association_radius_px": ASSOC_RADIUS_PX,
        "coincidence_eps_px": COINCIDENCE_EPS_PX,
        "definitions": {
            "paired_detection": "labeled-visible frame where the detector emitted a visible detection (any distance)",
            "coincident_pair": "paired detection where the label sits <= 0.5 px from the detector output; interpreted as an unmoved detector-seeded CVAT prelabel (circular; excluded from residual estimates)",
            "independent_pair": "paired detection whose label demonstrably did not copy the detector output (> 0.5 px away, or from the independently clicked SET A protocol)",
            "inlier": "independent pair with radial residual <= 20 px (localization-jitter mode)",
            "outlier_gt_20px": "independent pair > 20 px from the label (false-peak / mis-association mode)",
            "miss": "labeled-visible frame with no visible detection",
            "fp_at_labeled_hidden": "human-asserted not-visible frame where the detector fired",
            "missing_detection_gap": "interior run of consecutive frames with no visible detection in the DENSE detector track (upper bound on occlusion gaps; includes detector misses on visible balls)",
        },
        "label_caveats": [
            "SET A labels are human-reviewed clicks with status=corrected_unverified and not_ground_truth=true in their own artifact",
            "SET B labels are sparse CVAT keyframes (stride ~18 frames); outside=1 keyframes are treated as human not-visible assertions at that sampled frame",
            "SET B CVAT tasks were seeded from the same WASB prelabel tracks scored here: 126/209 paired labels sit <= 0.5 px from the detector output and are excluded from residual estimates as unmoved seeds; 4 of 6 clips are heavily affected, 73VurrTKCZ8_rally_0002 and wBu8bC4OfUY_rally_0001 show zero coincidences",
            "SET B human-moved residuals are conditioned on the annotator having chosen to move the seed, so they over-represent large errors; SET A is the primary localization-noise source",
            "regime tags marked visual_screen were assigned by visually inspecting one screening thumbnail per harvest video during WS5 and are approximate",
            "detector = WASB tennis zero-shot (heatmap threshold 0.5), NOT the w7 owner-retrained model; per-frame owner-retrain predictions do not exist locally (only aggregate LOSO reports were pulled from the VM)",
        ],
        "inputs": sorted(inputs, key=lambda x: x["path"]),
        "pooled_all": pooled_block(pooled_paired["all"]),
        "by_set": by_set,
        "by_regime": by_regime,
        "per_clip": {k: clips[k] for k in sorted(clips)},
        "blur_association": blur_block,
    }

    out = Path(args.out)
    out.write_text(json.dumps(round_floats(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
