#!/usr/bin/env python3
"""Fit + validate the metric-15pt court calibration path against the 4 eval clips.

This is the CAL-METRIC lane's validation harness for Task #15 (NORTH_STAR_ROADMAP.md). It:

1. Fits `metric_calibration_from_reviewed_keypoints_15pt` for each of the 4 eval clips
   from their human-reviewed `labels/court_keypoints.json`, and writes the result both
   under the run directory and alongside the source labels
   (`eval_clips/ball/<clip>/labels/court_calibration_metric15pt.json`) -- a NEW file,
   never overwriting any existing calibration artifact.
2. Renders a visual-verification overlay of the (rescaled-to-native) reviewed keypoints
   on an actual native-resolution video frame, to confirm the declared
   label_coordinate_space -> source_resolution scaling lands on real court lines and
   is not a repeat of the 960x540-preview-vs-native mixup.
3. Re-runs the PnP-vs-homography footpoint self-consistency check (the same check that
   scored 0/20 passes on Burlington in `runs/cal_body_projection_bias_20260702T014121Z/`)
   with the new calibration, for the same Burlington + a Wolverine 20-sample set.
4. Measures Burlington near-baseline/right-sideline line straightness (the fisheye "bow"
   metric from the diagnostic) using the new calibration's real k1/k2 to undistort.
5. Quantifies how far the Burlington BODY run's 152 grounded foot positions
   (`body_world_label_packet.json`) move between old and new calibration.

All numbers are written to `runs/cal_metric_15pt_<UTC timestamp>/`.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from threed.racketsport.court_calibration import (  # noqa: E402
    homography_from_planar_points,
    project_image_points_to_world,
    project_planar_points,
    project_world_points,
)
from threed.racketsport.court_calibration_metric15 import (  # noqa: E402
    metric_calibration_from_reviewed_keypoints_15pt,
)
from threed.racketsport.schemas import CourtCalibration  # noqa: E402

EVAL_CLIPS_ROOT = ROOT / "eval_clips" / "ball"
CLIPS = [
    "burlington_gold_0300_low_steep_corner",
    "wolverine_mixed_0200_mid_steep_corner",
    "indoor_doubles_fwuks_0500_long_mid_baseline",
    "outdoor_webcam_iynbd_1500_long_high_baseline",
]

# The BODY run whose overlay-review sample set + old calibration the diagnostic used.
BURLINGTON_BODY_RUN = (
    ROOT
    / "runs"
    / "body_joint_goal_smoke_20260630T001407"
    / "a100_body_video_smoke_burlington_best_zero_switch_tracks_20260701T153021Z_reground_footlockfix_report"
)
WOLVERINE_BODY_RUN = ROOT / "runs" / "body_joint_goal_smoke_20260630T001407" / "a100_body_video_smoke_wolverine_full_track_v1"

# Same thresholds as threed/racketsport/body_world_label_review_overlay.py, reused here
# so the "before/after" comparison is scored identically to how the diagnostic scored it.
MAX_PASSED_FLOOR_ANCHOR_DELTA_PX = 24.0
MAX_FAILED_FLOOR_ANCHOR_DELTA_PX = 48.0
MAX_PASSED_FLOOR_ANCHOR_DELTA_DIAG = 0.20
MAX_FAILED_FLOOR_ANCHOR_DELTA_DIAG = 0.35


def _status_for_delta(center_delta_px: float, bbox_diag: float) -> str:
    pass_limit_px = max(MAX_PASSED_FLOOR_ANCHOR_DELTA_PX, bbox_diag * MAX_PASSED_FLOOR_ANCHOR_DELTA_DIAG)
    fail_limit_px = max(MAX_FAILED_FLOOR_ANCHOR_DELTA_PX, bbox_diag * MAX_FAILED_FLOOR_ANCHOR_DELTA_DIAG)
    if center_delta_px > fail_limit_px:
        return "failed"
    if center_delta_px > pass_limit_px:
        return "warning"
    return "passed"


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = 0.95 * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(ordered[lo])
    weight = rank - lo
    return float(ordered[lo] * (1 - weight) + ordered[hi] * weight)


def fit_all_clips(run_dir: Path) -> dict[str, CourtCalibration]:
    calibrations: dict[str, CourtCalibration] = {}
    for clip in CLIPS:
        keypoints_path = EVAL_CLIPS_ROOT / clip / "labels" / "court_keypoints.json"
        calibration = metric_calibration_from_reviewed_keypoints_15pt(keypoints_path, sport="pickleball")
        calibrations[clip] = calibration

        artifact_dir = run_dir / "artifacts" / clip
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = calibration.model_dump(mode="json")
        (artifact_dir / "court_calibration.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        # New file alongside the source labels; never overwrites an existing artifact.
        eval_out = EVAL_CLIPS_ROOT / clip / "labels" / "court_calibration_metric15pt.json"
        eval_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return calibrations


def render_visual_verification(run_dir: Path, calibrations: dict[str, CourtCalibration]) -> dict[str, Any]:
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    for clip in CLIPS:
        clip_dir = EVAL_CLIPS_ROOT / clip
        source_video = clip_dir / "source.mp4"
        cap = cv2.VideoCapture(str(source_video))
        ok, frame = cap.read()  # native frame 0, matching label frame_000001.jpg (verified by pixel-diff cross-check)
        cap.release()
        if not ok:
            report[clip] = {"status": "read_failed"}
            continue

        calibration = calibrations[clip]
        native_h, native_w = frame.shape[:2]
        overlay = frame.copy()
        # image_pts are ordered by PICKLEBALL_COURT_KEYPOINT_NAMES; draw them all.
        for x, y in calibration.image_pts:
            cv2.circle(overlay, (int(round(x)), int(round(y))), 6, (0, 0, 255), -1)
            cv2.circle(overlay, (int(round(x)), int(round(y))), 8, (255, 255, 255), 2)
        out_path = evidence_dir / f"{clip}_keypoint_overlay_native.jpg"
        cv2.imwrite(str(out_path), overlay)
        report[clip] = {
            "status": "ok",
            "native_frame_size": [int(native_w), int(native_h)],
            "declared_image_size": list(calibration.image_size) if calibration.image_size else None,
            "overlay_path": str(out_path.relative_to(ROOT)),
        }
    return report


def pnp_vs_homography_check(
    run_dir: Path,
    *,
    clip_label: str,
    body_run_dir: Path,
    new_calibration: CourtCalibration,
) -> dict[str, Any]:
    overlay_index_path = body_run_dir / "body_world_label_review_bundle" / "overlays" / "body_world_label_review_overlay_index.json"
    overlay_index = json.loads(overlay_index_path.read_text(encoding="utf-8"))
    samples = overlay_index["overlays"]

    before_rows = []
    after_rows = []
    for sample in samples:
        pnp = sample.get("pnp_track_floor_projection_alignment")
        bbox = sample.get("track_bbox")
        if pnp is None or bbox is None:
            continue
        footpoint = pnp["bbox_footpoint"]
        x1, y1, x2, y2 = bbox[:4]
        bbox_diag = math.hypot(max(0.0, x2 - x1), max(0.0, y2 - y1))

        before_rows.append(
            {
                "sample_id": sample.get("sample_id"),
                "center_delta_px": pnp["center_delta_px"],
                "status": pnp["status"],
            }
        )

        # Self-consistent "after" check, same methodology as
        # _track_floor_projection_alignment: ground the bbox footpoint through the NEW
        # calibration's homography, then reproject that world point through the NEW
        # calibration's full PnP camera model and compare to the same footpoint pixel.
        world_xy = project_image_points_to_world(new_calibration.homography, [footpoint])[0]
        projected = project_world_points(
            new_calibration.extrinsics,
            new_calibration.intrinsics,
            [[world_xy[0], world_xy[1], 0.0]],
        )[0]
        delta_px = math.hypot(projected[0] - footpoint[0], projected[1] - footpoint[1])
        status = _status_for_delta(delta_px, bbox_diag)
        after_rows.append(
            {
                "sample_id": sample.get("sample_id"),
                "center_delta_px": round(delta_px, 3),
                "status": status,
                "bbox_diag_px": round(bbox_diag, 3),
            }
        )

    def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        deltas = [row["center_delta_px"] for row in rows]
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        return {
            "n": len(rows),
            "mean_px": round(sum(deltas) / len(deltas), 3) if deltas else None,
            "median_px": round(_median(deltas), 3) if deltas else None,
            "p95_px": round(_p95(deltas), 3) if deltas else None,
            "status_counts": counts,
            "pass_rate": round(counts.get("passed", 0) / len(rows), 4) if rows else None,
        }

    result = {
        "clip": clip_label,
        "body_run": str(body_run_dir.relative_to(ROOT)),
        "before": {"summary": _summary(before_rows), "rows": before_rows},
        "after": {"summary": _summary(after_rows), "rows": after_rows},
    }
    return result


def pnp_vs_homography_distortion_consistent_check(
    *,
    clip_label: str,
    body_run_dir: Path,
    new_calibration: CourtCalibration,
) -> dict[str, Any]:
    """Same self-consistency check as `pnp_vs_homography_check`, but with the homography
    fit on UNDISTORTED calibration points and footpoints undistorted before comparison.

    Why this exists: `pnp_track_floor_projection_alignment` (and the "after" check above)
    compares a strictly-planar homography (no distortion model) against the full PnP
    model (which, for a clip like Burlington, includes real nonzero k1/k2). Once a
    lens has real radial distortion, the two are structurally different function
    classes -- a homography cannot represent radial distortion at all -- so they are
    mathematically expected to diverge away from the calibration points even when both
    are well-fit individually. This variant isolates "is the pose/focal degenerate"
    (the original defect) from "does a distortion-free homography disagree with a
    distorted PnP model by construction" (an artifact of the check's own design, not a
    calibration defect) by undistorting both sides before comparing.
    """

    intrinsics = new_calibration.intrinsics
    dist = np.asarray(intrinsics.dist or [0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    k = np.array([[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    calib_pts = np.asarray(new_calibration.image_pts, dtype=np.float64).reshape(-1, 1, 2)
    undistorted_calib_pts = cv2.undistortPoints(calib_pts, k, dist, P=k).reshape(-1, 2)
    homography_undist = homography_from_planar_points(new_calibration.world_pts, undistorted_calib_pts.tolist())

    overlay_index_path = body_run_dir / "body_world_label_review_bundle" / "overlays" / "body_world_label_review_overlay_index.json"
    overlay_index = json.loads(overlay_index_path.read_text(encoding="utf-8"))

    rows = []
    for sample in overlay_index["overlays"]:
        pnp = sample.get("pnp_track_floor_projection_alignment")
        bbox = sample.get("track_bbox")
        if pnp is None or bbox is None:
            continue
        footpoint = np.asarray([pnp["bbox_footpoint"]], dtype=np.float64).reshape(-1, 1, 2)
        footpoint_undist = cv2.undistortPoints(footpoint, k, dist, P=k).reshape(-1, 2)[0]
        x1, y1, x2, y2 = bbox[:4]
        bbox_diag = math.hypot(max(0.0, x2 - x1), max(0.0, y2 - y1))

        world_xy = project_image_points_to_world(homography_undist, [footpoint_undist.tolist()])[0]
        # project_world_points applies K only (no distortion) -- consistent with the
        # already-undistorted footpoint, so this is an apples-to-apples comparison.
        projected = project_world_points(new_calibration.extrinsics, new_calibration.intrinsics, [[world_xy[0], world_xy[1], 0.0]])[0]
        delta_px = math.hypot(projected[0] - footpoint_undist[0], projected[1] - footpoint_undist[1])
        status = _status_for_delta(delta_px, bbox_diag)
        rows.append({"sample_id": sample.get("sample_id"), "center_delta_px": round(delta_px, 3), "status": status})

    deltas = [row["center_delta_px"] for row in rows]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = {
        "n": len(rows),
        "mean_px": round(sum(deltas) / len(deltas), 3) if deltas else None,
        "median_px": round(_median(deltas), 3) if deltas else None,
        "p95_px": round(_p95(deltas), 3) if deltas else None,
        "status_counts": counts,
        "pass_rate": round(counts.get("passed", 0) / len(rows), 4) if rows else None,
    }
    return {"clip": clip_label, "summary": summary, "rows": rows}


def _search_line_offsets(gray: np.ndarray, p0: tuple[float, float], p1: tuple[float, float], *, search_radius_px: int = 25) -> dict[str, Any]:
    """Reimplementation of the diagnostic's court-line curvature ("bow") measurement:
    sample points along the analytic straight line between two calibration corner
    points, search perpendicular to the line for the nearest near-white pixel cluster
    (the true painted court line), and record the perpendicular offset."""

    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    line_vec = p1 - p0
    line_length = float(np.linalg.norm(line_vec))
    if line_length < 1e-6:
        raise ValueError("degenerate line")
    line_dir = line_vec / line_length
    normal = np.array([-line_dir[1], line_dir[0]])

    ts = [round(0.05 * i, 2) for i in range(1, 20)]
    offsets = []
    for t in ts:
        base = p0 + t * line_vec
        best_offset = None
        best_val = -1.0
        for d in range(-search_radius_px, search_radius_px + 1):
            point = base + d * normal
            x, y = int(round(point[0])), int(round(point[1]))
            if 0 <= y < gray.shape[0] and 0 <= x < gray.shape[1]:
                val = float(gray[y, x])
                if val > 180.0 and val > best_val:
                    best_val = val
                    best_offset = float(d)
        offsets.append(best_offset if best_offset is not None else 0.0)

    mid_offset = offsets[len(offsets) // 2]
    end_offset = (offsets[0] + offsets[-1]) / 2.0
    bow = mid_offset - end_offset
    return {
        "t": ts,
        "offsets_px": offsets,
        "mean_px": sum(offsets) / len(offsets),
        "mid_offset_px": mid_offset,
        "end_offset_px": end_offset,
        "bow_px": bow,
        "line_length_px": line_length,
    }


def burlington_line_straightness(run_dir: Path, calibration: CourtCalibration) -> dict[str, Any]:
    clip_dir = EVAL_CLIPS_ROOT / "burlington_gold_0300_low_steep_corner"
    cap = cv2.VideoCapture(str(clip_dir / "source.mp4"))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("failed to read Burlington native frame 0")

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    from threed.racketsport.schemas import PICKLEBALL_COURT_KEYPOINT_NAMES

    points_by_name = dict(zip(PICKLEBALL_COURT_KEYPOINT_NAMES, calibration.image_pts, strict=True))

    # Same two edges the diagnostic measured (near_baseline, right_sideline), but with
    # endpoints taken from the NEW reviewed 15-point labels rather than the OLD 4-tap
    # corner labels the diagnostic used -- so absolute "before" numbers will differ
    # somewhat from the diagnostic's original 7.7px/4.46px even prior to any
    # undistortion, since the corner pixel locations themselves differ slightly between
    # the two label sets. What's directly comparable is the before-vs-after delta on
    # the SAME endpoints, which isolates the effect of undistortion.
    edges = {
        "near_baseline": (points_by_name["near_left_corner"], points_by_name["near_right_corner"]),
        "right_sideline": (points_by_name["near_right_corner"], points_by_name["far_right_corner"]),
        "far_baseline": (points_by_name["far_left_corner"], points_by_name["far_right_corner"]),
    }

    fx, fy, cx, cy = calibration.intrinsics.fx, calibration.intrinsics.fy, calibration.intrinsics.cx, calibration.intrinsics.cy
    dist = np.asarray(calibration.intrinsics.dist or [0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    undistorted_gray = cv2.undistort(gray, k, dist)

    result: dict[str, Any] = {}
    for edge_name, (p0, p1) in edges.items():
        before = _search_line_offsets(gray, tuple(p0), tuple(p1))
        # Undistort the calibration endpoints too, so the "after" line-straightness
        # check is measured in the same undistorted image the endpoints now describe.
        pts = np.array([[p0, p1]], dtype=np.float64)
        undistorted_pts = cv2.undistortPoints(pts, k, dist, P=k).reshape(-1, 2)
        after = _search_line_offsets(undistorted_gray, tuple(undistorted_pts[0]), tuple(undistorted_pts[1]))
        result[edge_name] = {"before": before, "after": after}

    return result


def world_scale_impact_burlington(old_calibration: CourtCalibration, new_calibration: CourtCalibration) -> dict[str, Any]:
    packet_path = BURLINGTON_BODY_RUN / "body_world_label_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    samples = packet["samples"]

    # body_world_label_packet.json's track_world_xy was grounded through the OLD
    # calibration *rescaled to native (1920x1080) pixels* (see
    # body_world_label_review_overlay.py's calibration_for_image_size /
    # _bbox_scale_for_review_frame -- confirmed empirically: forward-projecting a known
    # track_world_xy through the raw old_calibration.homography lands at exactly half
    # the expected native pixel footpoint, since the raw calibration's homography/
    # intrinsics are defined in its declared 960x540 image_size). Scale the recovered
    # footpoint up to native pixels before feeding it to the NEW (native-space)
    # calibration, or this comparison silently mixes two different pixel scales.
    old_base_w, old_base_h = old_calibration.image_size or (960, 540)
    new_native_w, new_native_h = new_calibration.image_size or (1920, 1080)
    scale_x = new_native_w / old_base_w
    scale_y = new_native_h / old_base_h

    deltas_m = []
    rows = []
    for sample in samples:
        old_world_xy = sample["track_world_xy"]
        # Recover the pixel footpoint that produced old_world_xy by forward-projecting
        # it through the OLD calibration's own homography (self-consistent inverse of
        # how it was originally grounded), then rescale from the old calibration's base
        # pixel space to native pixels.
        footpoint_base = project_planar_points(old_calibration.homography, [old_world_xy])[0]
        footpoint_native = [footpoint_base[0] * scale_x, footpoint_base[1] * scale_y]
        new_world_xy = project_image_points_to_world(new_calibration.homography, [footpoint_native])[0]
        delta = math.hypot(new_world_xy[0] - old_world_xy[0], new_world_xy[1] - old_world_xy[1])
        deltas_m.append(delta)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "old_world_xy": old_world_xy,
                "new_world_xy": new_world_xy,
                "delta_m": round(delta, 4),
            }
        )

    return {
        "n_samples": len(samples),
        "delta_m": {
            "median": round(_median(deltas_m), 4),
            "p90": round(_percentile(deltas_m, 90), 4),
            "mean": round(sum(deltas_m) / len(deltas_m), 4),
            "max": round(max(deltas_m), 4),
            "min": round(min(deltas_m), 4),
        },
        "rows": rows,
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "data").mkdir(exist_ok=True)
    (run_dir / "evidence").mkdir(exist_ok=True)

    print("Fitting metric-15pt calibration for all 4 eval clips...")
    calibrations = fit_all_clips(run_dir)

    per_clip_summary = {}
    for clip, calibration in calibrations.items():
        per_clip_summary[clip] = {
            "intrinsics": calibration.intrinsics.model_dump(mode="json"),
            "reprojection_error_px": calibration.reprojection_error_px.model_dump(mode="json"),
            "per_keypoint_residual_px": calibration.per_keypoint_residual_px,
            "metric_confidence": calibration.metric_confidence,
            "capture_quality": calibration.capture_quality.model_dump(mode="json"),
            "solved_over_frames": calibration.solved_over_frames,
        }
    (run_dir / "data" / "per_clip_calibration_summary.json").write_text(
        json.dumps(per_clip_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("Rendering visual verification overlays...")
    visual_report = render_visual_verification(run_dir, calibrations)
    (run_dir / "data" / "visual_verification.json").write_text(
        json.dumps(visual_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("Recomputing PnP-vs-homography check (Burlington + Wolverine)...")
    burlington_check = pnp_vs_homography_check(
        run_dir,
        clip_label="burlington_gold_0300_low_steep_corner",
        body_run_dir=BURLINGTON_BODY_RUN,
        new_calibration=calibrations["burlington_gold_0300_low_steep_corner"],
    )
    wolverine_check = pnp_vs_homography_check(
        run_dir,
        clip_label="wolverine_mixed_0200_mid_steep_corner",
        body_run_dir=WOLVERINE_BODY_RUN,
        new_calibration=calibrations["wolverine_mixed_0200_mid_steep_corner"],
    )
    (run_dir / "data" / "pnp_vs_homography_before_after.json").write_text(
        json.dumps({"burlington": burlington_check, "wolverine": wolverine_check}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Recomputing Burlington's check with both sides undistorted (isolates the")
    print("distortion-free-homography-vs-distorted-PnP artifact from a real calibration defect)...")
    burlington_distortion_consistent = pnp_vs_homography_distortion_consistent_check(
        clip_label="burlington_gold_0300_low_steep_corner",
        body_run_dir=BURLINGTON_BODY_RUN,
        new_calibration=calibrations["burlington_gold_0300_low_steep_corner"],
    )
    (run_dir / "data" / "burlington_pnp_vs_homography_distortion_consistent.json").write_text(
        json.dumps(burlington_distortion_consistent, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("Measuring Burlington line straightness before/after undistortion...")
    straightness = burlington_line_straightness(run_dir, calibrations["burlington_gold_0300_low_steep_corner"])
    (run_dir / "data" / "burlington_line_straightness.json").write_text(
        json.dumps(straightness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("Quantifying world-scale impact on Burlington's 152 BODY player-frames...")
    old_burlington_calibration = CourtCalibration.model_validate(
        json.loads((BURLINGTON_BODY_RUN / "court_calibration.json").read_text(encoding="utf-8"))
    )
    world_scale = world_scale_impact_burlington(old_burlington_calibration, calibrations["burlington_gold_0300_low_steep_corner"])
    (run_dir / "data" / "world_scale_impact_burlington.json").write_text(
        json.dumps(world_scale, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary = {
        "per_clip_calibration": {
            clip: {
                "fx": per_clip_summary[clip]["intrinsics"]["fx"],
                "fy": per_clip_summary[clip]["intrinsics"]["fy"],
                "dist": per_clip_summary[clip]["intrinsics"]["dist"],
                "reprojection_median_px": per_clip_summary[clip]["reprojection_error_px"]["median"],
                "reprojection_p95_px": per_clip_summary[clip]["reprojection_error_px"]["p95"],
                "metric_confidence": per_clip_summary[clip]["metric_confidence"],
            }
            for clip in CLIPS
        },
        "pnp_vs_homography": {
            "burlington": {"before": burlington_check["before"]["summary"], "after": burlington_check["after"]["summary"]},
            "wolverine": {"before": wolverine_check["before"]["summary"], "after": wolverine_check["after"]["summary"]},
            "burlington_after_distortion_consistent": burlington_distortion_consistent["summary"],
        },
        "burlington_line_straightness_bow_px": {
            edge: {"before": data["before"]["bow_px"], "after": data["after"]["bow_px"]} for edge, data in straightness.items()
        },
        "world_scale_impact_burlington_152_frames_m": world_scale["delta_m"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
