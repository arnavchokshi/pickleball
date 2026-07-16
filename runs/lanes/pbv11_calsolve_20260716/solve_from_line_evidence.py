#!/usr/bin/env python3
"""Solve the pbv11 preview calibration from accepted temporal court-line evidence.

This lane-local reproducer deliberately uses only our owner seed's derived line
evidence.  It never reads the competitor export.  All 15 image correspondences
are intersections of accepted, temporally aggregated court-line observations.
The three y=0 correspondences are the *ground-net* line, not top-net points.

The single planar view cannot uniquely identify unconstrained intrinsics.  The
repo solver therefore fixes the principal point at image center, constrains
square pixels, grid-searches focal length, and accepts k1/k2 only through its
15% residual-improvement gate.  The emitted calibration stays explicitly
corrected_unverified / preview-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.capture_quality import score_capture_quality  # noqa: E402
from threed.racketsport.coordinates import camera_matrix_from_intrinsics, invert_extrinsics  # noqa: E402
from threed.racketsport.court_calibration import (  # noqa: E402
    homography_from_planar_points,
    project_world_points,
)
from threed.racketsport.court_calibration_metric15 import fit_single_view_metric_camera  # noqa: E402
from threed.racketsport.court_positioning import (  # noqa: E402
    CameraFloorGeometry,
    estimate_ground_sample_distance,
    estimate_position_uncertainty,
)
from threed.racketsport.schemas import (  # noqa: E402
    CameraIntrinsics,
    CaptureQuality,
    CourtCalibration,
    CourtExtrinsics,
    validate_artifact_file,
)


DEFAULT_EVIDENCE = ROOT / (
    "runs/lanes/pbv11_headtohead_20260713/rerun_20260715/calseed_preflight/"
    "pbvision_11min_20260713/court_line_evidence.json"
)
DEFAULT_SEED = ROOT / (
    "runs/lanes/pbv11_headtohead_20260713/rerun_20260715/owner_cal_seed/"
    "court_corners_seed.json"
)
DEFAULT_VIDEO = ROOT / "data/pbvision_11min_20260713/source_video.mp4"
DEFAULT_OUT = ROOT / (
    "runs/lanes/pbv11_headtohead_20260713/rerun_20260715/owner_cal_seed/"
    "court_calibration_solved.json"
)
LANE_DIR = ROOT / "runs/lanes/pbv11_calsolve_20260716"
SOURCE_TAG = "line_evidence_intersections_15pt_single_view_planar"
EXPECTED_VIDEO_SHA256 = "272a2132ce7c72ea31fe6351c9ea05ac3016bbbfed0a5801d9c3a973ec628383"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_line(observations: dict[str, dict[str, Any]], line_ids: list[str]) -> np.ndarray:
    points = np.asarray(
        [point for line_id in line_ids for point in observations[line_id]["image_segment"]],
        dtype=np.float64,
    )
    center = points.mean(axis=0)
    _u, _s, vh = np.linalg.svd(points - center)
    direction = vh[0]
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    return np.asarray([normal[0], normal[1], -normal @ center], dtype=np.float64)


def _intersection(a: np.ndarray, b: np.ndarray) -> list[float]:
    xy = np.linalg.solve(np.asarray([a[:2], b[:2]]), -np.asarray([a[2], b[2]]))
    return [float(xy[0]), float(xy[1])]


def _camera_center(rotation: list[list[float]], translation: list[float]) -> list[float]:
    center = -np.asarray(rotation, dtype=np.float64).T @ np.asarray(translation, dtype=np.float64)
    return [float(value) for value in center]


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _build_correspondences(evidence: dict[str, Any]) -> tuple[list[str], list[list[float]], list[list[float]], dict[str, Any]]:
    accepted = set(evidence["aggregate"]["accepted_line_ids"])
    required = {
        "near_baseline",
        "near_centerline",
        "near_nvz",
        "net",
        "far_nvz",
        "far_centerline",
        "far_baseline",
        "left_sideline",
        "right_sideline",
    }
    missing = sorted(required - accepted)
    if missing:
        raise ValueError(f"required accepted line evidence missing: {missing}")
    observations = {row["line_id"]: row for row in evidence["line_observations"]}

    vertical = {
        "left": _fit_line(observations, ["left_sideline"]),
        "center": _fit_line(observations, ["near_centerline", "far_centerline"]),
        "right": _fit_line(observations, ["right_sideline"]),
    }
    horizontal = {
        "near_baseline": _fit_line(observations, ["near_baseline"]),
        "near_nvz": _fit_line(observations, ["near_nvz"]),
        "ground_net": _fit_line(observations, ["net"]),
        "far_nvz": _fit_line(observations, ["far_nvz"]),
        "far_baseline": _fit_line(observations, ["far_baseline"]),
    }

    # Schema-canonical order.  Unlike the repo metric15 reviewed-keypoint path,
    # the three net-row points here are explicitly ground-net z=0 evidence.
    rows = [
        ("near_left_corner", "near_baseline", "left", -3.048, -6.7056),
        ("near_baseline_center", "near_baseline", "center", 0.0, -6.7056),
        ("near_right_corner", "near_baseline", "right", 3.048, -6.7056),
        ("far_right_corner", "far_baseline", "right", 3.048, 6.7056),
        ("far_baseline_center", "far_baseline", "center", 0.0, 6.7056),
        ("far_left_corner", "far_baseline", "left", -3.048, 6.7056),
        ("near_nvz_left", "near_nvz", "left", -3.048, -2.1336),
        ("near_nvz_center", "near_nvz", "center", 0.0, -2.1336),
        ("near_nvz_right", "near_nvz", "right", 3.048, -2.1336),
        ("ground_net_left_sideline", "ground_net", "left", -3.048, 0.0),
        ("ground_net_center", "ground_net", "center", 0.0, 0.0),
        ("ground_net_right_sideline", "ground_net", "right", 3.048, 0.0),
        ("far_nvz_left", "far_nvz", "left", -3.048, 2.1336),
        ("far_nvz_center", "far_nvz", "center", 0.0, 2.1336),
        ("far_nvz_right", "far_nvz", "right", 3.048, 2.1336),
    ]
    names = [row[0] for row in rows]
    image_points = [_intersection(horizontal[row[1]], vertical[row[2]]) for row in rows]
    world_points = [[row[3], row[4], 0.0] for row in rows]
    line_equations = {
        **{name: line.tolist() for name, line in vertical.items()},
        **{name: line.tolist() for name, line in horizontal.items()},
    }
    return names, image_points, world_points, line_equations


def _build_calibration(
    names: list[str],
    image_points: list[list[float]],
    world_points: list[list[float]],
    evidence: dict[str, Any],
) -> tuple[CourtCalibration, dict[str, Any]]:
    fit = fit_single_view_metric_camera(world_points, image_points, (1280.0, 720.0))
    intrinsics = CameraIntrinsics(
        fx=fit.fx,
        fy=fit.fy,
        cx=fit.cx,
        cy=fit.cy,
        dist=[fit.k1, fit.k2, 0.0, 0.0],
        source=SOURCE_TAG,
    )
    center = _camera_center(fit.R, fit.t)
    extrinsics = CourtExtrinsics(
        R=fit.R,
        t=fit.t,
        camera_height_m=max(abs(center[2]), 1e-6),
    )
    homography = homography_from_planar_points(world_points, image_points)
    camera_to_world_rotation, _ = invert_extrinsics(extrinsics.R, extrinsics.t)
    geometry = CameraFloorGeometry(
        intrinsics={"fx": fit.fx, "fy": fit.fy, "cx": fit.cx, "cy": fit.cy},
        camera_origin_world=center,
        R_world_camera=camera_to_world_rotation.tolist(),
        floor_plane_point=[0.0, 0.0, 0.0],
        floor_plane_normal=[0.0, 0.0, 1.0],
    )
    gsd_samples = []
    for image_xy, world_xyz in zip(image_points, world_points, strict=True):
        gsd = estimate_ground_sample_distance(image_xy, geometry)
        sigma = estimate_position_uncertainty(
            pixel_error_px=fit.reprojection_error_px.median,
            gsd_m_per_px=gsd,
            plane_sigma_m=0.0,
            calibration_sigma_m=0.0,
        )
        gsd_samples.append(
            {
                "court_xy": world_xyz[:2],
                "gsd_m_per_px": gsd,
                "sigma_p_m": sigma,
            }
        )

    base_quality = score_capture_quality(
        corners_visible=len(image_points),
        reprojection_rmse_px=fit.reprojection_error_px.median,
    )
    reasons = list(
        dict.fromkeys(
            [
                *base_quality.reasons,
                "single_view_planar_full_calibration",
                f"distortion_model={fit.distortion_model}",
                "line_evidence_intersection_15pt_correspondences",
                "ground_net_points_not_top_net_points",
                "corrected_unverified",
                "preview_only_not_cal_accuracy_promotion",
            ]
        )
    )
    capture_quality = CaptureQuality(grade=base_quality.grade, reasons=reasons)
    frames = sorted(
        {
            int(frame)
            for row in evidence["line_observations"]
            for frame in row.get("frame_indexes", [])
        }
    )
    if fit.reprojection_error_px.median <= 2.0 and fit.reprojection_error_px.p95 <= 5.0:
        metric_confidence = "high"
    elif fit.reprojection_error_px.median <= 6.0 and fit.reprojection_error_px.p95 <= 15.0:
        metric_confidence = "med"
    else:
        metric_confidence = "low"

    payload = {
        "schema_version": 1,
        "sport": "pickleball",
        "coordinate_frame": "court_netcenter_z_up_m",
        "T_world_court": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "homography": homography,
        "intrinsics": intrinsics.model_dump(mode="json"),
        "image_size": [1280, 720],
        "extrinsics": extrinsics.model_dump(mode="json"),
        "reprojection_error_px": fit.reprojection_error_px.model_dump(mode="json"),
        "per_keypoint_residual_px": fit.per_point_residual_px,
        "metric_confidence": metric_confidence,
        "gsd_model": {
            "type": "analytic_ray_plane",
            "plane_sigma_m": 0.0,
            "calibration_sigma_m": 0.0,
            "samples": gsd_samples,
        },
        "capture_quality": capture_quality.model_dump(mode="json"),
        "image_pts": image_points,
        "world_pts": world_points,
        "source": SOURCE_TAG,
        "solved_over_frames": frames,
        "coordinate_contract": {
            "camera_matrix_K": camera_matrix_from_intrinsics(intrinsics),
            "camera_matrix_input_space": "camera_m",
            "camera_matrix_output_space": "pixels_undistorted_native",
            "extrinsics_convention": "world_to_camera_opencv_column",
            "extrinsics_input_space": "world_court_netcenter_z_up_m",
            "extrinsics_output_space": "camera_m",
            "homography_convention": "world_xy_to_image_column",
            "homography_input_space": "world_xy_homography_m",
            "homography_output_space": "pixels_raw_native",
            "homography_pixel_convention": "raw_pixels",
        },
    }
    calibration = CourtCalibration.model_validate(payload)
    diagnostics = {
        "correspondence_names": names,
        "fit": {
            "intrinsics": intrinsics.model_dump(mode="json"),
            "distortion_model": fit.distortion_model,
            "extrinsics": extrinsics.model_dump(mode="json"),
            "camera_center_world_m": center,
            "reprojection_error_px": fit.reprojection_error_px.model_dump(mode="json"),
            "per_correspondence_residual_px": [
                {"name": name, "residual_px": residual}
                for name, residual in zip(names, fit.per_point_residual_px, strict=True)
            ],
            "identifiability_notes": fit.identifiability_notes,
            "metric_confidence": metric_confidence,
        },
    }
    return calibration, diagnostics


def _render_overlay(video: Path, calibration: CourtCalibration, out: Path, *, frame_index: int = 300) -> None:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not decode frame {frame_index} from {video}")

    color_floor = (0, 255, 255)
    color_top_net = (255, 0, 255)
    xs = (-3.048, 0.0, 3.048)
    ys = (-6.7056, -2.1336, 0.0, 2.1336, 6.7056)
    floor_segments = []
    for x in xs:
        floor_segments.append(([x, -6.7056, 0.0], [x, -2.1336, 0.0]))
        floor_segments.append(([x, 2.1336, 0.0], [x, 6.7056, 0.0]))
    for y in ys:
        floor_segments.append(([-3.048, y, 0.0], [3.048, y, 0.0]))
    for start, end in floor_segments:
        p0, p1 = project_world_points(calibration.extrinsics, calibration.intrinsics, [start, end])
        cv2.line(frame, tuple(np.rint(p0).astype(int)), tuple(np.rint(p1).astype(int)), color_floor, 2, cv2.LINE_AA)

    # Regulation post-height line is a sanity projection only; it was not an input.
    top_z_m = 0.9144
    p0, p1 = project_world_points(
        calibration.extrinsics,
        calibration.intrinsics,
        [[-3.048, 0.0, top_z_m], [3.048, 0.0, top_z_m]],
    )
    cv2.line(frame, tuple(np.rint(p0).astype(int)), tuple(np.rint(p1).astype(int)), color_top_net, 2, cv2.LINE_AA)
    cv2.putText(frame, "yellow=floor court; magenta=projected top net (not solve input)", (22, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(frame, "yellow=floor court; magenta=projected top net (not solve input)", (22, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), frame):
        raise RuntimeError(f"could not write {out}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lane-dir", type=Path, default=LANE_DIR)
    args = parser.parse_args()

    video_sha = _sha256(args.video)
    if video_sha != EXPECTED_VIDEO_SHA256:
        raise ValueError(f"video sha mismatch: {video_sha}")
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    seed = json.loads(args.seed.read_text(encoding="utf-8"))
    names, image_points, world_points, line_equations = _build_correspondences(evidence)
    calibration, diagnostics = _build_calibration(names, image_points, world_points, evidence)

    seed_corners = seed["annotation"]["items"][0]["court_corners"]
    corner_indexes = {"near_left": 0, "near_right": 2, "far_right": 3, "far_left": 5}
    corner_deltas = {
        name: math.dist(image_points[index], seed_corners[name])
        for name, index in corner_indexes.items()
    }
    residuals = [float(row) for row in calibration.per_keypoint_residual_px or []]
    report = {
        "schema_version": 1,
        "artifact_type": "pbv11_line_evidence_calibration_solve",
        "status": "corrected_unverified",
        "verified": False,
        "preview_only": True,
        "competitor_camera_used_as_input": False,
        "inputs": {
            "video": str(args.video.relative_to(ROOT)),
            "video_sha256": video_sha,
            "owner_seed": str(args.seed.relative_to(ROOT)),
            "owner_seed_sha256": _sha256(args.seed),
            "line_evidence": str(args.evidence.relative_to(ROOT)),
            "line_evidence_sha256": _sha256(args.evidence),
            "line_evidence_aggregate": evidence["aggregate"],
        },
        "method": {
            "source_tag": SOURCE_TAG,
            "correspondence_count": len(image_points),
            "correspondence_derivation": "intersections of accepted temporally aggregated court-line observations",
            "centerline_method": "TLS fit over endpoints of accepted near_centerline and far_centerline observations",
            "net_row_semantics": "ground_net z=0 only; no top-net point was used as a solve input",
            "planar_scene_limit": "single-view planar intrinsics are not uniquely identifiable without assumptions",
        },
        "line_equations_ax_by_c_zero": line_equations,
        "correspondences": [
            {"name": name, "image_xy_px": image_xy, "world_xyz_m": world_xyz}
            for name, image_xy, world_xyz in zip(names, image_points, world_points, strict=True)
        ],
        "corner_intersection_delta_vs_owner_seed_px": corner_deltas,
        **diagnostics,
        "residual_summary_px": {
            "median": float(np.median(residuals)),
            "p95": _percentile(residuals, 95.0),
            "max": max(residuals),
            "line_evidence_mean": evidence["aggregate"]["mean_residual_px"],
            "line_evidence_p95": evidence["aggregate"]["p95_residual_px"],
        },
        "honesty": [
            "This is a corrected_unverified preview calibration, not a CAL accuracy promotion.",
            "All solve inputs came from our owner seed and our accepted line evidence.",
            "No competitor camera field was read or used by this solver.",
            "The planar solve assumptions and conditioning remain binding caveats even if the pipeline gate opens.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(calibration.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_artifact_file("court_calibration", args.out)
    args.lane_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = args.lane_dir / "solve_diagnostics.json"
    diagnostics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _render_overlay(args.video, calibration, args.lane_dir / "solved_calibration_overlay_frame300.jpg")
    print(json.dumps({"calibration": str(args.out), "diagnostics": str(diagnostics_path), "overlay": str(args.lane_dir / "solved_calibration_overlay_frame300.jpg"), "reprojection_error_px": calibration.reprojection_error_px.model_dump(mode="json"), "metric_confidence": calibration.metric_confidence}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
