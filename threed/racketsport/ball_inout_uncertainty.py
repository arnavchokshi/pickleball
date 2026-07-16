"""Camera-geometry-derived in/out uncertainty for BALL bounces.

Implements the sigma_bounce model from BALL_TRACKING_PIPELINE.md section 5.6
in place of a fixed uncertainty radius:

    sigma_bounce = sqrt(sigma_reproj^2 + sigma_depth^2 + sigma_ballradius^2 + sigma_localization^2)

``sigma_depth`` is the term this module adds: the ground-plane displacement
caused by the ball still being physically elevated at the frame the 2D
bounce detector fires (it can fire up to BOUNCE_DETECTION_FRAME_WINDOW
frames off the true ground-contact frame), projected onto the court-boundary
normal at the landing point. On a steep/low camera, a small residual height
maps to a large horizontal ground-plane error because the viewing ray grazes
the court at a shallow angle; on a near-overhead camera the same residual
height barely moves the ground intersection. This is exactly the mechanism
the 2026-07-02 owner review exposed: 6 of 8 reviewed "in" bounces were
called confidently "out" by the old fixed +/-0.05 m radius, all on
low/steep-angle cameras, with margins of 0.94-2.26 m.

Every constant below is a physical/engineering constant or an existing
repository convention -- never fit to the reviewed human labels:

  - STANDARD_GRAVITY_MPS2: universal physical constant (already used as
    literal -9.81 in ball_physics3d.py).
  - PICKLEBALL_DROP_TEST_HEIGHT_M: USA Pickleball equipment standard -- a
    ball dropped from 78 in (1.9812 m) onto a granite surface at 75F must
    rebound 30-34 in. This is an external, sport-specific regulatory fact
    about pickleball bounce dynamics, independent of this dataset or any
    review label. We use free-fall (zero-drag) kinematics from this
    reference height as a physically-grounded, deliberately conservative
    (drag would only lower the true speed) bound on vertical speed near a
    bounce.
  - BOUNCE_DETECTION_FRAME_WINDOW: this codebase's own bounce-timing
    tolerance (BALL_TRACKING_PIPELINE.md section 9, "Bounce: timing within
    +/-2 frames"; the same 2-frame tolerance ball_inout_gate.py already
    uses to match predicted bounces to reviewed ones).
  - BALL_RADIUS_UNCERTAINTY_M: BALL_TRACKING_PIPELINE.md section 6,
    "Ball radius uncertainty ~2 cm".
  - PIXEL_LOCALIZATION_PX: BALL_TRACKING_PIPELINE.md section 6,
    "Jitter target (straight seg) < 2 px std".
  - Calibration reprojection gate thresholds: reused from
    court_calibration.py's existing CALIBRATION_REPROJECTION_*_GATE_PX.

The camera pose used to compute sigma_depth is solved from the *same* 4
manual court corners already used to build the ground-plane homography (no
new human input): principal point = corner centroid, focal length found by
a 1D search that minimizes reprojection error of those same 4 corners. This
is standard single-view calibration off the calibration corners already in
use -- not a value fit to any bounce/in-out label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .coordinates import CoordinateSpace, validate_opencv_camera_seam
from .court_calibration import (
    CALIBRATION_REPROJECTION_MEDIAN_GATE_PX,
    CALIBRATION_REPROJECTION_P95_GATE_PX,
)
from .court_positioning import CameraFloorGeometry, back_project_pixel_to_floor, estimate_ground_sample_distance
from .court_templates import Sport, get_court_template

# ---------------------------------------------------------------------------
# Physics / engineering constants (not fit to any reviewed label)
# ---------------------------------------------------------------------------

STANDARD_GRAVITY_MPS2 = 9.81
PICKLEBALL_DROP_TEST_HEIGHT_M = 1.9812  # USA Pickleball equipment standard: 78 in drop test
BOUNCE_DETECTION_FRAME_WINDOW = 2.0  # matches ball_inout_gate.MAX_REVIEW_DELTA_FRAMES
BALL_RADIUS_UNCERTAINTY_M = 0.02
PIXEL_LOCALIZATION_PX = 2.0

# Search bracket for the focal-length self-calibration below. Wide enough to
# cover any plausible smartphone/webcam FOV; not tuned per clip or label.
FOCAL_LENGTH_SEARCH_MIN_PX = 150.0
FOCAL_LENGTH_SEARCH_MAX_PX = 10000.0
FOCAL_LENGTH_SEARCH_COARSE_STEPS = 40
FOCAL_LENGTH_SEARCH_REFINE_ITERATIONS = 60

METHOD_CAMERA_GEOMETRY = "camera_geometry_elevation_parallax_v1"
METHOD_FIXED_OVERRIDE = "fixed_override"

TERM_ELEVATION_PARALLAX = "camera_geometry_elevation_parallax"
TERM_CALIBRATION_REPROJECTION = "calibration_reprojection"
TERM_BALL_RADIUS = "ball_radius"
TERM_PIXEL_LOCALIZATION = "pixel_localization"


def reference_vertical_impact_speed_mps() -> float:
    """Free-fall speed after PICKLEBALL_DROP_TEST_HEIGHT_M, used as v_z_ref."""

    return math.sqrt(2.0 * STANDARD_GRAVITY_MPS2 * PICKLEBALL_DROP_TEST_HEIGHT_M)


@dataclass(frozen=True)
class CameraPose:
    """A solved (K, R, t) camera pose plus its fit quality."""

    fx: float
    fy: float
    cx: float
    cy: float
    R: list[list[float]]
    t: list[float]
    camera_center_world: list[float]
    camera_height_m: float
    reprojection_error_px_median: float
    reprojection_error_px_p95: float
    source: str

    def intrinsics_mapping(self) -> dict[str, float]:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}

    def passes_reprojection_gate(self) -> bool:
        return (
            self.reprojection_error_px_median < CALIBRATION_REPROJECTION_MEDIAN_GATE_PX
            and self.reprojection_error_px_p95 < CALIBRATION_REPROJECTION_P95_GATE_PX
        )


def solve_manual_corner_camera_pose(
    image_pts: Sequence[Sequence[float]],
    world_pts: Sequence[Sequence[float]],
    *,
    object_space: CoordinateSpace = CoordinateSpace.WORLD_COURT_NETCENTER_Z_UP_M,
    image_reference_space: CoordinateSpace = CoordinateSpace.PIXELS_RAW_NATIVE,
    projected_space: CoordinateSpace = CoordinateSpace.PIXELS_UNDISTORTED_NATIVE,
) -> CameraPose:
    """Solve a full 6-DOF camera pose from the same 4 corners used for the homography.

    Principal point is fixed at the corner centroid; focal length is found by
    a 1D search minimizing reprojection error of the 4 corners themselves.
    No frame-resolution metadata or reviewed label is used.
    """

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("solve_manual_corner_camera_pose requires opencv-python and numpy") from exc

    if len(image_pts) != len(world_pts) or len(image_pts) < 4:
        raise ValueError("solve_manual_corner_camera_pose requires at least 4 paired points")
    validate_opencv_camera_seam(
        object_space=object_space,
        image_reference_space=image_reference_space,
        projected_space=projected_space,
    )

    image_arr = np.asarray([[float(p[0]), float(p[1])] for p in image_pts], dtype=np.float64)
    world_arr = np.asarray([[float(p[0]), float(p[1]), float(p[2])] for p in world_pts], dtype=np.float64)
    cx = float(image_arr[:, 0].mean())
    cy = float(image_arr[:, 1].mean())

    def solve_for_focal(focal_px: float) -> tuple[float, Any, Any] | None:
        k = np.array([[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(world_arr, image_arr, k, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        projected, _ = cv2.projectPoints(world_arr, rvec, tvec, k, None)
        residuals = np.linalg.norm(projected.reshape(-1, 2) - image_arr, axis=1)
        return float(np.median(residuals)), rvec, tvec

    log_lo = math.log(FOCAL_LENGTH_SEARCH_MIN_PX)
    log_hi = math.log(FOCAL_LENGTH_SEARCH_MAX_PX)
    coarse = [
        log_lo + (log_hi - log_lo) * step / (FOCAL_LENGTH_SEARCH_COARSE_STEPS - 1)
        for step in range(FOCAL_LENGTH_SEARCH_COARSE_STEPS)
    ]
    best_log_f = None
    best_median = None
    for log_f in coarse:
        result = solve_for_focal(math.exp(log_f))
        if result is None:
            continue
        median_err = result[0]
        if best_median is None or median_err < best_median:
            best_median = median_err
            best_log_f = log_f

    if best_log_f is None:
        raise ValueError("manual corner camera pose: focal-length search failed to converge")

    step_width = (log_hi - log_lo) / (FOCAL_LENGTH_SEARCH_COARSE_STEPS - 1)
    bracket_lo = best_log_f - step_width
    bracket_hi = best_log_f + step_width

    golden_ratio = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = bracket_lo, bracket_hi
    c = b - golden_ratio * (b - a)
    d = a + golden_ratio * (b - a)
    f_c = solve_for_focal(math.exp(c))
    f_d = solve_for_focal(math.exp(d))
    for _ in range(FOCAL_LENGTH_SEARCH_REFINE_ITERATIONS):
        if f_c is None or (f_d is not None and f_c[0] > f_d[0]):
            a = c
            c = d
            f_c = f_d
            d = a + golden_ratio * (b - a)
            f_d = solve_for_focal(math.exp(d))
        else:
            b = d
            d = c
            f_d = f_c
            c = b - golden_ratio * (b - a)
            f_c = solve_for_focal(math.exp(c))

    best_focal = math.exp((a + b) / 2.0)
    final = solve_for_focal(best_focal)
    if final is None:
        final_focal = math.exp(best_log_f)
        final = solve_for_focal(final_focal)
        best_focal = final_focal
    if final is None:
        raise ValueError("manual corner camera pose: refinement failed to converge")

    _, rvec, tvec = final
    k = np.array([[best_focal, 0.0, cx], [0.0, best_focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    rotation, _ = cv2.Rodrigues(rvec)
    translation = tvec.reshape(3)
    projected, _ = cv2.projectPoints(world_arr, rvec, tvec, k, None)
    residuals = np.linalg.norm(projected.reshape(-1, 2) - image_arr, axis=1)
    camera_center = -(rotation.T @ translation)

    return CameraPose(
        fx=float(best_focal),
        fy=float(best_focal),
        cx=cx,
        cy=cy,
        R=rotation.tolist(),
        t=[float(value) for value in translation.tolist()],
        camera_center_world=[float(value) for value in camera_center.tolist()],
        camera_height_m=abs(float(camera_center[2])),
        reprojection_error_px_median=_percentile(residuals.tolist(), 50.0),
        reprojection_error_px_p95=_percentile(residuals.tolist(), 95.0),
        source="manual_corner_focal_length_search_v1",
    )


def _camera_floor_geometry(pose: CameraPose, height_m: float) -> CameraFloorGeometry:
    rotation_world_camera = [[pose.R[row][col] for row in range(3)] for col in range(3)]
    return CameraFloorGeometry(
        intrinsics=pose.intrinsics_mapping(),
        camera_origin_world=pose.camera_center_world,
        R_world_camera=rotation_world_camera,
        floor_plane_point=[0.0, 0.0, float(height_m)],
        floor_plane_normal=[0.0, 0.0, 1.0],
    )


def ground_point_at_height(pixel_uv: Sequence[float], pose: CameraPose, height_m: float) -> list[float]:
    """Where a camera ray through pixel_uv intersects the plane Z=height_m."""

    return back_project_pixel_to_floor(pixel_uv, _camera_floor_geometry(pose, height_m))


def binding_boundary_axis(world_xy: Sequence[float], *, sport: Sport) -> str:
    """Which axis (x=sideline, y=baseline) determines the in/out margin here."""

    template = get_court_template(sport)
    half_width = template.width_m / 2.0
    half_length = template.length_m / 2.0
    x, y = float(world_xy[0]), float(world_xy[1])
    margins = {
        "x": min(x + half_width, half_width - x),
        "y": min(y + half_length, half_length - y),
    }
    return "x" if margins["x"] <= margins["y"] else "y"


def bounce_geometric_uncertainty_m(
    *,
    contact_xy_img: Sequence[float],
    world_xy: Sequence[float],
    pose: CameraPose,
    sport: Sport = "pickleball",
    fps: float,
    reprojection_error_px_p95: float = 0.0,
) -> dict[str, Any]:
    """Per-bounce sigma_bounce breakdown from BALL_TRACKING_PIPELINE.md section 5.6."""

    if fps <= 0.0:
        raise ValueError("fps must be positive")

    ground_sample_distance_m_per_px = estimate_ground_sample_distance(contact_xy_img, _camera_floor_geometry(pose, 0.0))

    dt_s = BOUNCE_DETECTION_FRAME_WINDOW / float(fps)
    v_z_ref_mps = reference_vertical_impact_speed_mps()
    h_max_m = v_z_ref_mps * dt_s

    point_h0 = ground_point_at_height(contact_xy_img, pose, 0.0)
    point_hmax = ground_point_at_height(contact_xy_img, pose, h_max_m)
    displacement = [point_hmax[0] - point_h0[0], point_hmax[1] - point_h0[1]]

    axis = binding_boundary_axis(world_xy, sport=sport)
    sigma_depth_m = abs(displacement[0] if axis == "x" else displacement[1])

    sigma_reproj_m = float(reprojection_error_px_p95) * ground_sample_distance_m_per_px
    sigma_localization_m = PIXEL_LOCALIZATION_PX * ground_sample_distance_m_per_px
    sigma_ballradius_m = BALL_RADIUS_UNCERTAINTY_M

    terms = {
        TERM_ELEVATION_PARALLAX: sigma_depth_m,
        TERM_CALIBRATION_REPROJECTION: sigma_reproj_m,
        TERM_BALL_RADIUS: sigma_ballradius_m,
        TERM_PIXEL_LOCALIZATION: sigma_localization_m,
    }
    dominant_term = max(terms, key=lambda name: terms[name])

    uncertainty_m = math.sqrt(sigma_reproj_m**2 + sigma_depth_m**2 + sigma_ballradius_m**2 + sigma_localization_m**2)

    to_camera = [
        point_h0[0] - pose.camera_center_world[0],
        point_h0[1] - pose.camera_center_world[1],
        point_h0[2] - pose.camera_center_world[2],
    ]
    horizontal = math.hypot(to_camera[0], to_camera[1])
    grazing_angle_deg = math.degrees(math.atan2(abs(to_camera[2]), horizontal)) if horizontal > 0.0 else 90.0

    return {
        "uncertainty_m": uncertainty_m,
        "dominant_uncertainty_term": dominant_term,
        "breakdown": {
            "method": METHOD_CAMERA_GEOMETRY,
            "sigma_reproj_m": sigma_reproj_m,
            "sigma_depth_m": sigma_depth_m,
            "sigma_ballradius_m": sigma_ballradius_m,
            "sigma_localization_m": sigma_localization_m,
            "camera_height_m": pose.camera_height_m,
            "grazing_angle_deg": grazing_angle_deg,
            "h_max_m": h_max_m,
            "v_z_ref_mps": v_z_ref_mps,
            "dt_s": dt_s,
            "frames_window": BOUNCE_DETECTION_FRAME_WINDOW,
            "binding_axis": axis,
            "ground_sample_distance_m_per_px": ground_sample_distance_m_per_px,
            "pose_reprojection_error_px_median": pose.reprojection_error_px_median,
            "pose_source": pose.source,
        },
    }


def fixed_override_breakdown(uncertainty_m: float) -> dict[str, Any]:
    """Breakdown recorded when the caller supplies an explicit --uncertainty-m override."""

    return {
        "method": METHOD_FIXED_OVERRIDE,
        "sigma_reproj_m": 0.0,
        "sigma_depth_m": 0.0,
        "sigma_ballradius_m": 0.0,
        "sigma_localization_m": float(uncertainty_m),
    }


def physics_constants_manifest() -> dict[str, Any]:
    """Every constant the geometric uncertainty model uses, with its justification.

    Intended for embedding in run artifacts so the derivation is auditable
    without reading source: zero of these are derived from the reviewed
    human in/out labels.
    """

    return {
        "standard_gravity_mps2": {
            "value": STANDARD_GRAVITY_MPS2,
            "justification": "Universal physical constant; also used as -9.81 in ball_physics3d.py.",
        },
        "pickleball_drop_test_height_m": {
            "value": PICKLEBALL_DROP_TEST_HEIGHT_M,
            "justification": (
                "USA Pickleball equipment standard: a ball dropped from 78 in onto granite at 75F "
                "must rebound 30-34 in. External, sport-specific, non-label regulatory constant."
            ),
        },
        "reference_vertical_impact_speed_mps": {
            "value": reference_vertical_impact_speed_mps(),
            "justification": "sqrt(2 * g * drop_test_height); free-fall (zero-drag, conservative) speed.",
        },
        "bounce_detection_frame_window": {
            "value": BOUNCE_DETECTION_FRAME_WINDOW,
            "justification": (
                "This repo's own bounce-timing tolerance (BALL_TRACKING_PIPELINE.md section 9, "
                "'Bounce: timing within +/-2 frames'; ball_inout_gate.MAX_REVIEW_DELTA_FRAMES)."
            ),
        },
        "ball_radius_uncertainty_m": {
            "value": BALL_RADIUS_UNCERTAINTY_M,
            "justification": "BALL_TRACKING_PIPELINE.md section 6, 'Ball radius uncertainty ~2 cm'.",
        },
        "pixel_localization_px": {
            "value": PIXEL_LOCALIZATION_PX,
            "justification": "BALL_TRACKING_PIPELINE.md section 6, 'Jitter target (straight seg) < 2 px std'.",
        },
        "calibration_reprojection_gate_px": {
            "value": {"median": CALIBRATION_REPROJECTION_MEDIAN_GATE_PX, "p95": CALIBRATION_REPROJECTION_P95_GATE_PX},
            "justification": "Reused from court_calibration.py's existing calibration reprojection gate.",
        },
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


__all__ = [
    "BALL_RADIUS_UNCERTAINTY_M",
    "BOUNCE_DETECTION_FRAME_WINDOW",
    "CameraPose",
    "METHOD_CAMERA_GEOMETRY",
    "METHOD_FIXED_OVERRIDE",
    "PICKLEBALL_DROP_TEST_HEIGHT_M",
    "PIXEL_LOCALIZATION_PX",
    "STANDARD_GRAVITY_MPS2",
    "binding_boundary_axis",
    "bounce_geometric_uncertainty_m",
    "fixed_override_breakdown",
    "ground_point_at_height",
    "physics_constants_manifest",
    "reference_vertical_impact_speed_mps",
    "solve_manual_corner_camera_pose",
]
