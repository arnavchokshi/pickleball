"""Per-clip court calibration helpers and solvePnP-ready artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from .capture_quality import score_capture_quality
from .court_templates import Sport, get_court_template
from .schemas import (
    CameraIntrinsics,
    CaptureQuality,
    CaptureSidecar,
    CourtCalibration,
    CourtExtrinsics,
    CourtKeypoints,
    PICKLEBALL_COURT_KEYPOINT_NAMES,
    ReprojectionError,
)

CALIBRATION_REPROJECTION_MEDIAN_GATE_PX = 8.0
CALIBRATION_REPROJECTION_P95_GATE_PX = 15.0


def camera_matrix_from_intrinsics(intrinsics: CameraIntrinsics) -> list[list[float]]:
    return [
        [float(intrinsics.fx), 0.0, float(intrinsics.cx)],
        [0.0, float(intrinsics.fy), float(intrinsics.cy)],
        [0.0, 0.0, 1.0],
    ]


def load_capture_sidecar(path: str | Path) -> CaptureSidecar:
    with Path(path).open("r", encoding="utf-8") as handle:
        return CaptureSidecar.model_validate(json.load(handle))


def load_court_keypoints(path: str | Path) -> CourtKeypoints:
    with Path(path).open("r", encoding="utf-8") as handle:
        return CourtKeypoints.model_validate(json.load(handle))


def manual_tap_correspondences(
    sidecar: CaptureSidecar,
    *,
    sport: Sport,
) -> tuple[list[list[float]], list[list[float]]]:
    if len(sidecar.manual_court_taps) < 4:
        raise ValueError("at least 4 manual court taps are required")

    template = get_court_template(sport)
    return [list(point) for point in sidecar.manual_court_taps[:4]], [list(point) for point in template.corners_m]


def homography_from_planar_points(
    world_pts: Iterable[Iterable[float]],
    image_pts: Iterable[Iterable[float]],
) -> list[list[float]]:
    world = [[float(point[0]), float(point[1])] for point in world_pts]
    image = [[float(point[0]), float(point[1])] for point in image_pts]
    if len(world) != len(image) or len(world) < 4:
        raise ValueError("homography requires at least 4 paired world/image points")

    rows: list[tuple[list[float], float]] = []
    for (x, y), (u, v) in zip(world, image, strict=True):
        rows.append(([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y], u))
        rows.append(([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y], v))

    normal_matrix = [[0.0 for _ in range(8)] for _ in range(8)]
    normal_rhs = [0.0 for _ in range(8)]
    for coeffs, rhs in rows:
        for row_idx in range(8):
            normal_rhs[row_idx] += coeffs[row_idx] * rhs
            for col_idx in range(8):
                normal_matrix[row_idx][col_idx] += coeffs[row_idx] * coeffs[col_idx]

    h00, h01, h02, h10, h11, h12, h20, h21 = _solve_linear_system(normal_matrix, normal_rhs)
    return [[h00, h01, h02], [h10, h11, h12], [h20, h21, 1.0]]


def project_planar_points(
    homography: Iterable[Iterable[float]],
    world_pts: Iterable[Iterable[float]],
) -> list[list[float]]:
    h = [[float(value) for value in row] for row in homography]
    projected: list[list[float]] = []
    for point in world_pts:
        x, y = float(point[0]), float(point[1])
        u_num = h[0][0] * x + h[0][1] * y + h[0][2]
        v_num = h[1][0] * x + h[1][1] * y + h[1][2]
        scale = h[2][0] * x + h[2][1] * y + h[2][2]
        if math.isclose(scale, 0.0):
            raise ValueError("homogeneous projection has zero scale")
        projected.append([u_num / scale, v_num / scale])
    return projected


def project_image_points_to_world(
    homography: Iterable[Iterable[float]],
    image_pts: Iterable[Iterable[float]],
) -> list[list[float]]:
    inverse = _invert_homography(homography)
    projected: list[list[float]] = []
    for point in image_pts:
        u, v = float(point[0]), float(point[1])
        x_num = inverse[0][0] * u + inverse[0][1] * v + inverse[0][2]
        y_num = inverse[1][0] * u + inverse[1][1] * v + inverse[1][2]
        scale = inverse[2][0] * u + inverse[2][1] * v + inverse[2][2]
        if math.isclose(scale, 0.0):
            raise ValueError("homogeneous inverse projection has zero scale")
        projected.append([x_num / scale, y_num / scale])
    return projected


def project_world_points(
    extrinsics: CourtExtrinsics,
    intrinsics: CameraIntrinsics,
    world_pts: Iterable[Iterable[float]],
) -> list[list[float]]:
    projected: list[list[float]] = []
    rotation = [[float(value) for value in row] for row in extrinsics.R]
    translation = [float(value) for value in extrinsics.t]
    for point in world_pts:
        world = [float(point[0]), float(point[1]), float(point[2])]
        camera = [
            sum(rotation[row_idx][col_idx] * world[col_idx] for col_idx in range(3)) + translation[row_idx]
            for row_idx in range(3)
        ]
        if math.isclose(camera[2], 0.0):
            raise ValueError("world point projects with zero camera depth")
        projected.append(
            [
                intrinsics.fx * camera[0] / camera[2] + intrinsics.cx,
                intrinsics.fy * camera[1] / camera[2] + intrinsics.cy,
            ]
        )
    return projected


def reprojection_error(
    image_pts: Iterable[Iterable[float]],
    projected_pts: Iterable[Iterable[float]],
) -> ReprojectionError:
    observed = [[float(point[0]), float(point[1])] for point in image_pts]
    projected = [[float(point[0]), float(point[1])] for point in projected_pts]
    if len(observed) != len(projected) or not observed:
        raise ValueError("reprojection error requires paired non-empty point arrays")

    distances = [
        math.hypot(observed_point[0] - projected_point[0], observed_point[1] - projected_point[1])
        for observed_point, projected_point in zip(observed, projected, strict=True)
    ]
    distances = [0.0 if distance < 1e-9 else distance for distance in distances]
    return ReprojectionError(median=_percentile(distances, 50), p95=_percentile(distances, 95))


def passes_reprojection_gate(error: ReprojectionError) -> bool:
    return (
        error.median < CALIBRATION_REPROJECTION_MEDIAN_GATE_PX
        and error.p95 < CALIBRATION_REPROJECTION_P95_GATE_PX
    )


def calibration_image_size(
    calibration: CourtCalibration,
    *,
    fallback_target: tuple[float, float] | None = None,
    principal_point_tolerance: float = 0.10,
) -> tuple[float, float]:
    """Return calibration image size without treating off-center cx/cy as frame center."""

    if calibration.image_size is not None:
        width, height = calibration.image_size
        if width > 0 and height > 0:
            return float(width), float(height)

    inferred_width = float(calibration.intrinsics.cx) * 2.0
    inferred_height = float(calibration.intrinsics.cy) * 2.0
    if inferred_width <= 0.0 or inferred_height <= 0.0:
        raise ValueError("cannot infer calibration image size from intrinsics")

    if fallback_target is not None:
        target_width, target_height = fallback_target
        if target_width > 0.0 and target_height > 0.0:
            close_to_target = (
                abs(target_width / inferred_width - 1.0) <= principal_point_tolerance
                and abs(target_height / inferred_height - 1.0) <= principal_point_tolerance
            )
            if close_to_target:
                return float(target_width), float(target_height)

    return inferred_width, inferred_height


def solve_camera_pose(
    world_pts: Iterable[Iterable[float]],
    image_pts: Iterable[Iterable[float]],
    intrinsics: CameraIntrinsics,
) -> CourtExtrinsics:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("solve_camera_pose requires opencv-python and numpy") from exc

    world = np.asarray([[float(point[0]), float(point[1]), float(point[2])] for point in world_pts], dtype=np.float64)
    image = np.asarray([[float(point[0]), float(point[1])] for point in image_pts], dtype=np.float64)
    if world.shape[0] != image.shape[0] or world.shape[0] < 4:
        raise ValueError("solve_camera_pose requires at least 4 paired world/image points")

    distortion = np.asarray(intrinsics.dist, dtype=np.float64) if intrinsics.dist else None
    ok, rvec, tvec = cv2.solvePnP(
        world,
        image,
        np.asarray(camera_matrix_from_intrinsics(intrinsics), dtype=np.float64),
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("cv2.solvePnP failed to solve camera pose")

    rotation, _ = cv2.Rodrigues(rvec)
    translation = tvec.reshape(3)
    camera_center_world = -(rotation.T @ translation)
    return CourtExtrinsics(
        R=rotation.tolist(),
        t=[float(value) for value in translation.tolist()],
        camera_height_m=abs(float(camera_center_world[2])),
    )


def _camera_height_from_sidecar(sidecar: CaptureSidecar) -> float:
    if sidecar.arkit_camera_pose is None:
        return 0.0
    if sidecar.court_plane is None:
        return abs(float(sidecar.arkit_camera_pose.t[2]))

    camera = [float(value) for value in sidecar.arkit_camera_pose.t]
    plane_point = [float(value) for value in sidecar.court_plane.point]
    normal = [float(value) for value in sidecar.court_plane.normal]
    norm = math.sqrt(sum(value * value for value in normal))
    if math.isclose(norm, 0.0):
        return 0.0
    return abs(sum((camera[idx] - plane_point[idx]) * normal[idx] / norm for idx in range(3)))


def _merge_capture_quality(sidecar: CaptureSidecar, reprojection: ReprojectionError, corners_visible: int) -> CaptureQuality:
    scored = score_capture_quality(
        corners_visible=corners_visible,
        reprojection_rmse_px=reprojection.median,
        fps=sidecar.fps,
        exposure_s=sidecar.locked.exposure_s,
    )
    reasons = list(dict.fromkeys([*sidecar.capture_quality.reasons, *scored.reasons]))
    grades = {"good": 0, "warn": 1, "poor": 2}
    grade = sidecar.capture_quality.grade if grades[sidecar.capture_quality.grade] >= grades[scored.grade] else scored.grade
    return CaptureQuality(grade=grade, reasons=reasons)


def calibration_from_manual_taps(path: str | Path, *, sport: Sport) -> CourtCalibration:
    sidecar = load_capture_sidecar(path)
    image_pts, world_pts = manual_tap_correspondences(sidecar, sport=sport)
    return _build_calibration(sidecar, sport=sport, image_pts=image_pts, world_pts=world_pts)


def build_manual_tap_calibration_artifact(
    path: str | Path,
    *,
    sport: Sport,
    candidate_segments: Iterable[Sequence[Sequence[float]]] = (),
) -> dict[str, Any]:
    """Return a trusted manual-tap calibration payload with additive plausibility metadata.

    The canonical ``CourtCalibration`` schema is intentionally unchanged here. Callers that
    need a strict calibration model should keep using ``calibration_from_manual_taps``.
    """

    sidecar = load_capture_sidecar(path)
    image_pts, world_pts = manual_tap_correspondences(sidecar, sport=sport)
    calibration = _build_calibration(sidecar, sport=sport, image_pts=image_pts, world_pts=world_pts)
    plausibility = evaluate_manual_tap_plausibility(
        sidecar,
        sport=sport,
        candidate_segments=candidate_segments,
    )
    payload = calibration.model_dump(mode="json")
    payload["tap_plausibility"] = plausibility
    payload["needs_user_confirmation"] = bool(plausibility["needs_user_confirmation"])
    return payload


def evaluate_manual_tap_plausibility(
    sidecar: CaptureSidecar,
    *,
    sport: Sport,
    candidate_segments: Iterable[Sequence[Sequence[float]]] = (),
) -> dict[str, Any]:
    """Check whether trusted manual taps look like tennis-corner taps over pickleball evidence.

    Human taps remain authoritative for calibration. This function only annotates whether
    line evidence makes the pickleball interpretation suspicious enough to ask the owner to
    confirm the taps.
    """

    if sport != "pickleball":
        return {
            "verdict": "consistent",
            "needs_user_confirmation": False,
            "owner_tap_trusted": True,
            "reason": "tap_plausibility_check_only_applies_to_pickleball_manual_taps",
            "trust_note": {"grade": "good", "reason": "non_pickleball_tap_check_bypassed"},
            "evidence": {},
        }

    image_pts, _world_pts = manual_tap_correspondences(sidecar, sport=sport)
    candidates = [_segment2(candidate) for candidate in candidate_segments]
    if not candidates:
        return {
            "verdict": "consistent",
            "needs_user_confirmation": False,
            "owner_tap_trusted": True,
            "reason": "no_line_evidence_available",
            "trust_note": {"grade": "good", "reason": "trusted_manual_taps_without_line_evidence"},
            "evidence": {},
        }

    pickleball = _score_tap_template_against_candidates(
        image_pts=image_pts,
        sport="pickleball",
        candidate_segments=candidates,
        discriminating_line_ids=("near_nvz", "far_nvz", "near_centerline", "far_centerline", "net"),
    )
    tennis = _score_tap_template_against_candidates(
        image_pts=image_pts,
        sport="tennis",
        candidate_segments=candidates,
        discriminating_line_ids=("near_service_line", "far_service_line", "net"),
    )
    margin = tennis["evidence_mass"] - pickleball["evidence_mass"]
    pickleball_line_count = max(1, len(pickleball["line_scores"]))
    pickleball_mean_score = pickleball["evidence_mass"] / pickleball_line_count
    tennis_better = tennis["evidence_mass"] >= 1.6 and margin >= 0.35
    weak_pickleball = pickleball_mean_score < 0.55
    suspect = tennis_better or weak_pickleball
    verdict = "suspect_tennis_corners" if suspect else "consistent"
    if tennis_better:
        reason = "tennis_template_explains_line_evidence_better"
    elif weak_pickleball:
        reason = "pickleball_line_evidence_weak"
    else:
        reason = "pickleball_template_consistent"
    if tennis_better:
        trust_reason = "Trusted taps retained, but tennis-court evidence is stronger; owner confirmation required."
    elif weak_pickleball:
        trust_reason = "Trusted taps retained, but pickleball line evidence is weak; owner confirmation required."
    else:
        trust_reason = "Trusted taps are consistent with pickleball line evidence."
    return {
        "verdict": verdict,
        "needs_user_confirmation": suspect,
        "owner_tap_trusted": True,
        "reason": reason,
        "trust_note": {
            "grade": "warn" if suspect else "good",
            "reason": trust_reason,
        },
        "evidence": {
            "pickleball": pickleball,
            "tennis": tennis,
            "tennis_minus_pickleball_evidence_mass": round(margin, 4),
            "pickleball_mean_line_score": round(pickleball_mean_score, 4),
        },
    }


def calibration_from_manual_tap_frames(paths: Iterable[str | Path], *, sport: Sport) -> CourtCalibration:
    sidecars = [load_capture_sidecar(path) for path in paths]
    if not sidecars:
        raise ValueError("at least one sidecar frame is required")

    paired = [manual_tap_correspondences(sidecar, sport=sport) for sidecar in sidecars]
    first_world_pts = paired[0][1]
    point_count = len(paired[0][0])
    for _, world_pts in paired[1:]:
        if world_pts != first_world_pts:
            raise ValueError("manual tap frames must use the same world point order")

    averaged_image_pts = []
    for point_idx in range(point_count):
        x_values = [image_pts[point_idx][0] for image_pts, _ in paired]
        y_values = [image_pts[point_idx][1] for image_pts, _ in paired]
        averaged_image_pts.append([sum(x_values) / len(x_values), sum(y_values) / len(y_values)])

    return _build_calibration(sidecars[0], sport=sport, image_pts=averaged_image_pts, world_pts=first_world_pts)


def metric_calibration_from_sidecar_and_keypoints(
    sidecar_path: str | Path,
    keypoints_path: str | Path,
    *,
    sport: Sport,
) -> CourtCalibration:
    if sport != "pickleball":
        raise ValueError("metric court calibration currently supports pickleball")

    sidecar = load_capture_sidecar(sidecar_path)
    keypoints = load_court_keypoints(keypoints_path)
    if (
        sidecar.arkit_camera_pose is None
        or sidecar.court_plane is None
        or sidecar.intrinsics.source.lower() != "arkit"
    ):
        raise ValueError("trusted ARKit floor plane is required for metric court calibration")

    from .court_keypoint_net import PICKLEBALL_KEYPOINT_BY_NAME
    from .court_positioning import (
        CameraFloorGeometry,
        back_project_pixel_to_floor,
        estimate_ground_sample_distance,
        estimate_position_uncertainty,
        solve_metric_court_placement,
    )
    from .court_positioning_artifacts import build_metric_court_calibration_artifact

    geometry = CameraFloorGeometry(
        intrinsics=sidecar.intrinsics.model_dump(mode="json"),
        camera_origin_world=sidecar.arkit_camera_pose.t,
        R_world_camera=sidecar.arkit_camera_pose.R,
        floor_plane_point=sidecar.court_plane.point,
        floor_plane_normal=sidecar.court_plane.normal,
    )
    image_keypoints = {
        keypoint.name: {
            "uv": [float(keypoint.uv[0]), float(keypoint.uv[1])],
            "confidence": float(keypoint.confidence),
            "inlier_frames": [int(frame) for frame in keypoint.inlier_frames],
            "recovered": keypoint.recovered,
        }
        for keypoint in keypoints.keypoints
    }
    world_keypoints = {
        name: back_project_pixel_to_floor(image_keypoints[name]["uv"], geometry)
        for name in PICKLEBALL_COURT_KEYPOINT_NAMES
    }
    placement = solve_metric_court_placement(world_keypoints)
    if placement.solved_keypoints != PICKLEBALL_COURT_KEYPOINT_NAMES:
        raise ValueError("metric court calibration requires all 15 canonical pickleball keypoints")

    court_pts = [list(PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m) for name in placement.solved_keypoints]
    image_pts = [image_keypoints[name]["uv"] for name in placement.solved_keypoints]
    homography = homography_from_planar_points(court_pts, image_pts)
    projected_pts = project_planar_points(homography, court_pts)
    error = reprojection_error(image_pts, projected_pts)
    per_keypoint_residual_px = [
        math.hypot(observed[0] - projected[0], observed[1] - projected[1])
        for observed, projected in zip(image_pts, projected_pts, strict=True)
    ]
    per_keypoint_residual_px = [0.0 if residual < 1e-9 else residual for residual in per_keypoint_residual_px]
    calibration_sigma_m = 0.0 if placement.residual_error_m.p95 < 1e-9 else placement.residual_error_m.p95
    plane_sigma_m = 0.012
    gsd_samples = []
    for name in placement.solved_keypoints:
        uv = image_keypoints[name]["uv"]
        gsd = estimate_ground_sample_distance(uv, geometry)
        sigma = estimate_position_uncertainty(
            pixel_error_px=1.0,
            gsd_m_per_px=gsd,
            plane_sigma_m=plane_sigma_m,
            calibration_sigma_m=calibration_sigma_m,
        )
        canonical = PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m
        gsd_samples.append(
            {
                "court_xy": [float(canonical[0]), float(canonical[1])],
                "gsd_m_per_px": gsd,
                "sigma_p_m": sigma,
            }
        )

    extrinsics = _solve_or_seed_extrinsics(
        sidecar,
        world_pts=[world_keypoints[name] for name in placement.solved_keypoints],
        image_pts=image_pts,
    )
    payload = build_metric_court_calibration_artifact(
        placement=placement,
        intrinsics=sidecar.intrinsics.model_dump(mode="json"),
        homography=homography,
        image_keypoints=image_keypoints,
        extrinsics=extrinsics.model_dump(mode="json"),
        reprojection_error_px=error.model_dump(mode="json"),
        per_keypoint_residual_px=per_keypoint_residual_px,
        gsd_model={
            "type": "analytic_ray_plane",
            "plane_sigma_m": plane_sigma_m,
            "calibration_sigma_m": calibration_sigma_m,
            "samples": gsd_samples,
        },
        capture_quality=_merge_capture_quality(sidecar, error, len(image_pts)).model_dump(mode="json"),
        source="arkit_plane_keypoint_metric_solve_v1",
        solved_over_frames=keypoints.frame_indexes,
        sport=sport,
    )
    return CourtCalibration.model_validate(payload)


def _build_calibration(
    sidecar: CaptureSidecar,
    *,
    sport: Sport,
    image_pts: list[list[float]],
    world_pts: list[list[float]],
) -> CourtCalibration:
    homography = homography_from_planar_points(world_pts, image_pts)
    projected_pts = project_planar_points(homography, world_pts)
    error = reprojection_error(image_pts, projected_pts)
    extrinsics = _solve_or_seed_extrinsics(sidecar, world_pts=world_pts, image_pts=image_pts)
    return CourtCalibration(
        schema_version=1,
        sport=sport,
        homography=homography,
        intrinsics=sidecar.intrinsics,
        image_size=tuple(sidecar.resolution),
        extrinsics=extrinsics,
        reprojection_error_px=error,
        capture_quality=_merge_capture_quality(sidecar, error, len(image_pts)),
        image_pts=image_pts,
        world_pts=world_pts,
    )


def _score_tap_template_against_candidates(
    *,
    image_pts: Sequence[Sequence[float]],
    sport: Sport,
    candidate_segments: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    discriminating_line_ids: Sequence[str],
) -> dict[str, Any]:
    from .court_line_evidence import score_line_candidate

    template = get_court_template(sport)
    homography = homography_from_planar_points(template.corners_m, image_pts)
    line_scores: dict[str, float] = {}
    evidence_mass = 0.0
    for line_id in discriminating_line_ids:
        expected_world = template.line_segments_m.get(line_id)
        if expected_world is None:
            continue
        expected_image = project_planar_points(homography, expected_world)
        best = max((score_line_candidate(expected_image, segment) for segment in candidate_segments), key=lambda score: score.score)
        line_scores[line_id] = round(float(best.score), 4)
        evidence_mass += float(best.score)
    return {
        "evidence_mass": round(evidence_mass, 4),
        "line_scores": line_scores,
    }


def _segment2(segment: Sequence[Sequence[float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    if len(segment) != 2 or len(segment[0]) != 2 or len(segment[1]) != 2:
        raise ValueError("candidate segment must contain exactly two 2D points")
    return (
        (float(segment[0][0]), float(segment[0][1])),
        (float(segment[1][0]), float(segment[1][1])),
    )


def _solve_or_seed_extrinsics(
    sidecar: CaptureSidecar,
    *,
    world_pts: list[list[float]],
    image_pts: list[list[float]],
) -> CourtExtrinsics:
    try:
        return solve_camera_pose(world_pts, image_pts, sidecar.intrinsics)
    except (RuntimeError, ValueError):
        return CourtExtrinsics(
            R=sidecar.arkit_camera_pose.R if sidecar.arkit_camera_pose is not None else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=sidecar.arkit_camera_pose.t if sidecar.arkit_camera_pose is not None else [0.0, 0.0, 0.0],
            camera_height_m=_camera_height_from_sidecar(sidecar),
        )


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


def _invert_homography(homography: Iterable[Iterable[float]]) -> list[list[float]]:
    h = [[float(value) for value in row] for row in homography]
    if len(h) != 3 or any(len(row) != 3 for row in h):
        raise ValueError("homography must be a 3x3 matrix")

    a, b, c = h[0]
    d, e, f = h[1]
    g, i, j = h[2]
    cofactor = [
        [e * j - f * i, c * i - b * j, b * f - c * e],
        [f * g - d * j, a * j - c * g, c * d - a * f],
        [d * i - e * g, b * g - a * i, a * e - b * d],
    ]
    determinant = a * cofactor[0][0] + b * cofactor[1][0] + c * cofactor[2][0]
    if math.isclose(determinant, 0.0, abs_tol=1e-12):
        raise ValueError("homography is singular")
    return [[value / determinant for value in row] for row in cofactor]


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(rhs)
    augmented = [list(row) + [float(value)] for row, value in zip(matrix, rhs, strict=True)]

    for pivot_idx in range(size):
        pivot_row = max(range(pivot_idx, size), key=lambda row_idx: abs(augmented[row_idx][pivot_idx]))
        if math.isclose(augmented[pivot_row][pivot_idx], 0.0, abs_tol=1e-12):
            raise ValueError("degenerate homography")
        augmented[pivot_idx], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_idx]

        pivot = augmented[pivot_idx][pivot_idx]
        for col_idx in range(pivot_idx, size + 1):
            augmented[pivot_idx][col_idx] /= pivot

        for row_idx in range(size):
            if row_idx == pivot_idx:
                continue
            factor = augmented[row_idx][pivot_idx]
            if math.isclose(factor, 0.0, abs_tol=1e-12):
                continue
            for col_idx in range(pivot_idx, size + 1):
                augmented[row_idx][col_idx] -= factor * augmented[pivot_idx][col_idx]

    return [augmented[row_idx][size] for row_idx in range(size)]
