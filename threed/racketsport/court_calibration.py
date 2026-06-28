"""Per-clip court calibration helpers and solvePnP-ready artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from .capture_quality import score_capture_quality
from .court_templates import Sport, get_court_template
from .schemas import (
    CameraIntrinsics,
    CaptureQuality,
    CaptureSidecar,
    CourtCalibration,
    CourtExtrinsics,
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
