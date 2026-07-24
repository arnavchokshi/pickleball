"""Separate non-planar pickleball net-top stage.

The floor homography is used only to recover a camera pose.  The three net-top
points are then projected at regulation 3-D heights (36 inches at each post,
34 inches at center).  They never participate in fitting the floor transform.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from threed.racketsport.court_camera_geometry import (
    PinholeIntrinsics,
    project_camera_points_radial_k1,
    validate_bounded_k1,
)


NET_POST_HEIGHT_M = 0.9144
NET_CENTER_HEIGHT_M = 0.8636
NET_HALF_WIDTH_M = 3.048
NET_POINT_NAMES = ("net_left_sideline", "net_center", "net_right_sideline")
NET_WORLD_XYZ_M = np.asarray(
    (
        (-NET_HALF_WIDTH_M, 0.0, NET_POST_HEIGHT_M),
        (0.0, 0.0, NET_CENTER_HEIGHT_M),
        (NET_HALF_WIDTH_M, 0.0, NET_POST_HEIGHT_M),
    ),
    dtype=np.float64,
)


def decompose_floor_homography(
    homography_image_from_court: Sequence[Sequence[float]] | np.ndarray,
    intrinsics: PinholeIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover a proper world-to-camera pose from ``H = K [r1 r2 t]``."""

    homography = _homography(homography_image_from_court)
    normalized = np.linalg.inv(intrinsics.matrix) @ homography
    norm_1 = float(np.linalg.norm(normalized[:, 0]))
    norm_2 = float(np.linalg.norm(normalized[:, 1]))
    if min(norm_1, norm_2) <= 1.0e-9:
        raise ValueError("floor homography is degenerate for pose recovery")
    scale = 2.0 / (norm_1 + norm_2)
    raw_r1 = normalized[:, 0] * scale
    raw_r2 = normalized[:, 1] * scale
    raw_r3 = np.cross(raw_r1, raw_r2)
    raw_rotation = np.column_stack((raw_r1, raw_r2, raw_r3))
    left, _, right_t = np.linalg.svd(raw_rotation)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    translation = normalized[:, 2] * scale
    if translation[2] < 0.0:
        # H and -H encode the same plane mapping.  Choose the cheirality branch
        # that places the court in front of the camera.
        rotation[:, :2] *= -1.0
        rotation[:, 2] = np.cross(rotation[:, 0], rotation[:, 1])
        translation *= -1.0
    if np.linalg.det(rotation) <= 0.0 or translation[2] <= 0.0:
        raise ValueError("floor homography did not yield a positive-depth proper pose")
    return rotation, translation


def project_regulation_net_top(
    homography_image_from_court: Sequence[Sequence[float]] | np.ndarray,
    intrinsics: PinholeIntrinsics,
    *,
    k1: float = 0.0,
    transform_covariance: Sequence[Sequence[float]] | np.ndarray | None = None,
    k1_variance: float = 0.0,
    floor_court_confidence: float = 0.0,
) -> dict[str, Any]:
    """Project the constrained regulation net and report separate uncertainty."""

    coefficient = validate_bounded_k1(k1)
    if not math.isfinite(float(k1_variance)) or float(k1_variance) < 0.0:
        raise ValueError("k1_variance must be non-negative and finite")
    floor_confidence = min(1.0, max(0.0, float(floor_court_confidence)))
    rotation, translation = decompose_floor_homography(homography_image_from_court, intrinsics)
    camera_points = (rotation @ NET_WORLD_XYZ_M.T).T + translation
    image_points = project_camera_points_radial_k1(
        camera_points,
        intrinsics,
        k1=coefficient,
    )
    covariance_by_name = _net_projection_covariances(
        homography_image_from_court,
        intrinsics,
        k1=coefficient,
        transform_covariance=transform_covariance,
        k1_variance=float(k1_variance),
    )
    point_confidence: dict[str, float] = {}
    for name in NET_POINT_NAMES:
        covariance = covariance_by_name[name]
        radial_sigma = math.sqrt(max(float(np.linalg.eigvalsh(covariance).max()), 0.0))
        point_confidence[name] = floor_confidence * math.exp(-radial_sigma / 5.0)
    return {
        "schema_version": 1,
        "source": "regulation_3d_projection_from_floor_pose",
        "floor_homography_fit_used_net_top": False,
        "net_shape": "36in_post_34in_center_sag",
        "keypoints_xy": {
            name: [float(image_points[index, 0]), float(image_points[index, 1])]
            for index, name in enumerate(NET_POINT_NAMES)
        },
        "world_xyz_m": {
            name: [float(value) for value in NET_WORLD_XYZ_M[index]]
            for index, name in enumerate(NET_POINT_NAMES)
        },
        "point_covariance_px2": {
            name: covariance_by_name[name].tolist() for name in NET_POINT_NAMES
        },
        "point_confidence": point_confidence,
        "net_confidence": min(point_confidence.values()),
        "camera_pose": {
            "rotation_world_to_camera": rotation.tolist(),
            "translation_world_to_camera_m": translation.tolist(),
        },
        "distortion": {"model": "radial_k1", "k1": coefficient, "k1_variance": float(k1_variance)},
        "authority_state": "review_only",
        "measurement_valid": False,
    }


def apply_bounded_net_residual(
    projected_keypoints_xy: Mapping[str, Sequence[float]],
    residual_xy: Mapping[str, Sequence[float]],
    *,
    max_residual_px: float = 12.0,
) -> dict[str, list[float]]:
    """Apply a future learned residual head without allowing it to break the 3-D prior."""

    if not math.isfinite(float(max_residual_px)) or float(max_residual_px) <= 0.0:
        raise ValueError("max_residual_px must be positive and finite")
    result: dict[str, list[float]] = {}
    for name in NET_POINT_NAMES:
        base = np.asarray(projected_keypoints_xy[name], dtype=np.float64)
        raw = np.asarray(residual_xy.get(name, (0.0, 0.0)), dtype=np.float64)
        if base.shape != (2,) or raw.shape != (2,) or not np.isfinite(base).all() or not np.isfinite(raw).all():
            raise ValueError(f"invalid bounded net residual for {name}")
        bounded = float(max_residual_px) * np.tanh(raw / float(max_residual_px))
        result[name] = [float(value) for value in base + bounded]
    return result


def evaluate_net_reprojection(
    predicted_xy: Mapping[str, Sequence[float]],
    truth_xy: Mapping[str, Sequence[float]],
    *,
    predicted_world_xyz_m: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Exact-semantic 2-D/3-D evaluation for the separate net stage."""

    errors: dict[str, float] = {}
    for name in NET_POINT_NAMES:
        prediction = np.asarray(predicted_xy[name], dtype=np.float64)
        truth = np.asarray(truth_xy[name], dtype=np.float64)
        if prediction.shape != (2,) or truth.shape != (2,):
            raise ValueError(f"net evaluation point {name} must contain two coordinates")
        errors[name] = float(np.linalg.norm(prediction - truth))
    values = np.asarray(list(errors.values()), dtype=np.float64)
    height_errors_cm = None
    if predicted_world_xyz_m is not None:
        height_errors_cm = {
            name: abs(float(predicted_world_xyz_m[name][2]) - float(NET_WORLD_XYZ_M[index, 2])) * 100.0
            for index, name in enumerate(NET_POINT_NAMES)
        }
    return {
        "protocol": "exact_semantic_3d_net_reprojection_v1",
        "point_error_px": errors,
        "median_error_px": float(np.median(values)),
        "p95_error_px": float(np.percentile(values, 95)),
        "max_error_px": float(values.max()),
        "height_error_cm": height_errors_cm,
        "height_error_cm_max": None if height_errors_cm is None else max(height_errors_cm.values()),
    }


def _net_projection_covariances(
    homography: Sequence[Sequence[float]] | np.ndarray,
    intrinsics: PinholeIntrinsics,
    *,
    k1: float,
    transform_covariance: Sequence[Sequence[float]] | np.ndarray | None,
    k1_variance: float,
) -> dict[str, np.ndarray]:
    base_h = _homography(homography)
    if transform_covariance is None:
        covariance = np.eye(8, dtype=np.float64) * 4.0
    else:
        covariance = np.asarray(transform_covariance, dtype=np.float64)
        if covariance.shape != (8, 8) or not np.isfinite(covariance).all():
            raise ValueError("transform_covariance must be a finite 8x8 matrix")
    parameters = np.asarray(
        [base_h[0, 0], base_h[0, 1], base_h[0, 2], base_h[1, 0], base_h[1, 1], base_h[1, 2], base_h[2, 0], base_h[2, 1]],
        dtype=np.float64,
    )

    def projection(params: np.ndarray, coefficient: float) -> np.ndarray:
        h = np.asarray(
            [[params[0], params[1], params[2]], [params[3], params[4], params[5]], [params[6], params[7], 1.0]],
            dtype=np.float64,
        )
        rotation, translation = decompose_floor_homography(h, intrinsics)
        camera = (rotation @ NET_WORLD_XYZ_M.T).T + translation
        return project_camera_points_radial_k1(camera, intrinsics, k1=coefficient)

    base = projection(parameters, k1)
    jacobian = np.zeros((6, 8), dtype=np.float64)
    for index in range(8):
        epsilon = 1.0e-5 * max(1.0, abs(float(parameters[index])))
        perturbed = parameters.copy()
        perturbed[index] += epsilon
        try:
            jacobian[:, index] = ((projection(perturbed, k1) - base) / epsilon).reshape(-1)
        except ValueError:
            jacobian[:, index] = 0.0
    combined = jacobian @ covariance @ jacobian.T
    if k1_variance > 0.0:
        epsilon = 1.0e-5
        derivative = ((projection(parameters, k1 + epsilon) - base) / epsilon).reshape(-1, 1)
        combined += derivative @ derivative.T * k1_variance
    result: dict[str, np.ndarray] = {}
    for index, name in enumerate(NET_POINT_NAMES):
        block = combined[index * 2 : index * 2 + 2, index * 2 : index * 2 + 2]
        result[name] = 0.5 * (block + block.T) + np.eye(2, dtype=np.float64) * 1.0e-6
    return result


def _homography(value: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    homography = np.asarray(value, dtype=np.float64)
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError("homography must be a finite 3x3 matrix")
    if abs(float(homography[2, 2])) <= 1.0e-12 or np.linalg.matrix_rank(homography) < 3:
        raise ValueError("homography must be finite, full rank, and normalizable")
    return homography / float(homography[2, 2])


__all__ = [
    "NET_CENTER_HEIGHT_M",
    "NET_HALF_WIDTH_M",
    "NET_POINT_NAMES",
    "NET_POST_HEIGHT_M",
    "NET_WORLD_XYZ_M",
    "apply_bounded_net_residual",
    "decompose_floor_homography",
    "evaluate_net_reprojection",
    "project_regulation_net_top",
]
