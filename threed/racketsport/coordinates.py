"""Typed coordinate-space vocabulary and canonical rigid camera transforms.

The spaces named here are deliberately explicit about raster state, axes, and
units.  They are a small NS-01.4 slice, not a claim that every producer has
already adopted the vocabulary.

``pixels_raw_native``
    Native decoded-frame raster coordinates before lens undistortion.  The
    origin is the top-left of the native encoded image, +x points right, +y
    points down, and values are pixels in the calibration image dimensions.
``pixels_undistorted_native``
    The same native raster origin, axes, dimensions, and pixel units after
    applying the declared camera distortion model and native intrinsics.
``pixels_preview_scaled``
    Coordinates in a resized, cropped, and/or orientation-corrected preview.
    The origin is the preview's top-left, +x right, +y down, in preview pixels;
    these are not calibration coordinates without an explicit inverse map.
``camera_m``
    OpenCV pinhole camera coordinates in metres: origin at the optical centre,
    +x image-right, +y image-down, and +z forward along the optical axis.
``body_camera_root_relative_m``
    Fast-SAM-3D-Body/MHR camera-axis offsets in metres before ``pred_cam_t``.
    The body root is the translation origin; axes match ``camera_m``.
``world_court_netcenter_z_up_m``
    Metric court coordinates with origin at floor-level net centre, +x across
    court width, +y along court length (the net plane normal is [0, 1, 0]), and
    +z up.  The schema literal is ``court_netcenter_z_up_m``.
``world_xy_homography_m``
    The 2-D z=0 court-plane slice used by the pixel-to-court homography: origin
    at net centre, +x across width, +y along length, in metres.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Final, Literal, Mapping, Sequence

import numpy as np


class CoordinateSpace(str, Enum):
    """Stable names for coordinate-bearing artifacts and APIs."""

    __str__ = str.__str__

    PIXELS_RAW_NATIVE = "pixels_raw_native"
    PIXELS_UNDISTORTED_NATIVE = "pixels_undistorted_native"
    PIXELS_PREVIEW_SCALED = "pixels_preview_scaled"
    CAMERA_M = "camera_m"
    BODY_CAMERA_ROOT_RELATIVE_M = "body_camera_root_relative_m"
    WORLD_COURT_NETCENTER_Z_UP_M = "world_court_netcenter_z_up_m"
    WORLD_XY_HOMOGRAPHY_M = "world_xy_homography_m"


HomographyPixelConvention = Literal["raw_pixels", "undistorted_pixels"]
HOMOGRAPHY_PIXEL_CONVENTIONS: Final[tuple[HomographyPixelConvention, ...]] = (
    "raw_pixels",
    "undistorted_pixels",
)

LEGACY_COURT_WORLD_FRAME: Final = "court_Z0"
CANONICAL_COURT_WORLD_FRAME: Final = "court_netcenter_z_up_m"


def resolve_world_coordinate_space(
    payload: Mapping[str, Any],
    *,
    default: CoordinateSpace = CoordinateSpace.WORLD_COURT_NETCENTER_Z_UP_M,
) -> CoordinateSpace:
    """Resolve canonical and legacy world declarations without rewriting values.

    Older skeleton and paddle artifacts only carry ``world_frame="court_Z0"``.
    New artifacts may additionally carry the typed ``coordinate_space`` and the
    schema's canonical ``coordinate_frame``.  Missing declarations retain the
    historical court-world assumption so old callers and fixtures remain valid.
    Conflicting explicit declarations fail closed instead of silently relabeling
    coordinates.
    """

    declarations: list[CoordinateSpace] = []
    coordinate_space = payload.get("coordinate_space")
    if coordinate_space is not None:
        declarations.append(CoordinateSpace(str(coordinate_space)))

    coordinate_frame = payload.get("coordinate_frame")
    if coordinate_frame is not None:
        if str(coordinate_frame) != CANONICAL_COURT_WORLD_FRAME:
            raise ValueError(f"unsupported coordinate_frame: {coordinate_frame}")
        declarations.append(CoordinateSpace.WORLD_COURT_NETCENTER_Z_UP_M)

    world_frame = payload.get("world_frame")
    if world_frame is not None:
        if str(world_frame) not in {LEGACY_COURT_WORLD_FRAME, CANONICAL_COURT_WORLD_FRAME}:
            raise ValueError(f"unsupported world_frame: {world_frame}")
        declarations.append(CoordinateSpace.WORLD_COURT_NETCENTER_Z_UP_M)

    if not declarations:
        return default
    if any(space != declarations[0] for space in declarations[1:]):
        raise ValueError("conflicting world coordinate declarations")
    if declarations[0] != CoordinateSpace.WORLD_COURT_NETCENTER_Z_UP_M:
        raise ValueError(f"unsupported world coordinate space: {declarations[0]}")
    return declarations[0]


def project_world_points(
    extrinsics: Any,
    intrinsics: Any,
    world_points: Any,
    *,
    input_space: CoordinateSpace,
    output_space: CoordinateSpace,
    reference_space: CoordinateSpace,
) -> list[list[float]]:
    """Canonical typed adapter for the repository's legacy pinhole projection.

    The existing math produces ideal/native pinhole pixels and does not apply
    lens distortion, so ``output_space`` must be
    ``PIXELS_UNDISTORTED_NATIVE``.  ``reference_space`` declares the raster
    convention of evidence compared with the projection (for example raw
    detector boxes).  It is metadata at this parity-only seam: no distortion,
    resize, crop, or reference-frame transform is introduced here.
    """

    from .court_calibration import project_world_points_typed as _project

    return _project(
        extrinsics,
        intrinsics,
        world_points,
        input_space=input_space,
        output_space=output_space,
        reference_space=reference_space,
    )


def invert_extrinsics(R: Any, t: Any) -> tuple[np.ndarray, np.ndarray]:
    """Invert canonical world-to-camera extrinsics.

    The stored convention is ``camera_column = R @ world_column + t``.  The
    result is ``R_camera_to_world = R.T`` and camera centre
    ``C_world = -R.T @ t``.  For offsets, the repository's
    ``skeleton_upright.rotate_camera_offsets_row_times_R`` expression
    ``offset_row @ R`` is the row-vector equivalent of
    ``R.T @ offset_column``.
    """

    rotation = _rotation_matrix(R)
    translation = _vector3(t, name="t")
    camera_to_world = rotation.T
    camera_center_world = -(camera_to_world @ translation)
    return camera_to_world, camera_center_world


def world_to_camera_points(points_world: Any, R: Any, t: Any) -> np.ndarray:
    """Apply ``camera = R @ world + t`` to one or many 3-D points."""

    points = _points3(points_world, name="points_world")
    rotation = _rotation_matrix(R)
    translation = _vector3(t, name="t")
    return points @ rotation.T + translation


def camera_to_world_points(points_camera: Any, R: Any, t: Any) -> np.ndarray:
    """Apply the inverse of ``camera = R @ world + t`` to 3-D points."""

    points = _points3(points_camera, name="points_camera")
    camera_to_world, camera_center_world = invert_extrinsics(R, t)
    return points @ camera_to_world.T + camera_center_world


def apply_translation_once(
    points: Any,
    translation: Sequence[float] | None,
    already_applied: bool = False,
) -> list[list[float]]:
    """Return points with a 3-vector translation applied at most once.

    ``None`` points preserve the decode surfaces' historical empty-list
    behavior.  ``already_applied=True`` is the explicit escape hatch for
    sidecars whose coordinates already contain the translation.
    """

    if points is None:
        return []
    normalized = [[float(value) for value in point] for point in points]
    if translation is None or already_applied:
        return normalized
    if len(translation) != 3:
        raise ValueError("translation must be a 3-vector")
    delta = [float(translation[index]) for index in range(3)]
    return [[point[index] + delta[index] for index in range(3)] for point in normalized]


def camera_matrix_from_intrinsics(intrinsics: Any) -> list[list[float]]:
    """Build the blessed pinhole K matrix through court calibration code."""

    from .court_calibration import camera_matrix_from_intrinsics as _builder

    return _builder(intrinsics)


def _rotation_matrix(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError("R must be a 3x3 matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError("R must contain finite values")
    return array


def _vector3(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _points3(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1:] != (3,):
        raise ValueError(f"{name} must have shape (..., 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


__all__ = [
    "CANONICAL_COURT_WORLD_FRAME",
    "CoordinateSpace",
    "HOMOGRAPHY_PIXEL_CONVENTIONS",
    "HomographyPixelConvention",
    "LEGACY_COURT_WORLD_FRAME",
    "apply_translation_once",
    "camera_matrix_from_intrinsics",
    "camera_to_world_points",
    "invert_extrinsics",
    "project_world_points",
    "resolve_world_coordinate_space",
    "world_to_camera_points",
]
