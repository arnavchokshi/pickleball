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

LengthUnit = Literal["m", "meter", "meters", "cm", "centimeter", "centimeters"]

_RASTER_SPACES: Final[frozenset[CoordinateSpace]] = frozenset(
    {
        CoordinateSpace.PIXELS_RAW_NATIVE,
        CoordinateSpace.PIXELS_UNDISTORTED_NATIVE,
        CoordinateSpace.PIXELS_PREVIEW_SCALED,
    }
)


def homography_pixel_space(convention: HomographyPixelConvention | str) -> CoordinateSpace:
    """Map the frozen homography convention vocabulary to a raster space."""

    normalized = str(convention)
    if normalized == "raw_pixels":
        return CoordinateSpace.PIXELS_RAW_NATIVE
    if normalized == "undistorted_pixels":
        return CoordinateSpace.PIXELS_UNDISTORTED_NATIVE
    raise ValueError(f"unsupported homography_pixel_convention: {convention}")


def resolve_homography_pixel_convention(
    payload: Mapping[str, Any],
    *,
    default: HomographyPixelConvention = "raw_pixels",
) -> HomographyPixelConvention:
    """Resolve legacy/top-level and typed nested homography declarations.

    Missing declarations retain the historical raw-pixel default.  Any
    explicit disagreement fails closed instead of silently selecting one
    coordinate convention.
    """

    declarations: list[str] = []
    top_level = payload.get("homography_pixel_convention")
    if top_level is not None:
        declarations.append(str(top_level))

    contract = payload.get("coordinate_contract")
    if contract is not None:
        if not isinstance(contract, Mapping):
            raise ValueError("coordinate_contract must be an object")
        nested = contract.get("homography_pixel_convention")
        if nested is not None:
            declarations.append(str(nested))
        output_space = contract.get("homography_output_space")
        if output_space is not None:
            output = CoordinateSpace(str(output_space))
            if output == CoordinateSpace.PIXELS_RAW_NATIVE:
                declarations.append("raw_pixels")
            elif output == CoordinateSpace.PIXELS_UNDISTORTED_NATIVE:
                declarations.append("undistorted_pixels")
            else:
                raise ValueError(f"unsupported homography_output_space: {output_space}")

    if not declarations:
        declarations.append(str(default))
    for declaration in declarations:
        homography_pixel_space(declaration)
    if any(declaration != declarations[0] for declaration in declarations[1:]):
        raise ValueError("conflicting homography pixel declarations")
    return declarations[0]  # type: ignore[return-value]


def require_same_raster_space(
    input_space: CoordinateSpace,
    reference_space: CoordinateSpace,
) -> CoordinateSpace:
    """Fail closed unless two explicitly declared raster spaces match."""

    input_space = CoordinateSpace(input_space)
    reference_space = CoordinateSpace(reference_space)
    if input_space not in _RASTER_SPACES or reference_space not in _RASTER_SPACES:
        raise ValueError("input and reference spaces must both be raster spaces")
    if input_space != reference_space:
        raise ValueError(f"coordinate-space mismatch: {input_space} != {reference_space}")
    return input_space


def unproject_image_points_with_inverse(
    inverse_homography: Any,
    image_points: Any,
    *,
    input_space: CoordinateSpace,
    homography_space: CoordinateSpace,
    output_space: CoordinateSpace,
) -> np.ndarray:
    """Typed NumPy homography inverse preserving the placement arithmetic."""

    require_same_raster_space(input_space, homography_space)
    if CoordinateSpace(output_space) != CoordinateSpace.WORLD_XY_HOMOGRAPHY_M:
        raise ValueError(f"homography output must be world_xy_homography_m, got {output_space}")
    inverse = np.asarray(inverse_homography, dtype=float)
    if inverse.shape != (3, 3) or not np.all(np.isfinite(inverse)):
        raise ValueError("inverse_homography must be a finite 3x3 matrix")
    points = np.asarray(image_points, dtype=float)
    if points.ndim == 0 or points.shape[-1:] != (2,) or not np.all(np.isfinite(points)):
        raise ValueError("image_points must have shape (..., 2) with finite values")
    flat = points.reshape(-1, 2)
    output: list[np.ndarray] = []
    for pixel in flat:
        projected = inverse @ np.array([float(pixel[0]), float(pixel[1]), 1.0], dtype=float)
        if abs(float(projected[2])) < 1e-12:
            raise ValueError("homography projection reached zero scale")
        output.append(projected[:2] / projected[2])
    return np.asarray(output, dtype=float).reshape(points.shape)


def unproject_image_points_to_world(
    homography: Any,
    image_points: Any,
    *,
    input_space: CoordinateSpace,
    homography_space: CoordinateSpace,
    output_space: CoordinateSpace,
) -> list[list[float]]:
    """Canonical strict adapter around the legacy homography unprojection."""

    require_same_raster_space(input_space, homography_space)
    from .court_calibration import project_image_points_to_world_typed as _unproject

    return _unproject(
        homography,
        image_points,
        input_space=input_space,
        homography_space=homography_space,
        output_space=output_space,
    )


def project_world_xy_points(
    homography: Any,
    world_points: Any,
    *,
    input_space: CoordinateSpace,
    output_space: CoordinateSpace,
    homography_space: CoordinateSpace,
) -> list[list[float]]:
    """Canonical typed adapter for world-plane to image homographies."""

    from .court_calibration import project_planar_points_typed as _project

    return _project(
        homography,
        world_points,
        input_space=input_space,
        output_space=output_space,
        homography_space=homography_space,
    )


def scale_raster_points(
    points: Any,
    *,
    source_size: tuple[float, float],
    target_size: tuple[float, float],
    input_space: CoordinateSpace,
    output_space: CoordinateSpace,
) -> list[list[float]]:
    """Scale raster points with an explicit native/preview direction."""

    input_space = CoordinateSpace(input_space)
    output_space = CoordinateSpace(output_space)
    allowed = {
        (CoordinateSpace.PIXELS_RAW_NATIVE, CoordinateSpace.PIXELS_PREVIEW_SCALED),
        (CoordinateSpace.PIXELS_UNDISTORTED_NATIVE, CoordinateSpace.PIXELS_PREVIEW_SCALED),
        (CoordinateSpace.PIXELS_PREVIEW_SCALED, CoordinateSpace.PIXELS_RAW_NATIVE),
        (CoordinateSpace.PIXELS_PREVIEW_SCALED, CoordinateSpace.PIXELS_UNDISTORTED_NATIVE),
    }
    if (input_space, output_space) not in allowed:
        raise ValueError(f"unsupported raster scaling: {input_space} -> {output_space}")
    source_width, source_height = (float(source_size[0]), float(source_size[1]))
    target_width, target_height = (float(target_size[0]), float(target_size[1]))
    if min(source_width, source_height, target_width, target_height) <= 0.0:
        raise ValueError("source_size and target_size values must be > 0")
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    return [
        [float(point[0]) * scale_x, float(point[1]) * scale_y]
        for point in points
    ]


def project_world_array_pinhole(
    world_points: Any,
    *,
    rotation: Any,
    translation: Any,
    intrinsics: Any,
    input_space: CoordinateSpace,
    output_space: CoordinateSpace,
    reference_space: CoordinateSpace,
    np_module: Any = np,
) -> Any:
    """Typed camera-model seam preserving the ball solver's NumPy math."""

    if CoordinateSpace(input_space) != CoordinateSpace.WORLD_COURT_NETCENTER_Z_UP_M:
        raise ValueError(f"pinhole input must be court world metres, got {input_space}")
    if CoordinateSpace(output_space) != CoordinateSpace.PIXELS_UNDISTORTED_NATIVE:
        raise ValueError(f"pinhole output must be undistorted native pixels, got {output_space}")
    reference_space = CoordinateSpace(reference_space)
    if reference_space not in _RASTER_SPACES:
        raise ValueError(f"projection reference must be a raster space, got {reference_space}")
    world = np_module.asarray(world_points, dtype=float)
    camera_points = (np_module.asarray(rotation, dtype=float) @ world.T).T + np_module.asarray(
        translation, dtype=float
    )
    depth = camera_points[:, 2]
    depth = np_module.where(np_module.abs(depth) < 1e-9, 1e-9, depth)
    u = float(intrinsics.fx) * camera_points[:, 0] / depth + float(intrinsics.cx)
    v = float(intrinsics.fy) * camera_points[:, 1] / depth + float(intrinsics.cy)
    return np_module.column_stack([u, v])


def validate_opencv_camera_seam(
    *,
    object_space: CoordinateSpace,
    image_reference_space: CoordinateSpace,
    projected_space: CoordinateSpace,
) -> None:
    """Validate the declared spaces around unchanged solvePnP/projectPoints calls."""

    if CoordinateSpace(object_space) != CoordinateSpace.WORLD_COURT_NETCENTER_Z_UP_M:
        raise ValueError(f"solvePnP object space must be court world metres, got {object_space}")
    if CoordinateSpace(image_reference_space) not in _RASTER_SPACES:
        raise ValueError(f"solvePnP image reference must be a raster space, got {image_reference_space}")
    if CoordinateSpace(projected_space) != CoordinateSpace.PIXELS_UNDISTORTED_NATIVE:
        raise ValueError(f"projectPoints output must be undistorted native pixels, got {projected_space}")


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


def translation_to_metres(
    translation: Sequence[float],
    *,
    input_unit: LengthUnit | str,
) -> tuple[float, float, float]:
    """Normalize one explicit 3-D translation seam to metres.

    This is a unit declaration and scalar conversion only.  It deliberately
    preserves the historical ``float(value) * 0.01`` arithmetic used by the
    paddle IPPE path for centimetres, so adopting the canonical API does not
    alter its numeric payload.
    """

    if isinstance(translation, (str, bytes)):
        raise ValueError("translation must be a 3-vector")
    values = tuple(translation)
    if len(values) != 3:
        raise ValueError("translation must be a 3-vector")
    normalized = str(input_unit).strip().lower()
    if normalized in {"m", "meter", "meters"}:
        scale_to_m = 1.0
    elif normalized in {"cm", "centimeter", "centimeters"}:
        scale_to_m = 0.01
    else:
        raise ValueError("input_unit must be 'cm' or 'm'")
    converted = tuple(float(value) * scale_to_m for value in values)
    if not all(np.isfinite(value) for value in converted):
        raise ValueError("translation must contain finite values")
    return converted  # type: ignore[return-value]


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
    "LengthUnit",
    "apply_translation_once",
    "camera_matrix_from_intrinsics",
    "camera_to_world_points",
    "homography_pixel_space",
    "invert_extrinsics",
    "project_world_array_pinhole",
    "project_world_points",
    "project_world_xy_points",
    "require_same_raster_space",
    "resolve_homography_pixel_convention",
    "resolve_world_coordinate_space",
    "scale_raster_points",
    "translation_to_metres",
    "unproject_image_points_to_world",
    "unproject_image_points_with_inverse",
    "validate_opencv_camera_seam",
    "world_to_camera_points",
]
