"""Fail-closed adapter from a static court lock to legacy visualization artifacts.

The existing replay/overlay code consumes ``court_calibration.json``,
``court_zones.json``, and ``net_plane.json``.  A structured court lock is useful
for those visualizations before it earns measurement authority, but that bridge
must never upgrade trust.  This module therefore accepts only a bounded,
confirmed-static lock and emits schema-valid artifacts explicitly marked
``visualization_only``, ``review_only``, and ``measurement_valid=false``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from threed.racketsport.court_calibration import (
    homography_from_planar_points,
    project_planar_points,
    solve_camera_pose,
)
from threed.racketsport.court_camera_geometry import undistort_pixels_radial_k1
from threed.racketsport.court_keypoint_net import PICKLEBALL_KEYPOINTS
from threed.racketsport.court_static_lock import CourtLockArtifact
from threed.racketsport.court_zones import build_court_zones
from threed.racketsport.net_plane import build_net_plane
from threed.racketsport.schemas import (
    CameraIntrinsics,
    CaptureQuality,
    CourtCalibration,
    CourtExtrinsics,
    CourtZones,
    NetPlane,
    ReprojectionError,
)


FLOOR_KEYPOINTS = tuple(
    point for point in PICKLEBALL_KEYPOINTS if abs(float(point.world_xyz_m[2])) <= 1.0e-12
)


@dataclass(frozen=True)
class CourtLockVisualizationConfig:
    max_lock_p95_px: float = 20.0
    max_undistortion_refit_p95_px: float = 5.0
    max_k1_standard_deviation: float = 0.20
    min_supporting_frames: int = 1

    def __post_init__(self) -> None:
        for name in (
            "max_lock_p95_px",
            "max_undistortion_refit_p95_px",
            "max_k1_standard_deviation",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
        if isinstance(self.min_supporting_frames, bool) or not 1 <= int(self.min_supporting_frames) <= 8:
            raise ValueError("min_supporting_frames must be an integer in [1, 8]")


@dataclass(frozen=True)
class NamedFloorCorrespondence:
    semantic_name: str
    world_xyz_m: tuple[float, float, float]
    image_xy: tuple[float, float]
    image_coordinate_space: str
    raw_image_xy: tuple[float, float] | None
    source: str = "projected_from_accepted_static_lock"

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_name": self.semantic_name,
            "world_xyz_m": [float(value) for value in self.world_xyz_m],
            "image_xy": [float(value) for value in self.image_xy],
            "image_coordinate_space": self.image_coordinate_space,
            "raw_image_xy": (
                None if self.raw_image_xy is None else [float(value) for value in self.raw_image_xy]
            ),
            "source": self.source,
        }


@dataclass(frozen=True)
class CourtLockVisualizationTrust:
    authority_state: str = "review_only"
    measurement_valid: bool = False
    usage: str = "visualization_only"
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority_state": self.authority_state,
            "measurement_valid": self.measurement_valid,
            "usage": self.usage,
            "verified": self.verified,
        }


@dataclass(frozen=True)
class CourtLockVisualizationArtifacts:
    court_calibration: CourtCalibration
    court_zones: CourtZones
    net_plane: NetPlane
    named_floor_correspondences: tuple[NamedFloorCorrespondence, ...]
    trust: CourtLockVisualizationTrust
    adapter_diagnostics: Mapping[str, Any]

    def artifact_payloads(self) -> dict[str, dict[str, Any]]:
        """Return the three existing downstream artifact payloads."""

        return {
            "court_calibration.json": self.court_calibration.model_dump(mode="json"),
            "court_zones.json": self.court_zones.model_dump(mode="json"),
            "net_plane.json": self.net_plane.model_dump(mode="json"),
        }

    def metadata_payload(self) -> dict[str, Any]:
        """Return non-authoritative adapter provenance beside legacy artifacts."""

        return {
            "schema_version": 1,
            "artifact_type": "racketsport_court_lock_visualization_adapter",
            "trust": self.trust.to_dict(),
            "named_floor_correspondences": [
                correspondence.to_dict() for correspondence in self.named_floor_correspondences
            ],
            "adapter_diagnostics": dict(self.adapter_diagnostics),
        }


def adapt_court_lock_for_visualization(
    lock: CourtLockArtifact | Mapping[str, Any],
    *,
    image_size: tuple[int, int],
    config: CourtLockVisualizationConfig | None = None,
) -> CourtLockVisualizationArtifacts:
    """Build legacy visualization artifacts from one accepted static lock.

    The adapter cannot grant measurement authority.  A moving, ambiguous,
    unsupported-distortion, high-residual, or under-supported lock is rejected
    rather than converted into a plausible-looking calibration.
    """

    resolved = lock if isinstance(lock, CourtLockArtifact) else CourtLockArtifact.from_dict(lock)
    settings = config or CourtLockVisualizationConfig()
    width, height = _validated_image_size(image_size)
    _validate_visualization_lock(resolved, settings=settings)

    world_points = [
        [float(value) for value in point.world_xyz_m]
        for point in FLOOR_KEYPOINTS
    ]
    raw_or_undistorted_points = np.asarray(
        project_planar_points(resolved.homography_image_from_court, world_points),
        dtype=np.float64,
    )
    raw_points: np.ndarray | None
    if resolved.coordinate_space == "pixels_raw_native":
        raw_points = raw_or_undistorted_points.copy()
        if abs(float(resolved.distortion.k1)) > 1.0e-12:
            undistorted_points, distortion_diagnostics = undistort_pixels_radial_k1(
                raw_points,
                resolved.camera_parameters.intrinsics,
                k1=float(resolved.distortion.k1),
                bounds=resolved.distortion.k1_bounds,
                strict=True,
            )
            if not distortion_diagnostics.all_converged or distortion_diagnostics.any_ambiguous:
                raise ValueError("court lock radial correction is ambiguous")
        else:
            undistorted_points = raw_points.copy()
    else:
        raw_points = None
        undistorted_points = raw_or_undistorted_points

    if resolved.coordinate_space == "pixels_undistorted_native" or abs(float(resolved.distortion.k1)) <= 1.0e-12:
        homography_undistorted = _normalized_homography(resolved.homography_image_from_court)
    else:
        homography_undistorted = np.asarray(
            homography_from_planar_points(world_points, undistorted_points),
            dtype=np.float64,
        )
    refitted_points = np.asarray(
        project_planar_points(homography_undistorted, world_points),
        dtype=np.float64,
    )
    refit_residuals = np.linalg.norm(refitted_points - undistorted_points, axis=1)
    refit_median = float(np.median(refit_residuals))
    refit_p95 = float(np.percentile(refit_residuals, 95))
    if refit_p95 > settings.max_undistortion_refit_p95_px:
        raise ValueError(
            "radial correction cannot be represented by one visualization homography: "
            f"p95={refit_p95:.6g}px exceeds {settings.max_undistortion_refit_p95_px:.6g}px"
        )

    actual_intrinsics = CameraIntrinsics(
        fx=float(resolved.camera_parameters.intrinsics.fx),
        fy=float(resolved.camera_parameters.intrinsics.fy),
        cx=float(resolved.camera_parameters.intrinsics.cx),
        cy=float(resolved.camera_parameters.intrinsics.cy),
        dist=(
            []
            if abs(float(resolved.distortion.k1)) <= 1.0e-12
            else [float(resolved.distortion.k1), 0.0, 0.0, 0.0, 0.0]
        ),
        source=resolved.camera_parameters.source,
    )
    undistorted_pose_intrinsics = CameraIntrinsics(
        fx=float(actual_intrinsics.fx),
        fy=float(actual_intrinsics.fy),
        cx=float(actual_intrinsics.cx),
        cy=float(actual_intrinsics.cy),
        dist=[],
        source=f"{actual_intrinsics.source}:undistorted_pose_solve",
    )
    extrinsics, pose_source = _visualization_extrinsics(
        resolved,
        world_points=world_points,
        undistorted_image_points=refitted_points,
        intrinsics=undistorted_pose_intrinsics,
    )
    combined_median = max(float(resolved.residual_px.median_px), refit_median)
    combined_p95 = max(float(resolved.residual_px.p95_px), refit_p95, combined_median)
    trust_reasons = [
        "court_lock_visualization_only",
        "authority_state=review_only",
        "measurement_valid=false",
        "not_measurement_authority",
        f"court_lock_source={resolved.source}",
        f"pose_source={pose_source}",
    ]
    if resolved.coordinate_space == "pixels_raw_native" and abs(float(resolved.distortion.k1)) > 1.0e-12:
        trust_reasons.append("raw_lock_points_undistorted_before_homography_refit")
    calibration = CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=homography_undistorted.tolist(),
        intrinsics=actual_intrinsics,
        image_size=(width, height),
        extrinsics=extrinsics,
        reprojection_error_px=ReprojectionError(median=combined_median, p95=combined_p95),
        capture_quality=CaptureQuality(grade="warn", reasons=trust_reasons),
        image_pts=refitted_points.tolist(),
        world_pts=world_points,
        trust_band="preview",
        usage="visualization_only",
        authority_state="review_only",
        measurement_valid=False,
    )
    correspondences = tuple(
        NamedFloorCorrespondence(
            semantic_name=point.name,
            world_xyz_m=tuple(float(value) for value in point.world_xyz_m),
            image_xy=(float(refitted_points[index, 0]), float(refitted_points[index, 1])),
            image_coordinate_space="pixels_undistorted_native",
            raw_image_xy=(
                None
                if raw_points is None
                else (float(raw_points[index, 0]), float(raw_points[index, 1]))
            ),
        )
        for index, point in enumerate(FLOOR_KEYPOINTS)
    )
    return CourtLockVisualizationArtifacts(
        court_calibration=calibration,
        court_zones=build_court_zones("pickleball"),
        net_plane=build_net_plane("pickleball"),
        named_floor_correspondences=correspondences,
        trust=CourtLockVisualizationTrust(),
        adapter_diagnostics={
            "source_lock_coordinate_space": resolved.coordinate_space,
            "output_homography_coordinate_space": "pixels_undistorted_native",
            "floor_correspondence_count": len(correspondences),
            "undistortion_refit_median_px": refit_median,
            "undistortion_refit_p95_px": refit_p95,
            "pose_source": pose_source,
            "supporting_frame_indices": list(resolved.evidence.selected_frame_indices),
        },
    )


def _validate_visualization_lock(
    lock: CourtLockArtifact,
    *,
    settings: CourtLockVisualizationConfig,
) -> None:
    if lock.static_motion.status != "static" or not lock.static_motion.static_lock_usable:
        raise ValueError(
            "visualization adapter requires a confirmed static court lock; "
            f"got {lock.static_motion.status}"
        )
    if len(lock.evidence.selected_frame_indices) < settings.min_supporting_frames:
        raise ValueError(
            "visualization adapter received too few supporting frames: "
            f"{len(lock.evidence.selected_frame_indices)} < {settings.min_supporting_frames}"
        )
    if float(lock.residual_px.p95_px) > settings.max_lock_p95_px:
        raise ValueError(
            f"court lock p95 residual {lock.residual_px.p95_px:.6g}px exceeds visualization bound "
            f"{settings.max_lock_p95_px:.6g}px"
        )
    k1_standard_deviation = math.sqrt(max(0.0, float(lock.distortion.k1_variance)))
    if k1_standard_deviation > settings.max_k1_standard_deviation:
        raise ValueError(
            f"court lock k1 uncertainty {k1_standard_deviation:.6g} exceeds visualization bound "
            f"{settings.max_k1_standard_deviation:.6g}"
        )


def _visualization_extrinsics(
    lock: CourtLockArtifact,
    *,
    world_points: Sequence[Sequence[float]],
    undistorted_image_points: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[CourtExtrinsics, str]:
    rotation = lock.camera_parameters.rotation_world_to_camera
    translation = lock.camera_parameters.translation_world_to_camera_m
    if rotation is None or translation is None:
        return (
            solve_camera_pose(world_points, undistorted_image_points, intrinsics),
            "solve_camera_pose_from_named_floor_correspondences",
        )
    rotation_matrix = np.asarray(rotation, dtype=np.float64)
    translation_vector = np.asarray(translation, dtype=np.float64)
    camera_center_world = -(rotation_matrix.T @ translation_vector)
    camera_height = abs(float(camera_center_world[2]))
    if camera_height <= 1.0e-6:
        raise ValueError("court lock camera pose has zero camera height")
    return (
        CourtExtrinsics(
            R=rotation_matrix.tolist(),
            t=translation_vector.tolist(),
            camera_height_m=camera_height,
        ),
        "court_lock_camera_pose",
    )


def _validated_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    if len(image_size) != 2:
        raise ValueError("image_size must contain width and height")
    width, height = image_size
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or int(width) <= 0
        or int(height) <= 0
    ):
        raise ValueError("image_size width and height must be positive integers")
    return int(width), int(height)


def _normalized_homography(values: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("court lock homography must be a finite 3x3 matrix")
    if abs(float(matrix[2, 2])) <= 1.0e-12:
        raise ValueError("court lock homography h22 must be nonzero")
    matrix = matrix / float(matrix[2, 2])
    if abs(float(np.linalg.det(matrix))) <= 1.0e-12:
        raise ValueError("court lock homography must be nonsingular")
    return matrix


__all__ = [
    "CourtLockVisualizationArtifacts",
    "CourtLockVisualizationConfig",
    "CourtLockVisualizationTrust",
    "FLOOR_KEYPOINTS",
    "NamedFloorCorrespondence",
    "adapt_court_lock_for_visualization",
]
