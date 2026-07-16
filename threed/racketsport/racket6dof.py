"""CPU-only racket and paddle 6DoF pose primitives.

This module contains deterministic validation, planar paddle-corner PnP/IPPE,
projection, and motion-plausibility helpers. It does not run a detector, SAM2,
GigaPose/FoundPose, hand association, or a real UKF.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from .coordinates import invert_extrinsics, translation_to_metres


INCH_TO_CM = 2.54


@dataclass(frozen=True)
class PaddleDimensions:
    """Validated paddle face dimensions in inches."""

    length_in: float
    width_in: float

    def __post_init__(self) -> None:
        length_in = _require_finite_float(self.length_in, "paddle_dims_in.length")
        width_in = _require_finite_float(self.width_in, "paddle_dims_in.width")
        if length_in <= 0.0 or width_in <= 0.0:
            raise ValueError("paddle_dims_in values must be positive")
        object.__setattr__(self, "length_in", length_in)
        object.__setattr__(self, "width_in", width_in)

    @property
    def length_cm(self) -> float:
        return self.length_in * INCH_TO_CM

    @property
    def width_cm(self) -> float:
        return self.width_in * INCH_TO_CM


@dataclass(frozen=True)
class SE3PoseConfidence:
    """A minimal SE3 pose plus confidence placeholder for a future UKF stage."""

    R: Sequence[Sequence[float]]
    t: Sequence[float]
    confidence: float
    source: str = "ukf_placeholder"

    def __post_init__(self) -> None:
        object.__setattr__(self, "R", _validate_rotation_matrix(self.R))
        object.__setattr__(self, "t", _validate_vector(self.t, "t", length=3))
        confidence = _require_finite_float(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not self.source:
            raise ValueError("source must be non-empty")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class PlanarPaddlePoseEstimate:
    """Planar paddle PnP result with quality diagnostics."""

    pose: SE3PoseConfidence
    reprojection_error_px: float
    candidate_reprojection_errors_px: tuple[float, ...]
    ambiguity_margin_px: float | None
    ambiguous: bool
    alt_pose: SE3PoseConfidence | None = None

    def __post_init__(self) -> None:
        reprojection = _require_finite_float(self.reprojection_error_px, "reprojection_error_px")
        if reprojection < 0.0:
            raise ValueError("reprojection_error_px must be non-negative")
        errors = tuple(
            _require_finite_float(error, f"candidate_reprojection_errors_px/{index}")
            for index, error in enumerate(self.candidate_reprojection_errors_px)
        )
        if not errors:
            raise ValueError("candidate_reprojection_errors_px must be non-empty")
        if any(error < 0.0 for error in errors):
            raise ValueError("candidate_reprojection_errors_px values must be non-negative")
        if self.ambiguity_margin_px is not None:
            margin = _require_finite_float(self.ambiguity_margin_px, "ambiguity_margin_px")
            if margin < 0.0:
                raise ValueError("ambiguity_margin_px must be non-negative")
            object.__setattr__(self, "ambiguity_margin_px", margin)
        object.__setattr__(self, "reprojection_error_px", reprojection)
        object.__setattr__(self, "candidate_reprojection_errors_px", errors)

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_reprojection_errors_px)


@dataclass(frozen=True)
class TimedRacketPose:
    """A timestamped racket pose sample."""

    t: float
    pose: SE3PoseConfidence

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", _require_finite_float(self.t, "t"))


@dataclass(frozen=True)
class RacketMotionReport:
    """Frame-to-frame physical plausibility summary for racket SE(3) samples."""

    sample_count: int
    translation_clamp_count: int
    rotation_clamp_count: int
    max_translation_speed_per_s: float
    max_angular_speed_deg_s: float


@dataclass(frozen=True)
class ReboundConsistency:
    """Ball/racket normal-velocity consistency around a contact candidate."""

    consistent: bool
    normal_speed_before: float
    normal_speed_after: float
    notes: tuple[str, ...]


def validate_paddle_dimensions(paddle_dims_in: Mapping[str, float] | PaddleDimensions) -> PaddleDimensions:
    """Return normalized paddle dimensions from length/width or h/w inch keys."""

    if isinstance(paddle_dims_in, PaddleDimensions):
        return paddle_dims_in

    has_named_dims = "length" in paddle_dims_in and "width" in paddle_dims_in
    has_short_dims = "h" in paddle_dims_in and "w" in paddle_dims_in
    if not has_named_dims and not has_short_dims:
        raise ValueError("paddle_dims_in must include length/width or h/w")

    if has_named_dims:
        length_in = paddle_dims_in["length"]
        width_in = paddle_dims_in["width"]
    else:
        length_in = paddle_dims_in["h"]
        width_in = paddle_dims_in["w"]

    return PaddleDimensions(length_in=length_in, width_in=width_in)


def normalize_face_normal(face_normal: Sequence[float]) -> tuple[float, float, float]:
    """Return a unit face-normal vector from a 3-vector."""

    x, y, z = _validate_vector(face_normal, "face_normal", length=3)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("face_normal must be non-zero")
    return (x / norm, y / norm, z / norm)


def validate_contact_point_face_cm(
    contact_point_face_cm: Sequence[float],
    paddle_dims_in: Mapping[str, float] | PaddleDimensions,
) -> tuple[float, float]:
    """Validate a face-local contact point against paddle dimensions.

    Coordinates are centered on the paddle face: x spans paddle width and y
    spans paddle length, both measured in centimeters.
    """

    x_cm, y_cm = _validate_vector(contact_point_face_cm, "contact_point_face_cm", length=2)
    dims = validate_paddle_dimensions(paddle_dims_in)
    half_width_cm = dims.width_cm / 2.0
    half_length_cm = dims.length_cm / 2.0
    epsilon = 1e-9

    if abs(x_cm) > half_width_cm + epsilon:
        raise ValueError("contact_point_face_cm x coordinate exceeds paddle width")
    if abs(y_cm) > half_length_cm + epsilon:
        raise ValueError("contact_point_face_cm y coordinate exceeds paddle length")
    return (x_cm, y_cm)


def paddle_face_corners_object_cm(
    paddle_dims_in: Mapping[str, float] | PaddleDimensions,
) -> tuple[tuple[float, float, float], ...]:
    """Return planar paddle-face corner coordinates in object-local centimeters.

    The order is top-left, top-right, bottom-right, bottom-left when looking at
    the paddle face. The local x-axis spans width, y-axis spans length, and
    z=0 is the paddle face plane.
    """

    dims = validate_paddle_dimensions(paddle_dims_in)
    half_width = dims.width_cm / 2.0
    half_length = dims.length_cm / 2.0
    return (
        (-half_width, half_length, 0.0),
        (half_width, half_length, 0.0),
        (half_width, -half_length, 0.0),
        (-half_width, -half_length, 0.0),
    )


def estimate_planar_paddle_pose(
    image_points_px: Sequence[Sequence[float]],
    camera_matrix: Sequence[Sequence[float]],
    paddle_dims_in: Mapping[str, float] | PaddleDimensions,
    *,
    dist_coeffs: Sequence[float] | None = None,
) -> SE3PoseConfidence:
    """Estimate camera-space paddle SE(3) from four face-corner pixels.

    This is the deterministic geometry step behind the planned detector/SAM2/
    keypoint model: once a candidate produces clean face corners, IPPE solves
    the planar 6DoF ambiguity and the lowest-reprojection solution is retained.
    Translation is in centimeters because the object points are in centimeters.
    """

    return estimate_planar_paddle_pose_with_diagnostics(
        image_points_px,
        camera_matrix,
        paddle_dims_in,
        dist_coeffs=dist_coeffs,
    ).pose


def estimate_planar_paddle_pose_with_diagnostics(
    image_points_px: Sequence[Sequence[float]],
    camera_matrix: Sequence[Sequence[float]],
    paddle_dims_in: Mapping[str, float] | PaddleDimensions,
    *,
    dist_coeffs: Sequence[float] | None = None,
    ambiguity_margin_threshold_px: float = 1.0,
) -> PlanarPaddlePoseEstimate:
    """Estimate planar paddle pose and expose reprojection/ambiguity diagnostics."""

    threshold = _require_finite_float(ambiguity_margin_threshold_px, "ambiguity_margin_threshold_px")
    if threshold < 0.0:
        raise ValueError("ambiguity_margin_threshold_px must be non-negative")
    cv2, np = _cv2_np()
    image_points = _array2(image_points_px, "image_points_px", rows=4, cols=2, np=np)
    object_points = np.asarray(paddle_face_corners_object_cm(paddle_dims_in), dtype=np.float64)
    camera = _array2(camera_matrix, "camera_matrix", rows=3, cols=3, np=np)
    dist = np.asarray(dist_coeffs if dist_coeffs is not None else [], dtype=np.float64)

    flags = getattr(cv2, "SOLVEPNP_IPPE", None)
    candidates: list[tuple[float, object, object]] = []
    if flags is not None:
        ok, rvecs, tvecs, _errors = cv2.solvePnPGeneric(object_points, image_points, camera, dist, flags=flags)
        if ok:
            candidates.extend((0.0, rvec, tvec) for rvec, tvec in zip(rvecs, tvecs))

    if not candidates:
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError("paddle pose solvePnP failed")
        candidates.append((0.0, rvec, tvec))

    scored: list[tuple[float, object, object]] = []
    for _score, rvec, tvec in candidates:
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera, dist)
        projected = projected.reshape(-1, 2)
        reprojection_errors = np.linalg.norm(projected - image_points, axis=1)
        scored.append((float(np.mean(reprojection_errors)), rvec, tvec))
    scored = sorted(scored, key=lambda item: item[0])
    reprojection_error_px, best_rvec, best_tvec = scored[0]
    rotation, _ = cv2.Rodrigues(best_rvec)
    confidence = max(0.0, min(1.0, 1.0 / (1.0 + reprojection_error_px / 3.0)))
    candidate_errors = tuple(score for score, _rvec, _tvec in scored)
    ambiguity_margin = candidate_errors[1] - candidate_errors[0] if len(candidate_errors) > 1 else None
    pose = SE3PoseConfidence(
        R=rotation.tolist(),
        t=np.asarray(best_tvec, dtype=np.float64).reshape(3).tolist(),
        confidence=confidence,
        source="pnp_ippe",
    )
    alt_pose = None
    if len(scored) > 1:
        alt_error_px, alt_rvec, alt_tvec = scored[1]
        alt_rotation, _ = cv2.Rodrigues(alt_rvec)
        alt_translation = np.asarray(alt_tvec, dtype=np.float64).reshape(3).tolist()
        if alt_translation[2] <= 0.0:
            raise ValueError("second IPPE paddle pose is degenerate: non-positive camera depth")
        alt_pose = SE3PoseConfidence(
            R=alt_rotation.tolist(),
            t=alt_translation,
            confidence=max(0.0, min(1.0, 1.0 / (1.0 + alt_error_px / 3.0))),
            source="pnp_ippe_alt",
        )
    return PlanarPaddlePoseEstimate(
        pose=pose,
        reprojection_error_px=reprojection_error_px,
        candidate_reprojection_errors_px=candidate_errors,
        ambiguity_margin_px=ambiguity_margin,
        ambiguous=ambiguity_margin is not None and ambiguity_margin <= threshold,
        alt_pose=alt_pose,
    )


def project_paddle_corners(
    pose: SE3PoseConfidence,
    camera_matrix: Sequence[Sequence[float]],
    paddle_dims_in: Mapping[str, float] | PaddleDimensions,
    *,
    dist_coeffs: Sequence[float] | None = None,
) -> tuple[tuple[float, float], ...]:
    """Project a racket pose back to paddle-face corner pixels."""

    cv2, np = _cv2_np()
    object_points = np.asarray(paddle_face_corners_object_cm(paddle_dims_in), dtype=np.float64)
    camera = _array2(camera_matrix, "camera_matrix", rows=3, cols=3, np=np)
    dist = np.asarray(dist_coeffs if dist_coeffs is not None else [], dtype=np.float64)
    rotation = np.asarray(pose.R, dtype=np.float64)
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = np.asarray(pose.t, dtype=np.float64).reshape(3, 1)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera, dist)
    return tuple((float(x), float(y)) for x, y in projected.reshape(-1, 2))


def pose_face_normal(pose: SE3PoseConfidence) -> tuple[float, float, float]:
    """Return the camera/world-space normal of the paddle face."""

    return normalize_face_normal((pose.R[0][2], pose.R[1][2], pose.R[2][2]))


def camera_paddle_pose_to_court_world(
    pose: SE3PoseConfidence,
    calibration: object,
    *,
    input_translation_unit: str = "cm",
) -> SE3PoseConfidence:
    """Convert a camera-space paddle pose into the court_Z0 world frame.

    `estimate_planar_paddle_pose` returns translation in the same units as the
    paddle object model, currently centimeters. Court calibration extrinsics use
    meters, so this function scales translation before applying the inverse
    world-to-camera extrinsic transform.
    """

    extrinsics = getattr(calibration, "extrinsics", None)
    if extrinsics is None:
        raise ValueError("calibration must include extrinsics")
    world_to_camera_R = _validate_rotation_matrix(getattr(extrinsics, "R", None))
    world_to_camera_t = _validate_vector(getattr(extrinsics, "t", None), "calibration.extrinsics.t", length=3)

    camera_to_world_R_array, _camera_center_world = invert_extrinsics(
        world_to_camera_R,
        world_to_camera_t,
    )
    camera_to_world_R = tuple(
        tuple(float(value) for value in row)
        for row in camera_to_world_R_array.tolist()
    )
    pose_t_camera_m = translation_to_metres(
        pose.t,
        input_unit=input_translation_unit,
    )
    pose_t_minus_camera_origin = tuple(
        pose_t_camera_m[index] - world_to_camera_t[index]
        for index in range(3)
    )
    world_R = _matmul3(camera_to_world_R, pose.R)
    world_t = _matvec3(camera_to_world_R, pose_t_minus_camera_origin)
    return SE3PoseConfidence(
        R=world_R,
        t=world_t,
        confidence=pose.confidence,
        source=f"{pose.source}:court_Z0",
    )


def smooth_racket_pose_samples(
    samples: Sequence[TimedRacketPose | tuple[float, SE3PoseConfidence]],
    *,
    max_translation_speed_per_s: float,
    max_angular_speed_deg_s: float,
) -> tuple[list[TimedRacketPose], RacketMotionReport]:
    """Clamp implausible frame-to-frame racket jumps in SE(3).

    This is not a replacement for the planned UKF. It is the deterministic
    physical guardrail: a sample can move only as far as the configured
    translational and angular velocity limits allow for its frame interval.
    """

    if max_translation_speed_per_s <= 0.0:
        raise ValueError("max_translation_speed_per_s must be positive")
    if max_angular_speed_deg_s <= 0.0:
        raise ValueError("max_angular_speed_deg_s must be positive")

    np = _np()
    Rotation = _rotation_class()
    timed = [_coerce_timed_sample(sample) for sample in samples]
    if not timed:
        return [], RacketMotionReport(0, 0, 0, 0.0, 0.0)

    smoothed = [timed[0]]
    translation_clamps = 0
    rotation_clamps = 0
    max_translation_speed = 0.0
    max_angular_speed = 0.0

    for raw in timed[1:]:
        previous = smoothed[-1]
        dt = raw.t - previous.t
        if dt <= 0.0:
            raise ValueError("racket pose sample times must be strictly increasing")

        previous_t = np.asarray(previous.pose.t, dtype=np.float64)
        raw_t = np.asarray(raw.pose.t, dtype=np.float64)
        delta_t = raw_t - previous_t
        distance = float(np.linalg.norm(delta_t))
        max_translation_speed = max(max_translation_speed, distance / dt)
        max_distance = max_translation_speed_per_s * dt
        if distance > max_distance:
            raw_t = previous_t + delta_t * (max_distance / distance)
            translation_clamps += 1

        previous_r = Rotation.from_matrix(np.asarray(previous.pose.R, dtype=np.float64))
        raw_r = Rotation.from_matrix(np.asarray(raw.pose.R, dtype=np.float64))
        relative = previous_r.inv() * raw_r
        relative_rotvec = relative.as_rotvec()
        angle_rad = float(np.linalg.norm(relative_rotvec))
        angle_deg = math.degrees(angle_rad)
        max_angular_speed = max(max_angular_speed, angle_deg / dt)
        max_angle_rad = math.radians(max_angular_speed_deg_s * dt)
        if angle_rad > max_angle_rad:
            raw_r = previous_r * Rotation.from_rotvec(relative_rotvec * (max_angle_rad / angle_rad))
            rotation_clamps += 1

        source = raw.pose.source
        if distance > max_distance or angle_rad > max_angle_rad:
            source = f"{source}:motion_clamped"
        smoothed.append(
            TimedRacketPose(
                t=raw.t,
                pose=SE3PoseConfidence(
                    R=raw_r.as_matrix().tolist(),
                    t=raw_t.tolist(),
                    confidence=raw.pose.confidence,
                    source=source,
                ),
            )
        )

    return smoothed, RacketMotionReport(
        sample_count=len(smoothed),
        translation_clamp_count=translation_clamps,
        rotation_clamp_count=rotation_clamps,
        max_translation_speed_per_s=max_translation_speed,
        max_angular_speed_deg_s=max_angular_speed,
    )


def rebound_consistency(
    *,
    incoming_velocity: Sequence[float],
    outgoing_velocity: Sequence[float],
    face_normal: Sequence[float],
    min_normal_speed: float = 0.1,
) -> ReboundConsistency:
    """Check whether ball velocity flips across the paddle face normal."""

    if min_normal_speed < 0.0:
        raise ValueError("min_normal_speed must be non-negative")
    normal = normalize_face_normal(face_normal)
    before = _dot(_validate_vector(incoming_velocity, "incoming_velocity", length=3), normal)
    after = _dot(_validate_vector(outgoing_velocity, "outgoing_velocity", length=3), normal)
    notes: list[str] = []
    if abs(before) < min_normal_speed or abs(after) < min_normal_speed:
        notes.append("normal_component_too_small")
    if before * after >= 0.0:
        notes.append("normal_component_did_not_flip")
    return ReboundConsistency(
        consistent=not notes,
        normal_speed_before=before,
        normal_speed_after=after,
        notes=tuple(notes),
    )


def _require_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_vector(values: Sequence[float], name: str, *, length: int) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a {length}-vector")
    try:
        vector = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a {length}-vector") from exc
    if len(vector) != length:
        raise ValueError(f"{name} must be a {length}-vector")
    return tuple(_require_finite_float(value, f"{name}/{index}") for index, value in enumerate(vector))


def _validate_rotation_matrix(rows: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    if isinstance(rows, (str, bytes)):
        raise ValueError("R must be a 3x3 matrix")
    try:
        matrix = tuple(tuple(row) for row in rows)
    except TypeError as exc:
        raise ValueError("R must be a 3x3 matrix") from exc
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("R must be a 3x3 matrix")
    return tuple(
        tuple(_require_finite_float(value, f"R/{row_index}/{col_index}") for col_index, value in enumerate(row))
        for row_index, row in enumerate(matrix)
    )


def _translation_scale_to_m(unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in {"m", "meter", "meters"}:
        return 1.0
    if normalized in {"cm", "centimeter", "centimeters"}:
        return 0.01
    raise ValueError("input_translation_unit must be 'cm' or 'm'")


def _transpose3(matrix: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(matrix[row][col]) for row in range(3)) for col in range(3))


def _matmul3(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(
            sum(float(left[row][inner]) * float(right[inner][col]) for inner in range(3))
            for col in range(3)
        )
        for row in range(3)
    )


def _matvec3(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> tuple[float, float, float]:
    return tuple(
        sum(float(matrix[row][col]) * float(vector[col]) for col in range(3))
        for row in range(3)
    )


def _array2(values: Sequence[Sequence[float]], name: str, *, rows: int, cols: int, np: object) -> object:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (rows, cols):
        raise ValueError(f"{name} must be a {rows}x{cols} matrix")
    return array


def _coerce_timed_sample(sample: TimedRacketPose | tuple[float, SE3PoseConfidence]) -> TimedRacketPose:
    if isinstance(sample, TimedRacketPose):
        return sample
    t, pose = sample
    if not isinstance(pose, SE3PoseConfidence):
        raise ValueError("racket pose sample must contain an SE3PoseConfidence")
    return TimedRacketPose(float(t), pose)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def _cv2_np() -> tuple[object, object]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for planar paddle PnP") from exc
    return cv2, _np()


def _np() -> object:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("numpy is required for racket 6DoF geometry") from exc
    return np


def _rotation_class() -> object:
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise RuntimeError("scipy is required for racket pose smoothing") from exc
    return Rotation


__all__ = [
    "INCH_TO_CM",
    "PaddleDimensions",
    "PlanarPaddlePoseEstimate",
    "RacketMotionReport",
    "ReboundConsistency",
    "SE3PoseConfidence",
    "TimedRacketPose",
    "camera_paddle_pose_to_court_world",
    "estimate_planar_paddle_pose",
    "estimate_planar_paddle_pose_with_diagnostics",
    "normalize_face_normal",
    "paddle_face_corners_object_cm",
    "pose_face_normal",
    "project_paddle_corners",
    "rebound_consistency",
    "smooth_racket_pose_samples",
    "validate_contact_point_face_cm",
    "validate_paddle_dimensions",
]
