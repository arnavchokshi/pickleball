from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("value must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("value must be a finite number")
    return number


FiniteFloat = Annotated[float, BeforeValidator(_finite_number)]
NonNegativeFiniteFloat = Annotated[float, BeforeValidator(_finite_number), Field(ge=0.0)]
Vector2 = Annotated[list[FiniteFloat], Field(min_length=2, max_length=2)]
Vector3 = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
MatrixRow3 = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
Matrix3 = Annotated[list[MatrixRow3], Field(min_length=3, max_length=3)]
MatrixRow2 = Annotated[list[FiniteFloat], Field(min_length=2, max_length=2)]
Matrix2 = Annotated[list[MatrixRow2], Field(min_length=2, max_length=2)]
MatrixRow4 = Annotated[list[FiniteFloat], Field(min_length=4, max_length=4)]
Matrix4 = Annotated[list[MatrixRow4], Field(min_length=4, max_length=4)]
PICKLEBALL_COURT_KEYPOINT_NAMES: tuple[str, ...] = (
    "near_left_corner",
    "near_baseline_center",
    "near_right_corner",
    "far_right_corner",
    "far_baseline_center",
    "far_left_corner",
    "near_nvz_left",
    "near_nvz_center",
    "near_nvz_right",
    "net_left_sideline",
    "net_center",
    "net_right_sideline",
    "far_nvz_left",
    "far_nvz_center",
    "far_nvz_right",
)
BallVisibilityLevel = Literal["clear", "partial", "full", "out_of_frame"]
BallInputPreprocessingMode = Literal["official", "harness_v0"]
BALL_VISIBILITY_LEVELS: tuple[BallVisibilityLevel, ...] = ("clear", "partial", "full", "out_of_frame")
BALL_VISIBILITY_WBCE_WEIGHTS: dict[BallVisibilityLevel, int] = {
    "clear": 1,
    "partial": 2,
    "full": 3,
    "out_of_frame": 3,
}
LEGACY_BALL_VISIBILITY_MAPPING = (
    "Absent visibility_level means legacy-only visibility. A legacy visible=True ball frame or CVAT ball box is "
    "legacy_visible: it has a localizable ball but does not prove clear versus partial occlusion. A legacy "
    "visible=False frame or missing CVAT ball box is legacy_hidden: it does not distinguish full occlusion from "
    "out-of-frame. Legacy rows keep visibility_level=null and wbce_weight=null; the 1/2/3/3 WBCE weights apply "
    "only to explicit clear/partial/full/out_of_frame labels."
)


class StrictArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]


class CaptureQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grade: Literal["good", "warn", "poor"]
    reasons: list[str] = Field(default_factory=list)


class CameraIntrinsics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fx: FiniteFloat = Field(gt=0.0)
    fy: FiniteFloat = Field(gt=0.0)
    cx: FiniteFloat
    cy: FiniteFloat
    dist: list[FiniteFloat] = Field(default_factory=list)
    source: str


class RigidPose(BaseModel):
    model_config = ConfigDict(extra="forbid")

    R: Matrix3
    t: Vector3


class Plane(BaseModel):
    model_config = ConfigDict(extra="forbid")

    point: Vector3
    normal: Vector3


class LockedCapture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exposure_s: float
    iso: float
    focus: float
    wb_locked: bool


class ARTrackingSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Literal["normal", "limited", "unavailable"]
    quality: Literal["good", "limited", "unavailable"]
    reason: str | None = None


class ARKitFrameSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_pts_s: FiniteFloat
    arkit_timestamp_s: FiniteFloat | None = None
    camera_pose: RigidPose | None = None
    intrinsics: CameraIntrinsics | None = None
    tracking: ARTrackingSnapshot | None = None
    gravity: Vector3 | None = None
    provenance: Literal["arkit", "coremotion_only"]
    unavailable_reason: str | None = None


class ARKitSetupPassSidecar(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["available", "unavailable"]
    provenance: str
    intrinsics: CameraIntrinsics | None = None
    camera_pose: RigidPose | None = None
    court_plane: Plane | None = None
    gravity: Vector3 | None = None
    tracking_state: Literal["normal", "limited", "unavailable"]
    timestamp_s: FiniteFloat | None = None
    duration_s: FiniteFloat | None = None
    unavailable_reason: str | None = None


class CapturePolicyRequestedState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fps: int
    resolution: tuple[int, int]
    format: Literal["hevc", "prores422lt"]
    orientation: Literal["portrait", "landscape"]
    electronic_stabilization_enabled: bool
    exposure_locked: bool
    focus_locked: bool
    white_balance_locked: bool


class CapturePolicyAchievedState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fps: int | None = None
    resolution: tuple[int, int] | None = None
    format: Literal["hevc", "prores422lt"] | None = None
    orientation: Literal["portrait", "landscape"] | None = None
    electronic_stabilization_enabled: bool | None = None
    exposure_locked: bool | None = None
    focus_locked: bool | None = None
    white_balance_locked: bool | None = None


class CapturePolicyEnforcementReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested: CapturePolicyRequestedState
    achieved: CapturePolicyAchievedState | None = None
    violations: list[str] = Field(default_factory=list)


class ProfileCaptureStepRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["empty_court_clip", "calibration_grid_sweep", "paddle_orbit", "player_height_entry", "ball_pick"]
    status: Literal["pending", "complete"]
    artifact_ref: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class ProfileCapturePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[ProfileCaptureStepRecord]


class ReferenceCrop(BaseModel):
    """Crop rectangle in the native-intrinsics pixel raster.

    Absence means that the sidecar intrinsics already reference the encoded
    ``resolution`` raster.  The explicit ``_px`` suffix prevents normalized or
    processed-raster rectangles from being accepted by implication.
    """

    model_config = ConfigDict(extra="forbid")

    x_px: FiniteFloat = Field(ge=0.0)
    y_px: FiniteFloat = Field(ge=0.0)
    width_px: FiniteFloat = Field(gt=0.0)
    height_px: FiniteFloat = Field(gt=0.0)


class CaptureRollingShutter(BaseModel):
    """Optional device-declared sensor readout; absence means unavailable."""

    model_config = ConfigDict(extra="forbid")

    frame_readout_s: FiniteFloat = Field(gt=0.0)
    direction: Literal["top_to_bottom", "bottom_to_top"]


class CaptureSidecar(StrictArtifact):
    provenance: Literal["live_recording", "camera_roll_import"] | None = None
    device_tier: Literal["A_lidar", "B_standard", "fallback"]
    device_model: str
    fps: int
    format: Literal["hevc", "prores422lt"]
    resolution: tuple[int, int]
    orientation: Literal["portrait", "landscape"]
    capture_device_orientation: Literal["portrait", "portraitUpsideDown", "landscapeRight", "landscapeLeft"] | None = None
    video_rotation_angle_degrees: int | None = None
    recording_started_at: str | None = None
    recording_duration_s: float | None = None
    camera_position: str | None = None
    camera_lens: str | None = None
    locked: LockedCapture | None = None
    intrinsics: CameraIntrinsics | None = None
    reference_crop: ReferenceCrop | None = None
    rolling_shutter: CaptureRollingShutter | None = None
    arkit_camera_pose: RigidPose | None = None
    court_plane: Plane | None = None
    setup_pass: ARKitSetupPassSidecar | None = None
    manual_court_taps: list[Vector2] = Field(default_factory=list)
    gravity: Vector3 | None = None
    arkit_frame_samples: list[ARKitFrameSample] = Field(default_factory=list)
    lidar_depth_refs: list[str] = Field(default_factory=list)
    ondevice_pose_track: str | None = None
    unavailable_sensor_reasons: dict[str, str] = Field(default_factory=dict)
    policy_enforcement: CapturePolicyEnforcementReport | None = None
    profile_capture: ProfileCapturePayload | None = None
    capture_quality: CaptureQuality
    hdr_enabled: bool | None = None
    video_stabilization_enabled: bool | None = None
    exposure_locked: bool | None = None
    focus_locked: bool | None = None
    tripod_height_m: FiniteFloat | None = Field(default=None, ge=0.0)
    full_court_visible: bool | None = None
    court_lock_passed: bool | None = None
    ball_high_contrast: bool | None = None
    audio_recorded: bool | None = None

    @field_validator("video_rotation_angle_degrees")
    @classmethod
    def _rotation_must_be_cardinal(cls, value: int | None) -> int | None:
        if value is not None and value not in {0, 90, 180, 270}:
            raise ValueError("video_rotation_angle_degrees must be one of 0, 90, 180, 270")
        return value

    @field_validator("recording_duration_s")
    @classmethod
    def _recording_duration_must_be_nonnegative(cls, value: float | None) -> float | None:
        if value is not None and value < 0.0:
            raise ValueError("recording_duration_s must be nonnegative")
        return value


class ReprojectionError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    median: FiniteFloat = Field(ge=0.0)
    p95: FiniteFloat = Field(ge=0.0)


class CourtExtrinsics(RigidPose):
    camera_height_m: FiniteFloat = Field(gt=0.0)


class CourtGsdSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    court_xy: Vector2
    gsd_m_per_px: FiniteFloat = Field(ge=0.0)
    sigma_p_m: FiniteFloat = Field(ge=0.0)


class CourtGsdModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["analytic_ray_plane"]
    plane_sigma_m: FiniteFloat = Field(ge=0.0)
    calibration_sigma_m: FiniteFloat = Field(ge=0.0)
    samples: list[CourtGsdSample] = Field(default_factory=list)


class CourtCalibrationCoordinateContract(BaseModel):
    """Additive typed declarations beside legacy CAL matrices."""

    model_config = ConfigDict(extra="forbid")

    camera_matrix_K: Matrix3
    camera_matrix_input_space: Literal["camera_m"]
    camera_matrix_output_space: Literal["pixels_undistorted_native"]
    extrinsics_convention: Literal["world_to_camera_opencv_column"]
    extrinsics_input_space: Literal["world_court_netcenter_z_up_m"]
    extrinsics_output_space: Literal["camera_m"]
    homography_convention: Literal["world_xy_to_image_column"]
    homography_input_space: Literal["world_xy_homography_m"]
    homography_output_space: Literal["pixels_raw_native", "pixels_undistorted_native"]
    homography_pixel_convention: Literal["raw_pixels", "undistorted_pixels"]


class CourtCalibrationProvenance(BaseModel):
    """Reproducibility identity required by preview external-CAL imports."""

    model_config = ConfigDict(extra="forbid")

    method: str = Field(min_length=1)
    inputs: list[str] = Field(min_length=1)
    code_identity: str = Field(min_length=1)

    @field_validator("method", "code_identity")
    @classmethod
    def _identity_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("calibration provenance identity must not be blank")
        return value

    @field_validator("inputs")
    @classmethod
    def _inputs_must_not_contain_blank_identities(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("calibration provenance inputs must not contain blank identities")
        return value


class CourtCalibration(StrictArtifact):
    sport: Literal["pickleball", "tennis"]
    coordinate_frame: Literal["court_netcenter_z_up_m"] | None = None
    T_world_court: Matrix4 | None = None
    homography: Matrix3
    intrinsics: CameraIntrinsics
    image_size: tuple[int, int] | None = None
    extrinsics: CourtExtrinsics
    reprojection_error_px: ReprojectionError
    per_keypoint_residual_px: list[FiniteFloat] | None = None
    metric_confidence: Literal["high", "med", "low"] | None = None
    gsd_model: CourtGsdModel | None = None
    capture_quality: CaptureQuality
    image_pts: list[Vector2] = Field(min_length=4)
    world_pts: list[Vector3] = Field(min_length=4)
    source: str | None = None
    solved_over_frames: list[int] | None = None
    coordinate_contract: CourtCalibrationCoordinateContract | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    provenance: CourtCalibrationProvenance | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    trust_band: Literal["preview"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    usage: Literal["visualization_only"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    authority_state: Literal["review_only"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    measurement_valid: Literal[False] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def _point_lists_must_be_paired(self) -> CourtCalibration:
        if len(self.image_pts) != len(self.world_pts):
            raise ValueError("image_pts and world_pts must have the same length")
        metric_fields = {
            "coordinate_frame": self.coordinate_frame,
            "T_world_court": self.T_world_court,
            "per_keypoint_residual_px": self.per_keypoint_residual_px,
            "metric_confidence": self.metric_confidence,
            "gsd_model": self.gsd_model,
            "source": self.source,
            "solved_over_frames": self.solved_over_frames,
        }
        if any(value is not None for value in metric_fields.values()):
            missing = [name for name, value in metric_fields.items() if value is None]
            if missing:
                raise ValueError(f"metric court_calibration fields must be complete; missing {', '.join(missing)}")
            assert self.per_keypoint_residual_px is not None
            if len(self.per_keypoint_residual_px) != len(PICKLEBALL_COURT_KEYPOINT_NAMES):
                raise ValueError("per_keypoint_residual_px must contain exactly 15 values")
            if any(value < 0.0 for value in self.per_keypoint_residual_px):
                raise ValueError("per_keypoint_residual_px must be non-negative")
            assert self.solved_over_frames is not None
            if any(frame < 0 for frame in self.solved_over_frames):
                raise ValueError("solved_over_frames must be non-negative")
        if self.coordinate_contract is not None:
            if self.coordinate_frame != "court_netcenter_z_up_m":
                raise ValueError("coordinate_contract requires coordinate_frame=court_netcenter_z_up_m")
            expected_k = [
                [float(self.intrinsics.fx), 0.0, float(self.intrinsics.cx)],
                [0.0, float(self.intrinsics.fy), float(self.intrinsics.cy)],
                [0.0, 0.0, 1.0],
            ]
            if self.coordinate_contract.camera_matrix_K != expected_k:
                raise ValueError("coordinate_contract.camera_matrix_K conflicts with intrinsics")
            expected_homography_output = {
                "raw_pixels": "pixels_raw_native",
                "undistorted_pixels": "pixels_undistorted_native",
            }[self.coordinate_contract.homography_pixel_convention]
            if self.coordinate_contract.homography_output_space != expected_homography_output:
                raise ValueError("coordinate_contract homography declarations conflict")
        trust_fields = {
            "usage": self.usage,
            "authority_state": self.authority_state,
            "measurement_valid": self.measurement_valid,
        }
        if any(value is not None for value in trust_fields.values()):
            missing_trust = [name for name, value in trust_fields.items() if value is None]
            if missing_trust:
                raise ValueError(
                    "court calibration trust fields must be complete; missing "
                    + ", ".join(missing_trust)
                )
            if self.measurement_valid is not False or self.authority_state != "review_only":
                raise ValueError(
                    "visualization_only court calibration must be review_only and measurement_valid=false"
                )
            if self.trust_band != "preview":
                raise ValueError("visualization_only court calibration requires trust_band=preview")
        return self


class CourtZones(StrictArtifact):
    zones: dict[str, list[Vector2]]


class NetPlane(StrictArtifact):
    plane: Plane
    endpoints: tuple[Vector3, Vector3]
    center_height_in: float
    post_height_in: float


class ResidualSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean: FiniteFloat = Field(ge=0.0)
    p95: FiniteFloat = Field(ge=0.0)


class CourtLineObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line_id: str
    image_segment: list[Vector2]
    confidence: float
    frame_indexes: list[int]
    residual_px: ResidualSummary
    visible_fraction: float
    source: str

    @field_validator("image_segment")
    @classmethod
    def _must_be_segment(cls, value: list[Vector2]) -> list[Vector2]:
        if len(value) != 2 or any(len(point) != 2 for point in value):
            raise ValueError("image_segment must contain exactly two 2D points")
        return value

    @field_validator("confidence", "visible_fraction")
    @classmethod
    def _must_be_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("value must be in [0, 1]")
        return value


class CourtKeypointObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    image_xy: Vector2
    confidence: float
    frame_indexes: list[int]
    source: str

    @field_validator("image_xy")
    @classmethod
    def _must_be_image_point(cls, value: Vector2) -> Vector2:
        if len(value) != 2:
            raise ValueError("image_xy must be a 2D point")
        return value

    @field_validator("confidence")
    @classmethod
    def _must_be_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        return value


class NetLineObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    net_id: str
    image_points: list[Vector2]
    confidence: float
    frame_indexes: list[int]
    residual_px: ResidualSummary
    source: str

    @field_validator("image_points")
    @classmethod
    def _must_be_top_net_triplet(cls, value: list[Vector2]) -> list[Vector2]:
        if len(value) != 3 or any(len(point) != 2 for point in value):
            raise ValueError("image_points must contain left, center, and right 2D points")
        return value

    @field_validator("confidence")
    @classmethod
    def _must_be_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        return value


class CourtLineEvidenceAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted_line_ids: list[str] = Field(default_factory=list)
    rejected_line_ids: list[str] = Field(default_factory=list)
    missing_required_line_ids: list[str] = Field(default_factory=list)
    missing_required_net_ids: list[str] = Field(default_factory=list)
    mean_residual_px: FiniteFloat = Field(ge=0.0)
    p95_residual_px: FiniteFloat = Field(ge=0.0)
    temporal_stability_px: FiniteFloat = Field(ge=0.0)
    auto_calibration_ready: bool
    reasons: list[str] = Field(default_factory=list)


class CourtLineEvidence(StrictArtifact):
    sport: Literal["pickleball", "tennis"]
    source: str
    line_observations: list[CourtLineObservation] = Field(default_factory=list)
    keypoint_observations: list[CourtKeypointObservation] = Field(default_factory=list)
    net_observations: list[NetLineObservation] = Field(default_factory=list)
    aggregate: CourtLineEvidenceAggregate

    @model_validator(mode="after")
    def _ready_aggregate_must_be_backed_by_observations(self) -> CourtLineEvidence:
        if not self.aggregate.auto_calibration_ready:
            return self
        observed_line_ids = {observation.line_id for observation in self.line_observations}
        missing_accepted = [
            line_id for line_id in self.aggregate.accepted_line_ids
            if line_id not in observed_line_ids
        ]
        if not self.aggregate.accepted_line_ids or missing_accepted:
            raise ValueError("ready court_line_evidence must include observations for accepted lines")
        if not self.net_observations:
            raise ValueError("ready court_line_evidence must include net observations")
        return self


class CourtKeypointAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    uv: Vector2
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    inlier_frames: list[int] = Field(default_factory=list)
    recovered: bool

    @field_validator("inlier_frames")
    @classmethod
    def _inlier_frames_must_be_nonnegative(cls, value: list[int]) -> list[int]:
        if any(frame < 0 for frame in value):
            raise ValueError("inlier_frames must be non-negative")
        return value


class CourtKeypoints(StrictArtifact):
    artifact_type: Literal["racketsport_court_keypoints"]
    frame_indexes: list[int] = Field(min_length=1)
    coordinate_space: Literal["undistorted_source_video_pixels"]
    keypoints: list[CourtKeypointAggregate]
    target_court_score: FiniteFloat = Field(ge=0.0, le=1.0)
    source: str
    not_gate_verified: bool

    @field_validator("frame_indexes")
    @classmethod
    def _frame_indexes_must_be_nonnegative(cls, value: list[int]) -> list[int]:
        if any(frame < 0 for frame in value):
            raise ValueError("frame_indexes must be non-negative")
        return value

    @field_validator("not_gate_verified")
    @classmethod
    def _must_be_not_gate_verified(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("court_keypoints is not gate verified")
        return value

    @model_validator(mode="after")
    def _must_match_canonical_keypoint_schema(self) -> "CourtKeypoints":
        names = [keypoint.name for keypoint in self.keypoints]
        if len(names) != len(set(names)):
            raise ValueError("court_keypoints must not contain duplicate keypoint names")
        if tuple(names) != PICKLEBALL_COURT_KEYPOINT_NAMES:
            raise ValueError("court_keypoints must contain exactly the 15 canonical pickleball keypoints in schema order")
        return self


class CourtLock(StrictArtifact):
    """Versioned static structured-court lock.

    The detailed invariants live beside the solver serializer so the runtime
    writer and public artifact validator cannot silently drift apart.
    """

    artifact_type: Literal["racketsport_court_lock"]
    coordinate_space: Literal["pixels_raw_native", "pixels_undistorted_native"]
    homography_image_from_court: Matrix3
    camera_parameters: dict[str, Any]
    distortion: dict[str, Any]
    transform_covariance: list[list[FiniteFloat]] | None
    source: Literal[
        "owner_metric_15pt_reviewed",
        "multi_frame_point_and_line",
        "clearest_frame_point_and_line",
        "dense_line_only",
        "previous_static_lock",
        "camera_profile_prior",
    ]
    evidence: dict[str, Any]
    static_motion: dict[str, Any]
    residual_px: dict[str, Any]
    score_components: dict[str, FiniteFloat]
    scorer_version: str = Field(min_length=1)
    calibration_version: str = Field(min_length=1)
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    measurement_valid: bool
    authority_state: Literal["best_effort", "review_only", "authoritative"]
    verified: bool

    @model_validator(mode="after")
    def _must_match_runtime_contract(self) -> "CourtLock":
        # Import lazily so the general schema registry does not make the court
        # geometry stack a startup dependency for unrelated artifacts.
        from threed.racketsport.court_static_lock import CourtLockArtifact

        CourtLockArtifact.from_dict(self.model_dump(mode="python"))
        return self


class TrackFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_idx: int | None = Field(default=None, ge=0)
    t: FiniteFloat = Field(ge=0.0)
    bbox: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    world_xy: Vector2
    conf: FiniteFloat = Field(ge=0.0, le=1.0)
    interpolated: bool = Field(
        default=False,
        strict=True,
    )

    @model_serializer(mode="wrap")
    def _serialize_interpolation_provenance(self, handler: Any) -> dict[str, Any]:
        payload = handler(self)
        if not self.interpolated:
            payload.pop("interpolated", None)
        return payload

    @field_validator("bbox")
    @classmethod
    def _bbox_must_have_positive_extent(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = value
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must be ordered as x1, y1, x2, y2")
        return value


class PlayerTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    side: str
    role: str
    side_original: str | None = None
    role_original: str | None = None
    side_source: str | None = None
    role_source: str | None = None
    frames: list[TrackFrame]


class TimeSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t0: float = Field(ge=0.0)
    t1: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _times_must_be_ordered(self) -> "TimeSpan":
        if self.t1 < self.t0:
            raise ValueError("t1 must be greater than or equal to t0")
        return self


class Tracks(StrictArtifact):
    fps: float = Field(gt=0.0)
    players: list[PlayerTrack]
    rally_spans: list[TimeSpan] = Field(default_factory=list)
    placement_provenance: dict[str, Any] | None = None


class PlacementSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    xy: Vector2 | None = None
    sigma_m: Vector2 | None = None
    covariance_m2: Matrix2 | None = None
    used: bool
    reason: str | None = None
    sidecar_player_id: int | None = None
    mapped_player_id: int | None = None


class PlacementFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_idx: int = Field(ge=0)
    t: FiniteFloat = Field(ge=0.0)
    original_world_xy: Vector2
    fused_world_xy: Vector2
    smoothed_world_xy: Vector2
    covariance_m2: Matrix2
    stance: bool
    signals: list[PlacementSignal] = Field(default_factory=list)
    source_counts: dict[str, int] = Field(default_factory=dict)
    gap_hold: bool | None = None
    output_source: str | None = None
    visual_root_step_bounded: bool | None = None

    @field_validator("source_counts")
    @classmethod
    def _source_counts_must_be_nonnegative(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 0 for count in value.values()):
            raise ValueError("source_counts must be non-negative")
        return value


class PlacementPlayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    frames: list[PlacementFrame]


class PlacementSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_count: int = Field(ge=0)
    frame_count: int = Field(ge=0)
    coverage_unchanged: bool
    source_counts: dict[str, int] = Field(default_factory=dict)
    jitter_before_after_mps: dict[str, dict[str, FiniteFloat]] = Field(default_factory=dict)
    stance_wobble_before_after_m: dict[str, dict[str, FiniteFloat]] = Field(default_factory=dict)
    court_bounds_violations: int = Field(ge=0)
    sidecar_identity: dict[str, Any] = Field(default_factory=dict)
    boundary_guards: dict[str, Any] = Field(default_factory=dict)
    smoothing_guards: dict[str, Any] = Field(default_factory=dict)
    visual_smoothing: dict[str, Any] = Field(default_factory=dict)
    side_quadrant_consistency: dict[str, Any] = Field(default_factory=dict)
    camera_motion_path: str | None = None
    camera_motion_frames_used: int | None = Field(default=None, ge=0)
    camera_motion_frames_uncompensated: int | None = Field(default=None, ge=0)
    camera_motion_artifact_frame_count: int | None = Field(default=None, ge=0)
    camera_motion_artifact_compensated_frame_count: int | None = Field(default=None, ge=0)

    @field_validator("source_counts")
    @classmethod
    def _summary_source_counts_must_be_nonnegative(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 0 for count in value.values()):
            raise ValueError("source_counts must be non-negative")
        return value


class PlacementArtifact(StrictArtifact):
    artifact_type: Literal["racketsport_placement"]
    fps: FiniteFloat = Field(gt=0.0)
    source: str
    tracks_path: str
    backup_tracks_path: str
    refine_from_sam3d: bool
    homography_pixel_convention: str | None = None
    undistort_applied: bool
    players: list[PlacementPlayer]
    summary: PlacementSummary
    provenance: dict[str, Any] = Field(default_factory=dict)


class Sam3DKeypoint2D(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    index: int = Field(ge=0)
    xy_px: Vector2
    conf: FiniteFloat = Field(ge=0.0)


class Sam3DKeypoints2DFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_idx: int = Field(ge=0)
    t: FiniteFloat = Field(ge=0.0)
    keypoints: list[Sam3DKeypoint2D]


class Sam3DKeypoints2DPlayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    frames: list[Sam3DKeypoints2DFrame]


class Sam3DKeypoints2D(StrictArtifact):
    artifact_type: Literal["racketsport_sam3d_keypoints_2d"]
    source: str
    foot_keypoint_indices: dict[str, int] = Field(default_factory=dict)
    players: list[Sam3DKeypoints2DPlayer] = Field(default_factory=list)


class PlayerGroundFoot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    side: Literal["L", "R"]
    court_xy: Vector2
    height_m: FiniteFloat
    contact: bool
    sigma_p_m: FiniteFloat = Field(ge=0.0)
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    world_xyz: Vector3
    source_points: list[str] = Field(default_factory=list)


class PlayerGroundFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: FiniteFloat = Field(ge=0.0)
    feet: list[PlayerGroundFoot]
    root_world: Vector3
    joints_world: list[Vector3] = Field(default_factory=list)
    mesh_ref: str | None = None

    @model_validator(mode="after")
    def _must_have_both_feet(self) -> "PlayerGroundFrame":
        sides = {foot.side for foot in self.feet}
        if len(self.feet) != 2 or sides != {"L", "R"}:
            raise ValueError("player_ground frame must include both L and R feet")
        return self


class PlayerGroundPlayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    frames: list[PlayerGroundFrame]


class PlayerGroundArtifact(StrictArtifact):
    artifact_type: Literal["racketsport_player_ground"]
    fps: FiniteFloat = Field(gt=0.0)
    players: list[PlayerGroundPlayer]
    source: str
    not_gate_verified: bool

    @field_validator("not_gate_verified")
    @classmethod
    def _must_be_not_gate_verified(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("player_ground is not gate verified")
        return value


class CourtCallEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: FiniteFloat = Field(ge=0.0)
    player_id: int
    foot: Literal["L", "R"]
    boundary: Literal["kitchen", "sideline", "baseline", "centerline"]
    decision: Literal["in", "out", "kitchen", "too_close_to_call"]
    signed_dist_m: FiniteFloat
    sigma_p_m: FiniteFloat = Field(ge=0.0)
    frames: list[int] = Field(min_length=1)
    metric_confidence: Literal["high", "med", "low"]
    capture_quality_grade: Literal["good", "warn", "poor"]

    @field_validator("frames")
    @classmethod
    def _frames_must_be_nonnegative(cls, value: list[int]) -> list[int]:
        if any(frame < 0 for frame in value):
            raise ValueError("frames must be non-negative")
        return value


class CallsArtifact(StrictArtifact):
    artifact_type: Literal["racketsport_court_calls"]
    source: str
    events: list[CourtCallEvent]
    summary: dict[str, int | str]
    not_gate_verified: bool

    @field_validator("not_gate_verified")
    @classmethod
    def _must_be_not_gate_verified(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("court calls are not gate verified")
        return value

    @model_validator(mode="after")
    def _summary_must_match_events(self) -> "CallsArtifact":
        too_close = sum(1 for event in self.events if event.decision == "too_close_to_call")
        expected = {
            "total_events": len(self.events),
            "hard_call_count": len(self.events) - too_close,
            "too_close_to_call_count": too_close,
            "status": "not_gate_verified",
        }
        for key, value in expected.items():
            if self.summary.get(key) != value:
                raise ValueError(f"court calls summary {key} must match events")
        return self


class DriftLogCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame: int = Field(ge=0)
    p95_px: FiniteFloat = Field(ge=0.0)
    tripped: bool


class DriftRecalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_frame: int = Field(ge=0)
    to_frame: int = Field(ge=0)
    reason: str

    @model_validator(mode="after")
    def _frames_must_be_ordered(self) -> "DriftRecalibration":
        if self.to_frame < self.from_frame:
            raise ValueError("to_frame must be greater than or equal to from_frame")
        return self


class DriftLog(StrictArtifact):
    artifact_type: Literal["racketsport_drift_log"]
    checks: list[DriftLogCheck]
    recalibrations: list[DriftRecalibration] = Field(default_factory=list)


class PersonLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: int = Field(gt=0)
    bbox_xywh: tuple[float, float, float, float]
    ignored: bool = False
    visibility: float | None = None
    confidence: float | None = None
    class_id: int | None = None
    class_name: str | None = None
    person_class: bool = True

    @field_validator("bbox_xywh")
    @classmethod
    def _bbox_must_have_positive_extent(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        _, _, width, height = value
        if width <= 0.0 or height <= 0.0:
            raise ValueError("bbox_xywh width and height must be positive")
        return value

    @field_validator("visibility", "confidence")
    @classmethod
    def _optional_unit_interval(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("value must be in [0, 1]")
        return value


class PersonGroundTruthFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(ge=0)
    source_frame_id: int = Field(gt=0)
    labels: list[PersonLabel]


class PersonGroundTruthSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_count: int = Field(ge=0)
    valid_label_count: int = Field(ge=0)
    ignored_label_count: int = Field(ge=0)
    track_ids: list[int]
    max_valid_players_per_frame: int = Field(ge=0)

    @field_validator("track_ids")
    @classmethod
    def _track_ids_must_be_positive(cls, value: list[int]) -> list[int]:
        if any(track_id <= 0 for track_id in value):
            raise ValueError("track_ids must be positive")
        return value


class PersonGroundTruth(StrictArtifact):
    artifact_type: Literal["racketsport_person_ground_truth"]
    clip_id: str
    source_format: Literal["cvat_mot_1_1", "cvat_video_1_1"]
    source_path: str
    fps: float | None = Field(default=None, gt=0.0)
    frames: list[PersonGroundTruthFrame]
    summary: PersonGroundTruthSummary


class CvatVideoTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: int | None = None
    name: str | None = None
    size: int = Field(ge=0)
    mode: str | None = None
    start_frame: int = Field(ge=0)
    stop_frame: int = Field(ge=0)
    frame_filter: str | None = None
    original_size: tuple[int, int]
    source: str | None = None
    dumped: str | None = None

    @field_validator("original_size")
    @classmethod
    def _original_size_must_be_positive(cls, value: tuple[int, int]) -> tuple[int, int]:
        width, height = value
        if width <= 0 or height <= 0:
            raise ValueError("original_size values must be positive")
        return value


class CvatVideoBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: int = Field(ge=0)
    label: str
    frame_index: int = Field(ge=0)
    bbox_xyxy: tuple[float, float, float, float]
    bbox_xywh: tuple[float, float, float, float]
    keyframe: bool
    occluded: bool
    source: str | None = None
    visibility_level: BallVisibilityLevel | None = None
    center_convention: Literal["blur_midpoint", "disk_center", "unknown", "review_to_blur_streak_center"] | None = None
    blur_angle_deg: FiniteFloat | None = None
    blur_length_px: FiniteFloat | None = None
    blur_width_px: FiniteFloat | None = None
    blur_label_quality: Literal["clear", "weak", "absent", "unknown"] | None = None

    @field_validator("label")
    @classmethod
    def _label_must_be_nonempty(cls, value: str) -> str:
        if not value:
            raise ValueError("label must be non-empty")
        return value

    @field_validator("bbox_xyxy")
    @classmethod
    def _xyxy_must_have_positive_extent(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = value
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy must have positive extent")
        return value

    @field_validator("bbox_xywh")
    @classmethod
    def _xywh_must_have_positive_extent(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        _, _, width, height = value
        if width <= 0.0 or height <= 0.0:
            raise ValueError("bbox_xywh width and height must be positive")
        return value

    @field_validator("blur_length_px", "blur_width_px")
    @classmethod
    def _blur_extent_must_be_nonnegative(cls, value: float | None) -> float | None:
        if value is not None and value < 0.0:
            raise ValueError("blur extent must be nonnegative")
        return value


class CvatVideoFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(ge=0)
    boxes: list[CvatVideoBox]
    visibility_levels_by_label: dict[str, BallVisibilityLevel] = Field(default_factory=dict)


class CvatVideoTrackSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: int = Field(ge=0)
    label: str
    visible_box_count: int = Field(ge=0)
    outside_box_count: int = Field(ge=0)
    keyframe_count: int = Field(ge=0)
    first_visible_frame: int | None = None
    last_visible_frame: int | None = None


class CvatVideoAnnotationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_count: int = Field(ge=0)
    visible_box_count: int = Field(ge=0)
    outside_box_count: int = Field(ge=0)
    labels: list[str]
    track_count_by_label: dict[str, int]
    visible_box_count_by_label: dict[str, int]


class CvatVideoAnnotations(StrictArtifact):
    artifact_type: Literal["racketsport_cvat_video_annotations"]
    clip_id: str
    source_format: Literal["cvat_video_1_1", "cvat_images_1_1"]
    source_path: str
    reviewed_frame_indices: list[int] | None = None
    reviewed_frame_indices_source: Literal["cvat_frame_filter", "explicit", "dense_all_frames"] | None = None
    task: CvatVideoTask
    frames: list[CvatVideoFrame]
    tracks: list[CvatVideoTrackSummary]
    summary: CvatVideoAnnotationSummary

    @field_validator("reviewed_frame_indices")
    @classmethod
    def _reviewed_indices_must_be_sorted_unique(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if any(index < 0 for index in value):
            raise ValueError("reviewed_frame_indices must be nonnegative")
        if value != sorted(set(value)):
            raise ValueError("reviewed_frame_indices must be sorted and unique")
        return value


class OnDevicePersonDetection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_id: int = Field(gt=0)
    bbox_xywh: tuple[float, float, float, float]
    confidence: float
    source: str
    role: str | None = None

    @field_validator("bbox_xywh")
    @classmethod
    def _bbox_must_have_positive_extent(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        _, _, width, height = value
        if width <= 0.0 or height <= 0.0:
            raise ValueError("bbox_xywh width and height must be positive")
        return value

    @field_validator("confidence")
    @classmethod
    def _confidence_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        return value


class OnDevicePersonFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(ge=0)
    detections: list[OnDevicePersonDetection]


class OnDevicePersonTracksSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_count: int = Field(ge=0)
    detection_count: int = Field(ge=0)
    track_ids: list[int]

    @field_validator("track_ids")
    @classmethod
    def _track_ids_must_be_positive(cls, value: list[int]) -> list[int]:
        if any(track_id <= 0 for track_id in value):
            raise ValueError("track_ids must be positive")
        return value


class OnDevicePersonTracks(StrictArtifact):
    artifact_type: Literal["racketsport_on_device_person_tracks"]
    clip_id: str
    candidate: str
    device_model: str | None = None
    coordinate_space: Literal["source_video_pixels"] = "source_video_pixels"
    resolution: tuple[int, int] | None = None
    fps: float = Field(gt=0.0)
    frames: list[OnDevicePersonFrame]
    summary: OnDevicePersonTracksSummary


class OnDevicePersonTimingSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_index: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    processed: bool


class OnDevicePersonTimingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    processed_frame_count: int = Field(ge=0)
    dropped_frame_count: int = Field(ge=0)
    sustained_processed_fps: float = Field(ge=0.0)
    p50_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)


class OnDevicePersonTiming(StrictArtifact):
    artifact_type: Literal["racketsport_on_device_person_timing"]
    clip_id: str
    candidate: str
    mode: Literal["replay", "live"]
    device_model: str | None = None
    os_version: str | None = None
    wall_clock_seconds: float = Field(ge=0.0)
    dropped_frame_count: int = Field(ge=0)
    model_load_ms: float | None = Field(default=None, ge=0.0)
    mlpackage_size_mb: float | None = Field(default=None, ge=0.0)
    started_thermal_state: str | None = None
    ended_thermal_state: str | None = None
    samples: list[OnDevicePersonTimingSample] = Field(default_factory=list)
    summary: OnDevicePersonTimingSummary


class MobilePersonTrackingMetrics(StrictArtifact):
    artifact_type: Literal["racketsport_mobile_person_tracking_metrics"]
    clip_id: str
    candidate: str
    iou_threshold: float = Field(gt=0.0, le=1.0)
    frames: int = Field(ge=0)
    gt_detections: int = Field(ge=0)
    pred_detections: int = Field(ge=0)
    matches: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    id_switches: int = Field(ge=0)
    idf1: float = Field(ge=0.0, le=1.0)
    mota: float
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    expected_players: int = Field(ge=0)
    expected_player_coverage: float = Field(ge=0.0, le=1.0)
    expected_player_frames: int = Field(ge=0)
    exact_expected_player_frames: int = Field(ge=0)


class FootContact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: bool
    right: bool


class FootLockSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scaffold: str
    contact_frames: int = Field(ge=0)
    contact_samples: int = Field(ge=0)
    root_speed_limited_frames: int = Field(default=0, ge=0)
    max_slide_m: float = Field(ge=0.0)
    max_penetration_m: float = Field(ge=0.0)
    skate_free: bool


class SmplFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_idx: int | None = Field(default=None, ge=0)
    t: float = Field(ge=0.0)
    global_orient: list[float]
    body_pose: list[float]
    left_hand_pose: list[float] = Field(default_factory=list)
    right_hand_pose: list[float] = Field(default_factory=list)
    transl_world: Vector3
    track_world_xy: Vector2 | None = None
    temporal_smoothing_reset: bool = False
    temporal_smoothing_metadata: dict[str, Any] | None = None
    joints_world: list[Vector3]
    mesh_vertices_world: list[Vector3] = Field(default_factory=list)
    joint_conf: list[float]
    foot_contact: FootContact
    foot_lock: FootContact | None = None
    grf: list[Vector3] | None = None
    confidence_provenance: dict[str, Any] | None = None
    body_grounding_refinement: dict[str, Any] | None = None


class SmplPlayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    betas: list[float]
    # ADDITIVE (P2-2 GATE 1b, w5_p22latent_20260707): MHR scale_params
    # (28-dim bone-length/proportions correction), collapsed per-player the
    # same way `betas` is -- see worldhmr.py::compute_body_skeleton_and_metrics.
    # New field, no removals; default keeps every existing producer/consumer
    # that predates this lane valid (empty list = "not carried").
    scale: list[float] = Field(default_factory=list)
    frames: list[SmplFrame]
    skate_free: bool
    physics: str
    foot_lock: FootLockSummary | None = None


class SmplMotion(StrictArtifact):
    model: Literal["smpl", "smplx", "sam3dbody_world_joints", "sat_hmr_world_joints"]
    fps: float = Field(gt=0.0)
    world_frame: Literal["court_Z0"]
    mesh_faces: list[tuple[int, int, int]] = Field(default_factory=list)
    players: list[SmplPlayer]
    body_grounding_refinement: dict[str, Any] | None = None

    @field_validator("mesh_faces")
    @classmethod
    def _mesh_face_indices_must_be_nonnegative(cls, value: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        if any(index < 0 for face in value for index in face):
            raise ValueError("mesh_faces must not contain negative indices")
        return value


class TrustBand(BaseModel):
    """Per-entity confidence/gate-provenance for scrubber trust badges."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    gate_id: str
    gate_status: str
    badge: Literal["verified", "preview", "low_confidence"]
    reason: str
    evidence_path: str | None = None


class SkeletonPlausibility(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["pass", "low_confidence"]
    reasons: list[str] = Field(default_factory=list)
    joint_confidence_floor: FiniteFloat = Field(ge=0.0, le=1.0)
    max_bone_zscore: FiniteFloat = Field(gt=0.0)
    source: str


class SkeletonFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_idx: int | None = Field(default=None, ge=0)
    t: float = Field(ge=0.0)
    transl_world: Vector3 | None = None
    joints_world: list[Vector3]
    joint_conf: list[float]
    smoothing_flag: list[str] = Field(default_factory=list)
    temporal_smoothing_metadata: dict[str, Any] | None = None
    confidence_provenance: dict[str, Any] | None = None
    body_grounding_refinement: dict[str, Any] | None = None
    skeleton_implausible: bool | None = None
    skeleton_plausibility: SkeletonPlausibility | None = None
    trust_band: TrustBand | None = None


class SkeletonPlayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    frames: list[SkeletonFrame]


class Skeleton3D(StrictArtifact):
    artifact_type: Literal["racketsport_skeleton3d"] = "racketsport_skeleton3d"
    fps: float | None = Field(default=None, gt=0.0)
    world_frame: Literal["court_Z0", "relative_only"] | None = None
    source_model: str | None = None
    joint_names: list[str]
    preview_only: bool
    players: list[SkeletonPlayer]
    provenance: dict[str, Any] = Field(default_factory=dict)
    body_grounding_refinement: dict[str, Any] | None = None


class BallFrame(BaseModel):
    """Ball sample schema with additive four-level visibility labels.

    `visible` is the legacy raw bool and is intentionally preserved. New reviewed labels may set
    `visibility_level` to clear/partial/full/out_of_frame. Visibility-weighted BCE uses weights
    clear=1, partial=2, full=3, out_of_frame=3. If `visibility_level` is absent, keep the legacy
    bool readable as ambiguous legacy_visible/legacy_hidden instead of inventing a 4-level label.
    """

    model_config = ConfigDict(extra="forbid")

    t: FiniteFloat = Field(ge=0.0)
    xy: Vector2
    conf: FiniteFloat = Field(ge=0.0, le=1.0)
    visible: bool
    visibility_level: BallVisibilityLevel | None = None
    world_xyz: Vector3 | None = None
    spin_rpm: FiniteFloat | None = None
    speed_mps: FiniteFloat | None = Field(default=None, ge=0.0)
    approx: bool = False


class BounceUncertaintyBreakdown(BaseModel):
    """Inputs behind a per-bounce uncertainty_m, per BALL_TRACKING_PIPELINE.md section 5.6.

    sigma_bounce = sqrt(sigma_reproj^2 + sigma_depth^2 + sigma_ballradius^2 + sigma_localization^2).
    method "camera_geometry_elevation_parallax_v1" is the camera-geometry-derived
    default; "fixed_override" records an explicit caller-supplied uncertainty_m.
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["camera_geometry_elevation_parallax_v1", "fixed_override"]
    sigma_reproj_m: FiniteFloat = Field(ge=0.0)
    sigma_depth_m: FiniteFloat = Field(ge=0.0)
    sigma_ballradius_m: FiniteFloat = Field(ge=0.0)
    sigma_localization_m: FiniteFloat = Field(ge=0.0)
    camera_height_m: FiniteFloat | None = Field(default=None, gt=0.0)
    grazing_angle_deg: FiniteFloat | None = None
    h_max_m: FiniteFloat | None = Field(default=None, ge=0.0)
    v_z_ref_mps: FiniteFloat | None = Field(default=None, ge=0.0)
    dt_s: FiniteFloat | None = Field(default=None, ge=0.0)
    frames_window: FiniteFloat | None = Field(default=None, ge=0.0)
    binding_axis: Literal["x", "y"] | None = None
    ground_sample_distance_m_per_px: FiniteFloat | None = Field(default=None, ge=0.0)
    pose_reprojection_error_px_median: FiniteFloat | None = Field(default=None, ge=0.0)
    pose_source: str | None = None


class Bounce(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float = Field(ge=0.0)
    frame: int | None = Field(default=None, ge=0)
    world_xy: Vector2
    contact_xy_img: Vector2 | None = None
    p_bounce: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    audio_delta_ms: FiniteFloat | None = None
    source: str | None = None
    margin_m: FiniteFloat | None = None
    uncertainty_m: FiniteFloat | None = Field(default=None, ge=0.0)
    confidence: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    call: Literal["in", "out", "too_close_to_call"] | None = None
    nearest_line: str | None = None
    region: str | None = None
    dominant_uncertainty_term: str | None = None
    uncertainty_breakdown: BounceUncertaintyBreakdown | None = None
    not_ground_truth: bool = False
    render_only: bool = False
    not_for_detection_metrics: bool = False


class BallTrack(StrictArtifact):
    fps: float = Field(gt=0.0)
    source: Literal[
        "tracknet",
        "wasb",
        "fused",
        "tap",
        "pbmat",
        "totnet",
        "blurball",
        "vn_trajectories",
        "physics_filled",
        "sam31",
    ]
    input_preprocessing: BallInputPreprocessingMode | None = None
    frames: list[BallFrame]
    bounces: list[Bounce] = Field(default_factory=list)


class BallArcRenderPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_xyz: Vector3
    court_xy: Vector2


class BallArcRenderShot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: BallArcRenderPoint
    peak: BallArcRenderPoint
    end: BallArcRenderPoint
    speed_mps: NonNegativeFiniteFloat
    speed_mph: NonNegativeFiniteFloat
    height_over_net_m: FiniteFloat | None = None
    height_over_net_definition: str | None = None
    distance_m: NonNegativeFiniteFloat
    path_distance_m: NonNegativeFiniteFloat
    render_only: Literal[True]
    not_for_detection_metrics: Literal[True]


class BallArcRenderSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_id: int | str
    t0: FiniteFloat
    t1: FiniteFloat
    frame_start: int = Field(ge=0)
    frame_end: int = Field(ge=0)
    anchor_types: list[str]
    anchor_frames: list[int]
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    flight_sanity_verdict: str
    flight_sanity_reasons: list[str] = Field(default_factory=list)
    fit_status: str
    reprojection_rmse_px: FiniteFloat | None = Field(default=None, ge=0.0)
    max_reprojection_error_px: FiniteFloat | None = Field(default=None, ge=0.0)
    endpoint_error_m: FiniteFloat | None = Field(default=None, ge=0.0)
    net_clearance_m: FiniteFloat | None = None
    net_clearance_ok: bool | None = None
    bridge: Literal[False]
    render_only: Literal[True]
    not_for_detection_metrics: Literal[True]
    shot: BallArcRenderShot


class BallArcRenderBridge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bridge_id: str
    t0: FiniteFloat
    t1: FiniteFloat
    reason: str
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    render_only: Literal[True]
    not_for_detection_metrics: Literal[True]


class BallArcRenderSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: FiniteFloat
    frame_float: FiniteFloat | None = None
    segment_id: int | str
    world_xyz: Vector3
    court_xy: Vector2
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    band: str
    bridge: bool
    bridge_id: str | None = None
    render_only: Literal[True]
    not_for_detection_metrics: Literal[True]


class BallArcRender(StrictArtifact):
    artifact_type: Literal["racketsport_ball_arc_render"]
    clip_id: str
    generated_at: str | None = None
    source: str | None = None
    source_artifact: str
    solver_status: str
    solver_trusted_for_render: bool = False
    render_only: Literal[True]
    not_for_detection_metrics: Literal[True]
    trusted_for_ball_detection_metrics: Literal[False]
    policy: dict[str, Any] = Field(default_factory=dict)
    segments: list[BallArcRenderSegment]
    bridges: list[BallArcRenderBridge]
    samples: list[BallArcRenderSample]
    summary: dict[str, Any]


class BallCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    xy: Vector2
    score: FiniteFloat = Field(ge=0.0, le=1.0)
    source_detector: str


class BallCandidateFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame: int = Field(ge=0)
    candidates: list[BallCandidate] = Field(default_factory=list)


class BallCandidates(StrictArtifact):
    artifact_type: Literal["racketsport_ball_candidates"]
    fps: float = Field(gt=0.0)
    source: Literal["tracknet", "wasb", "pbmat", "totnet", "blurball"]
    source_mode: str
    input_preprocessing: BallInputPreprocessingMode | None = None
    primary_output: str
    max_candidates_per_frame: int = Field(ge=1)
    nms_radius_px: FiniteFloat | None = Field(default=None, ge=0.0)
    not_ground_truth: Literal[True]
    candidate_prediction: Literal[True]
    provenance: dict[str, Any] = Field(default_factory=dict)
    frames: list[BallCandidateFrame]

    @model_validator(mode="after")
    def _frames_must_be_unique_and_within_top_k(self) -> "BallCandidates":
        seen: set[int] = set()
        for frame in self.frames:
            if frame.frame in seen:
                raise ValueError("ball candidate frame ids must be unique")
            seen.add(frame.frame)
            if len(frame.candidates) > self.max_candidates_per_frame:
                raise ValueError("frame candidate count exceeds max_candidates_per_frame")
        return self


class BallLineCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: FiniteFloat = Field(ge=0.0)
    world_xy: Vector2
    court_call: Literal["in", "out", "unknown"]
    kitchen_call: Literal["nvz", "non_nvz", "unknown"]
    zone: str | None = None
    nearest_boundary_line_id: str | None = None
    nearest_kitchen_line_id: str | None = None
    boundary_margin_m: FiniteFloat
    kitchen_margin_m: FiniteFloat | None = None
    uncertainty_radius_m: FiniteFloat = Field(ge=0.0)
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class BallLineCallSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "blocked"]
    total_bounces: int = Field(ge=0)
    court_call_counts: dict[str, int]
    kitchen_call_counts: dict[str, int]
    reasons: list[str] = Field(default_factory=list)


class BallLineCalls(StrictArtifact):
    artifact_type: Literal["racketsport_ball_line_calls"]
    sport: Literal["pickleball", "tennis"]
    source: str
    rule_scope: Literal["ball_bounce_location_only"]
    world_frame: Literal["court_Z0"]
    input_ball_track: str | None = None
    uncertainty_radius_m: FiniteFloat = Field(ge=0.0)
    calls: list[BallLineCall] = Field(default_factory=list)
    summary: BallLineCallSummary
    not_gate_verified: bool

    @field_validator("not_gate_verified")
    @classmethod
    def _must_be_not_gate_verified(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("ball_line_calls is not gate verified")
        return value


class RuntimeEnvironment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    python: str | None = None
    platform: str | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    cuda_available: bool
    cuda_device_name: str | None = None
    cuda_visible_devices: str | None = None


class BallModelRuntimeProfile(StrictArtifact):
    artifact_type: Literal["racketsport_ball_model_runtime_profile"]
    candidate: str
    model_id: str
    clip_id: str
    video: str
    source_fps: FiniteFloat = Field(gt=0.0)
    batch_size: int = Field(gt=0)
    command: list[str]
    returncode: int | None = None
    status: Literal["ran", "failed", "blocked_missing_cuda"]
    wall_seconds: FiniteFloat | None = Field(default=None, ge=0.0)
    processed_frame_count: int | None = Field(default=None, ge=0)
    video_seconds_processed: FiniteFloat | None = Field(default=None, ge=0.0)
    effective_fps: FiniteFloat | None = Field(default=None, ge=0.0)
    realtime_factor: FiniteFloat | None = Field(default=None, ge=0.0)
    timing_breakdown: dict[str, FiniteFloat] = Field(default_factory=dict)
    runtime_env: RuntimeEnvironment
    gpu_verified: bool
    claim_scope: Literal[
        "cpu_profiler_smoke",
        "h100_runtime_profile_not_accuracy_gate",
        "blocked_missing_cuda",
    ]
    verified: bool
    not_ground_truth: bool
    not_accuracy_verified: bool
    stdout_tail: str = ""
    stderr_tail: str = ""
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _gpu_verified_requires_cuda_evidence(self) -> BallModelRuntimeProfile:
        if self.gpu_verified and (not self.runtime_env.cuda_available or not self.runtime_env.cuda_device_name):
            raise ValueError("gpu_verified requires CUDA evidence")
        if self.gpu_verified and self.claim_scope != "h100_runtime_profile_not_accuracy_gate":
            raise ValueError("gpu_verified requires h100 runtime claim scope")
        if self.verified:
            raise ValueError("runtime profile is not an accuracy or deployment verification gate")
        if not self.not_ground_truth or not self.not_accuracy_verified:
            raise ValueError("runtime profile must remain not_ground_truth and not_accuracy_verified")
        return self


def _validate_paddle_dimensions_dict(value: dict[str, float]) -> dict[str, float]:
    has_named_dims = {"length", "width"}.issubset(value)
    has_short_dims = {"h", "w"}.issubset(value)
    if not has_named_dims and not has_short_dims:
        raise ValueError("paddle_dims_in must include length/width or h/w")
    if any(dim <= 0 for dim in value.values()):
        raise ValueError("paddle_dims_in values must be positive")
    return value


class RacketCandidateFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float
    corners_px: list[Vector2]
    conf: float
    source: str

    @field_validator("corners_px")
    @classmethod
    def _must_have_four_ordered_corners(cls, value: list[Vector2]) -> list[Vector2]:
        if len(value) != 4 or any(len(point) != 2 for point in value):
            raise ValueError("corners_px must contain top-left, top-right, bottom-right, bottom-left 2D points")
        return value

    @field_validator("conf")
    @classmethod
    def _must_have_unit_confidence(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("conf must be in [0, 1]")
        return value

    @field_validator("source")
    @classmethod
    def _must_have_source(cls, value: str) -> str:
        if not value:
            raise ValueError("source must be non-empty")
        return value


class RacketCandidatePlayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    paddle_dims_in: dict[str, float]
    frames: list[RacketCandidateFrame]

    @field_validator("paddle_dims_in")
    @classmethod
    def _must_have_paddle_dimensions(cls, value: dict[str, float]) -> dict[str, float]:
        return _validate_paddle_dimensions_dict(value)


class RacketCandidates(StrictArtifact):
    artifact_type: Literal["racketsport_racket_candidates"]
    fps: float
    players: list[RacketCandidatePlayer]

    @field_validator("fps")
    @classmethod
    def _must_have_positive_fps(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("fps must be positive")
        return value


class SE3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    R: Matrix3
    t: Vector3

    @field_validator("R")
    @classmethod
    def _must_be_rotation_matrix(cls, value: Matrix3) -> Matrix3:
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("pose_se3.R must be a 3x3 matrix")
        _validate_rotation_orthonormal(value)
        return value

    @field_validator("t")
    @classmethod
    def _must_be_translation_vector(cls, value: Vector3) -> Vector3:
        if len(value) != 3:
            raise ValueError("pose_se3.t must be a 3-vector")
        return value


class RacketFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float = Field(ge=0.0)
    pose_se3: SE3
    conf: float = Field(ge=0.0, le=1.0)
    world_frame: Literal["camera", "court_Z0"] = "camera"
    translation_unit: Literal["cm", "m"] = "cm"
    source: str = "unspecified"
    reprojection_error_px: float | None = None
    ambiguous: bool = False

    @field_validator("conf")
    @classmethod
    def _must_have_unit_confidence(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("conf must be in [0, 1]")
        return value

    @field_validator("reprojection_error_px")
    @classmethod
    def _must_have_non_negative_reprojection_error(cls, value: float | None) -> float | None:
        if value is not None and value < 0.0:
            raise ValueError("reprojection_error_px must be non-negative")
        return value


class RacketContact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float = Field(ge=0.0)
    contact_point_face_cm: Vector2
    face_normal: Vector3
    conf: float = Field(ge=0.0, le=1.0)

    @field_validator("contact_point_face_cm")
    @classmethod
    def _must_be_face_point(cls, value: Vector2) -> Vector2:
        if len(value) != 2:
            raise ValueError("contact_point_face_cm must be a 2-vector")
        return value

    @field_validator("face_normal")
    @classmethod
    def _must_be_face_normal(cls, value: Vector3) -> Vector3:
        if len(value) != 3:
            raise ValueError("face_normal must be a 3-vector")
        return value


class RacketPlayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    paddle_dims_in: dict[str, float]
    frames: list[RacketFrame]
    contacts: list[RacketContact] = Field(default_factory=list)

    @field_validator("paddle_dims_in")
    @classmethod
    def _must_have_paddle_dimensions(cls, value: dict[str, float]) -> dict[str, float]:
        return _validate_paddle_dimensions_dict(value)


class RacketPose(StrictArtifact):
    fps: float = Field(gt=0.0)
    world_frame: Literal["camera", "court_Z0"] = "camera"
    translation_unit: Literal["cm", "m"] = "cm"
    players: list[RacketPlayer]


def _validate_rotation_orthonormal(value: Matrix3) -> None:
    tolerance = 1e-3
    rows = [[float(entry) for entry in row] for row in value]
    for row in rows:
        norm = math.sqrt(sum(entry * entry for entry in row))
        if abs(norm - 1.0) > tolerance:
            raise ValueError("pose_se3.R must be orthonormal")
    for left_index in range(3):
        for right_index in range(left_index + 1, 3):
            dot = sum(rows[left_index][col] * rows[right_index][col] for col in range(3))
            if abs(dot) > tolerance:
                raise ValueError("pose_se3.R must be orthonormal")
    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    if abs(determinant - 1.0) > tolerance:
        raise ValueError("pose_se3.R determinant must be 1")


class EventSources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: float | None = None
    wrist_vel: float
    ball_inflection: float
    human_review: float | None = None

    @field_validator("audio", "wrist_vel", "ball_inflection", "human_review")
    @classmethod
    def _must_be_unit_interval(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("source score must be in [0, 1]")
        return value


class ContactWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t0: float = Field(ge=0.0)
    t1: float = Field(ge=0.0)
    importance: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _times_must_be_ordered(self) -> "ContactWindow":
        if self.t1 < self.t0:
            raise ValueError("t1 must be greater than or equal to t0")
        return self


class ContactEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["contact", "bounce", "net_cross", "into_net"]
    t: float = Field(ge=0.0)
    frame: int = Field(ge=0)
    player_id: int | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    sources: EventSources
    window: ContactWindow
    trust_band_note: str | None = None


class ContactWindows(StrictArtifact):
    events: list[ContactEvent]


class ContactWindowCandidateSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_count: int = Field(ge=0)
    rejected_item_count: int = Field(ge=0)
    by_type: dict[str, int] = Field(default_factory=dict)
    by_status: dict[str, int] = Field(default_factory=dict)
    uncertainty_flags: list[str] = Field(default_factory=list)

    @field_validator("by_type", "by_status")
    @classmethod
    def _summary_counts_must_be_nonnegative(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 0 for count in value.values()):
            raise ValueError("summary counts must be nonnegative")
        return value


class ContactWindowCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    type: Literal["contact", "bounce", "net_cross", "into_net"]
    frame: int = Field(ge=0)
    t: float = Field(ge=0.0)
    xy_px: Vector2 | None = None
    source_label: str
    source_status: str
    source_confidence: float = Field(ge=0.0, le=1.0)
    candidate_confidence: float = Field(ge=0.0, le=1.0)
    window: ContactWindow


class ContactWindowCandidates(StrictArtifact):
    artifact_type: Literal["racketsport_contact_window_candidates"]
    clip: str
    fps: float = Field(gt=0.0)
    source_event_path: str
    not_gate_verified: Literal[True]
    trusted_for_body: Literal[False]
    promotion_target: Literal["contact_windows.json"]
    candidates: list[ContactWindowCandidate]
    summary: ContactWindowCandidateSummary

    @model_validator(mode="after")
    def _summary_must_match_candidates(self) -> "ContactWindowCandidates":
        if self.summary.candidate_count != len(self.candidates):
            raise ValueError("summary.candidate_count must equal candidates length")
        expected_by_type: dict[str, int] = {}
        expected_by_status: dict[str, int] = {}
        for candidate in self.candidates:
            expected_by_type[candidate.type] = expected_by_type.get(candidate.type, 0) + 1
            expected_by_status[candidate.source_status] = expected_by_status.get(candidate.source_status, 0) + 1
        if self.summary.by_type and self.summary.by_type != expected_by_type:
            raise ValueError("summary.by_type must match candidates")
        if self.summary.by_status and self.summary.by_status != expected_by_status:
            raise ValueError("summary.by_status must match candidates")
        return self


class ContactWindowReviewSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_count: int
    pending_count: int
    accepted_count: int
    rejected_count: int


class ContactWindowReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    decision: Literal["pending", "accepted", "rejected"]
    reviewer: str = ""
    reason: str = ""
    player_id: int | None = None
    t_override: float | None = None
    frame_override: int | None = None
    confidence_override: float | None = None
    window_override: ContactWindow | None = None


class ContactWindowReview(StrictArtifact):
    artifact_type: Literal["racketsport_contact_window_review"]
    clip: str
    candidate_path: str
    promotion_target: Literal["contact_windows.json"]
    status: Literal["pending_review", "partially_reviewed", "reviewed"]
    decisions: list[ContactWindowReviewDecision]
    summary: ContactWindowReviewSummary


class MetricValue(BaseModel):
    model_config = ConfigDict(extra="allow")

    value: Any
    conf: FiniteFloat
    gated: bool | None = None


class ShotAlternative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    confidence: float


class ShotMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float
    type: str
    type_conf: float
    top2: list[ShotAlternative] = Field(default_factory=list)
    metrics: dict[str, MetricValue]


class MetricsPlayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    shots: list[ShotMetrics]


class RacketSportMetrics(StrictArtifact):
    players: list[MetricsPlayer]


class ClipRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t0_sec: float
    t1_sec: float


class Drill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    duration_min: float


class Habit(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    title: str
    summary: str
    confidence: float
    clip_ref: ClipRef
    cue: str
    drill: Drill


class Coverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall: float
    skipped_reason_counts: dict[str, int]


class HabitReport(StrictArtifact):
    sport: Literal["pickleball", "tennis"]
    coverage: Coverage
    priority_habit_id: str
    replay_ref: dict[str, str] | None = None
    habits: list[Habit]


class ReplayPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    t0: float
    t1: float
    glb_url: str
    size_mb: float


class ReplayScene(StrictArtifact):
    world_frame: Literal["court_Z0"]
    fps: float
    court_glb: str
    players: list[int]
    points: list[ReplayPoint]


class ReplayViewerLabelOverlay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    label: str
    url: str
    trusted_for_metrics: bool
    not_ground_truth: bool


class ReplayViewerAnnotationSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    clip_id: str
    url: str
    trusted_for_metrics: bool


class ReplayViewerManifest(StrictArtifact):
    artifact_type: Literal["racketsport_replay_viewer_manifest"]
    clip: str
    video_url: str
    virtual_world_url: str
    replay_scene_url: str | None = None
    body_mesh_url: str | None = None
    body_mesh_index_url: str | None = None
    physics_refinement_url: str | None = None
    contact_windows_url: str | None = None
    ball_inflections_url: str | None = None
    ball_arc_solved_url: str | None = None
    ball_arc_render_url: str | None = None
    auto_bounce_candidates_url: str | None = None
    ball_bounce_candidates_url: str | None = None
    ball_flight_sanity_url: str | None = None
    reviewed_bounces_url: str | None = None
    coaching_card_facts_url: str | None = None
    rally_spans_url: str | None = None
    label_overlays: list[ReplayViewerLabelOverlay] = Field(default_factory=list)
    annotation_sources: list[ReplayViewerAnnotationSource] = Field(default_factory=list)
    mesh_status: Literal["windowed_index", "monolithic_unverified", "skeleton_only"] | None = None
    notes: list[str] = Field(default_factory=list)


class ConfidenceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    band: str
    display_band: str | None = None
    predictor: str
    horizon_frames: int = Field(ge=0)
    predicted_sigma_m: FiniteFloat | None = Field(default=None, ge=0.0)


class VirtualWorldNet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoints: list[Vector3]
    center_height_m: float
    post_height_m: float


class VirtualWorldPlacementCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str | None = None
    intrinsics_source: str | None = None
    capture_quality_grade: str | None = None
    metric_confidence: Literal["high", "med", "low"] | None = None
    evidence_path: str | None = None


class VirtualWorldCourt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sport: Literal["pickleball", "tennis"]
    coordinate_frame: str
    length_m: float
    width_m: float
    line_segments: dict[str, list[Vector3]]
    net: VirtualWorldNet
    trust_band: TrustBand | None = None
    placement_calibration: VirtualWorldPlacementCalibration | None = None


class VirtualWorldMeshRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact: str
    player_id: int
    frame_idx: int = Field(ge=0)
    t: float


class VirtualWorldPlayerFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float
    track_world_xy: Vector2 | None = None
    track_conf: float | None = None
    bbox: tuple[float, float, float, float] | None = None
    transl_world: Vector3 | None = None
    joints_world: list[Vector3] = Field(default_factory=list)
    joint_conf: list[float] = Field(default_factory=list)
    mesh_vertices_world: list[Vector3] = Field(default_factory=list)
    mesh_ref: VirtualWorldMeshRef | None = None
    joint_count: int
    mesh_vertex_count: int
    floor_world_xyz: Vector3 | None = None
    floor_source: str | None = None
    floor_offset_m: float | None = None
    min_mesh_z_m: float | None = None
    floor_penetration_m: float = 0.0
    foot_contact: FootContact | None = None
    contact_locked: bool = False
    physics: str | None = None
    grf: list[Vector3] | None = None
    trust_band: TrustBand | None = None
    confidence_provenance: ConfidenceProvenance | None = None


class VirtualWorldJointsSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skeleton3d: int = Field(default=0, ge=0)
    smpl_fill: int = Field(default=0, ge=0)


class VirtualWorldPlayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    side: str | None = None
    role: str | None = None
    representation: Literal["track_only", "joints", "mesh"]
    joints_source: VirtualWorldJointsSource | None = None
    frames: list[VirtualWorldPlayerFrame]
    trust_band: TrustBand | None = None


class VirtualWorldBallFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float
    xy: Vector2
    conf: float
    visible: bool
    world_xyz: Vector3 | None = None
    approx: bool = False
    trust_band: TrustBand | None = None
    confidence_provenance: ConfidenceProvenance | None = None
    render_only: bool = False
    not_for_detection_metrics: bool = False


class VirtualWorldBall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["tracknet", "wasb", "fused", "tap", "pbmat", "totnet", "vn_trajectories", "physics_filled", "sam31"] | None = None
    frames: list[VirtualWorldBallFrame] = Field(default_factory=list)
    trust_band: TrustBand | None = None


class VirtualWorldPaddleFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float
    pose_se3: SE3
    mesh_vertices_world: list[Vector3] = Field(default_factory=list)
    mesh_faces: list[tuple[int, int, int]] = Field(default_factory=list)
    conf: float
    world_frame: Literal["court_Z0"]
    translation_unit: Literal["m"]
    source: str
    reprojection_error_px: float | None = None
    ambiguous: bool = False
    trust_band: TrustBand | None = None
    confidence_provenance: ConfidenceProvenance | None = None
    render_only: bool | None = None
    not_for_detection_metrics: bool | None = None


class VirtualWorldPaddle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_id: int
    paddle_dims_in: dict[str, float]
    frames: list[VirtualWorldPaddleFrame]
    trust_band: TrustBand | None = None


class VirtualWorldCoverageEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_t: float | None = None
    max_t: float | None = None
    frame_span: float = 0.0
    coverage_fraction: float = Field(default=0.0, ge=0.0, le=1.0)


class VirtualWorldPlayerCoverageEntry(VirtualWorldCoverageEntry):
    player_id: int
    observed_frame_count: int = 0
    no_data_frame_count: int = 0


class VirtualWorldTemporalCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_min_t: float | None = None
    clip_max_t: float | None = None
    clip_frame_span: float = 0.0
    ball: VirtualWorldCoverageEntry
    players: list[VirtualWorldPlayerCoverageEntry] = Field(default_factory=list)


class VirtualWorldSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_count: int
    mesh_player_count: int
    mesh_player_frame_count: int = 0
    joint_player_frame_count: int = 0
    track_only_player_frame_count: int = 0
    floor_placed_player_frame_count: int = 0
    floor_contact_player_frame_count: int = 0
    max_floor_penetration_m: float = 0.0
    max_abs_floor_offset_m: float = 0.0
    physics_modes: list[str] = Field(default_factory=list)
    ball_frame_count: int
    approx_ball_frame_count: int = 0
    paddle_player_count: int = 0
    paddle_frame_count: int
    ambiguous_paddle_frame_count: int = 0
    paddle_source: str | None = None
    hidden_paddle_frame_count: int | None = Field(default=None, ge=0)
    warnings: list[str] = Field(default_factory=list)
    temporal_coverage: VirtualWorldTemporalCoverage | None = None


class VirtualWorld(StrictArtifact):
    artifact_type: Literal["racketsport_virtual_world"]
    world_frame: Literal["court_Z0"]
    fps: float
    joint_names: list[str] | None = None
    court: VirtualWorldCourt
    players: list[VirtualWorldPlayer]
    ball: VirtualWorldBall
    paddles: list[VirtualWorldPaddle]
    summary: VirtualWorldSummary
    confidence_gate: dict[str, Any] | None = None
    foot_pin: dict[str, Any] | None = None


class PhysicsRefinement(StrictArtifact):
    model_config = ConfigDict(extra="allow")

    artifact_type: Literal["racketsport_physics_refinement"]
    physics: str
    foot2_done: bool
    must_not_mark_done_verified: bool
    constraint_summary: dict[str, Any]
    execution_plan: dict[str, Any]


class BodyComputeExecution(StrictArtifact):
    model_config = ConfigDict(extra="allow")

    artifact_type: Literal["racketsport_body_compute_execution"]
    mode: str
    scheduled_frames: list[dict[str, Any]]
    skipped_frames: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any]


class BodySerializationTimingItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact: str
    path: str
    bytes: int = Field(ge=0)
    serialization_seconds: NonNegativeFiniteFloat
    skipped: bool = False
    reason: str | None = None


class BodySerializationTimingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_count: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    total_serialization_seconds: NonNegativeFiniteFloat
    json_format: str
    written_count: int | None = Field(default=None, ge=0)
    skipped_count: int | None = Field(default=None, ge=0)


class BodySerializationTiming(StrictArtifact):
    artifact_type: Literal["racketsport_body_serialization_timing"]
    artifacts: list[BodySerializationTimingItem]
    summary: BodySerializationTimingSummary


class BodyStagePhaseTiming(StrictArtifact):
    artifact_type: Literal["racketsport_body_stage_phase_timing"]
    stage_wall_seconds: NonNegativeFiniteFloat
    model_load_s: NonNegativeFiniteFloat | None = None
    orchestrator_model_setup_s: NonNegativeFiniteFloat | None = None
    compile_warmup_s: NonNegativeFiniteFloat | None = None
    inference_s: NonNegativeFiniteFloat | None = None
    subprocess_outer_call_s: NonNegativeFiniteFloat | None = None
    person_frame_count: int = Field(ge=0)
    ms_per_person_steady: NonNegativeFiniteFloat | None = None
    input_prep_s: NonNegativeFiniteFloat | None = None
    runner_request_parse_s: NonNegativeFiniteFloat | None = None
    runner_preprocessing_s: NonNegativeFiniteFloat | None = None
    runner_postprocessing_s: NonNegativeFiniteFloat | None = None
    runner_result_serialization_handoff_s: NonNegativeFiniteFloat | None = None
    runner_other_s: NonNegativeFiniteFloat | None = None
    subprocess_wrapper_handoff_s: NonNegativeFiniteFloat | None = None
    mesh_smpl_payload_assembly_s: NonNegativeFiniteFloat | None = None
    smpl_motion_payload_assembly_s: NonNegativeFiniteFloat | None = None
    array_native_gate_feed_s: NonNegativeFiniteFloat | None = None
    mesh_export_payload_assembly_s: NonNegativeFiniteFloat | None = None
    keypoints_2d_s: NonNegativeFiniteFloat | None = None
    contact_splice_s: NonNegativeFiniteFloat | None = None
    gates_s: NonNegativeFiniteFloat | None = None
    serialization_s: NonNegativeFiniteFloat | None = None
    index_build_s: NonNegativeFiniteFloat | None = None
    artifact_io_s: NonNegativeFiniteFloat | None = None
    attributed_s: NonNegativeFiniteFloat
    other_s: NonNegativeFiniteFloat
    per_bucket_timing: list[dict[str, Any]] = Field(default_factory=list)
    timing_sources: dict[str, str] = Field(default_factory=dict)
    phase_boundaries: dict[str, str] = Field(default_factory=dict)
    not_instrumentable: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class BodyMeshFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_idx: int = Field(ge=0)
    t: float = Field(ge=0.0)
    source_window_index: int | None = Field(default=None, ge=0)
    blend_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    mesh_vertices_world: list[Vector3]
    mesh_faces: list[tuple[int, int, int]] = Field(default_factory=list)
    smplx_params: dict[str, list[float]]
    joints_world: list[Vector3] = Field(default_factory=list)
    joint_conf: list[float] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    trust_badge: Literal["verified", "preview", "low_confidence"] | None = None

    @field_validator("mesh_faces")
    @classmethod
    def _face_indices_must_be_nonnegative(cls, value: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        if any(index < 0 for face in value for index in face):
            raise ValueError("mesh_faces must not contain negative indices")
        return value


class BodyMeshPlayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    frames: list[BodyMeshFrame]


class BodyMeshWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_window_index: int = Field(ge=0)
    frame_start: int = Field(ge=0)
    frame_end: int = Field(ge=0)
    t0: float = Field(ge=0.0)
    t1: float = Field(ge=0.0)
    frame_count: int = Field(ge=0)
    target_player_ids: list[int]
    target_representation: str
    fallback_representation: str
    reason_counts: dict[str, int] = Field(default_factory=dict)
    max_score: float = Field(ge=0.0)


class BodyMeshSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mesh_frame_count: int = Field(ge=0)
    player_count: int = Field(ge=0)
    contact_window_count: int = Field(ge=0)


class BodyMesh(StrictArtifact):
    artifact_type: Literal["racketsport_body_mesh"]
    clip: str
    model: str
    fps: float = Field(gt=0.0)
    world_frame: Literal["court_Z0"]
    faces_ref: str
    mesh_faces: list[tuple[int, int, int]] = Field(default_factory=list)
    joint_names: list[str] = Field(default_factory=list)
    windows: list[BodyMeshWindow] = Field(default_factory=list)
    players: list[BodyMeshPlayer]
    summary: BodyMeshSummary

    @field_validator("mesh_faces")
    @classmethod
    def _static_face_indices_must_be_nonnegative(cls, value: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        if any(index < 0 for face in value for index in face):
            raise ValueError("mesh_faces must not contain negative indices")
        return value


class BodyMeshReadiness(StrictArtifact):
    model_config = ConfigDict(extra="allow")

    artifact_type: Literal["racketsport_body_mesh_readiness"]
    clip: str | None = None
    status: str
    world_mesh_available: bool | None = None
    representation_decision: str
    trusted_for_body_promotion: bool
    summary: dict[str, Any] = Field(default_factory=dict)
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RacketPoseReadiness(StrictArtifact):
    model_config = ConfigDict(extra="allow")

    artifact_type: Literal["racketsport_racket_pose_readiness"]
    clip: str | None = None
    status: str
    blockers: list[str] = Field(default_factory=list)


class RacketPromotionAudit(StrictArtifact):
    model_config = ConfigDict(extra="allow")

    artifact_type: Literal["racketsport_racket_promotion_audit"]
    clip: str | None = None
    trusted_for_rkt_promotion: bool
    blockers: list[str] = Field(default_factory=list)


class ServingManifestSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    schema_version: Literal[1]


class ServingManifestExecution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cpu_only: Literal[True]
    starts_triton: Literal[False]
    downloads_models: Literal[False]
    uses_gpu: Literal[False]
    mutates_model_manifest: Literal[False]
    claims_env_or_eval_completion: Literal[False]


class ServingManifestPathSafety(BaseModel):
    model_config = ConfigDict(extra="forbid")

    safe: bool
    reason: str
    allowed_prefixes: list[str] | None = None


class ServingManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    stage: str
    status: str
    local_path: str | None
    license: str
    commercial_posture: str
    sha256_present: bool
    fallbacks: list[str]
    path_safety: ServingManifestPathSafety


class ServingManifestMissingOrPending(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None
    stage: str
    reason: str
    required_status: str | None = None
    path_reason: str | None = None


class ServingManifestComponent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component_id: str
    stage: str
    kind: Literal["checkpoint", "runtime"]
    role: str
    serving_backend: str
    required_status: Literal["available_on_h100", "available_runtime_on_h100"]
    checkpoint_available: bool | None
    runtime_available: bool | None
    safe_paths: bool
    inventory_ready: bool
    serving_ready: bool
    serving_blockers: list[str]
    entries: list[ServingManifestEntry]
    missing_or_pending: list[ServingManifestMissingOrPending]


class ServingManifestTier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier: Literal["offline_deep", "live_light"]
    checkpoint_runtime_inventory_ready: bool
    serving_ready: bool
    eval0_approval: Literal["not_evaluated_by_cpu_manifest"]
    triton_ensemble_status: Literal["scaffold_only_not_started"]
    components: list[ServingManifestComponent]


class ServingManifestTiers(BaseModel):
    model_config = ConfigDict(extra="forbid")

    offline_deep: ServingManifestTier
    live_light: ServingManifestTier


class ServingManifestSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_model_count: int = Field(ge=0)
    checkpoint_available_count: int = Field(ge=0)
    runtime_available_count: int = Field(ge=0)
    tier_count: int = Field(ge=0)
    component_count: int = Field(ge=0)
    inventory_ready_component_count: int = Field(ge=0)
    serving_ready_component_count: int = Field(ge=0)
    unsafe_model_path_count: int = Field(ge=0)
    unsafe_model_path_ids: list[str]
    pending_item_count: int = Field(ge=0)
    pending_item_ids: list[str]
    missing_component_ids: list[str]


class ServingManifest(StrictArtifact):
    artifact_type: Literal["racketsport_serving_manifest"]
    source_manifest: ServingManifestSource
    execution: ServingManifestExecution
    tiers: ServingManifestTiers
    summary: ServingManifestSummary
    notes: list[str]


class DrillRep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float
    quality: Literal["clean", "fault"]
    reasons: list[str] = Field(default_factory=list)


class DrillReport(StrictArtifact):
    drill: str
    reps: int
    clean_reps: int
    per_rep: list[DrillRep]


EvalStatus = Literal["pass", "fail", "blocked", "not_measured"]
EvalMetricStatus = Literal["measured", "not_measured"]
EvalMetricValue = float | int | bool | str | None


class EvalMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: EvalMetricValue = None
    unit: str | None = None
    gate: str
    passed: bool | None = None
    status: EvalMetricStatus


class EvalClipResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip: str
    run_dir: str
    labels_dir: str
    status: EvalStatus
    missing_label_files: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    metrics: dict[str, EvalMetric] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class EvalSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_clips: int
    ready_clips: int
    evaluated_clips: int
    passed_clips: int
    failed_clips: int
    blocked_clips: int


class PhaseEvalMetrics(StrictArtifact):
    phase: str
    evaluator: str
    root: str
    labels_root: str
    status: EvalStatus
    required_artifacts: list[str]
    summary: EvalSummary
    metrics: dict[str, EvalMetric] = Field(default_factory=dict)
    clips: list[EvalClipResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class PipelineStageRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: str
    status: str
    real_model: bool
    source_mode: str
    produced_artifacts: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    wall_seconds: NonNegativeFiniteFloat | None = None


class PipelineRun(StrictArtifact):
    artifact_type: Literal["racketsport_pipeline_run"]
    clip: str
    requested_stage: str
    status: str
    run_dir: str
    inputs_dir: str
    stages: list[PipelineStageRun] = Field(default_factory=list)
    review_artifacts: dict[str, Any] = Field(default_factory=dict)
    readiness: dict[str, Any] = Field(default_factory=dict)


ARTIFACT_MODELS: dict[str, type[BaseModel]] = {
    "capture_sidecar": CaptureSidecar,
    "court_calibration": CourtCalibration,
    "court_line_evidence": CourtLineEvidence,
    "court_keypoints": CourtKeypoints,
    "court_lock": CourtLock,
    "court_zones": CourtZones,
    "net_plane": NetPlane,
    "tracks": Tracks,
    "placement": PlacementArtifact,
    "sam3d_keypoints_2d": Sam3DKeypoints2D,
    "player_ground": PlayerGroundArtifact,
    "court_calls": CallsArtifact,
    "drift_log": DriftLog,
    "person_ground_truth": PersonGroundTruth,
    "cvat_video_annotations": CvatVideoAnnotations,
    "on_device_person_tracks": OnDevicePersonTracks,
    "on_device_person_timing": OnDevicePersonTiming,
    "mobile_person_tracking_metrics": MobilePersonTrackingMetrics,
    "smpl_motion": SmplMotion,
    "skeleton3d": Skeleton3D,
    "body_compute_execution": BodyComputeExecution,
    "body_serialization_timing": BodySerializationTiming,
    "body_stage_phase_timing": BodyStagePhaseTiming,
    "body_mesh": BodyMesh,
    "body_mesh_readiness": BodyMeshReadiness,
    "ball_track": BallTrack,
    "ball_arc_render": BallArcRender,
    "racketsport_ball_candidates": BallCandidates,
    "ball_line_calls": BallLineCalls,
    "ball_model_runtime_profile": BallModelRuntimeProfile,
    "contact_window_candidates": ContactWindowCandidates,
    "contact_window_review": ContactWindowReview,
    "racket_candidates": RacketCandidates,
    "racket_pose": RacketPose,
    "racket_pose_readiness": RacketPoseReadiness,
    "racket_promotion_audit": RacketPromotionAudit,
    "contact_windows": ContactWindows,
    "racket_sport_metrics": RacketSportMetrics,
    "habit_report": HabitReport,
    "coach_report": HabitReport,
    "replay_scene": ReplayScene,
    "replay_viewer_manifest": ReplayViewerManifest,
    "virtual_world": VirtualWorld,
    "physics_refinement": PhysicsRefinement,
    "serving_manifest": ServingManifest,
    "drill_report": DrillReport,
    "phase_eval_metrics": PhaseEvalMetrics,
    "pipeline_run": PipelineRun,
}


def validate_artifact_file(artifact: str, path: str | Path) -> BaseModel:
    model = ARTIFACT_MODELS[artifact]
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return model.model_validate(payload)


def load_ball_candidates_file(path: str | Path) -> BallCandidates:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return BallCandidates.model_validate(payload)
