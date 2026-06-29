from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("value must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("value must be a finite number")
    return number


FiniteFloat = Annotated[float, BeforeValidator(_finite_number)]
Vector2 = Annotated[list[FiniteFloat], Field(min_length=2, max_length=2)]
Vector3 = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
MatrixRow3 = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
Matrix3 = Annotated[list[MatrixRow3], Field(min_length=3, max_length=3)]


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


class CaptureSidecar(StrictArtifact):
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
    locked: LockedCapture
    intrinsics: CameraIntrinsics
    arkit_camera_pose: RigidPose | None = None
    court_plane: Plane | None = None
    manual_court_taps: list[Vector2] = Field(default_factory=list)
    gravity: Vector3
    lidar_depth_refs: list[str] = Field(default_factory=list)
    ondevice_pose_track: str | None = None
    capture_quality: CaptureQuality

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


class CourtCalibration(StrictArtifact):
    sport: Literal["pickleball", "tennis"]
    homography: Matrix3
    intrinsics: CameraIntrinsics
    image_size: tuple[int, int] | None = None
    extrinsics: CourtExtrinsics
    reprojection_error_px: ReprojectionError
    capture_quality: CaptureQuality
    image_pts: list[Vector2] = Field(min_length=4)
    world_pts: list[Vector3] = Field(min_length=4)

    @model_validator(mode="after")
    def _point_lists_must_be_paired(self) -> CourtCalibration:
        if len(self.image_pts) != len(self.world_pts):
            raise ValueError("image_pts and world_pts must have the same length")
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


class TrackFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: FiniteFloat = Field(ge=0.0)
    bbox: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    world_xy: Vector2
    conf: FiniteFloat = Field(ge=0.0, le=1.0)

    @field_validator("bbox")
    @classmethod
    def _bbox_must_have_positive_extent(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        _, _, width, height = value
        if width <= 0.0 or height <= 0.0:
            raise ValueError("bbox width and height must be positive")
        return value


class PlayerTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    side: str
    role: str
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


class SmplFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float = Field(ge=0.0)
    global_orient: list[float]
    body_pose: list[float]
    left_hand_pose: list[float] = Field(default_factory=list)
    right_hand_pose: list[float] = Field(default_factory=list)
    transl_world: Vector3
    joints_world: list[Vector3]
    mesh_vertices_world: list[Vector3] = Field(default_factory=list)
    joint_conf: list[float]
    foot_contact: FootContact
    grf: list[Vector3] | None = None


class SmplPlayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    betas: list[float]
    frames: list[SmplFrame]
    skate_free: bool
    physics: str


class SmplMotion(StrictArtifact):
    model: Literal["smpl", "smplx"]
    fps: float = Field(gt=0.0)
    world_frame: Literal["court_Z0"]
    players: list[SmplPlayer]


class SkeletonFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float = Field(ge=0.0)
    joints_world: list[Vector3]
    joint_conf: list[float]


class SkeletonPlayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    frames: list[SkeletonFrame]


class Skeleton3D(StrictArtifact):
    joint_names: list[str]
    preview_only: bool
    players: list[SkeletonPlayer]

    @field_validator("preview_only")
    @classmethod
    def _must_be_preview_only(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("skeleton3d is preview/triggering only")
        return value


class BallFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: FiniteFloat = Field(ge=0.0)
    xy: Vector2
    conf: FiniteFloat = Field(ge=0.0, le=1.0)
    visible: bool
    world_xyz: Vector3 | None = None
    spin_rpm: FiniteFloat | None = None
    approx: bool = False


class Bounce(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float = Field(ge=0.0)
    world_xy: Vector2


class BallTrack(StrictArtifact):
    fps: float = Field(gt=0.0)
    source: Literal["tracknet", "tap", "pbmat", "totnet"]
    frames: list[BallFrame]
    bounces: list[Bounce] = Field(default_factory=list)


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

    audio: float
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

    type: Literal["contact", "bounce", "net_cross"]
    t: float = Field(ge=0.0)
    frame: int = Field(ge=0)
    player_id: int | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    sources: EventSources
    window: ContactWindow


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
    type: Literal["contact", "bounce", "net_cross"]
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
    physics_refinement_url: str | None = None
    contact_windows_url: str | None = None
    label_overlays: list[ReplayViewerLabelOverlay] = Field(default_factory=list)
    annotation_sources: list[ReplayViewerAnnotationSource] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class VirtualWorldNet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoints: list[Vector3]
    center_height_m: float
    post_height_m: float


class VirtualWorldCourt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sport: Literal["pickleball", "tennis"]
    coordinate_frame: str
    length_m: float
    width_m: float
    line_segments: dict[str, list[Vector3]]
    net: VirtualWorldNet


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


class VirtualWorldPlayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    side: str | None = None
    role: str | None = None
    representation: Literal["track_only", "joints", "mesh"]
    frames: list[VirtualWorldPlayerFrame]


class VirtualWorldBallFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float
    xy: Vector2
    conf: float
    visible: bool
    world_xyz: Vector3 | None = None
    approx: bool = False


class VirtualWorldBall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["tracknet", "tap", "pbmat", "totnet"] | None = None
    frames: list[VirtualWorldBallFrame] = Field(default_factory=list)


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


class VirtualWorldPaddle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_id: int
    paddle_dims_in: dict[str, float]
    frames: list[VirtualWorldPaddleFrame]


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
    warnings: list[str] = Field(default_factory=list)


class VirtualWorld(StrictArtifact):
    artifact_type: Literal["racketsport_virtual_world"]
    world_frame: Literal["court_Z0"]
    fps: float
    court: VirtualWorldCourt
    players: list[VirtualWorldPlayer]
    ball: VirtualWorldBall
    paddles: list[VirtualWorldPaddle]
    summary: VirtualWorldSummary


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


ARTIFACT_MODELS: dict[str, type[BaseModel]] = {
    "capture_sidecar": CaptureSidecar,
    "court_calibration": CourtCalibration,
    "court_line_evidence": CourtLineEvidence,
    "court_zones": CourtZones,
    "net_plane": NetPlane,
    "tracks": Tracks,
    "person_ground_truth": PersonGroundTruth,
    "on_device_person_tracks": OnDevicePersonTracks,
    "on_device_person_timing": OnDevicePersonTiming,
    "mobile_person_tracking_metrics": MobilePersonTrackingMetrics,
    "smpl_motion": SmplMotion,
    "skeleton3d": Skeleton3D,
    "body_compute_execution": BodyComputeExecution,
    "body_mesh_readiness": BodyMeshReadiness,
    "ball_track": BallTrack,
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
}


def validate_artifact_file(artifact: str, path: str | Path) -> BaseModel:
    model = ARTIFACT_MODELS[artifact]
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return model.model_validate(payload)
