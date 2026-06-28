from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Vector2 = list[float]
Vector3 = list[float]
Matrix3 = list[list[float]]


class StrictArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]


class CaptureQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    grade: Literal["good", "warn", "poor"]
    reasons: list[str] = Field(default_factory=list)


class CameraIntrinsics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float] = Field(default_factory=list)
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
    orientation: Literal["landscape"]
    locked: LockedCapture
    intrinsics: CameraIntrinsics
    arkit_camera_pose: RigidPose | None = None
    court_plane: Plane | None = None
    manual_court_taps: list[Vector2] = Field(default_factory=list)
    gravity: Vector3
    lidar_depth_refs: list[str] = Field(default_factory=list)
    ondevice_pose_track: str | None = None
    capture_quality: CaptureQuality


class ReprojectionError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    median: float
    p95: float


class CourtExtrinsics(RigidPose):
    camera_height_m: float


class CourtCalibration(StrictArtifact):
    sport: Literal["pickleball", "tennis"]
    homography: Matrix3
    intrinsics: CameraIntrinsics
    extrinsics: CourtExtrinsics
    reprojection_error_px: ReprojectionError
    capture_quality: CaptureQuality
    image_pts: list[Vector2]
    world_pts: list[Vector3]


class CourtZones(StrictArtifact):
    zones: dict[str, list[Vector2]]


class NetPlane(StrictArtifact):
    plane: Plane
    endpoints: tuple[Vector3, Vector3]
    center_height_in: float
    post_height_in: float


class ResidualSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean: float
    p95: float


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
    mean_residual_px: float
    p95_residual_px: float
    temporal_stability_px: float
    auto_calibration_ready: bool
    reasons: list[str] = Field(default_factory=list)


class CourtLineEvidence(StrictArtifact):
    sport: Literal["pickleball", "tennis"]
    source: str
    line_observations: list[CourtLineObservation] = Field(default_factory=list)
    keypoint_observations: list[CourtKeypointObservation] = Field(default_factory=list)
    net_observations: list[NetLineObservation] = Field(default_factory=list)
    aggregate: CourtLineEvidenceAggregate


class TrackFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float
    bbox: tuple[float, float, float, float]
    world_xy: Vector2
    conf: float


class PlayerTrack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    side: str
    role: str
    frames: list[TrackFrame]


class TimeSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t0: float
    t1: float


class Tracks(StrictArtifact):
    fps: float
    players: list[PlayerTrack]
    rally_spans: list[TimeSpan] = Field(default_factory=list)


class FootContact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: bool
    right: bool


class SmplFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float
    global_orient: list[float]
    body_pose: list[float]
    left_hand_pose: list[float] = Field(default_factory=list)
    right_hand_pose: list[float] = Field(default_factory=list)
    transl_world: Vector3
    joints_world: list[Vector3]
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
    fps: float
    world_frame: Literal["court_Z0"]
    players: list[SmplPlayer]


class SkeletonFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float
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

    t: float
    xy: Vector2
    conf: float
    visible: bool
    world_xyz: Vector3 | None = None
    spin_rpm: float | None = None
    approx: bool = False


class Bounce(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float
    world_xy: Vector2


class BallTrack(StrictArtifact):
    fps: float
    source: Literal["tracknet", "tap"]
    frames: list[BallFrame]
    bounces: list[Bounce] = Field(default_factory=list)


class SE3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    R: Matrix3
    t: Vector3

    @field_validator("R")
    @classmethod
    def _must_be_rotation_matrix(cls, value: Matrix3) -> Matrix3:
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("pose_se3.R must be a 3x3 matrix")
        return value

    @field_validator("t")
    @classmethod
    def _must_be_translation_vector(cls, value: Vector3) -> Vector3:
        if len(value) != 3:
            raise ValueError("pose_se3.t must be a 3-vector")
        return value


class RacketFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float
    pose_se3: SE3
    conf: float


class RacketContact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float
    contact_point_face_cm: Vector2
    face_normal: Vector3
    conf: float

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
        has_named_dims = {"length", "width"}.issubset(value)
        has_short_dims = {"h", "w"}.issubset(value)
        if not has_named_dims and not has_short_dims:
            raise ValueError("paddle_dims_in must include length/width or h/w")
        if any(dim <= 0 for dim in value.values()):
            raise ValueError("paddle_dims_in values must be positive")
        return value


class RacketPose(StrictArtifact):
    fps: float
    players: list[RacketPlayer]


class EventSources(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: float
    wrist_vel: float
    ball_inflection: float


class ContactWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t0: float
    t1: float
    importance: float


class ContactEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["contact", "bounce", "net_cross"]
    t: float
    frame: int
    player_id: int | None = None
    confidence: float
    sources: EventSources
    window: ContactWindow


class ContactWindows(StrictArtifact):
    events: list[ContactEvent]


class MetricValue(BaseModel):
    model_config = ConfigDict(extra="allow")

    value: Any
    conf: float
    gated: bool | None = None


class ShotMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float
    type: str
    type_conf: float
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
    "smpl_motion": SmplMotion,
    "skeleton3d": Skeleton3D,
    "ball_track": BallTrack,
    "racket_pose": RacketPose,
    "contact_windows": ContactWindows,
    "racket_sport_metrics": RacketSportMetrics,
    "habit_report": HabitReport,
    "coach_report": HabitReport,
    "replay_scene": ReplayScene,
    "drill_report": DrillReport,
    "phase_eval_metrics": PhaseEvalMetrics,
}


def validate_artifact_file(artifact: str, path: str | Path) -> BaseModel:
    model = ARTIFACT_MODELS[artifact]
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return model.model_validate(payload)
