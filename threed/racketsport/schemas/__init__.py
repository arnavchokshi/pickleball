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


class RacketPlayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    paddle_dims_in: dict[str, float]
    frames: list[RacketFrame]
    contacts: list[RacketContact] = Field(default_factory=list)


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


ARTIFACT_MODELS: dict[str, type[BaseModel]] = {
    "capture_sidecar": CaptureSidecar,
    "court_calibration": CourtCalibration,
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
}


def validate_artifact_file(artifact: str, path: str | Path) -> BaseModel:
    model = ARTIFACT_MODELS[artifact]
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return model.model_validate(payload)

