"""Foot-ground contact detection and foot-slide measurement.

The court/world frame uses ``Z=0`` as the floor. Contact detection is based on
foot height above that floor plus low world-frame horizontal velocity with
hysteresis. The defaults are deliberately centimeter-scale because monocular
world joints in this repo have visible foot jitter: enter at 6 cm, stay until
10 cm, enter speed below 0.75 m/s, exit speed below 1.25 m/s. Those values are
loose enough to measure existing slide honestly rather than hide it by calling
noisy stance feet airborne.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from threed.racketsport.external_gt_body_prediction_schema import MHR70_JOINT_NAMES


FootName = str


@dataclass(frozen=True)
class ContactThresholds:
    enter_height_m: float = 0.060
    exit_height_m: float = 0.100
    enter_speed_mps: float = 0.75
    exit_speed_mps: float = 1.25
    min_confidence: float = 0.20
    min_phase_frames: int = 2
    low_foot_band_m: float = 0.025
    split_speed_mps: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


@dataclass(frozen=True)
class SkeletonFrame:
    player_id: str | int
    frame_index: int
    t: float | None
    joints_world: list[list[float]]
    joint_conf: list[float] | None = None
    source: Mapping[str, object] | None = None


@dataclass(frozen=True)
class FootJointIndices:
    left: tuple[int, ...]
    right: tuple[int, ...]

    def for_foot(self, foot: FootName) -> tuple[int, ...]:
        if foot == "left":
            return self.left
        if foot == "right":
            return self.right
        raise ValueError(f"unknown foot: {foot}")

    def all(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys((*self.left, *self.right)))


@dataclass(frozen=True)
class FootObservation:
    player_id: str | int
    foot: FootName
    frame_index: int
    t: float | None
    position_xyz: tuple[float, float, float]
    height_m: float
    speed_mps: float
    confidence: float


@dataclass(frozen=True)
class ContactPhase:
    player_id: str | int
    foot: FootName
    frame_indices: tuple[int, ...]
    start_time_s: float | None
    end_time_s: float | None
    anchor_position_xyz: tuple[float, float, float]
    max_height_m: float
    max_speed_mps: float
    min_confidence: float
    source: str = "body_foot_contact_detector"
    source_phase_foot: str | None = None
    foot_assignment: str = "per_foot_body_contact"
    weak: bool = False
    demoted: bool = False
    split: bool = False
    split_reason: str | None = None
    rejection_reason: str | None = None
    source_thresholds: Mapping[str, Any] | None = None
    assignment_evidence: Mapping[str, Any] | None = None

    @property
    def start_frame_index(self) -> int:
        return self.frame_indices[0]

    @property
    def end_frame_index(self) -> int:
        return self.frame_indices[-1]

    @property
    def frame_count(self) -> int:
        return len(self.frame_indices)

    def to_dict(self) -> dict[str, object]:
        return {
            "player_id": self.player_id,
            "foot": self.foot,
            "start_frame_index": self.start_frame_index,
            "end_frame_index": self.end_frame_index,
            "frame_indices": list(self.frame_indices),
            "frame_count": self.frame_count,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "anchor_position_xyz": list(self.anchor_position_xyz),
            "max_height_m": self.max_height_m,
            "max_speed_mps": self.max_speed_mps,
            "min_confidence": self.min_confidence,
            "source": self.source,
            "source_phase_foot": self.source_phase_foot if self.source_phase_foot is not None else self.foot,
            "foot_assignment": self.foot_assignment,
            "weak": self.weak,
            "demoted": self.demoted,
            "split": self.split,
            "split_reason": self.split_reason,
            "rejection_reason": self.rejection_reason,
            "source_thresholds": dict(self.source_thresholds or {}),
            "assignment_evidence": dict(self.assignment_evidence or {}),
        }


@dataclass(frozen=True)
class FootPhaseMetric:
    player_id: str | int
    foot: FootName
    start_frame_index: int
    end_frame_index: int
    frame_count: int
    slide_mm: float
    max_penetration_mm: float
    anchor_position_xyz: tuple[float, float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "player_id": self.player_id,
            "foot": self.foot,
            "start_frame_index": self.start_frame_index,
            "end_frame_index": self.end_frame_index,
            "frame_count": self.frame_count,
            "slide_mm": self.slide_mm,
            "max_penetration_mm": self.max_penetration_mm,
            "anchor_position_xyz": list(self.anchor_position_xyz),
        }


@dataclass(frozen=True)
class PlayerContactSummary:
    phase_count: int
    contact_frame_count: int
    median_slide_mm: float
    p95_slide_mm: float
    max_slide_mm: float
    max_penetration_mm: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class FootPenetrationSummary:
    max_penetration_mm: float
    penetrating_frame_count: int
    frame_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class ContactMetrics:
    phase_metrics: list[FootPhaseMetric]
    summary_by_player: dict[str | int, PlayerContactSummary]
    penetration: FootPenetrationSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "phase_metrics": [metric.to_dict() for metric in self.phase_metrics],
            "summary_by_player": {
                str(player_id): summary.to_dict() for player_id, summary in self.summary_by_player.items()
            },
            "penetration": self.penetration.to_dict(),
        }


_LEFT_NAMES = ("left_ankle", "left_big_toe", "left_big_toe_tip", "left_small_toe", "left_small_toe_tip", "left_heel")
_RIGHT_NAMES = (
    "right_ankle",
    "right_big_toe",
    "right_big_toe_tip",
    "right_small_toe",
    "right_small_toe_tip",
    "right_heel",
)


def resolve_foot_joint_indices(joint_names: Sequence[str], *, joint_count: int) -> FootJointIndices:
    names = tuple(joint_names)
    if joint_count == len(MHR70_JOINT_NAMES):
        names = MHR70_JOINT_NAMES
    if len(names) < joint_count:
        raise ValueError(f"joint_names has {len(names)} entries for {joint_count} joints")

    left = _indices_for_names(names, _LEFT_NAMES)
    right = _indices_for_names(names, _RIGHT_NAMES)
    if not left or not right:
        raise ValueError("joint_names must include left and right ankle/toe/heel joints")
    return FootJointIndices(left=left, right=right)


def detect_contact_phases(
    frames: Sequence[SkeletonFrame],
    *,
    joint_names: Sequence[str],
    thresholds: ContactThresholds = ContactThresholds(),
    court_z_m: float = 0.0,
) -> list[ContactPhase]:
    _validate_thresholds(thresholds)
    if not frames:
        return []
    indices = resolve_foot_joint_indices(joint_names, joint_count=len(frames[0].joints_world))
    phases: list[ContactPhase] = []
    for player_id, player_frames in _group_frames_by_player(frames).items():
        sorted_frames = sorted(player_frames, key=lambda frame: (frame.t if frame.t is not None else frame.frame_index))
        for foot in ("left", "right"):
            observations = _observations_for_foot(
                sorted_frames,
                foot=foot,
                indices=indices.for_foot(foot),
                thresholds=thresholds,
                court_z_m=court_z_m,
            )
            phases.extend(_phases_from_observations(player_id, foot, observations, thresholds))
    return phases


def measure_contact_metrics(
    frames: Sequence[SkeletonFrame],
    phases: Sequence[ContactPhase],
    *,
    joint_names: Sequence[str],
    court_z_m: float = 0.0,
) -> ContactMetrics:
    if not frames:
        return ContactMetrics(
            phase_metrics=[],
            summary_by_player={},
            penetration=FootPenetrationSummary(max_penetration_mm=0.0, penetrating_frame_count=0, frame_count=0),
        )
    indices = resolve_foot_joint_indices(joint_names, joint_count=len(frames[0].joints_world))
    frame_map = {(frame.player_id, frame.frame_index): frame for frame in frames}

    phase_metrics: list[FootPhaseMetric] = []
    for phase in phases:
        foot_indices = indices.for_foot(phase.foot)
        anchor = foot_contact_point(frame_map[(phase.player_id, phase.frame_indices[0])], foot_indices)
        max_slide_m = 0.0
        max_penetration_m = 0.0
        for frame_index in phase.frame_indices:
            frame = frame_map.get((phase.player_id, frame_index))
            if frame is None:
                continue
            position = foot_contact_point(frame, foot_indices)
            max_slide_m = max(max_slide_m, math.hypot(position[0] - anchor[0], position[1] - anchor[1]))
            max_penetration_m = max(max_penetration_m, max(0.0, court_z_m - position[2]))
        phase_metrics.append(
            FootPhaseMetric(
                player_id=phase.player_id,
                foot=phase.foot,
                start_frame_index=phase.start_frame_index,
                end_frame_index=phase.end_frame_index,
                frame_count=phase.frame_count,
                slide_mm=max_slide_m * 1000.0,
                max_penetration_mm=max_penetration_m * 1000.0,
                anchor_position_xyz=anchor,
            )
        )

    penetration = _penetration_summary(frames, indices, court_z_m=court_z_m)
    return ContactMetrics(
        phase_metrics=phase_metrics,
        summary_by_player=_summaries_by_player(phase_metrics),
        penetration=penetration,
    )


def foot_contact_point(
    frame: SkeletonFrame,
    foot_indices: Sequence[int],
    *,
    low_foot_band_m: float = 0.025,
) -> tuple[float, float, float]:
    points = [_point3(frame.joints_world[index], name=f"joints_world[{index}]") for index in foot_indices]
    min_z = min(point[2] for point in points)
    low_points = [point for point in points if point[2] <= min_z + low_foot_band_m]
    x = sum(point[0] for point in low_points) / len(low_points)
    y = sum(point[1] for point in low_points) / len(low_points)
    return (x, y, min_z)


def _observations_for_foot(
    frames: Sequence[SkeletonFrame],
    *,
    foot: FootName,
    indices: Sequence[int],
    thresholds: ContactThresholds,
    court_z_m: float,
) -> list[FootObservation]:
    positions = [foot_contact_point(frame, indices, low_foot_band_m=thresholds.low_foot_band_m) for frame in frames]
    speeds = [_horizontal_speed_mps(frames, positions, index) for index in range(len(frames))]
    observations: list[FootObservation] = []
    for frame, position, speed in zip(frames, positions, speeds, strict=True):
        observations.append(
            FootObservation(
                player_id=frame.player_id,
                foot=foot,
                frame_index=frame.frame_index,
                t=frame.t,
                position_xyz=position,
                height_m=position[2] - court_z_m,
                speed_mps=speed,
                confidence=_foot_confidence(frame, indices),
            )
        )
    return observations


def _phases_from_observations(
    player_id: str | int,
    foot: FootName,
    observations: Sequence[FootObservation],
    thresholds: ContactThresholds,
) -> list[ContactPhase]:
    phases: list[ContactPhase] = []
    active: list[FootObservation] = []
    in_contact = False
    for observation in observations:
        in_contact = _classify_observation(observation, previous_contact=in_contact, thresholds=thresholds)
        if in_contact:
            active.append(observation)
            continue
        if active:
            _append_phase(phases, player_id, foot, active, thresholds)
            active = []
    if active:
        _append_phase(phases, player_id, foot, active, thresholds)
    return phases


def _append_phase(
    phases: list[ContactPhase],
    player_id: str | int,
    foot: FootName,
    active: Sequence[FootObservation],
    thresholds: ContactThresholds,
) -> None:
    for segment, split_reason in _quality_segments(active, thresholds):
        if len(segment) < thresholds.min_phase_frames:
            continue
        anchor = segment[0].position_xyz
        phases.append(
            ContactPhase(
                player_id=player_id,
                foot=foot,
                frame_indices=tuple(observation.frame_index for observation in segment),
                start_time_s=segment[0].t,
                end_time_s=segment[-1].t,
                anchor_position_xyz=(anchor[0], anchor[1], 0.0),
                max_height_m=max(observation.height_m for observation in segment),
                max_speed_mps=max(observation.speed_mps for observation in segment),
                min_confidence=min(observation.confidence for observation in segment),
                split=split_reason is not None,
                split_reason=split_reason,
                source_thresholds=thresholds.to_dict(),
                assignment_evidence={
                    "support_frame_count": len(segment),
                    "source_phase_foot": foot,
                },
            )
        )


def _quality_segments(
    active: Sequence[FootObservation],
    thresholds: ContactThresholds,
) -> list[tuple[list[FootObservation], str | None]]:
    if len(active) < thresholds.min_phase_frames:
        return []
    split_speed = thresholds.split_speed_mps
    if split_speed is None:
        return [(list(active), None)]

    segments: list[list[FootObservation]] = []
    current: list[FootObservation] = []
    split_triggered = False
    for observation in active:
        if observation.speed_mps > split_speed + 1e-12:
            if current:
                segments.append(current)
                current = []
            split_triggered = True
            continue
        current.append(observation)
    if current:
        segments.append(current)
    split_reason = "internal_speed_inflection" if split_triggered else None
    return [(segment, split_reason) for segment in segments]


def _classify_observation(
    observation: FootObservation,
    *,
    previous_contact: bool,
    thresholds: ContactThresholds,
) -> bool:
    if observation.confidence < thresholds.min_confidence:
        return False
    if previous_contact:
        return observation.height_m <= thresholds.exit_height_m and observation.speed_mps <= thresholds.exit_speed_mps
    return observation.height_m <= thresholds.enter_height_m and observation.speed_mps <= thresholds.enter_speed_mps


def _horizontal_speed_mps(
    frames: Sequence[SkeletonFrame],
    positions: Sequence[tuple[float, float, float]],
    index: int,
) -> float:
    if len(frames) == 1:
        return 0.0
    if index == 0:
        other = 1
    elif index == len(frames) - 1:
        other = index - 1
    else:
        previous_speed = _pair_speed(frames[index - 1], positions[index - 1], frames[index], positions[index])
        next_speed = _pair_speed(frames[index], positions[index], frames[index + 1], positions[index + 1])
        return max(previous_speed, next_speed)
    return _pair_speed(frames[index], positions[index], frames[other], positions[other])


def _pair_speed(
    frame_a: SkeletonFrame,
    position_a: tuple[float, float, float],
    frame_b: SkeletonFrame,
    position_b: tuple[float, float, float],
) -> float:
    dt = _frame_dt_seconds(frame_a, frame_b)
    if dt <= 0:
        return 0.0
    return math.hypot(position_b[0] - position_a[0], position_b[1] - position_a[1]) / dt


def _frame_dt_seconds(frame_a: SkeletonFrame, frame_b: SkeletonFrame) -> float:
    if frame_a.t is not None and frame_b.t is not None:
        return abs(frame_b.t - frame_a.t)
    frame_delta = abs(frame_b.frame_index - frame_a.frame_index)
    return frame_delta / 30.0 if frame_delta else 0.0


def _penetration_summary(
    frames: Sequence[SkeletonFrame],
    indices: FootJointIndices,
    *,
    court_z_m: float,
) -> FootPenetrationSummary:
    max_penetration_m = 0.0
    penetrating_frames = 0
    foot_indices = indices.all()
    for frame in frames:
        frame_penetration_m = 0.0
        for index in foot_indices:
            point = _point3(frame.joints_world[index], name=f"joints_world[{index}]")
            frame_penetration_m = max(frame_penetration_m, court_z_m - point[2])
        if frame_penetration_m > 0:
            penetrating_frames += 1
            max_penetration_m = max(max_penetration_m, frame_penetration_m)
    return FootPenetrationSummary(
        max_penetration_mm=max_penetration_m * 1000.0,
        penetrating_frame_count=penetrating_frames,
        frame_count=len(frames),
    )


def _summaries_by_player(metrics: Sequence[FootPhaseMetric]) -> dict[str | int, PlayerContactSummary]:
    by_player: dict[str | int, list[FootPhaseMetric]] = {}
    for metric in metrics:
        by_player.setdefault(metric.player_id, []).append(metric)

    summaries: dict[str | int, PlayerContactSummary] = {}
    for player_id, player_metrics in by_player.items():
        slides = [metric.slide_mm for metric in player_metrics]
        penetrations = [metric.max_penetration_mm for metric in player_metrics]
        summaries[player_id] = PlayerContactSummary(
            phase_count=len(player_metrics),
            contact_frame_count=sum(metric.frame_count for metric in player_metrics),
            median_slide_mm=_percentile(slides, 50),
            p95_slide_mm=_percentile(slides, 95),
            max_slide_mm=max(slides) if slides else 0.0,
            max_penetration_mm=max(penetrations) if penetrations else 0.0,
        )
    return summaries


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (rank - lower)


def _group_frames_by_player(frames: Iterable[SkeletonFrame]) -> dict[str | int, list[SkeletonFrame]]:
    grouped: dict[str | int, list[SkeletonFrame]] = {}
    for frame in frames:
        grouped.setdefault(frame.player_id, []).append(frame)
    return grouped


def _indices_for_names(joint_names: Sequence[str], candidates: Sequence[str]) -> tuple[int, ...]:
    candidate_set = set(candidates)
    return tuple(index for index, name in enumerate(joint_names) if name in candidate_set)


def _foot_confidence(frame: SkeletonFrame, indices: Sequence[int]) -> float:
    if frame.joint_conf is None:
        return 1.0
    values = [frame.joint_conf[index] for index in indices if index < len(frame.joint_conf)]
    return min(values) if values else 1.0


def _point3(values: Sequence[float], *, name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must be a 3-vector")
    return (float(values[0]), float(values[1]), float(values[2]))


def _validate_thresholds(thresholds: ContactThresholds) -> None:
    if thresholds.enter_height_m < 0 or thresholds.exit_height_m < 0:
        raise ValueError("height thresholds must be non-negative")
    if thresholds.enter_speed_mps < 0 or thresholds.exit_speed_mps < 0:
        raise ValueError("speed thresholds must be non-negative")
    if thresholds.exit_height_m < thresholds.enter_height_m:
        raise ValueError("exit_height_m must be >= enter_height_m")
    if thresholds.exit_speed_mps < thresholds.enter_speed_mps:
        raise ValueError("exit_speed_mps must be >= enter_speed_mps")
    if thresholds.min_confidence < 0 or thresholds.min_confidence > 1:
        raise ValueError("min_confidence must be in [0, 1]")
    if thresholds.min_phase_frames < 1:
        raise ValueError("min_phase_frames must be >= 1")
    if thresholds.low_foot_band_m < 0:
        raise ValueError("low_foot_band_m must be non-negative")
    if thresholds.split_speed_mps is not None and thresholds.split_speed_mps <= 0:
        raise ValueError("split_speed_mps must be positive when provided")


__all__ = [
    "ContactMetrics",
    "ContactPhase",
    "ContactThresholds",
    "FootJointIndices",
    "FootObservation",
    "FootPenetrationSummary",
    "FootPhaseMetric",
    "PlayerContactSummary",
    "SkeletonFrame",
    "detect_contact_phases",
    "foot_contact_point",
    "measure_contact_metrics",
    "resolve_foot_joint_indices",
]
