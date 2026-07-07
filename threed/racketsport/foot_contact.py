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
FOOT_CONTACT_PHASES_ARTIFACT_TYPE = "foot_contact_phases"
FOOT_CONTACT_PHASES_SCHEMA_VERSION = 1
BODY_SKELETON_DIRECT_SOURCE_KIND = "body_skeleton_direct"
BODY_CONTACT_CONFIDENCE_MIN = 0.90
BODY_CONTACT_CONFIDENCE_FORMULA = (
    "confident iff foot in {left,right}, min_confidence >= 0.90, "
    "source_phase_foot agrees with foot when present, "
    "assignment_evidence.body_detector_agreement >= 0.90, required quality fields present, "
    "no simultaneous confident opposite-foot single overlaps the same frame, "
    "and no independent rejection reason"
)
INDEPENDENT_BODY_PHASE_REJECTION_REASONS = {
    "unknown_or_invalid_foot",
    "source_phase_foot_mismatch",
    "missing_confidence_fields",
    "low_body_contact_confidence",
    "low_body_detector_agreement",
    "phase_penetrates_ground",
    "simultaneous_bilateral_contact",
    "weak_bilateral_unknown_foot",
    "weak_phase",
    "demoted_phase",
}


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


def build_body_skeleton_foot_contact_phases(
    skeleton_payload: Mapping[str, Any],
    *,
    clip: str | None = None,
    thresholds: ContactThresholds | None = None,
    confidence_min: float = BODY_CONTACT_CONFIDENCE_MIN,
    court_z_m: float = 0.0,
) -> dict[str, Any]:
    """Build schema-compatible confident per-foot phases from BODY skeleton joints.

    Confidence is a measurable phase statistic:
    ``min_confidence >= confidence_min`` and
    ``assignment_evidence.body_detector_agreement >= confidence_min``. The
    BODY detector supplies exact left/right phases, so detector agreement is
    1.0 by construction for valid phases. Frames/phases that miss those
    measurable bars are rejected with an explicit reason rather than fabricated
    into confident contact.
    """

    if confidence_min < 0.0 or confidence_min > 1.0:
        raise ValueError("confidence_min must be in [0, 1]")
    effective_thresholds = thresholds or body_skeleton_direct_contact_thresholds()
    try:
        frames, joint_names = contact_frames_from_skeleton3d(skeleton_payload)
        phases = detect_contact_phases(
            frames,
            joint_names=joint_names,
            thresholds=effective_thresholds,
            court_z_m=court_z_m,
        )
        metrics = measure_contact_metrics(frames, phases, joint_names=joint_names, court_z_m=court_z_m).to_dict()
    except (TypeError, ValueError) as exc:
        return _body_phase_payload(
            clip=clip or str(skeleton_payload.get("clip") or ""),
            thresholds=effective_thresholds,
            phases=[],
            rejected=[],
            metrics={"phase_metrics": []},
            confidence_min=confidence_min,
            status="no_confident_phases_invalid_skeleton",
            notes=[f"BODY skeleton contact phase production failed closed: {type(exc).__name__}: {exc}"],
        )

    metric_by_key = {
        _metric_phase_key(row): row
        for row in metrics.get("phase_metrics", [])
        if isinstance(row, Mapping)
    }
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for ordinal, phase in enumerate(phases):
        row = _body_phase_dict(phase, ordinal=ordinal, thresholds=effective_thresholds)
        reason = _body_phase_rejection_reason(
            row,
            metric=metric_by_key.get(_contact_phase_key(phase)),
            confidence_min=confidence_min,
        )
        if reason is None:
            accepted.append(row)
            continue
        rejected.append(_rejected_phase_payload(row, reason=reason))
    accepted, overlap_rejected = _demote_simultaneous_confident_singles(accepted)
    rejected.extend(overlap_rejected)
    return _body_phase_payload(
        clip=clip or str(skeleton_payload.get("clip") or ""),
        thresholds=effective_thresholds,
        phases=accepted,
        rejected=rejected,
        metrics=metrics,
        confidence_min=confidence_min,
        status="ran",
        notes=[],
    )


def build_body_skeleton_foot_contact_phases_from_gate_stream(
    gate_stream: Mapping[str, Any],
    *,
    clip: str | None = None,
    thresholds: ContactThresholds | None = None,
    confidence_min: float = BODY_CONTACT_CONFIDENCE_MIN,
) -> dict[str, Any]:
    """Build BODY-direct phases from the persisted BODY foot-lock gate stream.

    This is the post-BODY/pre-refine producer surface for existing runs: the
    stream is already emitted by the BODY skeleton contact detector and carries
    the per-foot confidence, height, speed, and independent demotion reason
    for every candidate phase. Over-threshold slide is deliberately not a
    rejection input here.
    """

    if confidence_min < 0.0 or confidence_min > 1.0:
        raise ValueError("confidence_min must be in [0, 1]")
    effective_thresholds = thresholds or body_skeleton_direct_contact_thresholds()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rows = gate_stream.get("phase_rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return _body_phase_payload(
            clip=clip or str(gate_stream.get("clip") or ""),
            thresholds=effective_thresholds,
            phases=[],
            rejected=[],
            metrics={"phase_metrics": []},
            confidence_min=confidence_min,
            status="no_confident_phases_missing_gate_stream",
            notes=["BODY gate stream missing phase_rows; producer failed closed."],
        )
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        phase = _phase_dict_from_gate_stream_row(row, ordinal=ordinal, thresholds=effective_thresholds)
        reason = _gate_stream_phase_rejection_reason(phase, source_row=row, confidence_min=confidence_min)
        if reason is None:
            accepted.append(phase)
            continue
        rejected.append(_rejected_phase_payload(phase, reason=reason))
    accepted, overlap_rejected = _demote_simultaneous_confident_singles(accepted)
    rejected.extend(overlap_rejected)
    metrics = _metrics_from_gate_stream_rows(rows)
    return _body_phase_payload(
        clip=clip or str(gate_stream.get("clip") or ""),
        thresholds=effective_thresholds,
        phases=accepted,
        rejected=rejected,
        metrics=metrics,
        confidence_min=confidence_min,
        status="ran",
        notes=[],
    )


def body_skeleton_direct_contact_thresholds() -> ContactThresholds:
    """Return the BODY-direct producer thresholds shared with foot-lock gate contact."""

    base = ContactThresholds()
    return ContactThresholds(
        enter_height_m=base.enter_height_m,
        exit_height_m=base.exit_height_m,
        enter_speed_mps=base.enter_speed_mps,
        exit_speed_mps=base.exit_speed_mps,
        min_confidence=base.min_confidence,
        min_phase_frames=base.min_phase_frames,
        low_foot_band_m=base.low_foot_band_m,
        split_speed_mps=base.enter_speed_mps,
    )


def contact_frames_from_skeleton3d(skeleton_payload: Mapping[str, Any]) -> tuple[list[SkeletonFrame], list[str]]:
    """Extract BODY contact frames from a ``skeleton3d.json`` payload."""

    joint_names = [str(name) for name in skeleton_payload.get("joint_names", [])]
    frames: list[SkeletonFrame] = []
    fps = _optional_float(skeleton_payload.get("fps")) or 30.0
    players = skeleton_payload.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return frames, joint_names
    for player in players:
        if not isinstance(player, Mapping):
            continue
        player_id = str(player.get("id", player.get("player_id", "unknown")))
        player_frames = player.get("frames")
        if not isinstance(player_frames, Sequence) or isinstance(player_frames, (str, bytes)):
            continue
        for ordinal, frame in enumerate(player_frames):
            if not isinstance(frame, Mapping):
                continue
            joints = frame.get("joints_world")
            if not isinstance(joints, Sequence) or isinstance(joints, (str, bytes)) or not joints:
                continue
            t = _optional_float(frame.get("t"))
            frame_index = _frame_index(frame, fallback=round((t or 0.0) * fps) if t is not None else ordinal)
            joint_conf = frame.get("joint_conf")
            frames.append(
                SkeletonFrame(
                    player_id=player_id,
                    frame_index=frame_index,
                    t=t,
                    joints_world=[_point3_list(joint, name="joints_world[]") for joint in joints],
                    joint_conf=[float(value) for value in joint_conf]
                    if isinstance(joint_conf, Sequence) and not isinstance(joint_conf, (str, bytes))
                    else None,
                    source=frame,
                )
            )
    return frames, joint_names


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


def _body_phase_payload(
    *,
    clip: str,
    thresholds: ContactThresholds,
    phases: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    confidence_min: float,
    status: str,
    notes: Sequence[str],
) -> dict[str, Any]:
    rejected_reasons = _counts(
        str(phase.get("rejection_reason") or phase.get("reason") or "unknown")
        for phase in rejected
    )
    return {
        "artifact_type": FOOT_CONTACT_PHASES_ARTIFACT_TYPE,
        "schema_version": FOOT_CONTACT_PHASES_SCHEMA_VERSION,
        "clip": clip,
        "source_kind": BODY_SKELETON_DIRECT_SOURCE_KIND,
        "source": BODY_SKELETON_DIRECT_SOURCE_KIND,
        "source_phase_count": len(phases) + len(rejected),
        "phase_count": len(phases),
        "rejected_phase_count": len(rejected),
        "phases": [dict(phase) for phase in phases],
        "rejected_phases": [dict(phase) for phase in rejected],
        "summary": {
            "status": status,
            "confidence_formula": BODY_CONTACT_CONFIDENCE_FORMULA,
            "confidence_min": float(confidence_min),
            "phase_frame_count": _phase_frame_total(phases),
            "rejected_phase_frame_count": _phase_frame_total(rejected),
            "rejected_reasons": rejected_reasons,
            "max_candidate_phase_slide_m": _max_phase_slide_m(metrics),
            "candidate_phase_count": len(metrics.get("phase_metrics", []))
            if isinstance(metrics.get("phase_metrics"), list)
            else 0,
            "source_thresholds": thresholds.to_dict(),
            "notes": list(notes),
        },
        "policy": {
            "producer": BODY_SKELETON_DIRECT_SOURCE_KIND,
            "fail_closed_uncertain_frames": True,
            "rejects_on_gate_threshold": False,
            "protected_eval_labels_used": False,
        },
    }


def _body_phase_dict(phase: ContactPhase, *, ordinal: int, thresholds: ContactThresholds) -> dict[str, Any]:
    payload = phase.to_dict()
    payload["phase_id"] = (
        f"{phase.player_id}:{phase.foot}:{phase.start_frame_index}-{phase.end_frame_index}:{ordinal}"
    )
    payload["source"] = "body_foot_contact_detector"
    payload["source_kind"] = BODY_SKELETON_DIRECT_SOURCE_KIND
    payload["source_phase_foot"] = phase.foot
    payload["foot_assignment"] = "per_foot_body_contact"
    payload["weak"] = False
    payload["demoted"] = False
    payload["rejection_reason"] = None
    payload["source_thresholds"] = thresholds.to_dict()
    evidence = dict(payload.get("assignment_evidence") or {})
    evidence.update(
        {
            "body_detector_agreement": 1.0,
            "body_detector_exact_agreement": 1.0,
            "source_phase_foot": phase.foot,
            "support_frame_count": phase.frame_count,
        }
    )
    payload["assignment_evidence"] = evidence
    return payload


def _phase_dict_from_gate_stream_row(
    row: Mapping[str, Any],
    *,
    ordinal: int,
    thresholds: ContactThresholds,
) -> dict[str, Any]:
    start = int(row.get("start_frame_index", -1))
    end = int(row.get("end_frame_index", start))
    frame_indices = list(range(start, end + 1)) if start >= 0 and end >= start else []
    foot = str(row.get("foot"))
    player_id = str(row.get("player_id", "unknown"))
    phase_id = str(row.get("phase_id") or f"{player_id}:{foot}:{start}-{end}:{ordinal}")
    min_confidence = _optional_float(row.get("min_confidence"))
    max_height_m = _optional_float(row.get("max_height_m"))
    max_speed_mps = _optional_float(row.get("max_speed_mps"))
    source_phase_foot = str(row.get("source_phase_foot") or foot)
    source_evidence = row.get("assignment_evidence") if isinstance(row.get("assignment_evidence"), Mapping) else {}
    detector_agreement = _optional_float(source_evidence.get("body_detector_agreement"))
    if detector_agreement is None:
        detector_agreement = _optional_float(row.get("body_detector_agreement"))
    if detector_agreement is None and source_phase_foot == foot:
        detector_agreement = 1.0
    detector_exact_agreement = _optional_float(source_evidence.get("body_detector_exact_agreement"))
    if detector_exact_agreement is None:
        detector_exact_agreement = _optional_float(row.get("body_detector_exact_agreement"))
    if detector_exact_agreement is None:
        detector_exact_agreement = detector_agreement
    return {
        "phase_id": phase_id,
        "player_id": player_id,
        "foot": foot,
        "start_frame_index": start,
        "end_frame_index": end,
        "frame_indices": frame_indices,
        "frame_count": len(frame_indices),
        "start_time_s": row.get("start_time_s"),
        "end_time_s": row.get("end_time_s"),
        "anchor_position_xyz": list(row.get("anchor_position_xyz") or [0.0, 0.0, 0.0]),
        "max_height_m": max_height_m if max_height_m is not None else 0.0,
        "max_speed_mps": max_speed_mps if max_speed_mps is not None else 0.0,
        "min_confidence": min_confidence if min_confidence is not None else 0.0,
        "source": str(row.get("contact_source") or "body_foot_contact_detector"),
        "source_kind": BODY_SKELETON_DIRECT_SOURCE_KIND,
        "source_phase_foot": source_phase_foot,
        "foot_assignment": str(row.get("foot_assignment") or "per_foot_body_contact"),
        "weak": False,
        "demoted": False,
        "split": bool(row.get("split", False)),
        "split_reason": row.get("split_reason"),
        "rejection_reason": None,
        "source_thresholds": thresholds.to_dict(),
        "assignment_evidence": {
            "body_detector_agreement": detector_agreement,
            "body_detector_exact_agreement": detector_exact_agreement,
            "source_phase_foot": source_phase_foot,
            "support_frame_count": len(frame_indices),
            "gate_stream_phase_id": phase_id,
        },
    }


def _body_phase_rejection_reason(
    phase: Mapping[str, Any],
    *,
    metric: Mapping[str, Any] | None,
    confidence_min: float,
) -> str | None:
    foot = str(phase.get("foot"))
    if foot not in {"left", "right"}:
        return "unknown_or_invalid_foot"
    source_phase_foot = _phase_source_foot(phase)
    if source_phase_foot in {"left", "right"} and source_phase_foot != foot:
        return "source_phase_foot_mismatch"
    for field in ("min_confidence", "max_height_m", "max_speed_mps", "source_thresholds"):
        if field not in phase:
            return "missing_confidence_fields"
    confidence = _optional_float(phase.get("min_confidence"))
    if confidence is None or confidence < confidence_min:
        return "low_body_contact_confidence"
    evidence = phase.get("assignment_evidence") if isinstance(phase.get("assignment_evidence"), Mapping) else {}
    agreement = _optional_float(evidence.get("body_detector_agreement"))
    if agreement is None or agreement < confidence_min:
        return "low_body_detector_agreement"
    penetration_m = (
        float(metric.get("max_penetration_mm", 0.0)) / 1000.0
        if isinstance(metric, Mapping)
        else 0.0
    )
    if penetration_m > 0.0:
        return "phase_penetrates_ground"
    return None


def _gate_stream_phase_rejection_reason(
    phase: Mapping[str, Any],
    *,
    source_row: Mapping[str, Any],
    confidence_min: float,
) -> str | None:
    source_reason = source_row.get("rejection_reason")
    if source_reason and str(source_reason) in INDEPENDENT_BODY_PHASE_REJECTION_REASONS:
        return str(source_reason)
    if not source_reason and not bool(source_row.get("lock_metric_included", True)):
        return "not_lock_metric_included"
    return _body_phase_rejection_reason(phase, metric=None, confidence_min=confidence_min)


def _phase_source_foot(phase: Mapping[str, Any]) -> str | None:
    value = phase.get("source_phase_foot")
    if value is None:
        evidence = phase.get("assignment_evidence") if isinstance(phase.get("assignment_evidence"), Mapping) else {}
        value = evidence.get("source_phase_foot")
    if value is None:
        return None
    return str(value)


def _rejected_phase_payload(phase: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        **phase,
        "weak": True,
        "demoted": True,
        "rejection_reason": reason,
        "reason": reason,
    }


def _demote_simultaneous_confident_singles(
    phases: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_player_frame: dict[tuple[str, int], list[int]] = {}
    for index, phase in enumerate(phases):
        for frame_index in _phase_frame_indices(phase):
            by_player_frame.setdefault((str(phase.get("player_id")), frame_index), []).append(index)

    conflicted_indices: set[int] = set()
    for indices in by_player_frame.values():
        feet = {str(phases[index].get("foot")) for index in indices}
        if "left" in feet and "right" in feet:
            conflicted_indices.update(indices)
    if not conflicted_indices:
        return [dict(phase) for phase in phases], []

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, phase in enumerate(phases):
        if index in conflicted_indices:
            rejected.append(_rejected_phase_payload(phase, reason="simultaneous_bilateral_contact"))
        else:
            accepted.append(dict(phase))
    return accepted, rejected


def _phase_frame_indices(phase: Mapping[str, Any]) -> list[int]:
    frame_indices = phase.get("frame_indices")
    if isinstance(frame_indices, Sequence) and not isinstance(frame_indices, (str, bytes)):
        return [int(frame_index) for frame_index in frame_indices]
    start = int(phase.get("start_frame_index", -1))
    end = int(phase.get("end_frame_index", start))
    if start < 0 or end < start:
        return []
    return list(range(start, end + 1))


def _metrics_from_gate_stream_rows(rows: Sequence[Any]) -> dict[str, Any]:
    phase_metrics: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        slide_m = _optional_float(row.get("slide_m")) or 0.0
        phase_metrics.append(
            {
                "player_id": row.get("player_id"),
                "foot": row.get("foot"),
                "start_frame_index": row.get("start_frame_index"),
                "end_frame_index": row.get("end_frame_index"),
                "frame_count": row.get("frame_count"),
                "slide_mm": slide_m * 1000.0,
                "max_penetration_mm": 0.0,
                "anchor_position_xyz": row.get("anchor_position_xyz") or [0.0, 0.0, 0.0],
            }
        )
    return {"phase_metrics": phase_metrics}


def _contact_phase_key(phase: ContactPhase) -> tuple[str, str, int, int]:
    return (
        str(phase.player_id),
        str(phase.foot),
        int(phase.start_frame_index),
        int(phase.end_frame_index),
    )


def _metric_phase_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row.get("player_id", "unknown")),
        str(row.get("foot", "unknown")),
        int(row.get("start_frame_index", -1)),
        int(row.get("end_frame_index", -1)),
    )


def _phase_frame_total(phases: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for phase in phases:
        frame_indices = phase.get("frame_indices")
        if isinstance(frame_indices, Sequence) and not isinstance(frame_indices, (str, bytes)):
            total += len(frame_indices)
            continue
        total += int(phase.get("frame_count", 0) or 0)
    return total


def _max_phase_slide_m(metrics: Mapping[str, Any]) -> float:
    return max(
        (
            float(row.get("slide_mm", 0.0)) / 1000.0
            for row in metrics.get("phase_metrics", [])
            if isinstance(row, Mapping)
        ),
        default=0.0,
    )


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


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
                    "body_detector_agreement": 1.0,
                    "body_detector_exact_agreement": 1.0,
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


def _point3_list(values: Any, *, name: str) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 3:
        raise ValueError(f"{name} must be a 3-vector")
    return [float(values[0]), float(values[1]), float(values[2])]


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _frame_index(frame: Mapping[str, Any], *, fallback: int) -> int:
    for key in ("frame_idx", "frame_index"):
        if key in frame:
            try:
                return int(frame[key])
            except (TypeError, ValueError):
                break
    return int(fallback)


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
    "body_skeleton_direct_contact_thresholds",
    "build_body_skeleton_foot_contact_phases",
    "build_body_skeleton_foot_contact_phases_from_gate_stream",
    "contact_frames_from_skeleton3d",
    "detect_contact_phases",
    "foot_contact_point",
    "measure_contact_metrics",
    "resolve_foot_joint_indices",
]
