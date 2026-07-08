"""Post-hoc stance-phase foot pinning for world-frame skeleton artifacts."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Mapping, MutableMapping, Sequence

from threed.racketsport.external_gt_body_prediction_schema import MHR70_JOINT_NAMES
from threed.racketsport.foot_contact import (
    ContactPhase,
    ContactThresholds,
    SkeletonFrame,
    detect_contact_phases,
    foot_contact_point,
    measure_contact_metrics,
    resolve_foot_joint_indices,
)
from threed.racketsport.skeleton3d import SAM3D_BODY_MHR70_SEMANTIC_MAP


VERSION = 1
CORE_BODY_SPEED_CLAMP_MPS = 3.0
R3_MAX_XY_CORRECTION_M = 0.02
STANCE_MAX_XY_CORRECTION_M = 0.30


@dataclass(frozen=True)
class FootPinSettings:
    enter_height_m: float = 0.060
    exit_height_m: float = 0.100
    enter_speed_mps: float = 0.75
    exit_speed_mps: float = 1.25
    min_phase_confidence: float = 0.20
    min_phase_frames: int = 2
    low_foot_band_m: float = 0.025
    taper_frames: int = 0
    max_correction_m: float = STANCE_MAX_XY_CORRECTION_M
    max_smoothing_correction_m: float = 0.0
    interpolate_between_stances: bool = False
    root_speed_clamp_mps: float = CORE_BODY_SPEED_CLAMP_MPS
    soft_anchor_enabled: bool = True
    soft_anchor_max_height_m: float = 0.075
    soft_anchor_max_speed_mps: float = 0.80
    soft_anchor_min_frames: int = 3
    soft_anchor_ramp_frames: int = 3

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)

    def contact_thresholds(self, *, min_confidence: float = 0.0) -> ContactThresholds:
        return ContactThresholds(
            enter_height_m=self.enter_height_m,
            exit_height_m=self.exit_height_m,
            enter_speed_mps=self.enter_speed_mps,
            exit_speed_mps=self.exit_speed_mps,
            min_confidence=min_confidence,
            min_phase_frames=self.min_phase_frames,
            low_foot_band_m=self.low_foot_band_m,
        )


@dataclass(frozen=True)
class FootPinResult:
    payload: dict[str, Any]
    audit: dict[str, Any]


@dataclass(frozen=True)
class _FrameRef:
    player_id: str
    frame_index: int
    frame: MutableMapping[str, Any]
    joints_world: list[list[float]]
    joint_conf: list[float] | None
    t: float | None
    order: int


@dataclass(frozen=True)
class _PinnedPhase:
    source: ContactPhase
    anchor_xy: tuple[float, float]
    kind: str = "stance"

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return self.source.frame_indices

    @property
    def foot(self) -> str:
        return self.source.foot

    @property
    def player_id(self) -> str:
        return str(self.source.player_id)


@dataclass(frozen=True)
class _Correction:
    dx: float
    dy: float
    weight: float
    capped: bool
    active_contacts: tuple[dict[str, Any], ...]

    @property
    def magnitude(self) -> float:
        contact_magnitudes = [
            math.hypot(float(contact["delta_xy_m"][0]), float(contact["delta_xy_m"][1]))
            for contact in self.active_contacts
            if isinstance(contact.get("delta_xy_m"), Sequence) and len(contact["delta_xy_m"]) >= 2
        ]
        return max(contact_magnitudes, default=math.hypot(self.dx, self.dy))


def apply_foot_pin_to_payload(
    payload: Mapping[str, Any],
    *,
    settings: FootPinSettings = FootPinSettings(),
    contact_phases: Sequence[ContactPhase] | None = None,
    audit_path: str | None = None,
) -> FootPinResult:
    """Return a corrected copy plus an always-on audit JSON payload."""

    _validate_settings(settings)
    corrected = copy.deepcopy(dict(payload))
    frames, joint_names = _frame_refs(corrected)
    before_frames, _unused_names = _skeleton_frames_from_payload(payload)
    if not frames:
        audit = _empty_audit(settings=settings, audit_path=audit_path)
        _attach_provenance(corrected, settings=settings, audit=audit, audit_path=audit_path)
        return FootPinResult(payload=corrected, audit=audit)

    detection_frames = [
        SkeletonFrame(
            player_id=frame.player_id,
            frame_index=frame.frame_index,
            t=frame.t,
            joints_world=copy.deepcopy(frame.joints_world),
            joint_conf=copy.deepcopy(frame.joint_conf),
        )
        for frame in frames
    ]
    candidate_phases = (
        list(contact_phases)
        if contact_phases is not None
        else detect_contact_phases(
            detection_frames,
            joint_names=joint_names,
            thresholds=settings.contact_thresholds(min_confidence=0.0),
        )
    )
    eligible_source_phases = [
        phase for phase in candidate_phases if phase.min_confidence >= settings.min_phase_confidence
    ]
    skipped_low_confidence = [
        phase for phase in candidate_phases if phase.min_confidence < settings.min_phase_confidence
    ]
    pinned_phases = _median_anchor_phases(detection_frames, eligible_source_phases, joint_names, settings=settings)
    soft_static_phases = _soft_static_anchor_phases(
        detection_frames,
        existing_phases=pinned_phases,
        joint_names=joint_names,
        settings=settings,
    )
    pinned_phases = [*pinned_phases, *soft_static_phases]
    cap_exceeded_skips: list[dict[str, Any]] = []
    raw_corrections = _stance_corrections(
        detection_frames, pinned_phases, joint_names, settings=settings, cap_exceeded_skips=cap_exceeded_skips
    )
    corrections = _interpolated_corrections(frames, raw_corrections, settings=settings, joint_names=joint_names)

    before_roots = _root_xy_by_player(frames, corrections=None, joint_names=joint_names)
    _apply_corrections(frames, corrections, joint_names=joint_names)
    after_roots = _root_xy_by_player(frames, corrections=None, joint_names=joint_names)

    after_frames, _unused = _skeleton_frames_from_payload(corrected)
    before_metrics = measure_contact_metrics(before_frames, eligible_source_phases, joint_names=joint_names)
    after_metrics = measure_contact_metrics(after_frames, eligible_source_phases, joint_names=joint_names)
    audit = _audit_payload(
        settings=settings,
        joint_names=joint_names,
        frames=frames,
        before_frames=before_frames,
        after_frames=after_frames,
        candidate_phases=candidate_phases,
        eligible_phases=eligible_source_phases,
        skipped_low_confidence=skipped_low_confidence,
        corrections=corrections,
        before_metrics=before_metrics.to_dict(),
        after_metrics=after_metrics.to_dict(),
        before_roots=before_roots,
        after_roots=after_roots,
        audit_path=audit_path,
        cap_exceeded_skips=cap_exceeded_skips,
        soft_static_phases=soft_static_phases,
    )
    _attach_provenance(corrected, settings=settings, audit=audit, audit_path=audit_path)
    return FootPinResult(payload=corrected, audit=audit)


def _frame_refs(payload: MutableMapping[str, Any]) -> tuple[list[_FrameRef], tuple[str, ...]]:
    joint_names = _joint_names(payload)
    refs: list[_FrameRef] = []
    players = payload.get("players")
    if not isinstance(players, list):
        return refs, joint_names
    for player_order, player in enumerate(players):
        if not isinstance(player, MutableMapping):
            continue
        player_id = str(player.get("id", player.get("player_id", player_order)))
        player_frames = player.get("frames")
        if not isinstance(player_frames, list):
            continue
        for order, frame in enumerate(player_frames):
            if not isinstance(frame, MutableMapping):
                continue
            joints = _copy_joints(frame.get("joints_world"))
            if joints is None or not _has_usable_foot_joints(joints, joint_names):
                continue
            refs.append(
                _FrameRef(
                    player_id=player_id,
                    frame_index=_frame_index(frame, fallback=order),
                    frame=frame,
                    joints_world=joints,
                    joint_conf=_copy_conf(frame.get("joint_conf")),
                    t=_maybe_float(frame.get("t")),
                    order=order,
                )
            )
    return refs, joint_names


def _skeleton_frames_from_payload(payload: Mapping[str, Any]) -> tuple[list[SkeletonFrame], tuple[str, ...]]:
    joint_names = _joint_names(payload)
    frames: list[SkeletonFrame] = []
    players = payload.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return frames, joint_names
    for player_order, player in enumerate(players):
        if not isinstance(player, Mapping):
            continue
        player_id = str(player.get("id", player.get("player_id", player_order)))
        player_frames = player.get("frames")
        if not isinstance(player_frames, Sequence) or isinstance(player_frames, (str, bytes)):
            continue
        for order, frame in enumerate(player_frames):
            if not isinstance(frame, Mapping):
                continue
            joints = _copy_joints(frame.get("joints_world"))
            if joints is None or not _has_usable_foot_joints(joints, joint_names):
                continue
            frames.append(
                SkeletonFrame(
                    player_id=player_id,
                    frame_index=_frame_index(frame, fallback=order),
                    t=_maybe_float(frame.get("t")),
                    joints_world=joints,
                    joint_conf=_copy_conf(frame.get("joint_conf")),
                )
            )
    return frames, joint_names


def _median_anchor_phases(
    frames: Sequence[SkeletonFrame],
    phases: Sequence[ContactPhase],
    joint_names: Sequence[str],
    *,
    settings: FootPinSettings,
) -> list[_PinnedPhase]:
    if not frames:
        return []
    indices = resolve_foot_joint_indices(joint_names, joint_count=len(frames[0].joints_world))
    frame_map = {(str(frame.player_id), frame.frame_index): frame for frame in frames}
    pinned: list[_PinnedPhase] = []
    for phase in phases:
        foot_indices = indices.for_foot(phase.foot)
        points = [
            foot_contact_point(frame, foot_indices, low_foot_band_m=settings.low_foot_band_m)
            for frame_index in phase.frame_indices
            if (frame := frame_map.get((str(phase.player_id), frame_index))) is not None
        ]
        if not points:
            continue
        pinned.append(_PinnedPhase(source=phase, anchor_xy=(float(median(p[0] for p in points)), float(median(p[1] for p in points)))))
    return pinned


def _soft_static_anchor_phases(
    frames: Sequence[SkeletonFrame],
    *,
    existing_phases: Sequence[_PinnedPhase],
    joint_names: Sequence[str],
    settings: FootPinSettings,
) -> list[_PinnedPhase]:
    if not settings.soft_anchor_enabled or not frames:
        return []
    indices = resolve_foot_joint_indices(joint_names, joint_count=len(frames[0].joints_world))
    occupied = {
        (phase.player_id, phase.foot, frame_index)
        for phase in existing_phases
        for frame_index in phase.frame_indices
    }
    stance_bounds: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for phase in existing_phases:
        stance_bounds.setdefault((phase.player_id, phase.foot), []).append((phase.source.start_frame_index, phase.source.end_frame_index))

    soft: list[_PinnedPhase] = []
    for player_id, player_frames in _frames_by_player(frames).items():
        ordered = sorted(player_frames, key=lambda frame: frame.frame_index)
        for foot in ("left", "right"):
            foot_indices = indices.for_foot(foot)
            candidates: list[tuple[int, tuple[float, float, float], float, float]] = []
            previous_point: tuple[float, float, float] | None = None
            previous_t: float | None = None
            for frame in ordered:
                point = foot_contact_point(frame, foot_indices, low_foot_band_m=settings.low_foot_band_m)
                frame_t = frame.t if frame.t is not None else frame.frame_index / 30.0
                speed = 0.0
                if previous_point is not None and previous_t is not None:
                    dt = max(float(frame_t) - previous_t, 1.0 / 30.0)
                    speed = math.hypot(point[0] - previous_point[0], point[1] - previous_point[1]) / dt
                previous_point = point
                previous_t = float(frame_t)
                if (str(player_id), foot, frame.frame_index) in occupied:
                    continue
                if not _has_bracketing_stances(stance_bounds.get((str(player_id), foot), []), frame.frame_index):
                    continue
                confidence = _foot_confidence(frame, foot_indices)
                if (
                    point[2] <= settings.soft_anchor_max_height_m
                    and speed <= settings.soft_anchor_max_speed_mps
                    and confidence >= settings.min_phase_confidence
                ):
                    candidates.append((frame.frame_index, point, speed, confidence))
            for run in _contiguous_runs([item[0] for item in candidates]):
                if len(run) < settings.soft_anchor_min_frames:
                    continue
                by_frame = {item[0]: item for item in candidates}
                points = [by_frame[idx][1] for idx in run]
                speeds = [by_frame[idx][2] for idx in run]
                confidences = [by_frame[idx][3] for idx in run]
                phase = ContactPhase(
                    player_id=str(player_id),
                    foot=foot,
                    frame_indices=tuple(run),
                    start_time_s=run[0] / 30.0,
                    end_time_s=run[-1] / 30.0,
                    anchor_position_xyz=(
                        float(median(point[0] for point in points)),
                        float(median(point[1] for point in points)),
                        float(median(point[2] for point in points)),
                    ),
                    max_height_m=max(point[2] for point in points),
                    max_speed_mps=max(speeds, default=0.0),
                    min_confidence=min(confidences, default=1.0),
                )
                soft.append(
                    _PinnedPhase(
                        source=phase,
                        anchor_xy=(float(phase.anchor_position_xyz[0]), float(phase.anchor_position_xyz[1])),
                        kind="soft_static",
                    )
                )
    return soft


def _stance_corrections(
    frames: Sequence[SkeletonFrame],
    phases: Sequence[_PinnedPhase],
    joint_names: Sequence[str],
    *,
    settings: FootPinSettings,
    cap_exceeded_skips: list[dict[str, Any]] | None = None,
) -> dict[tuple[str, int], _Correction]:
    if not frames or not phases:
        return {}
    indices = resolve_foot_joint_indices(joint_names, joint_count=len(frames[0].joints_world))
    frame_map = {(str(frame.player_id), frame.frame_index): frame for frame in frames}

    # Phase-level cap filter: a stance phase whose anchor demands a correction beyond
    # the in-stance cap at ANY of its frames is inconsistent evidence. Pinning only its
    # under-cap frames would create an on/off discontinuity INSIDE the stance (measured
    # live as a 0.32 m foot jump on IMG_1605), so the whole phase is dropped fail-closed
    # and recorded; its feet stay unpinned and the slide metrics report them honestly.
    effective_cap = _effective_max_correction_m(settings)
    kept_phases: list[_PinnedPhase] = []
    for phase in phases:
        max_magnitude = 0.0
        for frame_index in phase.frame_indices:
            frame = frame_map.get((phase.player_id, frame_index))
            if frame is None:
                continue
            current = foot_contact_point(
                frame,
                indices.for_foot(phase.foot),
                low_foot_band_m=settings.low_foot_band_m,
            )
            max_magnitude = max(
                max_magnitude,
                math.hypot(phase.anchor_xy[0] - current[0], phase.anchor_xy[1] - current[1]),
            )
        if max_magnitude > effective_cap + 1e-9:
            if cap_exceeded_skips is not None:
                cap_exceeded_skips.append(
                    {
                        "kind": "phase_skipped",
                        "player_id": phase.player_id,
                        "foot": phase.foot,
                        "start_frame_index": phase.source.start_frame_index,
                        "end_frame_index": phase.source.end_frame_index,
                        "magnitude_m": round(max_magnitude, 6),
                        "cap_m": round(effective_cap, 6),
                    }
                )
            continue
        kept_phases.append(phase)

    active_by_frame: dict[tuple[str, int], list[_PinnedPhase]] = {}
    for phase in kept_phases:
        for frame_index in phase.frame_indices:
            active_by_frame.setdefault((phase.player_id, frame_index), []).append(phase)

    corrections: dict[tuple[str, int], _Correction] = {}
    for key, active in active_by_frame.items():
        frame = frame_map.get(key)
        if frame is None:
            continue
        weighted_dx_by_foot: dict[str, float] = {}
        weighted_dy_by_foot: dict[str, float] = {}
        weight_sum_by_foot: dict[str, float] = {}
        contacts: list[dict[str, Any]] = []
        for phase in active:
            phase_weight = _phase_weight(phase, frame.frame_index, settings)
            if phase_weight <= 0:
                continue
            current = foot_contact_point(
                frame,
                indices.for_foot(phase.foot),
                low_foot_band_m=settings.low_foot_band_m,
            )
            dx = phase.anchor_xy[0] - current[0]
            dy = phase.anchor_xy[1] - current[1]
            weighted_dx_by_foot[phase.foot] = weighted_dx_by_foot.get(phase.foot, 0.0) + dx * phase_weight
            weighted_dy_by_foot[phase.foot] = weighted_dy_by_foot.get(phase.foot, 0.0) + dy * phase_weight
            weight_sum_by_foot[phase.foot] = weight_sum_by_foot.get(phase.foot, 0.0) + phase_weight
            contacts.append(
                {
                    "foot": phase.foot,
                    "start_frame_index": phase.source.start_frame_index,
                    "end_frame_index": phase.source.end_frame_index,
                    "anchor_xy": [phase.anchor_xy[0], phase.anchor_xy[1]],
                    "weight": phase_weight,
                    "min_confidence": phase.source.min_confidence,
                    "source": phase.kind,
                }
            )
        if not weight_sum_by_foot:
            continue
        effective_cap = _effective_max_correction_m(settings)
        foot_deltas: dict[str, tuple[float, float, bool]] = {}
        for foot, weight_sum in weight_sum_by_foot.items():
            strength = min(float(weight_sum), 1.0)
            dx = (weighted_dx_by_foot[foot] / weight_sum) * strength
            dy = (weighted_dy_by_foot[foot] / weight_sum) * strength
            magnitude = math.hypot(dx, dy)
            if magnitude > effective_cap + 1e-9:
                # Fail closed per foot/frame: a correction beyond the in-stance cap is
                # untrustworthy evidence, so no pin is applied there. The skip is
                # recorded in the audit instead of killing the whole BODY refine.
                if cap_exceeded_skips is not None:
                    cap_exceeded_skips.append(
                        {
                            "player_id": key[0],
                            "frame_index": key[1],
                            "foot": foot,
                            "magnitude_m": round(magnitude, 6),
                            "cap_m": round(effective_cap, 6),
                        }
                    )
                continue
            foot_deltas[foot] = (*_cap_xy(dx, dy, effective_cap),)
        enriched_contacts: list[dict[str, Any]] = []
        for contact in contacts:
            delta = foot_deltas.get(str(contact["foot"]))
            if delta is None:
                continue
            enriched = dict(contact)
            enriched["delta_xy_m"] = [delta[0], delta[1]]
            enriched["capped"] = bool(delta[2])
            enriched_contacts.append(enriched)
        if not enriched_contacts:
            continue
        dx = sum(float(contact["delta_xy_m"][0]) for contact in enriched_contacts) / len(enriched_contacts)
        dy = sum(float(contact["delta_xy_m"][1]) for contact in enriched_contacts) / len(enriched_contacts)
        capped = any(bool(contact["capped"]) for contact in enriched_contacts)
        corrections[key] = _Correction(
            dx=dx,
            dy=dy,
            weight=sum(weight_sum_by_foot.values()) / len(active),
            capped=capped,
            active_contacts=tuple(enriched_contacts),
        )
    return corrections


def _phase_weight(phase: _PinnedPhase, frame_index: int, settings: FootPinSettings) -> float:
    taper_frames = settings.soft_anchor_ramp_frames if phase.kind == "soft_static" else settings.taper_frames
    if taper_frames <= 0 or len(phase.frame_indices) <= 1:
        return 1.0
    try:
        position = phase.frame_indices.index(frame_index)
    except ValueError:
        return 0.0
    edge_distance = min(position, len(phase.frame_indices) - 1 - position)
    if edge_distance >= taper_frames:
        return 1.0
    if phase.kind == "soft_static":
        return min(1.0, float(edge_distance + 1) / float(max(taper_frames, 1)))
    return float(edge_distance + 1) / float(taper_frames + 1)


def _interpolated_corrections(
    frames: Sequence[_FrameRef],
    raw: Mapping[tuple[str, int], _Correction],
    *,
    settings: FootPinSettings,
    joint_names: Sequence[str],
) -> dict[tuple[str, int], _Correction]:
    if not raw:
        return {}
    return dict(raw)


def _apply_corrections(
    frames: Sequence[_FrameRef],
    corrections: Mapping[tuple[str, int], _Correction],
    *,
    joint_names: Sequence[str],
) -> None:
    for frame in frames:
        correction = corrections.get((frame.player_id, frame.frame_index))
        if correction is None or correction.magnitude == 0.0:
            continue
        _translate_stance_chains(frame.frame, correction.active_contacts, joint_names=joint_names)


def _translate_stance_chains(
    frame: MutableMapping[str, Any],
    active_contacts: Sequence[Mapping[str, Any]],
    *,
    joint_names: Sequence[str],
) -> None:
    joints = frame.get("joints_world")
    if not isinstance(joints, list):
        return
    translated = [[float(value) for value in joint] for joint in joints]
    deltas = [
        (str(contact.get("foot", "")), float(contact["delta_xy_m"][0]), float(contact["delta_xy_m"][1]))
        for contact in active_contacts
        if isinstance(contact.get("delta_xy_m"), Sequence) and len(contact["delta_xy_m"]) >= 2
    ]
    if not deltas:
        return
    shared_dx = sum(delta[1] for delta in deltas) / len(deltas)
    shared_dy = sum(delta[2] for delta in deltas) / len(deltas)
    for joint_idx in _lower_body_chain_indices(joint_names, joint_count=len(translated)):
        translated[joint_idx] = _translated_vec3(translated[joint_idx], dx=shared_dx, dy=shared_dy)
    for foot, dx, dy in deltas:
        residual_dx = dx - shared_dx
        residual_dy = dy - shared_dy
        if abs(residual_dx) <= 1e-12 and abs(residual_dy) <= 1e-12:
            continue
        for joint_idx in _stance_foot_indices(joint_names, foot=foot, joint_count=len(translated)):
            translated[joint_idx] = _translated_vec3(translated[joint_idx], dx=residual_dx, dy=residual_dy)
    frame["joints_world"] = translated


def _translated_vec3(value: Any, *, dx: float, dy: float) -> Any:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return value
    return [float(value[0]) + dx, float(value[1]) + dy, float(value[2]), *list(value[3:])]


def _audit_payload(
    *,
    settings: FootPinSettings,
    joint_names: Sequence[str],
    frames: Sequence[_FrameRef],
    before_frames: Sequence[SkeletonFrame],
    after_frames: Sequence[SkeletonFrame],
    candidate_phases: Sequence[ContactPhase],
    eligible_phases: Sequence[ContactPhase],
    skipped_low_confidence: Sequence[ContactPhase],
    corrections: Mapping[tuple[str, int], _Correction],
    before_metrics: dict[str, Any],
    after_metrics: dict[str, Any],
    before_roots: Mapping[str, list[tuple[int, tuple[float, float]]]],
    after_roots: Mapping[str, list[tuple[int, tuple[float, float]]]],
    audit_path: str | None,
    cap_exceeded_skips: Sequence[Mapping[str, Any]] = (),
    soft_static_phases: Sequence[_PinnedPhase] = (),
) -> dict[str, Any]:
    players = _player_stats(
        frames=frames,
        before_frames=before_frames,
        after_frames=after_frames,
        joint_names=joint_names,
        eligible_phases=eligible_phases,
        skipped_low_confidence=skipped_low_confidence,
        corrections=corrections,
        before_roots=before_roots,
        after_roots=after_roots,
        settings=settings,
    )
    before_phase_slides = [float(metric["slide_mm"]) for metric in before_metrics.get("phase_metrics", [])]
    after_phase_slides = [float(metric["slide_mm"]) for metric in after_metrics.get("phase_metrics", [])]
    return {
        "schema_version": 1,
        "artifact_type": "foot_pin_audit",
        "foot_pin_version": VERSION,
        "audit_path": audit_path,
        "settings": settings.to_dict(),
        "phase_detection": {
            "candidate_phase_count": len(candidate_phases),
            "confident_phase_count": len(eligible_phases),
            "skipped_low_confidence_phase_count": len(skipped_low_confidence),
            "phases": [_phase_dict(phase) for phase in eligible_phases],
            "skipped_low_confidence_phases": [_phase_dict(phase) for phase in skipped_low_confidence],
            "cap_exceeded_skips": [dict(event) for event in cap_exceeded_skips],
            "soft_static_phase_count": len(soft_static_phases),
            "soft_static_phases": [_pinned_phase_dict(phase) for phase in soft_static_phases],
        },
        "summary": {
            "stance_slide_before_mm": _distribution(before_phase_slides),
            "stance_slide_after_mm": _distribution(after_phase_slides),
            "total_phase_count": len(eligible_phases),
            "total_skipped_low_confidence_phases": len(skipped_low_confidence),
            "cap_exceeded_skip_count": len(cap_exceeded_skips),
            "total_corrected_frame_count": len([corr for corr in corrections.values() if corr.magnitude > 0.0]),
            "max_correction_m": max((corr.magnitude for corr in corrections.values()), default=0.0),
            "max_limb_length_delta_m": max(
                (float(stats["max_limb_length_delta_m"]) for stats in players.values()),
                default=0.0,
            ),
            "max_non_foot_joint_displacement_m": max(
                (float(stats["max_non_foot_joint_displacement_m"]) for stats in players.values()),
                default=0.0,
            ),
            "max_wrist_displacement_m": max(
                (float(stats["max_wrist_displacement_m"]) for stats in players.values()),
                default=0.0,
            ),
        },
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
        "players": players,
    }


def _player_stats(
    *,
    frames: Sequence[_FrameRef],
    before_frames: Sequence[SkeletonFrame],
    after_frames: Sequence[SkeletonFrame],
    joint_names: Sequence[str],
    eligible_phases: Sequence[ContactPhase],
    skipped_low_confidence: Sequence[ContactPhase],
    corrections: Mapping[tuple[str, int], _Correction],
    before_roots: Mapping[str, list[tuple[int, tuple[float, float]]]],
    after_roots: Mapping[str, list[tuple[int, tuple[float, float]]]],
    settings: FootPinSettings,
) -> dict[str, Any]:
    player_ids = sorted({frame.player_id for frame in frames})
    before_map = {(str(frame.player_id), frame.frame_index): frame for frame in before_frames}
    after_map = {(str(frame.player_id), frame.frame_index): frame for frame in after_frames}
    foot_indices = set(resolve_foot_joint_indices(joint_names, joint_count=len(frames[0].joints_world)).all()) if frames else set()
    bone_pairs = _bone_pairs(joint_names)
    wrist_indices = _wrist_indices(joint_names)
    stats: dict[str, Any] = {}
    for player_id in player_ids:
        player_frames = sorted([frame for frame in frames if frame.player_id == player_id], key=lambda item: item.frame_index)
        phase_count = sum(1 for phase in eligible_phases if str(phase.player_id) == player_id)
        skipped_count = sum(1 for phase in skipped_low_confidence if str(phase.player_id) == player_id)
        frame_corrections = [
            _frame_correction_dict(frame, corrections[(player_id, frame.frame_index)])
            for frame in player_frames
            if (player_id, frame.frame_index) in corrections
        ]
        max_non_foot = 0.0
        max_wrist = 0.0
        max_bone_delta = 0.0
        for frame in player_frames:
            before = before_map.get((player_id, frame.frame_index))
            after = after_map.get((player_id, frame.frame_index))
            if before is None or after is None:
                continue
            max_non_foot = max(max_non_foot, _max_joint_displacement(before.joints_world, after.joints_world, excluded=foot_indices))
            max_wrist = max(max_wrist, _max_joint_displacement(before.joints_world, after.joints_world, included=wrist_indices))
            max_bone_delta = max(max_bone_delta, _max_bone_length_delta(before.joints_world, after.joints_world, bone_pairs))
        before_root_stats = _root_motion_stats(before_roots.get(player_id, []), settings=settings)
        after_root_stats = _root_motion_stats(after_roots.get(player_id, []), settings=settings)
        stats[player_id] = {
            "phase_count": phase_count,
            "skipped_low_confidence_phase_count": skipped_count,
            "corrected_frame_count": sum(1 for item in frame_corrections if item["correction_m"] > 0.0),
            "capped_correction_frame_count": sum(1 for item in frame_corrections if item["capped"]),
            "max_correction_m": max((float(item["correction_m"]) for item in frame_corrections), default=0.0),
            "max_non_foot_joint_displacement_m": max_non_foot,
            "max_wrist_displacement_m": max_wrist,
            "max_limb_length_delta_m": max_bone_delta,
            "root_jitter_before": before_root_stats,
            "root_jitter_after": after_root_stats,
            "clamp_engagement_fraction_before": before_root_stats["clamp_engagement_fraction"],
            "clamp_engagement_fraction_after": after_root_stats["clamp_engagement_fraction"],
            "frame_corrections": frame_corrections,
        }
    return stats


def _root_xy_by_player(
    frames: Sequence[_FrameRef],
    *,
    corrections: Mapping[tuple[str, int], _Correction] | None,
    joint_names: Sequence[str],
) -> dict[str, list[tuple[int, tuple[float, float]]]]:
    out: dict[str, list[tuple[int, tuple[float, float]]]] = {}
    for frame in frames:
        xy = _root_xy(frame.frame, frame.joints_world, joint_names=joint_names)
        correction = corrections.get((frame.player_id, frame.frame_index)) if corrections is not None else None
        if correction is not None:
            xy = (xy[0] + correction.dx, xy[1] + correction.dy)
        out.setdefault(frame.player_id, []).append((frame.frame_index, xy))
    for items in out.values():
        items.sort(key=lambda item: item[0])
    return out


def _root_xy(frame: Mapping[str, Any], joints_world: Sequence[Sequence[float]], *, joint_names: Sequence[str]) -> tuple[float, float]:
    track = frame.get("track_world_xy")
    if isinstance(track, Sequence) and not isinstance(track, (str, bytes)) and len(track) >= 2:
        return (float(track[0]), float(track[1]))
    transl = frame.get("transl_world")
    if isinstance(transl, Sequence) and not isinstance(transl, (str, bytes)) and len(transl) >= 2:
        return (float(transl[0]), float(transl[1]))
    names = _effective_joint_names(joint_names)
    hip_indices = [index for index, name in enumerate(names) if name in {"left_hip", "right_hip"} and index < len(joints_world)]
    if hip_indices:
        return (
            sum(float(joints_world[index][0]) for index in hip_indices) / len(hip_indices),
            sum(float(joints_world[index][1]) for index in hip_indices) / len(hip_indices),
        )
    if joints_world:
        return (
            sum(float(point[0]) for point in joints_world) / len(joints_world),
            sum(float(point[1]) for point in joints_world) / len(joints_world),
        )
    return (0.0, 0.0)


def _root_motion_stats(samples: Sequence[tuple[int, tuple[float, float]]], *, settings: FootPinSettings) -> dict[str, float | int]:
    distances: list[float] = []
    for (_frame_a, xy_a), (_frame_b, xy_b) in zip(samples, samples[1:]):
        distances.append(math.hypot(xy_b[0] - xy_a[0], xy_b[1] - xy_a[1]))
    clamp_threshold = settings.root_speed_clamp_mps / 30.0
    return {
        "sample_count": len(distances),
        "p90_frame_displacement_m": _percentile(distances, 90),
        "p50_frame_displacement_m": _percentile(distances, 50),
        "max_frame_displacement_m": max(distances) if distances else 0.0,
        "clamp_engagement_fraction": (
            sum(1 for distance in distances if distance >= clamp_threshold - 1e-9) / len(distances)
            if distances
            else 0.0
        ),
    }


def _frame_correction_dict(frame: _FrameRef, correction: _Correction) -> dict[str, Any]:
    return {
        "frame_index": frame.frame_index,
        "t": frame.t,
        "delta_xy_m": [correction.dx, correction.dy],
        "correction_m": correction.magnitude,
        "weight": correction.weight,
        "capped": correction.capped,
        "active_contacts": list(correction.active_contacts),
    }


def _max_joint_displacement(
    before: Sequence[Sequence[float]],
    after: Sequence[Sequence[float]],
    *,
    excluded: set[int] | None = None,
    included: Sequence[int] | None = None,
) -> float:
    included_set = set(included) if included is not None else None
    excluded_set = excluded or set()
    max_distance = 0.0
    for index, (left, right) in enumerate(zip(before, after, strict=False)):
        if included_set is not None and index not in included_set:
            continue
        if index in excluded_set:
            continue
        max_distance = max(max_distance, _distance3(left, right))
    return max_distance


def _max_bone_length_delta(
    before: Sequence[Sequence[float]],
    after: Sequence[Sequence[float]],
    bone_pairs: Sequence[tuple[int, int]],
) -> float:
    max_delta = 0.0
    for left, right in bone_pairs:
        if left >= len(before) or right >= len(before) or left >= len(after) or right >= len(after):
            continue
        before_len = _distance3(before[left], before[right])
        after_len = _distance3(after[left], after[right])
        max_delta = max(max_delta, abs(after_len - before_len))
    return max_delta


def _bone_pairs(joint_names: Sequence[str]) -> list[tuple[int, int]]:
    names = _effective_joint_names(joint_names)
    index_by_name = {name: index for index, name in enumerate(names)}
    pairs = [
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
    ]
    return [(index_by_name[left], index_by_name[right]) for left, right in pairs if left in index_by_name and right in index_by_name]


def _wrist_indices(joint_names: Sequence[str]) -> tuple[int, ...]:
    names = _effective_joint_names(joint_names)
    return tuple(index for index, name in enumerate(names) if name in {"left_wrist", "right_wrist"})


def _frames_by_player(frames: Sequence[SkeletonFrame]) -> dict[str, list[SkeletonFrame]]:
    grouped: dict[str, list[SkeletonFrame]] = {}
    for frame in frames:
        grouped.setdefault(str(frame.player_id), []).append(frame)
    return grouped


def _contiguous_runs(indices: Sequence[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for idx in sorted(dict.fromkeys(int(value) for value in indices)):
        if not runs or idx != runs[-1][-1] + 1:
            runs.append([idx])
        else:
            runs[-1].append(idx)
    return runs


def _has_bracketing_stances(bounds: Sequence[tuple[int, int]], frame_index: int) -> bool:
    has_before = any(end < frame_index for _start, end in bounds)
    has_after = any(start > frame_index for start, _end in bounds)
    return has_before and has_after


def _foot_confidence(frame: SkeletonFrame, foot_indices: Sequence[int]) -> float:
    if frame.joint_conf is None:
        return 1.0
    values = [
        float(frame.joint_conf[idx])
        for idx in foot_indices
        if idx < len(frame.joint_conf)
    ]
    return min(values, default=1.0)


def _lower_body_chain_indices(joint_names: Sequence[str], *, joint_count: int) -> tuple[int, ...]:
    names = _effective_joint_names(joint_names)
    wanted = {
        "left_hip",
        "left_knee",
        "left_ankle",
        "left_big_toe",
        "left_big_toe_tip",
        "left_small_toe",
        "left_small_toe_tip",
        "left_heel",
        "right_hip",
        "right_knee",
        "right_ankle",
        "right_big_toe",
        "right_big_toe_tip",
        "right_small_toe",
        "right_small_toe_tip",
        "right_heel",
    }
    return tuple(index for index, name in enumerate(names[:joint_count]) if name in wanted)


def _stance_foot_indices(joint_names: Sequence[str], *, foot: str, joint_count: int) -> tuple[int, ...]:
    names = _effective_joint_names(joint_names)
    if foot == "left":
        wanted = {"left_ankle", "left_big_toe", "left_big_toe_tip", "left_small_toe", "left_small_toe_tip", "left_heel"}
    elif foot == "right":
        wanted = {"right_ankle", "right_big_toe", "right_big_toe_tip", "right_small_toe", "right_small_toe_tip", "right_heel"}
    else:
        return ()
    return tuple(index for index, name in enumerate(names[:joint_count]) if name in wanted)


def _effective_joint_names(joint_names: Sequence[str]) -> tuple[str, ...]:
    if _looks_like_sam3d_joint_names(joint_names):
        semantic = [f"sam3dbody_joint_{index:03d}" for index in range(len(joint_names))]
        for name, index in SAM3D_BODY_MHR70_SEMANTIC_MAP.joints.items():
            if index < len(semantic):
                semantic[index] = name
        for index, name in enumerate(MHR70_JOINT_NAMES):
            if index < len(semantic):
                semantic[index] = name
        return tuple(semantic)
    return tuple(str(name) for name in joint_names)


def _phase_dict(phase: ContactPhase) -> dict[str, Any]:
    return phase.to_dict()


def _pinned_phase_dict(phase: _PinnedPhase) -> dict[str, Any]:
    payload = phase.source.to_dict()
    payload["source"] = phase.kind
    payload["anchor_xy"] = [phase.anchor_xy[0], phase.anchor_xy[1]]
    return payload


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "median": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "max": max(values) if values else 0.0,
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[lower])
    return float(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (rank - lower))


def _cap_xy(dx: float, dy: float, max_magnitude: float) -> tuple[float, float, bool]:
    magnitude = math.hypot(dx, dy)
    if max_magnitude <= 0:
        return 0.0, 0.0, magnitude > 0.0
    if magnitude <= max_magnitude:
        return dx, dy, False
    scale = max_magnitude / magnitude
    return dx * scale, dy * scale, True


def _effective_max_correction_m(settings: FootPinSettings) -> float:
    return float(settings.max_correction_m)


def _copy_joints(value: Any) -> list[list[float]] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    joints: list[list[float]] = []
    for point in value:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) < 3:
            return None
        joints.append([float(point[0]), float(point[1]), float(point[2])])
    return joints


def _has_usable_foot_joints(joints: Sequence[Sequence[float]], joint_names: Sequence[str]) -> bool:
    if not joints:
        return False
    try:
        indices = resolve_foot_joint_indices(joint_names, joint_count=len(joints))
    except ValueError:
        return False
    return all(index < len(joints) for index in indices.all())


def _copy_conf(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    return [float(item) for item in value]


def _joint_names(payload: Mapping[str, Any]) -> tuple[str, ...]:
    value = payload.get("joint_names")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(name) for name in value)
    return tuple(MHR70_JOINT_NAMES)


def _frame_index(frame: Mapping[str, Any], *, fallback: int) -> int:
    value = frame.get("frame_index", frame.get("frame_idx", fallback))
    return int(value)


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _distance3(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt((float(left[0]) - float(right[0])) ** 2 + (float(left[1]) - float(right[1])) ** 2 + (float(left[2]) - float(right[2])) ** 2)


def _looks_like_sam3d_joint_names(joint_names: Sequence[str]) -> bool:
    return len(joint_names) == 70 and all(str(name) == f"sam3dbody_joint_{index:03d}" for index, name in enumerate(joint_names))


def _validate_settings(settings: FootPinSettings) -> None:
    if settings.max_correction_m < 0:
        raise ValueError("max_correction_m must be non-negative")
    if settings.max_smoothing_correction_m < 0:
        raise ValueError("max_smoothing_correction_m must be non-negative")
    if settings.taper_frames < 0:
        raise ValueError("taper_frames must be non-negative")
    if settings.min_phase_confidence < 0 or settings.min_phase_confidence > 1:
        raise ValueError("min_phase_confidence must be in [0, 1]")
    if settings.min_phase_frames < 1:
        raise ValueError("min_phase_frames must be >= 1")
    if settings.soft_anchor_max_height_m < 0.0:
        raise ValueError("soft_anchor_max_height_m must be non-negative")
    if settings.soft_anchor_max_speed_mps < 0.0:
        raise ValueError("soft_anchor_max_speed_mps must be non-negative")
    if settings.soft_anchor_min_frames < 1:
        raise ValueError("soft_anchor_min_frames must be >= 1")
    if settings.soft_anchor_ramp_frames < 1:
        raise ValueError("soft_anchor_ramp_frames must be >= 1")


def _empty_audit(*, settings: FootPinSettings, audit_path: str | None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "foot_pin_audit",
        "foot_pin_version": VERSION,
        "audit_path": audit_path,
        "settings": settings.to_dict(),
        "phase_detection": {
            "candidate_phase_count": 0,
            "confident_phase_count": 0,
            "skipped_low_confidence_phase_count": 0,
            "phases": [],
            "skipped_low_confidence_phases": [],
        },
        "summary": {
            "stance_slide_before_mm": _distribution([]),
            "stance_slide_after_mm": _distribution([]),
            "total_phase_count": 0,
            "total_skipped_low_confidence_phases": 0,
            "total_corrected_frame_count": 0,
            "max_correction_m": 0.0,
            "max_limb_length_delta_m": 0.0,
            "max_non_foot_joint_displacement_m": 0.0,
            "max_wrist_displacement_m": 0.0,
        },
        "metrics_before": {},
        "metrics_after": {},
        "players": {},
    }


def _attach_provenance(
    payload: MutableMapping[str, Any],
    *,
    settings: FootPinSettings,
    audit: Mapping[str, Any],
    audit_path: str | None,
) -> None:
    payload["foot_pin"] = {
        "version": VERSION,
        "params": settings.to_dict(),
        "audit_path": audit_path,
        "audit": audit,
        "players": audit.get("players", {}),
    }


__all__ = [
    "FootPinResult",
    "FootPinSettings",
    "apply_foot_pin_to_payload",
]
