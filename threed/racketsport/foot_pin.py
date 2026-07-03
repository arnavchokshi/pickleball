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
    max_correction_m: float = 0.15
    max_smoothing_correction_m: float = 0.049
    interpolate_between_stances: bool = True
    root_speed_clamp_mps: float = CORE_BODY_SPEED_CLAMP_MPS

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
        return math.hypot(self.dx, self.dy)


def apply_foot_pin_to_payload(
    payload: Mapping[str, Any],
    *,
    settings: FootPinSettings = FootPinSettings(),
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
    candidate_phases = detect_contact_phases(
        detection_frames,
        joint_names=joint_names,
        thresholds=settings.contact_thresholds(min_confidence=0.0),
    )
    eligible_source_phases = [
        phase for phase in candidate_phases if phase.min_confidence >= settings.min_phase_confidence
    ]
    skipped_low_confidence = [
        phase for phase in candidate_phases if phase.min_confidence < settings.min_phase_confidence
    ]
    pinned_phases = _median_anchor_phases(detection_frames, eligible_source_phases, joint_names, settings=settings)
    raw_corrections = _stance_corrections(detection_frames, pinned_phases, joint_names, settings=settings)
    corrections = _interpolated_corrections(frames, raw_corrections, settings=settings, joint_names=joint_names)

    before_roots = _root_xy_by_player(frames, corrections=None, joint_names=joint_names)
    _apply_corrections(frames, corrections)
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


def _stance_corrections(
    frames: Sequence[SkeletonFrame],
    phases: Sequence[_PinnedPhase],
    joint_names: Sequence[str],
    *,
    settings: FootPinSettings,
) -> dict[tuple[str, int], _Correction]:
    if not frames or not phases:
        return {}
    indices = resolve_foot_joint_indices(joint_names, joint_count=len(frames[0].joints_world))
    frame_map = {(str(frame.player_id), frame.frame_index): frame for frame in frames}
    active_by_frame: dict[tuple[str, int], list[_PinnedPhase]] = {}
    for phase in phases:
        for frame_index in phase.frame_indices:
            active_by_frame.setdefault((phase.player_id, frame_index), []).append(phase)

    corrections: dict[tuple[str, int], _Correction] = {}
    for key, active in active_by_frame.items():
        frame = frame_map.get(key)
        if frame is None:
            continue
        weighted_dx = 0.0
        weighted_dy = 0.0
        weight_sum = 0.0
        contacts: list[dict[str, Any]] = []
        for phase in active:
            phase_weight = _phase_weight(phase, frame.frame_index, settings.taper_frames)
            if phase_weight <= 0:
                continue
            current = foot_contact_point(
                frame,
                indices.for_foot(phase.foot),
                low_foot_band_m=settings.low_foot_band_m,
            )
            dx = phase.anchor_xy[0] - current[0]
            dy = phase.anchor_xy[1] - current[1]
            weighted_dx += dx * phase_weight
            weighted_dy += dy * phase_weight
            weight_sum += phase_weight
            contacts.append(
                {
                    "foot": phase.foot,
                    "start_frame_index": phase.source.start_frame_index,
                    "end_frame_index": phase.source.end_frame_index,
                    "anchor_xy": [phase.anchor_xy[0], phase.anchor_xy[1]],
                    "weight": phase_weight,
                    "min_confidence": phase.source.min_confidence,
                }
            )
        if weight_sum <= 0:
            continue
        dx = weighted_dx / weight_sum
        dy = weighted_dy / weight_sum
        dx, dy, capped = _cap_xy(dx, dy, settings.max_correction_m)
        corrections[key] = _Correction(
            dx=dx,
            dy=dy,
            weight=weight_sum / len(active),
            capped=capped,
            active_contacts=tuple(contacts),
        )
    return corrections


def _phase_weight(phase: _PinnedPhase, frame_index: int, taper_frames: int) -> float:
    if taper_frames <= 0 or len(phase.frame_indices) <= 1:
        return 1.0
    try:
        position = phase.frame_indices.index(frame_index)
    except ValueError:
        return 0.0
    edge_distance = min(position, len(phase.frame_indices) - 1 - position)
    if edge_distance >= taper_frames:
        return 1.0
    return float(edge_distance + 1) / float(taper_frames + 1)


def _interpolated_corrections(
    frames: Sequence[_FrameRef],
    raw: Mapping[tuple[str, int], _Correction],
    *,
    settings: FootPinSettings,
    joint_names: Sequence[str],
) -> dict[tuple[str, int], _Correction]:
    if not settings.interpolate_between_stances or not raw:
        return dict(raw)
    by_player: dict[str, list[_FrameRef]] = {}
    for frame in frames:
        by_player.setdefault(frame.player_id, []).append(frame)

    out = dict(raw)
    for player_id, player_frames in by_player.items():
        ordered = sorted(player_frames, key=lambda item: (item.frame_index, item.order))
        knot_positions = [
            index
            for index, frame in enumerate(ordered)
            if (player_id, frame.frame_index) in raw
        ]
        player_candidate: dict[tuple[str, int], _Correction] = {
            key: value for key, value in raw.items() if key[0] == player_id
        }
        for left_pos, right_pos in zip(knot_positions, knot_positions[1:]):
            left = ordered[left_pos]
            right = ordered[right_pos]
            left_corr = raw[(player_id, left.frame_index)]
            right_corr = raw[(player_id, right.frame_index)]
            gap = right_pos - left_pos
            if gap <= 1:
                continue
            left_root = _root_xy(left.frame, left.joints_world, joint_names=joint_names)
            right_root = _root_xy(right.frame, right.joints_world, joint_names=joint_names)
            left_target = (left_root[0] + left_corr.dx, left_root[1] + left_corr.dy)
            right_target = (right_root[0] + right_corr.dx, right_root[1] + right_corr.dy)
            for pos in range(left_pos + 1, right_pos):
                frame = ordered[pos]
                key = (player_id, frame.frame_index)
                if key in out:
                    continue
                alpha = (pos - left_pos) / gap
                desired_root = (
                    left_target[0] + (right_target[0] - left_target[0]) * alpha,
                    left_target[1] + (right_target[1] - left_target[1]) * alpha,
                )
                current_root = _root_xy(frame.frame, frame.joints_world, joint_names=joint_names)
                dx = desired_root[0] - current_root[0]
                dy = desired_root[1] - current_root[1]
                dx, dy, capped = _cap_xy(
                    dx,
                    dy,
                    min(settings.max_correction_m, settings.max_smoothing_correction_m),
                )
                player_candidate[key] = _Correction(
                    dx=dx,
                    dy=dy,
                    weight=0.0,
                    capped=capped,
                    active_contacts=(),
                )
        candidate_p90 = _root_p90_with_corrections(
            ordered,
            player_candidate,
            joint_names=joint_names,
            settings=settings,
        )
        baseline_p90 = _root_p90_with_corrections(
            ordered,
            {},
            joint_names=joint_names,
            settings=settings,
        )
        if candidate_p90 <= baseline_p90 + 1e-12:
            out.update(player_candidate)
    return out


def _apply_corrections(frames: Sequence[_FrameRef], corrections: Mapping[tuple[str, int], _Correction]) -> None:
    for frame in frames:
        correction = corrections.get((frame.player_id, frame.frame_index))
        if correction is None or correction.magnitude == 0.0:
            continue
        _translate_frame(frame.frame, dx=correction.dx, dy=correction.dy)


def _translate_frame(frame: MutableMapping[str, Any], *, dx: float, dy: float) -> None:
    for field in ("joints_world", "mesh_vertices_world"):
        values = frame.get(field)
        if isinstance(values, list):
            frame[field] = [_translated_vec3(point, dx=dx, dy=dy) for point in values]
    if isinstance(frame.get("transl_world"), list):
        frame["transl_world"] = _translated_vec3(frame["transl_world"], dx=dx, dy=dy)
    if isinstance(frame.get("floor_world_xyz"), list):
        frame["floor_world_xyz"] = _translated_vec3(frame["floor_world_xyz"], dx=dx, dy=dy)
    if isinstance(frame.get("track_world_xy"), list) and len(frame["track_world_xy"]) >= 2:
        frame["track_world_xy"] = [float(frame["track_world_xy"][0]) + dx, float(frame["track_world_xy"][1]) + dy]


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
        },
        "summary": {
            "stance_slide_before_mm": _distribution(before_phase_slides),
            "stance_slide_after_mm": _distribution(after_phase_slides),
            "total_phase_count": len(eligible_phases),
            "total_skipped_low_confidence_phases": len(skipped_low_confidence),
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


def _root_p90_with_corrections(
    frames: Sequence[_FrameRef],
    corrections: Mapping[tuple[str, int], _Correction],
    *,
    joint_names: Sequence[str],
    settings: FootPinSettings,
) -> float:
    samples: list[tuple[int, tuple[float, float]]] = []
    for frame in frames:
        root = _root_xy(frame.frame, frame.joints_world, joint_names=joint_names)
        correction = corrections.get((frame.player_id, frame.frame_index))
        if correction is not None:
            root = (root[0] + correction.dx, root[1] + correction.dy)
        samples.append((frame.frame_index, root))
    return float(_root_motion_stats(samples, settings=settings)["p90_frame_displacement_m"])


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
