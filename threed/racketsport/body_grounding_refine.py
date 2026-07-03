"""Post-hoc rigid BODY grounding refinement.

The solver only applies root-level rigid translations to already-produced
world skeletons/meshes. It does not alter joint angles or bone-relative pose.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Mapping, Sequence

from threed.racketsport.confidence_gate import (
    BAND_PHYSICS_CORRECTED,
    BAND_PHYSICS_CORRECTED_WARN,
)
from threed.racketsport.external_gt_body_prediction_schema import MHR70_JOINT_NAMES
from threed.racketsport.foot_contact import (
    FootJointIndices,
    SkeletonFrame,
    foot_contact_point,
    resolve_foot_joint_indices,
)


ARTIFACT_TYPE = "body_grounding_refinement"
SCHEMA_VERSION = 1
DEFAULT_ROOT_JOINT_NAMES = ("left_hip", "right_hip")


@dataclass(frozen=True)
class GroundingRefineConfig:
    court_z_m: float = 0.0
    root_joint_names: tuple[str, ...] = DEFAULT_ROOT_JOINT_NAMES
    smoothness_weight: float = 0.15
    max_correction_warn_m: float = 0.15

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _FrameRef:
    player_id: str
    frame_index: int
    time_s: float | None
    frame: dict[str, Any]
    joint_names: tuple[str, ...]
    foot_indices: FootJointIndices


def refine_body_grounding(
    skeleton_payload: Mapping[str, Any],
    *,
    foot_contact_phases: Mapping[str, Any],
    tracks: Mapping[str, Any] | None = None,
    config: GroundingRefineConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a refined BODY payload plus a residual/correction report."""

    cfg = config or GroundingRefineConfig()
    _validate_config(cfg)
    refined = copy.deepcopy(dict(skeleton_payload))
    frame_refs = _frame_refs(refined)
    track_index = _TrackIndex(tracks or {})
    phase_deltas = _phase_deltas(frame_refs, foot_contact_phases=foot_contact_phases, tracks=track_index, config=cfg)
    raw_deltas = _raw_frame_deltas(
        frame_refs,
        foot_contact_phases=foot_contact_phases,
        phase_deltas=phase_deltas,
        tracks=track_index,
        config=cfg,
    )
    deltas = _smooth_deltas_by_player(raw_deltas, frame_refs=frame_refs, smoothness_weight=cfg.smoothness_weight)

    before_foot = _foot_plane_residuals(frame_refs, foot_contact_phases=foot_contact_phases, config=cfg)
    before_track = _track_residuals(frame_refs, tracks=track_index, config=cfg)
    frame_reports: dict[str, list[dict[str, Any]]] = {}
    for ref in frame_refs:
        delta = deltas.get((ref.player_id, ref.frame_index), (0.0, 0.0, 0.0))
        _apply_translation(ref.frame, delta)
        magnitude = _norm3(delta)
        band = BAND_PHYSICS_CORRECTED_WARN if magnitude > cfg.max_correction_warn_m else BAND_PHYSICS_CORRECTED
        provenance = {
            "band": band,
            "display_band": band,
            "predictor": "BodyGroundingRefine",
            "horizon_frames": 0,
            "predicted_sigma_m": None,
        }
        ref.frame["confidence_provenance"] = provenance
        ref.frame["body_grounding_refinement"] = {
            "translation_delta_xyz": [delta[0], delta[1], delta[2]],
            "yaw_delta_rad": 0.0,
            "correction_magnitude_m": magnitude,
            "band": band,
        }
        if band == BAND_PHYSICS_CORRECTED_WARN:
            ref.frame["body_grounding_refinement"]["warning"] = (
                f"correction_magnitude {magnitude:.3f}m exceeds {cfg.max_correction_warn_m:.3f}m"
            )
        frame_reports.setdefault(ref.player_id, []).append(
            {
                "frame_index": ref.frame_index,
                "t": ref.time_s,
                "translation_delta_xyz": [delta[0], delta[1], delta[2]],
                "yaw_delta_rad": 0.0,
                "correction_magnitude_m": magnitude,
                "confidence_provenance": dict(provenance),
            }
        )

    after_refs = _frame_refs(refined)
    after_foot = _foot_plane_residuals(after_refs, foot_contact_phases=foot_contact_phases, config=cfg)
    after_track = _track_residuals(after_refs, tracks=track_index, config=cfg)
    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "config": cfg.to_dict(),
        "source": {
            "input_artifact_type": skeleton_payload.get("artifact_type"),
            "track_source_present": bool(tracks),
            "foot_contact_phase_count": len(_phase_items(foot_contact_phases)),
        },
        "summary": {
            "frame_count": len(frame_refs),
            "player_count": len({ref.player_id for ref in frame_refs}),
            "foot_plane_residual_m": _residual_summary(before_foot, after_foot, absolute=True),
            "track_residual_m": _residual_summary(before_track, after_track, absolute=False),
            "correction_magnitude_m": _magnitude_summary(
                deltas.values(), max_correction_warn_m=cfg.max_correction_warn_m
            ),
            "residual_family_worse": {
                "foot_plane": _mean_abs(after_foot) > _mean_abs(before_foot) + 1e-12,
                "track": _mean(after_track) > _mean(before_track) + 1e-12,
            },
            "kill_recommended": (_mean_abs(after_foot) > _mean_abs(before_foot) + 1e-12)
            or (_mean(after_track) > _mean(before_track) + 1e-12),
        },
        "players": {player_id: {"frames": frames} for player_id, frames in sorted(frame_reports.items())},
        "policy": {
            "rigid_root_level_only": True,
            "joint_angles_changed": False,
            "yaw_rotation_enabled": False,
            "protected_eval_labels_used": False,
        },
    }
    refined["body_grounding_refinement"] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "summary": report["summary"],
        "policy": report["policy"],
    }
    return refined, report


def _validate_config(config: GroundingRefineConfig) -> None:
    if config.smoothness_weight < 0:
        raise ValueError("smoothness_weight must be >= 0")
    if config.max_correction_warn_m <= 0:
        raise ValueError("max_correction_warn_m must be positive")


def _frame_refs(payload: Mapping[str, Any]) -> list[_FrameRef]:
    players = payload.get("players")
    if not isinstance(players, list) or not players:
        raise ValueError("payload.players must contain at least one player")
    refs: list[_FrameRef] = []
    top_joint_names = tuple(str(name) for name in payload.get("joint_names", ()) if isinstance(name, str))
    for player in players:
        if not isinstance(player, dict):
            continue
        player_id = str(player.get("id", player.get("player_id", "unknown")))
        frames = player.get("frames")
        if not isinstance(frames, list):
            continue
        for ordinal, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue
            joints = frame.get("joints_world")
            if not isinstance(joints, list) or not joints:
                continue
            frame_index = _frame_index(frame, fallback=ordinal)
            joint_names = _effective_joint_names(top_joint_names, joint_count=len(joints))
            foot_indices = resolve_foot_joint_indices(joint_names, joint_count=len(joints))
            refs.append(
                _FrameRef(
                    player_id=player_id,
                    frame_index=frame_index,
                    time_s=_optional_float(frame.get("t")),
                    frame=frame,
                    joint_names=joint_names,
                    foot_indices=foot_indices,
                )
            )
    refs.sort(key=lambda ref: (ref.player_id, ref.frame_index, ref.time_s if ref.time_s is not None else 0.0))
    if not refs:
        raise ValueError("payload.players[].frames[] must contain joints_world")
    return refs


def _effective_joint_names(joint_names: Sequence[str], *, joint_count: int) -> tuple[str, ...]:
    if joint_count == len(MHR70_JOINT_NAMES):
        return tuple(MHR70_JOINT_NAMES)
    if len(joint_names) >= joint_count:
        return tuple(joint_names[:joint_count])
    raise ValueError(f"joint_names has {len(joint_names)} entries for {joint_count} joints")


def _phase_deltas(
    frame_refs: Sequence[_FrameRef],
    *,
    foot_contact_phases: Mapping[str, Any],
    tracks: "_TrackIndex",
    config: GroundingRefineConfig,
) -> dict[int, tuple[float, float, float]]:
    refs_by_key = {(ref.player_id, ref.frame_index): ref for ref in frame_refs}
    phase_delta_by_id: dict[int, tuple[float, float, float]] = {}
    for phase_id, phase in enumerate(_phase_items(foot_contact_phases)):
        foot = str(phase.get("foot", ""))
        player_id = str(phase.get("player_id", "unknown"))
        frame_indices = [int(index) for index in phase.get("frame_indices", [])]
        xyz_deltas: list[tuple[float, float, float]] = []
        for frame_index in frame_indices:
            ref = refs_by_key.get((player_id, frame_index))
            if ref is None:
                continue
            dz = _foot_plane_delta(ref, foot=foot, config=config)
            track_xy = tracks.track_xy(ref)
            dx, dy = (0.0, 0.0)
            if track_xy is not None:
                root_xy = _root_xy(ref, config=config)
                dx = track_xy[0] - root_xy[0]
                dy = track_xy[1] - root_xy[1]
            xyz_deltas.append((dx, dy, dz))
        if xyz_deltas:
            phase_delta_by_id[phase_id] = _mean_delta(xyz_deltas)
    return phase_delta_by_id


def _raw_frame_deltas(
    frame_refs: Sequence[_FrameRef],
    *,
    foot_contact_phases: Mapping[str, Any],
    phase_deltas: Mapping[int, tuple[float, float, float]],
    tracks: "_TrackIndex",
    config: GroundingRefineConfig,
) -> dict[tuple[str, int], tuple[float, float, float]]:
    phase_ids_by_frame: dict[tuple[str, int], list[int]] = {}
    for phase_id, phase in enumerate(_phase_items(foot_contact_phases)):
        player_id = str(phase.get("player_id", "unknown"))
        for frame_index in phase.get("frame_indices", []):
            phase_ids_by_frame.setdefault((player_id, int(frame_index)), []).append(phase_id)

    out: dict[tuple[str, int], tuple[float, float, float]] = {}
    z_by_contact_frame: dict[tuple[str, int], float] = {}
    for ref in frame_refs:
        key = (ref.player_id, ref.frame_index)
        active_phase_deltas = [phase_deltas[phase_id] for phase_id in phase_ids_by_frame.get(key, []) if phase_id in phase_deltas]
        if active_phase_deltas:
            dx, dy, dz = _mean_delta(active_phase_deltas)
            z_by_contact_frame[key] = dz
        else:
            dx, dy = 0.0, 0.0
            track_xy = tracks.track_xy(ref)
            if track_xy is not None:
                root_xy = _root_xy(ref, config=config)
                dx = track_xy[0] - root_xy[0]
                dy = track_xy[1] - root_xy[1]
            dz = 0.0
        out[key] = (dx, dy, dz)

    _fill_non_contact_z(out, frame_refs=frame_refs, z_by_contact_frame=z_by_contact_frame)
    return out


def _fill_non_contact_z(
    deltas: dict[tuple[str, int], tuple[float, float, float]],
    *,
    frame_refs: Sequence[_FrameRef],
    z_by_contact_frame: Mapping[tuple[str, int], float],
) -> None:
    by_player: dict[str, list[_FrameRef]] = {}
    for ref in frame_refs:
        by_player.setdefault(ref.player_id, []).append(ref)
    for player_id, refs in by_player.items():
        contact_refs = [ref for ref in refs if (player_id, ref.frame_index) in z_by_contact_frame]
        if not contact_refs:
            continue
        for ref in refs:
            key = (player_id, ref.frame_index)
            if key in z_by_contact_frame:
                continue
            nearest = min(contact_refs, key=lambda contact: abs(contact.frame_index - ref.frame_index))
            dx, dy, _dz = deltas[key]
            deltas[key] = (dx, dy, z_by_contact_frame[(player_id, nearest.frame_index)])


def _smooth_deltas_by_player(
    raw: Mapping[tuple[str, int], tuple[float, float, float]],
    *,
    frame_refs: Sequence[_FrameRef],
    smoothness_weight: float,
) -> dict[tuple[str, int], tuple[float, float, float]]:
    if smoothness_weight == 0:
        return dict(raw)
    out: dict[tuple[str, int], tuple[float, float, float]] = {}
    previous_by_player: dict[str, tuple[float, float, float]] = {}
    for ref in frame_refs:
        key = (ref.player_id, ref.frame_index)
        current = raw.get(key, (0.0, 0.0, 0.0))
        previous = previous_by_player.get(ref.player_id)
        if previous is None:
            smoothed = current
        else:
            denom = 1.0 + smoothness_weight
            smoothed = tuple((current[idx] + smoothness_weight * previous[idx]) / denom for idx in range(3))
        previous_by_player[ref.player_id] = smoothed
        out[key] = smoothed
    return out


class _TrackIndex:
    def __init__(self, tracks: Mapping[str, Any]) -> None:
        self._by_frame: dict[tuple[str, int], tuple[float, float]] = {}
        self._by_time: dict[tuple[str, int], tuple[float, float]] = {}
        for player in tracks.get("players", []) if isinstance(tracks.get("players"), list) else []:
            if not isinstance(player, Mapping):
                continue
            player_id = str(player.get("id", player.get("player_id", "unknown")))
            for ordinal, frame in enumerate(player.get("frames", []) or []):
                if not isinstance(frame, Mapping):
                    continue
                xy = _vec2(frame.get("world_xy", frame.get("track_world_xy")))
                if xy is None:
                    continue
                if "frame_idx" in frame or "frame_index" in frame:
                    self._by_frame[(player_id, _frame_index(frame, fallback=ordinal))] = xy
                time_s = _optional_float(frame.get("t"))
                if time_s is not None:
                    self._by_time[(player_id, _time_key(time_s))] = xy

    def track_xy(self, ref: _FrameRef) -> tuple[float, float] | None:
        direct = _vec2(ref.frame.get("track_world_xy"))
        if (ref.player_id, ref.frame_index) in self._by_frame:
            return self._by_frame[(ref.player_id, ref.frame_index)]
        if ref.time_s is not None and (ref.player_id, _time_key(ref.time_s)) in self._by_time:
            return self._by_time[(ref.player_id, _time_key(ref.time_s))]
        return direct


def _phase_items(foot_contact_phases: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    phases = foot_contact_phases.get("phases")
    return [phase for phase in phases if isinstance(phase, Mapping)] if isinstance(phases, list) else []


def _foot_plane_delta(ref: _FrameRef, *, foot: str, config: GroundingRefineConfig) -> float:
    indices = ref.foot_indices.for_foot(foot)
    point = foot_contact_point(
        SkeletonFrame(
            player_id=ref.player_id,
            frame_index=ref.frame_index,
            t=ref.time_s,
            joints_world=ref.frame["joints_world"],
            joint_conf=ref.frame.get("joint_conf"),
        ),
        indices,
    )
    return config.court_z_m - point[2]


def _root_xy(ref: _FrameRef, *, config: GroundingRefineConfig) -> tuple[float, float]:
    transl = _vec3(ref.frame.get("transl_world"))
    if transl is not None:
        return (transl[0], transl[1])
    name_to_index = {name: index for index, name in enumerate(ref.joint_names)}
    root_indices = [name_to_index[name] for name in config.root_joint_names if name in name_to_index]
    if not root_indices:
        root_indices = [0]
    joints = ref.frame["joints_world"]
    return (
        sum(float(joints[index][0]) for index in root_indices) / len(root_indices),
        sum(float(joints[index][1]) for index in root_indices) / len(root_indices),
    )


def _apply_translation(frame: dict[str, Any], delta: tuple[float, float, float]) -> None:
    for key in ("joints_world", "mesh_vertices_world"):
        values = frame.get(key)
        if isinstance(values, list):
            frame[key] = [[float(point[0]) + delta[0], float(point[1]) + delta[1], float(point[2]) + delta[2]] for point in values]
    for key in ("transl_world", "floor_world_xyz"):
        vector = _vec3(frame.get(key))
        if vector is not None:
            frame[key] = [vector[0] + delta[0], vector[1] + delta[1], vector[2] + delta[2]]
    if isinstance(frame.get("min_mesh_z_m"), (int, float)):
        frame["min_mesh_z_m"] = float(frame["min_mesh_z_m"]) + delta[2]


def _foot_plane_residuals(
    frame_refs: Sequence[_FrameRef],
    *,
    foot_contact_phases: Mapping[str, Any],
    config: GroundingRefineConfig,
) -> list[float]:
    refs_by_key = {(ref.player_id, ref.frame_index): ref for ref in frame_refs}
    residuals: list[float] = []
    for phase in _phase_items(foot_contact_phases):
        player_id = str(phase.get("player_id", "unknown"))
        foot = str(phase.get("foot", ""))
        for frame_index in phase.get("frame_indices", []) or []:
            ref = refs_by_key.get((player_id, int(frame_index)))
            if ref is None:
                continue
            residuals.append(-_foot_plane_delta(ref, foot=foot, config=config))
    return residuals


def _track_residuals(
    frame_refs: Sequence[_FrameRef],
    *,
    tracks: _TrackIndex,
    config: GroundingRefineConfig,
) -> list[float]:
    residuals: list[float] = []
    for ref in frame_refs:
        track_xy = tracks.track_xy(ref)
        if track_xy is None:
            continue
        root_xy = _root_xy(ref, config=config)
        residuals.append(math.dist(root_xy, track_xy))
    return residuals


def _residual_summary(before: Sequence[float], after: Sequence[float], *, absolute: bool) -> dict[str, float | int]:
    if absolute:
        before_values = [abs(value) for value in before]
        after_values = [abs(value) for value in after]
    else:
        before_values = list(before)
        after_values = list(after)
    return {
        "count": len(before_values),
        "mean_before": _mean(before_values),
        "mean_after": _mean(after_values),
        "mean_delta": _mean(after_values) - _mean(before_values),
        "mean_abs_before": _mean_abs(before),
        "mean_abs_after": _mean_abs(after),
        "max_before": max(before_values, default=0.0),
        "max_after": max(after_values, default=0.0),
        "rms_before": _rms(before_values),
        "rms_after": _rms(after_values),
    }


def _magnitude_summary(
    deltas: Sequence[tuple[float, float, float]] | Any, *, max_correction_warn_m: float
) -> dict[str, float | int]:
    magnitudes = [_norm3(delta) for delta in deltas]
    warn_count = sum(1 for magnitude in magnitudes if magnitude > max_correction_warn_m)
    return {
        "count": len(magnitudes),
        "mean": _mean(magnitudes),
        "max": max(magnitudes, default=0.0),
        "rms": _rms(magnitudes),
        "warn_count": warn_count,
    }


def _mean_delta(values: Sequence[tuple[float, float, float]]) -> tuple[float, float, float]:
    return (
        sum(value[0] for value in values) / len(values),
        sum(value[1] for value in values) / len(values),
        sum(value[2] for value in values) / len(values),
    )


def _frame_index(frame: Mapping[str, Any], *, fallback: int) -> int:
    value = frame.get("frame_idx", frame.get("frame_index", fallback))
    return int(value)


def _time_key(time_s: float) -> int:
    return round(time_s * 1_000_000)


def _vec2(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


def _vec3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _mean_abs(values: Sequence[float]) -> float:
    return _mean([abs(value) for value in values])


def _rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def _norm3(value: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) * float(item) for item in value[:3]))


__all__ = ["GroundingRefineConfig", "refine_body_grounding"]
