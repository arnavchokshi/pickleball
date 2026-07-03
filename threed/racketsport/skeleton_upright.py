"""Rotate camera-frame skeleton offsets upright in the court frame."""

from __future__ import annotations

import copy
import math
from statistics import median
from typing import Any, Mapping, Sequence


ROTATION_CONVENTION_OFFSET_ROW_TIMES_R = "offset_row_times_R"
ROTATION_CONVENTION_OFFSET_ROW_TIMES_R_TRANSPOSE = "offset_row_times_R_transpose"
ORIGINAL_NO_ROTATION = "original_no_rotation"
ARTIFACT_TYPE = "racketsport_skeleton_upright_repair"
SCHEMA_VERSION = 1
DEFAULT_STATURE_BAND_M = (1.4, 1.8)
SEVERE_FOOT_PENETRATION_CLAMP_M = 0.35
HEAD_JOINT_NAMES = ("nose", "head", "left_eye", "right_eye")
ANKLE_JOINT_NAMES = ("left_ankle", "right_ankle")
FOOT_JOINT_NAMES = (
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
)
ROOT_JOINT_NAMES = ("pelvis", "root", "mid_hip")


def rotate_camera_offsets_row_times_R(
    joints_camera_frame: Sequence[Sequence[float]],
    *,
    rotation: Sequence[Sequence[float]],
    joint_names: Sequence[str],
) -> list[list[float]]:
    """Rotate root-relative camera-frame offsets with the accepted row-vector convention."""

    return _rotate_offsets(
        joints_camera_frame,
        rotation=_mat3(rotation),
        joint_names=joint_names,
        convention=ROTATION_CONVENTION_OFFSET_ROW_TIMES_R,
    )


def score_upright_conventions(
    skeleton_payload: Mapping[str, Any],
    *,
    calibration_rotation: Sequence[Sequence[float]],
    z_smoothing_radius: int = 2,
    foot_contact_phases: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Try both supported extrinsic directions and score upright metrics."""

    rotation = _mat3(calibration_rotation)
    variants: dict[str, dict[str, Any]] = {
        ORIGINAL_NO_ROTATION: skeleton_upright_metrics(skeleton_payload),
    }
    for convention in (ROTATION_CONVENTION_OFFSET_ROW_TIMES_R, ROTATION_CONVENTION_OFFSET_ROW_TIMES_R_TRANSPOSE):
        repaired, _repair_stats = _repair_payload_with_convention(
            skeleton_payload,
            rotation=rotation,
            convention=convention,
            z_smoothing_radius=z_smoothing_radius,
            foot_contact_phases=foot_contact_phases,
        )
        variants[convention] = skeleton_upright_metrics(repaired)
    selected = max(
        (ROTATION_CONVENTION_OFFSET_ROW_TIMES_R, ROTATION_CONVENTION_OFFSET_ROW_TIMES_R_TRANSPOSE),
        key=lambda name: _selection_score(variants[name]),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "skeleton_upright_convention_selection",
        "selected_convention": selected,
        "selection_reason": (
            "highest weighted score over mean z-span closeness to 1.6m, "
            "heads-above-ankles rate, and feet-floor proximity"
        ),
        "variants": variants,
    }


def repair_skeleton_upright_payload(
    skeleton_payload: Mapping[str, Any],
    *,
    calibration_rotation: Sequence[Sequence[float]],
    calibration_path: str,
    z_smoothing_radius: int = 2,
    foot_contact_phases: Mapping[str, Any] | None = None,
    stature_band_m: tuple[float, float] = DEFAULT_STATURE_BAND_M,
    overlay_scale_suspect_caption: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a repaired skeleton3d payload and a repair report."""

    convention_report = score_upright_conventions(
        skeleton_payload,
        calibration_rotation=calibration_rotation,
        z_smoothing_radius=z_smoothing_radius,
        foot_contact_phases=foot_contact_phases,
    )
    selected = str(convention_report["selected_convention"])
    repaired, repair_stats = _repair_payload_with_convention(
        skeleton_payload,
        rotation=_mat3(calibration_rotation),
        convention=selected,
        z_smoothing_radius=z_smoothing_radius,
        foot_contact_phases=foot_contact_phases,
    )
    metrics_before = convention_report["variants"][ORIGINAL_NO_ROTATION]
    metrics_after = skeleton_upright_metrics(repaired)
    stature_check = build_stature_check(repaired, stature_band_m=stature_band_m)
    caption = overlay_scale_suspect_caption
    if caption is None and stature_check["scale_suspect"]:
        median_stature = stature_check.get("median_standing_z_span_m")
        if isinstance(median_stature, (int, float)) and math.isfinite(float(median_stature)):
            caption = f"SCALE SUSPECT: stature ~{float(median_stature):.2f}m — under investigation"
        else:
            caption = "SCALE SUSPECT: stature out of plausible band — under investigation"

    provenance = dict(repaired.get("provenance") or {})
    repair_provenance = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "rotation_convention": selected,
        "calibration_path": calibration_path,
        "z_grounding": {
            "floor_z_m": 0.0,
            "smoothing": "moving_median_by_player",
            "smoothing_radius_frames": int(z_smoothing_radius),
            "contact_phase_frames_prefer_raw_ankle_min_to_floor": bool(foot_contact_phases),
            "severe_foot_penetration_outlier_clamp": repair_stats[
                "severe_foot_penetration_outlier_clamp"
            ],
        },
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "stature_check": stature_check,
        "scale_suspect": bool(stature_check["scale_suspect"]),
        "overlay_caption_extra": caption,
        "protected_eval_labels_used": False,
    }
    provenance["skeleton_upright_repair"] = repair_provenance
    repaired["provenance"] = provenance
    report = {
        **repair_provenance,
        "selected_convention": selected,
        "convention_selection": convention_report,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "stature_check": stature_check,
    }
    return repaired, report


def skeleton_upright_metrics(skeleton_payload: Mapping[str, Any]) -> dict[str, Any]:
    names = _joint_names(skeleton_payload)
    head_indices = _indices(names, HEAD_JOINT_NAMES) or [0]
    ankle_indices = _indices(names, ANKLE_JOINT_NAMES) or _indices(names, FOOT_JOINT_NAMES) or [0]
    foot_indices = _indices(names, FOOT_JOINT_NAMES) or list(range(len(names)))
    spans: list[float] = []
    foot_abs: list[float] = []
    heads_above = 0
    feet_near = 0
    head_failures: list[dict[str, Any]] = []
    for item in _frame_items(skeleton_payload):
        joints = item["joints"]
        z_values = [point[2] for point in joints]
        span = max(z_values) - min(z_values)
        spans.append(span)
        head_z = max(joints[index][2] for index in head_indices if index < len(joints))
        ankle_z = min(joints[index][2] for index in ankle_indices if index < len(joints))
        if head_z > ankle_z:
            heads_above += 1
        else:
            head_failures.append(
                {
                    "player_id": item["player_id"],
                    "frame_idx": item["frame_idx"],
                    "t": item["t"],
                    "head_z_m": head_z,
                    "ankle_z_m": ankle_z,
                    "z_span_m": span,
                }
            )
        min_foot_abs = min(abs(joints[index][2]) for index in foot_indices if index < len(joints))
        foot_abs.append(min_foot_abs)
        if min_foot_abs <= 0.35:
            feet_near += 1
    frame_count = len(spans)
    mean_z_span = _mean(spans)
    heads_rate = heads_above / frame_count if frame_count else 0.0
    feet_rate = feet_near / frame_count if frame_count else 0.0
    return {
        "frame_count": frame_count,
        "mean_z_span_m": mean_z_span,
        "median_z_span_m": median(spans) if spans else None,
        "p95_z_span_m": _p95(spans),
        "heads_above_ankles_count": heads_above,
        "heads_above_ankles_rate": heads_rate,
        "feet_within_0_35m_count": feet_near,
        "feet_within_0_35m_rate": feet_rate,
        "foot_min_abs_p95_m": _p95(foot_abs),
        "passes_binding_gate_strict": bool(frame_count and 1.4 <= mean_z_span <= 1.8 and heads_rate >= 0.95 and feet_rate >= 0.90),
        "heads_not_above_ankles_frames": head_failures,
    }


def build_stature_check(
    skeleton_payload: Mapping[str, Any],
    *,
    stature_band_m: tuple[float, float] = DEFAULT_STATURE_BAND_M,
) -> dict[str, Any]:
    names = _joint_names(skeleton_payload)
    head_indices = _indices(names, HEAD_JOINT_NAMES) or [0]
    ankle_indices = _indices(names, ANKLE_JOINT_NAMES) or _indices(names, FOOT_JOINT_NAMES) or [0]
    foot_indices = _indices(names, FOOT_JOINT_NAMES) or list(range(len(names)))
    by_player: dict[str, list[float]] = {}
    for item in _frame_items(skeleton_payload):
        joints = item["joints"]
        if max(joints[index][2] for index in head_indices if index < len(joints)) <= min(
            joints[index][2] for index in ankle_indices if index < len(joints)
        ):
            continue
        if min(abs(joints[index][2]) for index in foot_indices if index < len(joints)) > 0.35:
            continue
        z_values = [point[2] for point in joints]
        by_player.setdefault(str(item["player_id"]), []).append(max(z_values) - min(z_values))

    low, high = float(stature_band_m[0]), float(stature_band_m[1])
    players: dict[str, dict[str, Any]] = {}
    all_medians: list[float] = []
    for player_id, values in sorted(by_player.items(), key=lambda pair: pair[0]):
        med = float(median(values)) if values else None
        if med is not None:
            all_medians.append(med)
        players[player_id] = {
            "standing_frame_count": len(values),
            "median_standing_z_span_m": med,
            "plausible_band_m": [low, high],
            "scale_suspect": bool(med is None or med < low or med > high),
        }
    overall = float(median(all_medians)) if all_medians else None
    return {
        "plausible_band_m": [low, high],
        "median_standing_z_span_m": overall,
        "scale_suspect": bool(overall is None or overall < low or overall > high or any(item["scale_suspect"] for item in players.values())),
        "players": players,
    }


def _repair_payload_with_convention(
    skeleton_payload: Mapping[str, Any],
    *,
    rotation: Sequence[Sequence[float]],
    convention: str,
    z_smoothing_radius: int,
    foot_contact_phases: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repaired = copy.deepcopy(dict(skeleton_payload))
    names = _joint_names(repaired)
    contact_keys = _contact_frame_keys(foot_contact_phases)
    clamp_stats = {
        "enabled": True,
        "floor_z_m": 0.0,
        "penetration_threshold_m": SEVERE_FOOT_PENETRATION_CLAMP_M,
        "clamped_joint_count": 0,
        "clamped_frame_count": 0,
    }
    for player in repaired.get("players", []):
        if not isinstance(player, dict):
            continue
        player_id = str(player.get("id", player.get("player_id", "unknown")))
        frames = player.get("frames")
        if not isinstance(frames, list):
            continue
        pending: list[tuple[dict[str, Any], list[list[float]], float]] = []
        raw_z_deltas: list[float] = []
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            joints = _parse_joints(frame.get("joints_world"))
            if not joints:
                continue
            rotated = _rotate_offsets(joints, rotation=rotation, joint_names=names, convention=convention)
            foot_indices = _indices(names, FOOT_JOINT_NAMES) or list(range(len(rotated)))
            dz = -min(rotated[index][2] for index in foot_indices if index < len(rotated))
            pending.append((frame, rotated, dz))
            raw_z_deltas.append(dz)
        smoothed_z = _moving_median(raw_z_deltas, radius=max(0, int(z_smoothing_radius)))
        for (frame, rotated, raw_dz), dz in zip(pending, smoothed_z):
            key = (player_id, _frame_index(frame))
            if key in contact_keys:
                dz = raw_dz
            shifted = [[point[0], point[1], point[2] + dz] for point in rotated]
            clamped_count = _clamp_severe_foot_penetration_outliers(shifted, names)
            if clamped_count:
                clamp_stats["clamped_joint_count"] += clamped_count
                clamp_stats["clamped_frame_count"] += 1
            frame["joints_world"] = shifted
            root = _root_xyz(shifted, names)
            existing_transl = frame.get("transl_world")
            if isinstance(existing_transl, list) and len(existing_transl) >= 3:
                frame["transl_world"] = [float(existing_transl[0]), float(existing_transl[1]), float(existing_transl[2]) + dz]
            else:
                frame["transl_world"] = root
    return repaired, {"severe_foot_penetration_outlier_clamp": clamp_stats}


def _rotate_offsets(
    joints: Sequence[Sequence[float]],
    *,
    rotation: Sequence[Sequence[float]],
    joint_names: Sequence[str],
    convention: str,
) -> list[list[float]]:
    parsed = [[float(point[0]), float(point[1]), float(point[2])] for point in joints]
    root = _root_xyz(parsed, joint_names)
    matrix = _transpose3(rotation) if convention == ROTATION_CONVENTION_OFFSET_ROW_TIMES_R_TRANSPOSE else _mat3(rotation)
    rotated: list[list[float]] = []
    for point in parsed:
        off = [point[idx] - root[idx] for idx in range(3)]
        rot = _row_times_matrix(off, matrix)
        rotated.append([root[idx] + rot[idx] for idx in range(3)])
    return rotated


def _clamp_severe_foot_penetration_outliers(joints: list[list[float]], names: Sequence[str]) -> int:
    clamped = 0
    for index in _indices(names, FOOT_JOINT_NAMES):
        if index >= len(joints):
            continue
        if joints[index][2] < -SEVERE_FOOT_PENETRATION_CLAMP_M:
            joints[index][2] = 0.0
            clamped += 1
    return clamped


def _frame_items(skeleton_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for player in skeleton_payload.get("players", []):
        if not isinstance(player, Mapping):
            continue
        player_id = player.get("id", player.get("player_id", "unknown"))
        frames = player.get("frames")
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            joints = _parse_joints(frame.get("joints_world"))
            if not joints:
                continue
            out.append({"player_id": player_id, "frame_idx": frame.get("frame_idx"), "t": frame.get("t"), "joints": joints})
    return out


def _parse_joints(values: Any) -> list[list[float]]:
    if not isinstance(values, list):
        return []
    out: list[list[float]] = []
    for point in values:
        if not isinstance(point, list) or len(point) < 3:
            continue
        out.append([float(point[0]), float(point[1]), float(point[2])])
    return out


def _joint_names(payload: Mapping[str, Any]) -> tuple[str, ...]:
    names = payload.get("joint_names")
    if not isinstance(names, list) or not names:
        raise ValueError("skeleton payload must include joint_names")
    return tuple(str(name) for name in names)


def _indices(names: Sequence[str], wanted: Sequence[str]) -> list[int]:
    by_name = {name: index for index, name in enumerate(names)}
    return [by_name[name] for name in wanted if name in by_name]


def _root_joint_names(names: Sequence[str]) -> list[str]:
    direct = [name for name in ROOT_JOINT_NAMES if name in names]
    if direct:
        return direct
    hips = [name for name in ("left_hip", "right_hip") if name in names]
    return hips or [str(names[0])]


def _root_xyz(joints: Sequence[Sequence[float]], names: Sequence[str]) -> list[float]:
    root_indices = _indices(names, ROOT_JOINT_NAMES)
    if not root_indices:
        root_indices = _indices(names, ("left_hip", "right_hip"))
    if not root_indices:
        root_indices = [0]
    points = [joints[index] for index in root_indices if index < len(joints)]
    return [_mean([point[axis] for point in points]) for axis in range(3)]


def _contact_frame_keys(foot_contact_phases: Mapping[str, Any] | None) -> set[tuple[str, int]]:
    if not isinstance(foot_contact_phases, Mapping):
        return set()
    phases = foot_contact_phases.get("phases")
    if not isinstance(phases, list):
        return set()
    keys: set[tuple[str, int]] = set()
    for phase in phases:
        if not isinstance(phase, Mapping):
            continue
        player_id = str(phase.get("player_id", "unknown"))
        frame_indices = phase.get("frame_indices")
        if isinstance(frame_indices, list):
            for frame_idx in frame_indices:
                keys.add((player_id, int(frame_idx)))
    return keys


def _frame_index(frame: Mapping[str, Any]) -> int:
    if frame.get("frame_idx") is not None:
        return int(frame["frame_idx"])
    if frame.get("frame_index") is not None:
        return int(frame["frame_index"])
    return -1


def _selection_score(metrics: Mapping[str, Any]) -> float:
    z_span = float(metrics.get("mean_z_span_m") or 0.0)
    z_score = max(0.0, 1.0 - min(abs(z_span - 1.6) / 1.6, 1.0))
    return (
        3.0 * z_score
        + 2.0 * float(metrics.get("heads_above_ankles_rate") or 0.0)
        + float(metrics.get("feet_within_0_35m_rate") or 0.0)
    )


def _moving_median(values: Sequence[float], *, radius: int) -> list[float]:
    if radius <= 0:
        return [float(value) for value in values]
    out: list[float] = []
    for index in range(len(values)):
        out.append(float(median(values[max(0, index - radius) : min(len(values), index + radius + 1)])))
    return out


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if len(values) < 20:
        return float(max(values))
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)
    return ordered[index]


def _mat3(values: Sequence[Sequence[float]]) -> list[list[float]]:
    if len(values) != 3:
        raise ValueError("rotation must be 3x3")
    matrix: list[list[float]] = []
    for row in values:
        if len(row) != 3:
            raise ValueError("rotation must be 3x3")
        matrix.append([float(value) for value in row])
    return matrix


def _transpose3(values: Sequence[Sequence[float]]) -> list[list[float]]:
    matrix = _mat3(values)
    return [[matrix[col][row] for col in range(3)] for row in range(3)]


def _row_times_matrix(vector: Sequence[float], matrix: Sequence[Sequence[float]]) -> list[float]:
    return [sum(float(vector[row]) * float(matrix[row][col]) for row in range(3)) for col in range(3)]
