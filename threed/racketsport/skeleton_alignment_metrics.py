"""Alignment metrics for comparing 3D skeletons against 2D keypoint evidence."""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import median
from typing import Any, Mapping, Sequence

from .skeleton_lift_2d import (
    BonePrior,
    Lift2DConfig,
    _bone_priors_for_player,
    _camera_from_payload,
    _frame_idx,
    _frames,
    _keypoints_by_joint,
    _players,
)


ARTIFACT_TYPE = "racketsport_skeleton_alignment_metrics"
SCHEMA_VERSION = 1

HEAD_JOINT_NAMES = {"nose", "head", "left_eye", "right_eye", "left_ear", "right_ear"}
LOWER_BODY_JOINT_NAMES = {
    "pelvis",
    "mid_hip",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
}
ARM_JOINT_NAMES = {"left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"}
TORSO_JOINT_NAMES = {"pelvis", "mid_hip", "neck", "left_shoulder", "right_shoulder", "left_hip", "right_hip"}
FOOT_JOINT_NAMES = {
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
}


def score_skeleton_alignment(
    skeleton_payload: Mapping[str, Any],
    keypoints_payload: Mapping[str, Any],
    calibration_payload: Mapping[str, Any],
    *,
    min_keypoint_confidence: float = 0.05,
) -> dict[str, Any]:
    """Return projection, jitter, stature, and bone stability metrics."""

    camera = _camera_from_payload(calibration_payload)
    joint_names = [str(name) for name in skeleton_payload.get("joint_names", [])]
    if not joint_names:
        raise ValueError("skeleton payload must include joint_names")
    skeleton_by_player = _skeleton_by_player_frame(skeleton_payload)
    keypoints_by_player = _keypoints_by_player_frame(keypoints_payload)

    projection_rows: list[dict[str, Any]] = []
    errors_by_joint: dict[str, list[float]] = defaultdict(list)
    errors_by_player: dict[str, list[float]] = defaultdict(list)
    for player_id, frames in keypoints_by_player.items():
        skeleton_frames = skeleton_by_player.get(player_id, {})
        for frame_idx, frame_keypoints in frames.items():
            skeleton_frame = skeleton_frames.get(frame_idx)
            if skeleton_frame is None:
                continue
            joints_world = _joints_world(skeleton_frame)
            for joint_index, joint_name in enumerate(joint_names):
                keypoint = frame_keypoints.get(joint_name)
                if keypoint is None or keypoint.conf < min_keypoint_confidence or joint_index >= len(joints_world):
                    continue
                joint_world = joints_world[joint_index]
                if not _finite_vec(joint_world, length=3):
                    continue
                projected = camera.project(joint_world)
                distance = math.hypot(projected[0] - keypoint.x_px, projected[1] - keypoint.y_px)
                if distance < 1e-9:
                    distance = 0.0
                projection_rows.append(
                    {
                        "player_id": player_id,
                        "frame_idx": frame_idx,
                        "joint": joint_name,
                        "distance_px": distance,
                    }
                )
                errors_by_joint[joint_name].append(distance)
                errors_by_player[player_id].append(distance)

    jitter = _jitter_metrics(skeleton_by_player, joint_names)
    stature = _stature_stability(skeleton_by_player, joint_names)
    bones = _bone_length_variance(skeleton_payload, keypoints_payload, joint_names)

    all_errors = [row["distance_px"] for row in projection_rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "projection_error_px": {
            "overall": _distribution(all_errors),
            "by_joint": {joint: _distribution(values) for joint, values in sorted(errors_by_joint.items())},
            "by_player": {player: _distribution(values) for player, values in sorted(errors_by_player.items())},
            "matched_observation_count": len(projection_rows),
        },
        "jitter_m_per_frame": jitter,
        "stature_stability": stature,
        "bone_length_variance": bones,
        "comparison_ready": {
            "projection_error_px": bool(projection_rows),
            "jitter": bool(jitter["groups"]["all"]["count"]),
            "stature": bool(stature["players"]),
            "bone_length_variance": bool(bones["bones"]),
        },
    }


def _skeleton_by_player_frame(payload: Mapping[str, Any]) -> dict[str, dict[int, Mapping[str, Any]]]:
    out: dict[str, dict[int, Mapping[str, Any]]] = {}
    for player in _players(payload):
        player_id = str(player.get("id") or player.get("player_id"))
        frames: dict[int, Mapping[str, Any]] = {}
        for frame in _frames(player):
            frames[_frame_idx(frame)] = frame
        if player_id:
            out[player_id] = frames
    return out


def _keypoints_by_player_frame(payload: Mapping[str, Any]) -> dict[str, dict[int, Mapping[str, Any]]]:
    out: dict[str, dict[int, Mapping[str, Any]]] = {}
    for player in _players(payload):
        player_id = str(player.get("id") or player.get("player_id"))
        frames: dict[int, Mapping[str, Any]] = {}
        for frame in _frames(player):
            frames[_frame_idx(frame)] = _keypoints_by_joint(frame)
        if player_id:
            out[player_id] = frames
    return out


def _jitter_metrics(
    skeleton_by_player: Mapping[str, Mapping[int, Mapping[str, Any]]],
    joint_names: Sequence[str],
) -> dict[str, Any]:
    group_values: dict[str, list[float]] = defaultdict(list)
    joint_values: dict[str, list[float]] = defaultdict(list)
    for player_id, frames in skeleton_by_player.items():
        del player_id
        previous: Mapping[str, Any] | None = None
        for _frame_idx_value, frame in sorted(frames.items()):
            if previous is None:
                previous = frame
                continue
            prev_joints = _joints_world(previous)
            joints = _joints_world(frame)
            for index, joint_name in enumerate(joint_names):
                if index >= len(prev_joints) or index >= len(joints):
                    continue
                if not _finite_vec(prev_joints[index], length=3) or not _finite_vec(joints[index], length=3):
                    continue
                value = math.dist(prev_joints[index], joints[index])
                joint_values[joint_name].append(value)
                group_values["all"].append(value)
                group_values[_joint_group(joint_name)].append(value)
            previous = frame
    if "all" not in group_values:
        group_values["all"] = []
    return {
        "groups": {group: _distribution(values) for group, values in sorted(group_values.items())},
        "by_joint": {joint: _distribution(values) for joint, values in sorted(joint_values.items())},
    }


def _stature_stability(
    skeleton_by_player: Mapping[str, Mapping[int, Mapping[str, Any]]],
    joint_names: Sequence[str],
) -> dict[str, Any]:
    head_indices = [index for index, name in enumerate(joint_names) if name in HEAD_JOINT_NAMES]
    foot_indices = [index for index, name in enumerate(joint_names) if name in FOOT_JOINT_NAMES]
    if not head_indices:
        head_indices = list(range(len(joint_names)))
    if not foot_indices:
        foot_indices = list(range(len(joint_names)))
    players: dict[str, Any] = {}
    all_statures: list[float] = []
    for player_id, frames in sorted(skeleton_by_player.items()):
        statures: list[float] = []
        for frame in frames.values():
            joints = _joints_world(frame)
            if not joints:
                continue
            head_z = [joints[index][2] for index in head_indices if index < len(joints) and _finite_vec(joints[index], length=3)]
            foot_z = [joints[index][2] for index in foot_indices if index < len(joints) and _finite_vec(joints[index], length=3)]
            if not head_z or not foot_z:
                continue
            statures.append(max(head_z) - min(foot_z))
        med = median(statures) if statures else None
        deviations = [abs(value - med) for value in statures] if med is not None else []
        players[player_id] = {
            "frame_count": len(statures),
            "median_stature_m": med,
            "stature_p50_abs_dev_m": _percentile(deviations, 50),
            "stature_p90_abs_dev_m": _percentile(deviations, 90),
        }
        all_statures.extend(statures)
    all_med = median(all_statures) if all_statures else None
    all_devs = [abs(value - all_med) for value in all_statures] if all_med is not None else []
    return {
        "overall": {
            "frame_count": len(all_statures),
            "median_stature_m": all_med,
            "stature_p50_abs_dev_m": _percentile(all_devs, 50),
            "stature_p90_abs_dev_m": _percentile(all_devs, 90),
        },
        "players": players,
    }


def _bone_length_variance(
    skeleton_payload: Mapping[str, Any],
    keypoints_payload: Mapping[str, Any],
    joint_names: Sequence[str],
) -> dict[str, Any]:
    index_by_name = {name: index for index, name in enumerate(joint_names)}
    skeleton_by_player = _skeleton_by_player_frame(skeleton_payload)
    bone_values: dict[str, list[float]] = defaultdict(list)
    priors_by_player: dict[str, list[BonePrior]] = {}
    for player in _players(skeleton_payload):
        player_id = str(player.get("id") or player.get("player_id"))
        keypoint_player = _matching_player(keypoints_payload, player_id)
        height_m = float(player.get("height_m") or (keypoint_player or {}).get("height_m") or 1.72)
        priors_by_player[player_id] = _bone_priors_for_player(
            keypoints_payload,
            keypoint_player or {},
            Lift2DConfig(),
            joint_names=joint_names,
            height_m=height_m,
        )
    for player_id, frames in skeleton_by_player.items():
        priors = priors_by_player.get(player_id, [])
        for frame in frames.values():
            joints = _joints_world(frame)
            for prior in priors:
                parent_idx = index_by_name.get(prior.parent)
                child_idx = index_by_name.get(prior.child)
                if parent_idx is None or child_idx is None or parent_idx >= len(joints) or child_idx >= len(joints):
                    continue
                if not _finite_vec(joints[parent_idx], length=3) or not _finite_vec(joints[child_idx], length=3):
                    continue
                bone_values[f"{prior.parent}->{prior.child}"].append(math.dist(joints[parent_idx], joints[child_idx]))
    bones = {name: _variance_stats(values) for name, values in sorted(bone_values.items())}
    cvs = [stats["cv"] for stats in bones.values() if stats["cv"] is not None]
    return {
        "summary": {
            "bone_count": len(bones),
            "max_cv": max(cvs) if cvs else None,
            "median_cv": median(cvs) if cvs else None,
        },
        "bones": bones,
    }


def _matching_player(payload: Mapping[str, Any], player_id: str) -> Mapping[str, Any] | None:
    for player in _players(payload):
        if str(player.get("id") or player.get("player_id")) == str(player_id):
            return player
    return None


def _variance_stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean_m": None, "std_m": None, "cv": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    return {
        "count": len(values),
        "mean_m": mean,
        "std_m": std,
        "cv": std / mean if abs(mean) > 1e-12 else None,
        "p50_m": _percentile(values, 50),
        "p90_m": _percentile(values, 90),
    }


def _joint_group(joint_name: str) -> str:
    if joint_name in HEAD_JOINT_NAMES:
        return "head"
    if joint_name in ARM_JOINT_NAMES:
        return "arms"
    if joint_name in TORSO_JOINT_NAMES:
        return "torso"
    if joint_name in LOWER_BODY_JOINT_NAMES:
        return "lower_body"
    return "other"


def _joints_world(frame: Mapping[str, Any]) -> list[list[float]]:
    raw = frame.get("joints_world") or frame.get("joints")
    if not isinstance(raw, Sequence):
        return []
    out: list[list[float]] = []
    for point in raw:
        if isinstance(point, Sequence) and len(point) >= 3:
            out.append([float(point[0]), float(point[1]), float(point[2])])
    return out


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "max": max(values) if values else None,
    }


def _percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(pct) / 100.0
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[int(position)]
    frac = position - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _finite_vec(values: Sequence[float], *, length: int) -> bool:
    return len(values) >= length and all(math.isfinite(float(values[index])) for index in range(length))


__all__ = ["score_skeleton_alignment", "ARTIFACT_TYPE"]
