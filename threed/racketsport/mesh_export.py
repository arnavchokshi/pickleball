"""Exports dedicated Lane B body mesh artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


ARTIFACT_TYPE = "racketsport_body_mesh"
DEFAULT_FACES_REF = "mhr_faces_static"


def build_body_mesh_export(
    smpl_motion: Mapping[str, Any],
    *,
    clip: str,
    body_compute_execution: Mapping[str, Any] | None = None,
    faces_ref: str = DEFAULT_FACES_REF,
) -> dict[str, Any]:
    scheduled = _scheduled_targets(body_compute_execution)
    windows = _scheduled_windows(body_compute_execution)
    joint_names = _joint_names(smpl_motion)
    mesh_faces = _mesh_faces(smpl_motion)
    players_payload: list[dict[str, Any]] = []
    contact_window_indexes: set[int] = set()
    mesh_frame_count = 0
    for player in smpl_motion.get("players", []):
        if not isinstance(player, Mapping):
            continue
        player_id = int(player.get("id", 0))
        betas = _float_list(player.get("betas", []))
        frames_payload: list[dict[str, Any]] = []
        for frame in player.get("frames", []):
            if not isinstance(frame, Mapping):
                continue
            frame_idx = int(frame.get("frame_idx", round(float(frame.get("t", 0.0)) * float(smpl_motion.get("fps", 30.0)))))
            scheduled_record = scheduled.get((frame_idx, player_id))
            if scheduled and scheduled_record is None:
                continue
            vertices = frame.get("mesh_vertices_world", [])
            if not isinstance(vertices, list) or not vertices:
                continue
            source_window_index = scheduled_record.get("source_window_index") if scheduled_record else None
            if source_window_index is not None:
                contact_window_indexes.add(int(source_window_index))
            frame_payload = {
                "frame_idx": frame_idx,
                "t": float(frame.get("t", frame_idx / float(smpl_motion.get("fps", 30.0)))),
                "source_window_index": source_window_index,
                "blend_weight": _blend_weight_for_frame(frame_idx, scheduled_record),
                "mesh_vertices_world": [[float(value) for value in vertex] for vertex in vertices],
                "smplx_params": {
                    "global_orient": _float_list(frame.get("global_orient", [])),
                    "body_pose": _float_list(frame.get("body_pose", [])),
                    "left_hand_pose": _float_list(frame.get("left_hand_pose", [])),
                    "right_hand_pose": _float_list(frame.get("right_hand_pose", [])),
                    "betas": betas,
                    "transl_world": _float_list(frame.get("transl_world", [])),
                },
                "reasons": list(scheduled_record.get("reasons", [])) if scheduled_record else [],
            }
            joints_world = _vector3_list(frame.get("joints_world", []))
            if joints_world:
                frame_payload["joints_world"] = joints_world
            joint_conf = _float_list(frame.get("joint_conf", []))
            if joint_conf:
                frame_payload["joint_conf"] = joint_conf
            frames_payload.append(frame_payload)
        if frames_payload:
            mesh_frame_count += len(frames_payload)
            players_payload.append({"id": player_id, "frames": frames_payload})

    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "clip": clip,
        "model": str(smpl_motion.get("model", "")),
        "fps": float(smpl_motion.get("fps", 0.0)),
        "world_frame": str(smpl_motion.get("world_frame", "")),
        "faces_ref": faces_ref,
        "mesh_faces": mesh_faces,
        "joint_names": joint_names,
        "windows": windows,
        "players": players_payload,
        "summary": {
            "mesh_frame_count": mesh_frame_count,
            "player_count": len(players_payload),
            "contact_window_count": len(contact_window_indexes) if scheduled else 0,
        },
    }


def write_body_mesh_export(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _scheduled_targets(body_compute_execution: Mapping[str, Any] | None) -> dict[tuple[int, int], dict[str, Any]]:
    if body_compute_execution is None:
        return {}
    scheduled: dict[tuple[int, int], dict[str, Any]] = {}
    for frame in body_compute_execution.get("scheduled_frames", []):
        if not isinstance(frame, Mapping):
            continue
        frame_idx = int(frame.get("frame_idx", -1))
        for player_id in frame.get("target_player_ids", []):
            scheduled[(frame_idx, int(player_id))] = dict(frame)
    return scheduled


def _scheduled_windows(body_compute_execution: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if body_compute_execution is None:
        return []
    fps = float(body_compute_execution.get("fps") or 30.0)
    windows: dict[int, dict[str, Any]] = {}
    for frame in body_compute_execution.get("scheduled_frames", []):
        if not isinstance(frame, Mapping):
            continue
        raw_index = frame.get("source_window_index")
        if raw_index is None:
            continue
        source_window_index = int(raw_index)
        frame_idx = int(frame.get("frame_idx", -1))
        if frame_idx < 0:
            continue
        frame_start = int(frame.get("window_frame_start", frame_idx))
        frame_end = int(frame.get("window_frame_end", frame_idx))
        target_player_ids = sorted({int(player_id) for player_id in frame.get("target_player_ids", [])})
        window = windows.setdefault(
            source_window_index,
            {
                "source_window_index": source_window_index,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "t0": float(frame.get("window_t0", frame_start / fps)),
                "t1": float(frame.get("window_t1", (frame_end + 1) / fps)),
                "frame_count": int(frame.get("window_frame_count", max(0, frame_end - frame_start + 1))),
                "target_player_ids": [],
                "target_representation": str(frame.get("target_representation", "world_mesh")),
                "fallback_representation": str(frame.get("fallback_representation", "lane_a_skeleton")),
                "reason_counts": {},
                "max_score": float(frame.get("max_score", 0.0) or 0.0),
            },
        )
        window["frame_start"] = min(int(window["frame_start"]), frame_start)
        window["frame_end"] = max(int(window["frame_end"]), frame_end)
        window["t0"] = min(float(window["t0"]), float(frame.get("window_t0", frame_start / fps)))
        window["t1"] = max(float(window["t1"]), float(frame.get("window_t1", (frame_end + 1) / fps)))
        window["frame_count"] = max(int(window["frame_count"]), int(frame.get("window_frame_count", frame_end - frame_start + 1)))
        window["target_player_ids"] = sorted(set(window["target_player_ids"]) | set(target_player_ids))
        window["max_score"] = max(float(window["max_score"]), float(frame.get("max_score", 0.0) or 0.0))
        reason_counts = frame.get("reason_counts")
        if isinstance(reason_counts, Mapping) and reason_counts:
            window["reason_counts"] = {str(reason): int(count) for reason, count in reason_counts.items()}
        elif not window["reason_counts"]:
            counts: dict[str, int] = {}
            for reason in frame.get("reasons", []):
                reason_key = str(reason)
                counts[reason_key] = counts.get(reason_key, 0) + 1
            window["reason_counts"] = counts
    return [windows[index] for index in sorted(windows)]


def _blend_weight_for_frame(frame_idx: int, scheduled_record: Mapping[str, Any] | None) -> float:
    if scheduled_record is None:
        return 1.0
    frame_start = int(scheduled_record.get("window_frame_start", frame_idx))
    frame_end = int(scheduled_record.get("window_frame_end", frame_idx))
    if frame_end <= frame_start:
        return 1.0
    progress = (frame_idx - frame_start) / (frame_end - frame_start)
    edge_progress = max(0.0, min(1.0, 1.0 - abs((progress * 2.0) - 1.0)))
    weight = 0.5 - (0.5 * math.cos(math.pi * edge_progress))
    return round(weight, 6)


def _float_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    return [float(value) for value in values]


def _vector3_list(values: Any) -> list[list[float]]:
    if not isinstance(values, list):
        return []
    vectors: list[list[float]] = []
    for value in values:
        if not isinstance(value, list) or len(value) != 3:
            return []
        vectors.append([float(component) for component in value])
    return vectors


def _mesh_faces(payload: Mapping[str, Any]) -> list[list[int]]:
    faces = payload.get("mesh_faces", payload.get("faces", []))
    if not isinstance(faces, list):
        return []
    parsed: list[list[int]] = []
    for face in faces:
        if not isinstance(face, list) or len(face) != 3:
            return []
        parsed_face: list[int] = []
        for value in face:
            index = int(value)
            if index < 0:
                return []
            parsed_face.append(index)
        parsed.append(parsed_face)
    return parsed


def _joint_names(smpl_motion: Mapping[str, Any]) -> list[str]:
    explicit = smpl_motion.get("joint_names")
    if isinstance(explicit, list) and all(isinstance(name, str) and name for name in explicit):
        return list(explicit)
    for player in smpl_motion.get("players", []):
        if not isinstance(player, Mapping):
            continue
        for frame in player.get("frames", []):
            if not isinstance(frame, Mapping):
                continue
            joints_world = frame.get("joints_world")
            if isinstance(joints_world, list) and len(joints_world) == 70:
                return [f"sam3dbody_joint_{idx:03d}" for idx in range(70)]
    return []


__all__ = ["build_body_mesh_export", "write_body_mesh_export"]
