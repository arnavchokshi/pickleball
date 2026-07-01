"""Splice Lane B contact mesh joints into continuous Lane A skeletons."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from .skeleton3d import SAM3D_BODY_MHR70_SEMANTIC_MAP


ARTIFACT_TYPE = "racketsport_contact_splice"
DEFAULT_MESH_SOURCE = "body_mesh.json"


def splice_contact_skeleton_with_body_mesh(
    skeleton3d: Mapping[str, Any],
    *,
    body_mesh: Mapping[str, Any],
    body_compute_execution: Mapping[str, Any],
    fallback_skeleton3d: Mapping[str, Any] | None = None,
    override_joint_names: Sequence[str] | None = None,
    mesh_source: str = DEFAULT_MESH_SOURCE,
    fallback_source: str = "body_pose_fallback.json",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a skeleton with scheduled hitter contact joints overridden by mesh joints.

    The splice is intentionally conservative: it only touches scheduled BODY
    player-frames, only overrides joints with a semantic name match, and leaves
    the Lane A skeleton untouched when mesh joints are unavailable.
    """

    output = deepcopy(dict(skeleton3d))
    skeleton_joint_names = _string_list(output.get("joint_names", []))
    skeleton_index = {name: idx for idx, name in enumerate(skeleton_joint_names)}
    requested_names = _override_joint_names(skeleton_joint_names, override_joint_names)
    skeleton_frames = _skeleton_frame_lookup(output)
    mesh_frames = _mesh_frame_lookup(body_mesh)
    mesh_joint_names = _string_list(body_mesh.get("joint_names", []))
    mesh_index = _mesh_joint_index(mesh_joint_names)
    fallback_frames = _skeleton_frame_lookup(dict(fallback_skeleton3d)) if isinstance(fallback_skeleton3d, Mapping) else {}
    fallback_joint_names = _string_list(fallback_skeleton3d.get("joint_names", [])) if isinstance(fallback_skeleton3d, Mapping) else []
    fallback_index = {name: idx for idx, name in enumerate(fallback_joint_names)}

    events: list[dict[str, Any]] = []
    spliced_contact_count = 0
    mesh_unavailable_count = 0
    fallback_spliced_count = 0
    overridden_joint_count = 0
    scheduled = _scheduled_targets(body_compute_execution)

    for target in scheduled:
        frame_idx = target["frame_idx"]
        player_id = target["player_id"]
        skeleton_frame = skeleton_frames.get((frame_idx, player_id))
        mesh_frame = mesh_frames.get((frame_idx, player_id))
        base_event = {
            "frame_idx": frame_idx,
            "player_id": player_id,
            "source_window_index": target.get("source_window_index"),
            "reasons": list(target.get("reasons", [])),
        }
        if skeleton_frame is None:
            mesh_unavailable_count += 1
            events.append({**base_event, "status": "mesh_unavailable", "mesh_unavailable": True, "overridden_joint_names": []})
            continue
        if not _mesh_frame_has_usable_joints(mesh_frame):
            mesh_unavailable_count += 1
            fallback_frame = fallback_frames.get((frame_idx, player_id))
            fallback_names = _override_joints_from_source(
                skeleton_frame=skeleton_frame,
                skeleton_index=skeleton_index,
                source_frame=fallback_frame,
                source_index=fallback_index,
                requested_names=requested_names,
            )
            if fallback_names:
                fallback_spliced_count += 1
                overridden_joint_count += len(fallback_names)
                events.append(
                    {
                        **base_event,
                        "status": "mesh_unavailable_pose_fallback",
                        "mesh_unavailable": True,
                        "fallback_source": fallback_source,
                        "overridden_joint_names": fallback_names,
                    }
                )
            else:
                events.append({**base_event, "status": "mesh_unavailable", "mesh_unavailable": True, "overridden_joint_names": []})
            continue

        overridden_names = _override_joints_from_source(
            skeleton_frame=skeleton_frame,
            skeleton_index=skeleton_index,
            source_frame=mesh_frame,
            source_index=mesh_index,
            requested_names=requested_names,
        )

        if overridden_names:
            spliced_contact_count += 1
            overridden_joint_count += len(overridden_names)
            events.append({**base_event, "status": "spliced", "mesh_unavailable": False, "overridden_joint_names": overridden_names})
        else:
            mesh_unavailable_count += 1
            events.append({**base_event, "status": "mesh_joint_unavailable", "mesh_unavailable": True, "overridden_joint_names": []})

    report = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "mesh_source": mesh_source,
        "override_joint_names": requested_names,
        "events": events,
        "summary": {
            "scheduled_contact_count": len(scheduled),
            "spliced_contact_count": spliced_contact_count,
            "mesh_unavailable_count": mesh_unavailable_count,
            "fallback_spliced_count": fallback_spliced_count,
            "overridden_joint_count": overridden_joint_count,
        },
    }
    provenance = dict(output.get("provenance", {}))
    provenance["contact_splice"] = {
        "artifact_type": ARTIFACT_TYPE,
        "mesh_source": mesh_source,
        "override_joint_names": requested_names,
        **report["summary"],
    }
    output["provenance"] = provenance
    return output, report


def _override_joints_from_source(
    *,
    skeleton_frame: dict[str, Any],
    skeleton_index: Mapping[str, int],
    source_frame: Mapping[str, Any] | None,
    source_index: Mapping[str, int],
    requested_names: Sequence[str],
) -> list[str]:
    if not isinstance(source_frame, Mapping):
        return []
    source_joints = source_frame.get("joints_world", [])
    if not isinstance(source_joints, list) or not source_joints:
        return []
    joints_world = skeleton_frame.get("joints_world")
    if not isinstance(joints_world, list):
        return []
    joint_conf = skeleton_frame.get("joint_conf")
    source_conf = source_frame.get("joint_conf")
    overridden_names: list[str] = []
    for name in requested_names:
        skeleton_joint_idx = skeleton_index.get(name)
        source_joint_idx = source_index.get(name)
        if skeleton_joint_idx is None or source_joint_idx is None:
            continue
        if skeleton_joint_idx >= len(joints_world) or source_joint_idx >= len(source_joints):
            continue
        joints_world[skeleton_joint_idx] = _vector3(source_joints[source_joint_idx])
        if (
            isinstance(joint_conf, list)
            and skeleton_joint_idx < len(joint_conf)
            and isinstance(source_conf, list)
            and source_joint_idx < len(source_conf)
        ):
            joint_conf[skeleton_joint_idx] = float(source_conf[source_joint_idx])
        overridden_names.append(name)
    return overridden_names


def _override_joint_names(skeleton_joint_names: Sequence[str], override_joint_names: Sequence[str] | None) -> list[str]:
    if override_joint_names is not None:
        candidates = [str(name) for name in override_joint_names]
    else:
        candidates = [
            name
            for name in skeleton_joint_names
            if name in {"left_wrist", "right_wrist"} or name.startswith("left_hand_") or name.startswith("right_hand_")
        ]
    skeleton_names = set(skeleton_joint_names)
    return [name for name in candidates if name in skeleton_names]


def _skeleton_frame_lookup(skeleton3d: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    fps = float(skeleton3d.get("fps") or 30.0)
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for player in skeleton3d.get("players", []):
        if not isinstance(player, dict):
            continue
        player_id = int(player.get("id", 0))
        for frame in player.get("frames", []):
            if not isinstance(frame, dict):
                continue
            frame_idx = _frame_idx(frame, fps=fps)
            lookup[(frame_idx, player_id)] = frame
    return lookup


def _mesh_frame_lookup(body_mesh: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    fps = float(body_mesh.get("fps") or 30.0)
    lookup: dict[tuple[int, int], Mapping[str, Any]] = {}
    for player in body_mesh.get("players", []):
        if not isinstance(player, Mapping):
            continue
        player_id = int(player.get("id", 0))
        for frame in player.get("frames", []):
            if not isinstance(frame, Mapping):
                continue
            frame_idx = _frame_idx(frame, fps=fps)
            lookup[(frame_idx, player_id)] = frame
    return lookup


def _scheduled_targets(body_compute_execution: Mapping[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for frame in body_compute_execution.get("scheduled_frames", []):
        if not isinstance(frame, Mapping):
            continue
        frame_idx = int(frame.get("frame_idx", -1))
        if frame_idx < 0:
            continue
        for player_id in frame.get("target_player_ids", []):
            targets.append(
                {
                    "frame_idx": frame_idx,
                    "player_id": int(player_id),
                    "source_window_index": frame.get("source_window_index"),
                    "reasons": list(frame.get("reasons", [])) if isinstance(frame.get("reasons"), list) else [],
                }
            )
    return targets


def _mesh_joint_index(mesh_joint_names: Sequence[str]) -> dict[str, int]:
    direct = {name: idx for idx, name in enumerate(mesh_joint_names)}
    if _looks_like_sam3d_body_mhr70(mesh_joint_names):
        for name, idx in SAM3D_BODY_MHR70_SEMANTIC_MAP.joints.items():
            direct.setdefault(name, idx)
    return direct


def _looks_like_sam3d_body_mhr70(mesh_joint_names: Sequence[str]) -> bool:
    if len(mesh_joint_names) != SAM3D_BODY_MHR70_SEMANTIC_MAP.source_joint_count:
        return False
    return all(name == f"sam3dbody_joint_{idx:03d}" for idx, name in enumerate(mesh_joint_names))


def _mesh_frame_has_usable_joints(frame: Mapping[str, Any] | None) -> bool:
    if not isinstance(frame, Mapping):
        return False
    joints = frame.get("joints_world")
    return isinstance(joints, list) and bool(joints) and bool(frame.get("mesh_vertices_world"))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _frame_idx(frame: Mapping[str, Any], *, fps: float) -> int:
    raw = frame.get("frame_idx")
    if raw is not None:
        return int(raw)
    return int(round(float(frame.get("t", 0.0)) * fps))


def _vector3(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("mesh joints_world entries must be 3-vectors")
    return [float(component) for component in value]


__all__ = ["splice_contact_skeleton_with_body_mesh"]
