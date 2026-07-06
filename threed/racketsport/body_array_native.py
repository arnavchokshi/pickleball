"""Array-native BODY artifact views for slim SAM-3D-Body runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import mesh_export as _mesh_export
from . import worldhmr as _worldhmr
from .schemas import CourtCalibration


@dataclass(frozen=True)
class BodyArrayNativeArtifacts:
    smpl_motion_view: dict[str, Any]
    skeleton3d: dict[str, Any]
    grounding_metrics: dict[str, Any]
    body_mesh_metadata: dict[str, Any]
    body_mesh_players: list[dict[str, Any]]
    body_mesh_summary: dict[str, Any]


def build_body_array_native_artifacts_from_fast_sam(
    samples: Sequence[Mapping[str, Any]],
    *,
    calibration: CourtCalibration,
    fps: float,
    clip: str,
    body_compute_execution: Mapping[str, Any] | None,
    smoothing_alpha: float = 0.65,
    max_root_speed_mps: float | None = None,
    max_track_anchor_smoothing_residual_m: float | None = None,
    model: str = "sam3dbody_world_joints",
    sam3d_wrist_bone_lock: bool = True,
    stance_index: Mapping[tuple[int | str, int], Mapping[str, Any]] | None = None,
    grounding_anchor_source: str | None = None,
    camera_motion_path: str | None = None,
    smoothing_gap_carry_frames: int = _worldhmr.DEFAULT_SMOOTHING_GAP_CARRY_FRAMES,
    smoothing_residual_identity_reset_m: float = _worldhmr.DEFAULT_SMOOTHING_RESIDUAL_IDENTITY_RESET_M,
    world_joint_visual_smoothing: bool = True,
) -> BodyArrayNativeArtifacts:
    """Build slim BODY gate/index inputs without materializing monolithic payloads."""

    computed = _worldhmr.compute_body_skeleton_and_metrics(
        samples,
        calibration=calibration,
        fps=fps,
        smoothing_alpha=smoothing_alpha,
        max_root_speed_mps=max_root_speed_mps,
        max_track_anchor_smoothing_residual_m=max_track_anchor_smoothing_residual_m,
        model=model,
        sam3d_wrist_bone_lock=sam3d_wrist_bone_lock,
        stance_index=stance_index,
        grounding_anchor_source=grounding_anchor_source,
        camera_motion_path=camera_motion_path,
        smoothing_gap_carry_frames=smoothing_gap_carry_frames,
        smoothing_residual_identity_reset_m=smoothing_residual_identity_reset_m,
        world_joint_visual_smoothing=world_joint_visual_smoothing,
    )
    metadata, body_mesh_players, body_mesh_summary = body_mesh_export_parts_from_smpl_motion_view(
        computed.smpl_motion_view,
        clip=clip,
        body_compute_execution=body_compute_execution,
    )
    return BodyArrayNativeArtifacts(
        smpl_motion_view=computed.smpl_motion_view,
        skeleton3d=computed.skeleton3d,
        grounding_metrics=computed.metrics,
        body_mesh_metadata=metadata,
        body_mesh_players=body_mesh_players,
        body_mesh_summary=body_mesh_summary,
    )


def body_mesh_export_parts_from_smpl_motion_view(
    smpl_motion: Mapping[str, Any],
    *,
    clip: str,
    body_compute_execution: Mapping[str, Any] | None,
    faces_ref: str = _mesh_export.DEFAULT_FACES_REF,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    scheduled = _mesh_export._scheduled_targets(body_compute_execution)
    windows = _mesh_export._scheduled_windows(body_compute_execution)
    joint_names = _mesh_export._joint_names(smpl_motion)
    mesh_faces = _mesh_export._mesh_faces(smpl_motion)
    players_payload: list[dict[str, Any]] = []
    contact_window_indexes: set[int] = set()
    mesh_frame_count = 0
    for player in smpl_motion.get("players", []):
        if not isinstance(player, Mapping):
            continue
        player_id = int(player.get("id", 0))
        betas = _mesh_export._float_list(player.get("betas", []))
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
                "blend_weight": _mesh_export._blend_weight_for_frame(frame_idx, scheduled_record),
                "mesh_vertices_world": vertices,
                "smplx_params": {
                    "global_orient": _mesh_export._float_list(frame.get("global_orient", [])),
                    "body_pose": _mesh_export._float_list(frame.get("body_pose", [])),
                    "left_hand_pose": _mesh_export._float_list(frame.get("left_hand_pose", [])),
                    "right_hand_pose": _mesh_export._float_list(frame.get("right_hand_pose", [])),
                    "betas": betas,
                    "transl_world": _mesh_export._float_list(frame.get("transl_world", [])),
                },
                "reasons": list(scheduled_record.get("reasons", [])) if scheduled_record else [],
            }
            joints_world = _mesh_export._vector3_list(frame.get("joints_world", []))
            if joints_world:
                frame_payload["joints_world"] = joints_world
            joint_conf = _mesh_export._float_list(frame.get("joint_conf", []))
            if joint_conf:
                frame_payload["joint_conf"] = joint_conf
            frames_payload.append(frame_payload)
        if frames_payload:
            mesh_frame_count += len(frames_payload)
            players_payload.append({"id": player_id, "frames": frames_payload})
    metadata = {
        "clip": clip,
        "model": str(smpl_motion.get("model", "")),
        "fps": float(smpl_motion.get("fps", 0.0)),
        "world_frame": str(smpl_motion.get("world_frame", "")),
        "faces_ref": faces_ref,
        "mesh_faces": mesh_faces,
        "joint_names": joint_names,
        "windows": windows,
    }
    summary = {
        "mesh_frame_count": mesh_frame_count,
        "player_count": len(players_payload),
        "contact_window_count": len(contact_window_indexes) if scheduled else 0,
    }
    return metadata, players_payload, summary


def body_mesh_payload_from_parts(
    metadata: Mapping[str, Any],
    players: Sequence[Mapping[str, Any]],
    *,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_body_mesh",
        "clip": metadata["clip"],
        "model": metadata["model"],
        "fps": metadata["fps"],
        "world_frame": metadata["world_frame"],
        "faces_ref": metadata["faces_ref"],
        "mesh_faces": metadata.get("mesh_faces", []),
        "joint_names": metadata.get("joint_names", []),
        "windows": metadata.get("windows", []),
        "players": [dict(player) for player in players],
        "summary": dict(summary),
    }


__all__ = [
    "BodyArrayNativeArtifacts",
    "body_mesh_export_parts_from_smpl_motion_view",
    "body_mesh_payload_from_parts",
    "build_body_array_native_artifacts_from_fast_sam",
]
