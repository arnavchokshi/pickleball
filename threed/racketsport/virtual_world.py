"""Assemble inspectable court_Z0 world-state artifacts for replay review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from .court_calibration import project_image_points_to_world
from .court_templates import get_court_template
from .racket6dof import SE3PoseConfidence, camera_paddle_pose_to_court_world, paddle_face_corners_object_cm
from .racket_true_corners import is_box_derived_source
from .schemas import (
    BallTrack,
    CourtCalibration,
    RacketPose,
    Skeleton3D,
    SmplMotion,
    Tracks,
    VirtualWorld,
    validate_artifact_file,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_virtual_world"
WORLD_FRAME = "court_Z0"


def build_virtual_world_state(
    *,
    court_calibration: CourtCalibration | Mapping[str, Any],
    tracks: Tracks | Mapping[str, Any] | None = None,
    smpl_motion: SmplMotion | Mapping[str, Any] | None = None,
    skeleton3d: Skeleton3D | Mapping[str, Any] | None = None,
    ball_track: BallTrack | Mapping[str, Any] | None = None,
    racket_pose: RacketPose | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one inspectable world artifact from already-produced stage outputs."""

    calibration = _court_calibration(court_calibration)
    tracks_obj = _tracks(tracks)
    smpl_obj = _smpl_motion(smpl_motion)
    skeleton_obj = _skeleton3d(skeleton3d)
    ball_obj = _ball_track(ball_track)
    racket_obj = _racket_pose(racket_pose)

    fps = _world_fps(tracks_obj, smpl_obj, skeleton_obj, ball_obj, racket_obj)
    players = _players(tracks_obj=tracks_obj, smpl_obj=smpl_obj, skeleton_obj=skeleton_obj)
    ball = _ball(ball_obj, calibration=calibration)
    paddles, paddle_warnings = _paddles(racket_obj, calibration=calibration)
    warnings = [*_warnings(players=players, ball=ball, paddles=paddles), *paddle_warnings]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "world_frame": WORLD_FRAME,
        "fps": fps,
        "court": _court(calibration),
        "players": players,
        "ball": ball,
        "paddles": paddles,
        "summary": _summary(players=players, ball=ball, paddles=paddles, warnings=warnings),
    }
    return VirtualWorld.model_validate(payload).model_dump(mode="json")


def build_virtual_world_state_from_files(
    *,
    court_calibration_path: str | Path,
    tracks_path: str | Path | None = None,
    smpl_motion_path: str | Path | None = None,
    skeleton3d_path: str | Path | None = None,
    ball_track_path: str | Path | None = None,
    racket_pose_path: str | Path | None = None,
) -> dict[str, Any]:
    calibration = validate_artifact_file("court_calibration", Path(court_calibration_path))
    if not isinstance(calibration, CourtCalibration):
        raise ValueError("court calibration artifact did not parse as CourtCalibration")
    return build_virtual_world_state(
        court_calibration=calibration,
        tracks=_optional_artifact("tracks", tracks_path, Tracks),
        smpl_motion=_optional_artifact("smpl_motion", smpl_motion_path, SmplMotion),
        skeleton3d=_optional_artifact("skeleton3d", skeleton3d_path, Skeleton3D),
        ball_track=_optional_artifact("ball_track", ball_track_path, BallTrack),
        racket_pose=_optional_artifact("racket_pose", racket_pose_path, RacketPose),
    )


def write_virtual_world(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _optional_artifact(artifact: str, path: str | Path | None, model: type[Any]) -> Any | None:
    if path is None:
        return None
    parsed = validate_artifact_file(artifact, Path(path))
    if not isinstance(parsed, model):
        raise ValueError(f"{artifact} artifact did not parse as {model.__name__}")
    return parsed


def _court(calibration: CourtCalibration) -> dict[str, Any]:
    template = get_court_template(calibration.sport)
    line_segments = {
        line_id: [list(start), list(end)]
        for line_id, (start, end) in template.line_segments_m.items()
    }
    net_start, net_end = template.line_segments_m["net"]
    return {
        "sport": calibration.sport,
        "coordinate_frame": template.coordinate_frame,
        "length_m": template.length_m,
        "width_m": template.width_m,
        "line_segments": line_segments,
        "net": {
            "endpoints": [list(net_start), list(net_end)],
            "center_height_m": template.center_net_height_m,
            "post_height_m": template.post_net_height_m,
        },
    }


def _players(
    *,
    tracks_obj: Tracks | None,
    smpl_obj: SmplMotion | None,
    skeleton_obj: Skeleton3D | None,
) -> list[dict[str, Any]]:
    track_meta, track_frames = _track_lookup(tracks_obj)
    if smpl_obj is not None:
        players = []
        smpl_players = {player.id: player for player in smpl_obj.players}
        player_ids = sorted(set(track_meta) | set(smpl_players))
        for player_id in player_ids:
            player = smpl_players.get(player_id)
            smpl_frames = {_time_key(frame.t): frame for frame in player.frames} if player is not None else {}
            frame_keys = sorted(
                {
                    key
                    for key_player_id, key in track_frames
                    if key_player_id == player_id
                }
                | set(smpl_frames)
            )
            frames = []
            for frame_key in frame_keys:
                track_frame = track_frames.get((player_id, frame_key))
                smpl_frame = smpl_frames.get(frame_key)
                frames.append(
                    _player_frame_from_sources(
                        track_frame=track_frame,
                        smpl_frame=smpl_frame,
                        physics=player.physics if player is not None else None,
                    )
                )
            meta = track_meta.get(player_id, {})
            players.append(
                {
                    "id": player_id,
                    "side": meta.get("side"),
                    "role": meta.get("role"),
                    "representation": _player_representation(frames),
                    "frames": frames,
                }
            )
        return players

    if skeleton_obj is not None:
        return [
            {
                "id": player.id,
                "side": track_meta.get(player.id, {}).get("side"),
                "role": track_meta.get(player.id, {}).get("role"),
                "representation": "joints",
                "frames": [
                    {
                        "t": frame.t,
                        "track_world_xy": None,
                        "track_conf": None,
                        "bbox": None,
                        "transl_world": None,
                        "joints_world": [list(joint) for joint in frame.joints_world],
                        "joint_conf": [float(conf) for conf in frame.joint_conf],
                        "mesh_vertices_world": [],
                        "joint_count": len(frame.joints_world),
                        "mesh_vertex_count": 0,
                        **_empty_floor_fields(),
                    }
                    for frame in player.frames
                ],
            }
            for player in skeleton_obj.players
        ]

    if tracks_obj is None:
        return []
    return [
        {
            "id": player.id,
            "side": player.side,
            "role": player.role,
            "representation": "track_only",
            "frames": [
                {
                    "t": frame.t,
                    "track_world_xy": list(frame.world_xy),
                    "track_conf": float(frame.conf),
                    "bbox": tuple(float(value) for value in frame.bbox),
                    "transl_world": None,
                    "joints_world": [],
                    "joint_conf": [],
                    "mesh_vertices_world": [],
                    "joint_count": 0,
                    "mesh_vertex_count": 0,
                    **_track_floor_fields(frame),
                }
                for frame in player.frames
            ],
        }
        for player in tracks_obj.players
    ]


def _player_frame_from_sources(
    *,
    track_frame: Any | None,
    smpl_frame: Any | None,
    physics: str | None = None,
) -> dict[str, Any]:
    if smpl_frame is None:
        if track_frame is None:
            raise ValueError("player frame requires track_frame or smpl_frame")
        return {
            "t": track_frame.t,
            "track_world_xy": list(track_frame.world_xy),
            "track_conf": float(track_frame.conf),
            "bbox": tuple(float(value) for value in track_frame.bbox),
            "transl_world": None,
            "joints_world": [],
            "joint_conf": [],
            "mesh_vertices_world": [],
            "joint_count": 0,
            "mesh_vertex_count": 0,
            **_track_floor_fields(track_frame),
        }
    mesh_vertices = [list(vertex) for vertex in smpl_frame.mesh_vertices_world]
    return {
        "t": smpl_frame.t,
        "track_world_xy": list(track_frame.world_xy) if track_frame is not None else None,
        "track_conf": float(track_frame.conf) if track_frame is not None else None,
        "bbox": tuple(float(value) for value in track_frame.bbox) if track_frame is not None else None,
        "transl_world": list(smpl_frame.transl_world),
        "joints_world": [list(joint) for joint in smpl_frame.joints_world],
        "joint_conf": [float(conf) for conf in smpl_frame.joint_conf],
        "mesh_vertices_world": mesh_vertices,
        "joint_count": len(smpl_frame.joints_world),
        "mesh_vertex_count": len(mesh_vertices),
        **_smpl_floor_fields(track_frame=track_frame, smpl_frame=smpl_frame, mesh_vertices=mesh_vertices, physics=physics),
    }


def _track_floor_fields(track_frame: Any) -> dict[str, Any]:
    return {
        "floor_world_xyz": [float(track_frame.world_xy[0]), float(track_frame.world_xy[1]), 0.0],
        "floor_source": "track_footpoint",
        "floor_offset_m": 0.0,
        "min_mesh_z_m": None,
        "floor_penetration_m": 0.0,
        "foot_contact": None,
        "contact_locked": False,
        "physics": None,
        "grf": None,
    }


def _smpl_floor_fields(
    *,
    track_frame: Any | None,
    smpl_frame: Any,
    mesh_vertices: list[list[float]],
    physics: str | None,
) -> dict[str, Any]:
    floor_xy: list[float] | None = None
    source = "smpl_world"
    if track_frame is not None:
        floor_xy = [float(track_frame.world_xy[0]), float(track_frame.world_xy[1])]
        source = "track_footpoint+smpl_world"
    elif smpl_frame.transl_world is not None:
        floor_xy = [float(smpl_frame.transl_world[0]), float(smpl_frame.transl_world[1])]

    min_mesh_z = min((float(vertex[2]) for vertex in mesh_vertices), default=None)
    foot_contact = {"left": bool(smpl_frame.foot_contact.left), "right": bool(smpl_frame.foot_contact.right)}
    return {
        "floor_world_xyz": [floor_xy[0], floor_xy[1], 0.0] if floor_xy is not None else None,
        "floor_source": source if floor_xy is not None else None,
        "floor_offset_m": float(smpl_frame.transl_world[2]) if smpl_frame.transl_world is not None else None,
        "min_mesh_z_m": min_mesh_z,
        "floor_penetration_m": max(0.0, -min_mesh_z) if min_mesh_z is not None else 0.0,
        "foot_contact": foot_contact,
        "contact_locked": bool(foot_contact["left"] or foot_contact["right"]),
        "physics": physics,
        "grf": [list(vector) for vector in smpl_frame.grf] if smpl_frame.grf is not None else None,
    }


def _empty_floor_fields() -> dict[str, Any]:
    return {
        "floor_world_xyz": None,
        "floor_source": None,
        "floor_offset_m": None,
        "min_mesh_z_m": None,
        "floor_penetration_m": 0.0,
        "foot_contact": None,
        "contact_locked": False,
        "physics": None,
        "grf": None,
    }


def _player_representation(frames: list[Mapping[str, Any]]) -> str:
    if any(int(frame["mesh_vertex_count"]) > 0 for frame in frames):
        return "mesh"
    if any(int(frame["joint_count"]) > 0 for frame in frames):
        return "joints"
    return "track_only"


def _track_lookup(tracks_obj: Tracks | None) -> tuple[dict[int, dict[str, str]], dict[tuple[int, str], Any]]:
    if tracks_obj is None:
        return {}, {}
    meta = {player.id: {"side": player.side, "role": player.role} for player in tracks_obj.players}
    frames = {
        (player.id, _time_key(frame.t)): frame
        for player in tracks_obj.players
        for frame in player.frames
    }
    return meta, frames


def _ball(ball_obj: BallTrack | None, *, calibration: CourtCalibration) -> dict[str, Any]:
    if ball_obj is None:
        return {"source": None, "frames": []}
    return {
        "source": ball_obj.source,
        "frames": [_ball_frame(frame, calibration=calibration) for frame in ball_obj.frames],
    }


def _ball_frame(frame: Any, *, calibration: CourtCalibration) -> dict[str, Any]:
    world_xyz = _ball_world_xyz(frame, calibration=calibration)
    return {
        "t": frame.t,
        "xy": list(frame.xy),
        "conf": float(frame.conf),
        "visible": frame.visible,
        "world_xyz": world_xyz,
        "approx": bool(frame.approx or (frame.world_xyz is None and world_xyz is not None)),
    }


def _ball_world_xyz(frame: Any, *, calibration: CourtCalibration) -> list[float] | None:
    if frame.world_xyz is not None:
        return list(frame.world_xyz)
    if not frame.visible:
        return None
    try:
        world_xy = project_image_points_to_world(calibration.homography, [frame.xy])[0]
    except ValueError:
        return None
    if not _world_xy_in_court(calibration, world_xy, margin_m=0.35):
        return None
    return [float(world_xy[0]), float(world_xy[1]), 0.0]


def _world_xy_in_court(calibration: CourtCalibration, world_xy: Any, *, margin_m: float) -> bool:
    template = get_court_template(calibration.sport)
    points = [point for segment in template.line_segments_m.values() for point in segment]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x = float(world_xy[0])
    y = float(world_xy[1])
    return min(xs) - margin_m <= x <= max(xs) + margin_m and min(ys) - margin_m <= y <= max(ys) + margin_m


def _paddles(racket_obj: RacketPose | None, *, calibration: CourtCalibration) -> tuple[list[dict[str, Any]], list[str]]:
    if racket_obj is None:
        return [], []
    paddles = []
    suppressed_box_derived_count = 0
    for player in racket_obj.players:
        paddle_dims = dict(player.paddle_dims_in)
        frames = []
        for frame in player.frames:
            if is_box_derived_source(frame.source):
                suppressed_box_derived_count += 1
                continue
            frames.append(_paddle_frame(frame, calibration=calibration, paddle_dims_in=paddle_dims))
        if frames:
            paddles.append({"player_id": player.id, "paddle_dims_in": dict(player.paddle_dims_in), "frames": frames})
    warnings = ["box_derived_paddle_pose_suppressed"] if suppressed_box_derived_count else []
    return paddles, warnings


def _paddle_frame(frame: Any, *, calibration: CourtCalibration, paddle_dims_in: Mapping[str, float]) -> dict[str, Any]:
    pose = frame.pose_se3
    source = frame.source
    if frame.world_frame == "court_Z0":
        scale = 0.01 if frame.translation_unit == "cm" else 1.0
        pose_se3 = {"R": [list(row) for row in pose.R], "t": [float(value) * scale for value in pose.t]}
        source = source if source.endswith(":court_Z0") else f"{source}:court_Z0"
    else:
        converted = camera_paddle_pose_to_court_world(
            SE3PoseConfidence(R=pose.R, t=pose.t, confidence=frame.conf, source=source),
            calibration,
            input_translation_unit=frame.translation_unit,
        )
        pose_se3 = {"R": [list(row) for row in converted.R], "t": list(converted.t)}
        source = converted.source
    mesh_vertices_world = _paddle_mesh_vertices_world(pose_se3, paddle_dims_in)
    return {
        "t": frame.t,
        "pose_se3": pose_se3,
        "mesh_vertices_world": mesh_vertices_world,
        "mesh_faces": [[0, 1, 2], [0, 2, 3]],
        "conf": float(frame.conf),
        "world_frame": WORLD_FRAME,
        "translation_unit": "m",
        "source": source,
        "reprojection_error_px": frame.reprojection_error_px,
        "ambiguous": frame.ambiguous,
    }


def _paddle_mesh_vertices_world(pose_se3: Mapping[str, Any], paddle_dims_in: Mapping[str, float]) -> list[list[float]]:
    rotation = pose_se3["R"]
    translation = pose_se3["t"]
    vertices = []
    for corner_cm in paddle_face_corners_object_cm(paddle_dims_in):
        corner_m = [float(value) / 100.0 for value in corner_cm]
        vertices.append(
            [
                sum(float(rotation[row][col]) * corner_m[col] for col in range(3)) + float(translation[row])
                for row in range(3)
            ]
        )
    return vertices


def _warnings(*, players: list[dict[str, Any]], ball: dict[str, Any], paddles: list[dict[str, Any]]) -> list[str]:
    warnings = []
    if not players:
        warnings.append("missing_players")
    elif not any(player["representation"] == "mesh" for player in players):
        warnings.append("missing_mesh_vertices")
    if not ball["frames"]:
        warnings.append("missing_ball_track")
    elif any(frame.get("visible") is True and frame.get("world_xyz") is None for frame in ball["frames"]):
        warnings.append("unprojected_visible_ball_frames")
    if not any(paddle["frames"] for paddle in paddles):
        warnings.append("missing_paddle_pose")
    if any(frame.get("ambiguous") for paddle in paddles for frame in paddle["frames"]):
        warnings.append("ambiguous_paddle_pose")
    return warnings


def _summary(
    *,
    players: list[dict[str, Any]],
    ball: dict[str, Any],
    paddles: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    player_frames = [
        frame
        for player in players
        for frame in player["frames"]
    ]
    paddle_frames = [
        frame
        for paddle in paddles
        for frame in paddle["frames"]
    ]
    ball_frames = ball["frames"]
    return {
        "player_count": len(players),
        "mesh_player_count": sum(1 for player in players if player["representation"] == "mesh"),
        "mesh_player_frame_count": sum(1 for frame in player_frames if int(frame["mesh_vertex_count"]) > 0),
        "joint_player_frame_count": sum(1 for frame in player_frames if int(frame["joint_count"]) > 0),
        "track_only_player_frame_count": sum(
            1
            for frame in player_frames
            if frame.get("track_world_xy") is not None
            and int(frame["mesh_vertex_count"]) == 0
            and int(frame["joint_count"]) == 0
        ),
        "floor_placed_player_frame_count": sum(1 for frame in player_frames if frame.get("floor_world_xyz") is not None),
        "floor_contact_player_frame_count": sum(1 for frame in player_frames if frame.get("contact_locked") is True),
        "max_floor_penetration_m": max((float(frame.get("floor_penetration_m") or 0.0) for frame in player_frames), default=0.0),
        "max_abs_floor_offset_m": max((abs(float(frame["floor_offset_m"])) for frame in player_frames if frame.get("floor_offset_m") is not None), default=0.0),
        "physics_modes": sorted({str(frame["physics"]) for frame in player_frames if frame.get("physics")}),
        "ball_frame_count": len(ball_frames),
        "approx_ball_frame_count": sum(1 for frame in ball_frames if frame.get("approx")),
        "paddle_player_count": sum(1 for paddle in paddles if paddle["frames"]),
        "paddle_frame_count": len(paddle_frames),
        "ambiguous_paddle_frame_count": sum(1 for frame in paddle_frames if frame.get("ambiguous")),
        "warnings": warnings,
    }


def _world_fps(*artifacts: Any) -> float:
    for artifact in artifacts:
        if artifact is not None and getattr(artifact, "fps", 0.0):
            return float(artifact.fps)
    return 30.0


def _time_key(t: float) -> str:
    return f"{float(t):.6f}"


def _court_calibration(value: CourtCalibration | Mapping[str, Any]) -> CourtCalibration:
    if isinstance(value, CourtCalibration):
        return value
    try:
        return CourtCalibration.model_validate(value)
    except ValidationError as exc:
        raise ValueError(f"court_calibration failed validation: {exc}") from exc


def _tracks(value: Tracks | Mapping[str, Any] | None) -> Tracks | None:
    if value is None or isinstance(value, Tracks):
        return value
    try:
        return Tracks.model_validate(value)
    except ValidationError as exc:
        raise ValueError(f"tracks failed validation: {exc}") from exc


def _smpl_motion(value: SmplMotion | Mapping[str, Any] | None) -> SmplMotion | None:
    if value is None or isinstance(value, SmplMotion):
        return value
    try:
        return SmplMotion.model_validate(value)
    except ValidationError as exc:
        raise ValueError(f"smpl_motion failed validation: {exc}") from exc


def _skeleton3d(value: Skeleton3D | Mapping[str, Any] | None) -> Skeleton3D | None:
    if value is None or isinstance(value, Skeleton3D):
        return value
    try:
        return Skeleton3D.model_validate(value)
    except ValidationError as exc:
        raise ValueError(f"skeleton3d failed validation: {exc}") from exc


def _ball_track(value: BallTrack | Mapping[str, Any] | None) -> BallTrack | None:
    if value is None or isinstance(value, BallTrack):
        return value
    try:
        return BallTrack.model_validate(value)
    except ValidationError as exc:
        raise ValueError(f"ball_track failed validation: {exc}") from exc


def _racket_pose(value: RacketPose | Mapping[str, Any] | None) -> RacketPose | None:
    if value is None or isinstance(value, RacketPose):
        return value
    try:
        return RacketPose.model_validate(value)
    except ValidationError as exc:
        raise ValueError(f"racket_pose failed validation: {exc}") from exc


__all__ = ["build_virtual_world_state", "build_virtual_world_state_from_files", "write_virtual_world"]
