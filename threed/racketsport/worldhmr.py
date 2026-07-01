"""World-grounded SMPL reconstruction helpers.

This module intentionally stops at deterministic CPU primitives. It does not
run Fast SAM-3D-Body or infer SMPL parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Mapping, Sequence

from .schemas import CourtCalibration


SCAFFOLD_NOTE = "cpu_worldhmr_primitives_no_sam3dbody_integration"
FLOOR_CONTACT_EPSILON_M = 0.035
LOW_GROUNDING_ANCHOR_HEIGHT_M = 0.08


@dataclass(frozen=True)
class WorldTranslationSample:
    """Single per-player root translation and optional mesh vertices."""

    frame_idx: int
    player_id: int
    root_xyz: list[float]
    mesh_vertices_xyz: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_vector3(self.root_xyz, name="root_xyz")
        for idx, vertex in enumerate(self.mesh_vertices_xyz):
            _validate_vector3(vertex, name=f"mesh_vertices_xyz/{idx}")

        object.__setattr__(self, "root_xyz", [float(value) for value in self.root_xyz])
        object.__setattr__(
            self,
            "mesh_vertices_xyz",
            [[float(value) for value in vertex] for vertex in self.mesh_vertices_xyz],
        )


@dataclass(frozen=True)
class WorldGroundingMetrics:
    """Residual summary for adjusted world root translations."""

    sample_count: int
    rms_root_residual_m: float
    max_root_residual_m: float
    rms_ground_z_error_m: float
    max_ground_z_error_m: float
    scaffold: str = SCAFFOLD_NOTE


def snap_player_translation_to_court(
    sample: WorldTranslationSample,
    *,
    court_z_m: float = 0.0,
) -> WorldTranslationSample:
    """Move a player's root translation to the court plane.

    The same vertical delta is applied to mesh vertices so root-relative mesh
    height is preserved.
    """

    z_delta = court_z_m - sample.root_xyz[2]
    return WorldTranslationSample(
        frame_idx=sample.frame_idx,
        player_id=sample.player_id,
        root_xyz=[sample.root_xyz[0], sample.root_xyz[1], court_z_m],
        mesh_vertices_xyz=[
            [vertex[0], vertex[1], vertex[2] + z_delta]
            for vertex in sample.mesh_vertices_xyz
        ],
    )


def smooth_world_translations(
    samples: Sequence[WorldTranslationSample],
    *,
    alpha: float = 0.5,
) -> list[WorldTranslationSample]:
    """Apply deterministic per-player EMA smoothing to root translations."""

    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("alpha must be greater than 0 and less than or equal to 1")

    previous_by_player: dict[int, list[float]] = {}
    smoothed: list[WorldTranslationSample] = []
    for sample in samples:
        previous = previous_by_player.get(sample.player_id)
        if previous is None:
            root_xyz = list(sample.root_xyz)
        else:
            root_xyz = [
                alpha * sample.root_xyz[idx] + (1.0 - alpha) * previous[idx]
                for idx in range(3)
            ]

        previous_by_player[sample.player_id] = root_xyz
        smoothed.append(
            WorldTranslationSample(
                frame_idx=sample.frame_idx,
                player_id=sample.player_id,
                root_xyz=root_xyz,
                mesh_vertices_xyz=sample.mesh_vertices_xyz,
            )
        )

    return smoothed


def residual_metrics(
    observed: Sequence[WorldTranslationSample],
    adjusted: Sequence[WorldTranslationSample],
    *,
    court_z_m: float = 0.0,
) -> WorldGroundingMetrics:
    """Compute root residuals and adjusted root z error against the court."""

    if len(observed) != len(adjusted):
        raise ValueError("observed and adjusted must have the same length")

    root_residuals: list[float] = []
    ground_z_errors: list[float] = []
    for observed_sample, adjusted_sample in zip(observed, adjusted):
        if observed_sample.frame_idx != adjusted_sample.frame_idx:
            raise ValueError("observed and adjusted frame_idx values must match")
        if observed_sample.player_id != adjusted_sample.player_id:
            raise ValueError("observed and adjusted player_id values must match")

        root_residuals.append(_distance3(observed_sample.root_xyz, adjusted_sample.root_xyz))
        ground_z_errors.append(abs(adjusted_sample.root_xyz[2] - court_z_m))

    return WorldGroundingMetrics(
        sample_count=len(root_residuals),
        rms_root_residual_m=_rms(root_residuals),
        max_root_residual_m=max(root_residuals, default=0.0),
        rms_ground_z_error_m=_rms(ground_z_errors),
        max_ground_z_error_m=max(ground_z_errors, default=0.0),
    )


def build_body_artifacts_from_fast_sam(
    samples: Sequence[Mapping[str, Any]],
    *,
    calibration: CourtCalibration,
    fps: float,
    smoothing_alpha: float = 0.65,
    max_root_speed_mps: float | None = None,
    max_track_anchor_smoothing_residual_m: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build BODY contract artifacts from real Fast SAM-3D-Body outputs."""

    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if max_root_speed_mps is not None and max_root_speed_mps <= 0.0:
        raise ValueError("max_root_speed_mps must be positive when provided")
    if max_track_anchor_smoothing_residual_m is not None and max_track_anchor_smoothing_residual_m <= 0.0:
        raise ValueError("max_track_anchor_smoothing_residual_m must be positive when provided")
    if not samples:
        raise ValueError("at least one Fast SAM-3D-Body sample is required")
    grounded = [
        _ground_fast_sam_sample(sample, calibration=calibration)
        for sample in sorted(samples, key=lambda item: (int(item["frame_idx"]), int(item["player_id"])))
    ]
    smoothed, smoothing_metrics = _smooth_grounded_frames(
        grounded,
        alpha=smoothing_alpha,
        max_root_speed_mps=max_root_speed_mps,
        max_track_anchor_smoothing_residual_m=max_track_anchor_smoothing_residual_m,
    )
    players: list[dict[str, Any]] = []
    skeleton_players: list[dict[str, Any]] = []
    max_joint_count = max(len(frame["joints_world"]) for frame in smoothed)
    for player_id in sorted({int(frame["player_id"]) for frame in smoothed}):
        player_frames = [frame for frame in smoothed if int(frame["player_id"]) == player_id]
        betas = _first_list(player_frames, "betas")
        contact_by_frame = [_infer_floor_contact(frame) for frame in player_frames]
        player_has_contact = any(contact["left"] or contact["right"] for contact in contact_by_frame)
        players.append(
            {
                "id": player_id,
                "betas": betas,
                "frames": [
                    {
                        "frame_idx": int(frame["frame_idx"]),
                        "t": float(frame["t"]),
                        "global_orient": list(frame["global_orient"]),
                        "body_pose": list(frame["body_pose"]),
                        "left_hand_pose": list(frame["left_hand_pose"]),
                        "right_hand_pose": list(frame["right_hand_pose"]),
                        "transl_world": list(frame["transl_world"]),
                        "track_world_xy": list(frame["track_world_xy"]),
                        "temporal_smoothing_reset": bool(frame["temporal_smoothing_reset"]),
                        "joints_world": [list(joint) for joint in frame["joints_world"]],
                        "mesh_vertices_world": [list(vertex) for vertex in frame["vertices_world"]],
                        "joint_conf": [float(frame["confidence"])] * len(frame["joints_world"]),
                        "foot_contact": contact,
                        "grf": None,
                    }
                    for frame, contact in zip(player_frames, contact_by_frame)
                ],
                "skate_free": False,
                "physics": "worldhmr_floor_contact_observation_only"
                if player_has_contact
                else "worldhmr_grounded_not_footlocked",
            }
        )
        skeleton_players.append(
            {
                "id": player_id,
                "frames": [
                    {
                        "frame_idx": int(frame["frame_idx"]),
                        "t": float(frame["t"]),
                        "joints_world": [list(joint) for joint in frame["joints_world"]],
                        "joint_conf": [float(frame["confidence"])] * len(frame["joints_world"]),
                    }
                    for frame in player_frames
                ],
            }
        )

    smpl_motion = {
        "schema_version": 1,
        "model": "sam3dbody_world_joints",
        "fps": float(fps),
        "world_frame": "court_Z0",
        "players": players,
    }
    skeleton3d = {
        "schema_version": 1,
        "joint_names": [f"sam3dbody_joint_{idx:03d}" for idx in range(max_joint_count)],
        "preview_only": True,
        "players": skeleton_players,
    }
    metrics = {
        "body_samples": len(samples),
        "players": len(players),
        "frames": len({int(frame["frame_idx"]) for frame in smoothed}),
        "world_frame": "court_Z0",
        "grounding": "camera_extrinsics_plus_track_footpoint_court_z0",
        "grounding_anchor": _common_grounding_anchor(smoothed),
        "smoothing_alpha": smoothing_alpha,
        "max_root_speed_mps": max_root_speed_mps,
        "max_track_anchor_smoothing_residual_m": max_track_anchor_smoothing_residual_m,
        **smoothing_metrics,
        "min_joint_z_m": min(
            (joint[2] for frame in smoothed for joint in frame["joints_world"]),
            default=0.0,
        ),
        "foot_contact_frames": sum(
            1
            for player in players
            for frame in player["frames"]
            if frame["foot_contact"]["left"] or frame["foot_contact"]["right"]
        ),
        "grf_frames": sum(
            1
            for player in players
            for frame in player["frames"]
            if frame["grf"] is not None
        ),
        "skate_free_players": sum(1 for player in players if player["skate_free"]),
    }
    return smpl_motion, skeleton3d, metrics


def _infer_floor_contact(frame: Mapping[str, Any], *, floor_z_m: float = 0.0) -> dict[str, bool]:
    joints = frame.get("joints_world")
    if not isinstance(joints, Sequence) or not joints:
        return {"left": False, "right": False}
    transl = frame.get("transl_world")
    root_x = float(transl[0]) if isinstance(transl, Sequence) and len(transl) >= 1 else 0.0
    low_points: list[tuple[float, float]] = []
    for joint in joints:
        if not isinstance(joint, Sequence) or len(joint) != 3:
            continue
        x = float(joint[0])
        z = float(joint[2])
        if abs(z - floor_z_m) <= FLOOR_CONTACT_EPSILON_M:
            low_points.append((x, z))
    if not low_points:
        return {"left": False, "right": False}
    left = any(x <= root_x for x, _z in low_points)
    right = any(x >= root_x for x, _z in low_points)
    if left and not right and len(low_points) >= 2:
        right = True
    elif right and not left and len(low_points) >= 2:
        left = True
    return {"left": left, "right": right}


def _distance3(left: Sequence[float], right: Sequence[float]) -> float:
    return sqrt(sum((left[idx] - right[idx]) ** 2 for idx in range(3)))


def _rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sqrt(sum(value * value for value in values) / len(values))


def _validate_vector3(values: Sequence[float], *, name: str) -> None:
    if len(values) != 3:
        raise ValueError(f"{name} must be a 3-vector")


def _ground_fast_sam_sample(sample: Mapping[str, Any], *, calibration: CourtCalibration) -> dict[str, Any]:
    joints_camera = _vector3_list(sample.get("joints_camera"), name="joints_camera")
    vertices_camera = _vector3_list(sample.get("vertices_camera", []), name="vertices_camera")
    if not joints_camera and not vertices_camera:
        raise ValueError("Fast SAM-3D-Body sample must include joints_camera or vertices_camera")
    camera_translation = _vector3(sample.get("camera_translation", [0.0, 0.0, 0.0]), name="camera_translation")
    track_world_xy = _vector2(sample.get("track_world_xy"), name="track_world_xy")
    t = float(sample["t"])
    confidence = float(sample.get("confidence", 0.0))
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be between 0 and 1")

    joints_world_raw = _camera_points_to_world(joints_camera, camera_translation, calibration)
    vertices_world_raw = _camera_points_to_world(vertices_camera, camera_translation, calibration)
    anchor_candidates = joints_world_raw or vertices_world_raw
    anchor_xy, anchor_name = _low_grounding_anchor_xy(anchor_candidates)
    all_world = joints_world_raw + vertices_world_raw
    min_z = min(point[2] for point in all_world)
    dx = track_world_xy[0] - anchor_xy[0]
    dy = track_world_xy[1] - anchor_xy[1]
    dz = -min_z

    return {
        "frame_idx": int(sample["frame_idx"]),
        "player_id": int(sample["player_id"]),
        "t": t,
        "confidence": confidence,
        "grounding_anchor": anchor_name,
        "track_world_xy": track_world_xy,
        "global_orient": _float_list(sample.get("global_orient", [0.0, 0.0, 0.0]), name="global_orient"),
        "body_pose": _float_list(sample.get("body_pose", []), name="body_pose"),
        "left_hand_pose": _float_list(sample.get("left_hand_pose", []), name="left_hand_pose"),
        "right_hand_pose": _float_list(sample.get("right_hand_pose", []), name="right_hand_pose"),
        "betas": _float_list(sample.get("betas", []), name="betas"),
        "transl_world": [track_world_xy[0], track_world_xy[1], 0.0],
        "joints_world": _translate_points(joints_world_raw or vertices_world_raw, dx=dx, dy=dy, dz=dz),
        "vertices_world": _translate_points(vertices_world_raw, dx=dx, dy=dy, dz=dz),
    }


def _camera_points_to_world(
    points_camera_relative: Sequence[Sequence[float]],
    camera_translation: Sequence[float],
    calibration: CourtCalibration,
) -> list[list[float]]:
    rotation = [[float(value) for value in row] for row in calibration.extrinsics.R]
    translation = [float(value) for value in calibration.extrinsics.t]
    points: list[list[float]] = []
    for point in points_camera_relative:
        camera_point = [float(point[idx]) + float(camera_translation[idx]) for idx in range(3)]
        camera_minus_t = [camera_point[idx] - translation[idx] for idx in range(3)]
        world = [
            sum(rotation[row_idx][col_idx] * camera_minus_t[row_idx] for row_idx in range(3))
            for col_idx in range(3)
        ]
        points.append(world)
    return points


def _low_grounding_anchor_xy(points_world: Sequence[Sequence[float]]) -> tuple[list[float], str]:
    points = [[float(point[0]), float(point[1]), float(point[2])] for point in points_world]
    min_z = min(point[2] for point in points)
    low_points = [point for point in points if point[2] <= min_z + LOW_GROUNDING_ANCHOR_HEIGHT_M]
    anchor_points = low_points or points[:1]
    return [
        sum(point[0] for point in anchor_points) / len(anchor_points),
        sum(point[1] for point in anchor_points) / len(anchor_points),
    ], "low_joint_cluster" if len(anchor_points) > 1 else "lowest_joint"


def _translate_points(points: Sequence[Sequence[float]], *, dx: float, dy: float, dz: float) -> list[list[float]]:
    return [[float(point[0]) + dx, float(point[1]) + dy, float(point[2]) + dz] for point in points]


def _smooth_grounded_frames(
    frames: Sequence[dict[str, Any]],
    *,
    alpha: float,
    max_root_speed_mps: float | None,
    max_track_anchor_smoothing_residual_m: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("alpha must be greater than 0 and less than or equal to 1")
    previous_by_player: dict[int, list[float]] = {}
    previous_t_by_player: dict[int, float] = {}
    smoothed: list[dict[str, Any]] = []
    root_speed_limited_frames = 0
    track_anchor_residuals: list[float] = []
    pre_reset_track_anchor_residuals: list[float] = []
    track_anchor_residual_reset_frames = 0
    for frame in frames:
        player_id = int(frame["player_id"])
        previous = previous_by_player.get(player_id)
        previous_t = previous_t_by_player.get(player_id)
        transl = [float(value) for value in frame["transl_world"]]
        if previous is None:
            smoothed_transl = transl
        else:
            smoothed_transl = [
                alpha * transl[idx] + (1.0 - alpha) * previous[idx]
                for idx in range(3)
            ]
            if max_root_speed_mps is not None and previous_t is not None:
                dt = max(float(frame["t"]) - previous_t, 0.0)
                if dt > 0.0:
                    smoothed_transl, limited = _limit_step(previous, smoothed_transl, max_distance=max_root_speed_mps * dt)
                    if limited:
                        root_speed_limited_frames += 1
        track_xy = [float(value) for value in frame["track_world_xy"]]
        pre_reset_residual = _distance2(smoothed_transl[:2], track_xy)
        pre_reset_track_anchor_residuals.append(pre_reset_residual)
        temporal_smoothing_reset = False
        if (
            previous is not None
            and max_track_anchor_smoothing_residual_m is not None
            and pre_reset_residual > max_track_anchor_smoothing_residual_m
        ):
            smoothed_transl = transl
            temporal_smoothing_reset = True
            track_anchor_residual_reset_frames += 1
        previous_by_player[player_id] = smoothed_transl
        previous_t_by_player[player_id] = float(frame["t"])
        delta = [smoothed_transl[idx] - transl[idx] for idx in range(3)]
        track_anchor_residuals.append(_distance2(smoothed_transl[:2], track_xy))
        smoothed_frame = dict(frame)
        smoothed_frame["grounding_anchor"] = frame.get("grounding_anchor", "")
        smoothed_frame["transl_world"] = smoothed_transl
        smoothed_frame["temporal_smoothing_reset"] = temporal_smoothing_reset
        smoothed_frame["joints_world"] = [
            [joint[0] + delta[0], joint[1] + delta[1], joint[2] + delta[2]]
            for joint in frame["joints_world"]
        ]
        smoothed_frame["vertices_world"] = [
            [vertex[0] + delta[0], vertex[1] + delta[1], vertex[2] + delta[2]]
            for vertex in frame["vertices_world"]
        ]
        smoothed.append(smoothed_frame)
    return smoothed, {
        "root_speed_limited_frames": root_speed_limited_frames,
        "track_anchor_residual_reset_frames": track_anchor_residual_reset_frames,
        "max_pre_reset_track_anchor_residual_m": max(pre_reset_track_anchor_residuals, default=0.0),
        "max_track_anchor_residual_m": max(track_anchor_residuals, default=0.0),
    }


def _limit_step(previous: Sequence[float], current: Sequence[float], *, max_distance: float) -> tuple[list[float], bool]:
    distance = _distance3(previous, current)
    if distance <= max_distance:
        return [float(value) for value in current], False
    if distance == 0.0:
        return [float(value) for value in current], False
    scale = max_distance / distance
    return [float(previous[idx]) + (float(current[idx]) - float(previous[idx])) * scale for idx in range(3)], True


def _common_grounding_anchor(frames: Sequence[Mapping[str, Any]]) -> str:
    anchors = sorted({str(frame.get("grounding_anchor", "")) for frame in frames if frame.get("grounding_anchor")})
    if not anchors:
        return ""
    return anchors[0] if len(anchors) == 1 else ",".join(anchors)


def _distance2(left: Sequence[float], right: Sequence[float]) -> float:
    return sqrt(sum((left[idx] - right[idx]) ** 2 for idx in range(2)))


def _first_list(frames: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    for frame in frames:
        values = _float_list(frame.get(field, []), name=field)
        if values:
            return values
    return []


def _vector3_list(values: Any, *, name: str) -> list[list[float]]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of 3-vectors")
    return [_vector3(vector, name=f"{name}/{idx}") for idx, vector in enumerate(values)]


def _vector3(values: Any, *, name: str) -> list[float]:
    result = _float_list(values, name=name)
    if len(result) != 3:
        raise ValueError(f"{name} must be a 3-vector")
    return result


def _vector2(values: Any, *, name: str) -> list[float]:
    result = _float_list(values, name=name)
    if len(result) != 2:
        raise ValueError(f"{name} must be a 2-vector")
    return result


def _float_list(values: Any, *, name: str) -> list[float]:
    if values is None or isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result: list[float] = []
    for idx, value in enumerate(values):
        try:
            result.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}/{idx} must be numeric") from exc
    return result
