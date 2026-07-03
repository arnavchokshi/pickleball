"""Lane A fast pose helpers for continuous world skeletons."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .court_positioning import CameraFloorGeometry, back_project_pixel_to_floor
from .model_manifest import verify_model_checkpoint
from .schemas import CourtCalibration, Tracks
from .skeleton_upright import ROTATION_CONVENTION_OFFSET_ROW_TIMES_R, rotate_camera_offsets_row_times_R


ARTIFACT_TYPE = "racketsport_skeleton3d"
RTMW3D_CONFIG_ENV = "RTMW3D_CONFIG_PATH"
RTMW3D_PROJECT_PYTHONPATH_ENV = "RTMW3D_PROJECT_PYTHONPATH"
RTMW3D_DEVICE_ENV = "RTMW3D_DEVICE"
RTMW3D_CONFIG_BY_MODEL_ID = {
    "rtmw3d_x": "rtmw3d-x_8xb32_cocktail14-384x288.py",
}
BODY_17_JOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
FOOT_6_JOINT_NAMES: tuple[str, ...] = (
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
)
FACE_68_JOINT_NAMES: tuple[str, ...] = tuple(f"face_{idx:02d}" for idx in range(68))
LEFT_HAND_21_JOINT_NAMES: tuple[str, ...] = tuple(f"left_hand_{idx:02d}" for idx in range(21))
RIGHT_HAND_21_JOINT_NAMES: tuple[str, ...] = tuple(f"right_hand_{idx:02d}" for idx in range(21))
RTMW3D_WHOLEBODY_133_JOINT_NAMES: tuple[str, ...] = (
    BODY_17_JOINT_NAMES
    + FOOT_6_JOINT_NAMES
    + FACE_68_JOINT_NAMES
    + LEFT_HAND_21_JOINT_NAMES
    + RIGHT_HAND_21_JOINT_NAMES
)
LANE_A_RTMW3D_JOINT_INDEXES: tuple[int, ...] = tuple(range(17)) + tuple(range(17, 23)) + tuple(range(91, 133))
LANE_A_RTMW3D_JOINT_NAMES: tuple[str, ...] = tuple(RTMW3D_WHOLEBODY_133_JOINT_NAMES[idx] for idx in LANE_A_RTMW3D_JOINT_INDEXES)
SUPPORT_FOOT_JOINT_NAMES = frozenset(FOOT_6_JOINT_NAMES)


@dataclass(frozen=True)
class PoseCropRequest:
    frame_idx: int
    player_id: int
    bbox_xyxy: list[float]
    track_world_xy: list[float]
    track_confidence: float


@dataclass(frozen=True)
class PoseCropResult:
    frame_idx: int
    player_id: int
    joints_m: Sequence[Sequence[float]]
    joint_conf: Sequence[float]
    joint_names: Sequence[str] = field(default_factory=lambda: RTMW3D_WHOLEBODY_133_JOINT_NAMES)
    joint_pixels: Sequence[Sequence[float]] | None = None


class RTMW3DPoseRuntime:
    """Runtime boundary for real RTMW3D inference.

    The production runner injects this boundary so tests can prove batching and
    artifact contracts without fabricating a model invocation.
    """

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        model_id: str = "rtmw3d_x",
        device: str | None = None,
        config_path: str | Path | None = None,
        project_pythonpath: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.model_id = model_id
        self.device = device
        self.config_path = Path(config_path) if config_path is not None else None
        self.project_pythonpath = Path(project_pythonpath) if project_pythonpath is not None else None
        self._manifest_verified = False
        self._manifest_entry: Any | None = None
        self._model: Any | None = None
        self._inference_topdown: Callable[..., Any] | None = None

    def infer_frame(self, image_path: Path, requests: list[PoseCropRequest]) -> list[PoseCropResult]:
        if not requests:
            return []
        model, inference_topdown = self._load_mmpose_runtime()
        np = _import_numpy()
        bboxes = np.asarray([request.bbox_xyxy for request in requests], dtype=np.float32)
        samples = inference_topdown(model, str(image_path), bboxes=bboxes, bbox_format="xyxy")
        if len(samples) != len(requests):
            raise RuntimeError(f"RTMW3D returned {len(samples)} results for {len(requests)} crop requests")
        return [_pose_result_from_mmpose_sample(request, sample) for request, sample in zip(requests, samples)]

    def _verify_manifest_checkpoint(self) -> Any:
        if self._manifest_verified:
            return self._manifest_entry
        self._manifest_entry = verify_model_checkpoint(self.manifest_path, self.model_id)
        self._manifest_verified = True
        return self._manifest_entry

    def _load_mmpose_runtime(self) -> tuple[Any, Callable[..., Any]]:
        if self._model is not None and self._inference_topdown is not None:
            return self._model, self._inference_topdown
        entry = self._verify_manifest_checkpoint()
        checkpoint_path = Path(entry.local_path)
        config_path = self._resolve_config_path(checkpoint_path)
        project_pythonpath = self._resolve_project_pythonpath()
        if project_pythonpath is not None:
            project_path = str(project_pythonpath)
            if project_path not in sys.path:
                sys.path.insert(0, project_path)
        try:
            from mmpose.apis import inference_topdown, init_model
            from mmpose.utils import register_all_modules
        except ImportError as exc:
            raise RuntimeError(
                "MMPose RTMW3D runtime dependencies are unavailable; install mmpose/mmcv/mmengine "
                "and expose the MMPose projects/rtmpose3d package on PYTHONPATH"
            ) from exc
        register_all_modules()
        self._model = init_model(str(config_path), str(checkpoint_path), device=self.device or os.environ.get(RTMW3D_DEVICE_ENV, "cuda:0"))
        self._inference_topdown = inference_topdown
        return self._model, self._inference_topdown

    def _resolve_config_path(self, checkpoint_path: Path) -> Path:
        config_path = self.config_path
        if config_path is None and os.environ.get(RTMW3D_CONFIG_ENV):
            config_path = Path(os.environ[RTMW3D_CONFIG_ENV])
        if config_path is None:
            config_name = RTMW3D_CONFIG_BY_MODEL_ID.get(self.model_id)
            if config_name:
                config_path = checkpoint_path.with_name(config_name)
        if config_path is None:
            raise RuntimeError(f"no RTMW3D config path is configured for model {self.model_id}")
        if not config_path.is_file():
            raise FileNotFoundError(f"missing RTMW3D config for {self.model_id}: {config_path}")
        return config_path

    def _resolve_project_pythonpath(self) -> Path | None:
        if self.project_pythonpath is not None:
            return self.project_pythonpath
        if os.environ.get(RTMW3D_PROJECT_PYTHONPATH_ENV):
            return Path(os.environ[RTMW3D_PROJECT_PYTHONPATH_ENV])
        return None


def build_lane_a_skeleton3d_from_rtmw3d(
    tracks: Tracks,
    pose_results: Sequence[PoseCropResult],
    *,
    world_frame: str,
    source_model: str,
    calibration: CourtCalibration | None = None,
) -> dict[str, Any]:
    """Build the real Lane A skeleton artifact from RTMW3D crop results."""

    if tracks.fps <= 0.0:
        raise ValueError("tracks.fps must be positive")
    result_lookup = {(int(result.frame_idx), int(result.player_id)): result for result in pose_results}
    players: list[dict[str, Any]] = []
    missing: list[dict[str, int]] = []
    pixel_grounding_count = 0
    grounding_fallback_count = 0
    for player in sorted(tracks.players, key=lambda item: int(item.id)):
        frames: list[dict[str, Any]] = []
        for track_frame in sorted(player.frames, key=lambda item: float(item.t)):
            frame_idx = int(round(float(track_frame.t) * tracks.fps))
            result = result_lookup.get((frame_idx, int(player.id)))
            if result is None:
                missing.append({"frame_idx": frame_idx, "player_id": int(player.id)})
                continue
            selected_joints, selected_conf = _lane_a_joints(result)
            rotation_applied = calibration is not None and world_frame == "court_Z0"
            grounding_joints = (
                rotate_camera_offsets_row_times_R(
                    selected_joints,
                    rotation=calibration.extrinsics.R,
                    joint_names=LANE_A_RTMW3D_JOINT_NAMES,
                )
                if rotation_applied
                else selected_joints
            )
            support_idx = _support_foot_index(grounding_joints, selected_conf)
            support_xy = None
            if calibration is not None and world_frame == "court_Z0":
                support_xy = _support_foot_ground_xy_from_pixels(result, support_idx=support_idx, calibration=calibration)
            if support_xy is None:
                support_xy = [float(value) for value in track_frame.world_xy]
                grounding_fallback_count += 1
            else:
                pixel_grounding_count += 1
            joints_world = _ground_to_support_foot(
                grounding_joints,
                support_idx=support_idx,
                support_world_xy=support_xy,
            )
            frames.append(
                {
                    "frame_idx": frame_idx,
                    "t": float(track_frame.t),
                    "transl_world": [float(track_frame.world_xy[0]), float(track_frame.world_xy[1]), 0.0],
                    "joints_world": joints_world,
                    "joint_conf": selected_conf,
                }
            )
        players.append({"id": int(player.id), "frames": frames})

    if missing:
        raise ValueError(f"missing Lane A pose results for tracked player frames: {missing[:10]}")

    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "fps": float(tracks.fps),
        "world_frame": world_frame,
        "source_model": source_model,
        "joint_names": list(LANE_A_RTMW3D_JOINT_NAMES),
        "preview_only": False,
        "players": players,
        "provenance": {
            "lane": "A",
            "source_joint_count": len(RTMW3D_WHOLEBODY_133_JOINT_NAMES),
            "output_joint_count": len(LANE_A_RTMW3D_JOINT_NAMES),
            "dropped_joint_groups": ["face_68"],
            "grounding": (
                "support_foot_pixel_backprojected_to_court_z0"
                if pixel_grounding_count > 0 and grounding_fallback_count == 0
                else "support_foot_to_track_world_xy"
            ),
            "pixel_grounding_count": pixel_grounding_count,
            "grounding_fallback_count": grounding_fallback_count,
            "camera_offset_rotation_convention": (
                ROTATION_CONVENTION_OFFSET_ROW_TIMES_R if calibration is not None and world_frame == "court_Z0" else None
            ),
        },
    }


def write_skeleton3d(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pose_result_from_mmpose_sample(request: PoseCropRequest, sample: Any) -> PoseCropResult:
    if not hasattr(sample, "pred_instances"):
        raise RuntimeError("RTMW3D result is missing pred_instances")
    instances = sample.pred_instances
    keypoints = _prediction_array(instances, "keypoints")
    scores = _prediction_array(instances, "keypoint_scores")
    keypoints = _squeeze_single_prediction(keypoints, name="keypoints")
    scores = _squeeze_single_prediction(scores, name="keypoint_scores")
    joint_pixels = _transformed_keypoints_or_none(instances)
    if keypoints.shape != (len(RTMW3D_WHOLEBODY_133_JOINT_NAMES), 3):
        raise RuntimeError(f"RTMW3D keypoints must have shape (133, 3), got {tuple(keypoints.shape)}")
    if scores.shape != (len(RTMW3D_WHOLEBODY_133_JOINT_NAMES),):
        raise RuntimeError(f"RTMW3D keypoint_scores must have shape (133,), got {tuple(scores.shape)}")
    return PoseCropResult(
        frame_idx=int(request.frame_idx),
        player_id=int(request.player_id),
        joints_m=keypoints.astype(float).tolist(),
        joint_conf=scores.astype(float).tolist(),
        joint_names=RTMW3D_WHOLEBODY_133_JOINT_NAMES,
        joint_pixels=joint_pixels,
    )


def _prediction_array(instances: Any, field_name: str) -> Any:
    if not hasattr(instances, field_name):
        raise RuntimeError(f"RTMW3D pred_instances is missing {field_name}")
    np = _import_numpy()
    return np.asarray(getattr(instances, field_name))


def _transformed_keypoints_or_none(instances: Any) -> list[list[float]] | None:
    if not hasattr(instances, "transformed_keypoints"):
        return None
    pixels = _prediction_array(instances, "transformed_keypoints")
    pixels = _squeeze_single_prediction(pixels, name="transformed_keypoints")
    if pixels.shape != (len(RTMW3D_WHOLEBODY_133_JOINT_NAMES), 2):
        raise RuntimeError(f"RTMW3D transformed_keypoints must have shape (133, 2), got {tuple(pixels.shape)}")
    return pixels.astype(float).tolist()


def _squeeze_single_prediction(values: Any, *, name: str) -> Any:
    if values.ndim >= 1 and values.shape[0] == 1:
        return values[0]
    return values


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("RTMW3D runtime requires numpy") from exc
    return np


def _lane_a_joints(result: PoseCropResult) -> tuple[list[list[float]], list[float]]:
    joint_names = tuple(str(name) for name in result.joint_names)
    if joint_names != RTMW3D_WHOLEBODY_133_JOINT_NAMES:
        raise ValueError("RTMW3D result joint_names must match the 133-keypoint whole-body layout")
    if len(result.joints_m) != len(RTMW3D_WHOLEBODY_133_JOINT_NAMES):
        raise ValueError("RTMW3D result must contain 133 joints")
    if len(result.joint_conf) != len(RTMW3D_WHOLEBODY_133_JOINT_NAMES):
        raise ValueError("RTMW3D result must contain 133 joint confidence values")
    joints = [_vector3(result.joints_m[idx], name=f"joints_m/{idx}") for idx in LANE_A_RTMW3D_JOINT_INDEXES]
    conf = [_unit(result.joint_conf[idx], name=f"joint_conf/{idx}") for idx in LANE_A_RTMW3D_JOINT_INDEXES]
    return joints, conf


def _support_foot_index(
    joints_m: Sequence[Sequence[float]],
    joint_conf: Sequence[float],
) -> int:
    foot_candidates = [
        (idx, float(joints_m[idx][2]), float(joint_conf[idx]))
        for idx, name in enumerate(LANE_A_RTMW3D_JOINT_NAMES)
        if name in SUPPORT_FOOT_JOINT_NAMES and float(joint_conf[idx]) >= 0.3
    ]
    if not foot_candidates:
        foot_candidates = [(idx, float(joint[2]), float(joint_conf[idx])) for idx, joint in enumerate(joints_m)]
    support_idx, _support_z, _support_conf = min(foot_candidates, key=lambda item: (item[1], -item[2], item[0]))
    return int(support_idx)


def _ground_to_support_foot(
    joints_m: Sequence[Sequence[float]],
    *,
    support_idx: int,
    support_world_xy: Sequence[float],
) -> list[list[float]]:
    support = [float(value) for value in joints_m[support_idx]]
    dx = float(support_world_xy[0]) - support[0]
    dy = float(support_world_xy[1]) - support[1]
    dz = -support[2]
    return [[float(joint[0]) + dx, float(joint[1]) + dy, float(joint[2]) + dz] for joint in joints_m]


def _support_foot_ground_xy_from_pixels(
    result: PoseCropResult,
    *,
    support_idx: int,
    calibration: CourtCalibration,
) -> list[float] | None:
    if result.joint_pixels is None:
        return None
    if len(result.joint_pixels) != len(RTMW3D_WHOLEBODY_133_JOINT_NAMES):
        return None
    full_joint_idx = LANE_A_RTMW3D_JOINT_INDEXES[support_idx]
    try:
        pixel = _vector2(result.joint_pixels[full_joint_idx], name=f"joint_pixels/{full_joint_idx}")
        court_xyz = _backproject_pixel_to_court_z0(calibration, pixel)
    except (ValueError, TypeError, IndexError):
        return None
    return [float(court_xyz[0]), float(court_xyz[1])]


def _backproject_pixel_to_court_z0(calibration: CourtCalibration, pixel_uv: Sequence[float]) -> list[float]:
    if calibration.T_world_court is not None:
        transform = _mat4(calibration.T_world_court, "calibration.T_world_court")
        geometry = CameraFloorGeometry(
            intrinsics=calibration.intrinsics.model_dump(mode="json"),
            camera_origin_world=list(calibration.extrinsics.t),
            R_world_camera=list(calibration.extrinsics.R),
            floor_plane_point=[transform[row][3] for row in range(3)],
            floor_plane_normal=[transform[row][2] for row in range(3)],
        )
        world_xyz = back_project_pixel_to_floor(pixel_uv, geometry)
        court_xyz = _world_to_court(world_xyz, transform)
        return [court_xyz[0], court_xyz[1], 0.0]

    rotation_camera_world = _mat3(calibration.extrinsics.R, "calibration.extrinsics.R")
    translation_camera = _vector3(calibration.extrinsics.t, name="calibration.extrinsics.t")
    rotation_world_camera = _transpose3(rotation_camera_world)
    camera_origin_world = [
        -sum(rotation_world_camera[row][col] * translation_camera[col] for col in range(3))
        for row in range(3)
    ]
    geometry = CameraFloorGeometry(
        intrinsics=calibration.intrinsics.model_dump(mode="json"),
        camera_origin_world=camera_origin_world,
        R_world_camera=rotation_world_camera,
        floor_plane_point=[0.0, 0.0, 0.0],
        floor_plane_normal=[0.0, 0.0, 1.0],
    )
    world_xyz = back_project_pixel_to_floor(pixel_uv, geometry)
    return [float(world_xyz[0]), float(world_xyz[1]), 0.0]


def _world_to_court(world_xyz: Sequence[float], transform: Sequence[Sequence[float]]) -> list[float]:
    point = _vector3(world_xyz, name="world_xyz")
    translation = [float(transform[row][3]) for row in range(3)]
    shifted = [point[idx] - translation[idx] for idx in range(3)]
    rotation = [[float(transform[row][col]) for col in range(3)] for row in range(3)]
    return [
        rotation[0][axis] * shifted[0] + rotation[1][axis] * shifted[1] + rotation[2][axis] * shifted[2]
        for axis in range(3)
    ]


def _vector3(values: Sequence[float], *, name: str) -> list[float]:
    if len(values) != 3:
        raise ValueError(f"{name} must be a 3-vector")
    return [float(value) for value in values]


def _vector2(values: Sequence[float], *, name: str) -> list[float]:
    if len(values) != 2:
        raise ValueError(f"{name} must be a 2-vector")
    return [float(value) for value in values]


def _mat3(values: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain 3 rows")
    rows = []
    for row_idx, row in enumerate(values):
        if len(row) != 3:
            raise ValueError(f"{name}[{row_idx}] must contain 3 values")
        rows.append([float(value) for value in row])
    return rows


def _mat4(values: Sequence[Sequence[float]], name: str) -> list[list[float]]:
    if len(values) != 4:
        raise ValueError(f"{name} must contain 4 rows")
    rows = []
    for row_idx, row in enumerate(values):
        if len(row) != 4:
            raise ValueError(f"{name}[{row_idx}] must contain 4 values")
        rows.append([float(value) for value in row])
    return rows


def _transpose3(values: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(values[row][col]) for row in range(3)] for col in range(3)]


def _unit(value: float, *, name: str) -> float:
    confidence = float(value)
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return confidence


__all__ = [
    "LANE_A_RTMW3D_JOINT_NAMES",
    "PoseCropRequest",
    "PoseCropResult",
    "RTMW3DPoseRuntime",
    "RTMW3D_WHOLEBODY_133_JOINT_NAMES",
    "build_lane_a_skeleton3d_from_rtmw3d",
    "write_skeleton3d",
]
