"""Temporal refinement for Lane A skeleton3d artifacts."""

from __future__ import annotations

import copy
import math
import os
import sys
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from .model_manifest import verify_model_checkpoint


DEFAULT_ONE_EURO_MINCUTOFF = 1.0
DEFAULT_ONE_EURO_BETA = 0.3
DEFAULT_ONE_EURO_DCUTOFF = 1.0
DEFAULT_MOTIONBERT_WINDOW_MAX_FRAMES = 243
MOTIONBERT_MODEL_ID = "motionbert_lift_smooth"
MOTIONBERT_CONFIG_ENV = "MOTIONBERT_CONFIG_PATH"
MOTIONBERT_PROJECT_PYTHONPATH_ENV = "MOTIONBERT_PROJECT_PYTHONPATH"
MOTIONBERT_DEVICE_ENV = "MOTIONBERT_DEVICE"
MOTIONBERT_DEFAULT_CONFIG = "configs/pose3d/MB_ft_h36m_global_lite.yaml"
MIN_BONE_CONFIDENCE = 0.3
GROUND_CONTACT_MAX_Z_M = 0.08
FOOT_LOCK_MAX_VERTICAL_SPEED_MPS = 1.0
FOOT_LOCK_REANCHOR_DISTANCE_M = 0.35

BODY17_JOINT_NAMES: tuple[str, ...] = (
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

BODY17_BONE_EDGES: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)

COCO_BODY17_TO_H36M_INDEX: dict[str, int] = {
    "nose": 9,
    "left_shoulder": 11,
    "right_shoulder": 14,
    "left_elbow": 12,
    "right_elbow": 15,
    "left_wrist": 13,
    "right_wrist": 16,
    "left_hip": 4,
    "right_hip": 1,
    "left_knee": 5,
    "right_knee": 2,
    "left_ankle": 6,
    "right_ankle": 3,
}


class MotionBERTTemporalRuntime:
    """Manifest-backed MotionBERT runtime for Lane A body-17 windows."""

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        model_id: str = MOTIONBERT_MODEL_ID,
        config_path: str | Path | None = None,
        project_pythonpath: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.model_id = model_id
        self.config_path = Path(config_path) if config_path is not None else None
        self.project_pythonpath = Path(project_pythonpath) if project_pythonpath is not None else None
        self.device = device
        self._manifest_verified = False
        self._manifest_entry: Any | None = None
        self._loaded: dict[str, Any] | None = None

    def refine_body17_window(
        self,
        *,
        player_id: int,
        frames: list[dict],
        joint_names: list[str],
    ) -> list[list[list[float]]]:
        del player_id
        if list(joint_names) != list(BODY17_JOINT_NAMES):
            raise ValueError("MotionBERTTemporalRuntime expects RTMW3D COCO body17 joint order")
        self._verify_manifest_checkpoint()
        loaded = self._load_model()
        torch = loaded["torch"]
        args = loaded["args"]
        model = loaded["model"]
        input_motion, alignments = _motionbert_input_from_body17_frames(frames, joint_names)
        device = str(loaded["device"])
        batch = torch.as_tensor(input_motion, dtype=torch.float32, device=device).unsqueeze(0)
        if bool(getattr(args, "no_conf", False)):
            batch = batch[..., :2]
        with torch.no_grad():
            if bool(getattr(args, "flip", False)) and loaded.get("flip_data") is not None:
                flipped = loaded["flip_data"](batch)
                predicted = (model(batch) + loaded["flip_data"](model(flipped))) / 2.0
            else:
                predicted = model(batch)
            if bool(getattr(args, "rootrel", False)):
                predicted[:, :, 0, :] = 0.0
        prediction = predicted.detach().cpu().numpy()[0]
        if prediction.shape != (len(frames), len(BODY17_JOINT_NAMES), 3):
            raise RuntimeError(f"MotionBERT output must have shape ({len(frames)}, 17, 3), got {tuple(prediction.shape)}")
        return _merge_motionbert_h36m_prediction(prediction, frames, joint_names, alignments)

    def _verify_manifest_checkpoint(self) -> Any:
        if self._manifest_verified:
            return self._manifest_entry
        self._manifest_entry = verify_model_checkpoint(self.manifest_path, self.model_id)
        self._manifest_verified = True
        return self._manifest_entry

    def _load_model(self) -> dict[str, Any]:
        if self._loaded is not None:
            return self._loaded
        entry = self._verify_manifest_checkpoint()
        config_path = self._resolve_config_path()
        project_pythonpath = self._resolve_project_pythonpath(config_path)
        if project_pythonpath is not None:
            project_path = str(project_pythonpath)
            if project_path not in sys.path:
                sys.path.insert(0, project_path)
        try:
            import torch
            import yaml
            from lib.utils.learning import load_backbone
            from lib.utils.utils_data import flip_data
        except ImportError as exc:
            raise RuntimeError("MotionBERT runtime dependencies are unavailable") from exc
        config_values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_values, Mapping):
            raise RuntimeError(f"MotionBERT config did not parse as a mapping: {config_path}")
        args = SimpleNamespace(**dict(config_values))
        device = self.device or os.environ.get(MOTIONBERT_DEVICE_ENV) or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model = load_backbone(args).to(device).eval()
        checkpoint = torch.load(str(entry.local_path), map_location="cpu")
        if not isinstance(checkpoint, Mapping) or "model_pos" not in checkpoint:
            raise RuntimeError("MotionBERT checkpoint must contain model_pos")
        state = {
            str(key).removeprefix("module."): value
            for key, value in checkpoint["model_pos"].items()
        }
        model.load_state_dict(state, strict=True)
        self._loaded = {"model": model, "args": args, "torch": torch, "flip_data": flip_data, "device": device}
        return self._loaded

    def _resolve_config_path(self) -> Path:
        config_path = self.config_path
        if config_path is None and os.environ.get(MOTIONBERT_CONFIG_ENV):
            config_path = Path(os.environ[MOTIONBERT_CONFIG_ENV])
        if config_path is None:
            project_pythonpath = self._resolve_project_pythonpath(None)
            if project_pythonpath is not None:
                config_path = project_pythonpath / MOTIONBERT_DEFAULT_CONFIG
        if config_path is None:
            raise RuntimeError("no MotionBERT config path is configured")
        if not config_path.is_file():
            raise FileNotFoundError(f"missing MotionBERT config: {config_path}")
        return config_path

    def _resolve_project_pythonpath(self, config_path: Path | None) -> Path | None:
        if self.project_pythonpath is not None:
            return self.project_pythonpath
        if os.environ.get(MOTIONBERT_PROJECT_PYTHONPATH_ENV):
            return Path(os.environ[MOTIONBERT_PROJECT_PYTHONPATH_ENV])
        if config_path is not None:
            parents = list(config_path.parents)
            if len(parents) >= 3 and (parents[2] / "lib").is_dir():
                return parents[2]
        return None


def refine_lane_a_skeleton3d(
    skeleton3d: Mapping[str, Any],
    *,
    fps: float | None = None,
    one_euro_mincutoff: float = DEFAULT_ONE_EURO_MINCUTOFF,
    one_euro_beta: float = DEFAULT_ONE_EURO_BETA,
    motionbert_window_max_frames: int = DEFAULT_MOTIONBERT_WINDOW_MAX_FRAMES,
    motionbert_runtime: Any | None = None,
) -> dict[str, Any]:
    """Return a refined Lane A skeleton payload without fabricating frames."""

    if one_euro_mincutoff <= 0.0:
        raise ValueError("one_euro_mincutoff must be positive")
    if one_euro_beta < 0.0:
        raise ValueError("one_euro_beta must be non-negative")
    if motionbert_window_max_frames <= 0:
        raise ValueError("motionbert_window_max_frames must be positive")
    joint_names = skeleton3d.get("joint_names")
    if not isinstance(joint_names, list) or not all(isinstance(name, str) for name in joint_names):
        raise ValueError("skeleton3d joint_names must be a list of strings")
    players = skeleton3d.get("players")
    if not isinstance(players, list):
        raise ValueError("skeleton3d players must be a list")

    inferred_fps = float(fps or skeleton3d.get("fps") or 0.0)
    if inferred_fps <= 0.0:
        inferred_fps = _infer_fps(players)

    output = copy.deepcopy(dict(skeleton3d))
    output_players: list[dict[str, Any]] = []
    grounding_metrics = _empty_grounding_metrics()
    motionbert_metrics = _empty_motionbert_metrics(motionbert_runtime)
    for player in players:
        if not isinstance(player, Mapping):
            continue
        player_output = copy.deepcopy(dict(player))
        frames = player_output.get("frames")
        if not isinstance(frames, list):
            player_output["frames"] = []
            output_players.append(player_output)
            continue
        sorted_frames = sorted(frames, key=lambda frame: (float(frame.get("t", 0.0)), int(frame.get("frame_idx", 0))))
        motionbert_frames, player_motionbert_metrics = _apply_motionbert_body17(
            sorted_frames,
            joint_names,
            player_id=int(player.get("id", 0)),
            runtime=motionbert_runtime,
            window_max_frames=motionbert_window_max_frames,
        )
        _add_motionbert_metrics(motionbert_metrics, player_motionbert_metrics)
        bone_lengths = _median_bone_lengths(motionbert_frames, joint_names)
        filtered_frames = _apply_one_euro(motionbert_frames, joint_names, fps=inferred_fps, mincutoff=one_euro_mincutoff, beta=one_euro_beta)
        constrained_frames = _apply_bone_lengths(filtered_frames, joint_names, bone_lengths)
        grounded_frames, player_grounding_metrics = _apply_lane_a_world_grounding(
            constrained_frames,
            joint_names,
            fps=inferred_fps,
        )
        _add_grounding_metrics(grounding_metrics, player_grounding_metrics)
        player_output["frames"] = grounded_frames
        output_players.append(player_output)

    output["players"] = output_players
    output["provenance"] = _provenance_with_temporal_refine(
        output.get("provenance"),
        mincutoff=one_euro_mincutoff,
        beta=one_euro_beta,
        motionbert_window_max_frames=motionbert_window_max_frames,
        motionbert_metrics=motionbert_metrics,
        grounding_metrics=grounding_metrics,
    )
    return output


def _apply_motionbert_body17(
    frames: Sequence[Mapping[str, Any]],
    joint_names: Sequence[str],
    *,
    player_id: int,
    runtime: Any | None,
    window_max_frames: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    copied_frames = [copy.deepcopy(dict(frame)) for frame in frames]
    metrics = {"motionbert_window_count": 0, "motionbert_frame_count": 0}
    if runtime is None or not copied_frames:
        return copied_frames, metrics
    body17_indexes = _body17_indexes(joint_names)
    if len(body17_indexes) != len(BODY17_JOINT_NAMES):
        raise ValueError("MotionBERT refine requires the 17 body joints in skeleton3d joint_names")
    body17_names = [joint_names[idx] for idx in body17_indexes]
    for window_start in range(0, len(copied_frames), window_max_frames):
        window = copied_frames[window_start : window_start + window_max_frames]
        refined_body17 = runtime.refine_body17_window(
            player_id=player_id,
            frames=window,
            joint_names=body17_names,
        )
        if len(refined_body17) != len(window):
            raise RuntimeError(
                f"MotionBERT returned {len(refined_body17)} frames for {len(window)} input frames"
            )
        for frame, refined_joints in zip(window, refined_body17):
            if len(refined_joints) != len(BODY17_JOINT_NAMES):
                raise RuntimeError(
                    f"MotionBERT body17 output must contain 17 joints, got {len(refined_joints)}"
                )
            joints = _joint_vectors(frame)
            for body_pos, joint_idx in enumerate(body17_indexes):
                joints[joint_idx] = _vector3(refined_joints[body_pos], name=f"motionbert_body17/{body_pos}")
            frame["joints_world"] = joints
        metrics["motionbert_window_count"] += 1
        metrics["motionbert_frame_count"] += len(window)
    return copied_frames, metrics


def _motionbert_input_from_body17_frames(
    frames: Sequence[Mapping[str, Any]],
    joint_names: Sequence[str],
) -> tuple[Any, list[dict[str, Any]]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("MotionBERT runtime requires numpy") from exc
    h36m_frames: list[list[list[float]]] = []
    alignments: list[dict[str, Any]] = []
    for frame in frames:
        joints = _joint_vectors(frame)
        conf = frame.get("joint_conf", [])
        body_by_name = {name: joints[idx] for idx, name in enumerate(joint_names) if idx < len(joints)}
        conf_by_name = {
            name: _joint_confidence(conf, idx)
            for idx, name in enumerate(joint_names)
        }
        h36m_world, h36m_conf = _coco_body17_to_h36m(body_by_name, conf_by_name)
        root = h36m_world[0]
        scale = _motionbert_body_scale(body_by_name)
        h36m_frames.append(
            [
                [
                    (joint[0] - root[0]) / scale,
                    (joint[1] - root[1]) / scale,
                    h36m_conf[idx],
                ]
                for idx, joint in enumerate(h36m_world)
            ]
        )
        alignments.append({"root_xyz": root, "scale": scale})
    return np.asarray(h36m_frames, dtype=np.float32), alignments


def _coco_body17_to_h36m(
    body_by_name: Mapping[str, Sequence[float]],
    conf_by_name: Mapping[str, float],
) -> tuple[list[list[float]], list[float]]:
    left_hip = _body_joint(body_by_name, "left_hip")
    right_hip = _body_joint(body_by_name, "right_hip")
    left_shoulder = _body_joint(body_by_name, "left_shoulder")
    right_shoulder = _body_joint(body_by_name, "right_shoulder")
    hip = _average_vectors(left_hip, right_hip)
    neck = _average_vectors(left_shoulder, right_shoulder)
    spine = _average_vectors(hip, neck)
    h36m = [[0.0, 0.0, 0.0] for _idx in range(17)]
    h36m[0] = hip
    h36m[1] = _body_joint(body_by_name, "right_hip")
    h36m[2] = _body_joint(body_by_name, "right_knee")
    h36m[3] = _body_joint(body_by_name, "right_ankle")
    h36m[4] = _body_joint(body_by_name, "left_hip")
    h36m[5] = _body_joint(body_by_name, "left_knee")
    h36m[6] = _body_joint(body_by_name, "left_ankle")
    h36m[7] = spine
    h36m[8] = neck
    h36m[9] = _body_joint(body_by_name, "nose")
    h36m[10] = _body_joint(body_by_name, "nose")
    h36m[11] = left_shoulder
    h36m[12] = _body_joint(body_by_name, "left_elbow")
    h36m[13] = _body_joint(body_by_name, "left_wrist")
    h36m[14] = right_shoulder
    h36m[15] = _body_joint(body_by_name, "right_elbow")
    h36m[16] = _body_joint(body_by_name, "right_wrist")
    h36m_conf = [0.0 for _idx in range(17)]
    h36m_conf[0] = _average_conf(conf_by_name, "left_hip", "right_hip")
    h36m_conf[1] = float(conf_by_name.get("right_hip", 0.0))
    h36m_conf[2] = float(conf_by_name.get("right_knee", 0.0))
    h36m_conf[3] = float(conf_by_name.get("right_ankle", 0.0))
    h36m_conf[4] = float(conf_by_name.get("left_hip", 0.0))
    h36m_conf[5] = float(conf_by_name.get("left_knee", 0.0))
    h36m_conf[6] = float(conf_by_name.get("left_ankle", 0.0))
    h36m_conf[7] = _average_conf(conf_by_name, "left_hip", "right_hip", "left_shoulder", "right_shoulder")
    h36m_conf[8] = _average_conf(conf_by_name, "left_shoulder", "right_shoulder")
    h36m_conf[9] = float(conf_by_name.get("nose", 0.0))
    h36m_conf[10] = float(conf_by_name.get("nose", 0.0))
    h36m_conf[11] = float(conf_by_name.get("left_shoulder", 0.0))
    h36m_conf[12] = float(conf_by_name.get("left_elbow", 0.0))
    h36m_conf[13] = float(conf_by_name.get("left_wrist", 0.0))
    h36m_conf[14] = float(conf_by_name.get("right_shoulder", 0.0))
    h36m_conf[15] = float(conf_by_name.get("right_elbow", 0.0))
    h36m_conf[16] = float(conf_by_name.get("right_wrist", 0.0))
    return h36m, h36m_conf


def _merge_motionbert_h36m_prediction(
    prediction: Any,
    frames: Sequence[Mapping[str, Any]],
    joint_names: Sequence[str],
    alignments: Sequence[Mapping[str, Any]],
) -> list[list[list[float]]]:
    merged_frames: list[list[list[float]]] = []
    for frame_idx, (predicted_h36m, frame, alignment) in enumerate(zip(prediction, frames, alignments)):
        joints = _joint_vectors(frame)
        if len(joints) < len(joint_names):
            raise RuntimeError(f"body17 frame {frame_idx} has {len(joints)} joints, expected {len(joint_names)}")
        root = [float(value) for value in alignment["root_xyz"]]
        scale = float(alignment["scale"])
        predicted_root = [float(value) for value in predicted_h36m[0]]
        h36m_world = [
            [
                root[axis] + (float(joint[axis]) - predicted_root[axis]) * scale
                for axis in range(3)
            ]
            for joint in predicted_h36m
        ]
        body17 = [list(joints[idx]) for idx in range(len(joint_names))]
        for name, h36m_idx in COCO_BODY17_TO_H36M_INDEX.items():
            if name in joint_names:
                body17[joint_names.index(name)] = list(h36m_world[h36m_idx])
        merged_frames.append(body17)
    return merged_frames


def _body_joint(body_by_name: Mapping[str, Sequence[float]], name: str) -> list[float]:
    if name not in body_by_name:
        raise ValueError(f"MotionBERT body17 conversion missing required joint: {name}")
    return _vector3(body_by_name[name], name=name)


def _average_vectors(*vectors: Sequence[float]) -> list[float]:
    return [
        sum(float(vector[axis]) for vector in vectors) / len(vectors)
        for axis in range(3)
    ]


def _average_conf(conf_by_name: Mapping[str, float], *names: str) -> float:
    return sum(float(conf_by_name.get(name, 0.0)) for name in names) / len(names)


def _motionbert_body_scale(body_by_name: Mapping[str, Sequence[float]]) -> float:
    left_shoulder = _body_joint(body_by_name, "left_shoulder")
    right_shoulder = _body_joint(body_by_name, "right_shoulder")
    left_hip = _body_joint(body_by_name, "left_hip")
    right_hip = _body_joint(body_by_name, "right_hip")
    return max(
        _xy_distance(left_shoulder, right_shoulder),
        _xy_distance(left_hip, right_hip),
        1.0,
    )


def _apply_one_euro(
    frames: Sequence[Mapping[str, Any]],
    joint_names: Sequence[str],
    *,
    fps: float,
    mincutoff: float,
    beta: float,
) -> list[dict[str, Any]]:
    filter_indexes = [
        idx
        for idx, name in enumerate(joint_names)
        if name in _foot_joint_names(joint_names) or name.startswith("left_hand_") or name.startswith("right_hand_")
    ]
    filters: dict[tuple[int, int], OneEuroFilter] = {}
    refined_frames: list[dict[str, Any]] = []
    previous_t: float | None = None
    for frame in frames:
        refined = copy.deepcopy(dict(frame))
        joints = _joint_vectors(refined)
        t = float(refined.get("t", 0.0))
        dt = (t - previous_t) if previous_t is not None else (1.0 / fps)
        current_fps = 1.0 / dt if dt > 0.0 else fps
        previous_t = t
        for joint_idx in filter_indexes:
            if joint_idx >= len(joints):
                continue
            for axis in range(3):
                key = (joint_idx, axis)
                filt = filters.setdefault(key, OneEuroFilter(freq=current_fps, mincutoff=mincutoff, beta=beta))
                joints[joint_idx][axis] = filt.filter(float(joints[joint_idx][axis]), freq=current_fps)
        refined["joints_world"] = joints
        refined_frames.append(refined)
    return refined_frames


def _apply_bone_lengths(
    frames: Sequence[Mapping[str, Any]],
    joint_names: Sequence[str],
    bone_lengths: Mapping[tuple[int, int], float],
) -> list[dict[str, Any]]:
    constrained: list[dict[str, Any]] = []
    for frame in frames:
        output = copy.deepcopy(dict(frame))
        joints = _joint_vectors(output)
        for parent_idx, child_idx in bone_lengths:
            if parent_idx >= len(joints) or child_idx >= len(joints):
                continue
            target = float(bone_lengths[(parent_idx, child_idx)])
            if target <= 0.0:
                continue
            parent = joints[parent_idx]
            child = joints[child_idx]
            direction = [child[axis] - parent[axis] for axis in range(3)]
            length = math.sqrt(sum(value * value for value in direction))
            if length <= 1e-9:
                continue
            scale = target / length
            joints[child_idx] = [parent[axis] + direction[axis] * scale for axis in range(3)]
        output["joints_world"] = joints
        constrained.append(output)
    return constrained


def _median_bone_lengths(frames: Sequence[Mapping[str, Any]], joint_names: Sequence[str]) -> dict[tuple[int, int], float]:
    index_by_name = {name: idx for idx, name in enumerate(joint_names)}
    lengths: dict[tuple[int, int], list[float]] = {}
    for parent_name, child_name in BODY17_BONE_EDGES:
        if parent_name not in index_by_name or child_name not in index_by_name:
            continue
        parent_idx = index_by_name[parent_name]
        child_idx = index_by_name[child_name]
        for frame in frames:
            joints = _joint_vectors(frame)
            conf = frame.get("joint_conf", [])
            if parent_idx >= len(joints) or child_idx >= len(joints):
                continue
            if not isinstance(conf, list) or parent_idx >= len(conf) or child_idx >= len(conf):
                continue
            if float(conf[parent_idx]) < MIN_BONE_CONFIDENCE or float(conf[child_idx]) < MIN_BONE_CONFIDENCE:
                continue
            length = math.dist(joints[parent_idx], joints[child_idx])
            if length > 0.0:
                lengths.setdefault((parent_idx, child_idx), []).append(length)
    return {edge: float(median(values)) for edge, values in lengths.items() if values}


def _provenance_with_temporal_refine(
    provenance: Any,
    *,
    mincutoff: float,
    beta: float,
    motionbert_window_max_frames: int,
    motionbert_metrics: Mapping[str, Any],
    grounding_metrics: Mapping[str, int],
) -> dict[str, Any]:
    output = dict(provenance) if isinstance(provenance, Mapping) else {}
    motionbert_status = "applied" if int(motionbert_metrics["motionbert_frame_count"]) > 0 else "not_configured"
    output["temporal_refine"] = {
        "motionbert": motionbert_status,
        "motionbert_model_id": str(motionbert_metrics.get("motionbert_model_id", "")),
        "motionbert_window_max_frames": motionbert_window_max_frames,
        "motionbert_window_count": int(motionbert_metrics["motionbert_window_count"]),
        "motionbert_frame_count": int(motionbert_metrics["motionbert_frame_count"]),
        "motionbert_body_format": "h36m_17",
        "one_euro": {
            "mincutoff": mincutoff,
            "beta": beta,
            "applied_joint_groups": ["feet", "hands"],
        },
        "bone_length_constraint": "body17_median_per_player",
    }
    output["world_grounding"] = {
        "support_foot_strategy": "max_conf_lowest_z_lowest_vertical_velocity_5f",
        "ground_plane_z_m": 0.0,
        "z_axis": "up",
        "support_frame_count": int(grounding_metrics["support_frame_count"]),
        "foot_lock": {
            "locked_frame_count": int(grounding_metrics["foot_lock_locked_frame_count"]),
            "max_vertical_speed_mps": FOOT_LOCK_MAX_VERTICAL_SPEED_MPS,
            "reanchor_distance_m": FOOT_LOCK_REANCHOR_DISTANCE_M,
        },
        "airborne": {
            "held_frame_count": int(grounding_metrics["airborne_held_frame_count"]),
            "reanchored_landing_count": int(grounding_metrics["reanchored_landing_count"]),
        },
    }
    return output


def _apply_lane_a_world_grounding(
    frames: Sequence[Mapping[str, Any]],
    joint_names: Sequence[str],
    *,
    fps: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    foot_indexes = _grounding_foot_indexes(joint_names)
    metrics = _empty_grounding_metrics()
    if not foot_indexes:
        return [copy.deepcopy(dict(frame)) for frame in frames], metrics

    grounded: list[dict[str, Any]] = []
    locked_xy_by_joint: dict[int, list[float]] = {}
    last_ground_anchor_xy: list[float] | None = None
    last_support_idx: int | None = None
    was_airborne = False
    for frame_pos, frame in enumerate(frames):
        output = copy.deepcopy(dict(frame))
        joints = _joint_vectors(output)
        if not joints:
            grounded.append(output)
            continue
        candidates = _support_foot_candidates(frames, frame_pos, joints, output, foot_indexes, fps=fps)
        contact_candidates = [candidate for candidate in candidates if candidate["z"] <= GROUND_CONTACT_MAX_Z_M]
        if not contact_candidates:
            reference_idx = _airborne_reference_idx(candidates, last_support_idx, len(joints))
            if reference_idx is not None and last_ground_anchor_xy is not None:
                dx = last_ground_anchor_xy[0] - joints[reference_idx][0]
                dy = last_ground_anchor_xy[1] - joints[reference_idx][1]
                _translate_xy(joints, dx=dx, dy=dy)
                metrics["airborne_held_frame_count"] += 1
            output["joints_world"] = joints
            grounded.append(output)
            locked_xy_by_joint.clear()
            was_airborne = True
            continue

        support = min(
            contact_candidates,
            key=lambda candidate: (
                -candidate["confidence"],
                candidate["z"],
                candidate["vertical_velocity_mps"],
                candidate["idx"],
            ),
        )
        support_idx = int(support["idx"])
        if was_airborne:
            metrics["reanchored_landing_count"] += 1
            locked_xy_by_joint.clear()
        support_xy = [float(joints[support_idx][0]), float(joints[support_idx][1])]
        lock_xy = locked_xy_by_joint.get(support_idx)
        low_slow = float(support["vertical_velocity_mps"]) <= FOOT_LOCK_MAX_VERTICAL_SPEED_MPS
        if low_slow:
            if lock_xy is None or _xy_distance(lock_xy, support_xy) > FOOT_LOCK_REANCHOR_DISTANCE_M:
                lock_xy = support_xy
                locked_xy_by_joint[support_idx] = lock_xy
            else:
                _translate_xy(joints, dx=lock_xy[0] - support_xy[0], dy=lock_xy[1] - support_xy[1])
                metrics["foot_lock_locked_frame_count"] += 1
        else:
            lock_xy = support_xy
            locked_xy_by_joint[support_idx] = lock_xy

        support_z = float(joints[support_idx][2])
        if abs(support_z) <= GROUND_CONTACT_MAX_Z_M:
            _translate_z(joints, dz=-support_z)
        output["joints_world"] = joints
        grounded.append(output)
        metrics["support_frame_count"] += 1
        last_ground_anchor_xy = list(lock_xy)
        last_support_idx = support_idx
        was_airborne = False
    return grounded, metrics


def _support_foot_candidates(
    frames: Sequence[Mapping[str, Any]],
    frame_pos: int,
    joints: Sequence[Sequence[float]],
    frame: Mapping[str, Any],
    foot_indexes: Sequence[int],
    *,
    fps: float,
) -> list[dict[str, float | int]]:
    conf = frame.get("joint_conf", [])
    candidates: list[dict[str, float | int]] = []
    for idx in foot_indexes:
        if idx >= len(joints):
            continue
        confidence = _joint_confidence(conf, idx)
        if confidence < MIN_BONE_CONFIDENCE:
            continue
        candidates.append(
            {
                "idx": idx,
                "confidence": confidence,
                "z": float(joints[idx][2]),
                "vertical_velocity_mps": _vertical_velocity_mps(frames, frame_pos, idx, fps=fps),
            }
        )
    return candidates


def _vertical_velocity_mps(
    frames: Sequence[Mapping[str, Any]],
    frame_pos: int,
    joint_idx: int,
    *,
    fps: float,
) -> float:
    current = frames[frame_pos]
    current_joints = _joint_vectors(current)
    if joint_idx >= len(current_joints):
        return 0.0
    current_t = float(current.get("t", frame_pos / fps))
    current_z = float(current_joints[joint_idx][2])
    for previous_pos in range(frame_pos - 1, max(-1, frame_pos - 6), -1):
        previous = frames[previous_pos]
        previous_joints = _joint_vectors(previous)
        if joint_idx >= len(previous_joints):
            continue
        previous_t = float(previous.get("t", previous_pos / fps))
        dt = current_t - previous_t
        if dt <= 0.0:
            continue
        return abs(current_z - float(previous_joints[joint_idx][2])) / dt
    return 0.0


def _grounding_foot_indexes(joint_names: Sequence[str]) -> list[int]:
    names = _foot_joint_names(joint_names) | {"left_ankle", "right_ankle"}
    return [idx for idx, name in enumerate(joint_names) if name in names]


def _airborne_reference_idx(
    candidates: Sequence[Mapping[str, float | int]],
    last_support_idx: int | None,
    joint_count: int,
) -> int | None:
    if last_support_idx is not None and last_support_idx < joint_count:
        return last_support_idx
    if not candidates:
        return None
    return int(min(candidates, key=lambda candidate: (float(candidate["z"]), -float(candidate["confidence"])))["idx"])


def _joint_confidence(conf: Any, joint_idx: int) -> float:
    if not isinstance(conf, list) or joint_idx >= len(conf):
        return 0.0
    return float(conf[joint_idx])


def _translate_xy(joints: list[list[float]], *, dx: float, dy: float) -> None:
    if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
        return
    for joint in joints:
        joint[0] = float(joint[0]) + dx
        joint[1] = float(joint[1]) + dy


def _translate_z(joints: list[list[float]], *, dz: float) -> None:
    if abs(dz) <= 1e-12:
        return
    for joint in joints:
        joint[2] = float(joint[2]) + dz


def _xy_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1]))


def _empty_grounding_metrics() -> dict[str, int]:
    return {
        "support_frame_count": 0,
        "foot_lock_locked_frame_count": 0,
        "airborne_held_frame_count": 0,
        "reanchored_landing_count": 0,
    }


def _add_grounding_metrics(total: dict[str, int], increment: Mapping[str, int]) -> None:
    for key in total:
        total[key] += int(increment.get(key, 0))


def _empty_motionbert_metrics(runtime: Any | None) -> dict[str, Any]:
    return {
        "motionbert_model_id": str(getattr(runtime, "model_id", "")) if runtime is not None else "",
        "motionbert_window_count": 0,
        "motionbert_frame_count": 0,
    }


def _add_motionbert_metrics(total: dict[str, Any], increment: Mapping[str, int]) -> None:
    total["motionbert_window_count"] = int(total["motionbert_window_count"]) + int(increment.get("motionbert_window_count", 0))
    total["motionbert_frame_count"] = int(total["motionbert_frame_count"]) + int(increment.get("motionbert_frame_count", 0))


def _foot_joint_names(joint_names: Sequence[str]) -> set[str]:
    return {name for name in joint_names if "toe" in name or name.endswith("_heel")}


def _body17_indexes(joint_names: Sequence[str]) -> list[int]:
    index_by_name = {name: idx for idx, name in enumerate(joint_names)}
    return [index_by_name[name] for name in BODY17_JOINT_NAMES if name in index_by_name]


def _joint_vectors(frame: Mapping[str, Any]) -> list[list[float]]:
    joints = frame.get("joints_world", [])
    if not isinstance(joints, list):
        return []
    return [[float(value) for value in joint] for joint in joints]


def _vector3(values: Sequence[float], *, name: str) -> list[float]:
    if len(values) != 3:
        raise RuntimeError(f"{name} must be a 3-vector")
    return [float(value) for value in values]


def _infer_fps(players: Sequence[Any]) -> float:
    deltas: list[float] = []
    for player in players:
        if not isinstance(player, Mapping):
            continue
        times = [
            float(frame["t"])
            for frame in player.get("frames", [])
            if isinstance(frame, Mapping) and "t" in frame
        ]
        for previous, current in zip(times, times[1:]):
            if current > previous:
                deltas.append(current - previous)
    if not deltas:
        return 30.0
    return 1.0 / median(deltas)


class OneEuroFilter:
    def __init__(
        self,
        *,
        freq: float,
        mincutoff: float,
        beta: float,
        dcutoff: float = DEFAULT_ONE_EURO_DCUTOFF,
    ) -> None:
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_previous: float | None = None
        self.x_hat_previous: float | None = None
        self.dx_hat_previous = 0.0

    def filter(self, value: float, *, freq: float | None = None) -> float:
        if freq is not None and freq > 0.0:
            self.freq = float(freq)
        if self.x_previous is None or self.x_hat_previous is None:
            self.x_previous = float(value)
            self.x_hat_previous = float(value)
            return float(value)
        dx = (float(value) - self.x_previous) * self.freq
        alpha_d = _alpha(self.freq, self.dcutoff)
        dx_hat = _exponential_smooth(alpha_d, dx, self.dx_hat_previous)
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        alpha = _alpha(self.freq, cutoff)
        x_hat = _exponential_smooth(alpha, float(value), self.x_hat_previous)
        self.x_previous = float(value)
        self.dx_hat_previous = dx_hat
        self.x_hat_previous = x_hat
        return x_hat


def _alpha(freq: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    te = 1.0 / freq
    return 1.0 / (1.0 + tau / te)


def _exponential_smooth(alpha: float, value: float, previous: float) -> float:
    return alpha * value + (1.0 - alpha) * previous


__all__ = ["OneEuroFilter", "refine_lane_a_skeleton3d"]
