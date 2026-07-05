"""Temporal refinement for production skeleton3d artifacts."""

from __future__ import annotations

import copy
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from .external_gt_body_prediction_schema import MHR70_JOINT_NAMES
from .model_manifest import verify_model_checkpoint
from .skeleton3d import SAM3D_BODY_MHR70_SEMANTIC_MAP
from .skeleton_lift_2d import _DEFAULT_LEG_DERIVED_RATIOS, LEG_BONE_JOINT_PAIRS


DEFAULT_ONE_EURO_MINCUTOFF = 1.0
DEFAULT_ONE_EURO_BETA = 0.3
DEFAULT_ONE_EURO_DCUTOFF = 1.0
DEFAULT_CORE_BODY_ONE_EURO_MINCUTOFF = 0.45
DEFAULT_CORE_BODY_ONE_EURO_BETA = 0.05
DEFAULT_WRIST_ONE_EURO_MINCUTOFF = 1000.0
DEFAULT_WRIST_ONE_EURO_BETA = 0.0
# Near pass-through, matching the verified wrist treatment: stance-phase measurement
# (runs/sam3d_foot_wander_20260703T1024Z/REPORT.md) showed the "feet" one-euro group
# inheriting the generic core-ish default lag was the dominant source of ANKLE/HEEL/TOE
# stance-phase slide in the SAM-3D refine chain, not bone-length or grounding.
DEFAULT_FOOT_ONE_EURO_MINCUTOFF = 1000.0
DEFAULT_FOOT_ONE_EURO_BETA = 0.0
DEFAULT_SAM3D_SMOOTHING_MAX_DISPLACEMENT_M = 0.03
DEFAULT_LOW_CONFIDENCE_JOINT_THRESHOLD = 0.25
DEFAULT_PLAUSIBILITY_JOINT_CONFIDENCE_FLOOR = 0.25
DEFAULT_PLAUSIBILITY_MAX_BONE_ZSCORE = 6.0
DEFAULT_PLAUSIBILITY_MIN_BONE_SAMPLES = 4
DEFAULT_PLAUSIBILITY_MIN_SIGMA_M = 0.03
DEFAULT_STATURE_BAND_M = (1.4, 1.8)
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
CORE_BODY_SPEED_FLAG_MPS = 3.0
SINGLE_FRAME_JUMP_FLAG_M = 0.5
CORE_SPEED_SUSTAINED_FRAME_COUNT = 2
NO_SMOOTHING_FLAG = "none"
SAM3D_BODY_JOINT_SOURCE = "sam3d_body_joints"
SAM3D_WRIST_BONE_LOCK_PROVENANCE_KEY = "sam3d_wrist_bone_lock"
DEFAULT_SAM3D_WRIST_LOCK_CONFIDENCE_FLOOR = 0.25
DEFAULT_SAM3D_WRIST_LOCK_DEGENERATE_EPSILON_M = 1e-6
DEFAULT_PLAYER_BONE_LENGTHS_PATH = (
    Path(__file__).resolve().parents[2] / "runs" / "bone_calib_20260703T0102Z" / "player_bone_lengths.json"
)

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
            raise ValueError("MotionBERTTemporalRuntime expects COCO body17 joint order")
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
    core_one_euro_mincutoff: float = DEFAULT_CORE_BODY_ONE_EURO_MINCUTOFF,
    core_one_euro_beta: float = DEFAULT_CORE_BODY_ONE_EURO_BETA,
    wrist_one_euro_mincutoff: float = DEFAULT_WRIST_ONE_EURO_MINCUTOFF,
    wrist_one_euro_beta: float = DEFAULT_WRIST_ONE_EURO_BETA,
    foot_one_euro_mincutoff: float | None = None,
    foot_one_euro_beta: float | None = None,
    smoothing_max_displacement_m: float | None = None,
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_JOINT_THRESHOLD,
    motionbert_window_max_frames: int = DEFAULT_MOTIONBERT_WINDOW_MAX_FRAMES,
    motionbert_runtime: Any | None = None,
    apply_world_grounding: bool = True,
) -> dict[str, Any]:
    """Return a refined Lane A skeleton payload without fabricating frames.

    ``foot_one_euro_mincutoff``/``foot_one_euro_beta`` default to ``None``, which keeps
    every existing caller's behavior unchanged: the "feet" joint group falls back to the
    generic ``one_euro_mincutoff``/``one_euro_beta`` params exactly as before. Pass explicit
    values to give the feet their own (typically lower-lag) one-euro tuning independent of
    the generic default; ``refine_sam3d_skeleton3d`` does this by default.
    """

    if one_euro_mincutoff <= 0.0:
        raise ValueError("one_euro_mincutoff must be positive")
    if one_euro_beta < 0.0:
        raise ValueError("one_euro_beta must be non-negative")
    if core_one_euro_mincutoff <= 0.0:
        raise ValueError("core_one_euro_mincutoff must be positive")
    if core_one_euro_beta < 0.0:
        raise ValueError("core_one_euro_beta must be non-negative")
    if wrist_one_euro_mincutoff <= 0.0:
        raise ValueError("wrist_one_euro_mincutoff must be positive")
    if wrist_one_euro_beta < 0.0:
        raise ValueError("wrist_one_euro_beta must be non-negative")
    if foot_one_euro_mincutoff is not None and foot_one_euro_mincutoff <= 0.0:
        raise ValueError("foot_one_euro_mincutoff must be positive")
    if foot_one_euro_beta is not None and foot_one_euro_beta < 0.0:
        raise ValueError("foot_one_euro_beta must be non-negative")
    if smoothing_max_displacement_m is not None and smoothing_max_displacement_m <= 0.0:
        raise ValueError("smoothing_max_displacement_m must be positive")
    if not 0.0 <= low_confidence_threshold <= 1.0:
        raise ValueError("low_confidence_threshold must be in [0, 1]")
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

    effective_foot_mincutoff = foot_one_euro_mincutoff if foot_one_euro_mincutoff is not None else one_euro_mincutoff
    effective_foot_beta = foot_one_euro_beta if foot_one_euro_beta is not None else one_euro_beta

    output = copy.deepcopy(dict(skeleton3d))
    output_players: list[dict[str, Any]] = []
    grounding_metrics = _empty_grounding_metrics()
    motionbert_metrics = _empty_motionbert_metrics(motionbert_runtime)
    smoothing_metrics = _empty_smoothing_metrics()
    core_clamp_engagement_by_player: dict[str, dict[str, Any]] = {}
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
        filtered_frames, player_smoothing_metrics = _apply_one_euro(
            motionbert_frames,
            joint_names,
            fps=inferred_fps,
            mincutoff=one_euro_mincutoff,
            beta=one_euro_beta,
            core_mincutoff=core_one_euro_mincutoff,
            core_beta=core_one_euro_beta,
            wrist_mincutoff=wrist_one_euro_mincutoff,
            wrist_beta=wrist_one_euro_beta,
            foot_mincutoff=effective_foot_mincutoff,
            foot_beta=effective_foot_beta,
            max_displacement_m=smoothing_max_displacement_m,
            low_confidence_threshold=low_confidence_threshold,
        )
        _add_smoothing_metrics(smoothing_metrics, player_smoothing_metrics)
        constrained_frames = _apply_bone_lengths(filtered_frames, joint_names, bone_lengths)
        if apply_world_grounding:
            grounded_frames, player_grounding_metrics = _apply_lane_a_world_grounding(
                constrained_frames,
                joint_names,
                fps=inferred_fps,
            )
        else:
            grounded_frames = [copy.deepcopy(dict(frame)) for frame in constrained_frames]
            player_grounding_metrics = _empty_grounding_metrics()
        guarded_frames, player_final_guard_metrics = _apply_final_core_jitter_guard(
            grounded_frames,
            joint_names,
            fps=inferred_fps,
            max_displacement_m=smoothing_max_displacement_m,
            raw_reference_frames=motionbert_frames if smoothing_max_displacement_m is not None else None,
        )
        _add_smoothing_metrics(smoothing_metrics, player_final_guard_metrics)
        _add_grounding_metrics(grounding_metrics, player_grounding_metrics)
        player_output["frames"] = guarded_frames
        player_id = str(player.get("id", len(output_players)))
        core_clamp_engagement_by_player[player_id] = _core_body_speed_clamp_engagement(
            guarded_frames,
            joint_names,
        )
        output_players.append(player_output)

    output["players"] = output_players
    output["provenance"] = _provenance_with_temporal_refine(
        output.get("provenance"),
        mincutoff=one_euro_mincutoff,
        beta=one_euro_beta,
        core_mincutoff=core_one_euro_mincutoff,
        core_beta=core_one_euro_beta,
        wrist_mincutoff=wrist_one_euro_mincutoff,
        wrist_beta=wrist_one_euro_beta,
        foot_mincutoff=effective_foot_mincutoff,
        foot_beta=effective_foot_beta,
        low_confidence_threshold=low_confidence_threshold,
        motionbert_window_max_frames=motionbert_window_max_frames,
        motionbert_metrics=motionbert_metrics,
        smoothing_metrics=smoothing_metrics,
        grounding_metrics=grounding_metrics,
        core_clamp_engagement_by_player=core_clamp_engagement_by_player,
        world_grounding_applied=apply_world_grounding,
        smoothing_max_displacement_m=smoothing_max_displacement_m,
    )
    return output


def refine_sam3d_skeleton3d(
    skeleton3d: Mapping[str, Any],
    *,
    fps: float | None = None,
    one_euro_mincutoff: float = DEFAULT_ONE_EURO_MINCUTOFF,
    one_euro_beta: float = DEFAULT_ONE_EURO_BETA,
    core_one_euro_mincutoff: float = DEFAULT_CORE_BODY_ONE_EURO_MINCUTOFF,
    core_one_euro_beta: float = DEFAULT_CORE_BODY_ONE_EURO_BETA,
    wrist_one_euro_mincutoff: float = DEFAULT_WRIST_ONE_EURO_MINCUTOFF,
    wrist_one_euro_beta: float = DEFAULT_WRIST_ONE_EURO_BETA,
    sam3d_foot_low_lag_smoothing: bool = True,
    foot_one_euro_mincutoff: float = DEFAULT_FOOT_ONE_EURO_MINCUTOFF,
    foot_one_euro_beta: float = DEFAULT_FOOT_ONE_EURO_BETA,
    smoothing_max_displacement_m: float | None = DEFAULT_SAM3D_SMOOTHING_MAX_DISPLACEMENT_M,
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_JOINT_THRESHOLD,
    plausibility_joint_confidence_floor: float = DEFAULT_PLAUSIBILITY_JOINT_CONFIDENCE_FLOOR,
    plausibility_max_bone_zscore: float = DEFAULT_PLAUSIBILITY_MAX_BONE_ZSCORE,
    plausibility_min_bone_samples: int = DEFAULT_PLAUSIBILITY_MIN_BONE_SAMPLES,
    plausibility_min_sigma_m: float = DEFAULT_PLAUSIBILITY_MIN_SIGMA_M,
    max_wrist_peak_delta_frames: int = 1,
    apply_world_grounding: bool = False,
) -> dict[str, Any]:
    """Port temporal/body plausibility gates onto SAM-3D body-mode joints.

    SAM-3D MHR70 artifacts usually carry generic ``sam3dbody_joint_###``
    names. This entrypoint keeps all 70 joints intact and uses the local
    semantic map only for wrist/limb/ankle gates.

    ``sam3d_foot_low_lag_smoothing`` (default on) gives the ANKLE/HEEL/TOE joint group its
    own near-pass-through one-euro tuning (``foot_one_euro_mincutoff``/``foot_one_euro_beta``,
    same style as the verified wrist treatment) instead of silently inheriting the
    core-body-ish generic default. Stage-by-stage measurement on real Wolverine artifacts
    (runs/sam3d_foot_wander_20260703T1024Z/REPORT.md) found this one-euro lag -- not
    bone-length enforcement or world grounding -- was the dominant source of stance-phase
    foot slide in this chain. Set to False to restore the pre-fix behavior for rollback.
    """

    if not _is_sam3d_skeleton_payload(skeleton3d):
        raise ValueError("SAM-3D post-processing requires a sam3d_body_joints skeleton3d payload")
    if not 0.0 <= plausibility_joint_confidence_floor <= 1.0:
        raise ValueError("plausibility_joint_confidence_floor must be in [0, 1]")
    if plausibility_max_bone_zscore <= 0.0:
        raise ValueError("plausibility_max_bone_zscore must be positive")
    if plausibility_min_bone_samples <= 0:
        raise ValueError("plausibility_min_bone_samples must be positive")
    if plausibility_min_sigma_m <= 0.0:
        raise ValueError("plausibility_min_sigma_m must be positive")
    if max_wrist_peak_delta_frames < 0:
        raise ValueError("max_wrist_peak_delta_frames must be non-negative")
    if smoothing_max_displacement_m is not None and smoothing_max_displacement_m <= 0.0:
        raise ValueError("smoothing_max_displacement_m must be positive")

    before = copy.deepcopy(dict(skeleton3d))
    plausibility_checked, plausibility = _apply_sam3d_skeleton_plausibility(
        before,
        confidence_floor=plausibility_joint_confidence_floor,
        max_bone_zscore=plausibility_max_bone_zscore,
        min_bone_samples=plausibility_min_bone_samples,
        min_sigma_m=plausibility_min_sigma_m,
    )
    refined = refine_lane_a_skeleton3d(
        skeleton3d,
        fps=fps,
        one_euro_mincutoff=one_euro_mincutoff,
        one_euro_beta=one_euro_beta,
        core_one_euro_mincutoff=core_one_euro_mincutoff,
        core_one_euro_beta=core_one_euro_beta,
        wrist_one_euro_mincutoff=wrist_one_euro_mincutoff,
        wrist_one_euro_beta=wrist_one_euro_beta,
        foot_one_euro_mincutoff=foot_one_euro_mincutoff if sam3d_foot_low_lag_smoothing else None,
        foot_one_euro_beta=foot_one_euro_beta if sam3d_foot_low_lag_smoothing else None,
        smoothing_max_displacement_m=smoothing_max_displacement_m,
        low_confidence_threshold=low_confidence_threshold,
        motionbert_runtime=None,
        apply_world_grounding=apply_world_grounding,
    )
    _copy_sam3d_plausibility_flags(refined, plausibility_checked)
    wrist_timing = compare_wrist_peak_timing(
        before,
        refined,
        max_allowed_delta_frames=max_wrist_peak_delta_frames,
    )
    provenance = dict(refined.get("provenance", {}))
    temporal = dict(provenance.get("temporal_refine", {}))
    one_euro = dict(temporal.get("one_euro", {}))
    one_euro["filtered_joint_count"] = len(refined.get("joint_names", []))
    temporal["one_euro"] = one_euro
    temporal["source"] = SAM3D_BODY_JOINT_SOURCE
    temporal["motionbert"] = "not_applicable_sam3d_body70"
    temporal["wrist_peak_timing"] = wrist_timing
    temporal["wrist_peak_timing_gate_pass"] = wrist_timing.get("status") == "pass"
    provenance["temporal_refine"] = temporal
    provenance["sam3d_skeleton_plausibility"] = plausibility
    provenance["sam3d_foot_treatment"] = {
        "low_lag_smoothing_enabled": sam3d_foot_low_lag_smoothing,
        "foot_one_euro_mincutoff": foot_one_euro_mincutoff if sam3d_foot_low_lag_smoothing else one_euro_mincutoff,
        "foot_one_euro_beta": foot_one_euro_beta if sam3d_foot_low_lag_smoothing else one_euro_beta,
        "heel_toe_canonical_name_resolution": "mhr70_positional_fallback",
        "bone_length_leg_chain_unchanged": True,
        "world_grounding_unchanged": True,
    }
    provenance["stature_check"] = _build_sam3d_stature_check(refined)
    provenance["sam3d_postprocess"] = {
        "source": SAM3D_BODY_JOINT_SOURCE,
        "all_joints_preserved": True,
        "joint_count": len(refined.get("joint_names", [])),
        "protected_eval_labels_used": False,
        "internal_val_only": True,
    }
    refined["provenance"] = provenance
    return refined


def apply_sam3d_wrist_bone_lock(
    skeleton3d: Mapping[str, Any],
    *,
    canonical_bone_lengths: Mapping[str, Any] | str | Path | None = None,
    enabled: bool = True,
    confidence_floor: float = DEFAULT_SAM3D_WRIST_LOCK_CONFIDENCE_FLOOR,
    degenerate_epsilon_m: float = DEFAULT_SAM3D_WRIST_LOCK_DEGENERATE_EPSILON_M,
    max_wrist_peak_delta_frames: int = 0,
) -> dict[str, Any]:
    """Project SAM-3D wrists to canonical lower-arm length without moving elbows."""

    if not enabled:
        return copy.deepcopy(dict(skeleton3d))
    if not 0.0 <= confidence_floor <= 1.0:
        raise ValueError("confidence_floor must be in [0, 1]")
    if degenerate_epsilon_m <= 0.0:
        raise ValueError("degenerate_epsilon_m must be positive")
    if max_wrist_peak_delta_frames < 0:
        raise ValueError("max_wrist_peak_delta_frames must be non-negative")

    before = copy.deepcopy(dict(skeleton3d))
    output = copy.deepcopy(dict(skeleton3d))
    provenance = dict(output.get("provenance", {}))
    lock_record = _empty_sam3d_wrist_bone_lock_record(
        status="applied",
        confidence_floor=confidence_floor,
        degenerate_epsilon_m=degenerate_epsilon_m,
    )

    if not _is_sam3d_skeleton_payload(output):
        lock_record["status"] = "skipped_non_sam3d"
        provenance[SAM3D_WRIST_BONE_LOCK_PROVENANCE_KEY] = lock_record
        output["provenance"] = provenance
        return output

    joint_names = _string_joint_names(output.get("joint_names"))
    index_by_name = _semantic_index_by_name(joint_names)
    wrist_edges = {
        "left_lower_arm": (index_by_name.get("left_elbow"), index_by_name.get("left_wrist")),
        "right_lower_arm": (index_by_name.get("right_elbow"), index_by_name.get("right_wrist")),
    }
    if any(parent is None or child is None for parent, child in wrist_edges.values()):
        lock_record["status"] = "skipped_missing_wrist_semantics"
        provenance[SAM3D_WRIST_BONE_LOCK_PROVENANCE_KEY] = lock_record
        output["provenance"] = provenance
        return output

    canonical_payload, canonical_source = _load_wrist_lock_canonical_payload(canonical_bone_lengths)
    lock_record["canonical_source"] = canonical_source
    players = output.get("players")
    if not isinstance(players, list):
        lock_record["status"] = "skipped_missing_players"
        provenance[SAM3D_WRIST_BONE_LOCK_PROVENANCE_KEY] = lock_record
        output["provenance"] = provenance
        return output

    total_locked = 0
    total_unlocked = 0
    for player_pos, player in enumerate(players):
        if not isinstance(player, dict):
            continue
        player_id = str(player.get("id", player_pos))
        frames = player.get("frames")
        if not isinstance(frames, list):
            frames = []
            player["frames"] = frames
        player_summary: dict[str, Any] = {}
        targets = {
            bone_name: _sam3d_wrist_lock_target_length(
                canonical_payload,
                player_id=player_id,
                bone_name=bone_name,
            )
            for bone_name in wrist_edges
        }
        for bone_name, (elbow_idx, wrist_idx) in wrist_edges.items():
            assert elbow_idx is not None and wrist_idx is not None
            target = targets[bone_name]
            summary = _empty_wrist_lock_bone_summary(
                target_length_m=target["length_m"],
                target_source=target["source"],
                frame_count=len(frames),
            )
            for frame in frames:
                if not isinstance(frame, dict):
                    summary["missing_joint_frame_count"] += 1
                    total_unlocked += 1
                    continue
                joints = frame.get("joints_world")
                if not isinstance(joints, list) or elbow_idx >= len(joints) or wrist_idx >= len(joints):
                    summary["missing_joint_frame_count"] += 1
                    total_unlocked += 1
                    continue
                elbow = _finite_joint3(joints[elbow_idx])
                wrist = _finite_joint3(joints[wrist_idx])
                if elbow is None or wrist is None:
                    summary["missing_joint_frame_count"] += 1
                    total_unlocked += 1
                    continue
                if _joint_confidence_safe(frame.get("joint_conf"), elbow_idx) < confidence_floor or _joint_confidence_safe(
                    frame.get("joint_conf"),
                    wrist_idx,
                ) < confidence_floor:
                    summary["low_confidence_frame_count"] += 1
                    _record_wrist_lock_length_sample(summary, elbow, wrist, target["length_m"])
                    total_unlocked += 1
                    continue
                direction = [wrist[axis] - elbow[axis] for axis in range(3)]
                length = math.sqrt(sum(value * value for value in direction))
                _record_wrist_lock_length_sample(summary, elbow, wrist, target["length_m"])
                if length <= degenerate_epsilon_m:
                    summary["degenerate_frame_count"] += 1
                    total_unlocked += 1
                    continue
                unit = [value / length for value in direction]
                locked_wrist = [
                    elbow[axis] + float(target["length_m"]) * unit[axis]
                    for axis in range(3)
                ]
                joints[wrist_idx] = locked_wrist
                summary["locked_frame_count"] += 1
                total_locked += 1
                _record_wrist_lock_post_length_sample(summary, elbow, locked_wrist, target["length_m"])
            _finalize_wrist_lock_bone_summary(summary)
            player_summary[bone_name] = summary
        player_summary["locked_frame_count"] = sum(int(item["locked_frame_count"]) for item in player_summary.values())
        player_summary["unlocked_frame_count"] = sum(
            int(item["frame_count"]) - int(item["locked_frame_count"])
            for item in player_summary.values()
            if isinstance(item, Mapping)
        )
        lock_record["players"][player_id] = player_summary

    lock_record["locked_frame_count"] = total_locked
    lock_record["unlocked_frame_count"] = total_unlocked
    lock_record["wrist_peak_timing_after_lock"] = compare_wrist_direction_peak_timing(
        before,
        output,
        max_allowed_delta_frames=max_wrist_peak_delta_frames,
        min_peak_direction_speed=0.0,
    )
    lock_record["wrist_world_velocity_peak_timing_after_lock"] = compare_wrist_peak_timing(
        before,
        output,
        max_allowed_delta_frames=max_wrist_peak_delta_frames,
        min_peak_speed_mps=0.0,
    )
    provenance[SAM3D_WRIST_BONE_LOCK_PROVENANCE_KEY] = lock_record
    output["provenance"] = provenance
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
    core_mincutoff: float,
    core_beta: float,
    wrist_mincutoff: float,
    wrist_beta: float,
    foot_mincutoff: float,
    foot_beta: float,
    low_confidence_threshold: float,
    max_displacement_m: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    filter_indexes = list(range(len(joint_names)))
    joint_groups = {
        idx: _joint_smoothing_group(name, joint_names)
        for idx, name in enumerate(joint_names)
    }
    filters: dict[tuple[int, int], OneEuroFilter] = {}
    previous_raw_by_joint: dict[int, list[float]] = {}
    previous_output_by_joint: dict[int, list[float]] = {}
    high_core_speed_streaks: dict[int, int] = {}
    refined_frames: list[dict[str, Any]] = []
    metrics = _empty_smoothing_metrics(filtered_joint_count=len(filter_indexes))
    previous_t: float | None = None
    for frame in frames:
        refined = copy.deepcopy(dict(frame))
        joints = _joint_vectors(refined)
        t = float(refined.get("t", 0.0))
        dt = (t - previous_t) if previous_t is not None else (1.0 / fps)
        current_fps = 1.0 / dt if dt > 0.0 else fps
        previous_t = t
        frame_flags: list[list[str]] = [[] for _idx in range(len(joints))]
        raw_joints = [list(joint) for joint in joints]
        conf = refined.get("joint_conf", [])
        for joint_idx in range(len(joints)):
            if _joint_confidence(conf, joint_idx) < low_confidence_threshold:
                _add_joint_flag(frame_flags, joint_idx, "low_confidence_joint")
        for joint_idx in filter_indexes:
            if joint_idx >= len(joints):
                continue
            group = joint_groups.get(joint_idx, "core_body")
            target = list(raw_joints[joint_idx])
            previous_raw = previous_raw_by_joint.get(joint_idx)
            previous_output = previous_output_by_joint.get(joint_idx)
            if previous_raw is not None and previous_output is not None and dt > 0.0:
                displacement_m = math.dist(previous_raw, target)
                output_displacement_m = math.dist(previous_output, target)
                speed_mps = displacement_m / dt
                clamp_flags: list[str] = []
                if displacement_m > SINGLE_FRAME_JUMP_FLAG_M or (
                    group == "wrists" and output_displacement_m > SINGLE_FRAME_JUMP_FLAG_M
                ):
                    clamp_flags.append("single_frame_jump_clamped")
                if group == "core_body":
                    if speed_mps > CORE_BODY_SPEED_FLAG_MPS:
                        high_core_speed_streaks[joint_idx] = high_core_speed_streaks.get(joint_idx, 0) + 1
                    else:
                        high_core_speed_streaks[joint_idx] = 0
                    if high_core_speed_streaks.get(joint_idx, 0) >= CORE_SPEED_SUSTAINED_FRAME_COUNT:
                        clamp_flags.append("core_speed_clamped")
                if clamp_flags:
                    target = _clamp_joint_step(
                        previous_output,
                        target,
                        max_step_m=_max_damped_step_m(group=group, dt=dt),
                    )
                    for flag in clamp_flags:
                        _add_joint_flag(frame_flags, joint_idx, flag)
            if group == "core_body":
                group_mincutoff = core_mincutoff
                group_beta = core_beta
            elif group == "wrists":
                group_mincutoff = wrist_mincutoff
                group_beta = wrist_beta
            elif group == "feet":
                group_mincutoff = foot_mincutoff
                group_beta = foot_beta
            else:
                group_mincutoff = mincutoff
                group_beta = beta
            for axis in range(3):
                key = (joint_idx, axis)
                filt = filters.setdefault(key, OneEuroFilter(freq=current_fps, mincutoff=group_mincutoff, beta=group_beta))
                joints[joint_idx][axis] = filt.filter(float(target[axis]), freq=current_fps)
            if max_displacement_m is not None:
                capped, was_capped = _cap_joint_displacement(
                    raw_joints[joint_idx],
                    joints[joint_idx],
                    max_displacement_m=max_displacement_m,
                )
                if was_capped:
                    joints[joint_idx] = capped
                    _set_one_euro_output_state(filters, joint_idx, capped)
                    _add_joint_flag(frame_flags, joint_idx, "smoothing_displacement_capped")
        for flags in frame_flags:
            for flag in flags:
                metrics["flag_counts"][flag] += 1
        refined["joints_world"] = joints
        refined["smoothing_flag"] = [_format_joint_flags(flags) for flags in frame_flags]
        refined_frames.append(refined)
        previous_raw_by_joint = {
            idx: list(raw_joints[idx])
            for idx in range(min(len(raw_joints), len(joint_names)))
        }
        previous_output_by_joint = {
            idx: list(joints[idx])
            for idx in range(min(len(joints), len(joint_names)))
        }
    return refined_frames, metrics


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


def _cap_joint_displacement(
    raw_joint: Sequence[float],
    candidate_joint: Sequence[float],
    *,
    max_displacement_m: float,
) -> tuple[list[float], bool]:
    displacement = math.dist(raw_joint, candidate_joint)
    if displacement <= float(max_displacement_m) or displacement <= 1e-12:
        return [float(value) for value in candidate_joint], False
    scale = float(max_displacement_m) / displacement
    return (
        [
            float(raw_joint[axis]) + (float(candidate_joint[axis]) - float(raw_joint[axis])) * scale
            for axis in range(3)
        ],
        True,
    )


def _set_one_euro_output_state(
    filters: Mapping[tuple[int, int], "OneEuroFilter"],
    joint_idx: int,
    joint: Sequence[float],
) -> None:
    for axis in range(3):
        filt = filters.get((joint_idx, axis))
        if filt is not None:
            filt.x_hat_previous = float(joint[axis])


def _median_bone_lengths(frames: Sequence[Mapping[str, Any]], joint_names: Sequence[str]) -> dict[tuple[int, int], float]:
    index_by_name = _semantic_index_by_name(joint_names)
    lengths: dict[tuple[int, int], list[float]] = {}
    for parent_name, child_name in BODY17_BONE_EDGES:
        if child_name in {"left_wrist", "right_wrist"}:
            continue
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


def _apply_sam3d_skeleton_plausibility(
    skeleton3d: Mapping[str, Any],
    *,
    confidence_floor: float,
    max_bone_zscore: float,
    min_bone_samples: int,
    min_sigma_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = copy.deepcopy(dict(skeleton3d))
    joint_names = _string_joint_names(output.get("joint_names"))
    bone_pairs = _available_semantic_bone_pairs(joint_names)
    players = output.get("players")
    if not isinstance(players, list):
        return output, _empty_sam3d_plausibility_summary()

    reason_counts: Counter[str] = Counter()
    checked_frame_count = 0
    implausible_frame_count = 0
    bone_stats_by_player: dict[int, dict[tuple[str, str], tuple[float, float]]] = {}
    for player_pos, player in enumerate(players):
        if not isinstance(player, Mapping):
            continue
        frames = player.get("frames")
        if not isinstance(frames, list):
            continue
        bone_stats_by_player[player_pos] = _sam3d_bone_length_stats(
            frames,
            joint_names=joint_names,
            bone_pairs=bone_pairs,
            confidence_floor=confidence_floor,
            min_bone_samples=min_bone_samples,
            min_sigma_m=min_sigma_m,
        )

    for player_pos, player in enumerate(players):
        if not isinstance(player, dict):
            continue
        frames = player.get("frames")
        if not isinstance(frames, list):
            continue
        bone_stats = bone_stats_by_player.get(player_pos, {})
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            checked_frame_count += 1
            reasons = _sam3d_frame_plausibility_reasons(
                frame,
                joint_names=joint_names,
                bone_pairs=bone_pairs,
                bone_stats=bone_stats,
                confidence_floor=confidence_floor,
                max_bone_zscore=max_bone_zscore,
            )
            frame["skeleton_implausible"] = bool(reasons)
            frame["skeleton_plausibility"] = {
                "status": "low_confidence" if reasons else "pass",
                "reasons": reasons,
                "joint_confidence_floor": confidence_floor,
                "max_bone_zscore": max_bone_zscore,
                "source": SAM3D_BODY_JOINT_SOURCE,
            }
            if reasons:
                implausible_frame_count += 1
                reason_counts.update(reason.split(":", 1)[0] for reason in reasons)
                frame["trust_band"] = {
                    "stage": "BODY",
                    "gate_id": "sam3d_skeleton_plausibility",
                    "gate_status": "low_confidence",
                    "badge": "low_confidence",
                    "reason": "; ".join(reasons),
                    "evidence_path": None,
                }

    return output, {
        "artifact_type": "racketsport_sam3d_skeleton_plausibility",
        "source": SAM3D_BODY_JOINT_SOURCE,
        "checked_frame_count": checked_frame_count,
        "implausible_frame_count": implausible_frame_count,
        "reason_counts": dict(sorted(reason_counts.items())),
        "bone_pair_count": len(bone_pairs),
        "joint_confidence_floor": confidence_floor,
        "max_bone_zscore": max_bone_zscore,
        "min_bone_samples": min_bone_samples,
        "min_sigma_m": min_sigma_m,
    }


def _copy_sam3d_plausibility_flags(target: dict[str, Any], checked: Mapping[str, Any]) -> None:
    checked_flags: dict[tuple[str, int | None, float | None, int], dict[str, Any]] = {}
    for player_pos, player in enumerate(checked.get("players", [])):
        if not isinstance(player, Mapping):
            continue
        player_id = str(player.get("id", player_pos))
        frames = player.get("frames")
        if not isinstance(frames, list):
            continue
        for frame_pos, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                continue
            payload = {
                key: copy.deepcopy(frame[key])
                for key in ("skeleton_implausible", "skeleton_plausibility", "trust_band")
                if key in frame
            }
            if not payload:
                continue
            checked_flags[_frame_identity_key(player_id, frame, frame_pos)] = payload

    for player_pos, player in enumerate(target.get("players", [])):
        if not isinstance(player, dict):
            continue
        player_id = str(player.get("id", player_pos))
        frames = player.get("frames")
        if not isinstance(frames, list):
            continue
        for frame_pos, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue
            payload = checked_flags.get(_frame_identity_key(player_id, frame, frame_pos))
            if not payload:
                continue
            frame.update(copy.deepcopy(payload))


def _frame_identity_key(player_id: str, frame: Mapping[str, Any], frame_pos: int) -> tuple[str, int | None, float | None, int]:
    frame_idx_value = frame.get("frame_idx")
    t_value = frame.get("t")
    try:
        frame_idx = int(frame_idx_value) if frame_idx_value is not None else None
    except (TypeError, ValueError):
        frame_idx = None
    try:
        t = float(t_value) if t_value is not None else None
    except (TypeError, ValueError):
        t = None
    return player_id, frame_idx, t, int(frame_pos)


def _empty_sam3d_plausibility_summary() -> dict[str, Any]:
    return {
        "artifact_type": "racketsport_sam3d_skeleton_plausibility",
        "source": SAM3D_BODY_JOINT_SOURCE,
        "checked_frame_count": 0,
        "implausible_frame_count": 0,
        "reason_counts": {},
        "bone_pair_count": 0,
    }


def _available_semantic_bone_pairs(joint_names: Sequence[str]) -> list[tuple[str, str]]:
    index_by_name = _semantic_index_by_name(joint_names)
    return [
        (parent, child)
        for parent, child in BODY17_BONE_EDGES
        if parent in index_by_name and child in index_by_name
    ]


def _sam3d_bone_length_stats(
    frames: Sequence[Mapping[str, Any]],
    *,
    joint_names: Sequence[str],
    bone_pairs: Sequence[tuple[str, str]],
    confidence_floor: float,
    min_bone_samples: int,
    min_sigma_m: float,
) -> dict[tuple[str, str], tuple[float, float]]:
    stats: dict[tuple[str, str], tuple[float, float]] = {}
    for bone in bone_pairs:
        lengths = [
            length
            for frame in frames
            if (length := _sam3d_frame_bone_length(frame, joint_names=joint_names, bone=bone, confidence_floor=confidence_floor)) is not None
        ]
        if len(lengths) < min_bone_samples:
            continue
        center = float(median(lengths))
        abs_deviations = [abs(length - center) for length in lengths]
        sigma = max(float(median(abs_deviations)) * 1.4826, min_sigma_m)
        stats[bone] = (center, sigma)
    return stats


def _sam3d_frame_plausibility_reasons(
    frame: Mapping[str, Any],
    *,
    joint_names: Sequence[str],
    bone_pairs: Sequence[tuple[str, str]],
    bone_stats: Mapping[tuple[str, str], tuple[float, float]],
    confidence_floor: float,
    max_bone_zscore: float,
) -> list[str]:
    reasons: list[str] = []
    low_conf = _sam3d_low_confidence_joints(
        frame,
        joint_names=joint_names,
        bone_pairs=bone_pairs,
        floor=confidence_floor,
    )
    if low_conf:
        reasons.append(f"joint_conf_below_floor:{','.join(low_conf[:6])}")
    for bone, (center, sigma) in bone_stats.items():
        length = _sam3d_frame_bone_length(
            frame,
            joint_names=joint_names,
            bone=bone,
            confidence_floor=0.0,
        )
        if length is None:
            continue
        zscore = abs(length - center) / sigma
        if zscore > max_bone_zscore:
            reasons.append(f"bone_length_zscore:{bone[0]}-{bone[1]}:{zscore:.2f}")
    return reasons


def _sam3d_low_confidence_joints(
    frame: Mapping[str, Any],
    *,
    joint_names: Sequence[str],
    bone_pairs: Sequence[tuple[str, str]],
    floor: float,
) -> list[str]:
    conf = frame.get("joint_conf")
    if not isinstance(conf, Sequence) or isinstance(conf, (str, bytes)):
        return []
    index_by_name = _semantic_index_by_name(joint_names)
    required = sorted({joint for bone in bone_pairs for joint in bone})
    low: list[str] = []
    for name in required:
        idx = index_by_name.get(name)
        if idx is None or idx >= len(conf):
            continue
        try:
            value = float(conf[idx])
        except (TypeError, ValueError):
            value = 0.0
        if value < floor:
            low.append(name)
    return low


def _sam3d_frame_bone_length(
    frame: Mapping[str, Any],
    *,
    joint_names: Sequence[str],
    bone: tuple[str, str],
    confidence_floor: float,
) -> float | None:
    index_by_name = _semantic_index_by_name(joint_names)
    parent_idx = index_by_name.get(bone[0])
    child_idx = index_by_name.get(bone[1])
    if parent_idx is None or child_idx is None:
        return None
    joints = _joint_vectors(frame)
    if parent_idx >= len(joints) or child_idx >= len(joints):
        return None
    conf = frame.get("joint_conf")
    if confidence_floor > 0.0:
        if _joint_confidence(conf, parent_idx) < confidence_floor or _joint_confidence(conf, child_idx) < confidence_floor:
            return None
    return math.dist(joints[parent_idx], joints[child_idx])


def _empty_sam3d_wrist_bone_lock_record(
    *,
    status: str,
    confidence_floor: float,
    degenerate_epsilon_m: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_sam3d_wrist_bone_lock",
        "status": status,
        "enabled": True,
        "source": "direction_preserving_canonical_lower_arm_projection",
        "joint_confidence_floor": confidence_floor,
        "degenerate_epsilon_m": degenerate_epsilon_m,
        "locked_frame_count": 0,
        "unlocked_frame_count": 0,
        "players": {},
    }


def _empty_wrist_lock_bone_summary(
    *,
    target_length_m: float,
    target_source: str,
    frame_count: int,
) -> dict[str, Any]:
    return {
        "target_length_m": round(float(target_length_m), 6),
        "target_source": target_source,
        "frame_count": int(frame_count),
        "locked_frame_count": 0,
        "missing_joint_frame_count": 0,
        "low_confidence_frame_count": 0,
        "degenerate_frame_count": 0,
        "mean_pre_length_m": None,
        "mean_post_length_m": None,
        "mean_abs_pre_length_delta_m": None,
        "mean_abs_post_length_delta_m": None,
        "_pre_lengths": [],
        "_post_lengths": [],
        "_pre_abs_deltas": [],
        "_post_abs_deltas": [],
    }


def _record_wrist_lock_length_sample(
    summary: dict[str, Any],
    elbow: Sequence[float],
    wrist: Sequence[float],
    target_length_m: float,
) -> None:
    length = math.dist(elbow, wrist)
    summary["_pre_lengths"].append(length)
    summary["_pre_abs_deltas"].append(abs(length - float(target_length_m)))


def _record_wrist_lock_post_length_sample(
    summary: dict[str, Any],
    elbow: Sequence[float],
    wrist: Sequence[float],
    target_length_m: float,
) -> None:
    length = math.dist(elbow, wrist)
    summary["_post_lengths"].append(length)
    summary["_post_abs_deltas"].append(abs(length - float(target_length_m)))


def _finalize_wrist_lock_bone_summary(summary: dict[str, Any]) -> None:
    for field, samples_key in (
        ("mean_pre_length_m", "_pre_lengths"),
        ("mean_post_length_m", "_post_lengths"),
        ("mean_abs_pre_length_delta_m", "_pre_abs_deltas"),
        ("mean_abs_post_length_delta_m", "_post_abs_deltas"),
    ):
        samples = [float(value) for value in summary.get(samples_key, [])]
        summary[field] = round(sum(samples) / len(samples), 6) if samples else None
        summary.pop(samples_key, None)


def _load_wrist_lock_canonical_payload(
    canonical_bone_lengths: Mapping[str, Any] | str | Path | None,
) -> tuple[dict[str, Any], str]:
    if canonical_bone_lengths is None:
        if DEFAULT_PLAYER_BONE_LENGTHS_PATH.is_file():
            return (
                json.loads(DEFAULT_PLAYER_BONE_LENGTHS_PATH.read_text(encoding="utf-8")),
                str(DEFAULT_PLAYER_BONE_LENGTHS_PATH.relative_to(Path(__file__).resolve().parents[2])),
            )
        return {}, "default_anthropometric_fallback_no_player_bone_lengths_file"
    if isinstance(canonical_bone_lengths, (str, Path)):
        path = Path(canonical_bone_lengths)
        return json.loads(path.read_text(encoding="utf-8")), str(path)
    return dict(canonical_bone_lengths), "argument"


def _sam3d_wrist_lock_target_length(
    canonical_payload: Mapping[str, Any],
    *,
    player_id: str,
    bone_name: str,
) -> dict[str, Any]:
    exact = _canonical_bone_median(canonical_payload, player_id=player_id, bone_name=bone_name)
    if exact is not None:
        return {
            "length_m": exact,
            "source": "player_bone_lengths_exact_lower_arm",
        }
    leg_scale = _canonical_leg_scale_m(canonical_payload, player_id=player_id)
    if leg_scale is not None:
        return {
            "length_m": float(_DEFAULT_LEG_DERIVED_RATIOS["lower_arm"]) * leg_scale,
            "source": "player_bone_lengths_leg_derived_anthropometric_fallback",
        }
    default_height_m = 1.72
    default_leg_scale_m = (0.245 + 0.246) * default_height_m
    return {
        "length_m": float(_DEFAULT_LEG_DERIVED_RATIOS["lower_arm"]) * default_leg_scale_m,
        "source": "default_anthropometric_fallback",
    }


def _canonical_bone_median(
    canonical_payload: Mapping[str, Any],
    *,
    player_id: str,
    bone_name: str,
) -> float | None:
    players = canonical_payload.get("players")
    if not isinstance(players, Mapping):
        return None
    player = players.get(str(player_id))
    if not isinstance(player, Mapping):
        return None
    bones = player.get("bones")
    if not isinstance(bones, Mapping):
        return None
    entry = bones.get(bone_name)
    if not isinstance(entry, Mapping):
        return None
    try:
        value = float(entry["median_m"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def _canonical_leg_scale_m(
    canonical_payload: Mapping[str, Any],
    *,
    player_id: str,
) -> float | None:
    lengths = {
        bone_name: _canonical_bone_median(canonical_payload, player_id=player_id, bone_name=bone_name)
        for bone_name in LEG_BONE_JOINT_PAIRS
    }
    if any(value is None for value in lengths.values()):
        return None
    thigh = (float(lengths["left_upper_leg"]) + float(lengths["right_upper_leg"])) / 2.0
    shin = (float(lengths["left_lower_leg"]) + float(lengths["right_lower_leg"])) / 2.0
    leg_scale = thigh + shin
    return leg_scale if math.isfinite(leg_scale) and leg_scale > 0.0 else None


def _finite_joint3(value: Any) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    try:
        joint = [float(value[axis]) for axis in range(3)]
    except (TypeError, ValueError):
        return None
    return joint if all(math.isfinite(axis) for axis in joint) else None


def _joint_confidence_safe(conf: Any, joint_idx: int) -> float:
    try:
        return _joint_confidence(conf, joint_idx)
    except (TypeError, ValueError):
        return 0.0


def _build_sam3d_stature_check(
    skeleton3d: Mapping[str, Any],
    *,
    stature_band_m: tuple[float, float] = DEFAULT_STATURE_BAND_M,
) -> dict[str, Any]:
    joint_names = _string_joint_names(skeleton3d.get("joint_names"))
    index_by_name = _semantic_index_by_name(joint_names)
    ankle_indices = [
        idx
        for name in ("left_ankle", "right_ankle")
        if (idx := index_by_name.get(name)) is not None
    ]
    low, high = float(stature_band_m[0]), float(stature_band_m[1])
    by_player: dict[str, list[float]] = {}
    for player in skeleton3d.get("players", []):
        if not isinstance(player, Mapping):
            continue
        player_id = str(player.get("id", "unknown"))
        for frame in player.get("frames", []):
            if not isinstance(frame, Mapping):
                continue
            joints = _joint_vectors(frame)
            if not joints:
                continue
            z_values = [float(joint[2]) for joint in joints if len(joint) >= 3 and math.isfinite(float(joint[2]))]
            if not z_values:
                continue
            foot_z_values = [
                float(joints[idx][2])
                for idx in ankle_indices
                if idx < len(joints) and math.isfinite(float(joints[idx][2]))
            ]
            if foot_z_values and min(abs(value) for value in foot_z_values) > 0.35:
                continue
            by_player.setdefault(player_id, []).append(max(z_values) - min(z_values))
    players: dict[str, dict[str, Any]] = {}
    medians: list[float] = []
    for player_id, values in sorted(by_player.items()):
        med = float(median(values)) if values else None
        if med is not None:
            medians.append(med)
        players[player_id] = {
            "standing_frame_count": len(values),
            "median_standing_z_span_m": med,
            "plausible_band_m": [low, high],
            "scale_suspect": bool(med is None or med < low or med > high),
        }
    overall = float(median(medians)) if medians else None
    return {
        "source": SAM3D_BODY_JOINT_SOURCE,
        "plausible_band_m": [low, high],
        "median_standing_z_span_m": overall,
        "scale_suspect": bool(overall is None or overall < low or overall > high or any(item["scale_suspect"] for item in players.values())),
        "players": players,
    }


def _provenance_with_temporal_refine(
    provenance: Any,
    *,
    mincutoff: float,
    beta: float,
    core_mincutoff: float,
    core_beta: float,
    wrist_mincutoff: float,
    wrist_beta: float,
    foot_mincutoff: float,
    foot_beta: float,
    low_confidence_threshold: float,
    motionbert_window_max_frames: int,
    motionbert_metrics: Mapping[str, Any],
    smoothing_metrics: Mapping[str, Any],
    grounding_metrics: Mapping[str, int],
    core_clamp_engagement_by_player: Mapping[str, Mapping[str, Any]],
    world_grounding_applied: bool,
    smoothing_max_displacement_m: float | None,
) -> dict[str, Any]:
    output = dict(provenance) if isinstance(provenance, Mapping) else {}
    motionbert_status = "applied" if int(motionbert_metrics["motionbert_frame_count"]) > 0 else "not_configured"
    smoothing_flag_counts = {
        str(flag): int(count)
        for flag, count in sorted(dict(smoothing_metrics.get("flag_counts", {})).items())
    }
    one_euro_record = {
        "mincutoff": mincutoff,
        "beta": beta,
        "core_body_mincutoff": core_mincutoff,
        "core_body_beta": core_beta,
        "wrist_mincutoff": wrist_mincutoff,
        "wrist_beta": wrist_beta,
        "foot_mincutoff": foot_mincutoff,
        "foot_beta": foot_beta,
        "applied_joint_groups": ["core_body", "feet", "hands", "wrists"],
        "filtered_joint_count": int(smoothing_metrics.get("filtered_joint_count", 0)),
    }
    if smoothing_max_displacement_m is not None:
        one_euro_record["smoothing_max_displacement_m"] = float(smoothing_max_displacement_m)
    physical_plausibility = {
        "core_body_speed_flag_mps": CORE_BODY_SPEED_FLAG_MPS,
        "core_body_speed_sustained_frame_count": CORE_SPEED_SUSTAINED_FRAME_COUNT,
        "core_body_speed_clamp_engagement_flag": "final_core_speed_clamped",
        "single_frame_jump_flag_m": SINGLE_FRAME_JUMP_FLAG_M,
        "flagged_joints_are_damped": True,
        "core_body_speed_clamp_engagement_by_player": {
            str(player_id): dict(summary)
            for player_id, summary in sorted(core_clamp_engagement_by_player.items())
        },
        "core_body_speed_clamp_engagement_overall": _core_body_speed_clamp_engagement_overall(
            core_clamp_engagement_by_player
        ),
    }
    if smoothing_max_displacement_m is not None:
        physical_plausibility["smoothing_max_displacement_m"] = float(smoothing_max_displacement_m)
    output["temporal_refine"] = {
        "motionbert": motionbert_status,
        "motionbert_model_id": str(motionbert_metrics.get("motionbert_model_id", "")),
        "motionbert_window_max_frames": motionbert_window_max_frames,
        "motionbert_window_count": int(motionbert_metrics["motionbert_window_count"]),
        "motionbert_frame_count": int(motionbert_metrics["motionbert_frame_count"]),
        "motionbert_body_format": "h36m_17",
        "one_euro": one_euro_record,
        "physical_plausibility": physical_plausibility,
        "low_confidence_threshold": low_confidence_threshold,
        "smoothing_flags": smoothing_flag_counts,
        "bone_length_constraint": "body17_median_per_player",
    }
    output["world_grounding"] = {
        "applied": bool(world_grounding_applied),
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


def _apply_final_core_jitter_guard(
    frames: Sequence[Mapping[str, Any]],
    joint_names: Sequence[str],
    *,
    fps: float,
    max_displacement_m: float | None = None,
    raw_reference_frames: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    guarded: list[dict[str, Any]] = []
    metrics = _empty_smoothing_metrics()
    core_indexes = [
        idx
        for idx, name in enumerate(joint_names)
        if _joint_smoothing_group(name, joint_names) == "core_body"
    ]
    previous_output_by_joint: dict[int, list[float]] = {}
    previous_raw_reference_by_joint: dict[int, list[float]] = {}
    high_core_speed_streaks: dict[int, int] = {}
    previous_t: float | None = None
    for frame_pos, frame in enumerate(frames):
        output = copy.deepcopy(dict(frame))
        joints = _joint_vectors(output)
        raw_reference_joints = (
            _joint_vectors(raw_reference_frames[frame_pos])
            if raw_reference_frames is not None and frame_pos < len(raw_reference_frames)
            else []
        )
        flags = [_parse_joint_flags(value) for value in output.get("smoothing_flag", [])]
        if len(flags) < len(joints):
            flags.extend([] for _idx in range(len(joints) - len(flags)))
        t = float(output.get("t", 0.0))
        dt = (t - previous_t) if previous_t is not None else (1.0 / fps)
        previous_t = t
        for joint_idx in core_indexes:
            if joint_idx >= len(joints):
                continue
            previous = previous_output_by_joint.get(joint_idx)
            if previous is None or dt <= 0.0:
                continue
            displacement_m = math.dist(previous, joints[joint_idx])
            speed_mps = displacement_m / dt
            raw_current = raw_reference_joints[joint_idx] if joint_idx < len(raw_reference_joints) else None
            raw_previous = previous_raw_reference_by_joint.get(joint_idx)
            raw_speed_mps: float | None = None
            raw_displacement_m: float | None = None
            if raw_current is not None and raw_previous is not None:
                raw_displacement_m = math.dist(raw_previous, raw_current)
                raw_speed_mps = raw_displacement_m / dt
            clamp_flags: list[str] = []
            single_frame_jump_is_anomaly = displacement_m > SINGLE_FRAME_JUMP_FLAG_M
            if raw_displacement_m is not None and raw_displacement_m > SINGLE_FRAME_JUMP_FLAG_M:
                single_frame_jump_is_anomaly = False
            if single_frame_jump_is_anomaly:
                clamp_flags.append("single_frame_jump_clamped")
            core_speed_is_anomaly = speed_mps > CORE_BODY_SPEED_FLAG_MPS
            if raw_speed_mps is not None and raw_speed_mps > CORE_BODY_SPEED_FLAG_MPS:
                core_speed_is_anomaly = False
            if core_speed_is_anomaly:
                high_core_speed_streaks[joint_idx] = high_core_speed_streaks.get(joint_idx, 0) + 1
            else:
                high_core_speed_streaks[joint_idx] = 0
            if high_core_speed_streaks.get(joint_idx, 0) >= CORE_SPEED_SUSTAINED_FRAME_COUNT:
                clamp_flags.append("core_speed_clamped")
            if not clamp_flags:
                continue
            joints[joint_idx] = _clamp_joint_step(
                previous,
                joints[joint_idx],
                max_step_m=_max_damped_step_m(group="core_body", dt=dt),
            )
            if max_displacement_m is not None and raw_current is not None:
                capped, was_capped = _cap_joint_displacement(
                    raw_current,
                    joints[joint_idx],
                    max_displacement_m=max_displacement_m,
                )
                if was_capped:
                    joints[joint_idx] = capped
                    _add_joint_flag(flags, joint_idx, "smoothing_displacement_capped")
                    metrics["flag_counts"]["smoothing_displacement_capped"] += 1
            for flag in clamp_flags:
                _add_joint_flag(flags, joint_idx, flag)
                metrics["flag_counts"][flag] += 1
                if flag == "core_speed_clamped":
                    _add_joint_flag(flags, joint_idx, "final_core_speed_clamped")
                    metrics["flag_counts"]["final_core_speed_clamped"] += 1
        output["joints_world"] = joints
        output["smoothing_flag"] = [_format_joint_flags(flag_values) for flag_values in flags]
        guarded.append(output)
        previous_output_by_joint = {
            idx: list(joints[idx])
            for idx in range(min(len(joints), len(joint_names)))
        }
        previous_raw_reference_by_joint = {
            idx: list(raw_reference_joints[idx])
            for idx in range(min(len(raw_reference_joints), len(joint_names)))
        }
    return guarded, metrics


def _core_body_speed_clamp_engagement(
    frames: Sequence[Mapping[str, Any]],
    joint_names: Sequence[str],
) -> dict[str, Any]:
    core_indexes = [
        idx
        for idx, name in enumerate(joint_names)
        if _joint_smoothing_group(name, joint_names) == "core_body"
    ]
    frame_count = 0
    clamped_frame_count = 0
    clamped_joint_sample_count = 0
    total_core_joint_sample_count = 0
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        frame_count += 1
        flags = frame.get("smoothing_flag")
        frame_clamped = False
        for joint_idx in core_indexes:
            total_core_joint_sample_count += 1
            if not isinstance(flags, list) or joint_idx >= len(flags):
                continue
            if "final_core_speed_clamped" in _parse_joint_flags(flags[joint_idx]):
                frame_clamped = True
                clamped_joint_sample_count += 1
        if frame_clamped:
            clamped_frame_count += 1
    return {
        "frame_count": frame_count,
        "clamped_frame_count": clamped_frame_count,
        "clamp_engagement_fraction": round(clamped_frame_count / frame_count, 6) if frame_count else 0.0,
        "core_joint_sample_count": total_core_joint_sample_count,
        "clamped_core_joint_sample_count": clamped_joint_sample_count,
        "core_joint_clamp_fraction": round(clamped_joint_sample_count / total_core_joint_sample_count, 6)
        if total_core_joint_sample_count
        else 0.0,
    }


def _core_body_speed_clamp_engagement_overall(
    by_player: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    frame_count = sum(int(summary.get("frame_count", 0)) for summary in by_player.values())
    clamped_frame_count = sum(int(summary.get("clamped_frame_count", 0)) for summary in by_player.values())
    core_joint_sample_count = sum(int(summary.get("core_joint_sample_count", 0)) for summary in by_player.values())
    clamped_core_joint_sample_count = sum(
        int(summary.get("clamped_core_joint_sample_count", 0))
        for summary in by_player.values()
    )
    return {
        "frame_count": frame_count,
        "clamped_frame_count": clamped_frame_count,
        "clamp_engagement_fraction": round(clamped_frame_count / frame_count, 6) if frame_count else 0.0,
        "core_joint_sample_count": core_joint_sample_count,
        "clamped_core_joint_sample_count": clamped_core_joint_sample_count,
        "core_joint_clamp_fraction": round(clamped_core_joint_sample_count / core_joint_sample_count, 6)
        if core_joint_sample_count
        else 0.0,
    }


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


def _empty_smoothing_metrics(*, filtered_joint_count: int = 0) -> dict[str, Any]:
    return {
        "filtered_joint_count": int(filtered_joint_count),
        "flag_counts": Counter(),
    }


def _add_smoothing_metrics(total: dict[str, Any], increment: Mapping[str, Any]) -> None:
    total["filtered_joint_count"] = max(int(total.get("filtered_joint_count", 0)), int(increment.get("filtered_joint_count", 0)))
    total_counts = total.setdefault("flag_counts", Counter())
    for flag, count in dict(increment.get("flag_counts", {})).items():
        total_counts[str(flag)] += int(count)


def _is_sam3d_skeleton_payload(payload: Mapping[str, Any]) -> bool:
    if payload.get("artifact_type") != "racketsport_skeleton3d":
        return False
    joint_names = _string_joint_names(payload.get("joint_names"))
    if len(joint_names) != SAM3D_BODY_MHR70_SEMANTIC_MAP.source_joint_count:
        return False
    accepted_sources = {SAM3D_BODY_JOINT_SOURCE, "sam3dbody_world_joints"}
    for explicit_key in ("source_model", "model"):
        explicit_value = str(payload.get(explicit_key, ""))
        if explicit_value and explicit_value not in accepted_sources:
            return False
    provenance = payload.get("provenance")
    source_values = {
        str(payload.get("source_model", "")),
        str(payload.get("model", "")),
    }
    if isinstance(provenance, Mapping):
        source_values.update(
            str(provenance.get(key, ""))
            for key in ("source", "model_family", "skeleton_source")
        )
    return bool(source_values & accepted_sources)


def _string_joint_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _semantic_index_by_name(joint_names: Sequence[str]) -> dict[str, int]:
    direct = {str(name): idx for idx, name in enumerate(joint_names)}
    if _looks_like_sam3d_mhr70_joint_names(joint_names):
        for name, idx in SAM3D_BODY_MHR70_SEMANTIC_MAP.joints.items():
            direct.setdefault(name, idx)
    return direct


def _looks_like_sam3d_mhr70_joint_names(joint_names: Sequence[str]) -> bool:
    if len(joint_names) != SAM3D_BODY_MHR70_SEMANTIC_MAP.source_joint_count:
        return False
    return all(str(name) == f"sam3dbody_joint_{idx:03d}" for idx, name in enumerate(joint_names))


def _canonical_joint_name(name: str, joint_names: Sequence[str]) -> str:
    if not _looks_like_sam3d_mhr70_joint_names(joint_names):
        return name
    try:
        idx = list(joint_names).index(name)
    except ValueError:
        return name
    for semantic_name, semantic_idx in SAM3D_BODY_MHR70_SEMANTIC_MAP.joints.items():
        if semantic_idx == idx:
            return semantic_name
    # SAM3D_BODY_MHR70_SEMANTIC_MAP only covers the 12 joints needed for wrist/limb/ankle
    # gates (see refine_sam3d_skeleton3d docstring). Foot joints outside that set (heel,
    # big/small toe tips) fell through to the generic sam3dbody_joint_### name here, which
    # made _foot_joint_names/_joint_smoothing_group silently bucket them as "core_body"
    # instead of "feet" -- runs/sam3d_foot_wander_20260703T1024Z/REPORT.md measured this as
    # the dominant source of ANKLE/HEEL/TOE stance-phase slide. MHR70_JOINT_NAMES is the
    # full canonical 70-joint layout that _looks_like_sam3d_mhr70_joint_names already
    # confirmed this payload matches positionally, so use it as a full-coverage fallback.
    if idx < len(MHR70_JOINT_NAMES):
        return MHR70_JOINT_NAMES[idx]
    return name


def _joint_smoothing_group(name: str, joint_names: Sequence[str]) -> str:
    canonical = _canonical_joint_name(name, joint_names)
    if canonical in {"left_wrist", "right_wrist"}:
        return "wrists"
    if canonical.startswith("left_hand_") or canonical.startswith("right_hand_"):
        return "hands"
    # Check the foot/toe/heel pattern directly against the canonical name rather than
    # membership in _foot_joint_names(joint_names) (which returns RAW joint_names entries,
    # not canonical ones): for SAM-3D generic sam3dbody_joint_### payloads, `canonical` is
    # already resolved (e.g. "left_heel"), so comparing it against a set of raw
    # "sam3dbody_joint_017"-style strings could never match, which silently bucketed heel
    # and toe-tip joints as "core_body" instead of "feet". See
    # runs/sam3d_foot_wander_20260703T1024Z/REPORT.md for the measured impact.
    if canonical in {"left_ankle", "right_ankle"} or "toe" in canonical or canonical.endswith("_heel"):
        return "feet"
    return "core_body"


def _add_joint_flag(frame_flags: list[list[str]], joint_idx: int, flag: str) -> None:
    if joint_idx >= len(frame_flags):
        return
    if flag not in frame_flags[joint_idx]:
        frame_flags[joint_idx].append(flag)


def _format_joint_flags(flags: Sequence[str]) -> str:
    if not flags:
        return NO_SMOOTHING_FLAG
    return "|".join(sorted(flags))


def _parse_joint_flags(value: Any) -> list[str]:
    if not isinstance(value, str) or value == NO_SMOOTHING_FLAG:
        return []
    return [part for part in value.split("|") if part]


def _clamp_joint_step(previous: Sequence[float], current: Sequence[float], *, max_step_m: float) -> list[float]:
    displacement = math.dist(previous, current)
    if displacement <= max_step_m or displacement <= 1e-12:
        return [float(value) for value in current]
    scale = max_step_m / displacement
    return [
        float(previous[axis]) + (float(current[axis]) - float(previous[axis])) * scale
        for axis in range(3)
    ]


def _max_damped_step_m(*, group: str, dt: float) -> float:
    if group == "core_body":
        return max(0.0, CORE_BODY_SPEED_FLAG_MPS * dt)
    return SINGLE_FRAME_JUMP_FLAG_M


def _foot_joint_names(joint_names: Sequence[str]) -> set[str]:
    return {
        name
        for name in joint_names
        if "toe" in _canonical_joint_name(name, joint_names) or _canonical_joint_name(name, joint_names).endswith("_heel")
    }


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


def compute_pose_jitter_audit(skeleton3d: Mapping[str, Any], *, source_path: str | Path | None = None) -> dict[str, Any]:
    """Measure frame-to-frame world-joint displacement by joint group."""

    joint_names = skeleton3d.get("joint_names")
    if not isinstance(joint_names, list) or not all(isinstance(name, str) for name in joint_names):
        raise ValueError("skeleton3d joint_names must be a list of strings")
    players = skeleton3d.get("players")
    if not isinstance(players, list):
        raise ValueError("skeleton3d players must be a list")
    fps = float(skeleton3d.get("fps") or _infer_fps(players))
    groups_by_idx = {
        idx: _joint_smoothing_group(name, joint_names)
        for idx, name in enumerate(joint_names)
    }
    group_samples: dict[str, list[float]] = {"all": [], "core_body": [], "feet": [], "hands": [], "wrists": []}
    per_joint_samples: dict[str, list[float]] = {name: [] for name in joint_names}
    for player in players:
        if not isinstance(player, Mapping):
            continue
        frames = player.get("frames")
        if not isinstance(frames, list):
            continue
        sorted_frames = sorted(frames, key=lambda frame: (float(frame.get("t", 0.0)), int(frame.get("frame_idx", 0))))
        for previous, current in zip(sorted_frames, sorted_frames[1:]):
            previous_joints = _joint_vectors(previous)
            current_joints = _joint_vectors(current)
            joint_count = min(len(previous_joints), len(current_joints), len(joint_names))
            for joint_idx in range(joint_count):
                displacement = math.dist(previous_joints[joint_idx], current_joints[joint_idx])
                joint_name = joint_names[joint_idx]
                group = groups_by_idx[joint_idx]
                group_samples["all"].append(displacement)
                group_samples[group].append(displacement)
                per_joint_samples[joint_name].append(displacement)

    group_stats = {
        group: _displacement_stats(samples)
        for group, samples in group_samples.items()
    }
    per_joint_stats = {
        joint_name: {
            **_displacement_stats(samples),
            "group": groups_by_idx.get(idx, "core_body"),
        }
        for idx, (joint_name, samples) in enumerate(per_joint_samples.items())
    }
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_pose_jitter_audit",
        "source_path": str(source_path or ""),
        "fps": fps,
        "joint_count": len(joint_names),
        "player_count": len(players),
        "thresholds": {
            "target_core_body_p90_frame_displacement_m": 0.3,
            "core_body_speed_flag_mps": CORE_BODY_SPEED_FLAG_MPS,
            "single_frame_jump_flag_m": SINGLE_FRAME_JUMP_FLAG_M,
        },
        "joint_groups": {
            "core_body": [name for idx, name in enumerate(joint_names) if groups_by_idx[idx] == "core_body"],
            "feet": [name for idx, name in enumerate(joint_names) if groups_by_idx[idx] == "feet"],
            "hands": [name for idx, name in enumerate(joint_names) if groups_by_idx[idx] == "hands"],
            "wrists": [name for idx, name in enumerate(joint_names) if groups_by_idx[idx] == "wrists"],
        },
        "group_stats": group_stats,
        "per_joint": per_joint_stats,
    }


def compare_wrist_peak_timing(
    before_skeleton3d: Mapping[str, Any],
    after_skeleton3d: Mapping[str, Any],
    *,
    top_k: int = 5,
    max_allowed_delta_frames: int = 1,
    min_peak_speed_mps: float = 4.0,
) -> dict[str, Any]:
    """Compare wrist velocity peak frame indexes before and after smoothing."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if min_peak_speed_mps < 0.0:
        raise ValueError("min_peak_speed_mps must be non-negative")
    before_names = before_skeleton3d.get("joint_names")
    after_names = after_skeleton3d.get("joint_names")
    if not isinstance(before_names, list) or not isinstance(after_names, list):
        raise ValueError("both skeleton payloads must include joint_names")
    before_players = _players_by_id(before_skeleton3d.get("players"))
    after_players = _players_by_id(after_skeleton3d.get("players"))
    before_fps = float(before_skeleton3d.get("fps") or _infer_fps(list(before_players.values())))
    after_fps = float(after_skeleton3d.get("fps") or _infer_fps(list(after_players.values())))
    comparisons: list[dict[str, Any]] = []
    skipped_clamped_before_peak_count = 0
    for joint_name in ("left_wrist", "right_wrist"):
        before_idx = _joint_index_for_semantic(before_names, joint_name)
        after_idx = _joint_index_for_semantic(after_names, joint_name)
        if before_idx is None or after_idx is None:
            continue
        for player_id, before_player in sorted(before_players.items()):
            after_player = after_players.get(player_id)
            if after_player is None:
                continue
            before_peaks = _top_wrist_speed_peaks(
                before_player.get("frames"),
                before_idx,
                fps=before_fps,
                top_k=top_k,
                min_peak_speed_mps=min_peak_speed_mps,
            )
            after_peaks = _top_wrist_speed_peaks(
                after_player.get("frames"),
                after_idx,
                fps=after_fps,
                top_k=100000,
                min_peak_speed_mps=min_peak_speed_mps,
            )
            used_after: set[int] = set()
            for peak in before_peaks:
                if _nearby_frame_has_flag(
                    after_player.get("frames"),
                    int(peak["frame"]),
                    after_idx,
                    "single_frame_jump_clamped",
                    radius=1,
                ):
                    skipped_clamped_before_peak_count += 1
                    continue
                if not after_peaks:
                    comparisons.append(
                        {
                            "player_id": player_id,
                            "joint_name": joint_name,
                            "before_frame": peak["frame"],
                            "after_frame": None,
                            "delta_frames": None,
                            "before_speed_mps": peak["speed_mps"],
                            "after_speed_mps": None,
                            "status": "missing_after_peak",
                        }
                    )
                    continue
                after_pos, after_peak = min(
                    (
                        (pos, candidate)
                        for pos, candidate in enumerate(after_peaks)
                        if pos not in used_after
                    ),
                    key=lambda item: abs(int(item[1]["frame"]) - int(peak["frame"])),
                    default=(0, after_peaks[0]),
                )
                used_after.add(after_pos)
                delta = int(after_peak["frame"]) - int(peak["frame"])
                comparisons.append(
                    {
                        "player_id": player_id,
                        "joint_name": joint_name,
                        "before_frame": int(peak["frame"]),
                        "after_frame": int(after_peak["frame"]),
                        "delta_frames": delta,
                        "abs_delta_frames": abs(delta),
                        "before_speed_mps": round(float(peak["speed_mps"]), 6),
                        "after_speed_mps": round(float(after_peak["speed_mps"]), 6),
                        "status": "matched",
                    }
                )
    deltas = [
        int(comparison["abs_delta_frames"])
        for comparison in comparisons
        if comparison.get("abs_delta_frames") is not None
    ]
    max_delta = max(deltas) if deltas else None
    status = "pass" if max_delta is not None and max_delta <= max_allowed_delta_frames else "fail"
    if not comparisons:
        status = "blocked"
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_wrist_peak_timing_comparison",
        "status": status,
        "max_allowed_delta_frames": max_allowed_delta_frames,
        "min_peak_speed_mps": min_peak_speed_mps,
        "ignored_single_frame_jump_threshold_m": SINGLE_FRAME_JUMP_FLAG_M,
        "max_abs_delta_frames": max_delta,
        "comparison_count": len(comparisons),
        "skipped_clamped_before_peak_count": skipped_clamped_before_peak_count,
        "comparisons": comparisons,
    }


def compare_wrist_direction_peak_timing(
    before_skeleton3d: Mapping[str, Any],
    after_skeleton3d: Mapping[str, Any],
    *,
    top_k: int = 5,
    max_allowed_delta_frames: int = 0,
    min_peak_direction_speed: float = 0.0,
) -> dict[str, Any]:
    """Compare elbow-relative wrist direction-change peak frame indexes."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if max_allowed_delta_frames < 0:
        raise ValueError("max_allowed_delta_frames must be non-negative")
    if min_peak_direction_speed < 0.0:
        raise ValueError("min_peak_direction_speed must be non-negative")
    before_names = before_skeleton3d.get("joint_names")
    after_names = after_skeleton3d.get("joint_names")
    if not isinstance(before_names, list) or not isinstance(after_names, list):
        raise ValueError("both skeleton payloads must include joint_names")
    before_players = _players_by_id(before_skeleton3d.get("players"))
    after_players = _players_by_id(after_skeleton3d.get("players"))
    before_fps = float(before_skeleton3d.get("fps") or _infer_fps(list(before_players.values())))
    after_fps = float(after_skeleton3d.get("fps") or _infer_fps(list(after_players.values())))
    comparisons: list[dict[str, Any]] = []
    for side in ("left", "right"):
        wrist_name = f"{side}_wrist"
        elbow_name = f"{side}_elbow"
        before_wrist_idx = _joint_index_for_semantic(before_names, wrist_name)
        before_elbow_idx = _joint_index_for_semantic(before_names, elbow_name)
        after_wrist_idx = _joint_index_for_semantic(after_names, wrist_name)
        after_elbow_idx = _joint_index_for_semantic(after_names, elbow_name)
        if (
            before_wrist_idx is None
            or before_elbow_idx is None
            or after_wrist_idx is None
            or after_elbow_idx is None
        ):
            continue
        for player_id, before_player in sorted(before_players.items()):
            after_player = after_players.get(player_id)
            if after_player is None:
                continue
            before_peaks = _top_wrist_direction_peaks(
                before_player.get("frames"),
                elbow_idx=before_elbow_idx,
                wrist_idx=before_wrist_idx,
                fps=before_fps,
                top_k=top_k,
                min_peak_direction_speed=min_peak_direction_speed,
            )
            after_peaks = _top_wrist_direction_peaks(
                after_player.get("frames"),
                elbow_idx=after_elbow_idx,
                wrist_idx=after_wrist_idx,
                fps=after_fps,
                top_k=100000,
                min_peak_direction_speed=min_peak_direction_speed,
            )
            used_after: set[int] = set()
            for peak in before_peaks:
                if not after_peaks:
                    comparisons.append(
                        {
                            "player_id": player_id,
                            "joint_name": wrist_name,
                            "before_frame": peak["frame"],
                            "after_frame": None,
                            "delta_frames": None,
                            "before_direction_speed": peak["direction_speed"],
                            "after_direction_speed": None,
                            "status": "missing_after_peak",
                        }
                    )
                    continue
                after_pos, after_peak = min(
                    (
                        (pos, candidate)
                        for pos, candidate in enumerate(after_peaks)
                        if pos not in used_after
                    ),
                    key=lambda item: abs(int(item[1]["frame"]) - int(peak["frame"])),
                    default=(0, after_peaks[0]),
                )
                used_after.add(after_pos)
                delta = int(after_peak["frame"]) - int(peak["frame"])
                comparisons.append(
                    {
                        "player_id": player_id,
                        "joint_name": wrist_name,
                        "before_frame": int(peak["frame"]),
                        "after_frame": int(after_peak["frame"]),
                        "delta_frames": delta,
                        "abs_delta_frames": abs(delta),
                        "before_direction_speed": round(float(peak["direction_speed"]), 6),
                        "after_direction_speed": round(float(after_peak["direction_speed"]), 6),
                        "status": "matched",
                    }
                )
    deltas = [
        int(comparison["abs_delta_frames"])
        for comparison in comparisons
        if comparison.get("abs_delta_frames") is not None
    ]
    max_delta = max(deltas) if deltas else None
    status = "pass" if max_delta is not None and max_delta <= max_allowed_delta_frames else "fail"
    if not comparisons:
        status = "blocked"
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_wrist_direction_peak_timing_comparison",
        "measurement": "elbow_relative_unit_direction_change",
        "status": status,
        "max_allowed_delta_frames": max_allowed_delta_frames,
        "min_peak_direction_speed": min_peak_direction_speed,
        "max_abs_delta_frames": max_delta,
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
    }


def _top_wrist_direction_peaks(
    frames: Any,
    *,
    elbow_idx: int,
    wrist_idx: int,
    fps: float,
    top_k: int,
    min_peak_direction_speed: float,
) -> list[dict[str, float | int]]:
    if not isinstance(frames, list):
        return []
    sorted_frames = sorted(frames, key=lambda frame: (float(frame.get("t", 0.0)), int(frame.get("frame_idx", 0))))
    speeds: list[dict[str, float | int]] = []
    for previous_pos, (previous, current) in enumerate(zip(sorted_frames, sorted_frames[1:])):
        previous_direction = _elbow_relative_wrist_unit(previous, elbow_idx=elbow_idx, wrist_idx=wrist_idx)
        current_direction = _elbow_relative_wrist_unit(current, elbow_idx=elbow_idx, wrist_idx=wrist_idx)
        if previous_direction is None or current_direction is None:
            continue
        previous_t = float(previous.get("t", previous_pos / fps))
        current_t = float(current.get("t", (previous_pos + 1) / fps))
        dt = current_t - previous_t
        if dt <= 0.0:
            frame_delta = int(current.get("frame_idx", previous_pos + 1)) - int(previous.get("frame_idx", previous_pos))
            dt = max(1, frame_delta) / fps
        direction_speed = math.dist(previous_direction, current_direction) / dt
        if direction_speed < min_peak_direction_speed:
            continue
        speeds.append(
            {
                "frame": int(current.get("frame_idx", previous_pos + 1)),
                "direction_speed": direction_speed,
            }
        )
    selected: list[dict[str, float | int]] = []
    for candidate in sorted(speeds, key=lambda item: float(item["direction_speed"]), reverse=True):
        if any(abs(int(candidate["frame"]) - int(kept["frame"])) <= 1 for kept in selected):
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return sorted(selected, key=lambda item: int(item["frame"]))


def _elbow_relative_wrist_unit(
    frame: Mapping[str, Any],
    *,
    elbow_idx: int,
    wrist_idx: int,
) -> list[float] | None:
    joints = frame.get("joints_world")
    if not isinstance(joints, list) or elbow_idx >= len(joints) or wrist_idx >= len(joints):
        return None
    elbow = _finite_joint3(joints[elbow_idx])
    wrist = _finite_joint3(joints[wrist_idx])
    if elbow is None or wrist is None:
        return None
    direction = [wrist[axis] - elbow[axis] for axis in range(3)]
    length = math.sqrt(sum(value * value for value in direction))
    if length <= 1e-12:
        return None
    return [value / length for value in direction]


def _displacement_stats(samples: Sequence[float]) -> dict[str, Any]:
    if not samples:
        return {
            "sample_count": 0,
            "p50_frame_displacement_m": None,
            "p90_frame_displacement_m": None,
            "max_frame_displacement_m": None,
        }
    return {
        "sample_count": len(samples),
        "p50_frame_displacement_m": round(_percentile(samples, 0.50), 6),
        "p90_frame_displacement_m": round(_percentile(samples, 0.90), 6),
        "max_frame_displacement_m": round(max(samples), 6),
    }


def _percentile(samples: Sequence[float], quantile: float) -> float:
    if not samples:
        raise ValueError("cannot compute percentile for empty samples")
    ordered = sorted(float(value) for value in samples)
    if len(ordered) == 1:
        return ordered[0]
    pos = quantile * (len(ordered) - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _players_by_id(players: Any) -> dict[int, Mapping[str, Any]]:
    if not isinstance(players, list):
        return {}
    output: dict[int, Mapping[str, Any]] = {}
    for player in players:
        if isinstance(player, Mapping):
            output[int(player.get("id", 0))] = player
    return output


def _joint_index_for_semantic(joint_names: Sequence[str], joint_name: str) -> int | None:
    index_by_name = _semantic_index_by_name(joint_names)
    return index_by_name.get(joint_name)


def _top_wrist_speed_peaks(
    frames: Any,
    joint_idx: int,
    *,
    fps: float,
    top_k: int,
    min_peak_speed_mps: float,
) -> list[dict[str, float | int]]:
    if not isinstance(frames, list):
        return []
    sorted_frames = sorted(frames, key=lambda frame: (float(frame.get("t", 0.0)), int(frame.get("frame_idx", 0))))
    speeds: list[dict[str, float | int]] = []
    for previous_pos, (previous, current) in enumerate(zip(sorted_frames, sorted_frames[1:])):
        previous_joints = _joint_vectors(previous)
        current_joints = _joint_vectors(current)
        if joint_idx >= len(previous_joints) or joint_idx >= len(current_joints):
            continue
        if _frame_joint_has_flag(previous, joint_idx, "single_frame_jump_clamped") or _frame_joint_has_flag(
            current, joint_idx, "single_frame_jump_clamped"
        ):
            continue
        previous_t = float(previous.get("t", previous_pos / fps))
        current_t = float(current.get("t", (previous_pos + 1) / fps))
        dt = current_t - previous_t
        if dt <= 0.0:
            frame_delta = int(current.get("frame_idx", previous_pos + 1)) - int(previous.get("frame_idx", previous_pos))
            dt = max(1, frame_delta) / fps
        displacement = math.dist(previous_joints[joint_idx], current_joints[joint_idx])
        if displacement > SINGLE_FRAME_JUMP_FLAG_M:
            continue
        if previous_pos > 0:
            pre_previous_joints = _joint_vectors(sorted_frames[previous_pos - 1])
            if (
                joint_idx < len(pre_previous_joints)
                and math.dist(pre_previous_joints[joint_idx], previous_joints[joint_idx]) > SINGLE_FRAME_JUMP_FLAG_M
            ):
                continue
        speed_mps = displacement / dt
        if speed_mps < min_peak_speed_mps:
            continue
        speeds.append(
            {
                "frame": int(current.get("frame_idx", previous_pos + 1)),
                "speed_mps": speed_mps,
            }
        )
    selected: list[dict[str, float | int]] = []
    for candidate in sorted(speeds, key=lambda item: float(item["speed_mps"]), reverse=True):
        if any(abs(int(candidate["frame"]) - int(kept["frame"])) <= 1 for kept in selected):
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return sorted(selected, key=lambda item: int(item["frame"]))


def _frame_joint_has_flag(frame: Mapping[str, Any], joint_idx: int, flag: str) -> bool:
    values = frame.get("smoothing_flag")
    if not isinstance(values, list) or joint_idx >= len(values):
        return False
    return flag in _parse_joint_flags(values[joint_idx])


def _nearby_frame_has_flag(frames: Any, frame_idx: int, joint_idx: int, flag: str, *, radius: int) -> bool:
    if not isinstance(frames, list):
        return False
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        current_idx = frame.get("frame_idx")
        if (
            isinstance(current_idx, int)
            and abs(current_idx - frame_idx) <= radius
            and _frame_joint_has_flag(frame, joint_idx, flag)
        ):
            return True
    return False


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


__all__ = [
    "OneEuroFilter",
    "apply_sam3d_wrist_bone_lock",
    "compare_wrist_direction_peak_timing",
    "compare_wrist_peak_timing",
    "compute_pose_jitter_audit",
    "refine_lane_a_skeleton3d",
    "refine_sam3d_skeleton3d",
]
