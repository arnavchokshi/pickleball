"""World-grounded SMPL reconstruction helpers.

This module intentionally stops at deterministic CPU primitives. It does not
run Fast SAM-3D-Body or infer SMPL parameters.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from .footlock import (
    ContactHysteresis,
    FootContactObservation,
    FootKinematics,
    classify_contact,
    foot_lock_metrics,
    snap_stance_foot,
)
from .foot_contact import (
    ContactPhase,
    ContactThresholds,
    SkeletonFrame as ContactSkeletonFrame,
    detect_contact_phases,
    foot_contact_point,
    measure_contact_metrics,
    resolve_foot_joint_indices,
)
from .foot_pin import FootPinSettings, apply_foot_pin_to_payload
from .body_postchain import BodyPostChainConfig, RAW_GROUNDED_JOINTS_ARTIFACT
from .external_gt_body_prediction_schema import MHR70_JOINT_NAMES
from .pose_temporal import apply_sam3d_wrist_bone_lock, compare_wrist_peak_timing, refine_sam3d_skeleton3d
from .schemas import CourtCalibration
from .skeleton_upright import ROTATION_CONVENTION_OFFSET_ROW_TIMES_R, rotate_camera_offsets_row_times_R
from .visual_quality import estimate_integer_lag_frames


SCAFFOLD_NOTE = "cpu_worldhmr_primitives_no_sam3dbody_integration"
FLOOR_CONTACT_EPSILON_M = 0.035
LOW_GROUNDING_ANCHOR_HEIGHT_M = 0.08
FOOT_LOCK_SKATE_FREE_MAX_SLIDE_M = 0.003
FOOT_LOCK_RELATIVE_ENTER_SPEED_MPS = 5.0
FOOT_LOCK_RELATIVE_EXIT_SPEED_MPS = 6.0
R3_GROUNDING_ANCHOR_SOURCE = "placement_track_world_xy"
DEFAULT_GROUNDING_ANCHOR_SOURCE = "track_world_xy"
STANCE_AWARE_STANCE_ALPHA_XY = 1.0
STANCE_AWARE_TRANSITION_ALPHA_XY = 0.92
STANCE_AWARE_TRANSITION_FALLBACK_ALPHA_XY = 0.85
STANCE_AWARE_STANCE_RESIDUAL_RESET_M = 0.02
STANCE_AWARE_TRANSITION_RESIDUAL_RESET_M = 0.20
STANCE_AWARE_TRANSITION_BOUNDARY_FRAMES = 12
STANCE_AWARE_HIGH_COVARIANCE_TRACE_M2 = 0.25
STANCE_AWARE_PHASE_ANCHOR_DRIFT_SPLIT_M = 0.25
FOOT_LOCK_MAX_XY_CORRECTION_M = 0.02
REFINED_STANCE_FOOT_PIN_MAX_CORRECTION_M = 0.30
FOOT_LOCK_GATE_STREAM_MAX_BYTES = 20_000_000
FOOT_LOCK_GATE_STREAM_MAX_FRAME_ROWS = 20_000
CAMERA_MOTION_ARTIFACT = "camera_motion.json"
DEFAULT_SMOOTHING_GAP_CARRY_FRAMES = 8
DEFAULT_SMOOTHING_RESIDUAL_IDENTITY_RESET_M = 1.0
WORLD_JOINT_VISUAL_SMOOTHING_WEIGHTS = (0.30, 0.40, 0.30)
WORLD_JOINT_VISUAL_SMOOTHING_BONE_PAIRS = (
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_big_toe_tip"),
    ("left_ankle", "left_small_toe_tip"),
    ("left_ankle", "left_heel"),
    ("right_ankle", "right_big_toe_tip"),
    ("right_ankle", "right_small_toe_tip"),
    ("right_ankle", "right_heel"),
)
SAM3D_FOOT_KEYPOINT_INDICES = {
    "left_ankle": 13,
    "right_ankle": 14,
    "left_toe": 15,
    "right_toe": 16,
    "left_heel": 17,
    "right_heel": 20,
}
FOOT_LOCK_CONTACT_HYSTERESIS = ContactHysteresis(
    enter_height_m=FLOOR_CONTACT_EPSILON_M,
    exit_height_m=FLOOR_CONTACT_EPSILON_M,
    enter_speed_mps=FOOT_LOCK_RELATIVE_ENTER_SPEED_MPS,
    exit_speed_mps=FOOT_LOCK_RELATIVE_EXIT_SPEED_MPS,
    min_confidence=0.5,
)


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


@dataclass(frozen=True)
class _CameraMotionObservation:
    matrix: list[list[float]]


@dataclass(frozen=True)
class _CameraMotionContext:
    path: Path
    frames: dict[int, _CameraMotionObservation]
    artifact_frame_count: int
    artifact_compensated_frame_count: int


@dataclass(frozen=True)
class BodySkeletonAndMetrics:
    """Shared BODY joint-compute result before optional monolith assembly."""

    smpl_motion_view: dict[str, Any]
    skeleton3d: dict[str, Any]
    metrics: dict[str, Any]
    raw_grounded_joints: dict[str, Any] | None = None


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
    model: str = "sam3dbody_world_joints",
    sam3d_wrist_bone_lock: bool = True,
    stance_index: Mapping[tuple[int | str, int], Mapping[str, Any]] | None = None,
    grounding_anchor_source: str | None = None,
    camera_motion_path: str | Path | None = None,
    smoothing_gap_carry_frames: int = DEFAULT_SMOOTHING_GAP_CARRY_FRAMES,
    smoothing_residual_identity_reset_m: float = DEFAULT_SMOOTHING_RESIDUAL_IDENTITY_RESET_M,
    world_joint_visual_smoothing: bool = True,
    body_postchain: BodyPostChainConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build BODY contract artifacts from real Fast SAM-3D-Body outputs."""

    computed = compute_body_skeleton_and_metrics(
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
        body_postchain=body_postchain,
    )
    return assemble_body_monolith_payloads(
        computed.smpl_motion_view,
        computed.skeleton3d,
        computed.metrics,
    )


def compute_body_skeleton_and_metrics(
    samples: Sequence[Mapping[str, Any]],
    *,
    calibration: CourtCalibration,
    fps: float,
    smoothing_alpha: float = 0.65,
    max_root_speed_mps: float | None = None,
    max_track_anchor_smoothing_residual_m: float | None = None,
    model: str = "sam3dbody_world_joints",
    sam3d_wrist_bone_lock: bool = True,
    stance_index: Mapping[tuple[int | str, int], Mapping[str, Any]] | None = None,
    grounding_anchor_source: str | None = None,
    camera_motion_path: str | Path | None = None,
    smoothing_gap_carry_frames: int = DEFAULT_SMOOTHING_GAP_CARRY_FRAMES,
    smoothing_residual_identity_reset_m: float = DEFAULT_SMOOTHING_RESIDUAL_IDENTITY_RESET_M,
    world_joint_visual_smoothing: bool = True,
    body_postchain: BodyPostChainConfig | None = None,
) -> BodySkeletonAndMetrics:
    """Compute shared BODY skeleton, joint metrics, and light SMPL motion view."""

    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if model not in {"sam3dbody_world_joints", "sat_hmr_world_joints"}:
        raise ValueError("model must be a supported world-joint BODY model")
    if max_root_speed_mps is not None and max_root_speed_mps <= 0.0:
        raise ValueError("max_root_speed_mps must be positive when provided")
    if max_track_anchor_smoothing_residual_m is not None and max_track_anchor_smoothing_residual_m <= 0.0:
        raise ValueError("max_track_anchor_smoothing_residual_m must be positive when provided")
    if smoothing_gap_carry_frames < 0:
        raise ValueError("smoothing_gap_carry_frames must be non-negative")
    if smoothing_residual_identity_reset_m <= 0.0:
        raise ValueError("smoothing_residual_identity_reset_m must be positive")
    if not samples:
        raise ValueError("at least one Fast SAM-3D-Body sample is required")
    postchain = body_postchain or BodyPostChainConfig(
        wrist_lock=sam3d_wrist_bone_lock,
        world_joint_visual_smoothing=world_joint_visual_smoothing,
    )
    sam3d_wrist_bone_lock = bool(postchain.wrist_lock)
    world_joint_visual_smoothing = bool(postchain.world_joint_visual_smoothing)
    camera_motion, camera_motion_warnings = _load_camera_motion_context(camera_motion_path)
    camera_motion_seen: set[int] = set()
    camera_motion_used: set[int] = set()
    grounded = []
    for sample in sorted(samples, key=lambda item: (int(item["frame_idx"]), int(item["player_id"]))):
        frame_idx = int(sample["frame_idx"])
        motion_frame = camera_motion.frames.get(frame_idx) if camera_motion is not None else None
        if camera_motion is not None:
            camera_motion_seen.add(frame_idx)
            if motion_frame is not None:
                camera_motion_used.add(frame_idx)
        grounded.append(_ground_fast_sam_sample(sample, calibration=calibration, camera_motion=motion_frame))
    camera_motion_metrics, camera_motion_provenance = _camera_motion_artifact_summary(
        camera_motion,
        warnings=camera_motion_warnings,
        frames_used=len(camera_motion_used),
        frames_uncompensated=len(camera_motion_seen - camera_motion_used),
    )
    anchor_source = str(
        grounding_anchor_source
        or (R3_GROUNDING_ANCHOR_SOURCE if stance_index else DEFAULT_GROUNDING_ANCHOR_SOURCE)
    )
    stance_aware_grounding = anchor_source == R3_GROUNDING_ANCHOR_SOURCE and stance_index is not None
    raw_grounded_joints = (
        _build_raw_grounded_joints_sidecar(
            grounded,
            fps=fps,
            model=model,
            grounding_anchor_source=anchor_source,
            postchain=postchain,
        )
        if postchain.is_raw
        else None
    )
    if not postchain.temporal_smoothing:
        smoothed, smoothing_metrics = _bypass_temporal_smoothing(grounded, stance_aware_grounding=stance_aware_grounding)
    elif stance_aware_grounding:
        smoothed, smoothing_metrics = _smooth_grounded_frames_stance_aware(
            grounded,
            stance_index=stance_index or {},
            fps=fps,
            max_root_speed_mps=max_root_speed_mps,
            max_track_anchor_smoothing_residual_m=max_track_anchor_smoothing_residual_m,
            smoothing_residual_identity_reset_m=smoothing_residual_identity_reset_m,
        )
    else:
        smoothed, smoothing_metrics = _smooth_grounded_frames(
            grounded,
            alpha=smoothing_alpha,
            max_root_speed_mps=max_root_speed_mps,
            max_track_anchor_smoothing_residual_m=max_track_anchor_smoothing_residual_m,
            smoothing_residual_identity_reset_m=smoothing_residual_identity_reset_m,
        )
    mesh_faces = _common_mesh_faces(smoothed)
    players: list[dict[str, Any]] = []
    skeleton_players: list[dict[str, Any]] = []
    max_joint_count = max(len(frame["joints_world"]) for frame in smoothed)
    foot_lock_player_summaries: list[dict[str, Any]] = []
    for player_id in sorted({int(frame["player_id"]) for frame in smoothed}):
        player_frames = [frame for frame in smoothed if int(frame["player_id"]) == player_id]
        if postchain.foot_lock:
            player_frames, foot_lock_summary = _apply_footlock_to_player_frames(
                player_frames,
                max_root_speed_mps=None if stance_aware_grounding else max_root_speed_mps,
                xy_translation_enabled=not stance_aware_grounding,
                max_allowed_xy_correction_m=FOOT_LOCK_MAX_XY_CORRECTION_M if stance_aware_grounding else None,
                smoothing_gap_carry_frames=smoothing_gap_carry_frames,
            )
        else:
            player_frames, foot_lock_summary = _bypass_footlock_for_player_frames(player_frames)
        foot_lock_player_summaries.append(foot_lock_summary)
        betas = _first_list(player_frames, "betas")
        # ADDITIVE (P2-2 GATE 1b, w5_p22latent_20260707): same per-player
        # collapse convention as `betas` above -- MHR scale_params is a
        # per-athlete bone-length/proportions correction, not a per-frame
        # quantity, so the first non-empty sample is representative.
        scale = _first_list(player_frames, "scale")
        contact_by_frame = [_infer_floor_contact(frame) for frame in player_frames]
        player_has_contact = any(contact["left"] or contact["right"] for contact in contact_by_frame)
        players.append(
            {
                "id": player_id,
                "betas": betas,
                "scale": scale,
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
                        "mesh_vertices_world": frame["vertices_world"],
                        "joint_conf": [float(frame["confidence"])] * len(frame["joints_world"]),
                        "foot_contact": contact,
                        "foot_lock": frame["foot_lock"],
                        "grf": None,
                        "confidence_provenance": {
                            "source": "sam3d_body_joints",
                            "model": model,
                            "grounding_anchor_source": anchor_source,
                        },
                        **_temporal_smoothing_metadata_export(frame),
                    }
                    for frame, contact in zip(player_frames, contact_by_frame)
                ],
                "foot_lock": _smpl_foot_lock_summary(foot_lock_summary),
                "skate_free": bool(foot_lock_summary["skate_free"]),
                "physics": (
                    "worldhmr_floor_contact_footlock_z_snap"
                    if player_has_contact and postchain.foot_lock
                    else "worldhmr_foot_lock_bypassed"
                    if player_has_contact
                    else "worldhmr_grounded_not_footlocked"
                ),
            }
        )
        skeleton_players.append(
            {
                "id": player_id,
                "frames": [
                    {
                        "frame_idx": int(frame["frame_idx"]),
                        "t": float(frame["t"]),
                        "transl_world": list(frame["transl_world"]),
                        "joints_world": [list(joint) for joint in frame["joints_world"]],
                        "joint_conf": [float(frame["confidence"])] * len(frame["joints_world"]),
                        "confidence_provenance": {
                            "source": "sam3d_body_joints",
                            "model": model,
                            "grounding_anchor_source": anchor_source,
                        },
                        **_temporal_smoothing_metadata_export(frame),
                    }
                    for frame in player_frames
                ],
            }
        )

    smpl_motion = {
        "schema_version": 1,
        "model": model,
        "fps": float(fps),
        "world_frame": "court_Z0",
        "players": players,
    }
    if mesh_faces:
        smpl_motion["mesh_faces"] = mesh_faces
    skeleton_provenance = {
        "lane": "BODY_TIER2",
        "source": "sam3d_body_joints",
        "model_family": model,
        "camera_offset_rotation_convention": ROTATION_CONVENTION_OFFSET_ROW_TIMES_R,
        "grounding": "camera_offset_row_times_R_plus_track_footpoint_court_z0",
        "grounding_anchor_source": anchor_source,
        "stance_aware_grounding": bool(stance_aware_grounding),
        "protected_eval_labels_used": False,
    }
    if camera_motion_provenance:
        skeleton_provenance["camera_motion"] = camera_motion_provenance
    skeleton3d = {
        "schema_version": 1,
        "artifact_type": "racketsport_skeleton3d",
        "fps": float(fps),
        "world_frame": "court_Z0",
        "source_model": "sam3d_body_joints",
        "joint_names": [f"sam3dbody_joint_{idx:03d}" for idx in range(max_joint_count)],
        "preview_only": False,
        "players": skeleton_players,
        "provenance": skeleton_provenance,
    }
    if postchain.temporal_smoothing:
        skeleton3d = apply_sam3d_temporal_refine_gate(
            skeleton3d,
            fps=fps,
            sam3d_wrist_bone_lock=sam3d_wrist_bone_lock,
        )
    else:
        skeleton3d = _with_sam3d_temporal_refine_status(
            skeleton3d,
            status="bypassed_body_postchain_temporal_smoothing_disabled",
            wrist_peak_timing_gate_pass=False,
        )
        if sam3d_wrist_bone_lock:
            skeleton3d = apply_sam3d_wrist_bone_lock(skeleton3d)
    refined_stance_metrics: dict[str, Any] = {}
    if anchor_source == R3_GROUNDING_ANCHOR_SOURCE and postchain.foot_pin:
        smpl_motion, skeleton3d, refined_stance_metrics = _apply_refined_stance_phase_lock_and_pin(
            smpl_motion,
            skeleton3d,
            fps=fps,
            deep_copy_payload=False,
        )
        players = smpl_motion["players"]
    elif anchor_source == R3_GROUNDING_ANCHOR_SOURCE and not postchain.foot_pin:
        refined_stance_metrics = _bypassed_foot_pin_metrics()
    mesh_vertices_by_key = _detach_smpl_motion_mesh_vertices(smpl_motion)
    smpl_motion, skeleton3d, world_joint_visual_smoothing_metrics = _apply_world_joint_visual_smoothing(
        smpl_motion,
        skeleton3d,
        fps=fps,
        enabled=world_joint_visual_smoothing,
        stance_index=stance_index,
    )
    _reattach_smpl_motion_mesh_vertices(smpl_motion, mesh_vertices_by_key)
    players = smpl_motion["players"]
    if not postchain.is_default:
        skeleton3d = _mark_body_postchain_bypasses(skeleton3d, postchain)
    sam3d_temporal_refine = dict(skeleton3d.get("provenance", {}).get("sam3d_temporal_refine", {}))
    sam3d_lock = dict(skeleton3d.get("provenance", {}).get("sam3d_wrist_bone_lock", {}))
    temporal_refine = dict(skeleton3d.get("provenance", {}).get("temporal_refine", {}))
    physical_plausibility = dict(temporal_refine.get("physical_plausibility", {}))
    metrics = {
        "body_samples": len(samples),
        "players": len(players),
        "frames": len({int(frame["frame_idx"]) for frame in smoothed}),
        "world_frame": "court_Z0",
        "grounding": "camera_offset_row_times_R_plus_track_footpoint_court_z0",
        "grounding_anchor_source": anchor_source,
        "stance_aware_grounding": bool(stance_aware_grounding),
        "camera_offset_rotation_convention": ROTATION_CONVENTION_OFFSET_ROW_TIMES_R,
        "grounding_anchor": _common_grounding_anchor(smoothed),
        "smoothing_alpha": smoothing_alpha,
        "sam3d_temporal_refine_status": sam3d_temporal_refine.get("status", "absent"),
        "sam3d_wrist_peak_timing_gate_pass": sam3d_temporal_refine.get("wrist_peak_timing_gate_pass"),
        "sam3d_wrist_bone_lock_status": sam3d_lock.get("status", "disabled" if not sam3d_wrist_bone_lock else "absent"),
        "sam3d_wrist_bone_lock_locked_frame_count": sam3d_lock.get("locked_frame_count", 0),
        "sam3d_core_body_speed_clamp_engagement_by_player": physical_plausibility.get(
            "core_body_speed_clamp_engagement_by_player",
            {},
        ),
        "max_root_speed_mps": max_root_speed_mps,
        "max_track_anchor_smoothing_residual_m": max_track_anchor_smoothing_residual_m,
        "smoothing_gap_carry_frames": smoothing_gap_carry_frames,
        "smoothing_residual_identity_reset_m": smoothing_residual_identity_reset_m,
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
        "foot_lock_contact_frames": sum(int(summary["contact_frames"]) for summary in foot_lock_player_summaries),
        "foot_lock_contact_samples": sum(int(summary["contact_samples"]) for summary in foot_lock_player_summaries),
        "foot_lock_root_speed_limited_frames": sum(
            int(summary["root_speed_limited_frames"]) for summary in foot_lock_player_summaries
        ),
        "foot_lock_xy_capped_frames": sum(
            int(summary.get("xy_capped_frames", 0)) for summary in foot_lock_player_summaries
        ),
        "foot_lock_gap_carried_frames": sum(
            int(summary.get("gap_carried_frames", 0)) for summary in foot_lock_player_summaries
        ),
        "foot_lock_gap_reset_frames": sum(
            int(summary.get("gap_reset_frames", 0)) for summary in foot_lock_player_summaries
        ),
        "max_foot_lock_slide_m": max(
            (float(summary["max_slide_m"]) for summary in foot_lock_player_summaries),
            default=0.0,
        ),
        "max_foot_lock_penetration_m": max(
            (float(summary["max_penetration_m"]) for summary in foot_lock_player_summaries),
            default=0.0,
        ),
        "foot_lock_skate_free_players": sum(1 for summary in foot_lock_player_summaries if summary["skate_free"]),
        "world_joint_visual_smoothing": world_joint_visual_smoothing_metrics,
        "grf_frames": sum(
            1
            for player in players
            for frame in player["frames"]
            if frame["grf"] is not None
        ),
        "skate_free_players": sum(1 for player in players if player["skate_free"]),
        **camera_motion_metrics,
    }
    metrics.update(refined_stance_metrics)
    if not postchain.is_default:
        metrics["body_postchain"] = postchain.to_artifact_dict()
        metrics["postchain_bypassed_stages"] = postchain.bypassed_stages()
        if raw_grounded_joints is not None:
            metrics["raw_grounded_joints_sidecar"] = RAW_GROUNDED_JOINTS_ARTIFACT
    return BodySkeletonAndMetrics(
        smpl_motion_view=smpl_motion,
        skeleton3d=skeleton3d,
        metrics=metrics,
        raw_grounded_joints=raw_grounded_joints,
    )


def assemble_body_monolith_payloads(
    smpl_motion_view: Mapping[str, Any],
    skeleton3d: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Materialize JSON-safe legacy BODY monolith payloads from shared compute."""

    return (
        copy.deepcopy(dict(smpl_motion_view)),
        copy.deepcopy(dict(skeleton3d)),
        copy.deepcopy(dict(metrics)),
    )


def _bypass_temporal_smoothing(
    grounded: Sequence[dict[str, Any]],
    *,
    stance_aware_grounding: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for frame in grounded:
        copied = copy.deepcopy(dict(frame))
        copied["grounding_anchor"] = frame.get("grounding_anchor", "")
        copied["temporal_smoothing_reset"] = False
        copied.pop("temporal_smoothing_metadata", None)
        frames.append(copied)
    metrics: dict[str, Any] = {
        "temporal_smoothing_status": "bypassed_body_postchain_temporal_smoothing_disabled",
        "root_speed_limited_frames": 0,
        "track_anchor_residual_reset_frames": 0,
        "track_anchor_residual_carried_frames": 0,
        "track_anchor_residual_identity_reset_frames": 0,
        "max_pre_reset_track_anchor_residual_m": 0.0,
        "max_track_anchor_residual_m": 0.0,
    }
    if stance_aware_grounding:
        metrics.update(
            {
                "root_speed_anomaly_frames": 0,
                "root_speed_clamp_engagement_overall": 0.0,
                "root_speed_anomaly_fraction_overall": 0.0,
                "root_speed_clamp_engagement_by_player": {},
                "root_speed_anomaly_fraction_by_player": {},
                "transition_anchor_lag_p95_m": 0.0,
                "transition_anchor_lag_median_m": 0.0,
                "body_marker_transition_divergence_p90_m": 0.0,
                "stance_aware_grounding": {
                    "source": R3_GROUNDING_ANCHOR_SOURCE,
                    "status": "bypassed_body_postchain_temporal_smoothing_disabled",
                    "stance_frame_count": 0,
                    "rejected_stance_frame_count": 0,
                    "rejected_stance_reasons": {},
                    "transition_frame_count": 0,
                    "track_anchor_residual_m": _distribution_m([]),
                    "transition_anchor_lag_m": _distribution_m([]),
                    "transition_anchor_lag_p95_m": 0.0,
                    "transition_anchor_lag_median_m": 0.0,
                    "body_marker_transition_divergence": _distribution_m([]),
                    "residual_reset_frames": 0,
                    "residual_carried_frames": 0,
                    "residual_identity_reset_frames": 0,
                    "root_speed_clamp_engagement_overall": 0.0,
                    "root_speed_clamp_engagement_by_player": {},
                },
            }
        )
    return frames, metrics


def _bypass_footlock_for_player_frames(
    player_frames: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    copied_frames: list[dict[str, Any]] = []
    for frame in player_frames:
        copied = copy.deepcopy(dict(frame))
        copied["foot_lock"] = {"left": False, "right": False}
        copied_frames.append(copied)
    return copied_frames, {
        "scaffold": "body_postchain_bypass",
        "status": "bypassed_body_postchain_foot_lock_disabled",
        "contact_frames": 0,
        "contact_samples": 0,
        "root_speed_limited_frames": 0,
        "xy_capped_frames": 0,
        "gap_carried_frames": 0,
        "gap_reset_frames": 0,
        "max_xy_correction_m": 0.0,
        "max_allowed_xy_correction_m": 0.0,
        "max_slide_m": 0.0,
        "max_penetration_m": 0.0,
        "skate_free": False,
    }


def _bypassed_foot_pin_metrics() -> dict[str, Any]:
    return {
        "refined_stance_phase_count": 0,
        "refined_stance_split_phase_count": 0,
        "refined_stance_phase_split_count": 0,
        "foot_pin_phase_count": 0,
        "foot_pin_corrected_frame_count": 0,
        "foot_pin_max_correction_m": 0.0,
        "foot_pin_status": "bypassed_body_postchain_foot_pin_disabled",
        "foot_lock_slide_metric": "body_postchain_foot_pin_bypassed",
        "max_foot_lock_slide_m": 0.0,
        "foot_lock_slide_p95_m": 0.0,
        "max_candidate_phase_slide_m": 0.0,
        "candidate_phase_count": 0,
        "candidate_phase_rejected_count": 0,
        "candidate_phase_rejection_reason_counts": {},
        "foot_lock_gate_stream": {
            "artifact_type": "foot_lock_gate_stream",
            "status": "bypassed_body_postchain_foot_pin_disabled",
            "phase_rows": [],
            "frame_rows": [],
            "summary": {"phase_count": 0, "frame_count": 0},
        },
    }


def _mark_body_postchain_bypasses(
    skeleton3d: Mapping[str, Any],
    postchain: BodyPostChainConfig,
) -> dict[str, Any]:
    output = copy.deepcopy(dict(skeleton3d))
    bypass_summary = postchain.bypass_summary()
    if bypass_summary is None:
        return output
    provenance = dict(output.get("provenance", {}))
    provenance["body_postchain_bypass"] = {
        **bypass_summary,
        "strict_mode_loud": True,
    }
    if not postchain.foot_pin:
        provenance.setdefault(
            "foot_pin",
            {
                "status": "bypassed_body_postchain_foot_pin_disabled",
                "strict_mode_loud": True,
            },
        )
    if not postchain.world_joint_visual_smoothing:
        provenance.setdefault(
            "worldhmr_visual_smoothing",
            {
                "enabled": False,
                "status": "bypassed_body_postchain_world_joint_visual_smoothing_disabled",
            },
        )
    output["provenance"] = provenance
    return output


def _build_raw_grounded_joints_sidecar(
    grounded: Sequence[dict[str, Any]],
    *,
    fps: float,
    model: str,
    grounding_anchor_source: str,
    postchain: BodyPostChainConfig,
) -> dict[str, Any]:
    if not grounded:
        raise ValueError("raw grounded joints sidecar requires at least one grounded BODY frame")
    player_frames: dict[int, list[dict[str, Any]]] = {}
    max_joint_count = 0
    min_joint_count: int | None = None
    for index, frame in enumerate(grounded):
        missing = [
            field
            for field in ("frame_idx", "player_id", "t", "track_world_xy", "transl_world", "joints_world")
            if field not in frame
        ]
        if missing:
            raise ValueError(f"raw grounded joints frame {index} missing required field(s): {', '.join(missing)}")
        player_id = int(frame["player_id"])
        joints = _vector3_list(frame["joints_world"], name=f"raw_grounded_joints/players/{player_id}/joints_world")
        if not joints:
            raise ValueError(f"raw grounded joints frame {index} has no joints_world samples")
        max_joint_count = max(max_joint_count, len(joints))
        min_joint_count = len(joints) if min_joint_count is None else min(min_joint_count, len(joints))
        confidence = float(frame.get("confidence", 0.0))
        player_frames.setdefault(player_id, []).append(
            {
                "frame_idx": int(frame["frame_idx"]),
                "t": float(frame["t"]),
                "track_world_xy": _vector2(frame["track_world_xy"], name="track_world_xy"),
                "transl_world": _vector3(frame["transl_world"], name="transl_world"),
                "joints_world": joints,
                "joint_conf": [confidence] * len(joints),
                "grounding_anchor": str(frame.get("grounding_anchor", "")),
            }
        )
    players = [
        {
            "id": player_id,
            "frames": sorted(frames, key=lambda item: int(item["frame_idx"])),
        }
        for player_id, frames in sorted(player_frames.items())
    ]
    sidecar = {
        "schema_version": 1,
        "artifact_type": "racketsport_body_raw_grounded_joints",
        "source": "worldhmr._ground_fast_sam_sample",
        "model": model,
        "fps": float(fps),
        "world_frame": "court_Z0",
        "grounding": "camera_offset_row_times_R_plus_track_footpoint_court_z0",
        "grounding_anchor_source": grounding_anchor_source,
        "postchain": postchain.to_artifact_dict(mode="raw"),
        "postchain_bypassed_stages": postchain.bypassed_stages(),
        "joint_names": [f"sam3dbody_joint_{idx:03d}" for idx in range(max_joint_count)],
        "players": players,
        "summary": {
            "player_count": len(players),
            "frame_count": len({int(frame["frame_idx"]) for frame in grounded}),
            "sample_count": len(grounded),
            "joint_count_min": int(min_joint_count or 0),
            "joint_count_max": int(max_joint_count),
        },
    }
    _validate_raw_grounded_joints_sidecar(sidecar)
    return sidecar


def _validate_raw_grounded_joints_sidecar(sidecar: Mapping[str, Any]) -> None:
    required = (
        "schema_version",
        "artifact_type",
        "fps",
        "world_frame",
        "postchain",
        "postchain_bypassed_stages",
        "joint_names",
        "players",
        "summary",
    )
    missing = [field for field in required if field not in sidecar]
    if missing:
        raise ValueError(f"raw grounded joints sidecar missing required field(s): {', '.join(missing)}")
    if sidecar["artifact_type"] != "racketsport_body_raw_grounded_joints":
        raise ValueError("raw grounded joints sidecar artifact_type mismatch")
    if sidecar["world_frame"] != "court_Z0":
        raise ValueError("raw grounded joints sidecar must use court_Z0 world_frame")
    if not isinstance(sidecar.get("players"), list) or not sidecar["players"]:
        raise ValueError("raw grounded joints sidecar must include at least one player")
    for player in sidecar["players"]:
        if not isinstance(player, Mapping) or "id" not in player or "frames" not in player:
            raise ValueError("raw grounded joints sidecar player rows require id and frames")
        frames = player["frames"]
        if not isinstance(frames, list) or not frames:
            raise ValueError("raw grounded joints sidecar players require non-empty frames")
        for frame in frames:
            if not isinstance(frame, Mapping):
                raise ValueError("raw grounded joints sidecar frame rows must be objects")
            for field in ("frame_idx", "t", "track_world_xy", "transl_world", "joints_world", "joint_conf"):
                if field not in frame:
                    raise ValueError(f"raw grounded joints sidecar frame missing {field}")
            joints = frame["joints_world"]
            if not isinstance(joints, list) or not joints:
                raise ValueError("raw grounded joints sidecar frame joints_world must be non-empty")


def apply_sam3d_temporal_refine_gate(
    skeleton3d: Mapping[str, Any],
    *,
    fps: float | None = None,
    sam3d_wrist_bone_lock: bool = True,
) -> dict[str, Any]:
    """Run SAM-3D temporal refinement and enforce the wrist peak gate.

    A failed or blocked gate returns the original grounded skeleton with a
    provenance flag instead of adopting smoothed coordinates.
    """

    original = copy.deepcopy(dict(skeleton3d))
    try:
        refined = refine_sam3d_skeleton3d(original, fps=fps, apply_world_grounding=False)
    except Exception as exc:
        failed = _with_sam3d_temporal_refine_status(
            original,
            status="skipped_refine_error",
            wrist_peak_timing_gate_pass=False,
            error=str(exc),
        )
        return apply_sam3d_wrist_bone_lock(failed) if sam3d_wrist_bone_lock else failed

    temporal = dict(refined.get("provenance", {}).get("temporal_refine", {}))
    wrist_timing = temporal.get("wrist_peak_timing")
    gate_pass = temporal.get("wrist_peak_timing_gate_pass") is True
    if gate_pass:
        accepted = _with_sam3d_temporal_refine_status(
            refined,
            status="applied",
            wrist_peak_timing_gate_pass=True,
            wrist_peak_timing=wrist_timing,
        )
        return apply_sam3d_wrist_bone_lock(accepted) if sam3d_wrist_bone_lock else accepted
    rejected = _with_sam3d_temporal_refine_status(
        original,
        status="rejected_wrist_peak_gate_failed",
        wrist_peak_timing_gate_pass=False,
        wrist_peak_timing=wrist_timing,
    )
    return apply_sam3d_wrist_bone_lock(rejected) if sam3d_wrist_bone_lock else rejected


def _apply_refined_stance_phase_lock_and_pin(
    smpl_motion: Mapping[str, Any],
    skeleton3d: Mapping[str, Any],
    *,
    fps: float,
    deep_copy_payload: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    phases = _detect_refined_contact_phases(skeleton3d)
    if not phases:
        return _copy_body_payload(smpl_motion, deep=deep_copy_payload), copy.deepcopy(dict(skeleton3d)), {
            "refined_stance_phase_count": 0,
            "refined_stance_phase_split_count": 0,
            "foot_pin_phase_count": 0,
            "foot_lock_slide_metric": "phase_anchored_contiguous_contact",
        }
    track_xy_by_key = _track_xy_by_key_from_smpl_motion(smpl_motion)
    split_phases, split_count = _split_contact_phases_by_anchor_drift(
        phases,
        track_xy_by_key=track_xy_by_key,
        max_anchor_drift_m=STANCE_AWARE_PHASE_ANCHOR_DRIFT_SPLIT_M,
    )
    locked_smpl, root_lock_summary = _apply_root_phase_median_lock_to_payload(
        smpl_motion,
        split_phases,
        track_xy_by_key=track_xy_by_key,
        translate_mesh=True,
        deep_copy_payload=deep_copy_payload,
    )
    locked_skeleton, _unused = _apply_root_phase_median_lock_to_payload(
        skeleton3d,
        split_phases,
        track_xy_by_key=track_xy_by_key,
        translate_mesh=False,
    )
    foot_pin = apply_foot_pin_to_payload(
        locked_skeleton,
        settings=FootPinSettings(
            taper_frames=0,
            max_correction_m=REFINED_STANCE_FOOT_PIN_MAX_CORRECTION_M,
            max_smoothing_correction_m=0.0,
            interpolate_between_stances=False,
        ),
    )
    pinned_skeleton = copy.deepcopy(foot_pin.payload)
    foot_pin_audit = dict(foot_pin.audit)
    pinned_skeleton.pop("foot_pin", None)
    provenance = dict(pinned_skeleton.get("provenance", {}))
    provenance["refined_stance_phase_lock"] = {
        "source": "post_temporal_refine_skeleton_contact_phases",
        "phase_count": len(phases),
        "split_phase_count": len(split_phases),
        "anchor_drift_split_m": STANCE_AWARE_PHASE_ANCHOR_DRIFT_SPLIT_M,
        **root_lock_summary,
    }
    provenance["foot_pin"] = {
        "version": foot_pin_audit.get("foot_pin_version", 1),
        "source": "post_root_lock_skeleton_contact_phases",
        "settings": foot_pin_audit.get("settings", {}),
        "phase_detection": {
            "candidate_phase_count": foot_pin_audit.get("phase_detection", {}).get("candidate_phase_count", 0),
            "confident_phase_count": foot_pin_audit.get("phase_detection", {}).get("confident_phase_count", 0),
            "skipped_low_confidence_phase_count": foot_pin_audit.get("phase_detection", {}).get(
                "skipped_low_confidence_phase_count",
                0,
            ),
        },
        "summary": foot_pin_audit.get("summary", {}),
        "players": {
            str(player_id): {
                "phase_count": player.get("phase_count", 0),
                "corrected_frame_count": player.get("corrected_frame_count", 0),
                "max_correction_m": player.get("max_correction_m", 0.0),
                "frame_corrections": player.get("frame_corrections", []),
            }
            for player_id, player in dict(foot_pin_audit.get("players", {})).items()
            if isinstance(player, Mapping)
        },
        "correction_scope": "stance_leg_chains_only_no_track_world_xy_mutation",
    }
    pinned_skeleton["provenance"] = provenance

    final_metrics, gate_stream = _contact_gate_stream_for_skeleton3d(
        pinned_skeleton,
        clip=str(pinned_skeleton.get("clip", "unknown")),
        threshold_m=0.03,
    )
    max_slide_m = max(
        (float(metric.get("slide_mm", 0.0)) / 1000.0 for metric in final_metrics.get("phase_metrics", [])),
        default=0.0,
    )
    p95_slide_m = _percentile(
        [float(metric.get("slide_mm", 0.0)) / 1000.0 for metric in final_metrics.get("phase_metrics", [])],
        95.0,
    )
    return locked_smpl, pinned_skeleton, {
        "refined_stance_phase_count": len(phases),
        "refined_stance_split_phase_count": len(split_phases),
        "refined_stance_phase_split_count": split_count,
        "refined_stance_root_lock": root_lock_summary,
        "foot_pin_phase_count": int(foot_pin_audit.get("summary", {}).get("total_phase_count", 0)),
        "foot_pin_corrected_frame_count": int(foot_pin_audit.get("summary", {}).get("total_corrected_frame_count", 0)),
        "foot_pin_max_correction_m": float(foot_pin_audit.get("summary", {}).get("max_correction_m", 0.0)),
        "foot_lock_slide_metric": "phase_anchored_contiguous_contact",
        "max_foot_lock_slide_m": max_slide_m,
        "foot_lock_slide_p95_m": p95_slide_m,
        "max_candidate_phase_slide_m": float(final_metrics.get("max_candidate_phase_slide_m", max_slide_m)),
        "candidate_phase_count": int(final_metrics.get("candidate_phase_count", 0)),
        "candidate_phase_rejected_count": int(final_metrics.get("candidate_phase_rejected_count", 0)),
        "candidate_phase_rejection_reason_counts": dict(
            final_metrics.get("candidate_phase_rejection_reason_counts", {})
        )
        if isinstance(final_metrics.get("candidate_phase_rejection_reason_counts"), Mapping)
        else {},
        "foot_lock_gate_stream": gate_stream,
    }


def _with_sam3d_temporal_refine_status(
    skeleton3d: Mapping[str, Any],
    *,
    status: str,
    wrist_peak_timing_gate_pass: bool,
    wrist_peak_timing: Any | None = None,
    error: str = "",
) -> dict[str, Any]:
    output = copy.deepcopy(dict(skeleton3d))
    provenance = dict(output.get("provenance", {}))
    record = {
        "status": status,
        "source": "refine_sam3d_skeleton3d",
        "wrist_peak_timing_gate_pass": bool(wrist_peak_timing_gate_pass),
        "protected_eval_labels_used": False,
        "internal_val_only": True,
    }
    if wrist_peak_timing is not None:
        record["wrist_peak_timing"] = wrist_peak_timing
    if error:
        record["error"] = error
    provenance["sam3d_temporal_refine"] = record
    output["provenance"] = provenance
    return output


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


def _detect_refined_contact_phases(skeleton3d: Mapping[str, Any]) -> list[ContactPhase]:
    frames, joint_names = _contact_frames_from_skeleton3d(skeleton3d)
    if not frames:
        return []
    try:
        return detect_contact_phases(frames, joint_names=joint_names, thresholds=_gate_contact_thresholds())
    except ValueError:
        return []


def _contact_metrics_for_skeleton3d(skeleton3d: Mapping[str, Any]) -> dict[str, Any]:
    metrics, _gate_stream = _contact_gate_stream_for_skeleton3d(skeleton3d, clip="unknown", threshold_m=0.03)
    return metrics


def _contact_gate_stream_for_skeleton3d(
    skeleton3d: Mapping[str, Any],
    *,
    clip: str,
    threshold_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frames, joint_names = _contact_frames_from_skeleton3d(skeleton3d)
    if not frames:
        metrics = {
            "phase_metrics": [],
            "summary_by_player": {},
            "penetration": {},
            "max_candidate_phase_slide_m": 0.0,
            "candidate_phase_count": 0,
            "candidate_phase_rejected_count": 0,
            "candidate_phase_rejection_reason_counts": {},
        }
        return metrics, _empty_gate_stream(clip=clip)
    phases = detect_contact_phases(frames, joint_names=joint_names, thresholds=_gate_contact_thresholds())
    raw_metrics = measure_contact_metrics(frames, phases, joint_names=joint_names).to_dict()
    phase_rejection_reasons = (
        _gate_phase_rejection_reasons(phases, raw_metrics=raw_metrics)
        if _use_lock_eligible_gate_metric(skeleton3d)
        else {}
    )
    if phase_rejection_reasons:
        accepted_phases = [phase for phase in phases if _contact_phase_key(phase) not in phase_rejection_reasons]
        metrics = measure_contact_metrics(frames, accepted_phases, joint_names=joint_names).to_dict()
    else:
        metrics = raw_metrics
    metrics["max_candidate_phase_slide_m"] = _max_phase_slide_m(raw_metrics)
    metrics["candidate_phase_count"] = len(raw_metrics.get("phase_metrics", []))
    metrics["candidate_phase_rejected_count"] = len(phase_rejection_reasons)
    metrics["candidate_phase_rejection_reason_counts"] = _counts(tuple(phase_rejection_reasons.values()))
    return metrics, _foot_lock_gate_stream(
        skeleton3d,
        frames=frames,
        phases=phases,
        metrics=metrics,
        joint_names=joint_names,
        clip=clip,
        threshold_m=threshold_m,
        phase_rejection_reasons=phase_rejection_reasons,
    )


def _gate_contact_thresholds() -> ContactThresholds:
    base = ContactThresholds()
    return ContactThresholds(
        enter_height_m=base.enter_height_m,
        exit_height_m=base.exit_height_m,
        enter_speed_mps=base.enter_speed_mps,
        exit_speed_mps=base.exit_speed_mps,
        min_confidence=base.min_confidence,
        min_phase_frames=base.min_phase_frames,
        low_foot_band_m=base.low_foot_band_m,
        split_speed_mps=base.enter_speed_mps,
    )


def _use_lock_eligible_gate_metric(skeleton3d: Mapping[str, Any]) -> bool:
    provenance = skeleton3d.get("provenance")
    return isinstance(provenance, Mapping) and (
        "refined_stance_phase_lock" in provenance or "foot_pin" in provenance
    )


def _gate_phase_rejection_reasons(
    phases: Sequence[ContactPhase],
    *,
    raw_metrics: Mapping[str, Any],
) -> dict[tuple[str, str, int, int], str]:
    metric_by_key = {
        _metric_phase_key(row): row
        for row in raw_metrics.get("phase_metrics", [])
        if isinstance(row, Mapping)
    }
    reasons: dict[tuple[str, str, int, int], str] = {}
    for phase in phases:
        key = _contact_phase_key(phase)
        reason: str | None = None
        if phase.weak:
            reason = phase.rejection_reason or "weak_phase"
        elif phase.demoted:
            reason = phase.rejection_reason or "demoted_phase"
        elif phase.min_confidence < 0.90:
            reason = "low_body_contact_confidence"
        else:
            metric = metric_by_key.get(key)
            penetration_m = float(metric.get("max_penetration_mm", 0.0)) / 1000.0 if isinstance(metric, Mapping) else 0.0
            if penetration_m > 0.0:
                reason = "phase_penetrates_ground"
        if reason is not None:
            reasons[key] = reason
    return reasons


def _max_phase_slide_m(metrics: Mapping[str, Any]) -> float:
    return max(
        (
            float(row.get("slide_mm", 0.0)) / 1000.0
            for row in metrics.get("phase_metrics", [])
            if isinstance(row, Mapping)
        ),
        default=0.0,
    )


def _contact_phase_key(phase: ContactPhase) -> tuple[str, str, int, int]:
    return (
        str(phase.player_id),
        str(phase.foot),
        int(phase.start_frame_index),
        int(phase.end_frame_index),
    )


def _metric_phase_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row.get("player_id", "unknown")),
        str(row.get("foot", "unknown")),
        int(row.get("start_frame_index", -1)),
        int(row.get("end_frame_index", -1)),
    )


def _counts(values: Sequence[str] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _contact_frames_from_skeleton3d(skeleton3d: Mapping[str, Any]) -> tuple[list[ContactSkeletonFrame], list[str]]:
    joint_names = [str(name) for name in skeleton3d.get("joint_names", [])]
    frames: list[ContactSkeletonFrame] = []
    for player in skeleton3d.get("players", []):
        if not isinstance(player, Mapping):
            continue
        player_id = str(player.get("id", "unknown"))
        for frame in player.get("frames", []):
            if not isinstance(frame, Mapping):
                continue
            joints = frame.get("joints_world")
            if not isinstance(joints, Sequence) or isinstance(joints, (str, bytes)) or not joints:
                continue
            frames.append(
                ContactSkeletonFrame(
                    player_id=player_id,
                    frame_index=int(frame.get("frame_idx", round(float(frame.get("t", 0.0)) * 30.0))),
                    t=float(frame.get("t", 0.0)),
                    joints_world=[[float(value) for value in joint] for joint in joints],
                    joint_conf=[float(value) for value in frame.get("joint_conf", [])]
                    if isinstance(frame.get("joint_conf"), Sequence)
                    else None,
                )
            )
    return frames, joint_names


def _foot_lock_gate_stream(
    skeleton3d: Mapping[str, Any],
    *,
    frames: Sequence[ContactSkeletonFrame],
    phases: Sequence[ContactPhase],
    metrics: Mapping[str, Any],
    joint_names: Sequence[str],
    clip: str,
    threshold_m: float,
    phase_rejection_reasons: Mapping[tuple[str, str, int, int], str] | None = None,
) -> dict[str, Any]:
    if not frames:
        return _empty_gate_stream(clip=clip)
    try:
        foot_indices = resolve_foot_joint_indices(joint_names, joint_count=len(frames[0].joints_world))
    except ValueError:
        return _empty_gate_stream(clip=clip)
    frame_by_key = {(str(frame.player_id), int(frame.frame_index)): frame for frame in frames}
    payload_by_key = _skeleton_frame_payload_by_key(skeleton3d)
    phase_rows: list[dict[str, Any]] = []
    frame_rows_unbounded: list[dict[str, Any]] = []
    rejection_reasons = phase_rejection_reasons or {}
    for phase_ordinal, phase in enumerate(phases):
        phase_id = f"{phase.player_id}:{phase.foot}:{phase.start_frame_index}-{phase.end_frame_index}:{phase_ordinal}"
        rejection_reason = rejection_reasons.get(_contact_phase_key(phase))
        anchor_frame = frame_by_key.get((str(phase.player_id), int(phase.start_frame_index)))
        if anchor_frame is None:
            continue
        anchor = foot_contact_point(anchor_frame, foot_indices.for_foot(phase.foot))
        max_slide_m = 0.0
        max_frame_index = int(phase.start_frame_index)
        for frame_index in phase.frame_indices:
            frame = frame_by_key.get((str(phase.player_id), int(frame_index)))
            if frame is None:
                continue
            point = foot_contact_point(frame, foot_indices.for_foot(phase.foot))
            slide_m = _distance2(point[:2], anchor[:2])
            if slide_m >= max_slide_m:
                max_slide_m = slide_m
                max_frame_index = int(frame_index)
            payload_frame = payload_by_key.get((str(phase.player_id), int(frame_index)), {})
            frame_rows_unbounded.append(
                {
                    "clip": clip,
                    "player_id": str(phase.player_id),
                    "foot": phase.foot,
                    "phase_id": phase_id,
                    "frame_idx": int(frame_index),
                    "contact_state": True,
                    "selected_foot": phase.foot,
                    "lock_anchor_xyz": [float(anchor[0]), float(anchor[1]), float(anchor[2])],
                    "raw_xy": [float(point[0]), float(point[1])],
                    "fused_xy": [float(point[0]), float(point[1])],
                    "smoothed_xy": [float(point[0]), float(point[1])],
                    "original_xy": [float(point[0]), float(point[1])],
                    "body_root_world": _body_root_world(payload_frame),
                    "output_source": str(payload_frame.get("output_source", payload_frame.get("grounding_anchor", "body"))),
                    "divergence_flag": _frame_has_divergence_flag(payload_frame),
                    "speed_cap_flag": _frame_has_speed_cap_flag(payload_frame),
                    "residuals": _frame_residuals(payload_frame),
                    "bbox_margin_px": payload_frame.get("bbox_margin_px"),
                    "source_counts": dict(payload_frame.get("source_counts", {}))
                    if isinstance(payload_frame.get("source_counts"), Mapping)
                    else {},
                    "foot_pin_correction_m": _foot_pin_correction_m(payload_frame),
                }
            )
        phase_rows.append(
            {
                "clip": clip,
                "player_id": str(phase.player_id),
                "foot": phase.foot,
                "phase_id": phase_id,
                "start_frame_index": int(phase.start_frame_index),
                "end_frame_index": int(phase.end_frame_index),
                "frame_count": int(phase.frame_count),
                "slide_m": float(max_slide_m),
                "max_contributing_frame_index": int(max_frame_index),
                "anchor_position_xyz": [float(anchor[0]), float(anchor[1]), float(anchor[2])],
                "contact_source": phase.source,
                "source_phase_id": phase_id,
                "foot_assignment": phase.foot_assignment,
                "source_phase_foot": phase.source_phase_foot or phase.foot,
                "weak": bool(phase.weak or rejection_reason),
                "demoted": bool(phase.demoted or rejection_reason),
                "split": bool(phase.split),
                "split_reason": phase.split_reason,
                "rejection_reason": rejection_reason or phase.rejection_reason,
                "lock_metric_included": rejection_reason is None,
                "min_confidence": float(phase.min_confidence),
                "max_height_m": float(phase.max_height_m),
                "max_speed_mps": float(phase.max_speed_mps),
            }
        )
    frame_rows, stride = _bounded_gate_frame_rows(frame_rows_unbounded)
    top = sorted(phase_rows, key=lambda row: float(row["slide_m"]), reverse=True)
    weak_reasons = _counts(
        str(row.get("rejection_reason"))
        for row in phase_rows
        if row.get("rejection_reason")
    )
    candidate_max_slide_m = max((float(row["slide_m"]) for row in phase_rows), default=0.0)
    return {
        "schema_version": 1,
        "artifact_type": "foot_lock_gate_stream",
        "clip": clip,
        "phase_rows": phase_rows,
        "frame_rows": frame_rows,
        "summary": {
            "top_20_phases_by_slide_m": top[:20],
            "phases_over_threshold": [row for row in top if float(row["slide_m"]) > threshold_m],
            "weak_rejection_reasons": weak_reasons,
            "candidate_phase_rejection_reason_counts": weak_reasons,
            "max_candidate_phase_slide_m": float(candidate_max_slide_m),
            "frame_row_stride": stride,
            "frame_rows_unstrided_count": len(frame_rows_unbounded),
            "phase_count": len(phase_rows),
            "threshold_m": float(threshold_m),
            "metric_phase_count": len(metrics.get("phase_metrics", [])) if isinstance(metrics.get("phase_metrics"), list) else 0,
        },
        "artifact_size_policy": _gate_artifact_size_policy(),
    }


def _empty_gate_stream(*, clip: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "foot_lock_gate_stream",
        "clip": clip,
        "phase_rows": [],
        "frame_rows": [],
        "summary": {
            "top_20_phases_by_slide_m": [],
            "phases_over_threshold": [],
            "weak_rejection_reasons": {},
            "candidate_phase_rejection_reason_counts": {},
            "max_candidate_phase_slide_m": 0.0,
            "frame_row_stride": 1,
            "frame_rows_unstrided_count": 0,
            "phase_count": 0,
            "threshold_m": 0.03,
            "metric_phase_count": 0,
        },
        "artifact_size_policy": _gate_artifact_size_policy(),
    }


def _bounded_gate_frame_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if len(rows) <= FOOT_LOCK_GATE_STREAM_MAX_FRAME_ROWS:
        return list(rows), 1
    stride = max(1, (len(rows) + FOOT_LOCK_GATE_STREAM_MAX_FRAME_ROWS - 1) // FOOT_LOCK_GATE_STREAM_MAX_FRAME_ROWS)
    return [dict(row) for index, row in enumerate(rows) if index % stride == 0], stride


def _gate_artifact_size_policy() -> dict[str, Any]:
    return {
        "max_bytes": FOOT_LOCK_GATE_STREAM_MAX_BYTES,
        "action": "stride_frame_rows_when_needed",
        "max_frame_rows_before_stride": FOOT_LOCK_GATE_STREAM_MAX_FRAME_ROWS,
    }


def _skeleton_frame_payload_by_key(skeleton3d: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    out: dict[tuple[str, int], Mapping[str, Any]] = {}
    for player in skeleton3d.get("players", []) or []:
        if not isinstance(player, Mapping):
            continue
        player_id = str(player.get("id", "unknown"))
        for frame in player.get("frames", []) or []:
            if not isinstance(frame, Mapping):
                continue
            frame_idx = int(frame.get("frame_idx", round(float(frame.get("t", 0.0)) * 30.0)))
            out[(player_id, frame_idx)] = frame
    return out


def _body_root_world(frame: Mapping[str, Any]) -> list[float] | None:
    transl = frame.get("transl_world")
    if isinstance(transl, Sequence) and not isinstance(transl, (str, bytes)) and len(transl) >= 3:
        return [float(transl[0]), float(transl[1]), float(transl[2])]
    joints = frame.get("joints_world")
    if isinstance(joints, Sequence) and not isinstance(joints, (str, bytes)) and joints:
        first = joints[0]
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes)) and len(first) >= 3:
            return [float(first[0]), float(first[1]), float(first[2])]
    return None


def _frame_has_divergence_flag(frame: Mapping[str, Any]) -> bool:
    output_source = str(frame.get("output_source", ""))
    metadata = frame.get("temporal_smoothing_metadata")
    return output_source.startswith("fused_divergence") or (
        isinstance(metadata, Mapping) and str(metadata.get("residual_status", "")) == "reset"
    )


def _frame_has_speed_cap_flag(frame: Mapping[str, Any]) -> bool:
    metadata = frame.get("temporal_smoothing_metadata")
    return isinstance(metadata, Mapping) and bool(metadata.get("root_speed_limited", False))


def _frame_residuals(frame: Mapping[str, Any]) -> dict[str, Any]:
    metadata = frame.get("temporal_smoothing_metadata")
    if not isinstance(metadata, Mapping):
        return {}
    keys = (
        "track_anchor_residual_m",
        "pre_reset_track_anchor_residual_m",
        "body_marker_transition_divergence_m",
        "residual_status",
        "reset_reason",
    )
    return {key: metadata.get(key) for key in keys if key in metadata}


def _foot_pin_correction_m(frame: Mapping[str, Any]) -> float:
    raw = frame.get("foot_pin")
    if isinstance(raw, Mapping) and isinstance(raw.get("correction_magnitude_m"), (int, float)):
        return float(raw["correction_magnitude_m"])
    raw = frame.get("body_grounding_refinement")
    if isinstance(raw, Mapping) and isinstance(raw.get("correction_magnitude_m"), (int, float)):
        return float(raw["correction_magnitude_m"])
    return 0.0


def _track_xy_by_key_from_smpl_motion(smpl_motion: Mapping[str, Any]) -> dict[tuple[str, int], list[float]]:
    out: dict[tuple[str, int], list[float]] = {}
    for player in smpl_motion.get("players", []):
        if not isinstance(player, Mapping):
            continue
        player_id = str(player.get("id", "unknown"))
        for frame in player.get("frames", []):
            if not isinstance(frame, Mapping):
                continue
            track_xy = frame.get("track_world_xy")
            if not isinstance(track_xy, Sequence) or isinstance(track_xy, (str, bytes)) or len(track_xy) < 2:
                continue
            frame_idx = int(frame.get("frame_idx", round(float(frame.get("t", 0.0)) * 30.0)))
            out[(player_id, frame_idx)] = [float(track_xy[0]), float(track_xy[1])]
    return out


def _split_contact_phases_by_anchor_drift(
    phases: Sequence[ContactPhase],
    *,
    track_xy_by_key: Mapping[tuple[str, int], Sequence[float]],
    max_anchor_drift_m: float,
) -> tuple[list[ContactPhase], int]:
    split_phases: list[ContactPhase] = []
    split_count = 0
    for phase in phases:
        current_segment: list[int] = []
        segment_start_xy: Sequence[float] | None = None
        for frame_idx in phase.frame_indices:
            track_xy = track_xy_by_key.get((str(phase.player_id), int(frame_idx)))
            if track_xy is None:
                continue
            if (
                segment_start_xy is not None
                and _distance2(track_xy, segment_start_xy) > max_anchor_drift_m
                and current_segment
            ):
                split_phases.append(_copy_contact_phase_with_frames(phase, current_segment))
                split_count += 1
                current_segment = []
                segment_start_xy = None
            if segment_start_xy is None:
                segment_start_xy = track_xy
            current_segment.append(int(frame_idx))
        if current_segment:
            split_phases.append(_copy_contact_phase_with_frames(phase, current_segment))
    return split_phases, split_count


def _copy_contact_phase_with_frames(phase: ContactPhase, frame_indices: Sequence[int]) -> ContactPhase:
    return ContactPhase(
        player_id=phase.player_id,
        foot=phase.foot,
        frame_indices=tuple(int(frame_idx) for frame_idx in frame_indices),
        start_time_s=phase.start_time_s,
        end_time_s=phase.end_time_s,
        anchor_position_xyz=phase.anchor_position_xyz,
        max_height_m=phase.max_height_m,
        max_speed_mps=phase.max_speed_mps,
        min_confidence=phase.min_confidence,
        source=phase.source,
        source_phase_foot=phase.source_phase_foot,
        foot_assignment=phase.foot_assignment,
        weak=phase.weak,
        demoted=phase.demoted,
        split=True,
        split_reason=phase.split_reason or "anchor_drift_split",
        rejection_reason=phase.rejection_reason,
        source_thresholds=phase.source_thresholds,
        assignment_evidence=phase.assignment_evidence,
    )


def _apply_root_phase_median_lock_to_payload(
    payload: Mapping[str, Any],
    phases: Sequence[ContactPhase],
    *,
    track_xy_by_key: Mapping[tuple[str, int], Sequence[float]],
    translate_mesh: bool,
    deep_copy_payload: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = _copy_body_payload(payload, deep=deep_copy_payload)
    targets: dict[tuple[str, int], list[float]] = {}
    for phase in phases:
        xys = [
            track_xy_by_key[(str(phase.player_id), int(frame_idx))]
            for frame_idx in phase.frame_indices
            if (str(phase.player_id), int(frame_idx)) in track_xy_by_key
        ]
        if not xys:
            continue
        anchor = [
            _median_sorted(sorted(float(xy[0]) for xy in xys)),
            _median_sorted(sorted(float(xy[1]) for xy in xys)),
        ]
        for frame_idx in phase.frame_indices:
            targets[(str(phase.player_id), int(frame_idx))] = list(anchor)

    corrected_frame_count = 0
    correction_magnitudes: list[float] = []
    for player in output.get("players", []):
        if not isinstance(player, MutableMapping):
            continue
        player_id = str(player.get("id", "unknown"))
        for frame in player.get("frames", []):
            if not isinstance(frame, MutableMapping):
                continue
            frame_idx = int(frame.get("frame_idx", round(float(frame.get("t", 0.0)) * 30.0)))
            target = targets.get((player_id, frame_idx))
            transl = frame.get("transl_world")
            if target is None or not isinstance(transl, Sequence) or isinstance(transl, (str, bytes)) or len(transl) < 3:
                continue
            dx = float(target[0]) - float(transl[0])
            dy = float(target[1]) - float(transl[1])
            if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
                continue
            correction_magnitudes.append(_distance2([0.0, 0.0], [dx, dy]))
            corrected_frame_count += 1
            frame["transl_world"] = [float(target[0]), float(target[1]), float(transl[2])]
            _translate_frame_vectors(frame, "joints_world", dx=dx, dy=dy)
            if translate_mesh:
                _translate_frame_vectors(frame, "mesh_vertices_world", dx=dx, dy=dy)
    return output, {
        "phase_count": len(phases),
        "corrected_frame_count": corrected_frame_count,
        "max_correction_m": max(correction_magnitudes, default=0.0),
        "p95_correction_m": _percentile(correction_magnitudes, 95.0),
    }


def _translate_frame_vectors(frame: MutableMapping[str, Any], field: str, *, dx: float, dy: float) -> None:
    vectors = frame.get(field)
    if not isinstance(vectors, list):
        return
    frame[field] = [[float(vector[0]) + dx, float(vector[1]) + dy, float(vector[2])] for vector in vectors]


def _copy_body_payload(payload: Mapping[str, Any], *, deep: bool) -> dict[str, Any]:
    if deep:
        return copy.deepcopy(dict(payload))
    output = dict(payload)
    players: list[dict[str, Any]] = []
    for player in payload.get("players", []):
        if not isinstance(player, Mapping):
            continue
        player_copy = dict(player)
        frames = player.get("frames")
        if isinstance(frames, list):
            player_copy["frames"] = [dict(frame) if isinstance(frame, Mapping) else frame for frame in frames]
        players.append(player_copy)
    output["players"] = players
    return output


def _apply_footlock_to_player_frames(
    frames: Sequence[dict[str, Any]],
    *,
    max_root_speed_mps: float | None = None,
    xy_translation_enabled: bool = True,
    max_allowed_xy_correction_m: float | None = None,
    smoothing_gap_carry_frames: int = DEFAULT_SMOOTHING_GAP_CARRY_FRAMES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples_by_joint: dict[int, list[FootKinematics]] = {}
    locked_xy_by_joint: dict[int, list[float]] = {}
    snapped_frames: list[dict[str, Any]] = []
    contact_frame_count = 0
    previous_frame_idx: int | None = None
    previous_transl: list[float] | None = None
    previous_t: float | None = None
    previous_relative_by_joint: dict[int, list[float]] = {}
    contact_by_joint: dict[int, bool] = {}
    root_speed_limited_frames = 0
    xy_capped_frames = 0
    max_observed_xy_correction_m = 0.0
    gap_carried_frames = 0
    gap_reset_frames = 0

    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        sparse_output_reset = False
        gap_metadata: dict[str, Any] | None = None
        if previous_frame_idx is not None and frame_idx > previous_frame_idx + 1:
            missing_frames = frame_idx - previous_frame_idx - 1
            gap_metadata = {
                "previous_frame_idx": previous_frame_idx,
                "frame_idx": frame_idx,
                "missing_frame_count": missing_frames,
                "carry_limit_frames": smoothing_gap_carry_frames,
            }
            if missing_frames <= smoothing_gap_carry_frames:
                gap_carried_frames += 1
                gap_metadata["status"] = "carried"
            else:
                for joint_idx, locked_xy in list(locked_xy_by_joint.items()):
                    _append_footlock_reset(samples_by_joint, joint_idx, locked_xy)
                locked_xy_by_joint.clear()
                previous_relative_by_joint.clear()
                contact_by_joint.clear()
                sparse_output_reset = True
                gap_reset_frames += 1
                gap_metadata["status"] = "reset"
        previous_frame_idx = frame_idx

        snapped = dict(frame)
        metadata = _temporal_smoothing_metadata(snapped)
        if gap_metadata is not None:
            metadata["gap"] = gap_metadata
            if sparse_output_reset:
                metadata["reset_reason"] = "sparse_output_gap"
        joints = [[float(value) for value in joint] for joint in frame["joints_world"]]
        vertices = [[float(value) for value in vertex] for vertex in frame["vertices_world"]]
        transl = [float(value) for value in frame["transl_world"]]
        frame_t = float(frame["t"])
        dt = frame_t - previous_t if previous_t is not None else 0.0
        confidence = float(frame.get("confidence", 0.0))
        contact_sample_count = 0
        contact_joint_indices: list[int] = []
        relative_by_joint: dict[int, list[float]] = {}
        for joint_idx, joint in enumerate(joints):
            relative = [joint[0] - transl[0], joint[1] - transl[1], joint[2]]
            relative_by_joint[joint_idx] = relative
            previous_relative = previous_relative_by_joint.get(joint_idx)
            relative_speed = (
                _distance3(previous_relative, relative)
                / dt
                if previous_relative is not None and dt > 0.0
                else 0.0
            )
            contact = classify_contact(
                FootContactObservation(
                    height_m=abs(joint[2]),
                    speed_mps=relative_speed,
                    confidence=confidence,
                ),
                previous_contact=contact_by_joint.get(joint_idx, False),
                hysteresis=FOOT_LOCK_CONTACT_HYSTERESIS,
            )
            contact_by_joint[joint_idx] = contact
            if not contact:
                continue
            locked = snap_stance_foot(
                FootKinematics(
                    position_xyz=joint,
                    velocity_xyz=[relative_speed, 0.0, 0.0],
                    contact=True,
                    frame_index=frame_idx,
                ),
                court_z_m=0.0,
            )
            joints[joint_idx] = locked.position_xyz
            contact_joint_indices.append(joint_idx)
            contact_sample_count += 1

        can_pin_contact_xy = xy_translation_enabled and bool(contact_joint_indices) and len(contact_joint_indices) < len(joints)
        if not can_pin_contact_xy:
            for joint_idx, locked_xy in list(locked_xy_by_joint.items()):
                _append_footlock_reset(samples_by_joint, joint_idx, locked_xy)
            locked_xy_by_joint.clear()

        active_contact_indices = set(contact_joint_indices)
        if can_pin_contact_xy:
            for joint_idx in list(locked_xy_by_joint):
                if joint_idx not in active_contact_indices:
                    _append_footlock_reset(samples_by_joint, joint_idx, locked_xy_by_joint[joint_idx])
                    del locked_xy_by_joint[joint_idx]

        lock_deltas = [
            [
                locked_xy_by_joint[joint_idx][0] - joints[joint_idx][0],
                locked_xy_by_joint[joint_idx][1] - joints[joint_idx][1],
            ]
            for joint_idx in contact_joint_indices
            if can_pin_contact_xy and joint_idx in locked_xy_by_joint
        ]
        if lock_deltas:
            dx = sum(delta[0] for delta in lock_deltas) / len(lock_deltas)
            dy = sum(delta[1] for delta in lock_deltas) / len(lock_deltas)
            max_observed_xy_correction_m = max(max_observed_xy_correction_m, _distance2([0.0, 0.0], [dx, dy]))
            if max_allowed_xy_correction_m is not None:
                dx, dy, capped = _limit_xy_delta(dx, dy, max_magnitude=max_allowed_xy_correction_m)
                if capped:
                    xy_capped_frames += 1
            transl = [transl[0] + dx, transl[1] + dy, transl[2]]
            joints = [[joint[0] + dx, joint[1] + dy, joint[2]] for joint in joints]
            vertices = [[vertex[0] + dx, vertex[1] + dy, vertex[2]] for vertex in vertices]

        for joint_idx in contact_joint_indices:
            if can_pin_contact_xy:
                locked_xy = locked_xy_by_joint.setdefault(joint_idx, [joints[joint_idx][0], joints[joint_idx][1]])
                joints[joint_idx][0] = locked_xy[0]
                joints[joint_idx][1] = locked_xy[1]

        temporal_smoothing_reset = bool(snapped.get("temporal_smoothing_reset")) or sparse_output_reset
        if (
            max_root_speed_mps is not None
            and previous_transl is not None
            and previous_t is not None
            and not temporal_smoothing_reset
        ):
            dt = float(frame["t"]) - previous_t
            if dt > 0.0:
                limited_transl, limited = _limit_step(
                    previous_transl,
                    transl,
                    max_distance=max_root_speed_mps * dt,
                )
                if limited:
                    delta = [limited_transl[idx] - transl[idx] for idx in range(3)]
                    transl = limited_transl
                    joints = [[joint[0] + delta[0], joint[1] + delta[1], joint[2] + delta[2]] for joint in joints]
                    vertices = [
                        [vertex[0] + delta[0], vertex[1] + delta[1], vertex[2] + delta[2]]
                        for vertex in vertices
                    ]
                    root_speed_limited_frames += 1

        previous_transl = list(transl)
        previous_t = frame_t
        previous_relative_by_joint = relative_by_joint

        for joint_idx in contact_joint_indices:
            samples_by_joint.setdefault(joint_idx, []).append(
                FootKinematics(
                    position_xyz=[joints[joint_idx][0], joints[joint_idx][1], joints[joint_idx][2]],
                    velocity_xyz=[0.0, 0.0, 0.0],
                    contact=True,
                    frame_index=frame_idx,
                )
            )

        if contact_sample_count:
            contact_frame_count += 1
        snapped["temporal_smoothing_reset"] = temporal_smoothing_reset
        if metadata:
            snapped["temporal_smoothing_metadata"] = metadata
        snapped["transl_world"] = transl
        snapped["joints_world"] = joints
        snapped["vertices_world"] = vertices
        snapped["foot_lock"] = _infer_floor_contact(snapped)
        snapped_frames.append(snapped)

    joint_metrics = [foot_lock_metrics(samples, court_z_m=0.0) for samples in samples_by_joint.values()]
    max_slide_m = max((metric.max_slide_m for metric in joint_metrics), default=0.0)
    max_penetration_m = max((metric.max_penetration_m for metric in joint_metrics), default=0.0)
    contact_samples = sum(metric.contact_frames for metric in joint_metrics)
    skate_free = (
        contact_frame_count >= 2
        and contact_samples > 0
        and max_slide_m <= FOOT_LOCK_SKATE_FREE_MAX_SLIDE_M
        and max_penetration_m <= 0.0
    )
    return snapped_frames, {
        "scaffold": "cpu_foot_lock_primitives_no_smpl_ik",
        "contact_frames": contact_frame_count,
        "contact_samples": contact_samples,
        "root_speed_limited_frames": root_speed_limited_frames,
        "xy_translation_enabled": bool(xy_translation_enabled),
        "xy_capped_frames": xy_capped_frames,
        "gap_carried_frames": gap_carried_frames,
        "gap_reset_frames": gap_reset_frames,
        "max_xy_correction_m": max_observed_xy_correction_m,
        "max_allowed_xy_correction_m": max_allowed_xy_correction_m if max_allowed_xy_correction_m is not None else 0.0,
        "max_slide_m": max_slide_m,
        "max_penetration_m": max_penetration_m,
        "skate_free": skate_free,
    }


def _append_footlock_reset(
    samples_by_joint: dict[int, list[FootKinematics]],
    joint_idx: int,
    locked_xy: Sequence[float],
) -> None:
    samples_by_joint.setdefault(joint_idx, []).append(
        FootKinematics(
            position_xyz=[float(locked_xy[0]), float(locked_xy[1]), 0.0],
            velocity_xyz=[0.0, 0.0, 0.0],
            contact=False,
        )
    )


def _smpl_foot_lock_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scaffold": str(summary.get("scaffold", "")),
        "contact_frames": int(summary.get("contact_frames", 0)),
        "contact_samples": int(summary.get("contact_samples", 0)),
        "root_speed_limited_frames": int(summary.get("root_speed_limited_frames", 0)),
        "max_slide_m": float(summary.get("max_slide_m", 0.0)),
        "max_penetration_m": float(summary.get("max_penetration_m", 0.0)),
        "skate_free": bool(summary.get("skate_free", False)),
    }


def _temporal_smoothing_metadata(frame: Mapping[str, Any]) -> dict[str, Any]:
    raw = frame.get("temporal_smoothing_metadata")
    return copy.deepcopy(dict(raw)) if isinstance(raw, Mapping) else {}


def _temporal_smoothing_metadata_export(frame: Mapping[str, Any]) -> dict[str, Any]:
    metadata = frame.get("temporal_smoothing_metadata")
    if not isinstance(metadata, Mapping) or not metadata:
        return {}
    return {"temporal_smoothing_metadata": copy.deepcopy(dict(metadata))}


def _detach_smpl_motion_mesh_vertices(smpl_motion: MutableMapping[str, Any]) -> dict[tuple[int, int], Any]:
    vertices_by_key: dict[tuple[int, int], Any] = {}
    for player in smpl_motion.get("players", []) or []:
        if not isinstance(player, Mapping):
            continue
        player_id = int(player.get("id", 0))
        frames = player.get("frames")
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, MutableMapping):
                continue
            frame_idx = int(frame.get("frame_idx", round(float(frame.get("t", 0.0)) * float(smpl_motion.get("fps", 30.0)))))
            vertices_by_key[(player_id, frame_idx)] = frame.pop("mesh_vertices_world", [])
    return vertices_by_key


def _reattach_smpl_motion_mesh_vertices(
    smpl_motion: MutableMapping[str, Any],
    vertices_by_key: Mapping[tuple[int, int], Any],
) -> None:
    for player in smpl_motion.get("players", []) or []:
        if not isinstance(player, Mapping):
            continue
        player_id = int(player.get("id", 0))
        frames = player.get("frames")
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, MutableMapping):
                continue
            frame_idx = int(frame.get("frame_idx", round(float(frame.get("t", 0.0)) * float(smpl_motion.get("fps", 30.0)))))
            frame["mesh_vertices_world"] = vertices_by_key.get((player_id, frame_idx), [])


def _apply_world_joint_visual_smoothing(
    smpl_motion: Mapping[str, Any],
    skeleton3d: Mapping[str, Any],
    *,
    fps: float,
    enabled: bool = True,
    stance_index: Mapping[tuple[int | str, int], Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    smpl_out = copy.deepcopy(dict(smpl_motion))
    skeleton_out = copy.deepcopy(dict(skeleton3d))
    joint_names = _semantic_joint_names(skeleton_out.get("joint_names", []))
    normalized_stance_index = _normalize_stance_index(stance_index or {})
    metrics: dict[str, Any] = {
        "enabled": bool(enabled),
        "filter": "centered_three_tap",
        "weights": list(WORLD_JOINT_VISUAL_SMOOTHING_WEIGHTS),
        "frames_smoothed": 0,
        "stance_lower_body_frames_protected": 0,
        "limb_length_max_delta_m": 0.0,
        "max_lag_frames": 0,
        "max_wrist_peak_delta_frames": 0,
    }
    if not enabled:
        return smpl_out, skeleton_out, metrics

    index_by_name = _semantic_joint_index_by_name(joint_names)
    total_smoothed = 0
    max_limb_delta = 0.0
    max_lag = 0
    for payload in (smpl_out, skeleton_out):
        for player in payload.get("players", []) or []:
            if not isinstance(player, MutableMapping):
                continue
            frames = player.get("frames")
            if not isinstance(frames, list) or len(frames) < 3:
                continue
            smoothed_joints, player_metrics = _smoothed_player_joints(
                frames,
                player_id=int(player.get("id", 0)),
                joint_names=joint_names,
                index_by_name=index_by_name,
                stance_index=normalized_stance_index,
            )
            total_smoothed += int(player_metrics["frames_smoothed"])
            metrics["stance_lower_body_frames_protected"] = int(metrics["stance_lower_body_frames_protected"]) + int(
                player_metrics["stance_lower_body_frames_protected"]
            )
            max_limb_delta = max(max_limb_delta, float(player_metrics["limb_length_max_delta_m"]))
            max_lag = max(max_lag, int(player_metrics["max_lag_frames"]))
            for frame, joints in zip(frames, smoothed_joints, strict=True):
                if joints is not None:
                    frame["joints_world"] = joints

    peak_restore_iterations = _restore_wrist_peak_timing_windows(
        smpl_motion,
        smpl_out,
        skeleton3d,
        skeleton_out,
    )
    wrist_timing = compare_wrist_peak_timing(
        skeleton3d,
        skeleton_out,
        max_allowed_delta_frames=0,
    )
    metrics["frames_smoothed"] = total_smoothed
    metrics["limb_length_max_delta_m"] = max_limb_delta
    metrics["max_lag_frames"] = max_lag
    metrics["wrist_peak_timing"] = wrist_timing
    metrics["max_wrist_peak_delta_frames"] = int(wrist_timing.get("max_abs_delta_frames", 0) or 0)
    metrics["wrist_peak_restore_iterations"] = peak_restore_iterations
    provenance = dict(skeleton_out.get("provenance", {}))
    provenance["worldhmr_visual_smoothing"] = metrics
    skeleton_out["provenance"] = provenance
    return smpl_out, skeleton_out, metrics


def _restore_wrist_peak_timing_windows(
    smpl_original: Mapping[str, Any],
    smpl_out: MutableMapping[str, Any],
    skeleton_original: Mapping[str, Any],
    skeleton_out: MutableMapping[str, Any],
) -> int:
    joint_names = _semantic_joint_names(skeleton_out.get("joint_names", []))
    index_by_name = _semantic_joint_index_by_name(joint_names)
    iterations = 0
    for _attempt in range(4):
        timing = compare_wrist_peak_timing(
            skeleton_original,
            skeleton_out,
            max_allowed_delta_frames=0,
        )
        if int(timing.get("max_abs_delta_frames", 0) or 0) == 0:
            break
        restore_plan: dict[tuple[int, int, int], set[int]] = {}
        for comparison in timing.get("comparisons", []) or []:
            if not isinstance(comparison, Mapping) or not comparison.get("abs_delta_frames"):
                continue
            player_id = int(comparison.get("player_id", 0))
            wrist_name = str(comparison.get("joint_name", ""))
            elbow_name = wrist_name.replace("_wrist", "_elbow")
            joint_indices = {
                idx
                for idx in (index_by_name.get(wrist_name), index_by_name.get(elbow_name))
                if idx is not None
            }
            frame_values = [comparison.get("before_frame"), comparison.get("after_frame")]
            for raw_frame in frame_values:
                if raw_frame is None:
                    continue
                frame_idx = int(raw_frame)
                for neighbor in (frame_idx - 1, frame_idx, frame_idx + 1):
                    for joint_idx in joint_indices:
                        restore_plan.setdefault((player_id, neighbor, joint_idx), set()).add(joint_idx)
        if not restore_plan:
            break
        _restore_joint_samples(skeleton_original, skeleton_out, restore_plan)
        _restore_joint_samples(smpl_original, smpl_out, restore_plan)
        iterations += 1
    return iterations


def _restore_joint_samples(
    original_payload: Mapping[str, Any],
    output_payload: MutableMapping[str, Any],
    restore_plan: Mapping[tuple[int, int, int], set[int]],
) -> None:
    original_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for player in original_payload.get("players", []) or []:
        if not isinstance(player, Mapping):
            continue
        player_id = int(player.get("id", 0))
        for frame in player.get("frames", []) or []:
            if isinstance(frame, Mapping):
                original_by_key[(player_id, int(frame.get("frame_idx", round(float(frame.get("t", 0.0)) * 30.0))))] = frame
    for player in output_payload.get("players", []) or []:
        if not isinstance(player, MutableMapping):
            continue
        player_id = int(player.get("id", 0))
        for frame in player.get("frames", []) or []:
            if not isinstance(frame, MutableMapping):
                continue
            frame_idx = int(frame.get("frame_idx", round(float(frame.get("t", 0.0)) * 30.0)))
            original_frame = original_by_key.get((player_id, frame_idx))
            if original_frame is None:
                continue
            original_joints = _valid_joint_list(original_frame.get("joints_world"))
            output_joints = _valid_joint_list(frame.get("joints_world"))
            if original_joints is None or output_joints is None:
                continue
            changed = False
            for player_key, frame_key, joint_idx in restore_plan:
                if player_key != player_id or frame_key != frame_idx:
                    continue
                if joint_idx < len(original_joints) and joint_idx < len(output_joints):
                    output_joints[joint_idx] = list(original_joints[joint_idx])
                    changed = True
            if changed:
                frame["joints_world"] = output_joints


def _smoothed_player_joints(
    frames: Sequence[Mapping[str, Any]],
    *,
    player_id: int,
    joint_names: Sequence[str],
    index_by_name: Mapping[str, int],
    stance_index: Mapping[tuple[int, int], Mapping[str, Any]],
) -> tuple[list[list[list[float]] | None], dict[str, Any]]:
    original = [_valid_joint_list(frame.get("joints_world")) for frame in frames]
    frame_indices = [int(frame.get("frame_idx", idx)) for idx, frame in enumerate(frames)]
    smoothed: list[list[list[float]] | None] = [copy.deepcopy(joints) if joints is not None else None for joints in original]
    protected_by_pos = _visual_smoothing_protected_joint_positions(
        original,
        frame_indices=frame_indices,
        index_by_name=index_by_name,
    )
    stance_lower_body_frames_protected = _add_stance_lower_body_protected_positions(
        protected_by_pos,
        player_id=player_id,
        frame_indices=frame_indices,
        stance_index=stance_index,
        index_by_name=index_by_name,
    )
    frames_smoothed = 0
    max_limb_delta = 0.0
    max_lag = 0
    for pos in range(1, len(frames) - 1):
        prev_joints = original[pos - 1]
        cur_joints = original[pos]
        next_joints = original[pos + 1]
        if (
            prev_joints is None
            or cur_joints is None
            or next_joints is None
            or len(prev_joints) != len(cur_joints)
            or len(next_joints) != len(cur_joints)
            or frame_indices[pos] != frame_indices[pos - 1] + 1
            or frame_indices[pos + 1] != frame_indices[pos] + 1
        ):
            continue
        smoothed_frame = []
        for joint_idx in range(len(cur_joints)):
            smoothed_frame.append(
                [
                    WORLD_JOINT_VISUAL_SMOOTHING_WEIGHTS[0] * prev_joints[joint_idx][axis]
                    + WORLD_JOINT_VISUAL_SMOOTHING_WEIGHTS[1] * cur_joints[joint_idx][axis]
                    + WORLD_JOINT_VISUAL_SMOOTHING_WEIGHTS[2] * next_joints[joint_idx][axis]
                    for axis in range(3)
                ]
            )
        max_limb_delta = max(
            max_limb_delta,
            _preserve_visual_smoothing_limb_lengths(
                original_joints=cur_joints,
                smoothed_joints=smoothed_frame,
                index_by_name=index_by_name,
            ),
        )
        for joint_idx in protected_by_pos.get(pos, set()):
            if joint_idx < len(smoothed_frame):
                smoothed_frame[joint_idx] = list(cur_joints[joint_idx])
        smoothed[pos] = smoothed_frame
        frames_smoothed += 1

    for joint_name in ("left_wrist", "right_wrist", "left_ankle", "right_ankle", "left_heel", "right_heel"):
        joint_idx = index_by_name.get(joint_name)
        if joint_idx is None:
            continue
        original_signal = [joints[joint_idx][0] for joints in original if joints is not None and joint_idx < len(joints)]
        smoothed_signal = [joints[joint_idx][0] for joints in smoothed if joints is not None and joint_idx < len(joints)]
        if len(original_signal) == len(smoothed_signal) and len(original_signal) >= 3:
            max_lag = max(
                max_lag,
                abs(estimate_integer_lag_frames(original_signal, smoothed_signal, max_lag_frames=1)),
            )
    return smoothed, {
        "frames_smoothed": frames_smoothed,
        "stance_lower_body_frames_protected": stance_lower_body_frames_protected,
        "limb_length_max_delta_m": max_limb_delta,
        "max_lag_frames": max_lag,
    }


def _add_stance_lower_body_protected_positions(
    protected: dict[int, set[int]],
    *,
    player_id: int,
    frame_indices: Sequence[int],
    stance_index: Mapping[tuple[int, int], Mapping[str, Any]],
    index_by_name: Mapping[str, int],
) -> int:
    protected_joint_names = (
        "left_hip",
        "left_knee",
        "left_ankle",
        "left_big_toe_tip",
        "left_small_toe_tip",
        "left_heel",
        "right_hip",
        "right_knee",
        "right_ankle",
        "right_big_toe_tip",
        "right_small_toe_tip",
        "right_heel",
    )
    protected_indices = {
        index_by_name[name]
        for name in protected_joint_names
        if name in index_by_name
    }
    if not protected_indices:
        return 0
    protected_frame_count = 0
    for pos, frame_idx in enumerate(frame_indices):
        if bool(stance_index.get((int(player_id), int(frame_idx)), {}).get("stance", False)):
            protected.setdefault(pos, set()).update(protected_indices)
            protected_frame_count += 1
    return protected_frame_count


def _visual_smoothing_protected_joint_positions(
    original: Sequence[list[list[float]] | None],
    *,
    frame_indices: Sequence[int],
    index_by_name: Mapping[str, int],
) -> dict[int, set[int]]:
    protected: dict[int, set[int]] = {}
    frame_pos_by_idx = {int(frame_idx): pos for pos, frame_idx in enumerate(frame_indices)}
    for wrist_name, elbow_name in (("left_wrist", "left_elbow"), ("right_wrist", "right_elbow")):
        wrist_idx = index_by_name.get(wrist_name)
        elbow_idx = index_by_name.get(elbow_name)
        if wrist_idx is None or elbow_idx is None:
            continue
        for peak_frame in _top_joint_speed_peak_frames(
            original,
            frame_indices=frame_indices,
            joint_idx=wrist_idx,
            top_k=5,
            min_peak_speed_mps=4.0,
            fps=30.0,
        ):
            for frame_idx in (peak_frame - 1, peak_frame):
                pos = frame_pos_by_idx.get(frame_idx)
                if pos is not None:
                    protected.setdefault(pos, set()).update({wrist_idx, elbow_idx})
    return protected


def _top_joint_speed_peak_frames(
    frames: Sequence[list[list[float]] | None],
    *,
    frame_indices: Sequence[int],
    joint_idx: int,
    top_k: int,
    min_peak_speed_mps: float,
    fps: float,
) -> list[int]:
    speeds: list[tuple[int, float]] = []
    for pos, (previous, current) in enumerate(zip(frames, frames[1:], strict=False), start=1):
        if previous is None or current is None or joint_idx >= len(previous) or joint_idx >= len(current):
            continue
        frame_delta = max(int(frame_indices[pos]) - int(frame_indices[pos - 1]), 1)
        displacement = _distance3(previous[joint_idx], current[joint_idx])
        if displacement > 0.5:
            continue
        speed = displacement / (frame_delta / fps)
        if speed >= min_peak_speed_mps:
            speeds.append((int(frame_indices[pos]), speed))
    selected: list[int] = []
    for frame_idx, _speed in sorted(speeds, key=lambda item: item[1], reverse=True):
        if any(abs(frame_idx - kept) <= 1 for kept in selected):
            continue
        selected.append(frame_idx)
        if len(selected) >= top_k:
            break
    return selected


def _preserve_visual_smoothing_limb_lengths(
    *,
    original_joints: Sequence[Sequence[float]],
    smoothed_joints: list[list[float]],
    index_by_name: Mapping[str, int],
) -> float:
    max_delta = 0.0
    for parent_name, child_name in WORLD_JOINT_VISUAL_SMOOTHING_BONE_PAIRS:
        parent_idx = index_by_name.get(parent_name)
        child_idx = index_by_name.get(child_name)
        if parent_idx is None or child_idx is None or parent_idx >= len(smoothed_joints) or child_idx >= len(smoothed_joints):
            continue
        original_length = _distance3(original_joints[parent_idx], original_joints[child_idx])
        if original_length <= 0.0:
            continue
        current = smoothed_joints[child_idx]
        parent = smoothed_joints[parent_idx]
        direction = [current[axis] - parent[axis] for axis in range(3)]
        direction_length = sqrt(sum(axis * axis for axis in direction))
        if direction_length <= 1e-12:
            original_direction = [original_joints[child_idx][axis] - original_joints[parent_idx][axis] for axis in range(3)]
            direction_length = sqrt(sum(axis * axis for axis in original_direction))
            if direction_length <= 1e-12:
                continue
            direction = original_direction
        scale = original_length / direction_length
        smoothed_joints[child_idx] = [parent[axis] + direction[axis] * scale for axis in range(3)]
        max_delta = max(max_delta, abs(_distance3(smoothed_joints[parent_idx], smoothed_joints[child_idx]) - original_length))
    return max_delta


def _valid_joint_list(value: Any) -> list[list[float]] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    joints: list[list[float]] = []
    for joint in value:
        if not isinstance(joint, Sequence) or isinstance(joint, (str, bytes)) or len(joint) < 3:
            return None
        try:
            point = [float(joint[0]), float(joint[1]), float(joint[2])]
        except (TypeError, ValueError):
            return None
        if not all(isfinite(axis) for axis in point):
            return None
        joints.append(point)
    return joints


def _semantic_joint_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, Sequence) and not isinstance(raw_names, (str, bytes)):
        names = [str(name) for name in raw_names]
    else:
        names = []
    if len(names) == len(MHR70_JOINT_NAMES) and all(name.startswith("sam3dbody_joint_") for name in names):
        return list(MHR70_JOINT_NAMES)
    return names


def _semantic_joint_index_by_name(joint_names: Sequence[str]) -> dict[str, int]:
    direct = {str(name): idx for idx, name in enumerate(joint_names)}
    if len(joint_names) == len(MHR70_JOINT_NAMES):
        for idx, name in enumerate(MHR70_JOINT_NAMES):
            direct.setdefault(str(name), idx)
    return direct


def _distance3(left: Sequence[float], right: Sequence[float]) -> float:
    return sqrt(sum((left[idx] - right[idx]) ** 2 for idx in range(3)))


def _rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sqrt(sum(value * value for value in values) / len(values))


def _validate_vector3(values: Sequence[float], *, name: str) -> None:
    if len(values) != 3:
        raise ValueError(f"{name} must be a 3-vector")


def _mesh_faces(values: Any, *, vertex_count: int) -> list[list[int]]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("mesh_faces must be a sequence of triangle index triples")
    faces: list[list[int]] = []
    for face_index, face in enumerate(values):
        if isinstance(face, (str, bytes)) or not isinstance(face, Sequence) or len(face) != 3:
            raise ValueError(f"mesh_faces/{face_index} must be a triangle index triple")
        parsed_face: list[int] = []
        for raw_index in face:
            if isinstance(raw_index, bool):
                raise ValueError(f"mesh_faces/{face_index} must be a triangle index triple")
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"mesh_faces/{face_index} must be a triangle index triple") from exc
            if index < 0:
                raise ValueError(f"mesh_faces/{face_index} must be a triangle index triple")
            if index >= vertex_count:
                raise ValueError(f"mesh_faces/{face_index} index {index} is outside vertices_camera")
            parsed_face.append(index)
        faces.append(parsed_face)
    return faces


def _common_mesh_faces(frames: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    canonical: list[list[int]] | None = None
    for frame in frames:
        faces = frame.get("mesh_faces", [])
        if not faces:
            continue
        face_rows = [[int(index) for index in face] for face in faces]
        if canonical is None:
            canonical = face_rows
            continue
        if canonical != face_rows:
            raise ValueError("Fast SAM-3D-Body samples produced inconsistent mesh_faces")
    return canonical or []


def _load_camera_motion_context(path: str | Path | None) -> tuple[_CameraMotionContext | None, list[str]]:
    motion_path = Path(path) if path is not None else Path.cwd() / CAMERA_MOTION_ARTIFACT
    if not motion_path.is_file():
        return None, []
    try:
        payload = json.loads(motion_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("payload must be a JSON object")
        raw_frames = payload.get("frames", [])
        if not isinstance(raw_frames, Sequence) or isinstance(raw_frames, (str, bytes)):
            raise ValueError("frames must be a sequence")
        frames: dict[int, _CameraMotionObservation] = {}
        artifact_frame_count = 0
        artifact_compensated_frame_count = 0
        fps = float(payload.get("fps") or 30.0)
        for item in raw_frames:
            if not isinstance(item, Mapping):
                continue
            frame_idx = int(item.get("frame_idx", round(float(item.get("t", 0.0)) * fps)))
            artifact_frame_count += 1
            if not bool(item.get("compensated", False)):
                continue
            frames[frame_idx] = _CameraMotionObservation(
                matrix=_matrix3(item.get("M"), name=f"camera_motion.frames[{frame_idx}].M")
            )
            artifact_compensated_frame_count += 1
        return (
            _CameraMotionContext(
                path=motion_path,
                frames=frames,
                artifact_frame_count=artifact_frame_count,
                artifact_compensated_frame_count=artifact_compensated_frame_count,
            ),
            [],
        )
    except Exception as exc:
        return None, [f"ignored malformed camera_motion.json ({motion_path.name}): {exc}"]


def _camera_motion_artifact_summary(
    camera_motion: _CameraMotionContext | None,
    *,
    warnings: Sequence[str],
    frames_used: int,
    frames_uncompensated: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if camera_motion is None and not warnings:
        return {}, {}
    if camera_motion is None:
        summary = {
            "camera_motion_status": "ignored_malformed",
            "camera_motion_frames_used": 0,
            "camera_motion_frames_uncompensated": 0,
            "camera_motion_warnings": list(warnings),
        }
        provenance = {
            "status": "ignored_malformed",
            "frames_used": 0,
            "frames_uncompensated": 0,
            "warnings": list(warnings),
        }
        return summary, provenance
    summary = {
        "camera_motion_status": "used",
        "camera_motion_path": camera_motion.path.name,
        "camera_motion_frames_used": int(frames_used),
        "camera_motion_frames_uncompensated": int(frames_uncompensated),
        "camera_motion_artifact_frame_count": int(camera_motion.artifact_frame_count),
        "camera_motion_artifact_compensated_frame_count": int(camera_motion.artifact_compensated_frame_count),
    }
    provenance = {
        "status": "used",
        "path": camera_motion.path.name,
        "frames_used": int(frames_used),
        "frames_uncompensated": int(frames_uncompensated),
        "artifact_frame_count": int(camera_motion.artifact_frame_count),
        "artifact_compensated_frame_count": int(camera_motion.artifact_compensated_frame_count),
        "scope": "pixel_to_world_grounding_inputs_only",
    }
    return summary, provenance


def _apply_camera_motion_grounding_correction(
    joints_world_raw: Sequence[Sequence[float]],
    *,
    sample: Mapping[str, Any],
    calibration: CourtCalibration,
    camera_motion: _CameraMotionObservation | None,
) -> list[list[float]]:
    corrected = [[float(value) for value in joint] for joint in joints_world_raw]
    if camera_motion is None or not corrected:
        return corrected
    homography_inv = _invert_matrix3(_matrix3(calibration.homography, name="calibration.homography"))
    corrected_indices: set[int] = set()
    for item in sample.get("pred_foot_keypoints_2d", []) or []:
        if not isinstance(item, Mapping):
            continue
        try:
            joint_index = int(item.get("index"))
        except (TypeError, ValueError):
            name = str(item.get("name", ""))
            if name not in SAM3D_FOOT_KEYPOINT_INDICES:
                continue
            joint_index = SAM3D_FOOT_KEYPOINT_INDICES[name]
        if joint_index < 0 or joint_index >= len(corrected):
            continue
        xy_px = _vector2(item.get("xy_px"), name=f"pred_foot_keypoints_2d/{joint_index}/xy_px")
        dx, dy = _camera_motion_world_delta(xy_px, homography_inv=homography_inv, camera_motion=camera_motion)
        corrected[joint_index][0] += dx
        corrected[joint_index][1] += dy
        corrected_indices.add(joint_index)
    if corrected_indices:
        return corrected
    bbox = sample.get("bbox_xyxy")
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) < 4:
        return corrected
    bbox_xyxy = _float_list(bbox[:4], name="bbox_xyxy")
    foot_pixel = [(bbox_xyxy[0] + bbox_xyxy[2]) / 2.0, bbox_xyxy[3]]
    dx, dy = _camera_motion_world_delta(foot_pixel, homography_inv=homography_inv, camera_motion=camera_motion)
    for joint_index in _low_joint_indices(corrected):
        corrected[joint_index][0] += dx
        corrected[joint_index][1] += dy
    return corrected


def _camera_motion_world_delta(
    pixel_xy: Sequence[float],
    *,
    homography_inv: Sequence[Sequence[float]],
    camera_motion: _CameraMotionObservation,
) -> tuple[float, float]:
    static_world = _apply_homography_xy(homography_inv, pixel_xy)
    reference_pixel = _apply_homography_xy(camera_motion.matrix, pixel_xy)
    reference_world = _apply_homography_xy(homography_inv, reference_pixel)
    return reference_world[0] - static_world[0], reference_world[1] - static_world[1]


def _low_joint_indices(joints_world: Sequence[Sequence[float]]) -> list[int]:
    if not joints_world:
        return []
    min_z = min(float(joint[2]) for joint in joints_world)
    return [
        index
        for index, joint in enumerate(joints_world)
        if float(joint[2]) <= min_z + LOW_GROUNDING_ANCHOR_HEIGHT_M
    ]


def _ground_fast_sam_sample(
    sample: Mapping[str, Any],
    *,
    calibration: CourtCalibration,
    camera_motion: _CameraMotionObservation | None = None,
) -> dict[str, Any]:
    joints_camera = _vector3_list(sample.get("joints_camera"), name="joints_camera")
    vertices_camera = _vector3_list(sample.get("vertices_camera", []), name="vertices_camera")
    if not joints_camera and not vertices_camera:
        raise ValueError("Fast SAM-3D-Body sample must include joints_camera or vertices_camera")
    track_world_xy = _vector2(sample.get("track_world_xy"), name="track_world_xy")
    t = float(sample["t"])
    confidence = float(sample.get("confidence", 0.0))
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be between 0 and 1")

    joints_world_raw = _camera_offsets_to_world(joints_camera, calibration=calibration)
    joints_world_raw = _apply_camera_motion_grounding_correction(
        joints_world_raw,
        sample=sample,
        calibration=calibration,
        camera_motion=camera_motion,
    )
    vertices_world_raw = _camera_offsets_to_world(
        vertices_camera,
        calibration=calibration,
        root_camera=joints_camera[0] if joints_camera else (vertices_camera[0] if vertices_camera else [0.0, 0.0, 0.0]),
    )
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
        # ADDITIVE (P2-2 GATE 1b, w5_p22latent_20260707): MHR scale_params
        # (28-dim bone-length/proportions correction), previously dropped
        # end-to-end -- see hmr_deep.py::normalize_fast_sam_body_output for
        # the producer-side note and measured decode-error evidence.
        "scale": _float_list(sample.get("scale", []), name="scale"),
        "transl_world": [track_world_xy[0], track_world_xy[1], 0.0],
        "joints_world": _translate_points(joints_world_raw or vertices_world_raw, dx=dx, dy=dy, dz=dz),
        "vertices_world": _translate_points(vertices_world_raw, dx=dx, dy=dy, dz=dz),
        "mesh_faces": _mesh_faces(sample.get("mesh_faces", []), vertex_count=len(vertices_camera)),
    }


def _camera_offsets_to_world(
    points_camera_relative: Sequence[Sequence[float]],
    *,
    calibration: CourtCalibration,
    root_camera: Sequence[float] | None = None,
) -> list[list[float]]:
    if not points_camera_relative:
        return []
    joint_names = [f"sam3dbody_joint_{idx:03d}" for idx in range(len(points_camera_relative))]
    if root_camera is None:
        return rotate_camera_offsets_row_times_R(
            points_camera_relative,
            rotation=calibration.extrinsics.R,
            joint_names=joint_names,
        )
    root = [float(root_camera[idx]) for idx in range(3)]
    rotation = [[float(value) for value in row] for row in calibration.extrinsics.R]
    rotated: list[list[float]] = []
    for point in points_camera_relative:
        offset = [float(point[idx]) - root[idx] for idx in range(3)]
        world_offset = [
            sum(offset[row_idx] * rotation[row_idx][col_idx] for row_idx in range(3))
            for col_idx in range(3)
        ]
        rotated.append([root[idx] + world_offset[idx] for idx in range(3)])
    return rotated


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


def _smooth_grounded_frames_stance_aware(
    frames: Sequence[dict[str, Any]],
    *,
    stance_index: Mapping[tuple[int | str, int], Mapping[str, Any]],
    fps: float,
    max_root_speed_mps: float | None,
    max_track_anchor_smoothing_residual_m: float | None,
    smoothing_residual_identity_reset_m: float = DEFAULT_SMOOTHING_RESIDUAL_IDENTITY_RESET_M,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_index, rejected_stance_reasons = _lock_eligible_stance_index(_normalize_stance_index(stance_index))
    transition_frames = _transition_frame_keys(frames, stance_index=normalized_index)
    phase_anchor_targets, phase_anchor_split_count = _stance_phase_anchor_targets(
        frames,
        stance_index=normalized_index,
        max_anchor_drift_m=STANCE_AWARE_PHASE_ANCHOR_DRIFT_SPLIT_M,
    )
    previous_by_player: dict[int, list[float]] = {}
    previous_t_by_player: dict[int, float] = {}
    previous_track_by_player: dict[int, list[float]] = {}
    smoothed: list[dict[str, Any]] = []
    track_anchor_residuals: list[float] = []
    transition_anchor_residuals: list[float] = []
    pre_reset_track_anchor_residuals: list[float] = []
    body_marker_transition_divergence: list[float] = []
    track_anchor_residual_reset_frames = 0
    stance_frame_count = 0
    transition_frame_count = 0
    root_speed_anomaly_frames = 0
    root_speed_anomaly_by_player: dict[int, int] = {}
    root_step_count_by_player: dict[int, int] = {}
    root_speed_limited_frames = 0
    rejected_stance_frame_count = sum(rejected_stance_reasons.values())
    track_anchor_residual_carried_frames = 0
    track_anchor_residual_identity_reset_frames = 0
    residual_reset_m = (
        min(max_track_anchor_smoothing_residual_m, STANCE_AWARE_TRANSITION_RESIDUAL_RESET_M)
        if max_track_anchor_smoothing_residual_m is not None
        else STANCE_AWARE_TRANSITION_RESIDUAL_RESET_M
    )

    for frame in frames:
        player_id = int(frame["player_id"])
        frame_idx = int(frame["frame_idx"])
        key = (player_id, frame_idx)
        info = normalized_index.get(key, {})
        is_stance = bool(info.get("stance", False))
        is_transition = key in transition_frames
        if is_stance:
            stance_frame_count += 1
        elif is_transition:
            transition_frame_count += 1
        previous = previous_by_player.get(player_id)
        previous_t = previous_t_by_player.get(player_id)
        previous_track = previous_track_by_player.get(player_id)
        transl = [float(value) for value in frame["transl_world"]]
        track_xy = [float(value) for value in frame["track_world_xy"]]
        frame_t = float(frame["t"])
        temporal_smoothing_reset = False
        reset_reason = ""
        residual_status = "ok"
        residual_threshold_m = STANCE_AWARE_STANCE_RESIDUAL_RESET_M if is_stance else residual_reset_m
        root_speed_limited_this_frame = False

        if is_stance:
            phase_target_xy = phase_anchor_targets.get(key, track_xy)
            pre_reset_residual = _distance2(previous[:2], phase_target_xy) if previous is not None else 0.0
            smoothed_xy = phase_target_xy
            would_reset = previous is not None and pre_reset_residual > STANCE_AWARE_STANCE_RESIDUAL_RESET_M
            if would_reset:
                if pre_reset_residual > smoothing_residual_identity_reset_m:
                    temporal_smoothing_reset = True
                    reset_reason = "stance_anchor_identity_reset"
                    track_anchor_residual_reset_frames += 1
                    track_anchor_residual_identity_reset_frames += 1
                    residual_status = "reset"
                else:
                    track_anchor_residual_carried_frames += 1
                    residual_status = "carried"
        elif previous is None:
            smoothed_xy = track_xy
            pre_reset_residual = 0.0
        else:
            dt = _positive_dt(frame_t, previous_t, fps=fps)
            velocity = _placement_velocity_xy(info, current_track_xy=track_xy, previous_track_xy=previous_track, dt=dt)
            alpha = (
                STANCE_AWARE_TRANSITION_FALLBACK_ALPHA_XY
                if velocity is None or _placement_covariance_is_high(info)
                else STANCE_AWARE_TRANSITION_ALPHA_XY
            )
            if velocity is None:
                predicted_xy = previous[:2]
            else:
                predicted_xy = [previous[0] + velocity[0] * dt, previous[1] + velocity[1] * dt]
            smoothed_xy = [
                alpha * track_xy[idx] + (1.0 - alpha) * predicted_xy[idx]
                for idx in range(2)
            ]
            pre_reset_residual = _distance2(smoothed_xy, track_xy)
            if pre_reset_residual > residual_reset_m:
                if pre_reset_residual > smoothing_residual_identity_reset_m:
                    smoothed_xy = track_xy
                    temporal_smoothing_reset = True
                    reset_reason = "transition_anchor_identity_reset"
                    track_anchor_residual_reset_frames += 1
                    track_anchor_residual_identity_reset_frames += 1
                    residual_status = "reset"
                else:
                    track_anchor_residual_carried_frames += 1
                    residual_status = "carried"

        smoothed_transl = [smoothed_xy[0], smoothed_xy[1], transl[2]]
        if previous is not None and max_root_speed_mps is not None and previous_t is not None:
            dt = _positive_dt(frame_t, previous_t, fps=fps)
            if dt > 0.0:
                root_step_count_by_player[player_id] = root_step_count_by_player.get(player_id, 0) + 1
                if _distance2(previous[:2], smoothed_transl[:2]) > max_root_speed_mps * dt + 1e-12:
                    root_speed_anomaly_frames += 1
                    root_speed_anomaly_by_player[player_id] = root_speed_anomaly_by_player.get(player_id, 0) + 1
        previous_by_player[player_id] = smoothed_transl
        previous_t_by_player[player_id] = frame_t
        previous_track_by_player[player_id] = track_xy
        delta = [smoothed_transl[idx] - transl[idx] for idx in range(3)]
        residual = _distance2(smoothed_transl[:2], track_xy)
        pre_reset_track_anchor_residuals.append(pre_reset_residual)
        track_anchor_residuals.append(residual)
        if is_transition:
            transition_anchor_residuals.append(residual)
            body_marker_transition_divergence.append(residual)
        smoothed_frame = dict(frame)
        smoothed_frame["grounding_anchor"] = frame.get("grounding_anchor", "")
        smoothed_frame["transl_world"] = smoothed_transl
        smoothed_frame["temporal_smoothing_reset"] = temporal_smoothing_reset
        metadata = _temporal_smoothing_metadata(frame)
        if residual_status != "ok":
            metadata["residual"] = {
                "status": residual_status,
                "reason": reset_reason or "stance_anchor_residual",
                "pre_reset_anchor_residual_m": pre_reset_residual,
                "threshold_m": residual_threshold_m,
                "identity_reset_m": smoothing_residual_identity_reset_m,
            }
        if reset_reason:
            metadata["reset_reason"] = reset_reason
        if root_speed_limited_this_frame:
            metadata["root_speed_limited"] = True
        if metadata:
            smoothed_frame["temporal_smoothing_metadata"] = metadata
        smoothed_frame["stance_aware_grounding"] = {
            "stance": is_stance,
            "transition": is_transition,
            "anchor_residual_m": residual,
            "pre_reset_anchor_residual_m": pre_reset_residual,
        }
        smoothed_frame["joints_world"] = [
            [joint[0] + delta[0], joint[1] + delta[1], joint[2] + delta[2]]
            for joint in frame["joints_world"]
        ]
        smoothed_frame["vertices_world"] = [
            [vertex[0] + delta[0], vertex[1] + delta[1], vertex[2] + delta[2]]
            for vertex in frame["vertices_world"]
        ]
        smoothed.append(smoothed_frame)

    clamp_fraction_by_player = {
        str(player_id): (
            root_speed_anomaly_by_player.get(player_id, 0) / step_count
            if step_count
            else 0.0
        )
        for player_id, step_count in sorted(root_step_count_by_player.items())
    }
    total_steps = sum(root_step_count_by_player.values())
    root_speed_engagement_overall = root_speed_anomaly_frames / total_steps if total_steps else 0.0
    transition_summary = _distribution_m(transition_anchor_residuals)
    divergence_summary = _distribution_m(body_marker_transition_divergence)
    return smoothed, {
        "root_speed_limited_frames": root_speed_limited_frames,
        "root_speed_anomaly_frames": root_speed_anomaly_frames,
        "root_speed_clamp_engagement_overall": root_speed_engagement_overall,
        "root_speed_anomaly_fraction_overall": root_speed_engagement_overall,
        "root_speed_clamp_engagement_by_player": clamp_fraction_by_player,
        "root_speed_anomaly_fraction_by_player": clamp_fraction_by_player,
        "track_anchor_residual_reset_frames": track_anchor_residual_reset_frames,
        "track_anchor_residual_carried_frames": track_anchor_residual_carried_frames,
        "track_anchor_residual_identity_reset_frames": track_anchor_residual_identity_reset_frames,
        "max_pre_reset_track_anchor_residual_m": max(pre_reset_track_anchor_residuals, default=0.0),
        "max_track_anchor_residual_m": max(track_anchor_residuals, default=0.0),
        "transition_anchor_lag_p95_m": transition_summary["p95"],
        "transition_anchor_lag_median_m": transition_summary["p50"],
        "body_marker_transition_divergence_p90_m": divergence_summary["p90"],
        "stance_aware_grounding": {
            "source": R3_GROUNDING_ANCHOR_SOURCE,
            "alpha_stance_xy": STANCE_AWARE_STANCE_ALPHA_XY,
            "alpha_transition_xy": STANCE_AWARE_TRANSITION_ALPHA_XY,
            "alpha_transition_fallback_xy": STANCE_AWARE_TRANSITION_FALLBACK_ALPHA_XY,
            "phase_anchor_drift_split_m": STANCE_AWARE_PHASE_ANCHOR_DRIFT_SPLIT_M,
            "phase_anchor_split_count": phase_anchor_split_count,
            "phase_median_anchor_frame_count": len(phase_anchor_targets),
            "stance_frame_count": stance_frame_count,
            "rejected_stance_frame_count": rejected_stance_frame_count,
            "rejected_stance_reasons": rejected_stance_reasons,
            "transition_frame_count": transition_frame_count,
            "track_anchor_residual_m": _distribution_m(track_anchor_residuals),
            "transition_anchor_lag_m": transition_summary,
            "transition_anchor_lag_p95_m": transition_summary["p95"],
            "transition_anchor_lag_median_m": transition_summary["p50"],
            "body_marker_transition_divergence": divergence_summary,
            "residual_reset_frames": track_anchor_residual_reset_frames,
            "residual_carried_frames": track_anchor_residual_carried_frames,
            "residual_identity_reset_frames": track_anchor_residual_identity_reset_frames,
            "root_speed_clamp_engagement_overall": root_speed_engagement_overall,
            "root_speed_clamp_engagement_by_player": clamp_fraction_by_player,
        },
    }


def _smooth_grounded_frames(
    frames: Sequence[dict[str, Any]],
    *,
    alpha: float,
    max_root_speed_mps: float | None,
    max_track_anchor_smoothing_residual_m: float | None,
    smoothing_residual_identity_reset_m: float,
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
    track_anchor_residual_carried_frames = 0
    track_anchor_residual_identity_reset_frames = 0
    for frame in frames:
        player_id = int(frame["player_id"])
        previous = previous_by_player.get(player_id)
        previous_t = previous_t_by_player.get(player_id)
        transl = [float(value) for value in frame["transl_world"]]
        if previous is None:
            smoothed_transl = transl
            root_speed_limited_this_frame = False
        else:
            smoothed_transl = [
                alpha * transl[idx] + (1.0 - alpha) * previous[idx]
                for idx in range(3)
            ]
            root_speed_limited_this_frame = False
            if max_root_speed_mps is not None and previous_t is not None:
                dt = max(float(frame["t"]) - previous_t, 0.0)
                if dt > 0.0:
                    smoothed_transl, limited = _limit_step(previous, smoothed_transl, max_distance=max_root_speed_mps * dt)
                    if limited:
                        root_speed_limited_frames += 1
                        root_speed_limited_this_frame = True
        track_xy = [float(value) for value in frame["track_world_xy"]]
        pre_reset_residual = _distance2(smoothed_transl[:2], track_xy)
        pre_reset_track_anchor_residuals.append(pre_reset_residual)
        temporal_smoothing_reset = False
        metadata = _temporal_smoothing_metadata(frame)
        if (
            previous is not None
            and max_track_anchor_smoothing_residual_m is not None
            and pre_reset_residual > max_track_anchor_smoothing_residual_m
        ):
            residual_metadata = {
                "pre_reset_anchor_residual_m": pre_reset_residual,
                "threshold_m": max_track_anchor_smoothing_residual_m,
                "identity_reset_m": smoothing_residual_identity_reset_m,
            }
            if pre_reset_residual > smoothing_residual_identity_reset_m:
                smoothed_transl = transl
                temporal_smoothing_reset = True
                track_anchor_residual_reset_frames += 1
                track_anchor_residual_identity_reset_frames += 1
                residual_metadata["status"] = "reset"
                residual_metadata["reason"] = "track_anchor_identity_reset"
                metadata["reset_reason"] = "track_anchor_identity_reset"
            else:
                track_anchor_residual_carried_frames += 1
                residual_metadata["status"] = "carried"
                residual_metadata["reason"] = "track_anchor_residual"
            metadata["residual"] = residual_metadata
        if root_speed_limited_this_frame:
            metadata["root_speed_limited"] = True
        previous_by_player[player_id] = smoothed_transl
        previous_t_by_player[player_id] = float(frame["t"])
        delta = [smoothed_transl[idx] - transl[idx] for idx in range(3)]
        track_anchor_residuals.append(_distance2(smoothed_transl[:2], track_xy))
        smoothed_frame = dict(frame)
        smoothed_frame["grounding_anchor"] = frame.get("grounding_anchor", "")
        smoothed_frame["transl_world"] = smoothed_transl
        smoothed_frame["temporal_smoothing_reset"] = temporal_smoothing_reset
        if metadata:
            smoothed_frame["temporal_smoothing_metadata"] = metadata
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
        "track_anchor_residual_carried_frames": track_anchor_residual_carried_frames,
        "track_anchor_residual_identity_reset_frames": track_anchor_residual_identity_reset_frames,
        "max_pre_reset_track_anchor_residual_m": max(pre_reset_track_anchor_residuals, default=0.0),
        "max_track_anchor_residual_m": max(track_anchor_residuals, default=0.0),
    }


def _normalize_stance_index(
    stance_index: Mapping[tuple[int | str, int], Mapping[str, Any]],
) -> dict[tuple[int, int], Mapping[str, Any]]:
    normalized: dict[tuple[int, int], Mapping[str, Any]] = {}
    for key, value in stance_index.items():
        if not isinstance(value, Mapping):
            continue
        try:
            player_id, frame_idx = key
            normalized[(int(player_id), int(frame_idx))] = value
        except (TypeError, ValueError):
            continue
    return normalized


def _lock_eligible_stance_index(
    stance_index: Mapping[tuple[int, int], Mapping[str, Any]],
) -> tuple[dict[tuple[int, int], Mapping[str, Any]], dict[str, int]]:
    filtered: dict[tuple[int, int], Mapping[str, Any]] = {}
    rejected_reasons: dict[str, int] = {}
    for key, info in stance_index.items():
        if not bool(info.get("stance", False)):
            filtered[key] = info
            continue
        reason = _stance_info_rejection_reason(info)
        if reason is None:
            filtered[key] = info
            continue
        rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
        filtered[key] = {**dict(info), "stance": False, "rejected_stance_reason": reason}
    return filtered, rejected_reasons


def _stance_info_rejection_reason(info: Mapping[str, Any]) -> str | None:
    if bool(info.get("weak", False)):
        return str(info.get("rejection_reason") or "weak_phase")
    if bool(info.get("demoted", False)):
        return str(info.get("rejection_reason") or "demoted_phase")
    foot_assignment = str(info.get("foot_assignment", ""))
    if foot_assignment == "bilateral_from_player_stance":
        return str(info.get("rejection_reason") or "weak_bilateral_unknown_foot")
    if foot_assignment not in {"per_foot_keypoint_support", "per_foot_body_contact"}:
        return "missing_per_foot_assignment"
    if str(info.get("source_phase_foot") or info.get("phase_foot") or "") not in {"left", "right"}:
        return "missing_per_foot_assignment"
    missing = [
        field
        for field in ("min_confidence", "max_height_m", "max_speed_mps", "source_thresholds")
        if field not in info
    ]
    if missing:
        return "missing_confidence_fields"
    confidence = _optional_float(info.get("min_confidence"))
    if confidence is None:
        return "invalid_min_confidence"
    if confidence < 0.90:
        return "low_body_contact_confidence"
    evidence = info.get("assignment_evidence") if isinstance(info.get("assignment_evidence"), Mapping) else {}
    agreement = _optional_float(info.get("body_detector_agreement"))
    if agreement is None:
        agreement = _optional_float(evidence.get("body_detector_agreement"))
    if agreement is None:
        return "missing_body_detector_agreement"
    if agreement < 0.90:
        return "low_body_detector_agreement"
    return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _transition_frame_keys(
    frames: Sequence[Mapping[str, Any]],
    *,
    stance_index: Mapping[tuple[int, int], Mapping[str, Any]],
) -> set[tuple[int, int]]:
    frames_by_player: dict[int, list[int]] = {}
    stance_by_player: dict[int, set[int]] = {}
    for frame in frames:
        player_id = int(frame["player_id"])
        frame_idx = int(frame["frame_idx"])
        frames_by_player.setdefault(player_id, []).append(frame_idx)
        if bool(stance_index.get((player_id, frame_idx), {}).get("stance", False)):
            stance_by_player.setdefault(player_id, set()).add(frame_idx)
    transitions: set[tuple[int, int]] = set()
    for player_id, frame_indices in frames_by_player.items():
        stance_frames = sorted(stance_by_player.get(player_id, set()))
        if not stance_frames:
            continue
        ordered = sorted(set(frame_indices))
        for frame_idx in ordered:
            if frame_idx in stance_frames:
                continue
            nearest = min(abs(frame_idx - stance_frame) for stance_frame in stance_frames)
            before = [stance_frame for stance_frame in stance_frames if stance_frame < frame_idx]
            after = [stance_frame for stance_frame in stance_frames if stance_frame > frame_idx]
            if nearest <= STANCE_AWARE_TRANSITION_BOUNDARY_FRAMES or (before and after):
                transitions.add((player_id, frame_idx))
    return transitions


def _stance_phase_anchor_targets(
    frames: Sequence[Mapping[str, Any]],
    *,
    stance_index: Mapping[tuple[int, int], Mapping[str, Any]],
    max_anchor_drift_m: float,
) -> tuple[dict[tuple[int, int], list[float]], int]:
    stance_runs: dict[tuple[int, str], list[tuple[int, list[float]]]] = {}
    fallback_run_by_player: dict[int, int] = {}
    previous_frame_by_player: dict[int, int] = {}
    previous_was_stance_by_player: dict[int, bool] = {}
    for frame in sorted(frames, key=lambda item: (int(item["player_id"]), int(item["frame_idx"]))):
        player_id = int(frame["player_id"])
        frame_idx = int(frame["frame_idx"])
        info = stance_index.get((player_id, frame_idx), {})
        if not bool(info.get("stance", False)):
            previous_was_stance_by_player[player_id] = False
            previous_frame_by_player[player_id] = frame_idx
            continue
        phase_id = info.get("phase_id")
        if phase_id is None:
            previous_frame = previous_frame_by_player.get(player_id)
            starts_new_run = (
                previous_frame is None
                or not previous_was_stance_by_player.get(player_id, False)
                or frame_idx > previous_frame + 1
            )
            if starts_new_run:
                fallback_run_by_player[player_id] = fallback_run_by_player.get(player_id, 0) + 1
            phase_id = f"stance_run_{fallback_run_by_player.get(player_id, 0)}"
        track_xy = [float(value) for value in frame["track_world_xy"][:2]]
        stance_runs.setdefault((player_id, str(phase_id)), []).append((frame_idx, track_xy))
        previous_was_stance_by_player[player_id] = True
        previous_frame_by_player[player_id] = frame_idx

    targets: dict[tuple[int, int], list[float]] = {}
    split_count = 0
    for (player_id, phase_id), samples in stance_runs.items():
        current_segment: list[tuple[int, list[float]]] = []
        segment_start_xy: list[float] | None = None
        for frame_idx, track_xy in samples:
            if (
                segment_start_xy is not None
                and _distance2(track_xy, segment_start_xy) > max_anchor_drift_m
                and current_segment
            ):
                _assign_phase_anchor_targets(targets, player_id=player_id, segment=current_segment)
                split_count += 1
                current_segment = []
                segment_start_xy = None
            if segment_start_xy is None:
                segment_start_xy = track_xy
            current_segment.append((frame_idx, track_xy))
        if current_segment:
            _assign_phase_anchor_targets(targets, player_id=player_id, segment=current_segment)
    return targets, split_count


def _assign_phase_anchor_targets(
    targets: dict[tuple[int, int], list[float]],
    *,
    player_id: int,
    segment: Sequence[tuple[int, Sequence[float]]],
) -> None:
    xs = sorted(float(track_xy[0]) for _frame_idx, track_xy in segment)
    ys = sorted(float(track_xy[1]) for _frame_idx, track_xy in segment)
    anchor = [_median_sorted(xs), _median_sorted(ys)]
    for frame_idx, _track_xy in segment:
        targets[(player_id, int(frame_idx))] = list(anchor)


def _median_sorted(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return (float(values[mid - 1]) + float(values[mid])) / 2.0


def _placement_velocity_xy(
    info: Mapping[str, Any],
    *,
    current_track_xy: Sequence[float],
    previous_track_xy: Sequence[float] | None,
    dt: float,
) -> list[float] | None:
    raw = info.get("velocity")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 2:
        try:
            return [float(raw[0]), float(raw[1])]
        except (TypeError, ValueError):
            return None
    if previous_track_xy is not None and dt > 0.0:
        return [
            (float(current_track_xy[0]) - float(previous_track_xy[0])) / dt,
            (float(current_track_xy[1]) - float(previous_track_xy[1])) / dt,
        ]
    return None


def _placement_covariance_is_high(info: Mapping[str, Any]) -> bool:
    raw = info.get("covariance_m2")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < 2:
        return False
    try:
        return float(raw[0][0]) + float(raw[1][1]) > STANCE_AWARE_HIGH_COVARIANCE_TRACE_M2
    except (TypeError, ValueError, IndexError):
        return False


def _positive_dt(current_t: float, previous_t: float | None, *, fps: float) -> float:
    if previous_t is None:
        return 1.0 / fps
    dt = float(current_t) - float(previous_t)
    return dt if dt > 0.0 else 1.0 / fps


def _limit_step(previous: Sequence[float], current: Sequence[float], *, max_distance: float) -> tuple[list[float], bool]:
    distance = _distance3(previous, current)
    if distance <= max_distance:
        return [float(value) for value in current], False
    if distance == 0.0:
        return [float(value) for value in current], False
    scale = max_distance / distance
    return [float(previous[idx]) + (float(current[idx]) - float(previous[idx])) * scale for idx in range(3)], True


def _limit_xy_delta(dx: float, dy: float, *, max_magnitude: float) -> tuple[float, float, bool]:
    magnitude = sqrt(dx * dx + dy * dy)
    if magnitude <= max_magnitude or magnitude == 0.0:
        return dx, dy, False
    scale = max_magnitude / magnitude
    return dx * scale, dy * scale, True


def _common_grounding_anchor(frames: Sequence[Mapping[str, Any]]) -> str:
    anchors = sorted({str(frame.get("grounding_anchor", "")) for frame in frames if frame.get("grounding_anchor")})
    if not anchors:
        return ""
    return anchors[0] if len(anchors) == 1 else ",".join(anchors)


def _distance2(left: Sequence[float], right: Sequence[float]) -> float:
    return sqrt(sum((left[idx] - right[idx]) ** 2 for idx in range(2)))


def _distribution_m(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "p50": _percentile(values, 50.0),
        "p90": _percentile(values, 90.0),
        "p95": _percentile(values, 95.0),
        "max": max(values, default=0.0),
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (rank - lower)


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


def _matrix3(values: Any, *, name: str) -> list[list[float]]:
    if values is None or isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != 3:
        raise ValueError(f"{name} must be a 3x3 matrix")
    rows = [_float_list(row, name=f"{name}/{idx}") for idx, row in enumerate(values)]
    if any(len(row) != 3 for row in rows):
        raise ValueError(f"{name} must be a 3x3 matrix")
    return rows


def _apply_homography_xy(matrix: Sequence[Sequence[float]], pixel_xy: Sequence[float]) -> list[float]:
    xy = _vector2(pixel_xy, name="pixel_xy")
    x = float(xy[0])
    y = float(xy[1])
    w = float(matrix[2][0]) * x + float(matrix[2][1]) * y + float(matrix[2][2])
    if abs(w) < 1e-12:
        raise ValueError("camera_motion homography projection reached zero scale")
    return [
        (float(matrix[0][0]) * x + float(matrix[0][1]) * y + float(matrix[0][2])) / w,
        (float(matrix[1][0]) * x + float(matrix[1][1]) * y + float(matrix[1][2])) / w,
    ]


def _invert_matrix3(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    a, b, c = [float(value) for value in matrix[0]]
    d, e, f = [float(value) for value in matrix[1]]
    g, h, i = [float(value) for value in matrix[2]]
    det = (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )
    if abs(det) < 1e-12:
        raise ValueError("calibration homography is singular")
    inv_det = 1.0 / det
    return [
        [(e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det],
        [(f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det],
        [(d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det],
    ]


def _float_list(values: Any, *, name: str) -> list[float]:
    if values is None or isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result: list[float] = []
    for idx, value in enumerate(values):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}/{idx} must be numeric") from exc
        if not isfinite(number):
            raise ValueError(f"{name}/{idx} must be finite")
        result.append(number)
    return result
