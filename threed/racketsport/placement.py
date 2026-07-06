"""Foot-keypoint placement fusion for track anchors on the court plane."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


UTC = timezone.utc
COURT_HALF_WIDTH_M = 3.048
COURT_HALF_LENGTH_M = 6.7056

NATIVE2D_FOOT_NAMES: dict[str, tuple[str, ...]] = {
    "left": ("left_ankle", "left_heel", "left_big_toe", "left_small_toe"),
    "right": ("right_ankle", "right_heel", "right_big_toe", "right_small_toe"),
}

SAM3D_FOOT_KEYPOINT_INDICES: dict[str, int] = {
    "left_ankle": 13,
    "right_ankle": 14,
    "left_toe": 15,
    "right_toe": 16,
    "left_heel": 17,
    "right_heel": 20,
}


@dataclass(frozen=True)
class FootSignal:
    name: str
    xy: Sequence[float]
    covariance: Sequence[Sequence[float]]
    used: bool = True
    reason: str | None = None


@dataclass(frozen=True)
class SmoothedPlacement:
    xy: list[float]
    covariance: list[list[float]]
    velocity: list[float]


@dataclass(frozen=True)
class PlacementConfig:
    keypoint_conf_min: float = 2.0
    sam3d_conf_min: float = 0.01
    keypoint_base_sigma_px: float = 4.0
    keypoint_min_sigma_px: float = 1.5
    keypoint_max_sigma_px: float = 22.0
    bbox_base_sigma_px: float = 24.0
    far_bbox_sigma_multiplier: float = 1.8
    crouch_bbox_sigma_multiplier: float = 2.2
    far_keypoint_sigma_multiplier: float = 1.15
    measurement_sigma_floor_m: float = 0.025
    process_noise_mps2: float = 2.0
    keypoint_stance_speed_mps: float = 0.30
    stance_speed_mps: float = 0.45
    stance_min_duration_s: float = 0.15
    stance_covariance_scale: float = 0.15
    zero_velocity_sigma_mps: float = 0.12
    court_margin_m: float = 0.60
    undistort: bool = True
    vote_share_min: float = 0.80
    vote_min: int = 10
    bbox_pad_frac: float = 0.10
    bbox_pad_min_px: float = 12.0
    foot_lower_half_frac: float = 0.50
    net_clamp_epsilon_m: float = 0.05
    centerline_gap_window_s: float = 0.33
    max_measurement_gap_s: float = 0.75
    gap_interp_min_gap_s: float = 0.50
    gap_interp_max_speed_mps: float = 8.0
    max_written_speed_mps: float = 8.0
    divergence_max_m: float = 0.75
    measurement_adherence_max_m: float = 0.30
    continuity_max_frame_displacement_m: float = 0.50
    continuity_max_supported_speed_mps: float = 2.50
    fallback_transition_blend_frames: int = 5
    visual_max_root_step_m: float | None = 0.10


@dataclass(frozen=True)
class PlacementRewriteResult:
    placement_path: Path
    backup_tracks_path: Path
    coverage_unchanged: bool
    source_counts: dict[str, int]
    court_bounds_violations: int
    summary: dict[str, Any]


@dataclass(frozen=True)
class _PixelObservation:
    source: str
    pixel_xy: list[float]
    confidence: float
    valid: bool = True
    reason: str | None = None
    sidecar_player_id: int | None = None
    mapped_player_id: int | None = None


@dataclass(frozen=True)
class _CameraMotionObservation:
    matrix: np.ndarray
    model: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class _FrameSignals:
    fused_xy: list[float]
    fused_covariance: list[list[float]]
    stance: bool
    signals: list[dict[str, Any]]
    source_counts: dict[str, int]


@dataclass(frozen=True)
class _StancePhase:
    player_id: int
    foot: str
    frame_indices: tuple[int, ...]
    source: str


@dataclass(frozen=True)
class _SegmentedSmoothingResult:
    smoothed: dict[int, SmoothedPlacement]
    gap_hold_frames: set[int]
    gap_interpolated_frames: set[int]
    gap_reacquisition_speed_violations: list[dict[str, Any]]


def rewrite_tracks_with_placement(
    *,
    tracks_path: str | Path,
    calibration_path: str | Path,
    placement_path: str | Path,
    native2d_keypoints_path: str | Path | None = None,
    sam3d_keypoints_path: str | Path | None = None,
    stance_phases_path: str | Path | None = None,
    camera_motion_path: str | Path | None = None,
    refine_from_sam3d: bool = False,
    config: PlacementConfig | None = None,
) -> PlacementRewriteResult:
    """Fuse foot evidence and rewrite ``tracks.json`` world anchors in place."""

    config = config or PlacementConfig()
    tracks_path = Path(tracks_path)
    calibration_path = Path(calibration_path)
    placement_path = Path(placement_path)
    native2d_keypoints_path = Path(native2d_keypoints_path) if native2d_keypoints_path is not None else None
    sam3d_keypoints_path = Path(sam3d_keypoints_path) if sam3d_keypoints_path is not None else None
    stance_phases_path = Path(stance_phases_path) if stance_phases_path is not None else None
    camera_motion_path = Path(camera_motion_path) if camera_motion_path is not None else None

    tracks_payload = _read_json(tracks_path)
    calibration_payload = _read_json(calibration_path)
    camera_motion_frames, camera_motion_metadata = (
        _load_camera_motion(camera_motion_path) if camera_motion_path is not None else ({}, {})
    )
    native2d = _load_native2d_foot_pixels(native2d_keypoints_path, config=config) if native2d_keypoints_path else {}
    sam3d = _load_sam3d_foot_pixels(sam3d_keypoints_path, config=config) if sam3d_keypoints_path else {}
    bbox_index = _build_track_bbox_index(tracks_payload)
    native2d, native2d_identity = _reassociate_sidecar_pixels(
        native2d,
        bbox_index=bbox_index,
        source_name="native2d",
        config=config,
    )
    sam3d, sam3d_identity = _reassociate_sidecar_pixels(
        sam3d,
        bbox_index=bbox_index,
        source_name="sam3d",
        config=config,
    )
    sidecar_identity_summary = {"native2d": native2d_identity, "sam3d": sam3d_identity}
    stance_phases = _load_stance_phases(stance_phases_path) if stance_phases_path else {}
    external_stance_phase_count = sum(len(phases) for phases in stance_phases.values())

    backup_path = tracks_path.with_name("tracks_prewrite_backup.json")
    if not backup_path.exists():
        backup_payload = copy.deepcopy(tracks_payload)
        backup_payload["placement_provenance"] = {
            "backup_of": tracks_path.name,
            "created_by": "threed.racketsport.placement",
            "source_sha256": _sha256_file(tracks_path),
        }
        _write_json(backup_path, backup_payload)

    homography = np.asarray(calibration_payload["homography"], dtype=float)
    intrinsics = calibration_payload.get("intrinsics", {})
    camera_matrix = _camera_matrix(intrinsics)
    dist = [float(value) for value in intrinsics.get("dist", []) or []]
    homography_pixel_convention = _homography_pixel_convention(calibration_payload)
    undistort_applied = bool(
        config.undistort and homography_pixel_convention == "undistorted_pixels" and _dist_nonzero(dist)
    )

    fps = float(tracks_payload.get("fps") or 30.0)
    placement_players: list[dict[str, Any]] = []
    rewritten_payload = copy.deepcopy(tracks_payload)
    total_source_counts: Counter[str] = Counter({"bbox": 0, "native2d": 0, "sam3d": 0})
    total_bounds_violations = 0
    frame_count = 0
    jitter_summary: dict[str, dict[str, float]] = {}
    stance_wobble_summary: dict[str, dict[str, float]] = {}
    stance_detection_players: dict[str, dict[str, Any]] = {}
    stance_detection_warnings: list[str] = []
    native_stance_phase_count = 0
    boundary_guard_players: dict[str, dict[str, int]] = {}
    boundary_guard_totals: Counter[str] = Counter({"net_gap_clamped_frames": 0, "centerline_gap_clamped_frames": 0})
    smoothing_guard_players: dict[str, dict[str, int]] = {}
    smoothing_guard_totals: Counter[str] = Counter(
        {
            "divergence_snap_frames": 0,
            "fallback_transition_blends": 0,
            "gap_interpolated_frames": 0,
            "gap_hold_frames": 0,
            "written_speed_capped_frames": 0,
        }
    )
    gap_reacquisition_speed_violations: list[dict[str, Any]] = []
    visual_smoothing_players: dict[str, dict[str, int | float]] = {}
    visual_smoothing_totals: Counter[str] = Counter({"frames_adjusted_count": 0, "infeasible_segment_count": 0})
    camera_motion_seen: set[int] = set()
    camera_motion_used: set[int] = set()

    for source_player, dest_player in zip(tracks_payload.get("players", []), rewritten_payload.get("players", []), strict=False):
        player_id = int(source_player["id"])
        side = str(source_player.get("side_original", source_player.get("side", "")))
        frames = list(source_player.get("frames", []))
        heights = [float(frame["bbox"][3]) - float(frame["bbox"][1]) for frame in frames if _valid_bbox(frame.get("bbox"))]
        height95 = float(np.quantile(heights, 0.95)) if heights else 1.0

        frame_signals: dict[int, _FrameSignals] = {}
        keypoint_xy_for_stance: dict[int, list[float]] = {}
        trajectory_xy_for_stance: dict[int, list[float]] = {}
        original_by_frame: dict[int, list[float]] = {}
        frame_indices: list[int] = []

        for frame in frames:
            frame_idx = _frame_index(frame, fps)
            camera_motion = camera_motion_frames.get(frame_idx)
            if camera_motion_path is not None:
                camera_motion_seen.add(frame_idx)
                if camera_motion is not None:
                    camera_motion_used.add(frame_idx)
            frame_indices.append(frame_idx)
            original_xy = _xy(frame.get("world_xy"), name="world_xy")
            original_by_frame[frame_idx] = original_xy
            signals = _signals_for_frame(
                frame=frame,
                player_id=player_id,
                frame_idx=frame_idx,
                side=side,
                height95=height95,
                native2d=native2d,
                sam3d=sam3d,
                homography=homography,
                camera_matrix=camera_matrix,
                dist=dist,
                undistort_applied=undistort_applied,
                pixel_transform=camera_motion.matrix if camera_motion is not None else None,
                config=config,
            )
            used_signals = [
                FootSignal(signal["name"], signal["xy"], signal["covariance_m2"], used=signal["used"], reason=signal.get("reason"))
                for signal in signals
                if signal["used"] and signal.get("xy") is not None
            ]
            if used_signals:
                fused_xy, fused_cov = inverse_covariance_fuse(used_signals)
            else:
                fused_xy = list(original_xy)
                fused_cov = [[0.5, 0.0], [0.0, 0.5]]
            source_counts = Counter(signal["name"] for signal in signals if signal["used"])
            for key, value in source_counts.items():
                total_source_counts[key] += value
            keypoint_signal = next((signal for signal in signals if signal["used"] and signal["name"] in {"native2d", "sam3d"}), None)
            if keypoint_signal and keypoint_signal.get("xy") is not None:
                keypoint_xy_for_stance[frame_idx] = [float(keypoint_signal["xy"][0]), float(keypoint_signal["xy"][1])]
            if used_signals:
                trajectory_xy_for_stance[frame_idx] = [float(fused_xy[0]), float(fused_xy[1])]
            frame_signals[frame_idx] = _FrameSignals(
                fused_xy=[float(fused_xy[0]), float(fused_xy[1])],
                fused_covariance=_covariance_to_list(fused_cov),
                stance=False,
                signals=_signals_for_artifact(signals),
                source_counts=dict(source_counts),
            )

        player_stance_phases = stance_phases.get(player_id, [])
        external_stance_frames = _phase_frame_set(player_stance_phases, frame_indices=frame_indices)
        keypoint_stance_frames = _detect_stance_frames(
            keypoint_xy_for_stance,
            frame_indices=frame_indices,
            fps=fps,
            config=config,
            speed_mps=config.keypoint_stance_speed_mps,
        )
        fallback_stance_frames = (
            set()
            if keypoint_stance_frames
            else _detect_stance_frames(
                trajectory_xy_for_stance,
                frame_indices=frame_indices,
                fps=fps,
                config=config,
                speed_mps=config.stance_speed_mps,
            )
        )
        native_stance_phases = _stance_phases_from_frames(
            player_id=player_id,
            stance_frames=keypoint_stance_frames or fallback_stance_frames,
            source="native_keypoint_low_speed" if keypoint_stance_frames else "native_placement_low_speed",
        )
        native_stance_phase_count += len(native_stance_phases)
        native_stance_frames = _phase_frame_set(native_stance_phases, frame_indices=frame_indices)
        stance_frames = set(native_stance_frames)
        stance_frames.update(external_stance_frames)
        if not native_stance_phases and not player_stance_phases:
            stance_detection_warnings.append(
                f"player {player_id}: emitted zero native stance phases at "
                f"{config.stance_speed_mps:.3f} m/s for >= {config.stance_min_duration_s:.3f}s"
            )
        stance_detection_players[str(player_id)] = {
            "native_phase_count": len(native_stance_phases),
            "external_phase_count": len(player_stance_phases),
            "stance_frame_count": len(stance_frames),
            "frame_count": len(frame_indices),
            "coverage_fraction": (len(stance_frames) / len(frame_indices)) if frame_indices else 0.0,
            "source": "native_keypoint_low_speed" if keypoint_stance_frames else "native_placement_low_speed",
        }
        measurements = {
            frame_idx: (
                frame_signals[frame_idx].fused_xy,
                frame_signals[frame_idx].fused_covariance,
                frame_idx in stance_frames,
            )
            for frame_idx in frame_indices
            if frame_idx in frame_signals and any(frame_signals[frame_idx].source_counts.values())
        }
        gap_hold_frames: set[int] = set()
        gap_interpolated_frames: set[int] = set()
        if measurements:
            smoothing_result = _smooth_measurement_segments(
                measurements,
                frame_indices=frame_indices,
                fps=fps,
                player_id=player_id,
                config=config,
            )
            smoothed = smoothing_result.smoothed
            gap_hold_frames = set(smoothing_result.gap_hold_frames)
            gap_interpolated_frames = set(smoothing_result.gap_interpolated_frames)
            gap_reacquisition_speed_violations.extend(smoothing_result.gap_reacquisition_speed_violations)
            for frame_idx in frame_indices:
                if frame_idx not in smoothed:
                    smoothed[frame_idx] = SmoothedPlacement(
                        xy=list(original_by_frame[frame_idx]),
                        covariance=[[0.5, 0.0], [0.0, 0.5]],
                        velocity=[0.0, 0.0],
                    )
        else:
            smoothed = {
                frame_idx: SmoothedPlacement(
                    xy=list(original_by_frame[frame_idx]),
                    covariance=[[0.5, 0.0], [0.0, 0.5]],
                    velocity=[0.0, 0.0],
                )
                for frame_idx in frame_indices
            }
        stance_anchor_phases = [*player_stance_phases, *native_stance_phases]
        if stance_anchor_phases:
            smoothed = _anchor_external_stance_phases(
                smoothed,
                phases=stance_anchor_phases,
                frame_indices=frame_indices,
                covariance_scale=config.stance_covariance_scale,
                court_margin_m=config.court_margin_m,
            )
        smoothed, boundary_counts = _apply_boundary_guards(
            smoothed,
            frame_signals=frame_signals,
            frame_indices=frame_indices,
            fps=fps,
            config=config,
        )
        boundary_guard_players[str(player_id)] = dict(boundary_counts)
        boundary_guard_totals.update(boundary_counts)

        placement_frames: list[dict[str, Any]] = []
        rewritten_frames = dest_player.get("frames", [])
        rewritten_by_frame: dict[int, list[float]] = {}
        smoothing_counts: Counter[str] = Counter(
            {
                "divergence_snap_frames": 0,
                "fallback_transition_blends": 0,
                "gap_interpolated_frames": len(gap_interpolated_frames),
                "gap_hold_frames": len(gap_hold_frames),
                "written_speed_capped_frames": 0,
            }
        )
        previous_written_xy: list[float] | None = None
        previous_written_source: str | None = None
        previous_written_frame_idx: int | None = None
        previous_used_measurement = False
        previous_stance = False
        for frame, rewritten_frame in zip(frames, rewritten_frames, strict=False):
            frame_idx = _frame_index(frame, fps)
            smoothed_frame = smoothed[frame_idx]
            fs = frame_signals[frame_idx]
            used_measurement = _frame_has_used_measurement(fs)
            xy = list(smoothed_frame.xy)
            if frame_idx in gap_interpolated_frames:
                written_source = "gap_interpolated"
            elif frame_idx in gap_hold_frames:
                written_source = "gap_hold"
            else:
                written_source = "smoothed"
            divergence_limit = min(float(config.divergence_max_m), float(config.measurement_adherence_max_m))
            if used_measurement and frame_idx not in stance_frames and _distance_xy(xy, fs.fused_xy) > divergence_limit:
                xy = list(fs.fused_xy)
                written_source = "fused_divergence"
                smoothing_counts["divergence_snap_frames"] += 1
            if not _inside_court_bounds(xy, margin_m=config.court_margin_m):
                fallback_xy = frame_signals[frame_idx].fused_xy
                if _inside_court_bounds(fallback_xy, margin_m=config.court_margin_m):
                    xy = list(fallback_xy)
                    written_source = "fused_court_bounds"
                else:
                    xy = list(original_by_frame[frame_idx])
                    written_source = "original_court_bounds"
            if previous_written_xy is not None:
                source_switched = written_source != previous_written_source
                consecutive_supported = previous_used_measurement and used_measurement
                divergence_snap = written_source.startswith("fused_divergence")
                gap_transition = previous_written_source is not None and (
                    "gap_hold" in previous_written_source or "gap_transition" in previous_written_source
                )
                step = _distance_xy(previous_written_xy, xy)
                if source_switched and step > 1e-9:
                    smoothing_counts["fallback_transition_blends"] += 1
                frame_delta = int(frame_idx - previous_written_frame_idx) if previous_written_frame_idx is not None else 1
                max_speed_mps = float(config.max_written_speed_mps)
                if consecutive_supported and not divergence_snap and not gap_transition:
                    max_speed_mps = min(max_speed_mps, float(config.continuity_max_supported_speed_mps))
                if previous_stance and frame_idx in stance_frames and not divergence_snap and not gap_transition:
                    max_speed_mps = min(max_speed_mps, float(config.stance_speed_mps))
                max_step = _max_written_step(
                    config=config,
                    fps=fps,
                    frame_delta=frame_delta,
                    max_speed_mps=max_speed_mps,
                )
                if step > max_step:
                    xy = _limit_step(previous_written_xy, xy, max_step=max_step)
                    if gap_transition:
                        written_source = f"{written_source}_gap_transition_blend"
                    else:
                        written_source = f"{written_source}_blend"
                    smoothing_counts["written_speed_capped_frames"] += 1
            original_inside = _inside_court_bounds(original_by_frame[frame_idx], margin_m=config.court_margin_m)
            if not _inside_court_bounds(xy, margin_m=config.court_margin_m) and original_inside:
                total_bounds_violations += 1
            rewritten_frame["world_xy"] = [float(xy[0]), float(xy[1])]
            rewritten_by_frame[frame_idx] = [float(xy[0]), float(xy[1])]
            frame_count += 1
            stance = frame_idx in stance_frames
            placement_frame = {
                "frame_idx": int(frame_idx),
                "t": float(frame.get("t", frame_idx / fps)),
                "original_world_xy": original_by_frame[frame_idx],
                "fused_world_xy": fs.fused_xy,
                "smoothed_world_xy": [float(xy[0]), float(xy[1])],
                "covariance_m2": smoothed_frame.covariance,
                "stance": bool(stance),
                "signals": fs.signals,
                "source_counts": fs.source_counts,
                "output_source": written_source,
            }
            if frame_idx in gap_interpolated_frames:
                placement_frame["gap_interpolated"] = True
            if frame_idx in gap_hold_frames:
                placement_frame["gap_hold"] = True
            placement_frames.append(placement_frame)
            previous_written_xy = [float(xy[0]), float(xy[1])]
            previous_written_source = written_source
            previous_written_frame_idx = int(frame_idx)
            previous_used_measurement = used_measurement
            previous_stance = stance
        visual_counts = _apply_visual_root_step_bound(
            placement_frames,
            rewritten_frames,
            config=config,
            fps=fps,
        )
        for frame in placement_frames:
            frame_idx = int(frame["frame_idx"])
            rewritten_by_frame[frame_idx] = [float(frame["smoothed_world_xy"][0]), float(frame["smoothed_world_xy"][1])]
        placement_players.append({"id": player_id, "frames": placement_frames})
        jitter_summary[str(player_id)] = _jitter_before_after(frames, rewritten_frames, fps=fps)
        player_stance_wobble = _stance_wobble_before_after(original_by_frame, rewritten_by_frame, stance_frames)
        if player_stance_wobble["phase_count"] > 0:
            stance_wobble_summary[str(player_id)] = player_stance_wobble
        smoothing_guard_players[str(player_id)] = dict(smoothing_counts)
        smoothing_guard_totals.update(smoothing_counts)
        visual_smoothing_players[str(player_id)] = visual_counts
        visual_smoothing_totals.update({key: int(value) for key, value in visual_counts.items() if key.endswith("_count")})

    side_consistency_summary = _recompute_side_roles_and_consistency(
        tracks_payload=tracks_payload,
        rewritten_payload=rewritten_payload,
        fps=fps,
    )
    coverage_unchanged = _track_frame_count(tracks_payload) == _track_frame_count(rewritten_payload)
    stance_phase_count = external_stance_phase_count + native_stance_phase_count
    boundary_guard_summary = {"totals": dict(boundary_guard_totals), "players": boundary_guard_players}
    smoothing_guard_summary = {"totals": dict(smoothing_guard_totals), "players": smoothing_guard_players}
    visual_smoothing_summary = {
        "visual_max_root_step_m": config.visual_max_root_step_m,
        "frames_adjusted_count": int(visual_smoothing_totals.get("frames_adjusted_count", 0)),
        "infeasible_segment_count": int(visual_smoothing_totals.get("infeasible_segment_count", 0)),
        "players": visual_smoothing_players,
    }
    camera_motion_counts: dict[str, Any] = {}
    if camera_motion_path is not None:
        camera_motion_counts = {
            "camera_motion_path": camera_motion_path.name,
            "camera_motion_frames_used": len(camera_motion_used),
            "camera_motion_frames_uncompensated": len(camera_motion_seen - camera_motion_used),
            "camera_motion_artifact_frame_count": camera_motion_metadata.get("artifact_frame_count", 0),
            "camera_motion_artifact_compensated_frame_count": camera_motion_metadata.get("artifact_compensated_frame_count", 0),
        }
    provenance = {
        "stage": "placement_refine" if refine_from_sam3d else "placement",
        "generated_at": _utc_now(),
        "tracks": tracks_path.name,
        "placement": placement_path.name,
        "tracks_backup": backup_path.name,
        "native2d_keypoints": native2d_keypoints_path.name if native2d_keypoints_path else None,
        "sam3d_keypoints": sam3d_keypoints_path.name if sam3d_keypoints_path else None,
        "stance_phases": stance_phases_path.name if stance_phases_path else None,
        "stance_phase_count": stance_phase_count,
        "external_stance_phase_count": external_stance_phase_count,
        "native_stance_phase_count": native_stance_phase_count,
        "stance_detection": {
            "method": "low_speed_dwell",
            "keypoint_speed_threshold_mps": config.keypoint_stance_speed_mps,
            "placement_speed_threshold_mps": config.stance_speed_mps,
            "min_duration_s": config.stance_min_duration_s,
            "players": stance_detection_players,
            "warnings": stance_detection_warnings,
        },
        "refine_from_sam3d": bool(refine_from_sam3d),
        "homography_pixel_convention": homography_pixel_convention,
        "undistort_applied": undistort_applied,
        "source_counts": dict(total_source_counts),
        "sidecar_identity": sidecar_identity_summary,
        "boundary_guards": boundary_guard_summary,
        "smoothing_guards": smoothing_guard_summary,
        "visual_smoothing": visual_smoothing_summary,
        "side_quadrant_consistency": side_consistency_summary,
        **camera_motion_counts,
    }
    if camera_motion_path is not None:
        provenance["camera_motion"] = {
            "path": camera_motion_path.name,
            "frames_used": len(camera_motion_used),
            "frames_uncompensated": len(camera_motion_seen - camera_motion_used),
            "artifact_frame_count": camera_motion_metadata.get("artifact_frame_count", 0),
            "artifact_compensated_frame_count": camera_motion_metadata.get("artifact_compensated_frame_count", 0),
            "verified": camera_motion_metadata.get("verified"),
            "not_gate_verified": camera_motion_metadata.get("not_gate_verified"),
        }
    if gap_reacquisition_speed_violations:
        provenance["gap_reacquisition_speed_violations"] = gap_reacquisition_speed_violations
    rewritten_payload["placement_provenance"] = provenance
    placement_summary = {
        "player_count": len(placement_players),
        "frame_count": frame_count,
        "coverage_unchanged": coverage_unchanged,
        "source_counts": dict(total_source_counts),
        "sidecar_identity": sidecar_identity_summary,
        "boundary_guards": boundary_guard_summary,
        "smoothing_guards": smoothing_guard_summary,
        "visual_smoothing": visual_smoothing_summary,
        "side_quadrant_consistency": side_consistency_summary,
        "jitter_before_after_mps": jitter_summary,
        "stance_wobble_before_after_m": stance_wobble_summary,
        "court_bounds_violations": total_bounds_violations,
    }
    if gap_reacquisition_speed_violations:
        placement_summary["gap_reacquisition_speed_violations"] = gap_reacquisition_speed_violations
    placement_summary.update(camera_motion_counts)

    placement_payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_placement",
        "fps": fps,
        "source": "threed.racketsport.placement",
        "tracks_path": tracks_path.name,
        "backup_tracks_path": backup_path.name,
        "refine_from_sam3d": bool(refine_from_sam3d),
        "homography_pixel_convention": homography_pixel_convention,
        "undistort_applied": undistort_applied,
        "players": placement_players,
        "summary": placement_summary,
        "provenance": provenance,
    }
    placement_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(placement_path, placement_payload)
    _write_json(tracks_path, rewritten_payload)
    return PlacementRewriteResult(
        placement_path=placement_path,
        backup_tracks_path=backup_path,
        coverage_unchanged=coverage_unchanged,
        source_counts=dict(total_source_counts),
        court_bounds_violations=total_bounds_violations,
        summary=placement_summary,
    )


def _homography_pixel_convention(calibration_payload: Mapping[str, Any]) -> str:
    declared = str(calibration_payload.get("homography_pixel_convention") or "raw_pixels")
    return "undistorted_pixels" if declared == "undistorted_pixels" else "raw_pixels"


def _build_track_bbox_index(tracks_payload: Mapping[str, Any]) -> dict[int, dict[int, list[float]]]:
    fps = float(tracks_payload.get("fps") or 30.0)
    index: dict[int, dict[int, list[float]]] = defaultdict(dict)
    for player in tracks_payload.get("players", []) or []:
        if not isinstance(player, Mapping):
            continue
        try:
            player_id = int(player["id"])
        except (KeyError, TypeError, ValueError):
            continue
        for frame in player.get("frames", []) or []:
            if not isinstance(frame, Mapping) or not _valid_bbox(frame.get("bbox")):
                continue
            index[_frame_index(frame, fps)][player_id] = _bbox(frame.get("bbox"))
    return {int(frame_idx): dict(boxes) for frame_idx, boxes in index.items()}


def _reassociate_sidecar_pixels(
    observations: Mapping[tuple[int, int], _PixelObservation],
    *,
    bbox_index: Mapping[int, Mapping[int, Sequence[float]]],
    source_name: str,
    config: PlacementConfig,
) -> tuple[dict[tuple[int, int], _PixelObservation], dict[str, Any]]:
    del source_name
    by_sidecar: dict[int, list[tuple[int, _PixelObservation]]] = defaultdict(list)
    for (sidecar_player_id, frame_idx), observation in observations.items():
        by_sidecar[int(sidecar_player_id)].append((int(frame_idx), observation))

    remapped: dict[tuple[int, int], _PixelObservation] = {}
    players_diag: dict[str, dict[str, Any]] = {}
    dropped_identity_mismatch_by_player: Counter[str] = Counter()
    totals: Counter[str] = Counter({"raw_obs": 0, "reassigned_obs": 0, "dropped_obs": 0, "used_obs": 0})

    for sidecar_player_id, player_observations in sorted(by_sidecar.items()):
        totals["raw_obs"] += len(player_observations)
        votes: Counter[int] = Counter()
        none_votes = 0
        for frame_idx, observation in player_observations:
            frame_boxes = bbox_index.get(frame_idx, {})
            if not frame_boxes:
                continue
            winning_track_id = _track_id_for_pixel(observation.pixel_xy, frame_boxes, config=config)
            if winning_track_id is None:
                none_votes += 1
            else:
                votes[int(winning_track_id)] += 1
        winning_track_id: int | None = None
        winning_votes = 0
        if votes:
            winning_track_id, winning_votes = sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0]
        non_none_votes = int(sum(votes.values()))
        vote_share = (float(winning_votes) / float(non_none_votes)) if non_none_votes else 0.0
        accepted = (
            winning_track_id is not None
            and winning_votes >= int(config.vote_min)
            and vote_share >= float(config.vote_share_min)
        )
        integer_match = winning_track_id == sidecar_player_id if winning_track_id is not None else False
        players_diag[str(sidecar_player_id)] = {
            "vote_share": vote_share,
            "votes": int(winning_votes),
            "non_none_votes": int(non_none_votes),
            "none_votes": int(none_votes),
            "mapped_track_id": int(winning_track_id) if accepted and winning_track_id is not None else None,
            "dropped": not accepted,
            "integer_match": bool(integer_match),
            "raw_obs": int(len(player_observations)),
        }

        if not accepted or winning_track_id is None:
            totals["dropped_obs"] += len(player_observations)
            continue

        for frame_idx, observation in player_observations:
            frame_boxes = bbox_index.get(frame_idx, {})
            mapped_bbox = frame_boxes.get(int(winning_track_id))
            key = (int(winning_track_id), int(frame_idx))
            if mapped_bbox is not None and _pixel_matches_mapped_foot_bbox(observation.pixel_xy, mapped_bbox, config=config):
                valid_observation = _PixelObservation(
                    source=observation.source,
                    pixel_xy=list(observation.pixel_xy),
                    confidence=float(observation.confidence),
                    valid=True,
                    reason=None,
                    sidecar_player_id=int(sidecar_player_id),
                    mapped_player_id=int(winning_track_id),
                )
                current = remapped.get(key)
                if current is None or not current.valid or valid_observation.confidence >= current.confidence:
                    remapped[key] = valid_observation
                totals["used_obs"] += 1
                if winning_track_id != sidecar_player_id:
                    totals["reassigned_obs"] += 1
            else:
                invalid_observation = _PixelObservation(
                    source=observation.source,
                    pixel_xy=list(observation.pixel_xy),
                    confidence=float(observation.confidence),
                    valid=False,
                    reason="identity_pixel_mismatch",
                    sidecar_player_id=int(sidecar_player_id),
                    mapped_player_id=int(winning_track_id),
                )
                current = remapped.get(key)
                if current is None:
                    remapped[key] = invalid_observation
                totals["dropped_obs"] += 1
                dropped_identity_mismatch_by_player[str(winning_track_id)] += 1

    return remapped, {
        "players": players_diag,
        "totals": dict(totals),
        "dropped_identity_mismatch_by_player": dict(dropped_identity_mismatch_by_player),
    }


def _track_id_for_pixel(
    pixel_xy: Sequence[float],
    frame_boxes: Mapping[int, Sequence[float]],
    *,
    config: PlacementConfig,
) -> int | None:
    containing: list[tuple[int, float]] = []
    for track_id, bbox in frame_boxes.items():
        padded = _padded_bbox(bbox, config=config)
        if _pixel_inside_bbox(pixel_xy, padded):
            containing.append((int(track_id), _bbox_bottom_center_distance(pixel_xy, bbox)))
    if not containing:
        return None
    return sorted(containing, key=lambda item: (item[1], item[0]))[0][0]


def _pixel_matches_mapped_foot_bbox(pixel_xy: Sequence[float], bbox: Sequence[float], *, config: PlacementConfig) -> bool:
    padded = _padded_bbox(bbox, config=config)
    if not _pixel_inside_bbox(pixel_xy, padded):
        return False
    box = _bbox(bbox)
    lower_half_y = box[1] + max(min(float(config.foot_lower_half_frac), 1.0), 0.0) * (box[3] - box[1])
    return float(pixel_xy[1]) >= lower_half_y


def _padded_bbox(bbox: Sequence[float], *, config: PlacementConfig) -> list[float]:
    box = _bbox(bbox)
    width = box[2] - box[0]
    height = box[3] - box[1]
    pad_x = max(float(config.bbox_pad_min_px), float(config.bbox_pad_frac) * width)
    pad_y = max(float(config.bbox_pad_min_px), float(config.bbox_pad_frac) * height)
    return [box[0] - pad_x, box[1] - pad_y, box[2] + pad_x, box[3] + pad_y]


def _pixel_inside_bbox(pixel_xy: Sequence[float], bbox: Sequence[float]) -> bool:
    x, y = float(pixel_xy[0]), float(pixel_xy[1])
    box = _bbox(bbox)
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _bbox_bottom_center_distance(pixel_xy: Sequence[float], bbox: Sequence[float]) -> float:
    box = _bbox(bbox)
    return float(math.hypot(float(pixel_xy[0]) - ((box[0] + box[2]) / 2.0), float(pixel_xy[1]) - box[3]))


def _smooth_measurement_segments(
    measurements: Mapping[int, tuple[Sequence[float], Sequence[Sequence[float]], bool]],
    *,
    frame_indices: Sequence[int],
    fps: float,
    player_id: int,
    config: PlacementConfig,
) -> _SegmentedSmoothingResult:
    indices = sorted(dict.fromkeys(int(idx) for idx in frame_indices))
    measurement_indices = sorted(int(idx) for idx in measurements)
    if not indices or not measurement_indices:
        return _SegmentedSmoothingResult(
            smoothed={},
            gap_hold_frames=set(),
            gap_interpolated_frames=set(),
            gap_reacquisition_speed_violations=[],
        )

    max_gap_frames = max(0, int(math.ceil(max(float(config.max_measurement_gap_s), 0.0) * fps)))
    gap_interp_min_frames = max(0, int(math.ceil(max(float(config.gap_interp_min_gap_s), 0.0) * fps)))
    interior_gap_frames = min(max_gap_frames, gap_interp_min_frames)
    segments: list[list[int]] = [[measurement_indices[0]]]
    interior_gaps: list[tuple[int, int]] = []
    for prev_idx, next_idx in zip(measurement_indices, measurement_indices[1:], strict=False):
        gap_frames = max(0, next_idx - prev_idx - 1)
        if gap_frames > interior_gap_frames:
            interior_gaps.append((prev_idx, next_idx))
            segments.append([next_idx])
        else:
            segments[-1].append(next_idx)

    smoothed: dict[int, SmoothedPlacement] = {}
    gap_hold_frames: set[int] = set()
    gap_interpolated_frames: set[int] = set()
    gap_reacquisition_speed_violations: list[dict[str, Any]] = []
    all_index_set = set(indices)
    first_frame_idx = indices[0]
    last_frame_idx = indices[-1]
    for segment_idx, segment_measurements in enumerate(segments):
        first_measurement = segment_measurements[0]
        last_measurement = segment_measurements[-1]
        segment_start = first_measurement
        segment_end = last_measurement
        if segment_idx == 0 and first_measurement - first_frame_idx <= max_gap_frames:
            segment_start = first_frame_idx
        if segment_idx == len(segments) - 1 and last_frame_idx - last_measurement <= max_gap_frames:
            segment_end = last_frame_idx
        segment_frame_indices = [idx for idx in indices if segment_start <= idx <= segment_end]
        segment_result = kalman_rts_smooth(
            {idx: measurements[idx] for idx in segment_measurements},
            frame_indices=segment_frame_indices,
            fps=fps,
            process_noise_mps2=config.process_noise_mps2,
            stance_covariance_scale=config.stance_covariance_scale,
            zero_velocity_sigma_mps=config.zero_velocity_sigma_mps,
        )
        smoothed.update(segment_result)

    for prev_idx, next_idx in interior_gaps:
        gap_indices = [idx for idx in range(prev_idx + 1, next_idx) if idx in all_index_set]
        if not gap_indices:
            continue
        prev_measurement_xy = _xy(measurements[prev_idx][0], name="measurement.xy")
        next_measurement_xy = _xy(measurements[next_idx][0], name="measurement.xy")
        displacement = _distance_xy(prev_measurement_xy, next_measurement_xy)
        dt = max((int(next_idx) - int(prev_idx)) / fps, 1.0 / fps)
        implied_speed = displacement / dt
        prev_smoothed = smoothed.get(
            prev_idx,
            SmoothedPlacement(
                xy=list(prev_measurement_xy),
                covariance=_covariance_to_list(measurements[prev_idx][1]),
                velocity=[0.0, 0.0],
            ),
        )
        next_smoothed = smoothed.get(
            next_idx,
            SmoothedPlacement(
                xy=list(next_measurement_xy),
                covariance=_covariance_to_list(measurements[next_idx][1]),
                velocity=[0.0, 0.0],
            ),
        )
        if implied_speed <= float(config.gap_interp_max_speed_mps) + 1e-9:
            span = max(int(next_idx) - int(prev_idx), 1)
            velocity = [
                (float(next_smoothed.xy[0]) - float(prev_smoothed.xy[0])) * fps / span,
                (float(next_smoothed.xy[1]) - float(prev_smoothed.xy[1])) * fps / span,
            ]
            prev_cov = np.asarray(_matrix2(prev_smoothed.covariance, name="prev_smoothed.covariance"), dtype=float)
            next_cov = np.asarray(_matrix2(next_smoothed.covariance, name="next_smoothed.covariance"), dtype=float)
            for idx in gap_indices:
                alpha = (int(idx) - int(prev_idx)) / span
                xy = [
                    float(prev_smoothed.xy[0]) + (float(next_smoothed.xy[0]) - float(prev_smoothed.xy[0])) * alpha,
                    float(prev_smoothed.xy[1]) + (float(next_smoothed.xy[1]) - float(prev_smoothed.xy[1])) * alpha,
                ]
                cov = prev_cov + (next_cov - prev_cov) * alpha
                smoothed[idx] = SmoothedPlacement(
                    xy=xy,
                    covariance=_inflate_covariance(cov, factor=100.0, min_variance=0.25),
                    velocity=[float(velocity[0]), float(velocity[1])],
                )
                gap_interpolated_frames.add(idx)
        else:
            hold_xy = list(prev_smoothed.xy)
            hold_cov = _inflate_covariance(prev_smoothed.covariance, factor=100.0, min_variance=0.25)
            for idx in gap_indices:
                smoothed[idx] = SmoothedPlacement(xy=list(hold_xy), covariance=hold_cov, velocity=[0.0, 0.0])
                gap_hold_frames.add(idx)
            gap_reacquisition_speed_violations.append(
                {
                    "player_id": int(player_id),
                    "gap_start_frame": int(gap_indices[0]),
                    "gap_end_frame": int(gap_indices[-1]),
                    "implied_speed_mps": float(implied_speed),
                    "displacement_m": float(displacement),
                }
            )

    last_measurement = measurement_indices[-1]
    trailing_gap = max(0, last_frame_idx - last_measurement)
    if trailing_gap > max_gap_frames:
        hold_xy = _xy(measurements[last_measurement][0], name="measurement.xy")
        hold_cov = _inflate_covariance(measurements[last_measurement][1], factor=100.0, min_variance=0.25)
        for idx in indices:
            if idx <= last_measurement:
                continue
            smoothed[idx] = SmoothedPlacement(xy=list(hold_xy), covariance=hold_cov, velocity=[0.0, 0.0])
            gap_hold_frames.add(idx)

    return _SegmentedSmoothingResult(
        smoothed=smoothed,
        gap_hold_frames=gap_hold_frames,
        gap_interpolated_frames=gap_interpolated_frames,
        gap_reacquisition_speed_violations=gap_reacquisition_speed_violations,
    )


def _apply_boundary_guards(
    smoothed: Mapping[int, SmoothedPlacement],
    *,
    frame_signals: Mapping[int, _FrameSignals],
    frame_indices: Sequence[int],
    fps: float,
    config: PlacementConfig,
) -> tuple[dict[int, SmoothedPlacement], dict[str, int]]:
    guarded = dict(smoothed)
    used_measurement_frames = [
        int(idx) for idx in frame_indices if idx in frame_signals and _frame_has_used_measurement(frame_signals[idx])
    ]
    if not used_measurement_frames:
        return guarded, {"net_gap_clamped_frames": 0, "centerline_gap_clamped_frames": 0}
    counts: Counter[str] = Counter({"net_gap_clamped_frames": 0, "centerline_gap_clamped_frames": 0})
    centerline_window_frames = int(math.ceil(max(float(config.centerline_gap_window_s), 0.0) * fps))
    for frame_idx in frame_indices:
        if frame_idx not in guarded or frame_idx not in frame_signals or _frame_has_used_measurement(frame_signals[frame_idx]):
            continue
        nearest_idx = min(used_measurement_frames, key=lambda idx: (abs(idx - int(frame_idx)), idx))
        nearest_xy = frame_signals[nearest_idx].fused_xy
        xy = list(guarded[frame_idx].xy)
        covariance = guarded[frame_idx].covariance
        changed = False
        nearest_y_sign = _axis_sign(nearest_xy[1])
        if nearest_y_sign and _axis_sign(xy[1]) != nearest_y_sign:
            xy[1] = nearest_y_sign * float(config.net_clamp_epsilon_m)
            covariance = _inflate_covariance(covariance, factor=25.0, min_variance=0.25)
            counts["net_gap_clamped_frames"] += 1
            changed = True
        has_nearby_used_measurement = any(
            abs(int(frame_idx) - used_idx) <= centerline_window_frames for used_idx in used_measurement_frames
        )
        nearest_x_sign = _axis_sign(nearest_xy[0])
        if not has_nearby_used_measurement and nearest_x_sign and _axis_sign(xy[0]) != nearest_x_sign:
            xy[0] = nearest_x_sign * float(config.net_clamp_epsilon_m)
            covariance = _inflate_covariance(covariance, factor=25.0, min_variance=0.25)
            counts["centerline_gap_clamped_frames"] += 1
            changed = True
        if changed:
            guarded[frame_idx] = SmoothedPlacement(
                xy=[float(xy[0]), float(xy[1])],
                covariance=covariance,
                velocity=list(guarded[frame_idx].velocity),
            )
    return guarded, dict(counts)


def _frame_has_used_measurement(frame_signals: _FrameSignals) -> bool:
    return any(int(count) > 0 for count in frame_signals.source_counts.values())


def _axis_sign(value: float) -> int:
    value = float(value)
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _inflate_covariance(
    covariance: Sequence[Sequence[float]],
    *,
    factor: float,
    min_variance: float,
) -> list[list[float]]:
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (2, 2):
        cov = np.asarray(_matrix2(covariance, name="covariance"), dtype=float)
    cov = cov * float(factor)
    cov[0, 0] = max(float(cov[0, 0]), float(min_variance))
    cov[1, 1] = max(float(cov[1, 1]), float(min_variance))
    return _covariance_to_list(cov)


def _distance_xy(left: Sequence[float], right: Sequence[float]) -> float:
    return float(math.hypot(float(right[0]) - float(left[0]), float(right[1]) - float(left[1])))


def _limit_step(previous_xy: Sequence[float], target_xy: Sequence[float], *, max_step: float) -> list[float]:
    distance = _distance_xy(previous_xy, target_xy)
    if distance <= max_step or distance <= 1e-12:
        return [float(target_xy[0]), float(target_xy[1])]
    scale = float(max_step) / distance
    return [
        float(previous_xy[0]) + (float(target_xy[0]) - float(previous_xy[0])) * scale,
        float(previous_xy[1]) + (float(target_xy[1]) - float(previous_xy[1])) * scale,
    ]


def _apply_visual_root_step_bound(
    placement_frames: list[dict[str, Any]],
    rewritten_frames: Sequence[dict[str, Any]],
    *,
    config: PlacementConfig,
    fps: float,
) -> dict[str, int | float]:
    if config.visual_max_root_step_m is None:
        return {
            "visual_max_root_step_m": 0.0,
            "frames_adjusted_count": 0,
            "infeasible_segment_count": 0,
        }
    max_step = float(config.visual_max_root_step_m)
    if max_step <= 0.0:
        raise ValueError("visual_max_root_step_m must be positive when set")
    xy_by_frame = {
        int(frame["frame_idx"]): [float(frame["smoothed_world_xy"][0]), float(frame["smoothed_world_xy"][1])]
        for frame in placement_frames
    }
    edge_max_steps = _visual_root_step_edge_limits(
        placement_frames,
        config=config,
        fps=fps,
        visual_max_step_m=max_step,
    )
    adjusted, frames_adjusted, infeasible = _redistribute_visual_root_steps(
        xy_by_frame,
        max_step_m=max_step,
        edge_max_steps=edge_max_steps,
        return_infeasible=True,
    )
    if frames_adjusted <= 0:
        return {
            "visual_max_root_step_m": max_step,
            "frames_adjusted_count": 0,
            "infeasible_segment_count": infeasible,
        }
    rewritten_by_frame = {
        _frame_index(frame, 30.0): frame
        for frame in rewritten_frames
        if isinstance(frame, dict)
    }
    for frame in placement_frames:
        frame_idx = int(frame["frame_idx"])
        bounded_xy = adjusted.get(frame_idx)
        if bounded_xy is None:
            continue
        old_xy = [float(frame["smoothed_world_xy"][0]), float(frame["smoothed_world_xy"][1])]
        if _distance_xy(old_xy, bounded_xy) <= 1e-12:
            continue
        frame["smoothed_world_xy"] = [float(bounded_xy[0]), float(bounded_xy[1])]
        frame["visual_root_step_bounded"] = True
        frame["output_source"] = f"{frame.get('output_source', 'smoothed')}_visual_bound"
        rewritten = rewritten_by_frame.get(frame_idx)
        if rewritten is not None:
            rewritten["world_xy"] = [float(bounded_xy[0]), float(bounded_xy[1])]
    return {
        "visual_max_root_step_m": max_step,
        "frames_adjusted_count": frames_adjusted,
        "infeasible_segment_count": infeasible,
    }


def _redistribute_visual_root_steps(
    xy_by_frame: Mapping[int, Sequence[float]],
    *,
    max_step_m: float,
    edge_max_steps: Mapping[tuple[int, int], float] | None = None,
    return_infeasible: bool = False,
) -> tuple[dict[int, list[float]], int] | tuple[dict[int, list[float]], int, int]:
    if max_step_m <= 0.0:
        raise ValueError("max_step_m must be positive")
    adjusted = {int(idx): [float(xy[0]), float(xy[1])] for idx, xy in xy_by_frame.items()}
    changed_frames: set[int] = set()
    infeasible_count = 0
    for run in _contiguous_runs(sorted(adjusted)):
        if len(run) < 3:
            continue
        start = adjusted[run[0]]
        end = adjusted[run[-1]]
        run_capacity = sum(
            _visual_root_step_edge_limit(left_idx, right_idx, max_step_m=max_step_m, edge_max_steps=edge_max_steps)
            for left_idx, right_idx in zip(run, run[1:], strict=False)
        )
        if _distance_xy(start, end) > run_capacity + 1e-9:
            infeasible_count += 1
            continue
        for _iteration in range(max(4, len(run) * 2)):
            changed = False
            for left_idx, right_idx in zip(run, run[1:], strict=False):
                left = adjusted[left_idx]
                right = adjusted[right_idx]
                edge_limit = _visual_root_step_edge_limit(
                    left_idx,
                    right_idx,
                    max_step_m=max_step_m,
                    edge_max_steps=edge_max_steps,
                )
                if _distance_xy(left, right) <= edge_limit + 1e-12:
                    continue
                limited = _limit_step(left, right, max_step=edge_limit)
                if right_idx != run[-1]:
                    adjusted[right_idx] = limited
                    changed_frames.add(right_idx)
                    changed = True
            for right_idx, left_idx in zip(reversed(run), reversed(run[:-1]), strict=False):
                right = adjusted[right_idx]
                left = adjusted[left_idx]
                edge_limit = _visual_root_step_edge_limit(
                    left_idx,
                    right_idx,
                    max_step_m=max_step_m,
                    edge_max_steps=edge_max_steps,
                )
                if _distance_xy(right, left) <= edge_limit + 1e-12:
                    continue
                limited = _limit_step(right, left, max_step=edge_limit)
                if left_idx != run[0]:
                    adjusted[left_idx] = limited
                    changed_frames.add(left_idx)
                    changed = True
            if not changed:
                break
        for left_idx, right_idx in zip(run, run[1:], strict=False):
            edge_limit = _visual_root_step_edge_limit(
                left_idx,
                right_idx,
                max_step_m=max_step_m,
                edge_max_steps=edge_max_steps,
            )
            if _distance_xy(adjusted[left_idx], adjusted[right_idx]) > edge_limit + 1e-9:
                infeasible_count += 1
                break
    if return_infeasible:
        return adjusted, len(changed_frames), infeasible_count
    return adjusted, len(changed_frames)


def _visual_root_step_edge_limits(
    placement_frames: Sequence[Mapping[str, Any]],
    *,
    config: PlacementConfig,
    fps: float,
    visual_max_step_m: float,
) -> dict[tuple[int, int], float]:
    sorted_frames = sorted(placement_frames, key=lambda frame: int(frame.get("frame_idx", 0)))
    limits: dict[tuple[int, int], float] = {}
    for previous, current in zip(sorted_frames, sorted_frames[1:], strict=False):
        previous_idx = int(previous.get("frame_idx", 0))
        current_idx = int(current.get("frame_idx", 0))
        if current_idx <= previous_idx:
            continue
        frame_delta = current_idx - previous_idx
        limit = float(visual_max_step_m)
        current_source = str(current.get("output_source") or "")
        previous_source = str(previous.get("output_source") or "")
        divergence_snap = current_source.startswith("fused_divergence")
        gap_transition = "gap_hold" in previous_source or "gap_transition" in previous_source
        if (
            _placement_frame_has_used_measurement(previous)
            and _placement_frame_has_used_measurement(current)
            and not divergence_snap
            and not gap_transition
        ):
            limit = min(
                limit,
                _max_written_step(
                    config=config,
                    fps=fps,
                    frame_delta=frame_delta,
                    max_speed_mps=float(config.continuity_max_supported_speed_mps),
                ),
            )
        if previous.get("stance") is True and current.get("stance") is True and not divergence_snap and not gap_transition:
            limit = min(
                limit,
                _max_written_step(
                    config=config,
                    fps=fps,
                    frame_delta=frame_delta,
                    max_speed_mps=float(config.stance_speed_mps),
                ),
            )
        limits[(previous_idx, current_idx)] = max(limit, 0.0)
    return limits


def _visual_root_step_edge_limit(
    left_idx: int,
    right_idx: int,
    *,
    max_step_m: float,
    edge_max_steps: Mapping[tuple[int, int], float] | None,
) -> float:
    if edge_max_steps is None:
        return float(max_step_m)
    key = (min(int(left_idx), int(right_idx)), max(int(left_idx), int(right_idx)))
    return max(float(edge_max_steps.get(key, max_step_m)), 0.0)


def _placement_frame_has_used_measurement(frame: Mapping[str, Any]) -> bool:
    source_counts = frame.get("source_counts") or {}
    if not isinstance(source_counts, Mapping):
        return False
    return any(int(count) > 0 for count in source_counts.values())


def _max_written_step(
    *,
    config: PlacementConfig,
    fps: float,
    frame_delta: int,
    max_speed_mps: float | None = None,
) -> float:
    frame_count = max(int(frame_delta), 1)
    max_speed = max(float(config.max_written_speed_mps if max_speed_mps is None else max_speed_mps), 0.0)
    max_speed = max(max_speed - 1e-9, 0.0)
    return float(max_speed * frame_count / max(float(fps), 1e-9))


def _recompute_side_roles_and_consistency(
    *,
    tracks_payload: Mapping[str, Any],
    rewritten_payload: dict[str, Any],
    fps: float,
) -> dict[str, Any]:
    original_by_id = {
        int(player["id"]): player
        for player in tracks_payload.get("players", []) or []
        if isinstance(player, Mapping) and "id" in player
    }
    recomputed: dict[int, dict[str, Any]] = {}
    summary_players: dict[str, dict[str, Any]] = {}

    for player in rewritten_payload.get("players", []) or []:
        if not isinstance(player, dict):
            continue
        player_id = int(player["id"])
        original_player = original_by_id.get(player_id, {})
        original_side = str(player.get("side_original", original_player.get("side", player.get("side", ""))))
        original_role = str(player.get("role_original", original_player.get("role", player.get("role", ""))))
        frames = [frame for frame in player.get("frames", []) or [] if isinstance(frame, Mapping)]
        xy_by_frame = {int(_frame_index(frame, fps)): _xy(frame.get("world_xy"), name="world_xy") for frame in frames}
        points = list(xy_by_frame.values())
        median_y = float(np.median([xy[1] for xy in points])) if points else 0.0
        side = "near" if median_y < 0.0 else "far"
        side_sign = -1 if side == "near" else 1
        service_x = [
            xy[0]
            for xy in points
            if 2.1336 <= abs(float(xy[1])) <= COURT_HALF_LENGTH_M and _axis_sign(float(xy[1])) == side_sign
        ]
        role_x_values = service_x if service_x else [xy[0] for xy in points]
        median_x = float(np.median(role_x_values)) if role_x_values else 0.0
        role = "left" if median_x < 0.0 else "right"
        player["side_original"] = original_side
        player["role_original"] = original_role
        player["side"] = side
        player["role"] = role
        player["side_source"] = "placement_recomputed"
        player["role_source"] = "placement_recomputed"

        y_match_count = sum(1 for xy in points if _axis_sign(float(xy[1])) == side_sign)
        first_label_mismatch_frame_idx = None
        original_side_sign = -1 if original_side == "near" else 1 if original_side == "far" else 0
        for frame_idx, xy in sorted(xy_by_frame.items()):
            if original_side_sign and _axis_sign(float(xy[1])) != original_side_sign:
                first_label_mismatch_frame_idx = int(frame_idx)
                break
        crossings = 0
        sorted_points = [xy_by_frame[idx] for idx in sorted(xy_by_frame)]
        for left, right in zip(sorted_points, sorted_points[1:], strict=False):
            if abs(float(left[1])) <= 0.3 or abs(float(right[1])) <= 0.3:
                continue
            if _axis_sign(float(left[1])) != _axis_sign(float(right[1])):
                crossings += 1

        recomputed[player_id] = {"side": side, "xy_by_frame": xy_by_frame}
        summary_players[str(player_id)] = {
            "side_label_original": original_side,
            "side_recomputed": side,
            "frac_y_sign_matches_recomputed": (float(y_match_count) / float(len(points))) if points else 0.0,
            "net_crossing_count": int(crossings),
            "role_original": original_role,
            "role_recomputed": role,
            "first_label_mismatch_frame_idx": first_label_mismatch_frame_idx,
            "same_side_pairs": {},
        }

    pair_summary: dict[str, dict[str, Any]] = {}
    for left_id, left in sorted(recomputed.items()):
        for right_id, right in sorted(recomputed.items()):
            if right_id <= left_id or left["side"] != right["side"]:
                continue
            shared_frames = sorted(set(left["xy_by_frame"]) & set(right["xy_by_frame"]))
            same_quadrant = 0
            for frame_idx in shared_frames:
                left_sign = -1 if float(left["xy_by_frame"][frame_idx][0]) < 0.0 else 1
                right_sign = -1 if float(right["xy_by_frame"][frame_idx][0]) < 0.0 else 1
                if left_sign == right_sign:
                    same_quadrant += 1
            fraction = (float(same_quadrant) / float(len(shared_frames))) if shared_frames else 0.0
            key = f"{left_id}-{right_id}"
            pair_item = {
                "players": [int(left_id), int(right_id)],
                "side": str(left["side"]),
                "shared_frame_count": int(len(shared_frames)),
                "same_x_quadrant_fraction": fraction,
            }
            pair_summary[key] = pair_item
            summary_players[str(left_id)]["same_side_pairs"][key] = pair_item
            summary_players[str(right_id)]["same_side_pairs"][key] = pair_item

    return {"players": summary_players, "same_side_player_pairs": pair_summary}


def inverse_covariance_fuse(signals: Sequence[FootSignal]) -> tuple[list[float], list[list[float]]]:
    precision_sum = np.zeros((2, 2), dtype=float)
    weighted_sum = np.zeros(2, dtype=float)
    used = [signal for signal in signals if signal.used]
    if not used:
        raise ValueError("at least one used signal is required")
    for signal in used:
        xy = np.asarray(_xy(signal.xy, name=f"{signal.name}.xy"), dtype=float)
        covariance = np.asarray(_matrix2(signal.covariance, name=f"{signal.name}.covariance"), dtype=float)
        precision = np.linalg.pinv(covariance)
        precision_sum += precision
        weighted_sum += precision @ xy
    fused_covariance = np.linalg.pinv(precision_sum)
    fused_xy = fused_covariance @ weighted_sum
    return [float(fused_xy[0]), float(fused_xy[1])], _covariance_to_list(fused_covariance)


def homography_world_covariance(
    homography: Sequence[Sequence[float]] | np.ndarray,
    pixel_xy: Sequence[float],
    *,
    sigma_px: float,
    sigma_floor_m: float = 0.0,
) -> np.ndarray:
    if sigma_px <= 0.0 or not math.isfinite(float(sigma_px)):
        raise ValueError("sigma_px must be a positive finite value")
    homography = np.asarray(homography, dtype=float)
    pixel = np.asarray(_xy(pixel_xy, name="pixel_xy"), dtype=float)
    h_inv = np.linalg.inv(homography)
    eps = 1.0
    j_cols = []
    for axis in range(2):
        delta = np.zeros(2, dtype=float)
        delta[axis] = eps
        plus = _unproject_pixel(h_inv, pixel + delta)
        minus = _unproject_pixel(h_inv, pixel - delta)
        j_cols.append((plus - minus) / (2.0 * eps))
    jacobian = np.column_stack(j_cols)
    covariance = jacobian @ (np.eye(2, dtype=float) * float(sigma_px) ** 2) @ jacobian.T
    if sigma_floor_m > 0.0:
        covariance += np.eye(2, dtype=float) * sigma_floor_m**2
    return covariance


def kalman_rts_smooth(
    measurements: Mapping[int, tuple[Sequence[float], Sequence[Sequence[float]], bool]],
    *,
    frame_indices: Sequence[int],
    fps: float,
    process_noise_mps2: float = 2.0,
    stance_covariance_scale: float = 0.15,
    zero_velocity_sigma_mps: float = 0.12,
) -> dict[int, SmoothedPlacement]:
    indices = sorted(dict.fromkeys(int(idx) for idx in frame_indices))
    if not indices:
        return {}
    if fps <= 0.0:
        raise ValueError("fps must be positive")

    first_xy = None
    for idx in indices:
        if idx in measurements:
            first_xy = _xy(measurements[idx][0], name="measurement.xy")
            break
    if first_xy is None:
        first_xy = [0.0, 0.0]

    state = np.array([first_xy[0], first_xy[1], 0.0, 0.0], dtype=float)
    covariance = np.diag([1.0, 1.0, 25.0, 25.0]).astype(float)
    filtered_states: dict[int, np.ndarray] = {}
    filtered_covs: dict[int, np.ndarray] = {}
    predicted_states: dict[int, np.ndarray] = {}
    predicted_covs: dict[int, np.ndarray] = {}
    transitions: dict[int, np.ndarray] = {}
    previous_idx = indices[0]

    for idx in indices:
        dt = max((idx - previous_idx) / fps, 1.0 / fps if idx != previous_idx else 0.0)
        transition = _constant_velocity_transition(dt)
        process = _constant_velocity_process_noise(dt, process_noise_mps2)
        if idx == indices[0]:
            predicted_state = state.copy()
            predicted_cov = covariance.copy()
        else:
            predicted_state = transition @ state
            predicted_cov = transition @ covariance @ transition.T + process
        predicted_states[idx] = predicted_state.copy()
        predicted_covs[idx] = predicted_cov.copy()
        transitions[idx] = transition
        state = predicted_state
        covariance = predicted_cov

        if idx in measurements:
            xy, cov, stance = measurements[idx]
            measurement_cov = np.asarray(_matrix2(cov, name="measurement.covariance"), dtype=float)
            if stance:
                measurement_cov = measurement_cov * stance_covariance_scale
            state, covariance = _kalman_update(
                state,
                covariance,
                np.asarray(_xy(xy, name="measurement.xy"), dtype=float),
                np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float),
                measurement_cov,
            )
            if stance:
                state, covariance = _kalman_update(
                    state,
                    covariance,
                    np.array([0.0, 0.0], dtype=float),
                    np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=float),
                    np.eye(2, dtype=float) * zero_velocity_sigma_mps**2,
                )
        filtered_states[idx] = state.copy()
        filtered_covs[idx] = covariance.copy()
        previous_idx = idx

    smoothed_states: dict[int, np.ndarray] = {indices[-1]: filtered_states[indices[-1]].copy()}
    smoothed_covs: dict[int, np.ndarray] = {indices[-1]: filtered_covs[indices[-1]].copy()}
    for current_idx, next_idx in zip(reversed(indices[:-1]), reversed(indices[1:]), strict=True):
        transition_next = transitions[next_idx]
        filt_cov = filtered_covs[current_idx]
        pred_cov_next = predicted_covs[next_idx]
        gain = filt_cov @ transition_next.T @ np.linalg.pinv(pred_cov_next)
        smoothed_states[current_idx] = filtered_states[current_idx] + gain @ (
            smoothed_states[next_idx] - predicted_states[next_idx]
        )
        smoothed_covs[current_idx] = filt_cov + gain @ (smoothed_covs[next_idx] - pred_cov_next) @ gain.T

    return {
        idx: SmoothedPlacement(
            xy=[float(smoothed_states[idx][0]), float(smoothed_states[idx][1])],
            covariance=_covariance_to_list(smoothed_covs[idx][:2, :2]),
            velocity=[float(smoothed_states[idx][2]), float(smoothed_states[idx][3])],
        )
        for idx in indices
    }


def undistort_pixel(
    pixel_xy: Sequence[float],
    camera_matrix: Sequence[Sequence[float]],
    dist: Sequence[float],
) -> list[float]:
    if not _dist_nonzero(dist):
        return _xy(pixel_xy, name="pixel_xy")
    import cv2  # type: ignore[import-not-found]

    k = np.asarray(camera_matrix, dtype=np.float64)
    d = np.asarray(list(dist), dtype=np.float64)
    points = np.asarray([[list(_xy(pixel_xy, name="pixel_xy"))]], dtype=np.float64)
    undistorted = cv2.undistortPoints(points, k, d, P=k)[0, 0]
    return [float(undistorted[0]), float(undistorted[1])]


def bbox_pixel_sigma(
    bbox_xyxy: Sequence[float],
    *,
    side: str,
    height_norm: float,
    base_sigma_px: float = 24.0,
) -> float:
    bbox = _bbox(bbox_xyxy)
    width = max(bbox[2] - bbox[0], 1e-6)
    height = max(bbox[3] - bbox[1], 1e-6)
    aspect = height / width
    sigma = float(base_sigma_px)
    if str(side).lower() == "far":
        sigma *= PlacementConfig.far_bbox_sigma_multiplier
    if height_norm < 0.8 or aspect < 1.35:
        sigma *= PlacementConfig.crouch_bbox_sigma_multiplier
    return sigma


def _signals_for_frame(
    *,
    frame: Mapping[str, Any],
    player_id: int,
    frame_idx: int,
    side: str,
    height95: float,
    native2d: Mapping[tuple[int, int], _PixelObservation],
    sam3d: Mapping[tuple[int, int], _PixelObservation],
    homography: np.ndarray,
    camera_matrix: np.ndarray,
    dist: Sequence[float],
    undistort_applied: bool,
    pixel_transform: np.ndarray | None,
    config: PlacementConfig,
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    bbox = _bbox(frame.get("bbox"))
    height_norm = (bbox[3] - bbox[1]) / max(height95, 1e-6)
    bbox_pixel = _apply_pixel_transform([(bbox[0] + bbox[2]) / 2.0, bbox[3]], pixel_transform)
    signals.append(
        _signal_from_pixel(
            name="bbox",
            pixel_xy=bbox_pixel,
            confidence=float(frame.get("conf", 1.0)),
            sigma_px=bbox_pixel_sigma(bbox, side=side, height_norm=height_norm, base_sigma_px=config.bbox_base_sigma_px),
            side=side,
            homography=homography,
            camera_matrix=camera_matrix,
            dist=dist,
            undistort_applied=undistort_applied,
            config=config,
        )
    )
    for source_name, source_map in (("native2d", native2d), ("sam3d", sam3d)):
        observation = source_map.get((player_id, frame_idx))
        if observation is None:
            signals.append({"name": source_name, "xy": None, "covariance_m2": None, "sigma_m": None, "used": False, "reason": "missing"})
            continue
        if not observation.valid:
            signals.append(
                {
                    "name": source_name,
                    "xy": None,
                    "covariance_m2": None,
                    "sigma_m": None,
                    "used": False,
                    "reason": observation.reason or "identity_pixel_mismatch",
                    "sidecar_player_id": observation.sidecar_player_id,
                    "mapped_player_id": observation.mapped_player_id,
                }
            )
            continue
        sigma = _keypoint_sigma_px(observation.confidence, base_sigma_px=config.keypoint_base_sigma_px, config=config)
        if str(side).lower() == "far":
            sigma *= config.far_keypoint_sigma_multiplier
        signal = _signal_from_pixel(
            name=source_name,
            pixel_xy=_apply_pixel_transform(observation.pixel_xy, pixel_transform),
            confidence=observation.confidence,
            sigma_px=sigma,
            side=side,
            homography=homography,
            camera_matrix=camera_matrix,
            dist=dist,
            undistort_applied=undistort_applied,
            config=config,
        )
        signal["sidecar_player_id"] = observation.sidecar_player_id
        signal["mapped_player_id"] = observation.mapped_player_id
        signals.append(signal)
    return signals


def _signal_from_pixel(
    *,
    name: str,
    pixel_xy: Sequence[float],
    confidence: float,
    sigma_px: float,
    side: str,
    homography: np.ndarray,
    camera_matrix: np.ndarray,
    dist: Sequence[float],
    undistort_applied: bool,
    config: PlacementConfig,
) -> dict[str, Any]:
    del side, confidence
    pixel = undistort_pixel(pixel_xy, camera_matrix, dist) if undistort_applied else _xy(pixel_xy, name="pixel_xy")
    xy = _unproject_pixel(np.linalg.inv(homography), np.asarray(pixel, dtype=float))
    covariance = homography_world_covariance(
        homography,
        pixel,
        sigma_px=sigma_px,
        sigma_floor_m=config.measurement_sigma_floor_m,
    )
    sigma_m = [float(math.sqrt(max(covariance[0, 0], 0.0))), float(math.sqrt(max(covariance[1, 1], 0.0)))]
    used = _inside_court_bounds(xy, margin_m=config.court_margin_m)
    return {
        "name": name,
        "xy": [float(xy[0]), float(xy[1])],
        "covariance_m2": _covariance_to_list(covariance),
        "sigma_m": sigma_m,
        "used": bool(used),
        "reason": None if used else "outside_court_bounds",
    }


def _apply_pixel_transform(pixel_xy: Sequence[float], matrix: np.ndarray | None) -> list[float]:
    pixel = _xy(pixel_xy, name="pixel_xy")
    if matrix is None:
        return pixel
    transformed = matrix @ np.asarray([pixel[0], pixel[1], 1.0], dtype=float)
    scale = float(transformed[2])
    if abs(scale) < 1e-12:
        raise ValueError("camera_motion projection reached zero scale")
    return [float(transformed[0] / scale), float(transformed[1] / scale)]


def _signals_for_artifact(signals: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for signal in signals:
        cov = signal.get("covariance_m2")
        sigma = signal.get("sigma_m")
        artifact_signal = {
            "name": str(signal["name"]),
            "xy": signal.get("xy"),
            "sigma_m": sigma if sigma is not None else None,
            "used": bool(signal.get("used", False)),
            "reason": signal.get("reason"),
        }
        if cov is not None:
            artifact_signal["covariance_m2"] = cov
        if "sidecar_player_id" in signal:
            artifact_signal["sidecar_player_id"] = signal.get("sidecar_player_id")
        if "mapped_player_id" in signal:
            artifact_signal["mapped_player_id"] = signal.get("mapped_player_id")
        out.append(artifact_signal)
    return out


def _load_camera_motion(path: Path) -> tuple[dict[int, _CameraMotionObservation], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"camera_motion path does not exist: {path}")
    payload = _read_json(path)
    frames: dict[int, _CameraMotionObservation] = {}
    artifact_frame_count = 0
    artifact_compensated_frame_count = 0
    for item in payload.get("frames", []) or []:
        if not isinstance(item, Mapping):
            continue
        frame_idx = int(item.get("frame_idx", round(float(item.get("t", 0.0)) * float(payload.get("fps") or 30.0))))
        artifact_frame_count += 1
        if not bool(item.get("compensated", False)):
            continue
        matrix = np.asarray(_matrix3(item.get("M"), name=f"camera_motion.frames[{frame_idx}].M"), dtype=float)
        frames[frame_idx] = _CameraMotionObservation(
            matrix=matrix,
            model=str(item.get("model")) if item.get("model") is not None else None,
            reason=str(item.get("reason")) if item.get("reason") is not None else None,
        )
        artifact_compensated_frame_count += 1
    return frames, {
        "artifact_frame_count": artifact_frame_count,
        "artifact_compensated_frame_count": artifact_compensated_frame_count,
        "verified": payload.get("verified"),
        "not_gate_verified": payload.get("not_gate_verified"),
    }


def _load_native2d_foot_pixels(path: Path | None, *, config: PlacementConfig) -> dict[tuple[int, int], _PixelObservation]:
    if path is None or not path.is_file():
        return {}
    payload = _read_json(path)
    out: dict[tuple[int, int], _PixelObservation] = {}
    for player in payload.get("players", []) or []:
        player_id = int(player["id"])
        for frame in player.get("frames", []) or []:
            frame_idx = int(frame.get("frame_idx", round(float(frame.get("t", 0.0)) * float(payload.get("fps") or 30.0))))
            by_name = {str(joint.get("name")): joint for joint in frame.get("joints", []) or [] if isinstance(joint, Mapping)}
            foot_points = []
            for names in NATIVE2D_FOOT_NAMES.values():
                point = _weighted_named_pixel(by_name, names, conf_min=config.keypoint_conf_min)
                if point is not None:
                    foot_points.append(point)
            combined = _combine_weighted_pixels(foot_points)
            if combined is not None:
                out[(player_id, frame_idx)] = _PixelObservation(
                    "native2d", combined[0], combined[1], sidecar_player_id=player_id
                )
    return out


def _load_sam3d_foot_pixels(path: Path | None, *, config: PlacementConfig) -> dict[tuple[int, int], _PixelObservation]:
    if path is None or not path.is_file():
        return {}
    payload = _read_json(path)
    out: dict[tuple[int, int], _PixelObservation] = {}
    for player in payload.get("players", []) or []:
        player_id = int(player["id"])
        for frame in player.get("frames", []) or []:
            frame_idx = int(frame["frame_idx"])
            by_name = {str(item.get("name")): item for item in frame.get("keypoints", []) or [] if isinstance(item, Mapping)}
            foot_points = []
            for names in (("left_ankle", "left_heel", "left_toe"), ("right_ankle", "right_heel", "right_toe")):
                point = _weighted_sidecar_pixel(by_name, names, conf_min=config.sam3d_conf_min)
                if point is not None:
                    foot_points.append(point)
            combined = _combine_weighted_pixels(foot_points)
            if combined is not None:
                out[(player_id, frame_idx)] = _PixelObservation(
                    "sam3d", combined[0], combined[1], sidecar_player_id=player_id
                )
    return out


def _load_stance_phases(path: Path | None) -> dict[int, list[_StancePhase]]:
    if path is None or not path.is_file():
        return {}
    payload = _read_json(path)
    raw_phases = payload.get("phases")
    if not isinstance(raw_phases, list):
        raw_phases = payload.get("metrics_before", {}).get("phase_metrics") if isinstance(payload.get("metrics_before"), Mapping) else None
    if not isinstance(raw_phases, list):
        raw_phases = payload.get("metrics_after", {}).get("phase_metrics") if isinstance(payload.get("metrics_after"), Mapping) else None
    if not isinstance(raw_phases, list):
        return {}

    by_player: dict[int, list[_StancePhase]] = defaultdict(list)
    source = str(payload.get("artifact_type") or path.name)
    for item in raw_phases:
        if not isinstance(item, Mapping):
            continue
        try:
            player_id = int(item.get("player_id"))
        except (TypeError, ValueError):
            continue
        foot = str(item.get("foot") or "unknown")
        frame_indices = _phase_frame_indices(item)
        if not frame_indices:
            continue
        by_player[player_id].append(
            _StancePhase(
                player_id=player_id,
                foot=foot,
                frame_indices=tuple(frame_indices),
                source=source,
            )
        )
    return dict(by_player)


def _phase_frame_indices(item: Mapping[str, Any]) -> list[int]:
    raw_indices = item.get("frame_indices")
    if isinstance(raw_indices, Sequence) and not isinstance(raw_indices, str | bytes):
        out = []
        for value in raw_indices:
            try:
                out.append(int(value))
            except (TypeError, ValueError):
                continue
        return sorted(set(out))
    try:
        start = int(item["start_frame_index"])
        end = int(item["end_frame_index"])
    except (KeyError, TypeError, ValueError):
        return []
    if end < start:
        return []
    return list(range(start, end + 1))


def _phase_frame_set(phases: Sequence[_StancePhase], *, frame_indices: Sequence[int]) -> set[int]:
    available = set(int(idx) for idx in frame_indices)
    out: set[int] = set()
    for phase in phases:
        out.update(idx for idx in phase.frame_indices if idx in available)
    return out


def _stance_phases_from_frames(*, player_id: int, stance_frames: set[int], source: str) -> list[_StancePhase]:
    phases = []
    for run in _contiguous_runs(sorted(stance_frames)):
        if len(run) < 2:
            continue
        phases.append(
            _StancePhase(
                player_id=int(player_id),
                foot="unknown",
                frame_indices=tuple(int(idx) for idx in run),
                source=source,
            )
        )
    return phases


def _anchor_external_stance_phases(
    smoothed: Mapping[int, SmoothedPlacement],
    *,
    phases: Sequence[_StancePhase],
    frame_indices: Sequence[int],
    covariance_scale: float,
    court_margin_m: float,
) -> dict[int, SmoothedPlacement]:
    anchored = dict(smoothed)
    available = set(int(idx) for idx in frame_indices)
    for phase in phases:
        indices = [idx for idx in phase.frame_indices if idx in available and idx in anchored]
        if len(indices) < 3:
            continue
        xy_values = np.asarray([anchored[idx].xy for idx in indices], dtype=float)
        anchor_xy = np.median(xy_values, axis=0)
        if not _inside_court_bounds(anchor_xy, margin_m=court_margin_m):
            continue
        cov_values = np.asarray([anchored[idx].covariance for idx in indices], dtype=float)
        anchor_cov = np.median(cov_values, axis=0) * max(float(covariance_scale), 1e-6)
        for idx in indices:
            anchored[idx] = SmoothedPlacement(
                xy=[float(anchor_xy[0]), float(anchor_xy[1])],
                covariance=_covariance_to_list(anchor_cov),
                velocity=[0.0, 0.0],
            )
    return anchored


def _weighted_named_pixel(
    by_name: Mapping[str, Mapping[str, Any]],
    names: Sequence[str],
    *,
    conf_min: float,
) -> tuple[list[float], float] | None:
    weighted = []
    for name in names:
        item = by_name.get(name)
        if item is None:
            continue
        conf = float(item.get("conf", 0.0))
        if conf < conf_min:
            continue
        weighted.append(([float(item["x_px"]), float(item["y_px"])], conf))
    return _combine_weighted_pixels(weighted)


def _weighted_sidecar_pixel(
    by_name: Mapping[str, Mapping[str, Any]],
    names: Sequence[str],
    *,
    conf_min: float,
) -> tuple[list[float], float] | None:
    weighted = []
    for name in names:
        item = by_name.get(name)
        if item is None:
            continue
        conf = float(item.get("conf", 1.0))
        if conf < conf_min:
            continue
        weighted.append((_xy(item["xy_px"], name=f"{name}.xy_px"), conf))
    return _combine_weighted_pixels(weighted)


def _combine_weighted_pixels(weighted: Sequence[tuple[Sequence[float], float]]) -> tuple[list[float], float] | None:
    if not weighted:
        return None
    total = sum(max(float(weight), 1e-6) for _point, weight in weighted)
    xy = [
        sum(float(point[axis]) * max(float(weight), 1e-6) for point, weight in weighted) / total
        for axis in range(2)
    ]
    confidence = total / len(weighted)
    return ([float(xy[0]), float(xy[1])], float(confidence))


def _detect_stance_frames(
    keypoint_xy: Mapping[int, Sequence[float]],
    *,
    frame_indices: Sequence[int],
    fps: float,
    config: PlacementConfig,
    speed_mps: float | None = None,
) -> set[int]:
    threshold = config.stance_speed_mps if speed_mps is None else float(speed_mps)
    min_frames = max(2, int(math.ceil(max(config.stance_min_duration_s, 0.0) * fps)))
    candidates: set[int] = set()
    sorted_indices = sorted(frame_indices)
    for prev, current in zip(sorted_indices, sorted_indices[1:], strict=False):
        if prev not in keypoint_xy or current not in keypoint_xy:
            continue
        dt = max((current - prev) / fps, 1.0 / fps)
        speed = math.hypot(
            float(keypoint_xy[current][0]) - float(keypoint_xy[prev][0]),
            float(keypoint_xy[current][1]) - float(keypoint_xy[prev][1]),
        ) / dt
        if speed <= threshold:
            candidates.add(prev)
            candidates.add(current)
    stance: set[int] = set()
    for run in _contiguous_runs(sorted(candidates)):
        if len(run) >= min_frames:
            stance.update(run)
    return stance


def _jitter_before_after(source_frames: Sequence[Mapping[str, Any]], dest_frames: Sequence[Mapping[str, Any]], *, fps: float) -> dict[str, float]:
    before = _frame_speeds(source_frames, fps=fps)
    after = _frame_speeds(dest_frames, fps=fps)
    return {
        "before_p50": _quantile(before, 0.50),
        "before_p90": _quantile(before, 0.90),
        "after_p50": _quantile(after, 0.50),
        "after_p90": _quantile(after, 0.90),
    }


def _stance_wobble_before_after(
    original_by_frame: Mapping[int, Sequence[float]],
    rewritten_by_frame: Mapping[int, Sequence[float]],
    stance_frames: set[int],
) -> dict[str, float]:
    before_ranges = []
    after_ranges = []
    for run in _contiguous_runs(sorted(stance_frames)):
        if len(run) < 3:
            continue
        before = _xy_range([original_by_frame[idx] for idx in run if idx in original_by_frame])
        after = _xy_range([rewritten_by_frame[idx] for idx in run if idx in rewritten_by_frame])
        if before is None or after is None:
            continue
        before_ranges.append(before)
        after_ranges.append(after)
    return {
        "phase_count": float(len(before_ranges)),
        "before_p50": _quantile(before_ranges, 0.50),
        "before_p90": _quantile(before_ranges, 0.90),
        "after_p50": _quantile(after_ranges, 0.50),
        "after_p90": _quantile(after_ranges, 0.90),
    }


def _contiguous_runs(indices: Sequence[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for idx in indices:
        idx = int(idx)
        if not runs or idx != runs[-1][-1] + 1:
            runs.append([idx])
        else:
            runs[-1].append(idx)
    return runs


def _xy_range(points: Sequence[Sequence[float]]) -> float | None:
    if len(points) < 3:
        return None
    arr = np.asarray(points, dtype=float)
    return float(max(np.ptp(arr[:, 0]), np.ptp(arr[:, 1])))


def _frame_speeds(frames: Sequence[Mapping[str, Any]], *, fps: float) -> list[float]:
    speeds = []
    for f0, f1 in zip(frames, frames[1:], strict=False):
        idx0 = _frame_index(f0, fps)
        idx1 = _frame_index(f1, fps)
        if idx1 <= idx0:
            continue
        xy0 = _xy(f0.get("world_xy"), name="world_xy")
        xy1 = _xy(f1.get("world_xy"), name="world_xy")
        dt = (idx1 - idx0) / fps
        speeds.append(math.hypot(xy1[0] - xy0[0], xy1[1] - xy0[1]) / max(dt, 1e-9))
    return speeds


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _keypoint_sigma_px(confidence: float, *, base_sigma_px: float, config: PlacementConfig) -> float:
    confidence = max(float(confidence), 1e-6)
    if confidence <= 1.5:
        sigma = base_sigma_px * math.sqrt(0.9 / confidence)
    else:
        sigma = base_sigma_px * math.sqrt(6.0 / confidence)
    return float(min(max(sigma, config.keypoint_min_sigma_px), config.keypoint_max_sigma_px))


def _camera_matrix(intrinsics: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            [float(intrinsics.get("fx", 1.0)), 0.0, float(intrinsics.get("cx", 0.0))],
            [0.0, float(intrinsics.get("fy", 1.0)), float(intrinsics.get("cy", 0.0))],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _unproject_pixel(h_inv: np.ndarray, pixel: np.ndarray) -> np.ndarray:
    projected = h_inv @ np.array([float(pixel[0]), float(pixel[1]), 1.0], dtype=float)
    if abs(float(projected[2])) < 1e-12:
        raise ValueError("homography projection reached zero scale")
    return projected[:2] / projected[2]


def _constant_velocity_transition(dt: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=float)


def _constant_velocity_process_noise(dt: float, acceleration_sigma: float) -> np.ndarray:
    q = float(acceleration_sigma) ** 2
    return q * np.array(
        [
            [0.25 * dt**4, 0.0, 0.5 * dt**3, 0.0],
            [0.0, 0.25 * dt**4, 0.0, 0.5 * dt**3],
            [0.5 * dt**3, 0.0, dt**2, 0.0],
            [0.0, 0.5 * dt**3, 0.0, dt**2],
        ],
        dtype=float,
    )


def _kalman_update(
    state: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    h: np.ndarray,
    r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    innovation = measurement - h @ state
    s = h @ covariance @ h.T + r
    gain = covariance @ h.T @ np.linalg.pinv(s)
    updated_state = state + gain @ innovation
    identity = np.eye(covariance.shape[0], dtype=float)
    updated_cov = (identity - gain @ h) @ covariance @ (identity - gain @ h).T + gain @ r @ gain.T
    return updated_state, updated_cov


def _inside_court_bounds(xy: Sequence[float] | np.ndarray, *, margin_m: float) -> bool:
    x, y = float(xy[0]), float(xy[1])
    return (
        -COURT_HALF_WIDTH_M - margin_m <= x <= COURT_HALF_WIDTH_M + margin_m
        and -COURT_HALF_LENGTH_M - margin_m <= y <= COURT_HALF_LENGTH_M + margin_m
    )


def _frame_index(frame: Mapping[str, Any], fps: float) -> int:
    if "frame_idx" in frame:
        return int(frame["frame_idx"])
    return int(round(float(frame.get("t", 0.0)) * fps))


def _track_frame_count(payload: Mapping[str, Any]) -> int:
    return sum(len(player.get("frames", []) or []) for player in payload.get("players", []) or [])


def _bbox(values: Any) -> list[float]:
    if not _valid_bbox(values):
        raise ValueError("bbox must contain ordered x1,y1,x2,y2 values")
    return [float(value) for value in values]


def _valid_bbox(values: Any) -> bool:
    if not isinstance(values, Sequence) or isinstance(values, str | bytes) or len(values) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(value) for value in values]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(value) for value in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1


def _xy(values: Any, *, name: str) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, str | bytes) or len(values) != 2:
        raise ValueError(f"{name} must be a 2-vector")
    out = [float(values[0]), float(values[1])]
    if not all(math.isfinite(value) for value in out):
        raise ValueError(f"{name} must contain finite values")
    return out


def _matrix2(values: Any, *, name: str) -> list[list[float]]:
    if not isinstance(values, Sequence) or len(values) != 2:
        raise ValueError(f"{name} must be a 2x2 matrix")
    matrix = [_xy(row, name=f"{name}/{idx}") for idx, row in enumerate(values)]
    return matrix


def _matrix3(values: Any, *, name: str) -> list[list[float]]:
    if not isinstance(values, Sequence) or isinstance(values, str | bytes) or len(values) != 3:
        raise ValueError(f"{name} must be a 3x3 matrix")
    out: list[list[float]] = []
    for row_idx, row in enumerate(values):
        if not isinstance(row, Sequence) or isinstance(row, str | bytes) or len(row) != 3:
            raise ValueError(f"{name}/{row_idx} must be a 3-vector")
        row_values = [float(row[0]), float(row[1]), float(row[2])]
        if not all(math.isfinite(value) for value in row_values):
            raise ValueError(f"{name}/{row_idx} must contain finite values")
        out.append(row_values)
    return out


def _covariance_to_list(covariance: Sequence[Sequence[float]] | np.ndarray) -> list[list[float]]:
    arr = np.asarray(covariance, dtype=float)
    return [[float(arr[0, 0]), float(arr[0, 1])], [float(arr[1, 0]), float(arr[1, 1])]]


def _dist_nonzero(dist: Sequence[float]) -> bool:
    return any(abs(float(value)) > 1e-12 for value in dist)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
