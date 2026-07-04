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
    process_noise_mps2: float = 0.1
    stance_speed_mps: float = 0.55
    stance_covariance_scale: float = 0.15
    zero_velocity_sigma_mps: float = 0.12
    court_margin_m: float = 0.60
    undistort: bool = True


@dataclass(frozen=True)
class PlacementRewriteResult:
    placement_path: Path
    backup_tracks_path: Path
    coverage_unchanged: bool
    source_counts: dict[str, int]
    court_bounds_violations: int


@dataclass(frozen=True)
class _PixelObservation:
    source: str
    pixel_xy: list[float]
    confidence: float


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


def rewrite_tracks_with_placement(
    *,
    tracks_path: str | Path,
    calibration_path: str | Path,
    placement_path: str | Path,
    native2d_keypoints_path: str | Path | None = None,
    sam3d_keypoints_path: str | Path | None = None,
    stance_phases_path: str | Path | None = None,
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

    tracks_payload = _read_json(tracks_path)
    calibration_payload = _read_json(calibration_path)
    native2d = _load_native2d_foot_pixels(native2d_keypoints_path, config=config) if native2d_keypoints_path else {}
    sam3d = _load_sam3d_foot_pixels(sam3d_keypoints_path, config=config) if sam3d_keypoints_path else {}
    stance_phases = _load_stance_phases(stance_phases_path) if stance_phases_path else {}
    stance_phase_count = sum(len(phases) for phases in stance_phases.values())

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
    undistort_applied = bool(config.undistort and _dist_nonzero(dist))

    fps = float(tracks_payload.get("fps") or 30.0)
    placement_players: list[dict[str, Any]] = []
    rewritten_payload = copy.deepcopy(tracks_payload)
    total_source_counts: Counter[str] = Counter({"bbox": 0, "native2d": 0, "sam3d": 0})
    total_bounds_violations = 0
    frame_count = 0
    jitter_summary: dict[str, dict[str, float]] = {}
    stance_wobble_summary: dict[str, dict[str, float]] = {}

    for source_player, dest_player in zip(tracks_payload.get("players", []), rewritten_payload.get("players", []), strict=False):
        player_id = int(source_player["id"])
        side = str(source_player.get("side", ""))
        frames = list(source_player.get("frames", []))
        heights = [float(frame["bbox"][3]) - float(frame["bbox"][1]) for frame in frames if _valid_bbox(frame.get("bbox"))]
        height95 = float(np.quantile(heights, 0.95)) if heights else 1.0

        frame_signals: dict[int, _FrameSignals] = {}
        keypoint_xy_for_stance: dict[int, list[float]] = {}
        original_by_frame: dict[int, list[float]] = {}
        frame_indices: list[int] = []

        for frame in frames:
            frame_idx = _frame_index(frame, fps)
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
            frame_signals[frame_idx] = _FrameSignals(
                fused_xy=[float(fused_xy[0]), float(fused_xy[1])],
                fused_covariance=_covariance_to_list(fused_cov),
                stance=False,
                signals=_signals_for_artifact(signals),
                source_counts=dict(source_counts),
            )

        player_stance_phases = stance_phases.get(player_id, [])
        external_stance_frames = _phase_frame_set(player_stance_phases, frame_indices=frame_indices)
        stance_frames = _detect_stance_frames(keypoint_xy_for_stance, frame_indices=frame_indices, fps=fps, config=config)
        stance_frames.update(external_stance_frames)
        measurements = {
            frame_idx: (
                frame_signals[frame_idx].fused_xy,
                frame_signals[frame_idx].fused_covariance,
                frame_idx in stance_frames,
            )
            for frame_idx in frame_indices
            if frame_idx in frame_signals and any(frame_signals[frame_idx].source_counts.values())
        }
        if measurements:
            smoothed = kalman_rts_smooth(
                measurements,
                frame_indices=frame_indices,
                fps=fps,
                process_noise_mps2=config.process_noise_mps2,
                stance_covariance_scale=config.stance_covariance_scale,
                zero_velocity_sigma_mps=config.zero_velocity_sigma_mps,
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
        if player_stance_phases:
            smoothed = _anchor_external_stance_phases(
                smoothed,
                phases=player_stance_phases,
                frame_indices=frame_indices,
                covariance_scale=config.stance_covariance_scale,
                court_margin_m=config.court_margin_m,
            )

        placement_frames: list[dict[str, Any]] = []
        rewritten_frames = dest_player.get("frames", [])
        rewritten_by_frame: dict[int, list[float]] = {}
        for frame, rewritten_frame in zip(frames, rewritten_frames, strict=False):
            frame_idx = _frame_index(frame, fps)
            smoothed_frame = smoothed[frame_idx]
            xy = smoothed_frame.xy
            if not _inside_court_bounds(xy, margin_m=config.court_margin_m):
                fallback_xy = frame_signals[frame_idx].fused_xy
                xy = fallback_xy if _inside_court_bounds(fallback_xy, margin_m=config.court_margin_m) else original_by_frame[frame_idx]
            original_inside = _inside_court_bounds(original_by_frame[frame_idx], margin_m=config.court_margin_m)
            if not _inside_court_bounds(xy, margin_m=config.court_margin_m) and original_inside:
                total_bounds_violations += 1
            rewritten_frame["world_xy"] = [float(xy[0]), float(xy[1])]
            rewritten_by_frame[frame_idx] = [float(xy[0]), float(xy[1])]
            frame_count += 1
            stance = frame_idx in stance_frames
            fs = frame_signals[frame_idx]
            placement_frames.append(
                {
                    "frame_idx": int(frame_idx),
                    "t": float(frame.get("t", frame_idx / fps)),
                    "original_world_xy": original_by_frame[frame_idx],
                    "fused_world_xy": fs.fused_xy,
                    "smoothed_world_xy": [float(xy[0]), float(xy[1])],
                    "covariance_m2": smoothed_frame.covariance,
                    "stance": bool(stance),
                    "signals": fs.signals,
                    "source_counts": fs.source_counts,
                }
            )
        placement_players.append({"id": player_id, "frames": placement_frames})
        jitter_summary[str(player_id)] = _jitter_before_after(frames, rewritten_frames, fps=fps)
        player_stance_wobble = _stance_wobble_before_after(original_by_frame, rewritten_by_frame, stance_frames)
        if player_stance_wobble["phase_count"] > 0:
            stance_wobble_summary[str(player_id)] = player_stance_wobble

    coverage_unchanged = _track_frame_count(tracks_payload) == _track_frame_count(rewritten_payload)
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
        "refine_from_sam3d": bool(refine_from_sam3d),
        "undistort_applied": undistort_applied,
        "source_counts": dict(total_source_counts),
    }
    rewritten_payload["placement_provenance"] = provenance

    placement_payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_placement",
        "fps": fps,
        "source": "threed.racketsport.placement",
        "tracks_path": tracks_path.name,
        "backup_tracks_path": backup_path.name,
        "refine_from_sam3d": bool(refine_from_sam3d),
        "undistort_applied": undistort_applied,
        "players": placement_players,
        "summary": {
            "player_count": len(placement_players),
            "frame_count": frame_count,
            "coverage_unchanged": coverage_unchanged,
            "source_counts": dict(total_source_counts),
            "jitter_before_after_mps": jitter_summary,
            "stance_wobble_before_after_m": stance_wobble_summary,
            "court_bounds_violations": total_bounds_violations,
        },
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
    )


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
    process_noise_mps2: float = 0.1,
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
    config: PlacementConfig,
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    bbox = _bbox(frame.get("bbox"))
    height_norm = (bbox[3] - bbox[1]) / max(height95, 1e-6)
    bbox_pixel = [(bbox[0] + bbox[2]) / 2.0, bbox[3]]
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
        sigma = _keypoint_sigma_px(observation.confidence, base_sigma_px=config.keypoint_base_sigma_px, config=config)
        if str(side).lower() == "far":
            sigma *= config.far_keypoint_sigma_multiplier
        signals.append(
            _signal_from_pixel(
                name=source_name,
                pixel_xy=observation.pixel_xy,
                confidence=observation.confidence,
                sigma_px=sigma,
                side=side,
                homography=homography,
                camera_matrix=camera_matrix,
                dist=dist,
                undistort_applied=undistort_applied,
                config=config,
            )
        )
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


def _signals_for_artifact(signals: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for signal in signals:
        cov = signal.get("covariance_m2")
        sigma = signal.get("sigma_m")
        out.append(
            {
                "name": str(signal["name"]),
                "xy": signal.get("xy"),
                "sigma_m": sigma if sigma is not None else None,
                "used": bool(signal.get("used", False)),
                "reason": signal.get("reason"),
            }
        )
        if cov is not None:
            out[-1]["covariance_m2"] = cov
    return out


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
                out[(player_id, frame_idx)] = _PixelObservation("native2d", combined[0], combined[1])
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
                out[(player_id, frame_idx)] = _PixelObservation("sam3d", combined[0], combined[1])
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
) -> set[int]:
    stance: set[int] = set()
    sorted_indices = sorted(frame_indices)
    for prev, current in zip(sorted_indices, sorted_indices[1:], strict=False):
        if prev not in keypoint_xy or current not in keypoint_xy:
            continue
        dt = max((current - prev) / fps, 1.0 / fps)
        speed = math.hypot(
            float(keypoint_xy[current][0]) - float(keypoint_xy[prev][0]),
            float(keypoint_xy[current][1]) - float(keypoint_xy[prev][1]),
        ) / dt
        if speed <= config.stance_speed_mps:
            stance.add(prev)
            stance.add(current)
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
