from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import cv2
import numpy as np


IDENTITY_3X3: list[list[float]] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
CAMERA_MOTION_AUTO_THRESHOLD = 2.5
CAMERA_MOTION_PROBE_FRAME_STEP = 75
CAMERA_MOTION_PROBE_MAX_FRAMES = 4
CAMERA_MOTION_PROBE_PROCESSING_SCALE = 0.3
CAMERA_MOTION_PROBE_MAX_CORNERS = 250
CAMERA_MOTION_DECODE_ORIENTATION_AUTO_REQUEST = 1


@dataclass(frozen=True)
class CameraMotionParams:
    estimator_mode: str = "hardened"
    use_person_masks: bool = True
    flow_backend: str = "lk"
    processing_scale: float = 0.6
    flow_mad_filter: bool = True
    flow_mad_z: float = 8.0
    min_flow_mad_survivors: int = 12
    temporal_smoothing: bool = True
    temporal_mad_z: float = 4.0
    temporal_gaussian_sigma_frames: float = 0.4
    temporal_gaussian_radius_frames: int = 2
    max_corners: int = 900
    quality_level: float = 0.01
    min_distance: float = 8.0
    block_size: int = 5
    apron_px: int = 80
    person_padding_px: int = 24
    lk_win_size: int = 31
    lk_max_level: int = 3
    lk_max_error_px: float = 80.0
    max_ransac_reproj_px: float = 3.0
    ransac_max_iters: int = 3000
    ransac_confidence: float = 0.995
    min_homography_inliers: int = 16
    min_similarity_inliers: int = 6
    max_homography_rms_px: float = 5.0
    max_similarity_rms_px: float = 5.0
    min_phase_response: float = 0.1
    min_area_scale: float = 0.9
    max_area_scale: float = 1.1
    max_corner_motion_px: float = 110.0
    max_projective_abs: float = 0.002
    rng_seed: int = 1337

    @classmethod
    def legacy(cls) -> "CameraMotionParams":
        return cls(
            estimator_mode="legacy",
            flow_mad_filter=False,
            temporal_smoothing=False,
            use_person_masks=True,
            flow_backend="lk",
            processing_scale=1.0,
        )


def estimate_camera_motion(
    video_path: str | Path,
    calibration_path: str | Path,
    *,
    tracks_path: str | Path | None = None,
    reference_frame_idx: int | None = None,
    max_frames: int | None = None,
    diagnostics_dir: str | Path | None = None,
    params: CameraMotionParams | None = None,
) -> dict[str, Any]:
    params = params or CameraMotionParams()
    _validate_params(params)
    video = Path(video_path)
    calibration_file = Path(calibration_path)
    calibration = _load_json(calibration_file)
    reference_idx = _reference_frame_idx(calibration, override=reference_frame_idx)
    tracks_by_frame = load_tracks_by_frame(tracks_path) if tracks_path is not None and params.use_person_masks else {}

    cap, orientation_policy = _open_camera_motion_capture(video)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video}")
    try:
        frame_count = _frame_count(cap, max_frames=max_frames)
        if frame_count <= 0:
            raise ValueError(f"video has no readable frames: {video}")
        if reference_idx < 0 or reference_idx >= frame_count:
            raise ValueError(f"reference frame {reference_idx} outside processed frame range 0..{frame_count - 1}")
        reference_bgr = _read_frame_at(cap, reference_idx)
        if reference_bgr is None:
            raise ValueError(f"could not read reference frame {reference_idx} from {video}")
        decode_telemetry = _capture_decode_telemetry(cap, orientation_policy, reference_bgr)
        decode_telemetry.update(_decode_orientation_policy_status(decode_telemetry, calibration))

        frame_shape = reference_bgr.shape[:2]
        processing_scale = float(params.processing_scale)
        estimation_params = _params_for_processing_scale(params, processing_scale)
        court_mask = build_court_mask(calibration, frame_shape, apron_px=params.apron_px)
        reference_mask_full = mask_people_for_frame(
            court_mask,
            tracks_by_frame,
            frame_idx=reference_idx,
            padding_px=params.person_padding_px,
        ) if params.use_person_masks else np.array(court_mask, copy=True)
        reference_bgr_est = _resize_for_processing(reference_bgr, processing_scale, is_mask=False)
        reference_mask = _resize_for_processing(reference_mask_full, processing_scale, is_mask=True)
        reference_gray = cv2.cvtColor(reference_bgr_est, cv2.COLOR_BGR2GRAY)
        reference_points = detect_reference_features(reference_bgr_est, reference_mask, estimation_params)
        court_polygon = court_reference_polygon(calibration, frame_shape)
        court_polygon_est = court_polygon * processing_scale
        centroid = _polygon_centroid(court_polygon)

        if len(reference_points) < params.min_similarity_inliers:
            frames = [
                _identity_frame(i, compensated=False, reason="too_few_reference_features")
                for i in range(frame_count)
            ]
            payload = _payload(
                video=video,
                reference_idx=reference_idx,
                params=params,
                frames=frames,
                drift_values=[0.0 for _ in frames],
                residual_values=[],
                smoothing_stats=_empty_smoothing_stats(),
                decode_telemetry=decode_telemetry,
                reference_feature_count=len(reference_points),
            )
            if diagnostics_dir is not None:
                _write_diagnostics(
                    video,
                    calibration,
                    payload,
                    diagnostics_dir=Path(diagnostics_dir),
                    selected_frames=[0, reference_idx],
                )
            validate_camera_motion_payload(payload)
            return payload

        frames: list[dict[str, Any]] = []
        drift_values: list[float] = []
        residual_values: list[float] = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_idx in range(frame_count):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx == reference_idx:
                entry = {
                    "frame_idx": frame_idx,
                    "M": _matrix_to_json(np.eye(3, dtype=np.float64)),
                    "inlier_count": int(len(reference_points)),
                    "rms_px": 0.0,
                    "compensated": True,
                    "model": "identity",
                    "flow_raw_track_count": int(len(reference_points)),
                    "flow_track_count": int(len(reference_points)),
                    "flow_mad_filtered_count": 0,
                    "inlier_ratio": 1.0 if len(reference_points) else 0.0,
                }
                frames.append(entry)
                drift_values.append(0.0)
                residual_values.append(0.0)
                continue

            frame_bgr_est = _resize_for_processing(frame_bgr, processing_scale, is_mask=False)
            frame_gray = cv2.cvtColor(frame_bgr_est, cv2.COLOR_BGR2GRAY)
            current_mask_full = mask_people_for_frame(
                court_mask,
                tracks_by_frame,
                frame_idx=frame_idx,
                padding_px=params.person_padding_px,
            ) if params.use_person_masks else np.array(court_mask, copy=True)
            current_mask = _resize_for_processing(current_mask_full, processing_scale, is_mask=True)
            estimate = _estimate_frame_transform(
                reference_gray,
                frame_gray,
                reference_points,
                current_mask,
                court_polygon_est,
                estimation_params,
            )
            estimate = _rescale_estimate_to_original(estimate, processing_scale)
            frames.append(_frame_entry(frame_idx, estimate))
            drift_values.append(_motion_magnitude_at_point(estimate.matrix_for_metrics, centroid))
            if estimate.compensated:
                residual_values.append(estimate.rms_px)

        frames, smoothing_stats = _smooth_camera_motion_frames(frames, params, reference_frame_idx=reference_idx)
        drift_values = [
            _motion_magnitude_at_point(np.asarray(frame["M"], dtype=np.float64), centroid)
            for frame in frames
        ]
        payload = _payload(
            video=video,
            reference_idx=reference_idx,
            params=params,
            frames=frames,
            drift_values=drift_values,
            residual_values=residual_values,
            smoothing_stats=smoothing_stats,
            decode_telemetry=decode_telemetry,
            reference_feature_count=len(reference_points),
        )
        if diagnostics_dir is not None:
            max_drift_idx = _max_drift_frame_idx(payload["frames"], drift_values)
            _write_diagnostics(
                video,
                calibration,
                payload,
                diagnostics_dir=Path(diagnostics_dir),
                selected_frames=[0, reference_idx, max_drift_idx],
            )
        validate_camera_motion_payload(payload)
        return payload
    finally:
        cap.release()


def estimate_camera_motion_probe(
    video_path: str | Path,
    calibration_path: str | Path,
    *,
    tracks_path: str | Path | None = None,
    reference_frame_idx: int | None = None,
    params: CameraMotionParams | None = None,
    frame_step: int = CAMERA_MOTION_PROBE_FRAME_STEP,
    max_probe_frames: int = CAMERA_MOTION_PROBE_MAX_FRAMES,
    threshold: float = CAMERA_MOTION_AUTO_THRESHOLD,
) -> dict[str, Any]:
    if params is None:
        params = replace(
            CameraMotionParams(),
            processing_scale=CAMERA_MOTION_PROBE_PROCESSING_SCALE,
            max_corners=CAMERA_MOTION_PROBE_MAX_CORNERS,
            temporal_smoothing=False,
        )
    _validate_params(params)
    video = Path(video_path)
    calibration_file = Path(calibration_path)
    calibration = _load_json(calibration_file)
    reference_idx = _reference_frame_idx(calibration, override=reference_frame_idx)
    tracks_by_frame = load_tracks_by_frame(tracks_path) if tracks_path is not None and params.use_person_masks else {}
    started = time.monotonic()

    cap, orientation_policy = _open_camera_motion_capture(video)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video}")
    try:
        frame_count = _frame_count(cap, max_frames=None)
        if frame_count <= 0:
            raise ValueError(f"video has no readable frames: {video}")
        if reference_idx < 0 or reference_idx >= frame_count:
            raise ValueError(f"reference frame {reference_idx} outside processed frame range 0..{frame_count - 1}")
        reference_bgr = _read_frame_at(cap, reference_idx)
        if reference_bgr is None:
            raise ValueError(f"could not read reference frame {reference_idx} from {video}")
        decode_telemetry = _capture_decode_telemetry(cap, orientation_policy, reference_bgr)
        decode_telemetry.update(_decode_orientation_policy_status(decode_telemetry, calibration))

        frame_shape = reference_bgr.shape[:2]
        processing_scale = float(params.processing_scale)
        estimation_params = _params_for_processing_scale(params, processing_scale)
        court_mask = build_court_mask(calibration, frame_shape, apron_px=params.apron_px)
        reference_mask_full = mask_people_for_frame(
            court_mask,
            tracks_by_frame,
            frame_idx=reference_idx,
            padding_px=params.person_padding_px,
        ) if params.use_person_masks else np.array(court_mask, copy=True)
        reference_bgr_est = _resize_for_processing(reference_bgr, processing_scale, is_mask=False)
        reference_mask = _resize_for_processing(reference_mask_full, processing_scale, is_mask=True)
        reference_gray = cv2.cvtColor(reference_bgr_est, cv2.COLOR_BGR2GRAY)
        reference_points = detect_reference_features(reference_bgr_est, reference_mask, estimation_params)
        sample_indices = _probe_frame_indices(
            frame_count,
            reference_idx=reference_idx,
            frame_step=frame_step,
            max_probe_frames=max_probe_frames,
        )
        court_polygon = court_reference_polygon(calibration, frame_shape)
        court_polygon_est = court_polygon * processing_scale
        centroid = _polygon_centroid(court_polygon)

        drift_values: list[float] = []
        compensated_count = 0
        failure_reasons: dict[str, int] = {}
        if len(reference_points) < params.min_similarity_inliers:
            failure_reasons["too_few_reference_features"] = len(sample_indices)
        else:
            for frame_idx in sample_indices:
                if frame_idx == reference_idx:
                    drift_values.append(0.0)
                    compensated_count += 1
                    continue
                frame_bgr = _read_frame_at(cap, frame_idx)
                if frame_bgr is None:
                    failure_reasons["frame_read_failed"] = failure_reasons.get("frame_read_failed", 0) + 1
                    continue
                frame_bgr_est = _resize_for_processing(frame_bgr, processing_scale, is_mask=False)
                frame_gray = cv2.cvtColor(frame_bgr_est, cv2.COLOR_BGR2GRAY)
                current_mask_full = mask_people_for_frame(
                    court_mask,
                    tracks_by_frame,
                    frame_idx=frame_idx,
                    padding_px=params.person_padding_px,
                ) if params.use_person_masks else np.array(court_mask, copy=True)
                current_mask = _resize_for_processing(current_mask_full, processing_scale, is_mask=True)
                estimate = _estimate_frame_transform(
                    reference_gray,
                    frame_gray,
                    reference_points,
                    current_mask,
                    court_polygon_est,
                    estimation_params,
                )
                estimate = _rescale_estimate_to_original(estimate, processing_scale)
                drift_values.append(_motion_magnitude_at_point(estimate.matrix_for_metrics, centroid))
                if estimate.compensated:
                    compensated_count += 1
                else:
                    reason = estimate.reason or "uncompensated"
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        motion_score = _percentile(drift_values, 95)
        orientation_untrusted = bool(decode_telemetry["orientation_policy_untrusted"])
        if orientation_untrusted:
            reason = str(decode_telemetry["orientation_policy_mismatch_reason"])
            failure_reasons[reason] = int(len(sample_indices))
        enabled = bool(motion_score > float(threshold)) and not orientation_untrusted
        forced = (
            f"auto_decode_orientation_untrusted:{decode_telemetry['orientation_policy_mismatch_reason']}"
            if orientation_untrusted
            else "auto"
        )
        return {
            "schema_version": 1,
            "artifact_type": "racketsport_camera_motion_probe",
            "video": video.as_posix(),
            "reference_frame_idx": int(reference_idx),
            "method": f"{_method_name(params)}_probe",
            "params": {
                **asdict(params),
                "frame_step": int(max(1, frame_step)),
                "max_probe_frames": int(max(1, max_probe_frames)),
            },
            "threshold": _round_float(float(threshold)),
            "motion_score": motion_score,
            "enabled": enabled,
            "forced": forced,
            "decode_orientation": decode_telemetry,
            "decode_orientation_mismatch": bool(decode_telemetry["orientation_policy_mismatch"]),
            "decode_orientation_consequential_mismatch": bool(
                decode_telemetry["orientation_policy_consequential_mismatch"]
            ),
            "decode_orientation_untrusted": orientation_untrusted,
            "decode_orientation_mismatch_reason": decode_telemetry["orientation_policy_mismatch_reason"],
            "decoded_frame_shape_hwc": decode_telemetry["decoded_frame_shape_hwc"],
            "decoded_frame_width_height": decode_telemetry["decoded_frame_width_height"],
            "reference_feature_count": int(len(reference_points)),
            "sampled_frame_indices": sample_indices,
            "sampled_frame_count": int(len(sample_indices)),
            "frame_step": int(max(1, frame_step)),
            "max_probe_frames": int(max(1, max_probe_frames)),
            "n_compensated": int(compensated_count),
            "drift_px_p50": _percentile(drift_values, 50),
            "drift_px_p95": motion_score,
            "drift_px_max": _max_or_zero(drift_values),
            "failure_reasons": failure_reasons,
            "wall_seconds": _round_float(time.monotonic() - started),
            "verified": False,
            "not_gate_verified": True,
        }
    finally:
        cap.release()


def write_camera_motion_json(payload: Mapping[str, Any], out_path: str | Path) -> None:
    validate_camera_motion_payload(payload)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_camera_motion_payload(payload: Mapping[str, Any]) -> None:
    required_top = {
        "schema_version",
        "artifact_type",
        "video",
        "reference_frame_idx",
        "method",
        "params",
        "frames",
        "summary",
        "verified",
        "not_gate_verified",
    }
    missing = required_top.difference(payload)
    if missing:
        raise ValueError(f"camera motion payload missing keys: {sorted(missing)}")
    if payload["artifact_type"] != "racketsport_camera_motion":
        raise ValueError("artifact_type must be racketsport_camera_motion")
    if payload["verified"] is not False or payload["not_gate_verified"] is not True:
        raise ValueError("camera motion payload must remain verified=false and not_gate_verified=true")
    if not isinstance(payload["reference_frame_idx"], int):
        raise ValueError("reference_frame_idx must be an int")
    frames = payload["frames"]
    if not isinstance(frames, list):
        raise ValueError("frames must be a list")
    for frame in frames:
        _validate_frame(frame)
    summary = payload["summary"]
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be an object")
    for key in (
        "n_frames",
        "n_compensated",
        "drift_px_p50",
        "drift_px_p95",
        "drift_px_max",
        "residual_px_p50",
        "residual_px_p95",
        "residual_px_max",
    ):
        if key not in summary:
            raise ValueError(f"summary missing {key}")


def build_court_mask(calibration: Mapping[str, Any], frame_shape: tuple[int, int], *, apron_px: int) -> np.ndarray:
    height, width = frame_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = court_reference_polygon(calibration, frame_shape)
    if len(polygon) < 3:
        mask[:] = 255
    else:
        cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 255)
    if apron_px > 0:
        kernel_size = max(1, int(apron_px) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def court_reference_polygon(calibration: Mapping[str, Any], frame_shape: tuple[int, int]) -> np.ndarray:
    image_pts = _points2(calibration.get("image_pts"))
    if len(image_pts) >= 3:
        return _convex_hull(image_pts)

    world_pts = _points2(calibration.get("world_pts"))
    homography = np.asarray(calibration.get("homography", []), dtype=np.float64)
    if len(world_pts) >= 3 and homography.shape == (3, 3):
        min_xy = world_pts.min(axis=0)
        max_xy = world_pts.max(axis=0)
        court_world = np.array(
            [
                [min_xy[0], min_xy[1]],
                [max_xy[0], min_xy[1]],
                [max_xy[0], max_xy[1]],
                [min_xy[0], max_xy[1]],
            ],
            dtype=np.float64,
        )
        return _convex_hull(_apply_homography(homography, court_world))

    height, width = frame_shape
    return np.array([[0.0, 0.0], [float(width - 1), 0.0], [float(width - 1), float(height - 1)], [0.0, float(height - 1)]])


def mask_people_for_frame(
    base_mask: np.ndarray,
    tracks_by_frame: Mapping[int, Sequence[Sequence[float]]],
    *,
    frame_idx: int,
    padding_px: int,
) -> np.ndarray:
    mask = np.array(base_mask, copy=True)
    height, width = mask.shape[:2]
    for bbox in tracks_by_frame.get(frame_idx, []):
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        left = max(0, int(math.floor(min(x1, x2) - padding_px)))
        top = max(0, int(math.floor(min(y1, y2) - padding_px)))
        right = min(width - 1, int(math.ceil(max(x1, x2) + padding_px)))
        bottom = min(height - 1, int(math.ceil(max(y1, y2) + padding_px)))
        if right >= left and bottom >= top:
            mask[top : bottom + 1, left : right + 1] = 0
    return mask


def detect_reference_features(image_bgr: np.ndarray, mask: np.ndarray, params: CameraMotionParams) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    cv2.setRNGSeed(params.rng_seed)
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=int(params.max_corners),
        qualityLevel=float(params.quality_level),
        minDistance=float(params.min_distance),
        mask=mask,
        blockSize=int(params.block_size),
        useHarrisDetector=False,
    )
    if points is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def load_tracks_by_frame(tracks_path: str | Path | None) -> dict[int, list[list[float]]]:
    if tracks_path is None:
        return {}
    payload = _load_json(Path(tracks_path))
    fps = float(payload.get("fps") or 30.0)
    by_frame: dict[int, list[list[float]]] = {}
    for player in payload.get("players", []) if isinstance(payload.get("players"), list) else []:
        frames = player.get("frames", []) if isinstance(player, Mapping) else []
        for item in frames if isinstance(frames, list) else []:
            if not isinstance(item, Mapping):
                continue
            bbox = item.get("bbox")
            if not _is_number_sequence(bbox, min_len=4):
                continue
            frame_idx = _track_frame_idx(item, fps=fps)
            if frame_idx is None:
                continue
            by_frame.setdefault(frame_idx, []).append([float(v) for v in bbox[:4]])
    return by_frame


@dataclass(frozen=True)
class _FrameEstimate:
    matrix: np.ndarray
    matrix_for_metrics: np.ndarray
    inlier_count: int
    rms_px: float
    compensated: bool
    model: str
    reason: str | None = None
    flow_raw_track_count: int = 0
    flow_track_count: int = 0
    flow_mad_filtered_count: int = 0


def _estimate_frame_transform(
    reference_gray: np.ndarray,
    frame_gray: np.ndarray,
    reference_points: np.ndarray,
    current_mask: np.ndarray,
    court_polygon: np.ndarray,
    params: CameraMotionParams,
) -> _FrameEstimate:
    if params.flow_backend == "raft-small":
        status = raft_small_backend_status()
        if status["status"] != "enabled":
            raise RuntimeError(f"raft_backend={status['status']}")
        raise RuntimeError("raft_backend=not_enabled_pending_weights")

    cv2.setRNGSeed(params.rng_seed)
    next_points, status, errors = cv2.calcOpticalFlowPyrLK(
        reference_gray,
        frame_gray,
        reference_points,
        None,
        winSize=(int(params.lk_win_size), int(params.lk_win_size)),
        maxLevel=int(params.lk_max_level),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status is None:
        return _failed_estimate("lk_failed")

    status_flat = status.reshape(-1).astype(bool)
    if errors is not None:
        status_flat &= errors.reshape(-1) <= float(params.lk_max_error_px)
    tracked_ref = reference_points.reshape(-1, 2)
    tracked_cur = next_points.reshape(-1, 2)
    ref = tracked_ref[status_flat]
    cur = tracked_cur[status_flat]
    if len(cur):
        keep = _mask_indices_in_mask(cur, current_mask)
        ref = ref[keep]
        cur = cur[keep]
    flow_raw_track_count = int(len(cur))

    if len(cur) < params.min_similarity_inliers:
        return _failed_estimate(
            "too_few_tracked_features",
            inlier_count=int(len(cur)),
            flow_raw_track_count=flow_raw_track_count,
            flow_track_count=int(len(cur)),
        )

    ref, cur, mad_stats = _mad_filter_flow_tracks(ref, cur, params)
    flow_track_count = int(len(cur))
    flow_mad_filtered_count = int(mad_stats["flow_mad_filtered_count"])

    homography_result = _fit_homography(cur, ref, params)
    similarity_result = _fit_similarity(cur, ref, params)
    phase_result = _fit_phase_translation(reference_gray, frame_gray, current_mask, cur, ref, params)
    valid_candidates: list[_FrameEstimate] = []
    invalid_sanity: list[tuple[int, str]] = []
    for candidate in (homography_result, similarity_result, phase_result):
        if candidate is None or not _is_valid_candidate(candidate, params):
            continue
        reason = transform_sanity_reason(candidate.matrix, court_polygon, params)
        if reason is None:
            valid_candidates.append(candidate)
        else:
            invalid_sanity.append((candidate.inlier_count, reason))
    if valid_candidates:
        return _with_flow_stats(
            _best_candidate(valid_candidates),
            flow_raw_track_count=flow_raw_track_count,
            flow_track_count=flow_track_count,
            flow_mad_filtered_count=flow_mad_filtered_count,
        )
    if invalid_sanity:
        inlier_count, reason = max(invalid_sanity, key=lambda item: item[0])
        return _failed_estimate(
            reason,
            inlier_count=inlier_count,
            flow_raw_track_count=flow_raw_track_count,
            flow_track_count=flow_track_count,
            flow_mad_filtered_count=flow_mad_filtered_count,
        )

    best = homography_result or similarity_result
    if best is None:
        return _failed_estimate(
            "transform_fit_failed",
            inlier_count=int(len(cur)),
            flow_raw_track_count=flow_raw_track_count,
            flow_track_count=flow_track_count,
            flow_mad_filtered_count=flow_mad_filtered_count,
        )
    if best.rms_px > max(params.max_homography_rms_px, params.max_similarity_rms_px):
        return _failed_estimate(
            "rms_above_threshold",
            inlier_count=best.inlier_count,
            metric_matrix=best.matrix,
            flow_raw_track_count=flow_raw_track_count,
            flow_track_count=flow_track_count,
            flow_mad_filtered_count=flow_mad_filtered_count,
        )
    return _failed_estimate(
        "too_few_transform_inliers",
        inlier_count=best.inlier_count,
        metric_matrix=best.matrix,
        flow_raw_track_count=flow_raw_track_count,
        flow_track_count=flow_track_count,
        flow_mad_filtered_count=flow_mad_filtered_count,
    )


def _is_valid_candidate(candidate: _FrameEstimate, params: CameraMotionParams) -> bool:
    if candidate.model == "homography":
        return candidate.inlier_count >= params.min_homography_inliers and candidate.rms_px <= params.max_homography_rms_px
    return candidate.inlier_count >= params.min_similarity_inliers and candidate.rms_px <= params.max_similarity_rms_px


def _best_candidate(candidates: Sequence[_FrameEstimate]) -> _FrameEstimate:
    priority = {"homography": 2, "similarity": 1, "identity": 0}
    strongest_support = max(
        candidates,
        key=lambda candidate: (
            candidate.inlier_count,
            -candidate.rms_px,
            priority.get(candidate.model, 0),
        ),
    )
    lowest_residual = min(candidates, key=lambda candidate: (candidate.rms_px, -candidate.inlier_count))
    if (
        lowest_residual.inlier_count >= 6
        and lowest_residual.rms_px + 0.5 < strongest_support.rms_px
    ):
        return lowest_residual
    return strongest_support


def transform_sanity_reason(matrix: np.ndarray, reference_polygon: np.ndarray, params: CameraMotionParams) -> str | None:
    candidate = np.asarray(matrix, dtype=np.float64)
    if candidate.shape != (3, 3) or not np.all(np.isfinite(candidate)):
        return "nonfinite_transform"
    if np.max(np.abs(candidate[2, :2])) > params.max_projective_abs:
        return "implausible_projective_terms"
    polygon = np.asarray(reference_polygon, dtype=np.float64).reshape(-1, 2)
    if len(polygon) < 3:
        return None
    transformed = _apply_homography(candidate, polygon)
    if not np.all(np.isfinite(transformed)):
        return "nonfinite_transformed_court"
    base_area = abs(float(cv2.contourArea(polygon.astype(np.float32))))
    transformed_area = abs(float(cv2.contourArea(transformed.astype(np.float32))))
    if base_area <= 1e-6 or transformed_area <= 1e-6:
        return "degenerate_court_area"
    area_scale = transformed_area / base_area
    if area_scale < params.min_area_scale or area_scale > params.max_area_scale:
        return "implausible_area_scale"
    max_motion = float(np.max(np.linalg.norm(transformed - polygon, axis=1)))
    if max_motion > params.max_corner_motion_px:
        return "implausible_corner_motion"
    return None


def _fit_homography(cur: np.ndarray, ref: np.ndarray, params: CameraMotionParams) -> _FrameEstimate | None:
    if len(cur) < 4:
        return None
    cv2.setRNGSeed(params.rng_seed)
    matrix, inliers = cv2.findHomography(
        cur.astype(np.float32),
        ref.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(params.max_ransac_reproj_px),
        maxIters=int(params.ransac_max_iters),
        confidence=float(params.ransac_confidence),
    )
    if matrix is None or inliers is None:
        return None
    inlier_mask = inliers.reshape(-1).astype(bool)
    count, rms = _inlier_rms(matrix, cur, ref, inlier_mask)
    return _FrameEstimate(
        matrix=_normalize_homography(matrix),
        matrix_for_metrics=_normalize_homography(matrix),
        inlier_count=count,
        rms_px=rms,
        compensated=True,
        model="homography",
    )


def _fit_similarity(cur: np.ndarray, ref: np.ndarray, params: CameraMotionParams) -> _FrameEstimate | None:
    if len(cur) < 2:
        return None
    cv2.setRNGSeed(params.rng_seed)
    affine, inliers = cv2.estimateAffinePartial2D(
        cur.astype(np.float32),
        ref.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=float(params.max_ransac_reproj_px),
        maxIters=int(params.ransac_max_iters),
        confidence=float(params.ransac_confidence),
        refineIters=10,
    )
    if affine is None or inliers is None:
        return None
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :] = affine
    inlier_mask = inliers.reshape(-1).astype(bool)
    count, rms = _inlier_rms(matrix, cur, ref, inlier_mask)
    return _FrameEstimate(
        matrix=_normalize_homography(matrix),
        matrix_for_metrics=_normalize_homography(matrix),
        inlier_count=count,
        rms_px=rms,
        compensated=True,
        model="similarity",
    )


def _fit_phase_translation(
    reference_gray: np.ndarray,
    frame_gray: np.ndarray,
    current_mask: np.ndarray,
    cur: np.ndarray,
    ref: np.ndarray,
    params: CameraMotionParams,
) -> _FrameEstimate | None:
    if len(cur) < params.min_similarity_inliers or np.count_nonzero(current_mask) == 0:
        return None
    ref_f = reference_gray.astype(np.float32)
    frame_f = frame_gray.astype(np.float32)
    mask_f = (current_mask > 0).astype(np.float32)
    if mask_f.shape != ref_f.shape:
        return None
    window = cv2.createHanningWindow((ref_f.shape[1], ref_f.shape[0]), cv2.CV_32F)
    try:
        (dx, dy), response = cv2.phaseCorrelate(ref_f * mask_f, frame_f * mask_f, window)
    except cv2.error:
        return None
    if not math.isfinite(dx) or not math.isfinite(dy) or response < params.min_phase_response:
        return None
    matrix = np.array([[1.0, 0.0, -float(dx)], [0.0, 1.0, -float(dy)], [0.0, 0.0, 1.0]], dtype=np.float64)
    residuals = np.linalg.norm(_apply_homography(matrix, cur) - ref, axis=1)
    inlier_mask = residuals <= params.max_ransac_reproj_px
    count = int(np.count_nonzero(inlier_mask))
    if count <= 0:
        return None
    rms = float(np.sqrt(np.mean(np.square(residuals[inlier_mask]))))
    return _FrameEstimate(
        matrix=matrix,
        matrix_for_metrics=matrix,
        inlier_count=count,
        rms_px=rms,
        compensated=True,
        model="similarity",
    )


def _mad_filter_flow_tracks(
    ref: np.ndarray,
    cur: np.ndarray,
    params: CameraMotionParams,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    ref_arr = np.asarray(ref, dtype=np.float32).reshape(-1, 2)
    cur_arr = np.asarray(cur, dtype=np.float32).reshape(-1, 2)
    count = int(min(len(ref_arr), len(cur_arr)))
    ref_arr = ref_arr[:count]
    cur_arr = cur_arr[:count]
    stats = {"flow_track_count": count, "flow_mad_filtered_count": 0}
    if (
        not params.flow_mad_filter
        or count < max(3, int(params.min_flow_mad_survivors))
    ):
        return ref_arr, cur_arr, stats

    flow = cur_arr - ref_arr
    median_flow = np.median(flow, axis=0)
    radial = np.linalg.norm(flow - median_flow, axis=1)
    radial_median = float(np.median(radial))
    mad = float(np.median(np.abs(radial - radial_median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= 1e-9:
        threshold = radial_median + 1e-6
    else:
        threshold = radial_median + float(params.flow_mad_z) * robust_sigma
    keep = radial <= threshold
    survivors = int(np.count_nonzero(keep))
    if survivors < int(params.min_flow_mad_survivors):
        return ref_arr, cur_arr, stats
    stats["flow_mad_filtered_count"] = int(count - survivors)
    return ref_arr[keep], cur_arr[keep], stats


def _tracks_from_dense_flow(
    reference_points: np.ndarray,
    dense_flow_xy: np.ndarray,
    current_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    points = np.asarray(reference_points, dtype=np.float32).reshape(-1, 2)
    flow = np.asarray(dense_flow_xy, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("dense_flow_xy must have shape HxWx2")
    height, width = flow.shape[:2]
    if current_mask.shape[:2] != (height, width):
        raise ValueError("current_mask shape must match dense_flow_xy")
    ref: list[list[float]] = []
    cur: list[list[float]] = []
    for x, y in points:
        ix = int(round(float(x)))
        iy = int(round(float(y)))
        if not (0 <= ix < width and 0 <= iy < height):
            continue
        dx, dy = flow[iy, ix]
        cx = float(x + dx)
        cy = float(y + dy)
        cix = int(round(cx))
        ciy = int(round(cy))
        if 0 <= cix < width and 0 <= ciy < height and current_mask[ciy, cix] > 0:
            ref.append([float(x), float(y)])
            cur.append([cx, cy])
    stats = {
        "flow_backend": "dense",
        "flow_track_count": len(ref),
        "flow_mad_filtered_count": 0,
    }
    return np.asarray(ref, dtype=np.float32), np.asarray(cur, dtype=np.float32), stats


def raft_small_backend_status() -> dict[str, Any]:
    try:
        import torch  # type: ignore
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small  # type: ignore  # noqa: F401
    except Exception as exc:
        return {
            "backend": "raft-small",
            "status": "not_available",
            "enabled": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    weights = Raft_Small_Weights.C_T_V2
    filename = os.path.basename(urlparse(str(weights.url)).path)
    checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if checkpoint_path.exists():
        return {
            "backend": "raft-small",
            "status": "enabled",
            "enabled": True,
            "weights": weights.name,
            "checkpoint_path": checkpoint_path.as_posix(),
        }
    return {
        "backend": "raft-small",
        "status": "not_enabled_pending_weights",
        "enabled": False,
        "weights": weights.name,
        "expected_checkpoint_path": checkpoint_path.as_posix(),
    }


def _failed_estimate(
    reason: str,
    *,
    inlier_count: int = 0,
    metric_matrix: np.ndarray | None = None,
    flow_raw_track_count: int = 0,
    flow_track_count: int = 0,
    flow_mad_filtered_count: int = 0,
) -> _FrameEstimate:
    identity = np.eye(3, dtype=np.float64)
    return _FrameEstimate(
        matrix=identity,
        matrix_for_metrics=metric_matrix if metric_matrix is not None else identity,
        inlier_count=int(inlier_count),
        rms_px=0.0,
        compensated=False,
        model="identity",
        reason=reason,
        flow_raw_track_count=int(flow_raw_track_count),
        flow_track_count=int(flow_track_count),
        flow_mad_filtered_count=int(flow_mad_filtered_count),
    )


def _with_flow_stats(
    estimate: _FrameEstimate,
    *,
    flow_raw_track_count: int,
    flow_track_count: int,
    flow_mad_filtered_count: int,
) -> _FrameEstimate:
    return _FrameEstimate(
        matrix=estimate.matrix,
        matrix_for_metrics=estimate.matrix_for_metrics,
        inlier_count=estimate.inlier_count,
        rms_px=estimate.rms_px,
        compensated=estimate.compensated,
        model=estimate.model,
        reason=estimate.reason,
        flow_raw_track_count=int(flow_raw_track_count),
        flow_track_count=int(flow_track_count),
        flow_mad_filtered_count=int(flow_mad_filtered_count),
    )


def _rescale_estimate_to_original(estimate: _FrameEstimate, scale: float) -> _FrameEstimate:
    if abs(float(scale) - 1.0) <= 1e-12:
        return estimate
    matrix = _rescale_matrix_to_original(estimate.matrix, scale)
    metric_matrix = _rescale_matrix_to_original(estimate.matrix_for_metrics, scale)
    return _FrameEstimate(
        matrix=matrix,
        matrix_for_metrics=metric_matrix,
        inlier_count=estimate.inlier_count,
        rms_px=_round_float(estimate.rms_px / scale) if scale > 1e-12 else estimate.rms_px,
        compensated=estimate.compensated,
        model=estimate.model,
        reason=estimate.reason,
        flow_raw_track_count=estimate.flow_raw_track_count,
        flow_track_count=estimate.flow_track_count,
        flow_mad_filtered_count=estimate.flow_mad_filtered_count,
    )


def _rescale_matrix_to_original(matrix: np.ndarray, scale: float) -> np.ndarray:
    scale = float(scale)
    to_scaled = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    to_original = np.array([[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return _normalize_homography(to_original @ np.asarray(matrix, dtype=np.float64) @ to_scaled)


def _mask_indices_in_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    keep = []
    for x, y in points:
        ix = int(round(float(x)))
        iy = int(round(float(y)))
        keep.append(0 <= ix < width and 0 <= iy < height and mask[iy, ix] > 0)
    return np.asarray(keep, dtype=bool)


def _frame_entry(frame_idx: int, estimate: _FrameEstimate) -> dict[str, Any]:
    inlier_ratio = (
        float(estimate.inlier_count) / float(estimate.flow_track_count)
        if estimate.flow_track_count > 0
        else 0.0
    )
    entry: dict[str, Any] = {
        "frame_idx": int(frame_idx),
        "M": _matrix_to_json(estimate.matrix),
        "inlier_count": int(estimate.inlier_count),
        "rms_px": _round_float(estimate.rms_px),
        "compensated": bool(estimate.compensated),
        "model": estimate.model,
        "flow_raw_track_count": int(estimate.flow_raw_track_count),
        "flow_track_count": int(estimate.flow_track_count),
        "flow_mad_filtered_count": int(estimate.flow_mad_filtered_count),
        "inlier_ratio": _round_float(inlier_ratio),
    }
    if estimate.reason is not None:
        entry["reason"] = estimate.reason
    return entry


def _identity_frame(frame_idx: int, *, compensated: bool, reason: str) -> dict[str, Any]:
    return {
        "frame_idx": int(frame_idx),
        "M": IDENTITY_3X3,
        "inlier_count": 0,
        "rms_px": 0.0,
        "compensated": bool(compensated),
        "model": "identity",
        "reason": reason,
        "flow_raw_track_count": 0,
        "flow_track_count": 0,
        "flow_mad_filtered_count": 0,
        "inlier_ratio": 0.0,
    }


def _smooth_camera_motion_frames(
    frames: Sequence[Mapping[str, Any]],
    params: CameraMotionParams,
    *,
    reference_frame_idx: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out = [dict(frame) for frame in frames]
    stats = _empty_smoothing_stats()
    if not params.temporal_smoothing:
        return out, stats
    compensated_indices = [
        idx
        for idx, frame in enumerate(out)
        if frame.get("compensated") is True and int(frame.get("frame_idx", -1)) != reference_frame_idx
    ]
    if len(compensated_indices) < 3:
        return out, stats

    vectors = np.asarray(
        [_matrix_to_smooth_vector(np.asarray(out[idx]["M"], dtype=np.float64)) for idx in compensated_indices],
        dtype=np.float64,
    )
    filtered = _replace_temporal_mad_outliers(
        vectors,
        z=float(params.temporal_mad_z),
        radius=max(1, int(params.temporal_gaussian_radius_frames)),
    )
    stats["temporal_mad_replaced_count"] = int(np.count_nonzero(np.any(np.abs(filtered - vectors) > 1e-9, axis=1)))
    smoothed = _gaussian_smooth_vectors(
        filtered,
        sigma=max(float(params.temporal_gaussian_sigma_frames), 1e-6),
        radius=max(1, int(params.temporal_gaussian_radius_frames)),
    )
    stats["temporal_smoothed_frame_count"] = int(len(compensated_indices))
    for row_idx, frame_idx in enumerate(compensated_indices):
        matrix = _smooth_vector_to_matrix(smoothed[row_idx])
        original_matrix = np.asarray(out[frame_idx]["M"], dtype=np.float64)
        out[frame_idx]["M"] = _matrix_to_json(matrix)
        if not np.allclose(matrix, original_matrix, atol=1e-9):
            out[frame_idx]["temporal_smoothed"] = True
    return out, stats


def _replace_temporal_mad_outliers(vectors: np.ndarray, *, z: float, radius: int) -> np.ndarray:
    out = np.array(vectors, copy=True)
    if len(out) < 3:
        return out
    for idx in range(len(vectors)):
        start = max(0, idx - radius)
        end = min(len(vectors), idx + radius + 1)
        local = vectors[start:end]
        if len(local) < 3:
            continue
        median = np.median(local, axis=0)
        distances = np.linalg.norm(local - median, axis=1)
        center_distance = float(np.linalg.norm(vectors[idx] - median))
        distance_median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - distance_median)))
        sigma = 1.4826 * mad
        threshold = distance_median + (z * sigma if sigma > 1e-9 else 1e-6)
        if center_distance > threshold:
            out[idx] = median
    return out


def _gaussian_smooth_vectors(vectors: np.ndarray, *, sigma: float, radius: int) -> np.ndarray:
    out = np.zeros_like(vectors, dtype=np.float64)
    for idx in range(len(vectors)):
        start = max(0, idx - radius)
        end = min(len(vectors), idx + radius + 1)
        offsets = np.arange(start, end, dtype=np.float64) - float(idx)
        weights = np.exp(-0.5 * np.square(offsets / sigma))
        weights_sum = float(np.sum(weights))
        if weights_sum <= 1e-12:
            out[idx] = vectors[idx]
        else:
            out[idx] = np.sum(vectors[start:end] * (weights / weights_sum)[:, None], axis=0)
    return out


def _matrix_to_smooth_vector(matrix: np.ndarray) -> np.ndarray:
    normalized = _normalize_homography(matrix)
    return np.array(
        [
            normalized[0, 0] - 1.0,
            normalized[0, 1],
            normalized[0, 2],
            normalized[1, 0],
            normalized[1, 1] - 1.0,
            normalized[1, 2],
            normalized[2, 0],
            normalized[2, 1],
        ],
        dtype=np.float64,
    )


def _smooth_vector_to_matrix(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64).reshape(8)
    return np.array(
        [
            [1.0 + values[0], values[1], values[2]],
            [values[3], 1.0 + values[4], values[5]],
            [values[6], values[7], 1.0],
        ],
        dtype=np.float64,
    )


def _empty_smoothing_stats() -> dict[str, int]:
    return {
        "temporal_mad_replaced_count": 0,
        "temporal_smoothed_frame_count": 0,
    }


def _method_name(params: CameraMotionParams) -> str:
    if params.estimator_mode == "legacy":
        return "shi_tomasi_reference_lk_ransac_homography_similarity_fallback_legacy"
    return "shi_tomasi_reference_lk_mad_ransac_temporal_smoothing"


def _validate_params(params: CameraMotionParams) -> None:
    if params.estimator_mode not in {"hardened", "legacy"}:
        raise ValueError("estimator_mode must be 'hardened' or 'legacy'")
    if params.flow_backend not in {"lk", "raft-small"}:
        raise ValueError("flow_backend must be 'lk' or 'raft-small'")
    if not (0.0 < float(params.processing_scale) <= 1.0):
        raise ValueError("processing_scale must be in the interval (0, 1]")
    if params.estimator_mode == "legacy" and (params.flow_mad_filter or params.temporal_smoothing):
        raise ValueError("legacy estimator_mode requires flow_mad_filter=false and temporal_smoothing=false")


def _open_camera_motion_capture(video: Path) -> tuple[cv2.VideoCapture, dict[str, Any]]:
    cap = cv2.VideoCapture(str(video))
    return cap, _apply_decode_orientation_policy(cap)


def _apply_decode_orientation_policy(cap: cv2.VideoCapture) -> dict[str, Any]:
    orientation_auto_prop = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    requested = CAMERA_MOTION_DECODE_ORIENTATION_AUTO_REQUEST
    set_ok: bool | None = None
    if orientation_auto_prop is not None:
        try:
            set_ok = bool(cap.set(int(orientation_auto_prop), float(requested)))
        except cv2.error:
            set_ok = False
    return {
        "orientation_auto_property_available": orientation_auto_prop is not None,
        "orientation_auto_requested": int(requested),
        "orientation_auto_set_ok": set_ok,
    }


def _capture_decode_telemetry(
    cap: cv2.VideoCapture,
    orientation_policy: Mapping[str, Any],
    first_decoded_frame: np.ndarray,
) -> dict[str, Any]:
    orientation_auto_prop = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    orientation_meta_prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    height, width = first_decoded_frame.shape[:2]
    telemetry = {
        **dict(orientation_policy),
        "orientation_auto_reported": _capture_prop_float(cap, orientation_auto_prop),
        "orientation_meta": _capture_prop_float(cap, orientation_meta_prop),
        "capture_frame_width_height": [
            _capture_prop_int(cap, cv2.CAP_PROP_FRAME_WIDTH),
            _capture_prop_int(cap, cv2.CAP_PROP_FRAME_HEIGHT),
        ],
        "decoded_frame_shape_hwc": [int(value) for value in first_decoded_frame.shape],
        "decoded_frame_width_height": [int(width), int(height)],
    }
    return telemetry


def _decode_orientation_policy_status(
    decode_telemetry: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    requested = decode_telemetry.get("orientation_auto_requested")
    reported = decode_telemetry.get("orientation_auto_reported")
    set_ok = decode_telemetry.get("orientation_auto_set_ok")
    reasons: list[str] = []
    if decode_telemetry.get("orientation_auto_property_available") is False:
        reasons.append("orientation_auto_property_unavailable")
    if set_ok is False:
        reasons.append("orientation_auto_set_failed")
    if requested is not None and reported is None and decode_telemetry.get("orientation_auto_property_available") is True:
        reasons.append("orientation_auto_readback_unavailable")
    if requested is not None and reported is not None and abs(float(reported) - float(requested)) > 1e-6:
        reasons.append("orientation_auto_readback_mismatch")

    calibration_image_size = _calibration_image_size(calibration)
    decoded_size = decode_telemetry.get("decoded_frame_width_height")
    dims_contradict = (
        calibration_image_size is not None
        and isinstance(decoded_size, Sequence)
        and not isinstance(decoded_size, (str, bytes))
        and [int(v) for v in decoded_size[:2]] != calibration_image_size
    )
    rotation_meta_nonzero = _rotation_meta_nonzero(decode_telemetry.get("orientation_meta"))
    mismatch = bool(reasons)
    consequential = bool(mismatch and (rotation_meta_nonzero or dims_contradict))
    reason_text = ";".join(reasons) if reasons else None
    if consequential and dims_contradict and "decoded_dims_contradict_calibration" not in reasons:
        reason_text = f"{reason_text};decoded_dims_contradict_calibration" if reason_text else "decoded_dims_contradict_calibration"
    return {
        "calibration_image_size": calibration_image_size,
        "decoded_dims_match_calibration_image_size": None if calibration_image_size is None else not dims_contradict,
        "rotation_metadata_nonzero": rotation_meta_nonzero,
        "decoded_dims_contradict_expected_orientation": bool(dims_contradict),
        "orientation_policy_mismatch": mismatch,
        "orientation_policy_consequential_mismatch": consequential,
        "orientation_policy_untrusted": consequential,
        "orientation_policy_mismatch_reason": reason_text,
    }


def _calibration_image_size(calibration: Mapping[str, Any]) -> list[int] | None:
    value = calibration.get("image_size")
    if not _is_number_sequence(value, min_len=2):
        return None
    width = int(round(float(value[0])))
    height = int(round(float(value[1])))
    if width <= 0 or height <= 0:
        return None
    return [width, height]


def _rotation_meta_nonzero(value: Any) -> bool:
    if value is None:
        return False
    try:
        rotation = float(value) % 360.0
    except (TypeError, ValueError):
        return False
    return rotation > 1e-6 and abs(rotation - 360.0) > 1e-6


def _capture_prop_float(cap: cv2.VideoCapture, prop: int | None) -> float | None:
    if prop is None:
        return None
    try:
        value = float(cap.get(int(prop)))
    except (cv2.error, TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return _round_float(value)


def _capture_prop_int(cap: cv2.VideoCapture, prop: int | None) -> int | None:
    value = _capture_prop_float(cap, prop)
    return None if value is None else int(round(value))


def _payload(
    *,
    video: Path,
    reference_idx: int,
    params: CameraMotionParams,
    frames: list[dict[str, Any]],
    drift_values: Sequence[float],
    residual_values: Sequence[float],
    smoothing_stats: Mapping[str, int],
    decode_telemetry: Mapping[str, Any] | None = None,
    reference_feature_count: int | None = None,
) -> dict[str, Any]:
    inlier_ratios = [
        float(frame.get("inlier_ratio", 0.0) or 0.0)
        for frame in frames
        if int(frame.get("flow_track_count", 0) or 0) > 0
    ]
    flow_track_total = int(sum(int(frame.get("flow_track_count", 0) or 0) for frame in frames))
    flow_raw_track_total = int(sum(int(frame.get("flow_raw_track_count", 0) or 0) for frame in frames))
    flow_mad_filtered_total = int(sum(int(frame.get("flow_mad_filtered_count", 0) or 0) for frame in frames))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "racketsport_camera_motion",
        "video": video.as_posix(),
        "reference_frame_idx": int(reference_idx),
        "method": _method_name(params),
        "params": asdict(params),
        "frames": frames,
        "summary": {
            "n_frames": int(len(frames)),
            "n_compensated": int(sum(1 for frame in frames if frame.get("compensated") is True)),
            "drift_px_p50": _percentile(drift_values, 50),
            "drift_px_p95": _percentile(drift_values, 95),
            "drift_px_max": _max_or_zero(drift_values),
            "residual_px_p50": _percentile(residual_values, 50),
            "residual_px_p95": _percentile(residual_values, 95),
            "residual_px_max": _max_or_zero(residual_values),
            "flow_raw_track_count_total": flow_raw_track_total,
            "flow_track_count_total": flow_track_total,
            "flow_mad_filtered_count_total": flow_mad_filtered_total,
            "inlier_ratio_p50": _percentile(inlier_ratios, 50),
            "inlier_ratio_p95": _percentile(inlier_ratios, 95),
            **{key: int(value) for key, value in smoothing_stats.items()},
        },
        "verified": False,
        "not_gate_verified": True,
    }
    if decode_telemetry is not None:
        decode_payload = dict(decode_telemetry)
        payload["decode_orientation"] = decode_payload
        payload["decode_orientation_mismatch"] = bool(decode_payload.get("orientation_policy_mismatch", False))
        payload["decode_orientation_consequential_mismatch"] = bool(
            decode_payload.get("orientation_policy_consequential_mismatch", False)
        )
        payload["decode_orientation_untrusted"] = bool(decode_payload.get("orientation_policy_untrusted", False))
        payload["decode_orientation_mismatch_reason"] = decode_payload.get("orientation_policy_mismatch_reason")
        payload["decoded_frame_shape_hwc"] = list(decode_payload.get("decoded_frame_shape_hwc", []))
        payload["decoded_frame_width_height"] = list(decode_payload.get("decoded_frame_width_height", []))
    if reference_feature_count is not None:
        payload["reference_feature_count"] = int(reference_feature_count)
    return payload


def _write_diagnostics(
    video: Path,
    calibration: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    diagnostics_dir: Path,
    selected_frames: Sequence[int],
) -> None:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    frames_by_idx = {int(frame["frame_idx"]): frame for frame in payload["frames"]}
    selected = _dedupe_ints([idx for idx in selected_frames if idx in frames_by_idx])
    if not selected:
        return
    cap, _orientation_policy = _open_camera_motion_capture(video)
    if not cap.isOpened():
        return
    try:
        frame_shape = None
        polygon = None
        for frame_idx in selected:
            frame_bgr = _read_frame_at(cap, frame_idx)
            if frame_bgr is None:
                continue
            if frame_shape is None:
                frame_shape = frame_bgr.shape[:2]
                polygon = court_reference_polygon(calibration, frame_shape)
            assert polygon is not None
            matrix = np.asarray(frames_by_idx[frame_idx]["M"], dtype=np.float64)
            overlay = draw_camera_motion_overlay(frame_bgr, polygon, matrix)
            out = diagnostics_dir / f"camera_motion_overlay_frame_{frame_idx:06d}.png"
            cv2.imwrite(str(out), overlay)
    finally:
        cap.release()


def draw_camera_motion_overlay(frame_bgr: np.ndarray, reference_polygon: np.ndarray, matrix_t_to_ref: np.ndarray) -> np.ndarray:
    image = frame_bgr.copy()
    static_poly = np.round(reference_polygon).astype(np.int32)
    cv2.polylines(image, [static_poly], isClosed=True, color=(0, 255, 255), thickness=3)
    try:
        inverse = np.linalg.inv(matrix_t_to_ref)
        compensated = _apply_homography(inverse, reference_polygon)
    except np.linalg.LinAlgError:
        compensated = reference_polygon
    cv2.polylines(image, [np.round(compensated).astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=3)
    cv2.putText(image, "static", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "motion_comp", (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return image


def _read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def _resize_for_processing(image: np.ndarray, scale: float, *, is_mask: bool) -> np.ndarray:
    scale = float(scale)
    if abs(scale - 1.0) <= 1e-12:
        return image
    height, width = image.shape[:2]
    out_width = max(8, int(round(width * scale)))
    out_height = max(8, int(round(height * scale)))
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(image, (out_width, out_height), interpolation=interpolation)


def _params_for_processing_scale(params: CameraMotionParams, scale: float) -> CameraMotionParams:
    scale = float(scale)
    if abs(scale - 1.0) <= 1e-12:
        return params
    return replace(
        params,
        min_distance=max(2.0, float(params.min_distance) * scale),
        block_size=max(3, _odd_int(round(float(params.block_size) * scale))),
        lk_win_size=max(15, _odd_int(round(float(params.lk_win_size) * scale))),
        max_ransac_reproj_px=max(0.75, float(params.max_ransac_reproj_px) * scale),
        max_homography_rms_px=max(1.0, float(params.max_homography_rms_px) * scale),
        max_similarity_rms_px=max(1.0, float(params.max_similarity_rms_px) * scale),
        max_corner_motion_px=max(10.0, float(params.max_corner_motion_px) * scale),
        processing_scale=scale,
    )


def _odd_int(value: float | int) -> int:
    ivalue = max(1, int(round(float(value))))
    return ivalue if ivalue % 2 == 1 else ivalue + 1


def _frame_count(cap: cv2.VideoCapture, *, max_frames: int | None) -> int:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        total = _count_readable_frames(cap)
    if max_frames is not None:
        total = min(total, int(max_frames))
    return total


def _probe_frame_indices(
    frame_count: int,
    *,
    reference_idx: int,
    frame_step: int,
    max_probe_frames: int,
) -> list[int]:
    frame_total = max(0, int(frame_count))
    if frame_total <= 0:
        return []
    step = max(1, int(frame_step))
    budget = max(1, int(max_probe_frames))
    indices = list(range(0, frame_total, step))
    if (frame_total - 1) not in indices:
        indices.append(frame_total - 1)
    ref = min(max(0, int(reference_idx)), frame_total - 1)
    if ref not in indices:
        indices.append(ref)
    indices = sorted(set(indices))
    if len(indices) <= budget:
        return indices
    positions = np.linspace(0, len(indices) - 1, num=budget)
    selected = {indices[int(round(pos))] for pos in positions}
    selected.add(ref)
    if len(selected) > budget and ref in selected:
        non_ref = sorted(idx for idx in selected if idx != ref)
        selected = set(non_ref[: max(0, budget - 1)])
        selected.add(ref)
    return sorted(selected)


def _count_readable_frames(cap: cv2.VideoCapture) -> int:
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    return count


def _reference_frame_idx(calibration: Mapping[str, Any], *, override: int | None) -> int:
    if override is not None:
        return int(override)
    for key in ("reference_frame_idx", "frame_idx", "frame_index"):
        value = calibration.get(key)
        if isinstance(value, int):
            return int(value)
    solved = calibration.get("solved_over_frames")
    if isinstance(solved, Sequence) and not isinstance(solved, (str, bytes)) and len(solved) > 0:
        return int(solved[0])
    return 0


def _track_frame_idx(item: Mapping[str, Any], *, fps: float) -> int | None:
    for key in ("frame_idx", "frame_index", "frame"):
        value = item.get(key)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)
    t = item.get("t")
    if isinstance(t, (int, float)):
        return int(round(float(t) * fps))
    return None


def _inlier_rms(matrix: np.ndarray, cur: np.ndarray, ref: np.ndarray, inlier_mask: np.ndarray) -> tuple[int, float]:
    count = int(np.count_nonzero(inlier_mask))
    if count <= 0:
        return 0, 0.0
    projected = _apply_homography(matrix, cur[inlier_mask])
    residual = np.linalg.norm(projected - ref[inlier_mask], axis=1)
    rms = float(np.sqrt(np.mean(np.square(residual)))) if len(residual) else 0.0
    return count, rms


def _motion_magnitude_at_point(matrix: np.ndarray, point_xy: np.ndarray) -> float:
    projected = _apply_homography(matrix, point_xy.reshape(1, 2))[0]
    return _round_float(float(np.linalg.norm(projected - point_xy)))


def _apply_homography(matrix: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    homo = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    projected = (np.asarray(matrix, dtype=np.float64) @ homo.T).T
    denom = projected[:, 2:3]
    denom[np.abs(denom) < 1e-12] = 1e-12
    return projected[:, :2] / denom


def _normalize_homography(matrix: np.ndarray) -> np.ndarray:
    out = np.asarray(matrix, dtype=np.float64)
    if abs(out[2, 2]) > 1e-12:
        out = out / out[2, 2]
    return out


def _matrix_to_json(matrix: np.ndarray) -> list[list[float]]:
    normalized = _normalize_homography(matrix)
    return [[_round_float(float(value)) for value in row] for row in normalized.tolist()]


def _validate_frame(frame: Mapping[str, Any]) -> None:
    for key in ("frame_idx", "M", "inlier_count", "rms_px", "compensated", "model"):
        if key not in frame:
            raise ValueError(f"frame missing {key}")
    if frame["model"] not in {"homography", "similarity", "identity"}:
        raise ValueError(f"invalid frame model: {frame['model']}")
    matrix = frame["M"]
    if not (
        isinstance(matrix, list)
        and len(matrix) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in matrix)
    ):
        raise ValueError("frame M must be a 3x3 matrix")
    if frame["compensated"] is False and frame["model"] != "identity":
        raise ValueError("uncompensated frames must use identity model")
    if frame["compensated"] is False and "reason" not in frame:
        raise ValueError("uncompensated frames must include reason")


def _points2(value: Any) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return np.empty((0, 2), dtype=np.float64)
    points: list[list[float]] = []
    for item in value:
        if _is_number_sequence(item, min_len=2):
            points.append([float(item[0]), float(item[1])])
    return np.asarray(points, dtype=np.float64) if points else np.empty((0, 2), dtype=np.float64)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32)).reshape(-1, 2)
    return hull.astype(np.float64)


def _polygon_centroid(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.array([0.0, 0.0], dtype=np.float64)
    return np.mean(points, axis=0).astype(np.float64)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    return _round_float(float(np.percentile(np.asarray(values, dtype=np.float64), percentile)))


def _max_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return _round_float(float(np.max(np.asarray(values, dtype=np.float64))))


def _max_drift_frame_idx(frames: Sequence[Mapping[str, Any]], drift_values: Sequence[float]) -> int:
    if not frames or not drift_values:
        return 0
    idx = int(np.argmax(np.asarray(drift_values, dtype=np.float64)))
    return int(frames[idx]["frame_idx"])


def _round_float(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    rounded = round(float(value), 6)
    return 0.0 if rounded == -0.0 else rounded


def _is_number_sequence(value: Any, *, min_len: int) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < min_len:
        return False
    return all(isinstance(item, (int, float)) for item in value[:min_len])


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dedupe_ints(values: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        ivalue = int(value)
        if ivalue not in seen:
            seen.add(ivalue)
            out.append(ivalue)
    return out
