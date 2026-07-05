from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


IDENTITY_3X3: list[list[float]] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


@dataclass(frozen=True)
class CameraMotionParams:
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
    video = Path(video_path)
    calibration_file = Path(calibration_path)
    calibration = _load_json(calibration_file)
    reference_idx = _reference_frame_idx(calibration, override=reference_frame_idx)
    tracks_by_frame = load_tracks_by_frame(tracks_path) if tracks_path is not None else {}

    cap = cv2.VideoCapture(str(video))
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

        frame_shape = reference_bgr.shape[:2]
        court_mask = build_court_mask(calibration, frame_shape, apron_px=params.apron_px)
        reference_mask = mask_people_for_frame(
            court_mask,
            tracks_by_frame,
            frame_idx=reference_idx,
            padding_px=params.person_padding_px,
        )
        reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
        reference_points = detect_reference_features(reference_bgr, reference_mask, params)
        court_polygon = court_reference_polygon(calibration, frame_shape)
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
                }
                frames.append(entry)
                drift_values.append(0.0)
                residual_values.append(0.0)
                continue

            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            current_mask = mask_people_for_frame(
                court_mask,
                tracks_by_frame,
                frame_idx=frame_idx,
                padding_px=params.person_padding_px,
            )
            estimate = _estimate_frame_transform(
                reference_gray,
                frame_gray,
                reference_points,
                current_mask,
                court_polygon,
                params,
            )
            frames.append(_frame_entry(frame_idx, estimate))
            drift_values.append(_motion_magnitude_at_point(estimate.matrix_for_metrics, centroid))
            if estimate.compensated:
                residual_values.append(estimate.rms_px)

        payload = _payload(
            video=video,
            reference_idx=reference_idx,
            params=params,
            frames=frames,
            drift_values=drift_values,
            residual_values=residual_values,
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


def _estimate_frame_transform(
    reference_gray: np.ndarray,
    frame_gray: np.ndarray,
    reference_points: np.ndarray,
    current_mask: np.ndarray,
    court_polygon: np.ndarray,
    params: CameraMotionParams,
) -> _FrameEstimate:
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

    if len(cur) < params.min_similarity_inliers:
        return _failed_estimate("too_few_tracked_features", inlier_count=int(len(cur)))

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
        return _best_candidate(valid_candidates)
    if invalid_sanity:
        inlier_count, reason = max(invalid_sanity, key=lambda item: item[0])
        return _failed_estimate(reason, inlier_count=inlier_count)

    best = homography_result or similarity_result
    if best is None:
        return _failed_estimate("transform_fit_failed", inlier_count=int(len(cur)))
    if best.rms_px > max(params.max_homography_rms_px, params.max_similarity_rms_px):
        return _failed_estimate("rms_above_threshold", inlier_count=best.inlier_count, metric_matrix=best.matrix)
    return _failed_estimate("too_few_transform_inliers", inlier_count=best.inlier_count, metric_matrix=best.matrix)


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


def _failed_estimate(
    reason: str,
    *,
    inlier_count: int = 0,
    metric_matrix: np.ndarray | None = None,
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
    )


def _mask_indices_in_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    keep = []
    for x, y in points:
        ix = int(round(float(x)))
        iy = int(round(float(y)))
        keep.append(0 <= ix < width and 0 <= iy < height and mask[iy, ix] > 0)
    return np.asarray(keep, dtype=bool)


def _frame_entry(frame_idx: int, estimate: _FrameEstimate) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "frame_idx": int(frame_idx),
        "M": _matrix_to_json(estimate.matrix),
        "inlier_count": int(estimate.inlier_count),
        "rms_px": _round_float(estimate.rms_px),
        "compensated": bool(estimate.compensated),
        "model": estimate.model,
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
    }


def _payload(
    *,
    video: Path,
    reference_idx: int,
    params: CameraMotionParams,
    frames: list[dict[str, Any]],
    drift_values: Sequence[float],
    residual_values: Sequence[float],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_camera_motion",
        "video": video.as_posix(),
        "reference_frame_idx": int(reference_idx),
        "method": "shi_tomasi_reference_lk_ransac_homography_similarity_fallback",
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
        },
        "verified": False,
        "not_gate_verified": True,
    }


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
    cap = cv2.VideoCapture(str(video))
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


def _frame_count(cap: cv2.VideoCapture, *, max_frames: int | None) -> int:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        total = _count_readable_frames(cap)
    if max_frames is not None:
        total = min(total, int(max_frames))
    return total


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
