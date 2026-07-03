"""Estimate and normalize per-player metric skeleton scale from court geometry."""

from __future__ import annotations

import copy
import math
from statistics import median
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
ESTIMATE_ARTIFACT_TYPE = "racketsport_player_scale_estimates"
NORMALIZATION_ARTIFACT_TYPE = "racketsport_player_scale_normalization"
DEFAULT_MIN_CONFIDENCE = 0.55
DEFAULT_WINDOW_SPREAD_MAX_M = 0.25
DEFAULT_ANTHROPOMETRIC_BAND_M = (1.40, 2.05)
DEFAULT_RAW_HEIGHT_BAND_M = (0.75, 2.60)
ROOT_JOINT_NAMES = ("pelvis", "root", "mid_hip")
HEAD_JOINT_NAMES = ("head", "nose", "left_eye", "right_eye")
FOOT_JOINT_NAMES = (
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
)


class PlayerScaleError(ValueError):
    """Raised when player scale estimation or normalization must fail closed."""


def estimate_player_metric_heights(
    tracks_payload: Mapping[str, Any],
    calibration_payload: Mapping[str, Any],
    *,
    skeleton_payload: Mapping[str, Any] | None = None,
    min_bbox_confidence: float = 0.20,
    min_valid_samples: int = 20,
    samples_per_window: int = 30,
    estimator_percentile: float = 80.0,
    window_spread_max_m: float = DEFAULT_WINDOW_SPREAD_MAX_M,
    raw_height_band_m: tuple[float, float] = DEFAULT_RAW_HEIGHT_BAND_M,
) -> dict[str, Any]:
    """Estimate player stature from accepted track bboxes and court-plane depth.

    Each track frame supplies a pixel bbox and a metric court position. The camera
    calibration projects a vertical metric segment at that court position into image
    pixels; a binary solve turns bbox pixel height into a per-frame metric-height
    sample. Per-player p80 and window p80 spread keep crouches and brief outliers from
    driving the estimate while still failing closed on unstable bbox quality.
    """

    players: dict[str, dict[str, Any]] = {}
    unstable_players: list[str] = []
    skeleton_statures = _skeleton_statures_by_player(skeleton_payload) if skeleton_payload is not None else {}

    for player in _track_players(tracks_payload):
        player_id = str(player.get("id", player.get("player_id", "unknown")))
        frames = player.get("frames")
        if not isinstance(frames, list):
            players[player_id] = _empty_player_report(player_id, "no_track_frames")
            continue

        raw_samples: list[dict[str, Any]] = []
        rejected = 0
        for ordinal, frame in enumerate(frames):
            if not isinstance(frame, Mapping):
                rejected += 1
                continue
            sample = _height_sample_from_track_frame(
                frame,
                ordinal=ordinal,
                calibration_payload=calibration_payload,
                min_bbox_confidence=min_bbox_confidence,
                raw_height_band_m=raw_height_band_m,
            )
            if sample is None:
                rejected += 1
                continue
            raw_samples.append(sample)

        samples = _mad_filtered_samples(raw_samples)
        if not samples:
            players[player_id] = {
                **_empty_player_report(player_id, "no_valid_samples"),
                "total_frame_count": len(frames),
                "rejected_sample_count": rejected,
            }
            continue

        heights = [float(sample["height_m"]) for sample in samples]
        window_heights = _window_percentiles(samples, samples_per_window=max(1, samples_per_window), percentile=estimator_percentile)
        window_spread = max(window_heights) - min(window_heights) if len(window_heights) >= 2 else None
        iqr = _percentile(heights, 75.0) - _percentile(heights, 25.0)
        bbox_conf_median = _percentile([float(sample["bbox_confidence"]) for sample in samples], 50.0)
        confidence = _confidence_score(
            valid_sample_count=len(samples),
            min_valid_samples=min_valid_samples,
            window_spread_m=window_spread,
            window_spread_max_m=window_spread_max_m,
            iqr_m=iqr,
            bbox_confidence=bbox_conf_median,
            calibration_payload=calibration_payload,
        )
        status = "ok"
        if len(samples) < min_valid_samples:
            status = "low_confidence"
            confidence = min(confidence, 0.34)
        if window_spread is not None and window_spread > window_spread_max_m:
            status = "unstable"
            confidence = min(confidence, 0.49)
            unstable_players.append(player_id)
        elif confidence < DEFAULT_MIN_CONFIDENCE:
            status = "low_confidence"

        players[player_id] = {
            "schema_version": SCHEMA_VERSION,
            "player_id": player_id,
            "status": status,
            "height_m": _round(_percentile(heights, estimator_percentile)),
            "estimator": f"p{int(estimator_percentile)}_bbox_depth_projection",
            "confidence": _round(confidence),
            "confidence_label": _confidence_label(confidence),
            "valid_sample_count": len(samples),
            "total_frame_count": len(frames),
            "rejected_sample_count": rejected + max(0, len(raw_samples) - len(samples)),
            "raw_sample_count": len(raw_samples),
            "median_frame_height_m": _round(_percentile(heights, 50.0)),
            "p20_frame_height_m": _round(_percentile(heights, 20.0)),
            "p80_frame_height_m": _round(_percentile(heights, 80.0)),
            "iqr_m": _round(iqr),
            "window_height_percentile": estimator_percentile,
            "window_heights_m": [_round(value) for value in window_heights],
            "window_height_spread_m": _round(window_spread) if window_spread is not None else None,
            "window_spread_max_m": float(window_spread_max_m),
            "median_bbox_height_px": _round(_percentile([float(sample["bbox_height_px"]) for sample in samples], 50.0)),
            "median_bbox_confidence": _round(bbox_conf_median),
            "skeleton_pre_scale_stature_m": _round(skeleton_statures.get(player_id)) if player_id in skeleton_statures else None,
        }

    unstable = bool(unstable_players)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ESTIMATE_ARTIFACT_TYPE,
        "status": "unstable" if unstable else "ok",
        "players": dict(sorted(players.items(), key=lambda item: _sort_key(item[0]))),
        "unstable": unstable,
        "unstable_players": unstable_players,
        "parameters": {
            "min_bbox_confidence": float(min_bbox_confidence),
            "min_valid_samples": int(min_valid_samples),
            "samples_per_window": int(samples_per_window),
            "estimator_percentile": float(estimator_percentile),
            "window_spread_max_m": float(window_spread_max_m),
            "raw_height_band_m": [float(raw_height_band_m[0]), float(raw_height_band_m[1])],
        },
        "calibration": _calibration_summary(calibration_payload),
        "reads_cvat_labels": False,
        "protected_eval_labels_used": False,
    }


def normalize_skeleton_scale_payload(
    skeleton_payload: Mapping[str, Any],
    estimate_report: Mapping[str, Any],
    *,
    estimate_path: str,
    pre_scale_backup_path: str,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    anthropometric_band_m: tuple[float, float] = DEFAULT_ANTHROPOMETRIC_BAND_M,
    require_stable: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Scale each skeleton player's joints about its per-frame root."""

    if require_stable and bool(estimate_report.get("unstable")):
        unstable_players = ", ".join(str(player) for player in estimate_report.get("unstable_players", []))
        raise PlayerScaleError(f"unstable height estimates exceed window spread gate for players: {unstable_players}")

    names = _joint_names(skeleton_payload)
    estimates = estimate_report.get("players")
    if not isinstance(estimates, Mapping):
        raise PlayerScaleError("estimate report must contain a players mapping")

    refusals: list[str] = []
    player_reports: dict[str, dict[str, Any]] = {}
    for player in skeleton_payload.get("players", []):
        if not isinstance(player, Mapping):
            continue
        player_id = str(player.get("id", player.get("player_id", "unknown")))
        estimate = estimates.get(player_id)
        if not isinstance(estimate, Mapping):
            refusals.append(f"player {player_id}: missing estimate")
            continue
        confidence = float(estimate.get("confidence") or 0.0)
        status = str(estimate.get("status") or "unknown")
        if status != "ok" or confidence < min_confidence:
            refusals.append(f"player {player_id}: low-confidence estimate status={status} confidence={confidence:.3f}")
            continue
        pre_stature = _player_stature_m(player, names)
        if pre_stature is None or pre_stature <= 1e-6:
            refusals.append(f"player {player_id}: invalid pre-scale stature")
            continue
        estimated_height = float(estimate.get("height_m") or 0.0)
        if estimated_height <= 0.0 or not math.isfinite(estimated_height):
            refusals.append(f"player {player_id}: invalid estimated height")
            continue
        target_height = _clamp(estimated_height, anthropometric_band_m[0], anthropometric_band_m[1])
        player_reports[player_id] = {
            "player_id": player_id,
            "estimated_height_m": _round(estimated_height),
            "target_height_m": _round(target_height),
            "clamped": not math.isclose(target_height, estimated_height),
            "pre_scale_stature_m": _round(pre_stature),
            "scale_factor": _round(target_height / pre_stature),
            "confidence": _round(confidence),
            "estimate_status": status,
            "valid_sample_count": int(estimate.get("valid_sample_count") or 0),
        }

    if refusals:
        raise PlayerScaleError("low-confidence or invalid player scale estimates refused: " + "; ".join(refusals))

    normalized = copy.deepcopy(dict(skeleton_payload))
    normalized_players = normalized.get("players")
    if not isinstance(normalized_players, list):
        raise PlayerScaleError("skeleton payload must contain a players list")

    for player in normalized_players:
        if not isinstance(player, dict):
            continue
        player_id = str(player.get("id", player.get("player_id", "unknown")))
        report = player_reports.get(player_id)
        if report is None:
            continue
        scale_factor = float(report["scale_factor"])
        frames = player.get("frames")
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            joints = _parse_joints_keep_tail(frame.get("joints_world"))
            if not joints:
                continue
            root = _root_xyz(joints, names)
            scaled: list[list[float]] = []
            for point in joints:
                scaled_xyz = [root[axis] + (point[axis] - root[axis]) * scale_factor for axis in range(3)]
                scaled.append([*scaled_xyz, *point[3:]])
            frame["joints_world"] = scaled

    report = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": NORMALIZATION_ARTIFACT_TYPE,
        "status": "normalized",
        "estimate_path": estimate_path,
        "pre_scale_backup": pre_scale_backup_path,
        "anthropometric_band_m": [float(anthropometric_band_m[0]), float(anthropometric_band_m[1])],
        "min_confidence": float(min_confidence),
        "require_stable": bool(require_stable),
        "players": dict(sorted(player_reports.items(), key=lambda item: _sort_key(item[0]))),
        "reads_cvat_labels": False,
        "protected_eval_labels_used": False,
    }
    provenance = dict(normalized.get("provenance") or {})
    provenance["player_scale_normalization"] = report
    normalized["provenance"] = provenance
    return normalized, report


def projected_metric_height_px(
    calibration_payload: Mapping[str, Any],
    world_xy: Sequence[float],
    height_m: float,
) -> float:
    """Return the image-y pixel extent of a vertical metric segment."""

    x, y = float(world_xy[0]), float(world_xy[1])
    foot = _project_world_point(calibration_payload, [x, y, 0.0])
    head = _project_world_point(calibration_payload, [x, y, float(height_m)])
    return abs(float(head[1]) - float(foot[1]))


def estimate_height_for_bbox(
    calibration_payload: Mapping[str, Any],
    *,
    world_xy: Sequence[float],
    bbox_height_px: float,
    max_height_m: float = 3.20,
) -> float:
    """Solve metric height whose projected image-y span matches bbox height."""

    target = float(bbox_height_px)
    if target <= 0.0 or not math.isfinite(target):
        raise PlayerScaleError("bbox height must be positive")
    lo, hi = 0.05, float(max_height_m)
    hi_px = projected_metric_height_px(calibration_payload, world_xy, hi)
    if hi_px <= 0.0 or not math.isfinite(hi_px):
        raise PlayerScaleError("calibration projects vertical height with invalid pixel scale")
    if target > hi_px:
        return target / hi_px * hi
    for _ in range(42):
        mid = (lo + hi) / 2.0
        mid_px = projected_metric_height_px(calibration_payload, world_xy, mid)
        if mid_px < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _height_sample_from_track_frame(
    frame: Mapping[str, Any],
    *,
    ordinal: int,
    calibration_payload: Mapping[str, Any],
    min_bbox_confidence: float,
    raw_height_band_m: tuple[float, float],
) -> dict[str, Any] | None:
    bbox = frame.get("bbox")
    world_xy = frame.get("track_world_xy", frame.get("world_xy", frame.get("court_xy")))
    if not _is_bbox(bbox) or not _is_xy(world_xy):
        return None
    confidence = float(frame.get("conf", frame.get("confidence", frame.get("score", 1.0))) or 0.0)
    if confidence < min_bbox_confidence:
        return None
    bbox_height = abs(float(bbox[3]) - float(bbox[1]))
    if bbox_height <= 0.0:
        return None
    try:
        height_m = estimate_height_for_bbox(calibration_payload, world_xy=world_xy, bbox_height_px=bbox_height)
    except (PlayerScaleError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if not (raw_height_band_m[0] <= height_m <= raw_height_band_m[1]):
        return None
    return {
        "frame_idx": int(frame.get("frame_idx", frame.get("frame_index", frame.get("frame", ordinal)))),
        "t": float(frame.get("t", frame.get("time", ordinal))),
        "height_m": float(height_m),
        "bbox_height_px": bbox_height,
        "bbox_confidence": confidence,
    }


def _project_world_point(calibration_payload: Mapping[str, Any], point: Sequence[float]) -> list[float]:
    intrinsics = calibration_payload["intrinsics"]
    extrinsics = calibration_payload["extrinsics"]
    rotation = _mat3(extrinsics["R"])
    translation = [float(value) for value in extrinsics["t"]]
    world = [float(point[0]), float(point[1]), float(point[2])]
    camera = [
        sum(rotation[row][col] * world[col] for col in range(3)) + translation[row]
        for row in range(3)
    ]
    if math.isclose(camera[2], 0.0):
        raise PlayerScaleError("world point projects with zero camera depth")
    return [
        float(intrinsics["fx"]) * camera[0] / camera[2] + float(intrinsics["cx"]),
        float(intrinsics["fy"]) * camera[1] / camera[2] + float(intrinsics["cy"]),
    ]


def _track_players(tracks_payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    players = tracks_payload.get("players")
    if isinstance(players, list):
        return [player for player in players if isinstance(player, Mapping)]
    return []


def _empty_player_report(player_id: str, status: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "player_id": str(player_id),
        "status": status,
        "height_m": None,
        "confidence": 0.0,
        "confidence_label": "none",
        "valid_sample_count": 0,
        "total_frame_count": 0,
        "rejected_sample_count": 0,
        "window_heights_m": [],
        "window_height_spread_m": None,
    }


def _mad_filtered_samples(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(samples) < 8:
        return list(samples)
    heights = [float(sample["height_m"]) for sample in samples]
    med = median(heights)
    deviations = [abs(value - med) for value in heights]
    mad = median(deviations)
    threshold = max(0.35, 4.0 * 1.4826 * mad)
    return [sample for sample in samples if abs(float(sample["height_m"]) - med) <= threshold]


def _window_percentiles(samples: Sequence[dict[str, Any]], *, samples_per_window: int, percentile: float) -> list[float]:
    ordered = sorted(samples, key=lambda sample: int(sample["frame_idx"]))
    windows: list[float] = []
    for start in range(0, len(ordered), samples_per_window):
        window = ordered[start : start + samples_per_window]
        if len(window) < max(3, min(samples_per_window, 6)):
            continue
        windows.append(_percentile([float(sample["height_m"]) for sample in window], percentile))
    return windows


def _confidence_score(
    *,
    valid_sample_count: int,
    min_valid_samples: int,
    window_spread_m: float | None,
    window_spread_max_m: float,
    iqr_m: float,
    bbox_confidence: float,
    calibration_payload: Mapping[str, Any],
) -> float:
    sample_score = min(float(valid_sample_count) / max(float(min_valid_samples) * 3.0, 1.0), 1.0)
    if window_spread_m is None:
        stability_score = 0.0
    else:
        stability_score = max(0.0, 1.0 - float(window_spread_m) / max(float(window_spread_max_m), 1e-6))
    iqr_score = max(0.0, 1.0 - float(iqr_m) / 0.35)
    det_score = _clamp(float(bbox_confidence), 0.0, 1.0)
    cal_score = _calibration_confidence_score(calibration_payload)
    return _clamp(
        0.30 * sample_score + 0.35 * stability_score + 0.20 * iqr_score + 0.10 * det_score + 0.05 * cal_score,
        0.0,
        1.0,
    )


def _calibration_confidence_score(calibration_payload: Mapping[str, Any]) -> float:
    value = str(calibration_payload.get("metric_confidence", calibration_payload.get("source", ""))).lower()
    if "high" in value or "good" in value:
        return 1.0
    if "medium" in value or "warn" in value:
        return 0.8
    if "low" in value:
        return 0.6
    return 0.7


def _calibration_summary(calibration_payload: Mapping[str, Any]) -> dict[str, Any]:
    intrinsics = calibration_payload.get("intrinsics") if isinstance(calibration_payload.get("intrinsics"), Mapping) else {}
    image_size = calibration_payload.get("image_size")
    return {
        "source": calibration_payload.get("source"),
        "metric_confidence": calibration_payload.get("metric_confidence"),
        "image_size": list(image_size) if isinstance(image_size, list) else None,
        "intrinsics_source": intrinsics.get("source") if isinstance(intrinsics, Mapping) else None,
        "uses_extrinsic_projection": True,
        "distortion_ignored_for_height_scale": bool(intrinsics.get("dist") if isinstance(intrinsics, Mapping) else False),
    }


def _skeleton_statures_by_player(skeleton_payload: Mapping[str, Any] | None) -> dict[str, float]:
    if not isinstance(skeleton_payload, Mapping):
        return {}
    names = _joint_names(skeleton_payload)
    out: dict[str, float] = {}
    for player in skeleton_payload.get("players", []):
        if not isinstance(player, Mapping):
            continue
        player_id = str(player.get("id", player.get("player_id", "unknown")))
        stature = _player_stature_m(player, names)
        if stature is not None:
            out[player_id] = stature
    return out


def _player_stature_m(player: Mapping[str, Any], names: Sequence[str]) -> float | None:
    frames = player.get("frames")
    if not isinstance(frames, list):
        return None
    spans: list[float] = []
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        joints = _parse_joints_keep_tail(frame.get("joints_world"))
        if not joints:
            continue
        head_indices = _indices(names, HEAD_JOINT_NAMES)
        foot_indices = _indices(names, FOOT_JOINT_NAMES)
        if head_indices and foot_indices:
            head_values = [float(joints[index][2]) for index in head_indices if index < len(joints)]
            foot_values = [float(joints[index][2]) for index in foot_indices if index < len(joints)]
            if head_values and foot_values:
                spans.append(max(head_values) - min(foot_values))
        else:
            z_values = [float(point[2]) for point in joints]
            spans.append(max(z_values) - min(z_values))
    if not spans:
        return None
    return _percentile(spans, 80.0)


def _joint_names(payload: Mapping[str, Any]) -> tuple[str, ...]:
    names = payload.get("joint_names")
    if not isinstance(names, list) or not names:
        raise PlayerScaleError("skeleton payload must include joint_names")
    return tuple(str(name) for name in names)


def _parse_joints_keep_tail(values: Any) -> list[list[float]]:
    if not isinstance(values, list):
        return []
    out: list[list[float]] = []
    for point in values:
        if not isinstance(point, list) or len(point) < 3:
            continue
        parsed = [float(value) for value in point[:3]]
        parsed.extend(point[3:])
        out.append(parsed)
    return out


def _root_xyz(joints: Sequence[Sequence[float]], names: Sequence[str]) -> list[float]:
    root_indices = _indices(names, ROOT_JOINT_NAMES)
    if not root_indices:
        root_indices = _indices(names, ("left_hip", "right_hip"))
    if not root_indices:
        root_indices = [0]
    points = [joints[index] for index in root_indices if index < len(joints)]
    return [_mean([float(point[axis]) for point in points]) for axis in range(3)]


def _indices(names: Sequence[str], wanted: Sequence[str]) -> list[int]:
    by_name = {name: index for index, name in enumerate(names)}
    return [by_name[name] for name in wanted if name in by_name]


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (float(percentile) / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _mat3(values: Sequence[Sequence[float]]) -> list[list[float]]:
    if len(values) != 3:
        raise PlayerScaleError("rotation must be 3x3")
    matrix: list[list[float]] = []
    for row in values:
        if len(row) != 3:
            raise PlayerScaleError("rotation must be 3x3")
        matrix.append([float(value) for value in row])
    return matrix


def _is_bbox(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 4 and all(isinstance(item, (int, float)) for item in value[:4])


def _is_xy(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2])


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.80:
        return "high"
    if confidence >= DEFAULT_MIN_CONFIDENCE:
        return "medium"
    if confidence > 0.0:
        return "low"
    return "none"


def _sort_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):08d}") if str(value).isdigit() else (1, str(value))
