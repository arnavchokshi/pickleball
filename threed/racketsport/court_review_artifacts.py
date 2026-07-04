"""Reviewed court-calibration artifacts for user-adjusted video submissions."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .court_calibration import (
    homography_from_planar_points,
    project_planar_points,
    reprojection_error,
    solve_camera_pose,
)
from .court_keypoint_net import PICKLEBALL_KEYPOINT_BY_NAME
from .schemas import CameraIntrinsics, CaptureQuality, CourtCalibration, CourtExtrinsics, PICKLEBALL_COURT_KEYPOINT_NAMES

COURT_REVIEW_SCHEMA_VERSION = 1
COURT_REVIEW_ARTIFACT_TYPE = "racketsport_reviewed_court_calibration"
COURT_REVIEW_INDEX_ARTIFACT_TYPE = "racketsport_reviewed_court_calibration_index"
COURT_REVIEW_STATUS_HUMAN = "human_reviewed"
COURT_REVIEW_STATUS_AUTO = "auto_predicted_unreviewed"
COURT_REVIEW_STATUSES = frozenset({COURT_REVIEW_STATUS_HUMAN, COURT_REVIEW_STATUS_AUTO})
LOW_CONFIDENCE_THRESHOLD = 0.25
MANUAL_MOVE_EPSILON_PX = 0.75
NET_TOP_KEYPOINT_NAMES = frozenset({"net_left_sideline", "net_center", "net_right_sideline"})
CORNER_ORDER = ("near_left_corner", "near_right_corner", "far_right_corner", "far_left_corner")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reviewed_court_points(
    *,
    adjusted_points: Mapping[str, Any],
    predicted_points: Mapping[str, Any],
    image_size: Sequence[int | float],
) -> dict[str, Any]:
    width, height = _image_size_tuple(image_size)
    adjusted = _point_xy_map(adjusted_points, allow_missing=True)
    predicted = _prediction_point_map(predicted_points, allow_missing=True)
    warnings: list[dict[str, Any]] = []

    for name in PICKLEBALL_COURT_KEYPOINT_NAMES:
        if name not in adjusted:
            warnings.append(_warning("missing_point", name, f"{name} is missing from the reviewed court layout."))
            continue
        x, y = adjusted[name]
        if x < 0.0 or x > width or y < 0.0 or y > height:
            warnings.append(_warning("out_of_frame", name, f"{name} is outside the source frame."))
        confidence = predicted.get(name, {}).get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and float(confidence) < LOW_CONFIDENCE_THRESHOLD:
            warnings.append(_warning("low_prediction_confidence", name, f"{name} started from a low-confidence auto prediction."))

    geometry_warning = _geometry_warning(adjusted, width=width, height=height)
    if geometry_warning is not None:
        warnings.append(geometry_warning)

    return {
        "status": "pass" if not warnings else "warn",
        "warnings": warnings,
        "point_count": len(adjusted),
        "required_point_count": len(PICKLEBALL_COURT_KEYPOINT_NAMES),
    }


def build_reviewed_court_artifacts(
    *,
    video_id: str,
    video_path: str,
    video_sha256: str,
    image_size: Sequence[int | float],
    frame_index: int,
    frame_time_s: float,
    auto_prediction_source: str,
    predicted_points: Mapping[str, Any],
    adjusted_points: Mapping[str, Any],
    created_at: str | None = None,
    review_status: str = COURT_REVIEW_STATUS_HUMAN,
) -> tuple[dict[str, Any], dict[str, Any]]:
    width, height = _image_size_tuple(image_size)
    created = created_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if review_status not in COURT_REVIEW_STATUSES:
        raise ValueError(f"review_status must be one of: {', '.join(sorted(COURT_REVIEW_STATUSES))}")
    is_human_reviewed = review_status == COURT_REVIEW_STATUS_HUMAN
    predicted = _prediction_point_map(predicted_points, allow_missing=False)
    adjusted = _point_xy_map(adjusted_points, allow_missing=False)
    validation = validate_reviewed_court_points(adjusted_points=adjusted, predicted_points=predicted, image_size=(width, height))
    protected_eval = _looks_like_protected_eval(video_path)

    points: dict[str, dict[str, Any]] = {}
    for name in PICKLEBALL_COURT_KEYPOINT_NAMES:
        predicted_xy = predicted[name]["xy"]
        adjusted_xy = adjusted[name]
        points[name] = {
            "predicted_xy": [float(predicted_xy[0]), float(predicted_xy[1])],
            "adjusted_xy": [float(adjusted_xy[0]), float(adjusted_xy[1])],
            "confidence": float(predicted[name]["confidence"]),
            "manual_moved": _distance(predicted_xy, adjusted_xy) > MANUAL_MOVE_EPSILON_PX,
        }

    artifact = {
        "schema_version": COURT_REVIEW_SCHEMA_VERSION,
        "artifact_type": COURT_REVIEW_ARTIFACT_TYPE,
        "review_status": review_status,
        "source_video": {
            "id": str(video_id),
            "path": str(video_path),
            "sha256": str(video_sha256),
        },
        "frame": {
            "index": int(frame_index),
            "time_s": float(frame_time_s),
            "image_size": [int(width), int(height)],
        },
        "auto_prediction": {
            "source": str(auto_prediction_source),
            "verified": False,
            "not_cal3_verified": True,
            "points": {
                name: {"xy": list(predicted[name]["xy"]), "confidence": float(predicted[name]["confidence"])}
                for name in PICKLEBALL_COURT_KEYPOINT_NAMES
            },
        },
        "points": points,
        "validation": validation,
        "pipeline": {
            "derived_artifact": "court_calibration.json",
            "handoff": "--court-calibration",
            "trust": "reviewed_manual_court_layout" if is_human_reviewed else "auto_predicted_unreviewed_court_layout",
        },
        "training": {
            "usable_for_court_detector_training": is_human_reviewed and not protected_eval,
            "training_policy": "human_reviewed_not_eval_promoted" if is_human_reviewed else "auto_prediction_not_training_ready",
            "protected_eval_clip": protected_eval,
        },
        "created_at": created,
    }
    calibration = court_calibration_from_review_artifact(artifact)
    return artifact, calibration


def court_calibration_from_review_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    points = artifact.get("points")
    frame = artifact.get("frame")
    if not isinstance(points, Mapping):
        raise ValueError("review artifact points must be an object")
    if not isinstance(frame, Mapping):
        raise ValueError("review artifact frame must be an object")
    width, height = _image_size_tuple(frame.get("image_size") or ())
    image_pts = [_review_point_xy(points[name], name=name) for name in PICKLEBALL_COURT_KEYPOINT_NAMES]
    world_pts = [list(PICKLEBALL_KEYPOINT_BY_NAME[name].world_xyz_m) for name in PICKLEBALL_COURT_KEYPOINT_NAMES]
    homography = homography_from_planar_points(world_pts, image_pts)
    projected = project_planar_points(homography, world_pts)
    error = reprojection_error(image_pts, projected)
    intrinsics = CameraIntrinsics(
        fx=float(max(width, height)) * 1.2,
        fy=float(max(width, height)) * 1.2,
        cx=float(width) / 2.0,
        cy=float(height) / 2.0,
        dist=[],
        source="estimated_from_reviewed_court_calibration",
    )
    try:
        extrinsics = solve_camera_pose(world_pts, image_pts, intrinsics)
    except Exception:
        extrinsics = CourtExtrinsics(
            R=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t=[0.0, 0.0, 10.0],
            camera_height_m=10.0,
        )
    review_status = str(artifact.get("review_status") or COURT_REVIEW_STATUS_HUMAN)
    if review_status == COURT_REVIEW_STATUS_HUMAN:
        quality_reasons = [
            "human_reviewed_court_correction",
            "estimated_intrinsics",
            "auto_prediction_unverified_manual_review_authority",
        ]
    else:
        quality_reasons = [
            "auto_predicted_court_layout_unreviewed",
            "estimated_intrinsics",
            "auto_prediction_unverified",
        ]
    calibration = CourtCalibration(
        schema_version=1,
        sport="pickleball",
        homography=homography,
        intrinsics=intrinsics,
        image_size=(int(width), int(height)),
        extrinsics=extrinsics,
        reprojection_error_px=error,
        capture_quality=CaptureQuality(grade="warn", reasons=quality_reasons),
        image_pts=image_pts,
        world_pts=world_pts,
    )
    return calibration.model_dump(mode="json")


def save_reviewed_court_artifacts(
    *,
    artifact: Mapping[str, Any],
    court_calibration: Mapping[str, Any],
    root: str | Path,
) -> dict[str, str]:
    root_path = Path(root)
    source_video = artifact.get("source_video")
    frame = artifact.get("frame")
    if not isinstance(source_video, Mapping):
        raise ValueError("review artifact source_video must be an object")
    if not isinstance(frame, Mapping):
        raise ValueError("review artifact frame must be an object")
    video_id = str(source_video.get("id") or "unknown_video")
    video_sha256 = str(source_video.get("sha256") or "")
    created_at = str(artifact.get("created_at") or datetime.now(UTC).isoformat())
    slug = _safe_slug(video_id)
    entry_id = _safe_slug(f"{created_at.replace(':', '').replace('-', '').replace('+', 'Z')}_{video_sha256[:12] or 'nohash'}")
    entry_dir = root_path / slug / entry_id
    entry_dir.mkdir(parents=True, exist_ok=True)

    review_path = entry_dir / "reviewed_court_calibration.json"
    calibration_path = entry_dir / "court_calibration.json"
    _write_json(review_path, dict(artifact))
    _write_json(calibration_path, dict(court_calibration))

    index_path = root_path / "reviewed_court_calibrations_index.json"
    index = _read_json(index_path) if index_path.is_file() else _empty_index()
    entries = [entry for entry in index.get("entries", []) if isinstance(entry, dict)]
    entries.append(
        {
            "video_id": video_id,
            "video_sha256": video_sha256,
            "created_at": created_at,
            "frame_index": int(frame.get("index", 0)),
            "review_path": str(review_path),
            "court_calibration_path": str(calibration_path),
            "review_status": str(artifact.get("review_status") or COURT_REVIEW_STATUS_HUMAN),
            "training_policy": str(
                (artifact.get("training") if isinstance(artifact.get("training"), Mapping) else {}).get(
                    "training_policy",
                    "human_reviewed_not_eval_promoted",
                )
            ),
            "usable_for_court_detector_training": bool(
                (artifact.get("training") if isinstance(artifact.get("training"), Mapping) else {}).get(
                    "usable_for_court_detector_training",
                    False,
                )
            ),
        }
    )
    index["entries"] = entries
    _write_json(index_path, index)
    return {
        "review_path": str(review_path),
        "court_calibration_path": str(calibration_path),
        "index_path": str(index_path),
    }


def _review_point_xy(value: Any, *, name: str) -> list[float]:
    if isinstance(value, Mapping):
        if "adjusted_xy" in value:
            return _xy(value["adjusted_xy"], name=name)
        if "xy" in value:
            return _xy(value["xy"], name=name)
    return _xy(value, name=name)


def _prediction_point_map(points: Mapping[str, Any], *, allow_missing: bool) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for name in PICKLEBALL_COURT_KEYPOINT_NAMES:
        if name not in points:
            if allow_missing:
                continue
            raise ValueError(f"missing predicted point: {name}")
        value = points[name]
        if isinstance(value, Mapping):
            xy_value = value.get("xy", value.get("predicted_xy"))
            confidence = value.get("confidence", 0.0)
        else:
            xy_value = value
            confidence = 0.0
        normalized[name] = {"xy": _xy(xy_value, name=name), "confidence": _unit_confidence(confidence, name=name)}
    return normalized


def _point_xy_map(points: Mapping[str, Any], *, allow_missing: bool) -> dict[str, list[float]]:
    normalized: dict[str, list[float]] = {}
    for name in PICKLEBALL_COURT_KEYPOINT_NAMES:
        if name not in points:
            if allow_missing:
                continue
            raise ValueError(f"missing adjusted point: {name}")
        normalized[name] = _review_point_xy(points[name], name=name)
    return normalized


def _xy(value: Any, *, name: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must be a 2D point")
    x, y = value
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not isinstance(y, (int, float)) or isinstance(y, bool):
        raise ValueError(f"{name} must contain finite numbers")
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
        raise ValueError(f"{name} must contain finite numbers")
    return [float(x), float(y)]


def _unit_confidence(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} confidence must be finite")
    return max(0.0, min(1.0, float(value)))


def _image_size_tuple(image_size: Sequence[int | float]) -> tuple[int, int]:
    if len(image_size) != 2:
        raise ValueError("image_size must contain width and height")
    width, height = image_size
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        raise ValueError("image_size must contain numbers")
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    return int(width), int(height)


def _geometry_warning(points: Mapping[str, list[float]], *, width: int, height: int) -> dict[str, Any] | None:
    if any(name not in points for name in CORNER_ORDER):
        return None
    near_left, near_right, far_right, far_left = [points[name] for name in CORNER_ORDER]
    area = abs(_polygon_area([near_left, near_right, far_right, far_left]))
    min_area = float(width * height) * 0.025
    near_mid_y = (near_left[1] + near_right[1]) / 2.0
    far_mid_y = (far_left[1] + far_right[1]) / 2.0
    left_right_swapped = near_left[0] >= near_right[0] or far_left[0] >= far_right[0]
    far_not_above_near = far_mid_y >= near_mid_y
    if area < min_area or left_right_swapped or far_not_above_near:
        return {
            "code": "bad_geometry",
            "severity": "warning",
            "message": "Reviewed court corners do not form a plausible full-court quadrilateral.",
            "points": list(CORNER_ORDER),
        }
    return None


def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        total += float(point[0]) * float(next_point[1]) - float(next_point[0]) * float(point[1])
    return total / 2.0


def _warning(code: str, point: str, message: str) -> dict[str, Any]:
    return {"code": code, "severity": "warning", "point": point, "message": message}


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _looks_like_protected_eval(video_path: str) -> bool:
    normalized = video_path.replace("\\", "/")
    return "/eval_clips/" in normalized or normalized.startswith("eval_clips/")


def _empty_index() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": COURT_REVIEW_INDEX_ARTIFACT_TYPE,
        "entries": [],
    }


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return slug or "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
