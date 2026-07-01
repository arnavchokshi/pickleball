"""World-speed and pixel-jump ghost gate for BALL tracks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .ball_court_filter import load_court_calibration
from .ball_overlay import load_ball_track
from .court_calibration import calibration_image_size, project_image_points_to_world, project_planar_points
from .schemas import BallTrack, CourtCalibration


ARTIFACT_TYPE = "racketsport_ball_world_speed_gate"
STATUS_TESTED = "TESTED-ON-REAL-DATA"


def filter_ball_track_world_speed(
    *,
    ball_track_path: str | Path,
    calibration: CourtCalibration,
    target_size: tuple[int, int] | None = None,
    max_world_speed_mps: float = 30.0,
    base_jump_px: float = 60.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if max_world_speed_mps <= 0.0 or not math.isfinite(float(max_world_speed_mps)):
        raise ValueError("max_world_speed_mps must be > 0")
    if base_jump_px < 0.0 or not math.isfinite(float(base_jump_px)):
        raise ValueError("base_jump_px must be >= 0")

    track = load_ball_track(ball_track_path)
    payload = track.model_dump(mode="json")
    visible_before = 0
    visible_after = 0
    evaluated_link_count = 0
    rejected_world_speed_count = 0
    rejected_pixel_jump_count = 0
    max_observed_world_speed_mps = 0.0
    max_observed_pixel_jump_px = 0.0
    max_observed_allowed_pixel_jump_px = 0.0
    last_accepted: dict[str, Any] | None = None

    for frame in payload["frames"]:
        if not bool(frame["visible"]):
            continue
        visible_before += 1
        sample = _sample_from_frame(frame, calibration=calibration, target_size=target_size)
        if last_accepted is None:
            visible_after += 1
            last_accepted = sample
            continue

        dt = sample["t"] - last_accepted["t"]
        evaluated_link_count += 1
        world_distance_m = _distance(sample["world_xy"], last_accepted["world_xy"])
        pixel_distance_px = _distance(sample["xy"], last_accepted["xy"])
        speed_mps = math.inf if dt <= 0.0 else world_distance_m / dt
        local_px_per_m = _local_px_per_m(last_accepted["world_xy"], calibration=calibration, target_size=target_size)
        allowed_pixel_jump_px = float(base_jump_px) + float(max_world_speed_mps) * max(dt, 0.0) * local_px_per_m

        if math.isfinite(speed_mps):
            max_observed_world_speed_mps = max(max_observed_world_speed_mps, speed_mps)
        else:
            max_observed_world_speed_mps = math.inf
        max_observed_pixel_jump_px = max(max_observed_pixel_jump_px, pixel_distance_px)
        max_observed_allowed_pixel_jump_px = max(max_observed_allowed_pixel_jump_px, allowed_pixel_jump_px)

        rejects_world_speed = speed_mps > float(max_world_speed_mps)
        rejects_pixel_jump = pixel_distance_px > allowed_pixel_jump_px
        if rejects_world_speed or rejects_pixel_jump:
            _hide_frame(frame)
            if rejects_world_speed:
                rejected_world_speed_count += 1
            if rejects_pixel_jump:
                rejected_pixel_jump_count += 1
            continue

        visible_after += 1
        last_accepted = sample

    BallTrack.model_validate(payload)
    summary = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": STATUS_TESTED,
        "source_ball_track": str(ball_track_path),
        "sport": calibration.sport,
        "target_size": list(target_size) if target_size is not None else None,
        "coordinate_model": "court_plane_xy",
        "max_world_speed_mps": float(max_world_speed_mps),
        "base_jump_px": float(base_jump_px),
        "frame_count": len(payload["frames"]),
        "visible_before": visible_before,
        "visible_after": visible_after,
        "evaluated_link_count": evaluated_link_count,
        "rejected_world_speed_count": rejected_world_speed_count,
        "rejected_pixel_jump_count": rejected_pixel_jump_count,
        "max_observed_world_speed_mps": _json_finite_or_string(max_observed_world_speed_mps),
        "max_observed_pixel_jump_px": float(max_observed_pixel_jump_px),
        "max_observed_allowed_pixel_jump_px": float(max_observed_allowed_pixel_jump_px),
        "uses_human_clicks": False,
        "not_ground_truth": True,
    }
    return payload, summary


def write_world_speed_filtered_ball_track(
    *,
    ball_track_path: str | Path,
    calibration_path: str | Path,
    out_path: str | Path,
    summary_path: str | Path,
    target_size: tuple[int, int] | None = None,
    max_world_speed_mps: float = 30.0,
    base_jump_px: float = 60.0,
) -> dict[str, Any]:
    calibration = load_court_calibration(calibration_path)
    payload, summary = filter_ball_track_world_speed(
        ball_track_path=ball_track_path,
        calibration=calibration,
        target_size=target_size,
        max_world_speed_mps=max_world_speed_mps,
        base_jump_px=base_jump_px,
    )
    summary = {**summary, "calibration_path": str(calibration_path)}

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_out = Path(summary_path)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _sample_from_frame(
    frame: dict[str, Any],
    *,
    calibration: CourtCalibration,
    target_size: tuple[int, int] | None,
) -> dict[str, Any]:
    xy = [float(frame["xy"][0]), float(frame["xy"][1])]
    return {
        "t": float(frame["t"]),
        "xy": xy,
        "world_xy": _target_image_xy_to_world_xy(xy, calibration=calibration, target_size=target_size),
    }


def _target_image_xy_to_world_xy(
    xy: list[float],
    *,
    calibration: CourtCalibration,
    target_size: tuple[int, int] | None,
) -> list[float]:
    x, y = float(xy[0]), float(xy[1])
    if target_size is not None:
        calibration_width, calibration_height = _calibration_image_size(calibration)
        target_width, target_height = target_size
        if target_width <= 0 or target_height <= 0:
            raise ValueError("target_size values must be > 0")
        x *= calibration_width / float(target_width)
        y *= calibration_height / float(target_height)
    world = project_image_points_to_world(calibration.homography, [[x, y]])[0]
    return [float(world[0]), float(world[1])]


def _world_xy_to_target_image_xy(
    world_xy: list[float],
    *,
    calibration: CourtCalibration,
    target_size: tuple[int, int] | None,
) -> list[float]:
    image_xy = project_planar_points(calibration.homography, [world_xy])[0]
    x, y = float(image_xy[0]), float(image_xy[1])
    if target_size is not None:
        calibration_width, calibration_height = _calibration_image_size(calibration)
        target_width, target_height = target_size
        if target_width <= 0 or target_height <= 0:
            raise ValueError("target_size values must be > 0")
        x *= float(target_width) / calibration_width
        y *= float(target_height) / calibration_height
    return [x, y]


def _local_px_per_m(
    world_xy: list[float],
    *,
    calibration: CourtCalibration,
    target_size: tuple[int, int] | None,
) -> float:
    origin = _world_xy_to_target_image_xy(world_xy, calibration=calibration, target_size=target_size)
    x_axis = _world_xy_to_target_image_xy(
        [float(world_xy[0]) + 1.0, float(world_xy[1])],
        calibration=calibration,
        target_size=target_size,
    )
    y_axis = _world_xy_to_target_image_xy(
        [float(world_xy[0]), float(world_xy[1]) + 1.0],
        calibration=calibration,
        target_size=target_size,
    )
    return max(_distance(origin, x_axis), _distance(origin, y_axis))


def _calibration_image_size(calibration: CourtCalibration) -> tuple[float, float]:
    width, height = calibration_image_size(calibration)
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0.0 or height <= 0.0:
        raise ValueError("cannot infer calibration image size from intrinsics")
    return float(width), float(height)


def _hide_frame(frame: dict[str, Any]) -> None:
    frame["visible"] = False
    frame["conf"] = 0.0
    frame["approx"] = False
    frame.pop("world_xyz", None)
    frame.pop("speed_mps", None)


def _distance(left: list[float], right: list[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _json_finite_or_string(value: float) -> float | str:
    return float(value) if math.isfinite(float(value)) else "inf"


__all__ = [
    "ARTIFACT_TYPE",
    "filter_ball_track_world_speed",
    "write_world_speed_filtered_ball_track",
]
