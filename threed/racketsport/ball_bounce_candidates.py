"""Label-free ball bounce candidate proposals from 2D track geometry."""

from __future__ import annotations

import json
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .ball_arc_solver import BALL_RADIUS_M, intersect_ray_z, pixel_ray_world
from .io_decode import frame_time_lookup, time_for_frame


@dataclass(frozen=True)
class BounceCandidateConfig:
    smoothing_window: int = 5
    min_run_len: int = 7
    max_visible_gap_frames: int = 3
    min_candidate_separation_frames: int = 6
    court_margin_m: float = 2.0
    sharpness_relative_floor: float = 0.15
    min_gap_hidden_frames: int = 2
    max_gap_hidden_frames: int = 24
    gap_window_frames: int = 4
    gap_min_image_speed_px_s: float = 40.0
    gap_max_ray_rms_m: float = 2.0
    gap_min_vertical_speed_mps: float = 0.5
    gap_max_speed_mps: float = 35.0


def build_bounce_candidate_payload(
    ball_track: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    clip_id: str = "",
    config: BounceCandidateConfig | None = None,
    frame_times: Any = None,
) -> dict[str, Any]:
    cfg = config or BounceCandidateConfig()
    bounds = _court_bounds(calibration)
    frame_time_map = frame_time_lookup(frame_times if frame_times is not None else ball_track.get("frame_times"))
    runs = _visible_runs(ball_track, max_gap_frames=cfg.max_visible_gap_frames, frame_times=frame_time_map)
    all_candidates: list[dict[str, Any]] = []
    for run in runs:
        all_candidates.extend(_find_cusp_candidates_in_run(run, calibration, bounds, cfg))
    all_candidates.extend(_find_gap_ballistic_candidates(runs, calibration, bounds, cfg, frame_times=frame_time_map))
    raw_count = len(all_candidates)
    in_bounds = [candidate for candidate in all_candidates if candidate["in_extended_court_bounds"]]
    deduped = _dedupe(in_bounds, cfg.min_candidate_separation_frames)
    payload = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_bounce_candidates_track_geometry",
        "clip_id": clip_id,
        "source": "track_geometry_candidate",
        "not_ground_truth": True,
        "human_reviewed": False,
        "candidate_prediction": True,
        "diagnostic_only": False,
        "policy": {
            "notes": [
                "Built from the 2D ball track + court calibration geometry only.",
                "No reviewed label or reviewed-bounce file is required or consumed.",
                "These are algorithmic geometric candidates, not human-reviewed ground truth.",
            ]
        },
        "method": {
            "smoothing_window": cfg.smoothing_window,
            "min_run_len": cfg.min_run_len,
            "max_visible_gap_frames": cfg.max_visible_gap_frames,
            "min_candidate_separation_frames": cfg.min_candidate_separation_frames,
            "court_margin_m": cfg.court_margin_m,
            "sharpness_relative_floor": cfg.sharpness_relative_floor,
            "min_gap_hidden_frames": cfg.min_gap_hidden_frames,
            "max_gap_hidden_frames": cfg.max_gap_hidden_frames,
            "gap_window_frames": cfg.gap_window_frames,
            "gap_min_image_speed_px_s": cfg.gap_min_image_speed_px_s,
            "gap_max_ray_rms_m": cfg.gap_max_ray_rms_m,
            "gap_min_vertical_speed_mps": cfg.gap_min_vertical_speed_mps,
            "gap_max_speed_mps": cfg.gap_max_speed_mps,
        },
        "summary": {
            "raw_local_extrema_count": sum(1 for item in all_candidates if item["method"] == "image_y_cusp"),
            "gap_ballistic_intersection_count": sum(1 for item in all_candidates if item["method"] == "gap_ballistic_intersection"),
            "raw_candidate_count": raw_count,
            "in_court_bounds_count": len(in_bounds),
            "final_candidate_count": len(deduped),
        },
        "candidates": deduped,
    }
    return payload


def write_bounce_candidate_payload(
    *,
    ball_track_path: Path,
    calibration_path: Path,
    out_path: Path,
    clip_id: str = "",
    config: BounceCandidateConfig | None = None,
    frame_times: Any = None,
    frame_times_path: Path | None = None,
) -> dict[str, Any]:
    payload = build_bounce_candidate_payload(
        json.loads(ball_track_path.read_text(encoding="utf-8")),
        json.loads(calibration_path.read_text(encoding="utf-8")),
        clip_id=clip_id,
        config=config,
        frame_times=frame_times_path if frame_times_path is not None else frame_times,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _court_bounds(calibration: Mapping[str, Any]) -> tuple[float, float, float, float]:
    world_pts = calibration.get("world_pts")
    if not isinstance(world_pts, Sequence) or isinstance(world_pts, (str, bytes)):
        return (-10.0, 10.0, -15.0, 15.0)
    xs: list[float] = []
    ys: list[float] = []
    for point in world_pts:
        if isinstance(point, Sequence) and not isinstance(point, (str, bytes)) and len(point) >= 2:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
    if not xs or not ys:
        return (-10.0, 10.0, -15.0, 15.0)
    return (min(xs), max(xs), min(ys), max(ys))


def _visible_runs(
    ball_track: Mapping[str, Any],
    *,
    max_gap_frames: int,
    frame_times: Any = None,
) -> list[list[dict[str, Any]]]:
    frames = ball_track.get("frames")
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        return []
    fps = _float_or_none(ball_track.get("fps")) or 30.0
    observations: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or frame.get("visible") is not True:
            continue
        xy = _xy_tuple(frame.get("xy"))
        if xy is None:
            continue
        observations.append(
            {
                "frame": index,
                "t": _float_or_none(frame.get("t"))
                if _float_or_none(frame.get("t")) is not None
                else time_for_frame(index, frame_times=frame_times, fps=fps),
                "xy": xy,
                "conf": _float_or_none(frame.get("conf")) or 1.0,
            }
        )
    observations.sort(key=lambda item: item["frame"])
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    prev_frame: int | None = None
    for obs in observations:
        if prev_frame is not None and int(obs["frame"]) - prev_frame > max_gap_frames:
            if current:
                runs.append(current)
            current = []
        current.append(obs)
        prev_frame = int(obs["frame"])
    if current:
        runs.append(current)
    return runs


def _find_cusp_candidates_in_run(
    run: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    bounds: tuple[float, float, float, float],
    config: BounceCandidateConfig,
) -> list[dict[str, Any]]:
    if len(run) < config.min_run_len:
        return []
    ts = [float(item["t"]) for item in run]
    xs_raw = [float(item["xy"][0]) for item in run]
    ys_raw = [float(item["xy"][1]) for item in run]
    xs = _moving_average(xs_raw, config.smoothing_window)
    ys = _moving_average(ys_raw, config.smoothing_window)
    del xs  # x smoothing is retained for parity with the diagnostic prototype.
    n = len(run)
    vy = [0.0] * n
    for i in range(1, n - 1):
        dt = ts[i + 1] - ts[i - 1]
        vy[i] = (ys[i + 1] - ys[i - 1]) / dt if dt > 1e-9 else 0.0
    median_abs_vy = sorted(abs(v) for v in vy)[n // 2] if n else 0.0
    candidates: list[dict[str, Any]] = []
    for i in range(2, n - 2):
        if not (vy[i - 1] > 0.0 and vy[i + 1] < 0.0):
            continue
        sharpness = abs(vy[i - 1]) + abs(vy[i + 1])
        if median_abs_vy > 1e-6 and sharpness < config.sharpness_relative_floor * median_abs_vy * 2:
            continue
        xy = (xs_raw[i], ys_raw[i])
        candidate = _candidate_from_xy(
            frame=int(run[i]["frame"]),
            t=float(run[i]["t"]),
            xy=xy,
            calibration=calibration,
            bounds=bounds,
            config=config,
            method="image_y_cusp",
            extra={
                "vy_before_px_s": round(vy[i - 1], 3),
                "vy_after_px_s": round(vy[i + 1], 3),
                "sharpness_px_s": round(sharpness, 3),
                "conf": float(run[i].get("conf", 1.0)),
            },
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _find_gap_ballistic_candidates(
    runs: Sequence[Sequence[Mapping[str, Any]]],
    calibration: Mapping[str, Any],
    bounds: tuple[float, float, float, float],
    config: BounceCandidateConfig,
    *,
    frame_times: Mapping[int, float] | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for left, right in zip(runs, runs[1:]):
        if not left or not right:
            continue
        gap_frames = int(right[0]["frame"]) - int(left[-1]["frame"]) - 1
        if gap_frames < config.min_gap_hidden_frames or gap_frames > config.max_gap_hidden_frames:
            continue
        left_slope = _image_y_slope(left[-config.gap_window_frames :])
        right_slope = _image_y_slope(right[: config.gap_window_frames])
        fit = _best_gap_ballistic_fit(
            left,
            right,
            calibration,
            bounds,
            config,
            left_image_y_slope=left_slope,
            right_image_y_slope=right_slope,
            frame_times=frame_times,
        )
        if fit is None:
            continue
        frame = int(fit["frame"])
        t = float(fit["t"])
        xy = fit["xy"]
        candidate = _candidate_from_xy(
            frame=frame,
            t=t,
            xy=xy,
            calibration=calibration,
            bounds=bounds,
            config=config,
            method="gap_ballistic_intersection",
            extra={
                "gap_start_frame": int(left[-1]["frame"]) + 1,
                "gap_end_frame": int(right[0]["frame"]) - 1,
                "gap_hidden_frame_count": gap_frames,
                "pre_gap_image_y_slope_px_s": None if left_slope is None else round(left_slope, 3),
                "post_gap_image_y_slope_px_s": None if right_slope is None else round(right_slope, 3),
                "pre_gap_ray_rms_m": round(float(fit["pre_rms_m"]), 6),
                "post_gap_ray_rms_m": round(float(fit["post_rms_m"]), 6),
                "pre_gap_bounce_vz_mps": round(float(fit["pre_velocity_mps"][2]), 6),
                "post_gap_bounce_vz_mps": round(float(fit["post_velocity_mps"][2]), 6),
                "pre_gap_speed_mps": round(_norm3(fit["pre_velocity_mps"]), 6),
                "post_gap_speed_mps": round(_norm3(fit["post_velocity_mps"]), 6),
            },
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _best_gap_ballistic_fit(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    bounds: tuple[float, float, float, float],
    config: BounceCandidateConfig,
    *,
    left_image_y_slope: float | None,
    right_image_y_slope: float | None,
    frame_times: Mapping[int, float] | None,
) -> dict[str, Any] | None:
    left_samples = list(left[-config.gap_window_frames :])
    right_samples = list(right[: config.gap_window_frames])
    if len(left_samples) < 2 or len(right_samples) < 2:
        return None
    start_frame = int(left[-1]["frame"])
    end_frame = int(right[0]["frame"])
    boundary_dt = max(float(right[0]["t"]) - float(left[-1]["t"]), 1e-9)
    fps = (end_frame - start_frame) / boundary_dt
    if len(left) <= 2:
        candidate_frames = list(range(start_frame, end_frame + 1))
    else:
        candidate_frames = list(range(start_frame + 1, end_frame))
        if not candidate_frames:
            candidate_frames = list(range(start_frame, end_frame + 1))
    best: dict[str, Any] | None = None
    for frame in candidate_frames:
        t = _candidate_time_for_frame(
            frame,
            frame_times=frame_times,
            fallback_fps=fps,
            start_frame=start_frame,
            start_t=float(left[-1]["t"]),
            end_frame=end_frame,
            end_t=float(right[0]["t"]),
        )
        if frame == start_frame:
            t = float(left[-1]["t"])
            xy = _xy_tuple(left[-1].get("xy"))
        elif frame == end_frame:
            t = float(right[0]["t"])
            xy = _xy_tuple(right[0].get("xy"))
        else:
            xy = _interpolate_xy(left[-1], right[0], t)
        if xy is None:
            continue
        try:
            origin, direction = pixel_ray_world(calibration, xy)
            bounce_xyz = intersect_ray_z(origin, direction, BALL_RADIUS_M)
        except ValueError:
            continue
        if not _world_xy_in_bounds(bounce_xyz, bounds, config.court_margin_m):
            continue
        pre_fit = _fit_velocity_to_rays(left_samples, calibration, bounce_xyz, t)
        post_fit = _fit_velocity_to_rays(right_samples, calibration, bounce_xyz, t)
        if pre_fit is None or post_fit is None:
            continue
        pre_velocity, pre_rms = pre_fit
        post_velocity, post_rms = post_fit
        pre_descending = pre_velocity[2] <= -config.gap_min_vertical_speed_mps or (
            left_image_y_slope is not None and left_image_y_slope >= config.gap_min_image_speed_px_s
        )
        post_ascending = post_velocity[2] >= config.gap_min_vertical_speed_mps or (
            right_image_y_slope is not None and right_image_y_slope <= -config.gap_min_image_speed_px_s
        )
        if not pre_descending:
            continue
        if not post_ascending:
            continue
        if _norm3(pre_velocity) > config.gap_max_speed_mps or _norm3(post_velocity) > config.gap_max_speed_mps:
            continue
        if pre_rms > config.gap_max_ray_rms_m or post_rms > config.gap_max_ray_rms_m:
            continue
        score = pre_rms + post_rms
        candidate = {
            "frame": frame,
            "t": t,
            "xy": xy,
            "pre_velocity_mps": pre_velocity,
            "post_velocity_mps": post_velocity,
            "pre_rms_m": pre_rms,
            "post_rms_m": post_rms,
            "score": score,
        }
        if best is None or score < float(best["score"]):
            best = candidate
    return best


def _candidate_time_for_frame(
    frame: int,
    *,
    frame_times: Mapping[int, float] | None,
    fallback_fps: float,
    start_frame: int,
    start_t: float,
    end_frame: int,
    end_t: float,
) -> float:
    if frame_times and frame in frame_times:
        return float(frame_times[frame])
    if end_frame != start_frame:
        alpha = (float(frame) - float(start_frame)) / (float(end_frame) - float(start_frame))
        return start_t + alpha * (end_t - start_t)
    return time_for_frame(frame, frame_times=frame_times, fps=fallback_fps)


def _fit_velocity_to_rays(
    samples: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    bounce_xyz: tuple[float, float, float],
    bounce_t: float,
) -> tuple[tuple[float, float, float], float] | None:
    try:
        import numpy as np
    except ImportError:
        return None
    rows: list[list[float]] = []
    values: list[float] = []
    eye = np.eye(3)
    gravity = np.array([0.0, 0.0, -9.80665], dtype=float)
    p_bounce = np.array(bounce_xyz, dtype=float)
    for sample in samples:
        xy = _xy_tuple(sample.get("xy"))
        if xy is None:
            continue
        try:
            origin, direction = pixel_ray_world(calibration, xy)
        except ValueError:
            continue
        d = np.array(direction, dtype=float)
        projection = eye - np.outer(d, d)
        dt = float(sample["t"]) - bounce_t
        constant = p_bounce + 0.5 * gravity * dt * dt - np.array(origin, dtype=float)
        matrix = projection * dt
        rhs = -projection @ constant
        rows.extend(matrix.tolist())
        values.extend(rhs.tolist())
    if len(rows) < 6:
        return None
    a = np.array(rows, dtype=float)
    b = np.array(values, dtype=float)
    velocity, *_ = np.linalg.lstsq(a, b, rcond=None)
    residuals: list[float] = []
    for sample in samples:
        xy = _xy_tuple(sample.get("xy"))
        if xy is None:
            continue
        origin, direction = pixel_ray_world(calibration, xy)
        dt = float(sample["t"]) - bounce_t
        point = tuple((p_bounce + velocity * dt + 0.5 * gravity * dt * dt).tolist())
        residuals.append(_distance_point_to_ray(point, origin, direction))
    if not residuals:
        return None
    rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    return (float(velocity[0]), float(velocity[1]), float(velocity[2])), rms


def _candidate_from_xy(
    *,
    frame: int,
    t: float,
    xy: tuple[float, float],
    calibration: Mapping[str, Any],
    bounds: tuple[float, float, float, float],
    config: BounceCandidateConfig,
    method: str,
    extra: Mapping[str, Any],
) -> dict[str, Any] | None:
    try:
        origin, direction = pixel_ray_world(calibration, xy)
        world_xyz = intersect_ray_z(origin, direction, BALL_RADIUS_M)
    except ValueError:
        return None
    wx, wy, _ = world_xyz
    x_min, x_max, y_min, y_max = bounds
    in_bounds = (
        (x_min - config.court_margin_m) <= wx <= (x_max + config.court_margin_m)
        and (y_min - config.court_margin_m) <= wy <= (y_max + config.court_margin_m)
    )
    return {
        "frame": int(frame),
        "t": float(t),
        "xy": [round(float(xy[0]), 3), round(float(xy[1]), 3)],
        "method": method,
        "source": "track_geometry_candidate",
        "not_ground_truth": True,
        "human_reviewed": False,
        "candidate_prediction": True,
        "world_xy_at_ball_radius": [round(wx, 4), round(wy, 4)],
        "in_extended_court_bounds": bool(in_bounds),
        **dict(extra),
    }


def _world_xy_in_bounds(world_xyz: Sequence[float], bounds: tuple[float, float, float, float], margin_m: float) -> bool:
    x_min, x_max, y_min, y_max = bounds
    wx = float(world_xyz[0])
    wy = float(world_xyz[1])
    return (x_min - margin_m) <= wx <= (x_max + margin_m) and (y_min - margin_m) <= wy <= (y_max + margin_m)


def _distance_point_to_ray(
    point: Sequence[float],
    origin: Sequence[float],
    direction: Sequence[float],
) -> float:
    op = (float(point[0]) - float(origin[0]), float(point[1]) - float(origin[1]), float(point[2]) - float(origin[2]))
    along = op[0] * float(direction[0]) + op[1] * float(direction[1]) + op[2] * float(direction[2])
    closest = (
        float(origin[0]) + along * float(direction[0]),
        float(origin[1]) + along * float(direction[1]),
        float(origin[2]) + along * float(direction[2]),
    )
    return math.sqrt(
        (float(point[0]) - closest[0]) ** 2
        + (float(point[1]) - closest[1]) ** 2
        + (float(point[2]) - closest[2]) ** 2
    )


def _norm3(value: Sequence[float]) -> float:
    return math.sqrt(float(value[0]) ** 2 + float(value[1]) ** 2 + float(value[2]) ** 2)


def _dedupe(candidates: Sequence[Mapping[str, Any]], min_separation_frames: int) -> list[dict[str, Any]]:
    priority = {"gap_ballistic_intersection": 0, "image_y_cusp": 1}
    candidates_sorted = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda item: (priority.get(str(item.get("method")), 9), -float(item.get("sharpness_px_s", 0.0))),
    )
    kept: list[dict[str, Any]] = []
    kept_frames: list[int] = []
    for candidate in candidates_sorted:
        frame = int(candidate["frame"])
        if any(abs(frame - existing) < min_separation_frames for existing in kept_frames):
            continue
        kept.append(candidate)
        kept_frames.append(frame)
    return sorted(kept, key=lambda item: (int(item["frame"]), str(item.get("method"))))


def _moving_average(values: Sequence[float], window: int) -> list[float]:
    n = len(values)
    half = max(0, int(window) // 2)
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def _image_y_slope(run: Sequence[Mapping[str, Any]]) -> float | None:
    if len(run) < 2:
        return None
    first = run[0]
    last = run[-1]
    dt = float(last["t"]) - float(first["t"])
    if dt <= 1e-9:
        return None
    return (float(last["xy"][1]) - float(first["xy"][1])) / dt


def _interpolate_xy(left: Mapping[str, Any], right: Mapping[str, Any], t: float) -> tuple[float, float] | None:
    left_xy = _xy_tuple(left.get("xy"))
    right_xy = _xy_tuple(right.get("xy"))
    if left_xy is None or right_xy is None:
        return None
    dt = float(right["t"]) - float(left["t"])
    alpha = 0.0 if dt <= 1e-9 else (t - float(left["t"])) / dt
    alpha = max(0.0, min(1.0, alpha))
    return (left_xy[0] + (right_xy[0] - left_xy[0]) * alpha, left_xy[1] + (right_xy[1] - left_xy[1]) * alpha)


def _xy_tuple(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    try:
        xy = (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None
    if not all(component == component and component not in (float("inf"), float("-inf")) for component in xy):
        return None
    return xy


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


__all__ = [
    "BounceCandidateConfig",
    "build_bounce_candidate_payload",
    "write_bounce_candidate_payload",
]
