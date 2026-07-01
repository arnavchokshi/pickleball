"""CPU local pixel-search recovery for ball tracks.

This post-processes a schema-valid ``BallTrack`` against source video pixels.
It does not consume sparse human click labels.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ball_court_filter import build_target_court_polygon, load_court_calibration, point_in_polygon_with_margin
from .ball_overlay import load_ball_track
from .schemas import BallTrack


DEFAULT_SUPPRESS_CONF_THRESHOLD = 1.01


@dataclass(frozen=True)
class _AcceptedSample:
    frame_index: int
    xy: tuple[float, float]


@dataclass(frozen=True)
class _SearchHit:
    xy: tuple[float, float]
    contrast: float
    luma: float
    local_mean: float


def filter_ball_track_local_search(
    *,
    video_path: str | Path,
    ball_track_path: str | Path,
    court_calibration_path: str | Path | None = None,
    search_radius_px: int = 12,
    min_contrast: float = 35.0,
    max_speed_px_per_second: float = 1800.0,
    base_jump_px: float = 20.0,
    max_prediction_gap_frames: int = 6,
    suppress_conf_threshold: float = DEFAULT_SUPPRESS_CONF_THRESHOLD,
    court_margin_px: float = 20.0,
    cv2_module: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover missing samples and suppress weak off-path samples using local pixels."""

    video = Path(video_path)
    if not video.is_file():
        raise ValueError(f"missing video file: {video}")
    _validate_positive_int(search_radius_px, "search_radius_px")
    _validate_positive(min_contrast, "min_contrast")
    _validate_positive(max_speed_px_per_second, "max_speed_px_per_second")
    _validate_nonnegative(base_jump_px, "base_jump_px")
    if max_prediction_gap_frames < 1:
        raise ValueError("max_prediction_gap_frames must be >= 1")
    _validate_nonnegative(suppress_conf_threshold, "suppress_conf_threshold")
    _validate_nonnegative(court_margin_px, "court_margin_px")

    track = load_ball_track(ball_track_path)
    payload = deepcopy(track.model_dump(mode="json"))
    samples_by_index = _payload_samples_by_frame_index(payload, fps=float(track.fps))
    visible_before = sum(1 for frame in payload["frames"] if bool(frame["visible"]))

    cv2 = cv2_module or _cv2()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video}")

    source_fps = _positive_float(cap.get(cv2.CAP_PROP_FPS)) or float(track.fps) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"cannot determine video frame size: {video}")

    court_polygon = _load_court_polygon(
        court_calibration_path,
        target_size=(width, height),
    )

    accepted: list[_AcceptedSample] = []
    processed_indices: set[int] = set()
    recovered_count = 0
    relocated_off_path_count = 0
    suppressed_off_path_count = 0
    evidence_miss_count = 0
    court_rejected_count = 0
    video_frame_count = 0
    source_index = 0

    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            video_frame_count += 1
            track_index = _track_index_for_source_frame(
                source_index=source_index,
                source_fps=source_fps,
                ball_fps=float(track.fps),
            )
            source_index += 1
            if track_index in processed_indices:
                continue
            frame = samples_by_index.get(track_index)
            if frame is None:
                continue
            processed_indices.add(track_index)

            prediction = _predict_xy(
                accepted=accepted,
                samples_by_index=samples_by_index,
                frame_index=track_index,
                fps=float(track.fps),
                max_speed_px_per_second=max_speed_px_per_second,
                base_jump_px=base_jump_px,
                max_prediction_gap_frames=max_prediction_gap_frames,
            )
            current_visible = bool(frame["visible"])
            current_xy = _xy(frame)
            if prediction is None:
                if current_visible:
                    accepted.append(_AcceptedSample(track_index, current_xy))
                continue

            hit = _find_local_pixel_evidence(
                image,
                center=prediction,
                radius_px=search_radius_px,
                min_contrast=min_contrast,
            )
            if hit is not None and not _court_allows(
                hit.xy,
                court_polygon=court_polygon,
                margin_px=court_margin_px,
            ):
                hit = None
                court_rejected_count += 1

            if not current_visible:
                if hit is None or not _motion_link_allowed_from_accepted(
                    accepted,
                    hit.xy,
                    frame_index=track_index,
                    fps=float(track.fps),
                    max_speed_px_per_second=max_speed_px_per_second,
                    base_jump_px=base_jump_px,
                ):
                    evidence_miss_count += 1
                    continue
                _apply_hit(frame, hit)
                accepted.append(_AcceptedSample(track_index, hit.xy))
                recovered_count += 1
                continue

            if _motion_link_allowed_from_accepted(
                accepted,
                current_xy,
                frame_index=track_index,
                fps=float(track.fps),
                max_speed_px_per_second=max_speed_px_per_second,
                base_jump_px=base_jump_px,
            ):
                accepted.append(_AcceptedSample(track_index, current_xy))
                continue

            if hit is not None and _motion_link_allowed_from_accepted(
                accepted,
                hit.xy,
                frame_index=track_index,
                fps=float(track.fps),
                max_speed_px_per_second=max_speed_px_per_second,
                base_jump_px=base_jump_px,
            ):
                _apply_hit(frame, hit)
                accepted.append(_AcceptedSample(track_index, hit.xy))
                relocated_off_path_count += 1
                continue

            if float(frame["conf"]) <= float(suppress_conf_threshold):
                _hide_frame(frame)
                suppressed_off_path_count += 1
            else:
                evidence_miss_count += 1
    finally:
        cap.release()

    BallTrack.model_validate(payload)
    visible_after = sum(1 for frame in payload["frames"] if bool(frame["visible"]))
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_local_search_filter",
        "status": "filtered_not_gate_verified",
        "source_ball_track": str(ball_track_path),
        "source_video": str(video),
        "court_calibration": str(court_calibration_path) if court_calibration_path is not None else None,
        "frame_count": len(payload["frames"]),
        "video_frame_count": video_frame_count,
        "processed_track_frame_count": len(processed_indices),
        "visible_before": visible_before,
        "visible_after": visible_after,
        "recovered_count": recovered_count,
        "relocated_off_path_count": relocated_off_path_count,
        "suppressed_off_path_count": suppressed_off_path_count,
        "evidence_miss_count": evidence_miss_count,
        "court_rejected_count": court_rejected_count,
        "search_radius_px": int(search_radius_px),
        "min_contrast": float(min_contrast),
        "max_speed_px_per_second": float(max_speed_px_per_second),
        "base_jump_px": float(base_jump_px),
        "max_prediction_gap_frames": int(max_prediction_gap_frames),
        "suppress_conf_threshold": float(suppress_conf_threshold),
        "court_margin_px": float(court_margin_px),
        "uses_human_clicks": False,
        "not_ground_truth": True,
    }
    return payload, summary


def write_local_search_ball_track(
    *,
    video_path: str | Path,
    ball_track_path: str | Path,
    out_path: str | Path,
    summary_path: str | Path,
    court_calibration_path: str | Path | None = None,
    search_radius_px: int = 12,
    min_contrast: float = 35.0,
    max_speed_px_per_second: float = 1800.0,
    base_jump_px: float = 20.0,
    max_prediction_gap_frames: int = 6,
    suppress_conf_threshold: float = DEFAULT_SUPPRESS_CONF_THRESHOLD,
    court_margin_px: float = 20.0,
    cv2_module: Any | None = None,
) -> dict[str, Any]:
    payload, summary = filter_ball_track_local_search(
        video_path=video_path,
        ball_track_path=ball_track_path,
        court_calibration_path=court_calibration_path,
        search_radius_px=search_radius_px,
        min_contrast=min_contrast,
        max_speed_px_per_second=max_speed_px_per_second,
        base_jump_px=base_jump_px,
        max_prediction_gap_frames=max_prediction_gap_frames,
        suppress_conf_threshold=suppress_conf_threshold,
        court_margin_px=court_margin_px,
        cv2_module=cv2_module,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_out = Path(summary_path)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _payload_samples_by_frame_index(payload: dict[str, Any], *, fps: float) -> dict[int, dict[str, Any]]:
    return {int(round(float(frame["t"]) * fps)): frame for frame in payload["frames"]}


def _track_index_for_source_frame(*, source_index: int, source_fps: float, ball_fps: float) -> int:
    return int(round((float(source_index) / source_fps) * ball_fps))


def _predict_xy(
    *,
    accepted: list[_AcceptedSample],
    samples_by_index: dict[int, dict[str, Any]],
    frame_index: int,
    fps: float,
    max_speed_px_per_second: float,
    base_jump_px: float,
    max_prediction_gap_frames: int,
) -> tuple[float, float] | None:
    if len(accepted) >= 2:
        prev = accepted[-2]
        last = accepted[-1]
        span = last.frame_index - prev.frame_index
        gap = frame_index - last.frame_index
        if span > 0 and 0 < gap <= max_prediction_gap_frames:
            velocity_x = (last.xy[0] - prev.xy[0]) / float(span)
            velocity_y = (last.xy[1] - prev.xy[1]) / float(span)
            return last.xy[0] + velocity_x * gap, last.xy[1] + velocity_y * gap

    if accepted:
        last = accepted[-1]
        gap = frame_index - last.frame_index
        if gap <= 0 or gap > max_prediction_gap_frames:
            return None
        future = _next_plausible_visible(
            samples_by_index=samples_by_index,
            last=last,
            after_index=frame_index,
            fps=fps,
            max_speed_px_per_second=max_speed_px_per_second,
            base_jump_px=base_jump_px,
            max_prediction_gap_frames=max_prediction_gap_frames,
        )
        if future is None:
            return last.xy
        total_gap = future.frame_index - last.frame_index
        if total_gap <= 0:
            return last.xy
        alpha = (frame_index - last.frame_index) / float(total_gap)
        return (
            last.xy[0] + (future.xy[0] - last.xy[0]) * alpha,
            last.xy[1] + (future.xy[1] - last.xy[1]) * alpha,
        )
    return None


def _next_plausible_visible(
    *,
    samples_by_index: dict[int, dict[str, Any]],
    last: _AcceptedSample,
    after_index: int,
    fps: float,
    max_speed_px_per_second: float,
    base_jump_px: float,
    max_prediction_gap_frames: int,
) -> _AcceptedSample | None:
    for candidate_index in sorted(index for index in samples_by_index if index > after_index):
        total_gap = candidate_index - last.frame_index
        if total_gap > max_prediction_gap_frames:
            return None
        frame = samples_by_index[candidate_index]
        if not bool(frame["visible"]):
            continue
        candidate_xy = _xy(frame)
        if _motion_link_allowed(
            last.xy,
            candidate_xy,
            gap_frames=total_gap,
            fps=fps,
            max_speed_px_per_second=max_speed_px_per_second,
            base_jump_px=base_jump_px,
        ):
            return _AcceptedSample(candidate_index, candidate_xy)
    return None


def _find_local_pixel_evidence(
    frame: Any,
    *,
    center: tuple[float, float],
    radius_px: int,
    min_contrast: float,
) -> _SearchHit | None:
    width, height = _frame_width_height(frame)
    center_x = int(round(center[0]))
    center_y = int(round(center[1]))
    x0 = max(0, center_x - radius_px)
    x1 = min(width - 1, center_x + radius_px)
    y0 = max(0, center_y - radius_px)
    y1 = min(height - 1, center_y + radius_px)
    if x0 > x1 or y0 > y1:
        return None

    samples: list[tuple[float, int, int]] = []
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            samples.append((_luma(_pixel_at(frame, x, y)), x, y))
    if not samples:
        return None

    local_mean = sum(value for value, _x, _y in samples) / float(len(samples))
    best_luma = 0.0
    best_x = center_x
    best_y = center_y
    best_contrast = -1.0
    best_distance = float("inf")
    for value, x, y in samples:
        contrast = abs(value - local_mean)
        distance = math.hypot(float(x) - center[0], float(y) - center[1])
        if contrast > best_contrast or (math.isclose(contrast, best_contrast) and distance < best_distance):
            best_luma = value
            best_x = x
            best_y = y
            best_contrast = contrast
            best_distance = distance

    if best_contrast < float(min_contrast):
        return None
    return _SearchHit(
        xy=(float(best_x), float(best_y)),
        contrast=float(best_contrast),
        luma=float(best_luma),
        local_mean=float(local_mean),
    )


def _apply_hit(frame: dict[str, Any], hit: _SearchHit) -> None:
    frame["xy"] = [float(hit.xy[0]), float(hit.xy[1])]
    frame["conf"] = _confidence_from_contrast(hit.contrast)
    frame["visible"] = True
    frame["approx"] = True
    frame.pop("world_xyz", None)


def _confidence_from_contrast(contrast: float) -> float:
    return max(0.05, min(0.99, float(contrast) / 255.0))


def _hide_frame(frame: dict[str, Any]) -> None:
    frame["visible"] = False
    frame["conf"] = 0.0
    frame["approx"] = False
    frame.pop("world_xyz", None)


def _motion_link_allowed_from_accepted(
    accepted: list[_AcceptedSample],
    xy: tuple[float, float],
    *,
    frame_index: int,
    fps: float,
    max_speed_px_per_second: float,
    base_jump_px: float,
) -> bool:
    if not accepted:
        return True
    last = accepted[-1]
    gap = frame_index - last.frame_index
    return _motion_link_allowed(
        last.xy,
        xy,
        gap_frames=gap,
        fps=fps,
        max_speed_px_per_second=max_speed_px_per_second,
        base_jump_px=base_jump_px,
    )


def _motion_link_allowed(
    left_xy: tuple[float, float],
    right_xy: tuple[float, float],
    *,
    gap_frames: int,
    fps: float,
    max_speed_px_per_second: float,
    base_jump_px: float,
) -> bool:
    if gap_frames <= 0:
        return False
    allowed = float(base_jump_px) + (float(max_speed_px_per_second) * float(gap_frames) / float(fps))
    return _distance(left_xy, right_xy) <= allowed


def _load_court_polygon(
    court_calibration_path: str | Path | None,
    *,
    target_size: tuple[int, int],
) -> list[list[float]] | None:
    if court_calibration_path is None:
        return None
    calibration = load_court_calibration(court_calibration_path)
    return build_target_court_polygon(calibration, target_size=target_size)


def _court_allows(
    xy: tuple[float, float],
    *,
    court_polygon: list[list[float]] | None,
    margin_px: float,
) -> bool:
    if court_polygon is None:
        return True
    return point_in_polygon_with_margin(xy, court_polygon, margin_px=margin_px)


def _frame_width_height(frame: Any) -> tuple[int, int]:
    shape = getattr(frame, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    height = len(frame)
    width = len(frame[0]) if height else 0
    return int(width), int(height)


def _pixel_at(frame: Any, x: int, y: int) -> Any:
    try:
        return frame[y, x]
    except (TypeError, KeyError, IndexError):
        return frame[y][x]


def _luma(pixel: Any) -> float:
    try:
        values = list(pixel)
    except TypeError:
        return float(pixel)
    if len(values) >= 3:
        b, g, r = values[:3]
        return 0.114 * float(b) + 0.587 * float(g) + 0.299 * float(r)
    if len(values) == 1:
        return float(values[0])
    return sum(float(value) for value in values) / float(len(values))


def _xy(frame: dict[str, Any]) -> tuple[float, float]:
    return float(frame["xy"][0]), float(frame["xy"][1])


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result <= 0.0:
        return None
    return result


def _validate_positive(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be > 0")


def _validate_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be >= 0")


def _validate_positive_int(value: int, name: str) -> None:
    if int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _cv2() -> Any:
    import cv2

    return cv2


__all__ = [
    "DEFAULT_SUPPRESS_CONF_THRESHOLD",
    "filter_ball_track_local_search",
    "write_local_search_ball_track",
]
