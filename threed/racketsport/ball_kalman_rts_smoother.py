"""Constant-acceleration Kalman + RTS smoothing for BALL tracks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from .ball_overlay import load_ball_track
from .schemas import BallTrack


ARTIFACT_TYPE = "racketsport_ball_kalman_rts_smoother"
STATUS_TESTED = "TESTED-ON-REAL-DATA"


def smooth_ball_track_kalman_rts(
    *,
    ball_track_path: str | Path,
    max_gap_fill_frames: int = 6,
    measurement_variance_px: float = 4.0,
    process_variance: float = 0.05,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if max_gap_fill_frames < 0:
        raise ValueError("max_gap_fill_frames must be >= 0")
    if measurement_variance_px <= 0.0 or not math.isfinite(float(measurement_variance_px)):
        raise ValueError("measurement_variance_px must be > 0")
    if process_variance < 0.0 or not math.isfinite(float(process_variance)):
        raise ValueError("process_variance must be >= 0")

    track = load_ball_track(ball_track_path)
    payload = track.model_dump(mode="json")
    fps = float(track.fps)
    dt = 1.0 / fps
    segments = _smoothing_segments(payload["frames"], max_gap_fill_frames=max_gap_fill_frames)
    filled_gap_frame_count = 0
    max_filled_gap_frames = 0
    smoothed_visible_count = 0

    for start, end in segments:
        segment_frames = payload["frames"][start : end + 1]
        observations_x = [_measurement(frame, axis=0) for frame in segment_frames]
        observations_y = [_measurement(frame, axis=1) for frame in segment_frames]
        smoothed_x = _smooth_axis(
            observations_x,
            dt=dt,
            measurement_variance=float(measurement_variance_px),
            process_variance=float(process_variance),
        )
        smoothed_y = _smooth_axis(
            observations_y,
            dt=dt,
            measurement_variance=float(measurement_variance_px),
            process_variance=float(process_variance),
        )
        gap_run = 0
        for offset, frame in enumerate(segment_frames):
            was_visible = bool(frame["visible"])
            if was_visible:
                gap_run = 0
            else:
                gap_run += 1
                filled_gap_frame_count += 1
                max_filled_gap_frames = max(max_filled_gap_frames, gap_run)
            frame["xy"] = [float(smoothed_x[offset][0]), float(smoothed_y[offset][0])]
            frame["visible"] = True
            frame["approx"] = bool(frame.get("approx", False)) or not was_visible
            if not was_visible:
                frame["conf"] = _filled_confidence(segment_frames, offset)
            smoothed_visible_count += 1

    BallTrack.model_validate(payload)
    jitter_px_std = _jitter_px_std(payload["frames"])
    summary = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": STATUS_TESTED,
        "source_ball_track": str(ball_track_path),
        "motion_model": "constant_acceleration_xy",
        "smoother": "forward_kalman_backward_rts",
        "max_gap_fill_frames": int(max_gap_fill_frames),
        "measurement_variance_px": float(measurement_variance_px),
        "process_variance": float(process_variance),
        "frame_count": len(payload["frames"]),
        "segment_count": len(segments),
        "filled_gap_frame_count": filled_gap_frame_count,
        "max_filled_gap_frames": max_filled_gap_frames,
        "smoothed_visible_count": smoothed_visible_count,
        "jitter_metric": "std_second_difference_px_on_visible_runs",
        "jitter_px_std": float(jitter_px_std),
        "uses_human_clicks": False,
        "not_ground_truth": True,
    }
    return payload, summary


def write_kalman_rts_smoothed_ball_track(
    *,
    ball_track_path: str | Path,
    out_path: str | Path,
    summary_path: str | Path,
    max_gap_fill_frames: int = 6,
    measurement_variance_px: float = 4.0,
    process_variance: float = 0.05,
) -> dict[str, Any]:
    payload, summary = smooth_ball_track_kalman_rts(
        ball_track_path=ball_track_path,
        max_gap_fill_frames=max_gap_fill_frames,
        measurement_variance_px=measurement_variance_px,
        process_variance=process_variance,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_out = Path(summary_path)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _smoothing_segments(frames: Sequence[dict[str, Any]], *, max_gap_fill_frames: int) -> list[tuple[int, int]]:
    visible_indices = [index for index, frame in enumerate(frames) if bool(frame["visible"])]
    if not visible_indices:
        return []
    segments: list[tuple[int, int]] = []
    start = visible_indices[0]
    previous = visible_indices[0]
    for index in visible_indices[1:]:
        missing_count = index - previous - 1
        if missing_count > max_gap_fill_frames:
            segments.append((start, previous))
            start = index
        previous = index
    segments.append((start, previous))
    return segments


def _measurement(frame: dict[str, Any], *, axis: int) -> float | None:
    return float(frame["xy"][axis]) if bool(frame["visible"]) else None


def _smooth_axis(
    observations: Sequence[float | None],
    *,
    dt: float,
    measurement_variance: float,
    process_variance: float,
) -> list[list[float]]:
    initial = _initial_state(observations, dt=dt)
    state = initial
    covariance = _diag([measurement_variance, 1000.0, 1000.0])
    transition = _transition(dt)
    process = _process_noise(process_variance)
    filtered_states: list[list[float]] = []
    filtered_covariances: list[list[list[float]]] = []
    predicted_states: list[list[float]] = []
    predicted_covariances: list[list[list[float]]] = []

    for observation in observations:
        pred_state = _mat_vec_mul(transition, state)
        pred_covariance = _mat_add(_mat_mul(_mat_mul(transition, covariance), _transpose(transition)), process)
        predicted_states.append(pred_state)
        predicted_covariances.append(pred_covariance)
        if observation is None:
            state = pred_state
            covariance = pred_covariance
        else:
            innovation = float(observation) - pred_state[0]
            innovation_variance = pred_covariance[0][0] + measurement_variance
            gain = [pred_covariance[row][0] / innovation_variance for row in range(3)]
            state = [pred_state[row] + gain[row] * innovation for row in range(3)]
            covariance = _mat_mul(_measurement_update_matrix(gain), pred_covariance)
        filtered_states.append(state)
        filtered_covariances.append(covariance)

    smoothed_states = [list(state) for state in filtered_states]
    smoothed_covariances = [[list(row) for row in covariance] for covariance in filtered_covariances]
    transition_t = _transpose(transition)
    for index in range(len(observations) - 2, -1, -1):
        inv_pred = _invert_3x3(predicted_covariances[index + 1])
        gain = _mat_mul(_mat_mul(filtered_covariances[index], transition_t), inv_pred)
        delta = _vec_sub(smoothed_states[index + 1], predicted_states[index + 1])
        smoothed_states[index] = _vec_add(filtered_states[index], _mat_vec_mul(gain, delta))
        cov_delta = _mat_sub(smoothed_covariances[index + 1], predicted_covariances[index + 1])
        smoothed_covariances[index] = _mat_add(
            filtered_covariances[index],
            _mat_mul(_mat_mul(gain, cov_delta), _transpose(gain)),
        )
    return smoothed_states


def _initial_state(observations: Sequence[float | None], *, dt: float) -> list[float]:
    measured = [(index, value) for index, value in enumerate(observations) if value is not None]
    if not measured:
        return [0.0, 0.0, 0.0]
    first_index, first_value = measured[0]
    if len(measured) < 2:
        return [float(first_value), 0.0, 0.0]
    second_index, second_value = measured[1]
    delta_t = max((second_index - first_index) * dt, dt)
    velocity = (float(second_value) - float(first_value)) / delta_t
    return [float(first_value), velocity, 0.0]


def _filled_confidence(frames: Sequence[dict[str, Any]], offset: int) -> float:
    left = next((float(frames[index]["conf"]) for index in range(offset - 1, -1, -1) if bool(frames[index]["visible"])), 0.0)
    right = next((float(frames[index]["conf"]) for index in range(offset + 1, len(frames)) if bool(frames[index]["visible"])), 0.0)
    support = min(value for value in (left, right) if value > 0.0) if left > 0.0 and right > 0.0 else max(left, right)
    return max(0.0, min(1.0, support * 0.5))


def _jitter_px_std(frames: Sequence[dict[str, Any]]) -> float:
    second_differences: list[float] = []
    visible_run: list[dict[str, Any]] = []
    previous_index: int | None = None
    for index, frame in enumerate(frames):
        if bool(frame["visible"]) and (previous_index is None or index == previous_index + 1):
            visible_run.append(frame)
            previous_index = index
            continue
        _extend_second_differences(second_differences, visible_run)
        visible_run = [frame] if bool(frame["visible"]) else []
        previous_index = index if bool(frame["visible"]) else None
    _extend_second_differences(second_differences, visible_run)
    if not second_differences:
        return 0.0
    mean = sum(second_differences) / float(len(second_differences))
    variance = sum((value - mean) ** 2 for value in second_differences) / float(len(second_differences))
    return math.sqrt(variance)


def _extend_second_differences(target: list[float], run: Sequence[dict[str, Any]]) -> None:
    if len(run) < 3:
        return
    for left, middle, right in zip(run, run[1:], run[2:], strict=False):
        ddx = float(right["xy"][0]) - 2.0 * float(middle["xy"][0]) + float(left["xy"][0])
        ddy = float(right["xy"][1]) - 2.0 * float(middle["xy"][1]) + float(left["xy"][1])
        target.append(math.hypot(ddx, ddy))


def _transition(dt: float) -> list[list[float]]:
    return [[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]]


def _process_noise(process_variance: float) -> list[list[float]]:
    return _diag([process_variance, process_variance, process_variance])


def _measurement_update_matrix(gain: Sequence[float]) -> list[list[float]]:
    return [
        [1.0 - gain[0], 0.0, 0.0],
        [-gain[1], 1.0, 0.0],
        [-gain[2], 0.0, 1.0],
    ]


def _diag(values: Sequence[float]) -> list[list[float]]:
    return [[float(value) if row == col else 0.0 for col, value in enumerate(values)] for row in range(len(values))]


def _mat_mul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    return [
        [
            sum(float(left[row][inner]) * float(right[inner][col]) for inner in range(len(right)))
            for col in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def _mat_vec_mul(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(float(row[col]) * float(vector[col]) for col in range(len(vector))) for row in matrix]


def _mat_add(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(left[row][col]) + float(right[row][col]) for col in range(len(left[0]))] for row in range(len(left))]


def _mat_sub(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(left[row][col]) - float(right[row][col]) for col in range(len(left[0]))] for row in range(len(left))]


def _vec_add(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [float(left[index]) + float(right[index]) for index in range(len(left))]


def _vec_sub(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [float(left[index]) - float(right[index]) for index in range(len(left))]


def _transpose(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(matrix[row][col]) for row in range(len(matrix))] for col in range(len(matrix[0]))]


def _invert_3x3(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    determinant = (
        a * (e * i - f * h)
        - b * (d * i - f * g)
        + c * (d * h - e * g)
    )
    if abs(determinant) < 1e-12:
        raise ValueError("singular covariance in RTS smoother")
    inv_det = 1.0 / determinant
    return [
        [(e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det],
        [(f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det],
        [(d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det],
    ]


__all__ = [
    "ARTIFACT_TYPE",
    "smooth_ball_track_kalman_rts",
    "write_kalman_rts_smoothed_ball_track",
]
