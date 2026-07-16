"""Generalizable temporal filtering for ball tracks.

This post-processes a candidate ``BallTrack`` using only the candidate's own
samples and a motion budget.  It must not read sparse human click labels.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ball_overlay import load_ball_track
from .schemas import BallTrack


@dataclass(frozen=True)
class _VisibleNode:
    position: int
    frame_index: int
    xy: tuple[float, float]
    conf: float


def filter_ball_track_temporal_path(
    *,
    ball_track_path: str | Path,
    max_speed_px_per_second: float = 7200.0,
    base_jump_px: float = 60.0,
    max_link_gap_frames: int = 10,
    max_interpolate_gap_frames: int = 3,
    min_chain_visible_frames: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drop off-path detections and bridge short gaps without human labels."""

    _validate_positive(max_speed_px_per_second, "max_speed_px_per_second")
    _validate_nonnegative(base_jump_px, "base_jump_px")
    if max_link_gap_frames < 1:
        raise ValueError("max_link_gap_frames must be >= 1")
    if max_interpolate_gap_frames < 0:
        raise ValueError("max_interpolate_gap_frames must be >= 0")
    if min_chain_visible_frames < 1:
        raise ValueError("min_chain_visible_frames must be >= 1")

    track = load_ball_track(ball_track_path)
    payload = deepcopy(track.model_dump(mode="json"))
    samples_by_index = _payload_samples_by_frame_index(payload, fps=float(track.fps))
    visible_nodes = _visible_nodes(samples_by_index)
    chain_indices = _longest_motion_chain(
        visible_nodes,
        fps=float(track.fps),
        max_speed_px_per_second=max_speed_px_per_second,
        base_jump_px=base_jump_px,
        max_link_gap_frames=max_link_gap_frames,
    )
    if len(chain_indices) < min_chain_visible_frames:
        chain_indices = set()

    visible_before = sum(1 for frame in payload["frames"] if bool(frame["visible"]))
    rejected_off_path = 0
    for frame_index, frame in samples_by_index.items():
        if not bool(frame["visible"]):
            continue
        if frame_index in chain_indices:
            continue
        _hide_frame(frame)
        frame["approx"] = False
        rejected_off_path += 1

    confidence_repairs = _interpolate_chain_gaps(
        samples_by_index=samples_by_index,
        chain_indices=chain_indices,
        fps=float(track.fps),
        max_speed_px_per_second=max_speed_px_per_second,
        base_jump_px=base_jump_px,
        max_interpolate_gap_frames=max_interpolate_gap_frames,
    )
    BallTrack.model_validate(payload)
    visible_after = sum(1 for frame in payload["frames"] if bool(frame["visible"]))
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_temporal_filter",
        "status": "filtered_not_gate_verified",
        "source_ball_track": str(ball_track_path),
        "frame_count": len(payload["frames"]),
        "visible_before": visible_before,
        "visible_after": visible_after,
        "chain_visible_count": len(chain_indices),
        "rejected_off_path_count": rejected_off_path,
        "interpolated_count": len(confidence_repairs),
        "confidence_repairs": confidence_repairs,
        "max_speed_px_per_second": float(max_speed_px_per_second),
        "base_jump_px": float(base_jump_px),
        "max_link_gap_frames": int(max_link_gap_frames),
        "max_interpolate_gap_frames": int(max_interpolate_gap_frames),
        "min_chain_visible_frames": int(min_chain_visible_frames),
        "uses_human_clicks": False,
        "not_ground_truth": True,
    }
    return payload, summary


def filter_ball_track_temporal_outliers(
    *,
    ball_track_path: str | Path,
    max_speed_px_per_second: float = 7200.0,
    base_jump_px: float = 60.0,
    max_neighbor_gap_frames: int = 4,
    max_iterations: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove isolated impossible jumps while preserving most detections."""

    _validate_positive(max_speed_px_per_second, "max_speed_px_per_second")
    _validate_nonnegative(base_jump_px, "base_jump_px")
    if max_neighbor_gap_frames < 1:
        raise ValueError("max_neighbor_gap_frames must be >= 1")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    track = load_ball_track(ball_track_path)
    payload = deepcopy(track.model_dump(mode="json"))
    samples_by_index = _payload_samples_by_frame_index(payload, fps=float(track.fps))
    visible_before = sum(1 for frame in payload["frames"] if bool(frame["visible"]))
    rejected = 0
    iterations = 0
    for iteration in range(max_iterations):
        to_hide = _isolated_outlier_indices(
            samples_by_index=samples_by_index,
            fps=float(track.fps),
            max_speed_px_per_second=max_speed_px_per_second,
            base_jump_px=base_jump_px,
            max_neighbor_gap_frames=max_neighbor_gap_frames,
        )
        if not to_hide:
            break
        for frame_index in to_hide:
            frame = samples_by_index[frame_index]
            if bool(frame["visible"]):
                _hide_frame(frame)
                frame["approx"] = False
                rejected += 1
        iterations = iteration + 1

    BallTrack.model_validate(payload)
    visible_after = sum(1 for frame in payload["frames"] if bool(frame["visible"]))
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_temporal_outlier_filter",
        "status": "filtered_not_gate_verified",
        "source_ball_track": str(ball_track_path),
        "frame_count": len(payload["frames"]),
        "visible_before": visible_before,
        "visible_after": visible_after,
        "rejected_isolated_outlier_count": rejected,
        "iterations": iterations,
        "max_speed_px_per_second": float(max_speed_px_per_second),
        "base_jump_px": float(base_jump_px),
        "max_neighbor_gap_frames": int(max_neighbor_gap_frames),
        "uses_human_clicks": False,
        "not_ground_truth": True,
    }
    return payload, summary


def filter_ball_track_local_trajectory_outliers(
    *,
    ball_track_path: str | Path,
    window_frames: int = 20,
    max_error_px: float = 80.0,
    min_pair_predictions: int = 4,
    max_iterations: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject points far from the local trajectory implied by surrounding points."""

    if window_frames < 1:
        raise ValueError("window_frames must be >= 1")
    _validate_positive(max_error_px, "max_error_px")
    if min_pair_predictions < 1:
        raise ValueError("min_pair_predictions must be >= 1")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    track = load_ball_track(ball_track_path)
    payload = deepcopy(track.model_dump(mode="json"))
    samples_by_index = _payload_samples_by_frame_index(payload, fps=float(track.fps))
    visible_before = sum(1 for frame in payload["frames"] if bool(frame["visible"]))
    rejected_count = 0
    evaluated_count = 0
    iterations = 0

    for iteration in range(max_iterations):
        to_hide: set[int] = set()
        visible = sorted((index, frame) for index, frame in samples_by_index.items() if bool(frame["visible"]))
        for frame_index, frame in visible:
            prediction = _local_pairwise_prediction(
                visible=visible,
                frame_index=frame_index,
                window_frames=window_frames,
                min_pair_predictions=min_pair_predictions,
            )
            if prediction is None:
                continue
            evaluated_count += 1
            if _distance(_xy(frame), prediction) > max_error_px:
                to_hide.add(frame_index)
        if not to_hide:
            break
        for frame_index in to_hide:
            frame = samples_by_index[frame_index]
            if bool(frame["visible"]):
                _hide_frame(frame)
                frame["approx"] = False
                rejected_count += 1
        iterations = iteration + 1

    BallTrack.model_validate(payload)
    visible_after = sum(1 for frame in payload["frames"] if bool(frame["visible"]))
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_local_trajectory_filter",
        "status": "filtered_not_gate_verified",
        "source_ball_track": str(ball_track_path),
        "frame_count": len(payload["frames"]),
        "visible_before": visible_before,
        "visible_after": visible_after,
        "evaluated_local_trajectory_count": evaluated_count,
        "rejected_local_trajectory_outlier_count": rejected_count,
        "iterations": iterations,
        "window_frames": int(window_frames),
        "max_error_px": float(max_error_px),
        "min_pair_predictions": int(min_pair_predictions),
        "uses_human_clicks": False,
        "not_ground_truth": True,
    }
    return payload, summary


def filter_ball_track_ballistic_outliers(
    *,
    ball_track_path: str | Path,
    window_frames: int = 24,
    max_residual_px: float = 60.0,
    min_fit_points: int = 5,
    max_iterations: int = 2,
    require_bracket: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject detections that do not fit a local constant-acceleration image arc."""

    if window_frames < 1:
        raise ValueError("window_frames must be >= 1")
    _validate_positive(max_residual_px, "max_residual_px")
    if min_fit_points < 3:
        raise ValueError("min_fit_points must be >= 3")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    track = load_ball_track(ball_track_path)
    payload = deepcopy(track.model_dump(mode="json"))
    samples_by_index = _payload_samples_by_frame_index(payload, fps=float(track.fps))
    visible_before = sum(1 for frame in payload["frames"] if bool(frame["visible"]))
    rejected_count = 0
    evaluated_count = 0
    iterations = 0

    for iteration in range(max_iterations):
        to_hide: set[int] = set()
        visible = sorted((index, frame) for index, frame in samples_by_index.items() if bool(frame["visible"]))
        for frame_index, frame in visible:
            prediction = _local_ballistic_prediction(
                visible=visible,
                frame_index=frame_index,
                window_frames=window_frames,
                max_residual_px=max_residual_px,
                min_fit_points=min_fit_points,
                require_bracket=require_bracket,
            )
            if prediction is None:
                continue
            evaluated_count += 1
            if _distance(_xy(frame), prediction) > max_residual_px:
                to_hide.add(frame_index)
        if not to_hide:
            break
        for frame_index in to_hide:
            frame = samples_by_index[frame_index]
            if bool(frame["visible"]):
                _hide_frame(frame)
                frame["approx"] = False
                rejected_count += 1
        iterations = iteration + 1

    BallTrack.model_validate(payload)
    visible_after = sum(1 for frame in payload["frames"] if bool(frame["visible"]))
    summary = {
        "schema_version": 1,
        "artifact_type": "racketsport_ball_ballistic_filter",
        "status": "filtered_not_gate_verified",
        "source_ball_track": str(ball_track_path),
        "frame_count": len(payload["frames"]),
        "visible_before": visible_before,
        "visible_after": visible_after,
        "evaluated_ballistic_count": evaluated_count,
        "rejected_ballistic_outlier_count": rejected_count,
        "iterations": iterations,
        "window_frames": int(window_frames),
        "max_residual_px": float(max_residual_px),
        "min_fit_points": int(min_fit_points),
        "require_bracket": bool(require_bracket),
        "uses_human_clicks": False,
        "not_ground_truth": True,
    }
    return payload, summary


def write_temporal_filtered_ball_track(
    *,
    ball_track_path: str | Path,
    out_path: str | Path,
    summary_path: str | Path,
    mode: str = "path",
    max_speed_px_per_second: float = 7200.0,
    base_jump_px: float = 60.0,
    max_link_gap_frames: int = 10,
    max_interpolate_gap_frames: int = 3,
    min_chain_visible_frames: int = 3,
    max_neighbor_gap_frames: int = 4,
    max_iterations: int = 3,
    local_trajectory_window_frames: int = 20,
    local_trajectory_max_error_px: float = 80.0,
    local_trajectory_min_pair_predictions: int = 4,
    ballistic_window_frames: int = 24,
    ballistic_max_residual_px: float = 60.0,
    ballistic_min_fit_points: int = 5,
    ballistic_require_bracket: bool = True,
) -> dict[str, Any]:
    if mode == "path":
        payload, summary = filter_ball_track_temporal_path(
            ball_track_path=ball_track_path,
            max_speed_px_per_second=max_speed_px_per_second,
            base_jump_px=base_jump_px,
            max_link_gap_frames=max_link_gap_frames,
            max_interpolate_gap_frames=max_interpolate_gap_frames,
            min_chain_visible_frames=min_chain_visible_frames,
        )
    elif mode == "outlier":
        payload, summary = filter_ball_track_temporal_outliers(
            ball_track_path=ball_track_path,
            max_speed_px_per_second=max_speed_px_per_second,
            base_jump_px=base_jump_px,
            max_neighbor_gap_frames=max_neighbor_gap_frames,
            max_iterations=max_iterations,
        )
    elif mode == "local_trajectory":
        payload, summary = filter_ball_track_local_trajectory_outliers(
            ball_track_path=ball_track_path,
            window_frames=local_trajectory_window_frames,
            max_error_px=local_trajectory_max_error_px,
            min_pair_predictions=local_trajectory_min_pair_predictions,
            max_iterations=max_iterations,
        )
    elif mode == "ballistic":
        payload, summary = filter_ball_track_ballistic_outliers(
            ball_track_path=ball_track_path,
            window_frames=ballistic_window_frames,
            max_residual_px=ballistic_max_residual_px,
            min_fit_points=ballistic_min_fit_points,
            max_iterations=max_iterations,
            require_bracket=ballistic_require_bracket,
        )
    else:
        raise ValueError("mode must be 'path', 'outlier', 'local_trajectory', or 'ballistic'")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_out = Path(summary_path)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _payload_samples_by_frame_index(payload: dict[str, Any], *, fps: float) -> dict[int, dict[str, Any]]:
    return {int(round(float(frame["t"]) * fps)): frame for frame in payload["frames"]}


def _visible_nodes(samples_by_index: dict[int, dict[str, Any]]) -> list[_VisibleNode]:
    nodes: list[_VisibleNode] = []
    for position, (frame_index, frame) in enumerate(sorted(samples_by_index.items())):
        if not bool(frame["visible"]):
            continue
        nodes.append(
            _VisibleNode(
                position=position,
                frame_index=frame_index,
                xy=(float(frame["xy"][0]), float(frame["xy"][1])),
                conf=float(frame["conf"]),
            )
        )
    return nodes


def _longest_motion_chain(
    nodes: list[_VisibleNode],
    *,
    fps: float,
    max_speed_px_per_second: float,
    base_jump_px: float,
    max_link_gap_frames: int,
) -> set[int]:
    if not nodes:
        return set()
    scores = [1.0 + nodes[index].conf * 0.01 for index in range(len(nodes))]
    prev: list[int | None] = [None for _node in nodes]
    for current_index, current in enumerate(nodes):
        for candidate_index in range(current_index - 1, -1, -1):
            candidate = nodes[candidate_index]
            gap = current.frame_index - candidate.frame_index
            if gap <= 0:
                continue
            if gap > max_link_gap_frames:
                break
            if not _motion_link_allowed(
                candidate.xy,
                current.xy,
                gap_frames=gap,
                fps=fps,
                max_speed_px_per_second=max_speed_px_per_second,
                base_jump_px=base_jump_px,
            ):
                continue
            continuity_bonus = max(0.0, 0.02 - (_distance(candidate.xy, current.xy) * 0.00001))
            candidate_score = scores[candidate_index] + 1.0 + current.conf * 0.01 + continuity_bonus
            if candidate_score > scores[current_index]:
                scores[current_index] = candidate_score
                prev[current_index] = candidate_index

    best_index = max(range(len(scores)), key=lambda index: scores[index])
    chain: set[int] = set()
    cursor: int | None = best_index
    while cursor is not None:
        chain.add(nodes[cursor].frame_index)
        cursor = prev[cursor]
    return chain


def _interpolate_chain_gaps(
    *,
    samples_by_index: dict[int, dict[str, Any]],
    chain_indices: set[int],
    fps: float,
    max_speed_px_per_second: float,
    base_jump_px: float,
    max_interpolate_gap_frames: int,
) -> list[dict[str, Any]]:
    if max_interpolate_gap_frames == 0 or len(chain_indices) < 2:
        return []
    confidence_repairs: list[dict[str, Any]] = []
    ordered = sorted(chain_indices)
    for left_index, right_index in zip(ordered, ordered[1:]):
        gap = right_index - left_index
        if gap <= 1 or gap - 1 > max_interpolate_gap_frames:
            continue
        left = samples_by_index[left_index]
        right = samples_by_index[right_index]
        if not _motion_link_allowed(
            _xy(left),
            _xy(right),
            gap_frames=gap,
            fps=fps,
            max_speed_px_per_second=max_speed_px_per_second,
            base_jump_px=base_jump_px,
        ):
            continue
        for frame_index in range(left_index + 1, right_index):
            frame = samples_by_index.get(frame_index)
            if frame is None:
                continue
            alpha = (frame_index - left_index) / float(gap)
            frame["xy"] = [
                float(left["xy"][0]) + (float(right["xy"][0]) - float(left["xy"][0])) * alpha,
                float(left["xy"][1]) + (float(right["xy"][1]) - float(left["xy"][1])) * alpha,
            ]
            frame["conf"] = min(float(left["conf"]), float(right["conf"])) * 0.5
            frame["visible"] = True
            frame["approx"] = True
            frame.pop("world_xyz", None)
            confidence_repairs.append(
                {
                    "frame_index": frame_index,
                    "t": float(frame["t"]),
                    "conf": float(frame["conf"]),
                    "conf_source": "interpolated_endpoint_min_half",
                    "repaired": True,
                }
            )
    return confidence_repairs


def _isolated_outlier_indices(
    *,
    samples_by_index: dict[int, dict[str, Any]],
    fps: float,
    max_speed_px_per_second: float,
    base_jump_px: float,
    max_neighbor_gap_frames: int,
) -> set[int]:
    visible_indices = sorted(index for index, frame in samples_by_index.items() if bool(frame["visible"]))
    to_hide: set[int] = set()
    for position, frame_index in enumerate(visible_indices):
        prev_index = _neighbor_index(visible_indices, position, direction=-1, max_gap=max_neighbor_gap_frames)
        next_index = _neighbor_index(visible_indices, position, direction=1, max_gap=max_neighbor_gap_frames)
        if prev_index is None or next_index is None:
            continue
        prev_frame = samples_by_index[prev_index]
        frame = samples_by_index[frame_index]
        next_frame = samples_by_index[next_index]
        prev_gap = frame_index - prev_index
        next_gap = next_index - frame_index
        bridge_gap = next_index - prev_index
        prev_to_current = _motion_link_allowed(
            _xy(prev_frame),
            _xy(frame),
            gap_frames=prev_gap,
            fps=fps,
            max_speed_px_per_second=max_speed_px_per_second,
            base_jump_px=base_jump_px,
        )
        current_to_next = _motion_link_allowed(
            _xy(frame),
            _xy(next_frame),
            gap_frames=next_gap,
            fps=fps,
            max_speed_px_per_second=max_speed_px_per_second,
            base_jump_px=base_jump_px,
        )
        prev_to_next = _motion_link_allowed(
            _xy(prev_frame),
            _xy(next_frame),
            gap_frames=bridge_gap,
            fps=fps,
            max_speed_px_per_second=max_speed_px_per_second,
            base_jump_px=base_jump_px,
        )
        if not prev_to_current and not current_to_next and prev_to_next:
            to_hide.add(frame_index)
    return to_hide


def _local_pairwise_prediction(
    *,
    visible: list[tuple[int, dict[str, Any]]],
    frame_index: int,
    window_frames: int,
    min_pair_predictions: int,
) -> tuple[float, float] | None:
    before = [
        (index, frame)
        for index, frame in visible
        if 0 < frame_index - index <= window_frames
    ]
    after = [
        (index, frame)
        for index, frame in visible
        if 0 < index - frame_index <= window_frames
    ]
    predictions: list[tuple[float, float]] = []
    for before_index, before_frame in before:
        before_xy = _xy(before_frame)
        for after_index, after_frame in after:
            span = after_index - before_index
            if span <= 0:
                continue
            alpha = (frame_index - before_index) / float(span)
            after_xy = _xy(after_frame)
            predictions.append(
                (
                    before_xy[0] + (after_xy[0] - before_xy[0]) * alpha,
                    before_xy[1] + (after_xy[1] - before_xy[1]) * alpha,
                )
            )
    if len(predictions) < min_pair_predictions:
        return None
    return (
        _median([prediction[0] for prediction in predictions]),
        _median([prediction[1] for prediction in predictions]),
    )


def _local_ballistic_prediction(
    *,
    visible: list[tuple[int, dict[str, Any]]],
    frame_index: int,
    window_frames: int,
    max_residual_px: float,
    min_fit_points: int,
    require_bracket: bool,
) -> tuple[float, float] | None:
    neighbors = [
        (index, frame)
        for index, frame in visible
        if index != frame_index and 0 < abs(index - frame_index) <= window_frames
    ]
    if require_bracket:
        has_before = any(index < frame_index for index, _frame in neighbors)
        has_after = any(index > frame_index for index, _frame in neighbors)
        if not has_before or not has_after:
            return None
    if len(neighbors) < min_fit_points:
        return None

    active = list(neighbors)
    while len(active) >= min_fit_points:
        fit = _fit_quadratic_xy(active, origin_frame_index=frame_index)
        residuals = [
            (
                position,
                _distance(
                    _xy(frame),
                    _eval_quadratic_xy(fit, index - frame_index),
                ),
            )
            for position, (index, frame) in enumerate(active)
        ]
        worst_position, worst_residual = max(residuals, key=lambda item: item[1])
        if worst_residual <= max_residual_px or len(active) == min_fit_points:
            return _eval_quadratic_xy(fit, 0)
        active.pop(worst_position)
    return None


def _fit_quadratic_xy(
    samples: list[tuple[int, dict[str, Any]]],
    *,
    origin_frame_index: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    rows = [
        [1.0, float(index - origin_frame_index), float(index - origin_frame_index) ** 2]
        for index, _frame in samples
    ]
    x_coeffs = _least_squares(rows, [_xy(frame)[0] for _index, frame in samples])
    y_coeffs = _least_squares(rows, [_xy(frame)[1] for _index, frame in samples])
    return x_coeffs, y_coeffs


def _least_squares(rows: list[list[float]], values: list[float]) -> tuple[float, float, float]:
    if not rows:
        raise ValueError("least squares requires at least one row")
    width = len(rows[0])
    normal = [[0.0 for _ in range(width)] for _index in range(width)]
    rhs = [0.0 for _index in range(width)]
    for row, value in zip(rows, values):
        if len(row) != width:
            raise ValueError("least squares rows must have consistent width")
        for i in range(width):
            rhs[i] += row[i] * value
            for j in range(width):
                normal[i][j] += row[i] * row[j]
    solution = _solve_linear_system(normal, rhs)
    return float(solution[0]), float(solution[1]), float(solution[2])


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda row: abs(matrix[row][col]))
        if abs(matrix[pivot_row][col]) < 1e-12:
            raise ValueError("cannot fit ballistic trajectory to singular sample times")
        if pivot_row != col:
            matrix[col], matrix[pivot_row] = matrix[pivot_row], matrix[col]
            rhs[col], rhs[pivot_row] = rhs[pivot_row], rhs[col]

        pivot = matrix[col][col]
        for j in range(col, n):
            matrix[col][j] /= pivot
        rhs[col] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = matrix[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n):
                matrix[row][j] -= factor * matrix[col][j]
            rhs[row] -= factor * rhs[col]
    return rhs


def _eval_quadratic_xy(
    fit: tuple[tuple[float, float, float], tuple[float, float, float]],
    frame_offset: int,
) -> tuple[float, float]:
    dt = float(frame_offset)
    x_coeffs, y_coeffs = fit
    return (
        x_coeffs[0] + x_coeffs[1] * dt + x_coeffs[2] * dt * dt,
        y_coeffs[0] + y_coeffs[1] * dt + y_coeffs[2] * dt * dt,
    )


def _neighbor_index(visible_indices: list[int], position: int, *, direction: int, max_gap: int) -> int | None:
    neighbor_position = position + direction
    if neighbor_position < 0 or neighbor_position >= len(visible_indices):
        return None
    current = visible_indices[position]
    neighbor = visible_indices[neighbor_position]
    if abs(neighbor - current) > max_gap:
        return None
    return neighbor


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
    allowed = base_jump_px + (float(max_speed_px_per_second) * float(gap_frames) / float(fps))
    return _distance(left_xy, right_xy) <= allowed


def _hide_frame(frame: dict[str, Any]) -> None:
    frame["visible"] = False
    frame["conf"] = 0.0


def _xy(frame: dict[str, Any]) -> tuple[float, float]:
    return float(frame["xy"][0]), float(frame["xy"][1])


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _validate_positive(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be > 0")


def _validate_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be >= 0")


__all__ = [
    "filter_ball_track_ballistic_outliers",
    "filter_ball_track_local_trajectory_outliers",
    "filter_ball_track_temporal_outliers",
    "filter_ball_track_temporal_path",
    "write_temporal_filtered_ball_track",
]
