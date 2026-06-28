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

    interpolated_count = _interpolate_chain_gaps(
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
        "interpolated_count": interpolated_count,
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
    else:
        raise ValueError("mode must be 'path' or 'outlier'")
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
) -> int:
    if max_interpolate_gap_frames == 0 or len(chain_indices) < 2:
        return 0
    interpolated_count = 0
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
            interpolated_count += 1
    return interpolated_count


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


def _validate_positive(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be > 0")


def _validate_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be >= 0")


__all__ = [
    "filter_ball_track_temporal_outliers",
    "filter_ball_track_temporal_path",
    "write_temporal_filtered_ball_track",
]
