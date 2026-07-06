"""Offline visual-smoothness metrics for completed replay bundle directories."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from threed.racketsport.external_gt_body_prediction_schema import MHR70_JOINT_NAMES


VERSION = 1
FOOT_JOINTS: dict[str, tuple[str, ...]] = {
    "left": ("left_ankle", "left_big_toe", "left_small_toe", "left_heel"),
    "right": ("right_ankle", "right_big_toe", "right_small_toe", "right_heel"),
}
WRIST_JOINTS = ("left_wrist", "right_wrist")


def estimate_integer_lag_frames(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    max_lag_frames: int = 1,
) -> int:
    """Return the integer lag with the strongest normalized cross-correlation."""

    if max_lag_frames < 0:
        raise ValueError("max_lag_frames must be non-negative")
    ref = [float(value) for value in reference if math.isfinite(float(value))]
    cand = [float(value) for value in candidate if math.isfinite(float(value))]
    if len(ref) != len(cand):
        raise ValueError("reference and candidate must have the same finite length")
    if len(ref) < 3:
        return 0
    best_lag = 0
    best_score = -math.inf
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        if lag < 0:
            left = ref[-lag:]
            right = cand[: len(cand) + lag]
        elif lag > 0:
            left = ref[: len(ref) - lag]
            right = cand[lag:]
        else:
            left = ref
            right = cand
        score = _normalized_correlation(left, right)
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and abs(lag) < abs(best_lag)):
            best_score = score
            best_lag = lag
    return best_lag


def measure_visual_quality(run_dir: str | Path) -> dict[str, Any]:
    """Measure visual-smoothness metrics from existing completed clip artifacts."""

    root = Path(run_dir)
    skeleton = _read_json(root / "skeleton3d.json")
    placement = _read_json(root / "placement.json")
    virtual_world = _read_json(root / "virtual_world.json")
    body_quality = _read_json(root / "body_joint_quality.json")

    fps = float(
        virtual_world.get("fps")
        or skeleton.get("fps")
        or placement.get("fps")
        or 30.0
    )
    joint_names = _joint_names(virtual_world) or _joint_names(skeleton)
    placement_by_player = _placement_frames_by_player(placement)
    skeleton_by_player = _frames_by_player(skeleton)

    players: dict[str, Any] = {}
    for player in virtual_world.get("players", []) or []:
        if not isinstance(player, Mapping) or "id" not in player:
            continue
        player_id = str(player["id"])
        world_frames = _sorted_frames(player.get("frames", []))
        if not world_frames:
            world_frames = skeleton_by_player.get(player_id, [])
        placement_frames = placement_by_player.get(player_id, {})
        players[player_id] = _measure_player(
            player_id=player_id,
            frames=world_frames,
            placement_frames=placement_frames,
            joint_names=joint_names,
            fps=fps,
        )

    body_summary = body_quality.get("summary", {}) if isinstance(body_quality.get("summary"), Mapping) else {}
    placement_summary = placement.get("summary", {}) if isinstance(placement.get("summary"), Mapping) else {}
    return {
        "schema_version": VERSION,
        "artifact_type": "racketsport_visual_quality",
        "source_run_dir": str(root),
        "fps": fps,
        "inputs": {
            "skeleton3d": "skeleton3d.json",
            "placement": "placement.json",
            "virtual_world": "virtual_world.json",
            "body_joint_quality": "body_joint_quality.json",
        },
        "units": {
            "foot_slide": "mm/frame",
            "world_jitter": "mm/frame^2",
            "root_step": "m/frame",
        },
        "summary": {
            "player_count": len(players),
            "temporal_smoothing_reset_count": int(body_summary.get("temporal_smoothing_reset_count", 0) or 0),
            "root_motion_temporal_jump_count": int(body_summary.get("root_motion_temporal_jump_count", 0) or 0),
            "court_bounds_violations": int(placement_summary.get("court_bounds_violations", 0) or 0),
            "side_quadrant_consistency": placement_summary.get("side_quadrant_consistency", {}),
        },
        "players": players,
    }


def write_visual_quality(run_dir: str | Path, *, out_dir: str | Path | None = None) -> tuple[Path, Path, dict[str, Any]]:
    metrics = measure_visual_quality(run_dir)
    output_dir = Path(out_dir) if out_dir is not None else Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "visual_quality.json"
    md_path = output_dir / "visual_quality.md"
    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_summary(metrics), encoding="utf-8")
    return json_path, md_path, metrics


def _measure_player(
    *,
    player_id: str,
    frames: Sequence[Mapping[str, Any]],
    placement_frames: Mapping[int, Mapping[str, Any]],
    joint_names: Sequence[str],
    fps: float,
) -> dict[str, Any]:
    del fps
    root_points = [_root_point(frame, joint_names=joint_names) for frame in frames]
    frame_indices = [_frame_idx(frame, fallback=idx) for idx, frame in enumerate(frames)]
    stance_by_frame = {
        int(frame_idx): bool(placement_frames.get(int(frame_idx), {}).get("stance", False))
        for frame_idx in frame_indices
    }
    foot_points = {
        foot: [_foot_point(frame, joint_names=joint_names, foot=foot) for frame in frames]
        for foot in FOOT_JOINTS
    }
    foot_slide_all: list[float] = []
    foot_slide_stance: list[float] = []
    foot_slide_nonstance: list[float] = []
    for pos, (left_idx, right_idx) in enumerate(zip(frame_indices, frame_indices[1:], strict=False)):
        pair_is_stance = stance_by_frame.get(int(left_idx), False) and stance_by_frame.get(int(right_idx), False)
        for foot in FOOT_JOINTS:
            left = foot_points[foot][pos]
            right = foot_points[foot][pos + 1]
            if left is None or right is None:
                continue
            distance_mm = _distance_xy(left, right) * 1000.0
            foot_slide_all.append(distance_mm)
            if pair_is_stance:
                foot_slide_stance.append(distance_mm)
            else:
                foot_slide_nonstance.append(distance_mm)

    root_steps = _steps_per_frame(root_points, frame_indices)
    placement_root_steps = _placement_root_steps(placement_frames)
    return {
        "frame_count": len(frames),
        "foot_slide_mm_per_frame": {
            "all": _distribution(foot_slide_all),
            "stance": _distribution(foot_slide_stance),
            "non_stance": _distribution(foot_slide_nonstance),
        },
        "world_jitter_mm_per_frame2": _world_jitter(frames, joint_names=joint_names, root_points=root_points),
        "root_step_m": _distribution(root_steps),
        "placement_root_step_m": _distribution(placement_root_steps),
        "smoothing_reset_count": _count_smoothing_resets(frames),
        "longest_unanchored_foot_run_frames": _longest_false_run([stance_by_frame.get(idx, False) for idx in frame_indices]),
    }


def _world_jitter(
    frames: Sequence[Mapping[str, Any]],
    *,
    joint_names: Sequence[str],
    root_points: Sequence[tuple[float, float, float] | None],
) -> dict[str, Any]:
    index_by_name = _joint_index_by_name(joint_names)
    foot_indices = [
        index_by_name[name]
        for names in FOOT_JOINTS.values()
        for name in names
        if name in index_by_name
    ]
    wrist_indices = [index_by_name[name] for name in WRIST_JOINTS if name in index_by_name]
    joints_by_frame = [_joints(frame) for frame in frames]
    return {
        "root": _acceleration_distribution(root_points),
        "feet": _joint_group_acceleration(joints_by_frame, foot_indices, joint_names),
        "wrists": _joint_group_acceleration(joints_by_frame, wrist_indices, joint_names),
    }


def _joint_group_acceleration(
    joints_by_frame: Sequence[Sequence[Sequence[float]]],
    indices: Sequence[int],
    joint_names: Sequence[str],
) -> dict[str, Any]:
    all_values: list[float] = []
    by_joint: dict[str, Any] = {}
    for idx in indices:
        points = [
            _point3(joints[idx]) if idx < len(joints) else None
            for joints in joints_by_frame
        ]
        values = _accelerations_mm(points)
        all_values.extend(values)
        name = str(joint_names[idx]) if idx < len(joint_names) else f"joint_{idx}"
        by_joint[name] = _distribution(values)
    out = _distribution(all_values)
    out["joints"] = by_joint
    return out


def _acceleration_distribution(points: Sequence[tuple[float, float, float] | None]) -> dict[str, float | int]:
    return _distribution(_accelerations_mm(points))


def _accelerations_mm(points: Sequence[tuple[float, float, float] | None]) -> list[float]:
    values: list[float] = []
    for left, mid, right in zip(points, points[1:], points[2:], strict=False):
        if left is None or mid is None or right is None:
            continue
        accel = (
            right[0] - 2.0 * mid[0] + left[0],
            right[1] - 2.0 * mid[1] + left[1],
            right[2] - 2.0 * mid[2] + left[2],
        )
        values.append(math.sqrt(sum(axis * axis for axis in accel)) * 1000.0)
    return values


def _count_smoothing_resets(frames: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for frame in frames:
        flags = frame.get("smoothing_flag")
        if isinstance(flags, Sequence) and not isinstance(flags, (str, bytes)):
            count += sum(1 for value in flags if "temporal_smoothing_reset" in str(value))
    return count


def _placement_frames_by_player(payload: Mapping[str, Any]) -> dict[str, dict[int, Mapping[str, Any]]]:
    out: dict[str, dict[int, Mapping[str, Any]]] = {}
    for player in payload.get("players", []) or []:
        if not isinstance(player, Mapping) or "id" not in player:
            continue
        frames = {
            _frame_idx(frame, fallback=idx): frame
            for idx, frame in enumerate(_sorted_frames(player.get("frames", [])))
        }
        out[str(player["id"])] = frames
    return out


def _placement_root_steps(placement_frames: Mapping[int, Mapping[str, Any]]) -> list[float]:
    points: list[tuple[float, float] | None] = []
    frame_indices: list[int] = []
    for frame_idx in sorted(placement_frames):
        point = _point2(placement_frames[frame_idx].get("smoothed_world_xy"))
        points.append(point)
        frame_indices.append(int(frame_idx))
    return _steps_per_frame(points, frame_indices)


def _steps_per_frame(
    points: Sequence[Sequence[float] | None],
    frame_indices: Sequence[int],
) -> list[float]:
    values: list[float] = []
    for left, right, left_idx, right_idx in zip(points, points[1:], frame_indices, frame_indices[1:], strict=False):
        if left is None or right is None:
            continue
        frame_delta = max(int(right_idx) - int(left_idx), 1)
        values.append(_distance_xy(left, right) / float(frame_delta))
    return values


def _frames_by_player(payload: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    for player in payload.get("players", []) or []:
        if isinstance(player, Mapping) and "id" in player:
            out[str(player["id"])] = _sorted_frames(player.get("frames", []))
    return out


def _sorted_frames(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    frames = [frame for frame in value if isinstance(frame, Mapping)]
    return sorted(frames, key=lambda frame: _frame_idx(frame, fallback=0))


def _root_point(frame: Mapping[str, Any], *, joint_names: Sequence[str]) -> tuple[float, float, float] | None:
    for key in ("transl_world", "track_world_xy", "floor_world_xyz"):
        point = frame.get(key)
        parsed = _point3(point)
        if parsed is not None:
            return parsed
        parsed2 = _point2(point)
        if parsed2 is not None:
            return (parsed2[0], parsed2[1], 0.0)
    joints = _joints(frame)
    index_by_name = _joint_index_by_name(joint_names)
    hips = [index_by_name[name] for name in ("left_hip", "right_hip") if name in index_by_name]
    points = [_point3(joints[idx]) for idx in hips if idx < len(joints)]
    points = [point for point in points if point is not None]
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
        sum(point[2] for point in points) / len(points),
    )


def _foot_point(frame: Mapping[str, Any], *, joint_names: Sequence[str], foot: str) -> tuple[float, float, float] | None:
    joints = _joints(frame)
    index_by_name = _joint_index_by_name(joint_names)
    points = [
        _point3(joints[index_by_name[name]])
        for name in FOOT_JOINTS[foot]
        if name in index_by_name and index_by_name[name] < len(joints)
    ]
    points = [point for point in points if point is not None]
    if not points:
        return None
    min_z = min(point[2] for point in points)
    low = [point for point in points if point[2] <= min_z + 0.025]
    return (
        sum(point[0] for point in low) / len(low),
        sum(point[1] for point in low) / len(low),
        sum(point[2] for point in low) / len(low),
    )


def _joint_index_by_name(joint_names: Sequence[str]) -> dict[str, int]:
    names = list(joint_names)
    direct = {str(name): idx for idx, name in enumerate(names)}
    if len(names) == len(MHR70_JOINT_NAMES):
        for idx, name in enumerate(MHR70_JOINT_NAMES):
            direct.setdefault(str(name), idx)
    return direct


def _joint_names(payload: Mapping[str, Any]) -> list[str]:
    value = payload.get("joint_names")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _joints(frame: Mapping[str, Any]) -> list[Sequence[float]]:
    value = frame.get("joints_world")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [joint for joint in value if isinstance(joint, Sequence) and not isinstance(joint, (str, bytes))]


def _frame_idx(frame: Mapping[str, Any], *, fallback: int) -> int:
    if "frame_idx" in frame:
        return int(frame["frame_idx"])
    return fallback


def _point3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    return (float(value[0]), float(value[1]), float(value[2]))


def _point2(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    return (float(value[0]), float(value[1]))


def _distance_xy(left: Sequence[float], right: Sequence[float]) -> float:
    return float(math.hypot(float(right[0]) - float(left[0]), float(right[1]) - float(left[1])))


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0, "rms": 0.0}
    return {
        "count": len(clean),
        "p50": _percentile(clean, 50.0),
        "p95": _percentile(clean, 95.0),
        "max": max(clean),
        "rms": math.sqrt(sum(value * value for value in clean) / len(clean)),
    }


def _normalized_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return -math.inf
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denom = math.sqrt(sum(value * value for value in left_centered) * sum(value * value for value in right_centered))
    if denom <= 0.0:
        return 1.0 if all(abs(left[idx] - right[idx]) <= 1e-12 for idx in range(len(left))) else 0.0
    return sum(left_centered[idx] * right_centered[idx] for idx in range(len(left))) / denom


def _percentile(values: Sequence[float], pct: float) -> float:
    clean = sorted(float(value) for value in values)
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * pct / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return clean[int(rank)]
    weight = rank - low
    return clean[low] * (1.0 - weight) + clean[high] * weight


def _longest_false_run(values: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _markdown_summary(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# Visual Quality Summary",
        "",
        f"- source_run_dir: `{metrics.get('source_run_dir')}`",
        f"- temporal_smoothing_reset_count: {metrics.get('summary', {}).get('temporal_smoothing_reset_count', 0)}",
        f"- root_motion_temporal_jump_count: {metrics.get('summary', {}).get('root_motion_temporal_jump_count', 0)}",
        f"- court_bounds_violations: {metrics.get('summary', {}).get('court_bounds_violations', 0)}",
        "",
        "| player | world root p95 m/frame | world root max m/frame | placement root p95 m/frame | placement root max m/frame | slide all p95 mm/frame | slide stance p95 mm/frame | unanchored run frames |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    players = metrics.get("players", {})
    if isinstance(players, Mapping):
        for player_id, player in sorted(players.items(), key=lambda item: str(item[0])):
            if not isinstance(player, Mapping):
                continue
            root = player.get("root_step_m", {})
            placement_root = player.get("placement_root_step_m", {})
            slide = player.get("foot_slide_mm_per_frame", {})
            all_slide = slide.get("all", {}) if isinstance(slide, Mapping) else {}
            stance_slide = slide.get("stance", {}) if isinstance(slide, Mapping) else {}
            lines.append(
                "| {player} | {root_p95:.6f} | {root_max:.6f} | {placement_root_p95:.6f} | {placement_root_max:.6f} | {all_p95:.3f} | {stance_p95:.3f} | {run} |".format(
                    player=player_id,
                    root_p95=float(root.get("p95", 0.0)) if isinstance(root, Mapping) else 0.0,
                    root_max=float(root.get("max", 0.0)) if isinstance(root, Mapping) else 0.0,
                    placement_root_p95=float(placement_root.get("p95", 0.0)) if isinstance(placement_root, Mapping) else 0.0,
                    placement_root_max=float(placement_root.get("max", 0.0)) if isinstance(placement_root, Mapping) else 0.0,
                    all_p95=float(all_slide.get("p95", 0.0)) if isinstance(all_slide, Mapping) else 0.0,
                    stance_p95=float(stance_slide.get("p95", 0.0)) if isinstance(stance_slide, Mapping) else 0.0,
                    run=int(player.get("longest_unanchored_foot_run_frames", 0)),
                )
            )
    return "\n".join(lines) + "\n"


__all__ = ["estimate_integer_lag_frames", "measure_visual_quality", "write_visual_quality"]
