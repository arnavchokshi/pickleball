"""Prediction packet for reviewed BODY world-joint labeling."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_body_world_label_packet"
DEFAULT_REVIEW_MIN_SAMPLE_COUNT = 20
DEFAULT_REVIEW_MIN_COVERAGE_RATIO = 0.10


def build_body_world_label_packet(
    *,
    clip: str,
    smpl_motion: Mapping[str, Any] | None = None,
    skeleton3d: Mapping[str, Any] | None = None,
    body_compute_execution: Mapping[str, Any] | None = None,
    source_video: str = "",
    suggested_label_path: str = "labels/body_world_joints.json",
    smpl_motion_path: str | None = None,
    skeleton3d_path: str | None = None,
    body_compute_execution_path: str | None = None,
) -> dict[str, Any]:
    """Create a review packet from BODY predictions.

    The packet is not ground truth. It intentionally uses
    ``predicted_joints_world`` instead of ``joints_world`` so the world-MPJPE
    gate cannot accidentally consume it as reviewed labels.
    """

    fps = _fps(smpl_motion)
    scheduled = _scheduled_player_frames(body_compute_execution)
    joint_names = _joint_names(skeleton3d)
    samples: list[dict[str, Any]] = []
    player_ids: set[int] = set()
    frame_indexes: set[int] = set()
    joint_counts: list[int] = []

    for player in _players(smpl_motion):
        player_id = _maybe_int(player.get("id"))
        frames = player.get("frames")
        if player_id is None or not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            frame_index = _frame_index(frame, fps=fps)
            if frame_index is None:
                continue
            if scheduled and (frame_index, player_id) not in scheduled:
                continue
            joints = _vectors(frame.get("joints_world"))
            if not joints:
                continue
            conf = _number_list(frame.get("joint_conf"))
            sample = {
                "sample_id": f"frame_{frame_index:06d}_player_{player_id}",
                "frame_index": frame_index,
                "t": round(frame_index / fps, 6) if fps > 0.0 else frame.get("t"),
                "player_id": player_id,
                "track_world_xy": _vector(frame.get("track_world_xy"), length=2),
                "predicted_joints_world": joints,
                "joint_conf": conf if len(conf) == len(joints) else [],
                "joint_count": len(joints),
                "review_required": True,
            }
            if frame.get("temporal_smoothing_reset") is True:
                sample["temporal_smoothing_reset"] = True
            samples.append(sample)
            player_ids.add(player_id)
            frame_indexes.add(frame_index)
            joint_counts.append(len(joints))

    blockers = [] if samples else ["missing_body_predictions_for_label_packet"]
    review_plan = _review_plan(samples)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip": clip,
        "status": "needs_review" if samples else "blocked",
        "not_ground_truth": True,
        "trusted_for_world_mpjpe": False,
        "source_video": source_video,
        "suggested_label_path": suggested_label_path,
        "joint_names": joint_names,
        "samples": samples,
        "review_plan": review_plan,
        "summary": {
            "sample_count": len(samples),
            "player_count": len(player_ids),
            "frame_count": len(frame_indexes),
            "joint_count_min": min(joint_counts) if joint_counts else 0,
            "joint_count_max": max(joint_counts) if joint_counts else 0,
        },
        "blockers": blockers,
        "review_instructions": [
            "Use predicted_joints_world only as a visual/reference aid.",
            "Start with review_plan.selected_sample_ids so reviewed labels are representative enough for the world-MPJPE gate.",
            "Write reviewed BODY labels to suggested_label_path using samples[].joints_world only after human or trusted teacher review.",
            "Do not rename this packet to body_world_joints.json; it is not ground truth.",
        ],
        "paths": {
            "smpl_motion": smpl_motion_path or "",
            "skeleton3d": skeleton3d_path or "",
            "body_compute_execution": body_compute_execution_path or "",
        },
    }


def build_body_world_label_packet_from_paths(
    *,
    clip: str,
    smpl_motion_path: str | Path | None,
    skeleton3d_path: str | Path | None,
    body_compute_execution_path: str | Path | None = None,
    source_video: str = "",
    suggested_label_path: str = "labels/body_world_joints.json",
) -> dict[str, Any]:
    return build_body_world_label_packet(
        clip=clip,
        smpl_motion=_read_optional_json(smpl_motion_path),
        skeleton3d=_read_optional_json(skeleton3d_path),
        body_compute_execution=_read_optional_json(body_compute_execution_path),
        source_video=source_video,
        suggested_label_path=suggested_label_path,
        smpl_motion_path=str(smpl_motion_path or ""),
        skeleton3d_path=str(skeleton3d_path or ""),
        body_compute_execution_path=str(body_compute_execution_path or ""),
    )


def write_body_world_label_packet(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _players(payload: Mapping[str, Any] | None) -> list[Any]:
    players = payload.get("players") if isinstance(payload, Mapping) else None
    return players if isinstance(players, list) else []


def _fps(payload: Mapping[str, Any] | None) -> float:
    value = payload.get("fps") if isinstance(payload, Mapping) else None
    if isinstance(value, bool):
        return 30.0
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return 30.0
    return fps if fps > 0.0 else 30.0


def _scheduled_player_frames(body_compute_execution: Mapping[str, Any] | None) -> set[tuple[int, int]]:
    frames = body_compute_execution.get("scheduled_frames") if isinstance(body_compute_execution, Mapping) else None
    if not isinstance(frames, list):
        return set()
    scheduled: set[tuple[int, int]] = set()
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        frame_index = _maybe_int(frame.get("frame_idx"))
        target_ids = frame.get("target_player_ids")
        if frame_index is None or not isinstance(target_ids, list):
            continue
        for player_id in target_ids:
            parsed = _maybe_int(player_id)
            if parsed is not None:
                scheduled.add((frame_index, parsed))
    return scheduled


def _joint_names(skeleton3d: Mapping[str, Any] | None) -> list[str]:
    names = skeleton3d.get("joint_names") if isinstance(skeleton3d, Mapping) else None
    if not isinstance(names, list):
        return []
    return [str(name) for name in names]


def _review_plan(samples: list[dict[str, Any]]) -> dict[str, Any]:
    expected_sample_count = len(samples)
    required_sample_count = _required_review_sample_count(expected_sample_count)
    selected_ids = _evenly_spaced_sample_ids(samples, required_sample_count)
    return {
        "expected_sample_count": expected_sample_count,
        "required_sample_count": required_sample_count,
        "selected_sample_count": len(selected_ids),
        "selected_sample_ids": selected_ids,
        "min_sample_count": DEFAULT_REVIEW_MIN_SAMPLE_COUNT,
        "min_coverage_ratio": DEFAULT_REVIEW_MIN_COVERAGE_RATIO,
    }


def _required_review_sample_count(expected_sample_count: int) -> int:
    if expected_sample_count <= 0:
        return 0
    min_samples = max(1, DEFAULT_REVIEW_MIN_SAMPLE_COUNT)
    ratio_samples = max(1, math.ceil(expected_sample_count * DEFAULT_REVIEW_MIN_COVERAGE_RATIO))
    return min(expected_sample_count, max(min_samples, ratio_samples))


def _evenly_spaced_sample_ids(samples: list[dict[str, Any]], count: int) -> list[str]:
    if count <= 0 or not samples:
        return []
    ordered = sorted(samples, key=lambda sample: (sample.get("frame_index", 0), sample.get("player_id", 0)))
    if count >= len(ordered):
        return [str(sample["sample_id"]) for sample in ordered]
    if count == 1:
        return [str(ordered[0]["sample_id"])]
    indexes = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [str(ordered[index]["sample_id"]) for index in indexes]


def _frame_index(frame: Mapping[str, Any], *, fps: float) -> int | None:
    value = _maybe_int(frame.get("frame_idx"))
    if value is not None:
        return value
    value = _maybe_int(frame.get("frame_index"))
    if value is not None:
        return value
    t = _maybe_float(frame.get("t"))
    if t is None:
        return None
    return int(round(t * fps))


def _vectors(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    out: list[list[float]] = []
    for item in value:
        vector = _vector(item, length=3)
        if vector:
            out.append(vector)
    return out


def _vector(value: Any, *, length: int) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) != length:
        return []
    out: list[float] = []
    for item in value:
        if isinstance(item, bool):
            return []
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return []
    return out


def _number_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        if isinstance(item, bool):
            return []
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return []
    return out


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
