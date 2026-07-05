"""All-human source-only identity fragments for TRK diagnostics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .court_calibration import project_image_points_to_world
from .person_court_membership import NON_TARGET_CLASSES, classify_world_position


def build_identity_fragment_artifacts(
    *,
    raw_pool_payload: Mapping[str, Any] | None = None,
    tracks_payload: Mapping[str, Any] | None = None,
    calibration_payload: Mapping[str, Any] | None = None,
    source_name: str = "unknown",
    expected_target_players: int = 4,
    split_gap_frames: int = 24,
) -> dict[str, Any]:
    observations = (
        _observations_from_raw_pool(raw_pool_payload, calibration_payload=calibration_payload)
        if raw_pool_payload is not None
        else []
    )
    if not observations and tracks_payload is not None:
        observations = _observations_from_tracks(tracks_payload)
    fragments = _build_fragments(observations, split_gap_frames=split_gap_frames)
    target_count = sum(1 for fragment in fragments if fragment["eligible_for_target_selection"])
    non_target_count = len(fragments) - target_count
    return {
        "human_observations": {
            "schema_version": 1,
            "artifact_type": "racketsport_human_observations",
            "source_only": True,
            "uses_cvat_labels": False,
            "source_name": source_name,
            "observation_count": len(observations),
            "observations": observations,
        },
        "identity_fragments": {
            "schema_version": 1,
            "artifact_type": "racketsport_identity_fragments",
            "source_only": True,
            "uses_cvat_labels": False,
            "source_name": source_name,
            "fragment_count": len(fragments),
            "fragments": fragments,
        },
        "identity_association_report": {
            "schema_version": 1,
            "artifact_type": "racketsport_identity_association_report",
            "source_only": True,
            "uses_cvat_labels": False,
            "source_name": source_name,
            "input_human_observation_count": len(observations),
            "fragment_count": len(fragments),
            "expected_target_players": expected_target_players,
            "target_player_candidate_count": target_count,
            "non_target_diagnostic_fragment_count": non_target_count,
            "final_tracks_written": False,
            "notes": [
                "All detected humans are preserved as diagnostics before any exactly-four target-player selection.",
                "This artifact does not rewrite tracks.json and does not use CVAT labels.",
            ],
        },
    }


def _observations_from_raw_pool(
    payload: Mapping[str, Any] | None,
    *,
    calibration_payload: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("raw_pool_payload must contain a frames list")
    observations: list[dict[str, Any]] = []
    for default_frame_idx, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            continue
        frame_idx = _frame_idx(frame, default_frame_idx)
        detections = frame.get("detections")
        if not isinstance(detections, list):
            continue
        for det_idx, detection in enumerate(detections):
            if not isinstance(detection, Mapping) or not _is_person(detection):
                continue
            bbox = _bbox(detection)
            source_id = _source_id(detection, det_idx + 1)
            world_xy = _world_xy(detection) or _project_bbox_footpoint(bbox, calibration_payload)
            observations.append(
                _observation(
                    detection_id=f"{frame_idx}:{source_id}:{det_idx}",
                    fragment_id=f"track_{source_id}",
                    frame_idx=frame_idx,
                    source_tracker_id=source_id,
                    bbox=bbox,
                    conf=_conf(detection),
                    world_xy=world_xy,
                    raw_player_id=_int_or_none(detection.get("player_id")),
                )
            )
    return sorted(observations, key=lambda row: (row["frame_idx"], row["source_tracker_id"], row["detection_id"]))


def _observations_from_tracks(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    fps = _fps(payload)
    players = payload.get("players")
    if not isinstance(players, list):
        return []
    observations: list[dict[str, Any]] = []
    for player in players:
        if not isinstance(player, Mapping):
            continue
        player_id = _int_or_none(player.get("id"))
        if player_id is None:
            continue
        frames = player.get("frames")
        if not isinstance(frames, list):
            continue
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            frame_idx = _frame_idx(frame, int(round(float(frame.get("t", 0.0)) * fps)))
            bbox = _bbox(frame)
            observations.append(
                _observation(
                    detection_id=f"{frame_idx}:{player_id}",
                    fragment_id=f"track_{player_id}",
                    frame_idx=frame_idx,
                    source_tracker_id=player_id,
                    bbox=bbox,
                    conf=_conf(frame),
                    world_xy=_world_xy(frame),
                    raw_player_id=player_id,
                )
            )
    return sorted(observations, key=lambda row: (row["frame_idx"], row["source_tracker_id"]))


def _build_fragments(observations: list[dict[str, Any]], *, split_gap_frames: int) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        grouped[int(observation["source_tracker_id"])].append(observation)
    fragments: list[dict[str, Any]] = []
    next_id = 1
    for source_id, rows in sorted(grouped.items()):
        current: list[dict[str, Any]] = []
        previous_frame: int | None = None
        for row in sorted(rows, key=lambda item: int(item["frame_idx"])):
            frame_idx = int(row["frame_idx"])
            if previous_frame is not None and frame_idx - previous_frame > split_gap_frames:
                fragments.append(_fragment(next_id, source_id, current))
                next_id += 1
                current = []
            current.append(row)
            previous_frame = frame_idx
        if current:
            fragments.append(_fragment(next_id, source_id, current))
            next_id += 1
    return fragments


def _fragment(fragment_id: int, source_id: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    classes = [classify_world_position(row.get("projected_world_xy"))["membership_class"] for row in rows]
    class_counts = {name: classes.count(name) for name in sorted(set(classes))}
    non_target = sum(class_counts.get(name, 0) for name in NON_TARGET_CLASSES)
    eligible = non_target < max(1, len(rows) / 2)
    speeds = _speeds(rows)
    blockers: list[str] = []
    if not eligible:
        blockers.append("outside_target_court_geometry")
    return {
        "fragment_id": f"frag_{fragment_id}",
        "source_tracker_id": source_id,
        "detection_ids": [row["detection_id"] for row in rows],
        "start_frame": int(rows[0]["frame_idx"]),
        "end_frame": int(rows[-1]["frame_idx"]),
        "coverage_frames": len(rows),
        "bbox_center_trajectory": [_bbox_center(row["bbox_xyxy"]) for row in rows],
        "bbox_height_trajectory": [round(float(row["bbox_xyxy"][3]) - float(row["bbox_xyxy"][1]), 6) for row in rows],
        "world_trajectory": [row.get("projected_world_xy") for row in rows],
        "court_membership_summary": class_counts,
        "speed_m_s_p50": _percentile(speeds, 50),
        "speed_m_s_p90": _percentile(speeds, 90),
        "speed_teleport_flags": [speed for speed in speeds if speed > 10.0],
        "same_frame_conflicts": [],
        "candidate_role_side_evidence": {},
        "eligible_for_target_selection": eligible,
        "target_selection_blockers": blockers,
    }


def _observation(
    *,
    detection_id: str,
    fragment_id: str,
    frame_idx: int,
    source_tracker_id: int,
    bbox: tuple[float, float, float, float],
    conf: float,
    world_xy: list[float] | None,
    raw_player_id: int | None,
) -> dict[str, Any]:
    return {
        "detection_id": detection_id,
        "fragment_id": fragment_id,
        "frame_idx": frame_idx,
        "bbox_xyxy": [round(value, 6) for value in bbox],
        "detector_conf": round(conf, 6),
        "source_tracker_id": source_tracker_id,
        "raw_player_id": raw_player_id,
        "osnet_embedding_id": None,
        "footpoint_px": [round((bbox[0] + bbox[2]) / 2.0, 6), round(bbox[3], 6)],
        "projected_world_xy": [round(float(world_xy[0]), 6), round(float(world_xy[1]), 6)] if world_xy else None,
        "court_membership_evidence": {},
        "body_observation_ids": [],
        "source_only": True,
        "uses_cvat_labels": False,
    }


def _is_person(detection: Mapping[str, Any]) -> bool:
    value = detection.get("class", "person")
    return value == 0 or str(value).lower() in {"person", "player", "0"}


def _bbox(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    raw = row.get("bbox") or row.get("bbox_xyxy")
    if not isinstance(raw, Sequence) or len(raw) < 4:
        raise ValueError("person detection/frame must contain bbox/bbox_xyxy")
    x1, y1, x2, y2 = (float(raw[index]) for index in range(4))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must be ordered as x1, y1, x2, y2")
    return x1, y1, x2, y2


def _world_xy(row: Mapping[str, Any]) -> list[float] | None:
    raw = row.get("world_xy") or row.get("projected_world_xy")
    if not isinstance(raw, Sequence) or len(raw) < 2:
        return None
    try:
        x, y = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return [x, y]


def _project_bbox_footpoint(
    bbox: tuple[float, float, float, float],
    calibration_payload: Mapping[str, Any] | None,
) -> list[float] | None:
    if not isinstance(calibration_payload, Mapping):
        return None
    homography = calibration_payload.get("homography")
    if not isinstance(homography, Sequence):
        return None
    footpoint = [(bbox[0] + bbox[2]) / 2.0, bbox[3]]
    try:
        projected = project_image_points_to_world(homography, [footpoint])[0]
    except (TypeError, ValueError, IndexError):
        return None
    if len(projected) < 2:
        return None
    x, y = float(projected[0]), float(projected[1])
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return [x, y]


def _frame_idx(row: Mapping[str, Any], default: int) -> int:
    for key in ("frame_idx", "frame", "frame_index"):
        if key in row:
            return int(row[key])
    return int(default)


def _source_id(row: Mapping[str, Any], fallback: int) -> int:
    for key in ("track_id", "source_track_id", "player_id", "id"):
        value = _int_or_none(row.get(key))
        if value is not None:
            return value
    return fallback


def _fps(payload: Mapping[str, Any]) -> float:
    try:
        fps = float(payload.get("fps") or 30.0)
    except (TypeError, ValueError):
        fps = 30.0
    return fps if fps > 0.0 else 30.0


def _conf(row: Mapping[str, Any]) -> float:
    try:
        return float(row.get("conf", row.get("confidence", 1.0)))
    except (TypeError, ValueError):
        return 1.0


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bbox_center(bbox: Sequence[float]) -> list[float]:
    return [round((float(bbox[0]) + float(bbox[2])) / 2.0, 6), round((float(bbox[1]) + float(bbox[3])) / 2.0, 6)]


def _speeds(rows: list[dict[str, Any]]) -> list[float]:
    speeds: list[float] = []
    for previous, current in zip(rows, rows[1:]):
        prev_world = previous.get("projected_world_xy")
        curr_world = current.get("projected_world_xy")
        if not prev_world or not curr_world:
            continue
        dt = int(current["frame_idx"]) - int(previous["frame_idx"])
        if dt <= 0:
            continue
        distance = math.hypot(float(curr_world[0]) - float(prev_world[0]), float(curr_world[1]) - float(prev_world[1]))
        speeds.append(round(distance / dt, 6))
    return speeds


def _percentile(values: Sequence[float], percentile: float) -> float:
    cleaned = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not cleaned:
        return 0.0
    if len(cleaned) == 1:
        return round(cleaned[0], 6)
    rank = (len(cleaned) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(cleaned[int(lower)], 6)
    weight = rank - lower
    return round(cleaned[int(lower)] * (1.0 - weight) + cleaned[int(upper)] * weight, 6)


__all__ = ["build_identity_fragment_artifacts"]
