"""Bind SAM3D/BODY artifacts back to source person identity evidence."""

from __future__ import annotations

import math
from statistics import mean
from typing import Any, Mapping, Sequence


INHERITED_ANCHOR = "placement_track_world_xy"
ROOT_TRACK_RESIDUAL_WARNING_M = 0.5
ROOT_TRACK_RESIDUAL_CONFLICT_M = 1.0


def build_sam3d_identity_evidence(
    *,
    tracks_payload: Mapping[str, Any] | None = None,
    sam3d_body_input_prep: Mapping[str, Any] | None = None,
    sam3d_keypoints_2d: Mapping[str, Any] | None = None,
    skeleton3d_payload: Mapping[str, Any] | None = None,
    smpl_motion_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tracks = _track_index(tracks_payload)
    keypoints = _frame_player_index(sam3d_keypoints_2d)
    skeleton = _frame_player_index(skeleton3d_payload)
    smpl = _frame_player_index(smpl_motion_payload)
    inherited_root = _payload_has_inherited_anchor(skeleton3d_payload) or _payload_has_inherited_anchor(smpl_motion_payload)

    body_observations: list[dict[str, Any]] = []
    for record in _prep_records(sam3d_body_input_prep):
        frame_idx = _int_or_none(record.get("frame_idx"))
        player_id = _int_or_none(record.get("player_id"))
        if frame_idx is None or player_id is None:
            continue
        key = (player_id, frame_idx)
        track_frame = tracks.get(key)
        keypoint_frame = keypoints.get(key)
        skeleton_frame = skeleton.get(key)
        smpl_frame = smpl.get(key)
        body_frame = skeleton_frame or smpl_frame
        risk_flags: list[str] = []
        frame_inherited = inherited_root or _payload_has_inherited_anchor(body_frame)
        if frame_inherited:
            risk_flags.append("placement_track_world_xy_anchor")
        crop_bbox = _bbox(record.get("original_bbox_xyxy"))
        track_bbox = _bbox(track_frame.get("bbox") if track_frame else None)
        track_world = _world_xy(track_frame.get("world_xy") if track_frame else None)
        transl = _vec3((body_frame or {}).get("transl_world") if body_frame else None)
        foot_summary = _footpoint_summary(keypoint_frame, track_bbox)
        root_residual = _root_residual(transl, track_world)
        if root_residual is not None and root_residual > ROOT_TRACK_RESIDUAL_WARNING_M:
            risk_flags.append("sam3d_root_track_residual_over_0_5m")
        if root_residual is not None and root_residual > ROOT_TRACK_RESIDUAL_CONFLICT_M:
            risk_flags.append("sam3d_root_track_residual_over_1m")
        row = {
            "body_observation_id": str(record.get("request_id") or f"{frame_idx}:{player_id}"),
            "frame_idx": frame_idx,
            "player_id": player_id,
            "detection_id": str(record.get("detection_id") or f"{frame_idx}:{player_id}"),
            "fragment_id": str(record.get("fragment_id") or f"track_{player_id}"),
            "sam3d_request_id": record.get("request_id"),
            "original_crop_bbox_xyxy": list(crop_bbox) if crop_bbox else None,
            "prepared_crop_bbox_xyxy": list(_bbox(record.get("prepared_bbox_xyxy")) or ()),
            "track_bbox_xyxy": list(track_bbox) if track_bbox else None,
            "crop_detection_residual_px": _bbox_residual(crop_bbox, track_bbox),
            "joint_confidence_mean": _joint_confidence(body_frame),
            "keypoint_confidence_mean": foot_summary["keypoint_confidence_mean"],
            "footpoint_inside_track_bbox": foot_summary["footpoint_inside_track_bbox"],
            "footpoint_vs_bbox_agreement": foot_summary["footpoint_vs_bbox_agreement"],
            "root_track_residual_m": root_residual,
            "limb_stability": {"status": "not_computed_single_frame"},
            "stance_consistency": {"status": "not_computed_single_frame"},
            "transl_world_independent": not frame_inherited,
            "risk_flags": risk_flags,
            "source_only": True,
            "uses_cvat_labels": False,
        }
        body_observations.append(row)

    inherited_count = sum(1 for row in body_observations if not row["transl_world_independent"])
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_sam3d_identity_evidence",
        "source_only": True,
        "uses_cvat_labels": False,
        "summary": {
            "body_observation_count": len(body_observations),
            "inherited_anchor_risk_count": inherited_count,
            "crop_bound_observation_count": sum(1 for row in body_observations if row["original_crop_bbox_xyxy"] is not None),
            "root_track_residual_over_0_5m_count": sum(
                1
                for row in body_observations
                if row["root_track_residual_m"] is not None and row["root_track_residual_m"] > ROOT_TRACK_RESIDUAL_WARNING_M
            ),
            "root_track_residual_over_1m_count": sum(
                1
                for row in body_observations
                if row["root_track_residual_m"] is not None and row["root_track_residual_m"] > ROOT_TRACK_RESIDUAL_CONFLICT_M
            ),
        },
        "body_observations": body_observations,
    }


def _prep_records(payload: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    records = payload.get("records")
    return [row for row in records if isinstance(row, Mapping)] if isinstance(records, list) else []


def _track_index(payload: Mapping[str, Any] | None) -> dict[tuple[int, int], Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return {}
    fps = _fps(payload)
    index: dict[tuple[int, int], Mapping[str, Any]] = {}
    for player in _players(payload):
        player_id = _int_or_none(player.get("id"))
        if player_id is None:
            continue
        for frame in _frames(player):
            frame_idx = _frame_idx(frame, fps=fps)
            if frame_idx is not None:
                index[(player_id, frame_idx)] = frame
    return index


def _frame_player_index(payload: Mapping[str, Any] | None) -> dict[tuple[int, int], Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return {}
    fps = _fps(payload)
    index: dict[tuple[int, int], Mapping[str, Any]] = {}
    for player in _players(payload):
        player_id = _int_or_none(player.get("id"))
        if player_id is None:
            continue
        for frame in _frames(player):
            frame_idx = _frame_idx(frame, fps=fps)
            if frame_idx is not None:
                index[(player_id, frame_idx)] = frame
    return index


def _players(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    players = payload.get("players")
    return [player for player in players if isinstance(player, Mapping)] if isinstance(players, list) else []


def _frames(player: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    frames = player.get("frames")
    return [frame for frame in frames if isinstance(frame, Mapping)] if isinstance(frames, list) else []


def _fps(payload: Mapping[str, Any]) -> float:
    try:
        fps = float(payload.get("fps") or 30.0)
    except (TypeError, ValueError):
        fps = 30.0
    return fps if fps > 0.0 else 30.0


def _frame_idx(frame: Mapping[str, Any], *, fps: float) -> int | None:
    for key in ("frame_idx", "frame", "frame_index"):
        if key in frame:
            return _int_or_none(frame.get(key))
    try:
        return int(round(float(frame.get("t")) * fps))
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bbox(raw: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, Sequence) or len(raw) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(raw[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _vec3(raw: Any) -> tuple[float, float, float] | None:
    if not isinstance(raw, Sequence) or len(raw) < 3:
        return None
    try:
        values = (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(value) for value in values) else None


def _world_xy(raw: Any) -> tuple[float, float] | None:
    if not isinstance(raw, Sequence) or len(raw) < 2:
        return None
    try:
        values = (float(raw[0]), float(raw[1]))
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(value) for value in values) else None


def _bbox_residual(left: tuple[float, float, float, float] | None, right: tuple[float, float, float, float] | None) -> float | None:
    if left is None or right is None:
        return None
    return round(max(abs(a - b) for a, b in zip(left, right, strict=True)), 6)


def _joint_confidence(frame: Mapping[str, Any] | None) -> float | None:
    if not isinstance(frame, Mapping):
        return None
    raw = frame.get("joint_conf")
    if not isinstance(raw, Sequence):
        return None
    values = [float(value) for value in raw if isinstance(value, int | float) and math.isfinite(float(value))]
    return round(mean(values), 6) if values else None


def _footpoint_summary(frame: Mapping[str, Any] | None, bbox: tuple[float, float, float, float] | None) -> dict[str, Any]:
    if not isinstance(frame, Mapping):
        return {
            "keypoint_confidence_mean": None,
            "footpoint_inside_track_bbox": None,
            "footpoint_vs_bbox_agreement": "missing_keypoints",
        }
    keypoints = frame.get("keypoints")
    rows = [row for row in keypoints if isinstance(row, Mapping)] if isinstance(keypoints, list) else []
    foot_rows = [row for row in rows if any(token in str(row.get("name", "")).lower() for token in ("ankle", "heel", "toe"))]
    confs = []
    inside = []
    for row in foot_rows:
        try:
            confs.append(float(row.get("conf", 0.0)))
        except (TypeError, ValueError):
            pass
        xy = row.get("xy_px")
        if bbox is not None and isinstance(xy, Sequence) and len(xy) >= 2:
            x, y = float(xy[0]), float(xy[1])
            inside.append(bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3])
    inside_value = all(inside) if inside else None
    return {
        "keypoint_confidence_mean": round(mean(confs), 6) if confs else None,
        "footpoint_inside_track_bbox": inside_value,
        "footpoint_vs_bbox_agreement": "inside" if inside_value is True else "outside_or_missing",
    }


def _root_residual(transl: tuple[float, float, float] | None, track_world: tuple[float, float] | None) -> float | None:
    if transl is None or track_world is None:
        return None
    return round(math.hypot(transl[0] - track_world[0], transl[1] - track_world[1]), 6)


def _payload_has_inherited_anchor(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if payload.get("grounding_anchor_source") == INHERITED_ANCHOR:
        return True
    for key in ("provenance", "confidence_provenance", "grounding_metrics"):
        nested = payload.get(key)
        if isinstance(nested, Mapping) and nested.get("grounding_anchor_source") == INHERITED_ANCHOR:
            return True
    return False


__all__ = ["build_sam3d_identity_evidence"]
