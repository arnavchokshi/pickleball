"""Render-only wrist-anchored paddle proxy from BODY skeleton joints.

This module does not estimate or claim true paddle 6DoF. It creates a
`racket_pose_estimate.json`-shaped preview artifact so the world renderer can
show an honest, low-confidence paddle near the player's wrist while the RKT
gate remains blocked on true-corner/reference GT.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .skeleton3d import semanticize_skeleton_payload


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "racketsport_racket_pose_estimate"
SOURCE = "wrist_proxy"
TRUST = "estimated_from_wrist"
DEFAULT_PADDLE_DIMS_IN = {"length": 15.5, "width": 7.5}
DEFAULT_WRIST_OFFSET_M = 0.16
DEFAULT_HANDLE_LENGTH_IN = 5.25
DEFAULT_GRIP_OFFSET_M = 0.04
DEFAULT_MIN_JOINT_CONFIDENCE = 0.25
DEFAULT_SMOOTHING_ALPHA = 0.55
DEFAULT_MOTION_HINT_MAX_BALL_TIME_DELTA_S = 0.12
WORLD_FRAME = "court_Z0"
RENDER_MESH_STYLE = "paddle_face_with_handle"
JOINT_ALIASES = {
    "left_wrist": {"left_wrist", "leftwrist", "lwrist", "l_wrist", "left_hand", "lefthand", "lhand"},
    "right_wrist": {"right_wrist", "rightwrist", "rwrist", "r_wrist", "right_hand", "righthand", "rhand"},
    "left_elbow": {"left_elbow", "leftelbow", "lelbow", "l_elbow"},
    "right_elbow": {"right_elbow", "rightelbow", "relbow", "r_elbow"},
    "left_shoulder": {"left_shoulder", "leftshoulder", "lshoulder", "l_shoulder"},
    "right_shoulder": {"right_shoulder", "rightshoulder", "rshoulder", "r_shoulder"},
}


def build_paddle_proxy_from_file(
    skeleton3d_path: str | Path,
    *,
    clip_id: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    path = Path(skeleton3d_path)
    payload = _read_json_object(path, "skeleton3d")
    return build_paddle_proxy_from_skeleton(
        payload,
        clip_id=clip_id or path.parent.name or path.stem,
        source_path=path,
        **kwargs,
    )


def build_paddle_proxy_from_skeleton(
    skeleton3d: Mapping[str, Any],
    *,
    clip_id: str,
    source_path: str | Path | None = None,
    dominant_hand: str = "right",
    dominant_hand_by_player: Mapping[int | str, str] | None = None,
    paddle_dims_in: Mapping[str, float] | None = None,
    wrist_offset_m: float | None = None,
    grip_offset_m: float = DEFAULT_GRIP_OFFSET_M,
    min_joint_confidence: float = DEFAULT_MIN_JOINT_CONFIDENCE,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
    ball_track: Mapping[str, Any] | None = None,
    motion_hint_max_ball_time_delta_s: float = DEFAULT_MOTION_HINT_MAX_BALL_TIME_DELTA_S,
) -> dict[str, Any]:
    """Build render-only paddle proxy frames from semantic wrist/elbow joints."""

    if dominant_hand not in {"right", "left", "auto"}:
        raise ValueError("dominant_hand must be one of: right, left, auto")
    if wrist_offset_m is not None and wrist_offset_m < 0.0:
        raise ValueError("wrist_offset_m must be non-negative")
    if grip_offset_m < 0.0:
        raise ValueError("grip_offset_m must be non-negative")
    if not 0.0 <= min_joint_confidence <= 1.0:
        raise ValueError("min_joint_confidence must be in [0, 1]")
    if not 0.0 < smoothing_alpha <= 1.0:
        raise ValueError("smoothing_alpha must be in (0, 1]")
    if motion_hint_max_ball_time_delta_s < 0.0:
        raise ValueError("motion_hint_max_ball_time_delta_s must be non-negative")

    paddle_dims = _paddle_dims(paddle_dims_in)
    grip_to_face_center_m = _grip_to_face_center_m(paddle_dims)
    placement_model = (
        "explicit_wrist_face_center_offset"
        if wrist_offset_m is not None
        else "hand_grip_to_face_center_from_paddle_dimensions"
    )
    hand_overrides = _hand_overrides(dominant_hand_by_player)
    ball_motion = _ball_motion_samples(ball_track)
    skeleton = semanticize_skeleton_payload(skeleton3d) or dict(skeleton3d)
    joint_names = skeleton.get("joint_names")
    joint_indexes = _joint_indexes(joint_names)
    players_in = skeleton.get("players")
    if not isinstance(players_in, Sequence) or isinstance(players_in, (str, bytes)):
        raise ValueError("skeleton3d players must be a list")

    blockers: list[str] = []
    required = {"left_wrist", "right_wrist", "left_elbow", "right_elbow"}
    if dominant_hand == "right":
        required = {"right_wrist", "right_elbow"}
    elif dominant_hand == "left":
        required = {"left_wrist", "left_elbow"}
    missing = sorted(required - set(joint_indexes))
    if missing:
        blockers.append("missing_required_wrist_forearm_joints")

    output_players: list[dict[str, Any]] = []
    hidden_frames: list[dict[str, Any]] = []
    input_frame_count = 0
    estimate_frame_count = 0
    motion_hint_frame_count = 0
    ball_reflection_orientation_frame_count = 0
    selected_hands_by_player: dict[str, str] = {}

    for player in players_in:
        if not isinstance(player, Mapping):
            continue
        player_id = _get(player, "id")
        player_id_int = _maybe_int(player_id)
        frames_in = _get(player, "frames")
        if not isinstance(frames_in, Sequence) or isinstance(frames_in, (str, bytes)):
            continue
        hand_selection = _hand_selection_for_player(
            frames_in,
            player_id=player_id_int,
            joint_indexes=joint_indexes,
            dominant_hand=dominant_hand,
            dominant_hand_by_player=hand_overrides,
            min_joint_confidence=min_joint_confidence,
        )
        side = str(hand_selection["selected_side"])
        if player_id_int is not None:
            selected_hands_by_player[str(player_id_int)] = side
        state: dict[str, Any] = {}
        frames_out: list[dict[str, Any]] = []
        player_hidden = 0
        for frame in sorted((item for item in frames_in if isinstance(item, Mapping)), key=lambda item: float(_get(item, "t", 0.0))):
            input_frame_count += 1
            built, hidden = _build_proxy_frame(
                frame,
                side=side,
                joint_indexes=joint_indexes,
                paddle_dims=paddle_dims,
                wrist_offset_m=wrist_offset_m,
                grip_offset_m=grip_offset_m,
                grip_to_face_center_m=grip_to_face_center_m,
                placement_model=placement_model,
                min_joint_confidence=min_joint_confidence,
                smoothing_alpha=smoothing_alpha,
                ball_motion=ball_motion,
                motion_hint_max_ball_time_delta_s=motion_hint_max_ball_time_delta_s,
                state=state,
            )
            if hidden is not None:
                hidden_frames.append({"player_id": player_id, **hidden})
                player_hidden += 1
                state.clear()
                continue
            if built is not None:
                frames_out.append(built)
                estimate_frame_count += 1
                if _get(built, "orientation_hint"):
                    motion_hint_frame_count += 1
                    if _get(built["orientation_hint"], "face_normal_solver") == "ball_reflection_impulse_projected_to_forearm_plane":
                        ball_reflection_orientation_frame_count += 1
        output_players.append(
            {
                "id": player_id,
                "dominant_hand": side,
                "hand_selection": hand_selection,
                "paddle_dims_in": dict(paddle_dims),
                "frames": frames_out,
                "hidden_frame_count": player_hidden,
            }
        )

    hidden_counts_by_reason = _counts_by_reason(hidden_frames)
    if hidden_frames:
        blockers.append("hidden_low_confidence_or_missing_wrist_frames")
    status = "preview" if estimate_frame_count else "blocked"
    warnings = [
        "wrist_proxy_render_only_not_rkt_gate",
        "rkt_gate_unscoreable_without_true_corner_gt",
    ]
    if motion_hint_frame_count:
        warnings.append("motion_ball_orientation_hints_estimated_only")
    if ball_reflection_orientation_frame_count:
        warnings.append("physics_ball_orientation_estimated_only")
    if hidden_frames:
        warnings.append("hidden_paddle_proxy_frames")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "clip_id": str(clip_id),
        "status": status,
        "source": SOURCE,
        "render_only": True,
        "not_for_detection_metrics": True,
        "trusted_for_rkt_promotion": False,
        "never_canonical_racket_pose": True,
        "canonical_output_forbidden": "racket_pose.json",
        "rkt_gate_unscoreable": True,
        "trust": TRUST,
        "world_frame": WORLD_FRAME,
        "translation_unit": "m",
        "fps": float(_get(skeleton, "fps", 30.0) or 30.0),
        "source_path": str(source_path or ""),
        "parameters": {
            "dominant_hand": dominant_hand,
            "dominant_hand_by_player": {str(key): value for key, value in sorted(hand_overrides.items())},
            "wrist_offset_m": round(float(wrist_offset_m), 6) if wrist_offset_m is not None else None,
            "grip_offset_m": round(float(grip_offset_m), 6),
            "grip_to_face_center_m": round(float(grip_to_face_center_m), 6),
            "placement_model": placement_model,
            "min_joint_confidence": round(float(min_joint_confidence), 6),
            "smoothing_alpha": round(float(smoothing_alpha), 6),
            "motion_hint_max_ball_time_delta_s": round(float(motion_hint_max_ball_time_delta_s), 6),
            "paddle_dims_in": dict(paddle_dims),
            "render_mesh_style": RENDER_MESH_STYLE,
            "orientation": "forearm_length_axis_cross_world_up_face_normal_with_temporal_smoothing",
        },
        "summary": {
            "input_player_count": len([player for player in players_in if isinstance(player, Mapping)]),
            "player_count": len(output_players),
            "input_frame_count": input_frame_count,
            "estimate_frame_count": estimate_frame_count,
            "motion_hint_frame_count": motion_hint_frame_count,
            "ball_reflection_orientation_frame_count": ball_reflection_orientation_frame_count,
            "hidden_frame_count": len(hidden_frames),
            "hidden_frame_counts_by_reason": hidden_counts_by_reason,
            "dominant_hand_by_player": selected_hands_by_player,
            "render_only": True,
            "trust": TRUST,
            "rkt_gate_unscoreable": True,
        },
        "warnings": warnings,
        "blockers": sorted(set(blockers)),
        "players": output_players,
        "hidden_frames": hidden_frames,
        "notes": [
            "This is an estimated wrist proxy for render continuity only, not true paddle 6DoF.",
            "The RKT face-angle/contact gate remains unscoreable until true paddle-corner/reference GT exists.",
        ],
    }


def write_paddle_proxy(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_proxy_frame(
    frame: Mapping[str, Any],
    *,
    side: str,
    joint_indexes: Mapping[str, int],
    paddle_dims: Mapping[str, float],
    wrist_offset_m: float | None,
    grip_offset_m: float,
    grip_to_face_center_m: float,
    placement_model: str,
    min_joint_confidence: float,
    smoothing_alpha: float,
    ball_motion: Sequence[Mapping[str, Any]],
    motion_hint_max_ball_time_delta_s: float,
    state: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    wrist_name = f"{side}_wrist"
    elbow_name = f"{side}_elbow"
    wrist_idx = joint_indexes.get(wrist_name)
    elbow_idx = joint_indexes.get(elbow_name)
    t = float(_get(frame, "t", 0.0))
    frame_idx = _maybe_int(_first_present(frame, "frame_idx", "frame_index", "frame"))
    hidden_base = {"frame_idx": frame_idx, "t": t, "side": side}
    if wrist_idx is None or elbow_idx is None:
        return None, {**hidden_base, "reason": "missing_joint_mapping", "joint_confidence": 0.0}
    joints = _get(frame, "joints_world")
    wrist = _point_at(joints, wrist_idx)
    elbow = _point_at(joints, elbow_idx)
    if wrist is None or elbow is None:
        return None, {**hidden_base, "reason": "missing_joint", "joint_confidence": 0.0}
    joint_conf = min(_confidence_at(_get(frame, "joint_conf"), wrist_idx), _confidence_at(_get(frame, "joint_conf"), elbow_idx))
    if joint_conf < min_joint_confidence:
        return None, {**hidden_base, "reason": "low_joint_confidence", "joint_confidence": round(joint_conf, 6)}
    forearm = _sub(wrist, elbow)
    if _norm(forearm) <= 1e-9:
        return None, {**hidden_base, "reason": "unstable_forearm_direction", "joint_confidence": round(joint_conf, 6)}
    y_axis_raw = _normalize(forearm)
    orientation_hint = _orientation_hint(
        t=t,
        wrist=wrist,
        y_axis=y_axis_raw,
        previous_wrist=state.get("previous_wrist"),
        previous_t=state.get("previous_t"),
        ball_motion=ball_motion,
        max_ball_time_delta_s=motion_hint_max_ball_time_delta_s,
    )
    hint_axis = _orientation_hint_axis(orientation_hint, y_axis=y_axis_raw)
    z_axis_raw = _face_normal_candidate(y_axis_raw, previous=state.get("z_axis"), hint_axis=hint_axis)
    grip_raw = _add(wrist, _scale(y_axis_raw, grip_offset_m))
    if wrist_offset_m is None:
        position_raw = _add(grip_raw, _scale(y_axis_raw, grip_to_face_center_m))
        face_center_offset_from_wrist_m = grip_offset_m + grip_to_face_center_m
    else:
        position_raw = _add(wrist, _scale(y_axis_raw, wrist_offset_m))
        face_center_offset_from_wrist_m = wrist_offset_m
        grip_to_face_center_m = max(0.0, wrist_offset_m - grip_offset_m)
    position = _smooth_vector(state.get("position"), position_raw, smoothing_alpha)
    y_axis = _normalize(_smooth_vector(state.get("y_axis"), y_axis_raw, smoothing_alpha))
    z_axis = _orthogonalize(_smooth_vector(state.get("z_axis"), z_axis_raw, smoothing_alpha), y_axis)
    x_axis = _normalize(_cross(y_axis, z_axis))
    z_axis = _normalize(_cross(x_axis, y_axis))
    state["position"] = position
    state["y_axis"] = y_axis
    state["z_axis"] = z_axis
    state["previous_wrist"] = wrist
    state["previous_t"] = t
    rotation = _rotation_from_axes(x_axis=x_axis, y_axis=y_axis, z_axis=z_axis)
    payload = {
        "t": t,
        "frame": frame_idx,
        "pose_se3": {"R": rotation, "t": _vec_json(position)},
        "conf": round(joint_conf, 6),
        "world_frame": WORLD_FRAME,
        "translation_unit": "m",
        "source": SOURCE,
        "reprojection_error_px": None,
        "ambiguous": False,
        "render_only": True,
        "not_for_detection_metrics": True,
        "trust": TRUST,
        "trust_band": {
            "status": TRUST,
            "stage": "RKT",
            "gate_id": "wrist_proxy_estimated_paddle",
            "badge": "low_confidence",
            "note": "estimated from wrist/forearm joints for render only; not true paddle 6DoF",
        },
        "confidence_provenance": {
            "band": TRUST,
            "display_band": "low_confidence",
            "predictor": _orientation_predictor(orientation_hint),
            "horizon_frames": 0,
            "predicted_sigma_m": None,
        },
        "render_mesh": {
            "style": RENDER_MESH_STYLE,
            "face_vertex_count": 4,
            "handle_vertex_count": 4,
        },
        "attachment": {
            "hand_side": side,
            "joint_basis": f"{wrist_name}+{elbow_name}",
            "attachment_point_world": _vec_json(wrist),
            "grip_center_world": _vec_json(grip_raw),
            "face_center_world": _vec_json(position),
            "forearm_axis_world": _vec_json(y_axis_raw),
            "grip_offset_m": round(float(grip_offset_m), 6),
            "grip_to_face_center_m": round(float(grip_to_face_center_m), 6),
            "face_center_offset_from_wrist_m": round(float(face_center_offset_from_wrist_m), 6),
            "placement_model": placement_model,
            "placement_trust": TRUST,
        },
        "proxy_inputs": {
            "side": side,
            "wrist_world": _vec_json(wrist),
            "elbow_world": _vec_json(elbow),
            "grip_center_world": _vec_json(grip_raw),
            "face_center_world": _vec_json(position),
            "forearm_direction_world": _vec_json(y_axis_raw),
            "paddle_dims_in": dict(paddle_dims),
            "grip_to_face_center_m": round(float(grip_to_face_center_m), 6),
            "placement_model": placement_model,
            "joint_confidence": round(joint_conf, 6),
            "wrist_offset_m": round(float(wrist_offset_m), 6) if wrist_offset_m is not None else None,
            "grip_offset_m": round(float(grip_offset_m), 6),
        },
    }
    if orientation_hint:
        payload["orientation_hint"] = orientation_hint
    return payload, None


def _hand_selection_for_player(
    frames: Sequence[Any],
    *,
    player_id: int | None,
    joint_indexes: Mapping[str, int],
    dominant_hand: str,
    dominant_hand_by_player: Mapping[int, str],
    min_joint_confidence: float,
) -> dict[str, Any]:
    side_scores = _hand_side_scores(frames, joint_indexes=joint_indexes, min_joint_confidence=min_joint_confidence)
    if player_id is not None and player_id in dominant_hand_by_player:
        selected = dominant_hand_by_player[player_id]
        reason = "player_override"
    elif dominant_hand != "auto":
        selected = dominant_hand
        reason = "global_dominant_hand"
    else:
        selected = _best_scored_side(side_scores)
        reason = "auto_side_score"
    return {
        "selected_side": selected,
        "selection_reason": reason,
        "side_scores": side_scores,
    }


def _hand_side_scores(
    frames: Sequence[Any],
    *,
    joint_indexes: Mapping[str, int],
    min_joint_confidence: float,
) -> dict[str, dict[str, float | int]]:
    scores: dict[str, dict[str, float | int]] = {}
    for side in ("right", "left"):
        wrist_idx = joint_indexes.get(f"{side}_wrist")
        elbow_idx = joint_indexes.get(f"{side}_elbow")
        if wrist_idx is None or elbow_idx is None:
            scores[side] = {
                "usable_frame_count": 0,
                "mean_joint_confidence": 0.0,
                "wrist_motion_m": 0.0,
                "score": 0.0,
            }
            continue
        usable = 0
        confidence_sum = 0.0
        wrist_motion_m = 0.0
        previous_wrist: tuple[float, float, float] | None = None
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            joints = _get(frame, "joints_world")
            wrist = _point_at(joints, wrist_idx)
            elbow = _point_at(joints, elbow_idx)
            if wrist is None or elbow is None:
                continue
            conf = min(_confidence_at(_get(frame, "joint_conf"), wrist_idx), _confidence_at(_get(frame, "joint_conf"), elbow_idx))
            if conf >= min_joint_confidence:
                usable += 1
                confidence_sum += conf
                if previous_wrist is not None:
                    wrist_motion_m += _norm(_sub(wrist, previous_wrist))
                previous_wrist = wrist
        mean_conf = confidence_sum / usable if usable else 0.0
        score = float(usable) * 100.0 + mean_conf + min(wrist_motion_m, 10.0)
        scores[side] = {
            "usable_frame_count": usable,
            "mean_joint_confidence": round(mean_conf, 6),
            "wrist_motion_m": round(wrist_motion_m, 6),
            "score": round(score, 6),
        }
    return scores


def _best_scored_side(side_scores: Mapping[str, Mapping[str, float | int]]) -> str:
    right = side_scores.get("right", {})
    left = side_scores.get("left", {})
    right_key = (
        int(right.get("usable_frame_count", 0)),
        float(right.get("mean_joint_confidence", 0.0)),
        float(right.get("wrist_motion_m", 0.0)),
    )
    left_key = (
        int(left.get("usable_frame_count", 0)),
        float(left.get("mean_joint_confidence", 0.0)),
        float(left.get("wrist_motion_m", 0.0)),
    )
    return "left" if left_key > right_key else "right"


def _joint_indexes(joint_names: Any) -> dict[str, int]:
    if not isinstance(joint_names, Sequence) or isinstance(joint_names, (str, bytes)):
        return {}
    mapping: dict[str, int] = {}
    for index, name in enumerate(joint_names):
        normalized = _normalize_joint_name(name)
        for semantic, aliases in JOINT_ALIASES.items():
            if normalized in aliases:
                mapping[semantic] = index
    return mapping


def _face_normal_candidate(
    y_axis: tuple[float, float, float],
    *,
    previous: tuple[float, float, float] | None,
    hint_axis: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    if hint_axis is not None and _norm(hint_axis) > 1e-6:
        candidate = _orthogonalize(hint_axis, y_axis)
        return candidate
    else:
        up = (0.0, 0.0, 1.0)
        candidate = _cross(y_axis, up)
        if _norm(candidate) <= 1e-6:
            candidate = _cross(y_axis, (0.0, 1.0, 0.0))
        if _norm(candidate) <= 1e-6:
            candidate = (1.0, 0.0, 0.0)
        candidate = _normalize(candidate)
    if previous is not None and _dot(candidate, previous) < 0.0:
        candidate = _scale(candidate, -1.0)
    return candidate


def _orientation_hint(
    *,
    t: float,
    wrist: tuple[float, float, float],
    y_axis: tuple[float, float, float],
    previous_wrist: tuple[float, float, float] | None,
    previous_t: float | None,
    ball_motion: Sequence[Mapping[str, Any]],
    max_ball_time_delta_s: float,
) -> dict[str, Any] | None:
    sources: list[str] = []
    payload: dict[str, Any] = {"trust": "estimated_from_motion"}
    if previous_wrist is not None and previous_t is not None and t > previous_t:
        swing_delta = _sub(wrist, previous_wrist)
        swing_distance = _norm(swing_delta)
        if swing_distance > 1e-6:
            sources.append("wrist_swing")
            payload["swing_direction_world"] = _vec_json(_normalize(swing_delta))
            payload["swing_speed_mps"] = round(swing_distance / max(t - previous_t, 1e-9), 6)
    ball_change = _ball_trajectory_change_near_t(ball_motion, t=t, max_delta_s=max_ball_time_delta_s)
    if ball_change is not None:
        payload["ball_time_delta_s"] = round(float(ball_change["time_delta_s"]), 6)
        if _get(ball_change, "path_direction") is not None:
            payload["ball_direction_world"] = _vec_json(ball_change["path_direction"])
        if _get(ball_change, "incoming_direction") is not None:
            payload["incoming_ball_direction_world"] = _vec_json(ball_change["incoming_direction"])
        if _get(ball_change, "outgoing_direction") is not None:
            payload["outgoing_ball_direction_world"] = _vec_json(ball_change["outgoing_direction"])
        impulse = _point3(_get(ball_change, "impulse_direction"))
        if impulse is not None:
            projected_impulse = _project_perpendicular(impulse, y_axis)
            if projected_impulse is not None:
                sources.append("ball_reflection")
                payload["ball_impulse_direction_world"] = _vec_json(impulse)
                payload["face_normal_world"] = _vec_json(projected_impulse)
                payload["face_normal_solver"] = "ball_reflection_impulse_projected_to_forearm_plane"
                payload["orientation_model"] = "hand_anchor_ball_reflection_estimate"
        if "face_normal_world" not in payload:
            path_direction = _point3(_get(ball_change, "path_direction"))
            projected_path = _project_perpendicular(path_direction, y_axis) if path_direction is not None else None
            if projected_path is not None:
                sources.append("ball_path")
                payload["face_normal_world"] = _vec_json(projected_path)
                payload["face_normal_solver"] = "ball_path_projected_to_forearm_plane"
                payload["orientation_model"] = "hand_anchor_ball_path_estimate"
    if not sources:
        return None
    payload.setdefault("orientation_model", "hand_anchor_motion_estimate")
    payload["sources"] = sources
    return payload


def _orientation_hint_axis(
    hint: Mapping[str, Any] | None,
    *,
    y_axis: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    if not hint:
        return None
    raw = (
        _point3(_get(hint, "face_normal_world"))
        or _point3(_get(hint, "ball_impulse_direction_world"))
        or _point3(_get(hint, "ball_direction_world"))
        or _point3(_get(hint, "swing_direction_world"))
    )
    if raw is None:
        return None
    projected = _project_perpendicular(raw, y_axis)
    if projected is None:
        return None
    return projected


def _ball_motion_samples(ball_track: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(ball_track, Mapping):
        return []
    samples = []
    for frame in _sequence(_get(ball_track, "frames")):
        t = _maybe_float(_get(frame, "t"))
        point = _point3(_first_present(frame, "world_xyz", "world"))
        if t is None or point is None:
            continue
        samples.append({"t": t, "point": point})
    return sorted(samples, key=lambda item: float(item["t"]))


def _ball_trajectory_change_near_t(
    samples: Sequence[Mapping[str, Any]],
    *,
    t: float,
    max_delta_s: float,
) -> dict[str, Any] | None:
    if len(samples) < 2:
        return None
    nearest_index = min(range(len(samples)), key=lambda index: abs(float(samples[index]["t"]) - t))
    nearest_t = float(samples[nearest_index]["t"])
    time_delta = abs(nearest_t - t)
    if time_delta > max_delta_s:
        return None
    previous_index = max(0, nearest_index - 1)
    next_index = min(len(samples) - 1, nearest_index + 1)
    if previous_index == next_index:
        return None
    previous_point = _point3(samples[previous_index].get("point"))
    nearest_point = _point3(samples[nearest_index].get("point"))
    next_point = _point3(samples[next_index].get("point"))
    if previous_point is None or nearest_point is None or next_point is None:
        return None
    path_direction = _normalized_or_none(_sub(next_point, previous_point))
    incoming_direction = (
        _normalized_or_none(_sub(nearest_point, previous_point))
        if previous_index != nearest_index
        else None
    )
    outgoing_direction = (
        _normalized_or_none(_sub(next_point, nearest_point))
        if next_index != nearest_index
        else None
    )
    if path_direction is None and incoming_direction is None and outgoing_direction is None:
        return None
    impulse_direction = (
        _normalized_or_none(_sub(outgoing_direction, incoming_direction))
        if incoming_direction is not None and outgoing_direction is not None
        else None
    )
    return {
        "path_direction": path_direction,
        "incoming_direction": incoming_direction,
        "outgoing_direction": outgoing_direction,
        "impulse_direction": impulse_direction,
        "time_delta_s": time_delta,
    }


def _orientation_predictor(orientation_hint: Mapping[str, Any] | None) -> str:
    if not orientation_hint:
        return SOURCE
    if _get(orientation_hint, "face_normal_solver") == "ball_reflection_impulse_projected_to_forearm_plane":
        return "wrist_proxy+ball_reflection_hint"
    return "wrist_proxy+motion_hint"


def _project_perpendicular(
    vector: tuple[float, float, float],
    axis: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    projected = _sub(vector, _scale(axis, _dot(vector, axis)))
    if _norm(projected) <= 1e-6:
        return None
    return _normalize(projected)


def _normalized_or_none(vector: tuple[float, float, float]) -> tuple[float, float, float] | None:
    if _norm(vector) <= 1e-6:
        return None
    return _normalize(vector)


def _orthogonalize(vector: tuple[float, float, float], axis: tuple[float, float, float]) -> tuple[float, float, float]:
    corrected = _sub(vector, _scale(axis, _dot(vector, axis)))
    if _norm(corrected) <= 1e-9:
        corrected = _face_normal_candidate(axis, previous=None)
    return _normalize(corrected)


def _rotation_from_axes(
    *,
    x_axis: tuple[float, float, float],
    y_axis: tuple[float, float, float],
    z_axis: tuple[float, float, float],
) -> list[list[float]]:
    return [
        [round(float(x_axis[row]), 9), round(float(y_axis[row]), 9), round(float(z_axis[row]), 9)]
        for row in range(3)
    ]


def _smooth_vector(
    previous: tuple[float, float, float] | None,
    current: tuple[float, float, float],
    alpha: float,
) -> tuple[float, float, float]:
    if previous is None:
        return current
    return tuple(float(previous[i]) * (1.0 - alpha) + float(current[i]) * alpha for i in range(3))  # type: ignore[return-value]


def _counts_by_reason(hidden_frames: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in hidden_frames:
        reason = str(frame.get("reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _hand_overrides(value: Mapping[int | str, str] | None) -> dict[int, str]:
    overrides: dict[int, str] = {}
    if value is None:
        return overrides
    for raw_player_id, raw_side in value.items():
        player_id = _maybe_int(raw_player_id)
        if player_id is None:
            raise ValueError("dominant_hand_by_player keys must be player ids")
        side = str(raw_side)
        if side not in {"right", "left"}:
            raise ValueError("dominant_hand_by_player values must be right or left")
        overrides[player_id] = side
    return overrides


def _paddle_dims(value: Mapping[str, float] | None) -> dict[str, float]:
    dims = dict(value or DEFAULT_PADDLE_DIMS_IN)
    if not ({"length", "width"}.issubset(dims) or {"h", "w"}.issubset(dims)):
        raise ValueError("paddle_dims_in must include length/width or h/w")
    if any(float(item) <= 0.0 for item in dims.values()):
        raise ValueError("paddle_dims_in values must be positive")
    return {str(key): float(val) for key, val in dims.items()}


def _grip_to_face_center_m(paddle_dims_in: Mapping[str, float]) -> float:
    face_length_in = float(paddle_dims_in.get("length", paddle_dims_in.get("h", DEFAULT_PADDLE_DIMS_IN["length"])))
    handle_length_in = float(
        paddle_dims_in.get(
            "handle_length",
            paddle_dims_in.get("handle_length_in", DEFAULT_HANDLE_LENGTH_IN),
        )
    )
    if face_length_in <= 0.0:
        raise ValueError("paddle face length must be positive")
    if handle_length_in <= 0.0:
        raise ValueError("paddle handle length must be positive")
    return (face_length_in + handle_length_in) * 0.0254 / 2.0


def _read_json_object(path: str | Path, label: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _point_at(joints: Any, index: int) -> tuple[float, float, float] | None:
    if not isinstance(joints, Sequence) or isinstance(joints, (str, bytes)) or index >= len(joints):
        return None
    point = joints[index]
    if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 3:
        return None
    try:
        return (float(point[0]), float(point[1]), float(point[2]))
    except (TypeError, ValueError):
        return None


def _point3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    try:
        point = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(component) for component in point):
        return None
    return point


def _confidence_at(conf: Any, index: int) -> float:
    if not isinstance(conf, Sequence) or isinstance(conf, (str, bytes)) or index >= len(conf):
        return 0.0
    try:
        return max(0.0, min(1.0, float(conf[index])))
    except (TypeError, ValueError):
        return 0.0


def _vec_json(vector: Sequence[float]) -> list[float]:
    return [round(float(value), 9) for value in vector]


def _normalize_joint_name(value: Any) -> str:
    text = str(value).lower().replace("-", "_")
    return "_".join(part for part in text.split("_") if part)


def _first_present(value: Any, *fields: str) -> Any:
    for field in fields:
        candidate = _get(value, field)
        if candidate is not None:
            return candidate
    return None


def _get(value: Any, field: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(field, default)
    return getattr(value, field, default)


def _maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2]))


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))


def _scale(vector: Sequence[float], scalar: float) -> tuple[float, float, float]:
    return (float(vector[0]) * scalar, float(vector[1]) * scalar, float(vector[2]) * scalar)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Sequence[float]) -> tuple[float, float, float]:
    norm = _norm(vector)
    if norm <= 1e-12:
        raise ValueError("cannot normalize zero-length vector")
    return (float(vector[0]) / norm, float(vector[1]) / norm, float(vector[2]) / norm)


__all__ = [
    "DEFAULT_GRIP_OFFSET_M",
    "DEFAULT_MIN_JOINT_CONFIDENCE",
    "DEFAULT_MOTION_HINT_MAX_BALL_TIME_DELTA_S",
    "DEFAULT_PADDLE_DIMS_IN",
    "DEFAULT_SMOOTHING_ALPHA",
    "DEFAULT_WRIST_OFFSET_M",
    "SOURCE",
    "TRUST",
    "build_paddle_proxy_from_file",
    "build_paddle_proxy_from_skeleton",
    "write_paddle_proxy",
]
