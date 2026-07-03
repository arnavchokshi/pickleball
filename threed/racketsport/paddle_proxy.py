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
DEFAULT_MIN_JOINT_CONFIDENCE = 0.25
DEFAULT_SMOOTHING_ALPHA = 0.55
WORLD_FRAME = "court_Z0"
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
    paddle_dims_in: Mapping[str, float] | None = None,
    wrist_offset_m: float = DEFAULT_WRIST_OFFSET_M,
    min_joint_confidence: float = DEFAULT_MIN_JOINT_CONFIDENCE,
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
) -> dict[str, Any]:
    """Build render-only paddle proxy frames from semantic wrist/elbow joints."""

    if dominant_hand not in {"right", "left", "auto"}:
        raise ValueError("dominant_hand must be one of: right, left, auto")
    if wrist_offset_m < 0.0:
        raise ValueError("wrist_offset_m must be non-negative")
    if not 0.0 <= min_joint_confidence <= 1.0:
        raise ValueError("min_joint_confidence must be in [0, 1]")
    if not 0.0 < smoothing_alpha <= 1.0:
        raise ValueError("smoothing_alpha must be in (0, 1]")

    paddle_dims = _paddle_dims(paddle_dims_in)
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

    for player in players_in:
        if not isinstance(player, Mapping):
            continue
        player_id = _get(player, "id")
        frames_in = _get(player, "frames")
        if not isinstance(frames_in, Sequence) or isinstance(frames_in, (str, bytes)):
            continue
        side = _dominant_side_for_player(
            frames_in,
            joint_indexes=joint_indexes,
            dominant_hand=dominant_hand,
            min_joint_confidence=min_joint_confidence,
        )
        state: dict[str, tuple[float, float, float]] = {}
        frames_out: list[dict[str, Any]] = []
        player_hidden = 0
        for frame in sorted((item for item in frames_in if isinstance(item, Mapping)), key=lambda item: float(_get(item, "t", 0.0))):
            input_frame_count += 1
            built, hidden = _build_proxy_frame(
                frame,
                side=side,
                joint_indexes=joint_indexes,
                wrist_offset_m=wrist_offset_m,
                min_joint_confidence=min_joint_confidence,
                smoothing_alpha=smoothing_alpha,
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
        output_players.append(
            {
                "id": player_id,
                "dominant_hand": side,
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
            "wrist_offset_m": round(float(wrist_offset_m), 6),
            "min_joint_confidence": round(float(min_joint_confidence), 6),
            "smoothing_alpha": round(float(smoothing_alpha), 6),
            "paddle_dims_in": dict(paddle_dims),
            "orientation": "forearm_length_axis_cross_world_up_face_normal_with_temporal_smoothing",
        },
        "summary": {
            "input_player_count": len([player for player in players_in if isinstance(player, Mapping)]),
            "player_count": len(output_players),
            "input_frame_count": input_frame_count,
            "estimate_frame_count": estimate_frame_count,
            "hidden_frame_count": len(hidden_frames),
            "hidden_frame_counts_by_reason": hidden_counts_by_reason,
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
    wrist_offset_m: float,
    min_joint_confidence: float,
    smoothing_alpha: float,
    state: dict[str, tuple[float, float, float]],
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
    z_axis_raw = _face_normal_candidate(y_axis_raw, previous=state.get("z_axis"))
    position_raw = _add(wrist, _scale(y_axis_raw, wrist_offset_m))
    position = _smooth_vector(state.get("position"), position_raw, smoothing_alpha)
    y_axis = _normalize(_smooth_vector(state.get("y_axis"), y_axis_raw, smoothing_alpha))
    z_axis = _orthogonalize(_smooth_vector(state.get("z_axis"), z_axis_raw, smoothing_alpha), y_axis)
    x_axis = _normalize(_cross(y_axis, z_axis))
    z_axis = _normalize(_cross(x_axis, y_axis))
    state["position"] = position
    state["y_axis"] = y_axis
    state["z_axis"] = z_axis
    rotation = _rotation_from_axes(x_axis=x_axis, y_axis=y_axis, z_axis=z_axis)
    return {
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
            "predictor": SOURCE,
            "horizon_frames": 0,
            "predicted_sigma_m": None,
        },
        "proxy_inputs": {
            "side": side,
            "wrist_world": _vec_json(wrist),
            "elbow_world": _vec_json(elbow),
            "forearm_direction_world": _vec_json(y_axis_raw),
            "joint_confidence": round(joint_conf, 6),
            "wrist_offset_m": round(float(wrist_offset_m), 6),
        },
    }, None


def _dominant_side_for_player(
    frames: Sequence[Any],
    *,
    joint_indexes: Mapping[str, int],
    dominant_hand: str,
    min_joint_confidence: float,
) -> str:
    if dominant_hand != "auto":
        return dominant_hand
    usable = {"right": 0, "left": 0}
    for side in ("right", "left"):
        wrist_idx = joint_indexes.get(f"{side}_wrist")
        elbow_idx = joint_indexes.get(f"{side}_elbow")
        if wrist_idx is None or elbow_idx is None:
            continue
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            joints = _get(frame, "joints_world")
            if _point_at(joints, wrist_idx) is None or _point_at(joints, elbow_idx) is None:
                continue
            conf = min(_confidence_at(_get(frame, "joint_conf"), wrist_idx), _confidence_at(_get(frame, "joint_conf"), elbow_idx))
            if conf >= min_joint_confidence:
                usable[side] += 1
    return "left" if usable["left"] > usable["right"] else "right"


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
) -> tuple[float, float, float]:
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


def _paddle_dims(value: Mapping[str, float] | None) -> dict[str, float]:
    dims = dict(value or DEFAULT_PADDLE_DIMS_IN)
    if not ({"length", "width"}.issubset(dims) or {"h", "w"}.issubset(dims)):
        raise ValueError("paddle_dims_in must include length/width or h/w")
    if any(float(item) <= 0.0 for item in dims.values()):
        raise ValueError("paddle_dims_in values must be positive")
    return {str(key): float(val) for key, val in dims.items()}


def _read_json_object(path: str | Path, label: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


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
    "DEFAULT_MIN_JOINT_CONFIDENCE",
    "DEFAULT_PADDLE_DIMS_IN",
    "DEFAULT_SMOOTHING_ALPHA",
    "DEFAULT_WRIST_OFFSET_M",
    "SOURCE",
    "TRUST",
    "build_paddle_proxy_from_file",
    "build_paddle_proxy_from_skeleton",
    "write_paddle_proxy",
]
