"""Transfer-oriented shot-label baseline from existing contact and ball cues.

This module is deliberately conservative. It does not claim BST/PoseConv3D
accuracy; it gives a deterministic label suggestion with provenance so current
pickleball runs can produce visual review artifacts before learned SHOT-1
training is complete.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from threed.racketsport.skeleton3d import semanticize_skeleton_payload


CLASSIFIER_NAME = "shot_transfer_baseline_v1"
DEFAULT_MAX_BALL_DT_S = 0.30
DEFAULT_LOOSE_BALL_DT_S = 0.75
NVZ_HALF_DEPTH_M = 2.1336
BASELINE_Y_M = 6.7056


def classify_shots_from_payloads(
    *,
    clip_id: str,
    contact_windows_payload: Mapping[str, Any],
    ball_inflections_payload: Mapping[str, Any] | None = None,
    skeleton3d_payload: Mapping[str, Any] | None = None,
    smpl_motion_payload: Mapping[str, Any] | None = None,
    tracks_payload: Mapping[str, Any] | None = None,
    ball_track_payload: Mapping[str, Any] | None = None,
    max_ball_dt_s: float = DEFAULT_MAX_BALL_DT_S,
) -> dict[str, object]:
    """Build a shot-classification artifact from cue payload dictionaries."""

    if not clip_id:
        raise ValueError("clip_id is required")
    max_ball_dt_s = _require_non_negative(max_ball_dt_s, "max_ball_dt_s")
    contacts = _contact_events(contact_windows_payload)
    inflections = _ball_inflections(ball_inflections_payload or {})
    pose_payload = (
        semanticize_skeleton_payload(skeleton3d_payload)
        or semanticize_skeleton_payload(smpl_motion_payload)
        or {}
    )
    tracks = _track_players(tracks_payload or {})
    ball_frames = _ball_track_frames(ball_track_payload or {})

    shots: list[dict[str, object]] = []
    for index, contact in enumerate(contacts):
        match = _nearest_inflection(contact["t"], inflections, max_ball_dt_s=max_ball_dt_s)
        fallback = _pose_track_fallback(
            contact=contact,
            match=match,
            pose_payload=pose_payload,
            tracks=tracks,
            ball_frames=ball_frames,
            max_ball_dt_s=max_ball_dt_s,
        )
        shots.append(
            _classify_contact(
                index=index,
                contact=contact,
                match=match,
                fallback=fallback,
                max_ball_dt_s=max_ball_dt_s,
            )
        )

    unknown_count = sum(1 for shot in shots if shot["type"] == "unknown")
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_shot_classification",
        "clip_id": clip_id,
        "classifier": {
            "name": CLASSIFIER_NAME,
            "family": "transfer_or_heuristic",
            "not_gate_verified": True,
            "trained_model": None,
            "notes": [
                "Uses court/ball/contact cues as a provisional transfer baseline.",
                "Do not treat these labels as pickleball accuracy until SHOT-1 has human-reviewed labels and held-out metrics.",
            ],
        },
        "shots": shots,
        "summary": {
            "shot_count": len(shots),
            "unknown_count": unknown_count,
            "known_count": len(shots) - unknown_count,
        },
    }


def _classify_contact(
    *,
    index: int,
    contact: dict[str, Any],
    match: dict[str, Any] | None,
    fallback: dict[str, Any] | None,
    max_ball_dt_s: float,
) -> dict[str, object]:
    base = {
        "id": f"shot_{index:04d}",
        "t": _round_time(contact["t"]),
        "frame": int(contact["frame"]),
        "player_id": contact.get("player_id") if contact.get("player_id") is not None else (fallback or {}).get("player_id"),
    }
    if match is None:
        if fallback is not None:
            confidence = round(float(contact["confidence"]) * float(fallback["confidence"]), 6)
            abstract_label = _abstract_side_label(str(fallback["type"]))
            second = "bh_shot" if abstract_label == "fh_shot" else "fh_shot"
            return {
                **base,
                "type": abstract_label,
                "specific_type_candidate": fallback["type"],
                "type_conf": confidence,
                "gated": False,
                "gate_reasons": [],
                "top2": [
                    {"type": abstract_label, "confidence": confidence},
                    {"type": second, "confidence": round(max(0.0, confidence - 0.16), 6)},
                ],
                "evidence": {
                    "contact": _contact_evidence(contact),
                    "matched_ball_inflection": None,
                    "pose_track_fallback": _fallback_evidence(fallback),
                    "rules": [f"no_ball_inflection_within_{max_ball_dt_s:.3f}s", "fallback_label_from_pose_or_track"],
                },
            }
        return {
            **base,
            "type": "unknown",
            "type_conf": 0.0,
            "gated": True,
            "gate_reasons": [f"no ball inflection within {max_ball_dt_s:.3f}s"],
            "top2": [],
            "evidence": {"contact": _contact_evidence(contact), "matched_ball_inflection": None},
        }

    label, score, second = _heuristic_label(match)
    if fallback is not None and label in {"fh_drive", "bh_drive"} and fallback["type"] in {"fh_drive", "bh_drive"}:
        label = str(fallback["type"])
        second = "bh_drive" if label == "fh_drive" else "fh_drive"
    confidence = _combined_confidence(contact, match, score)
    return {
        **base,
        "type": label,
        "type_conf": confidence,
        "gated": False,
        "gate_reasons": [],
        "top2": [
            {"type": label, "confidence": confidence},
            {"type": second, "confidence": round(max(0.0, confidence - 0.18), 6)},
        ],
        "evidence": {
            "contact": _contact_evidence(contact),
            "matched_ball_inflection": _inflection_evidence(match),
            "pose_track_fallback": _fallback_evidence(fallback) if fallback is not None else None,
            "rules": _rule_notes(match),
        },
    }


def _heuristic_label(match: Mapping[str, Any]) -> tuple[str, float, str]:
    xyz = match.get("ball_world_xyz") or [0.0, 0.0, 0.0]
    x = float(xyz[0])
    y = float(xyz[1])
    before = float(match.get("speed_before_mps", 0.0) or 0.0)
    after = float(match.get("speed_after_mps", 0.0) or 0.0)
    speed = max(before, after)
    abs_y = abs(y)

    if abs_y <= NVZ_HALF_DEPTH_M and speed <= 5.5:
        return "dink", 0.84, "reset_block"
    if abs_y <= NVZ_HALF_DEPTH_M and speed <= 8.0:
        return "reset_block", 0.70, "dink"
    if abs_y >= BASELINE_Y_M * 0.72 and speed <= 7.0:
        return "third_shot_drop", 0.66, "lob"
    if speed >= 8.0:
        return ("fh_drive" if x >= 0 else "bh_drive"), 0.72, "reset_block"
    if after >= 6.0 and abs_y >= BASELINE_Y_M * 0.55:
        return "lob", 0.62, "third_shot_drop"
    return "reset_block", 0.58, "dink"


def _abstract_side_label(label: str) -> str:
    if label.startswith("fh_"):
        return "fh_shot"
    if label.startswith("bh_"):
        return "bh_shot"
    return label


def _combined_confidence(contact: Mapping[str, Any], match: Mapping[str, Any], rule_score: float) -> float:
    contact_conf = float(contact.get("confidence", 0.0) or 0.0)
    ball_conf = float(match.get("confidence", 0.0) or 0.0)
    dt_penalty = min(0.20, float(match["dt_s"]) / max(DEFAULT_MAX_BALL_DT_S, 1e-6) * 0.20)
    confidence = (0.40 * contact_conf) + (0.30 * ball_conf) + (0.30 * rule_score) - dt_penalty
    return round(max(0.0, min(1.0, confidence)), 6)


def _contact_events(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    events = payload.get("events", [])
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise ValueError("contact_windows.events must be a sequence")
    contacts = []
    for item in events:
        if not isinstance(item, Mapping) or item.get("type", "contact") != "contact":
            continue
        contacts.append(
            {
                "t": _require_non_negative(item.get("t"), "contact.t"),
                "frame": _require_non_negative_int(item.get("frame", 0), "contact.frame"),
                "player_id": item.get("player_id"),
                "confidence": _unit_interval(item.get("confidence", 0.0), "contact.confidence"),
                "sources": dict(item.get("sources", {})) if isinstance(item.get("sources", {}), Mapping) else {},
                "window": dict(item.get("window", {})) if isinstance(item.get("window", {}), Mapping) else {},
            }
        )
    contacts.sort(key=lambda item: (item["t"], item["frame"]))
    return contacts


def _ball_inflections(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = payload.get("candidates", payload.get("ball_inflections", []))
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("ball_inflections candidates must be a sequence")
    inflections = []
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        time_s = item.get("time_s", item.get("t"))
        xyz = item.get("ball_world_xyz", item.get("world_xyz", [0.0, 0.0, 0.0]))
        if not isinstance(xyz, Sequence) or isinstance(xyz, (str, bytes)) or len(xyz) != 3:
            xyz = [0.0, 0.0, 0.0]
        inflections.append(
            {
                "time_s": _require_non_negative(time_s, "ball.time_s"),
                "frame": _require_non_negative_int(item.get("frame", 0), "ball.frame"),
                "ball_world_xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                "confidence": _unit_interval(item.get("confidence", 0.0), "ball.confidence"),
                "speed_before_mps": _finite_or_zero(item.get("speed_before_mps")),
                "speed_after_mps": _finite_or_zero(item.get("speed_after_mps")),
                "turn_angle_deg": _finite_or_zero(item.get("turn_angle_deg")),
                "approx": bool(item.get("approx", False)),
            }
        )
    inflections.sort(key=lambda item: item["time_s"])
    return inflections


def _nearest_inflection(
    contact_t: float,
    inflections: Sequence[dict[str, Any]],
    *,
    max_ball_dt_s: float,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for item in inflections:
        dt = abs(float(item["time_s"]) - contact_t)
        if dt > max_ball_dt_s:
            continue
        if best is None or dt < float(best["dt_s"]):
            best = {**item, "dt_s": round(dt, 6)}
    return best


def _pose_track_fallback(
    *,
    contact: Mapping[str, Any],
    match: Mapping[str, Any] | None,
    pose_payload: Mapping[str, Any],
    tracks: Sequence[dict[str, Any]],
    ball_frames: Sequence[dict[str, Any]],
    max_ball_dt_s: float,
) -> dict[str, Any] | None:
    ball_world = (match or {}).get("ball_world_xyz")
    pose_hint = _semantic_pose_hint(contact=contact, pose_payload=pose_payload, ball_world=ball_world)
    if pose_hint is not None:
        return pose_hint

    ball_frame = _nearest_ball_frame(float(contact["t"]), ball_frames, max_dt_s=max_ball_dt_s)
    if ball_frame is not None:
        track_hint = _track_ball_side_hint(contact=contact, tracks=tracks, ball_frame=ball_frame, max_dt_s=max_ball_dt_s)
        if track_hint is not None:
            return track_hint

    loose_max_dt_s = max(DEFAULT_LOOSE_BALL_DT_S, max_ball_dt_s)
    loose_ball_frame = ball_frame or _nearest_ball_frame(float(contact["t"]), ball_frames, max_dt_s=loose_max_dt_s)
    if loose_ball_frame is None:
        return None
    return _ball_track_image_side_hint(ball_frame=loose_ball_frame, max_dt_s=loose_max_dt_s)


def _semantic_pose_hint(
    *,
    contact: Mapping[str, Any],
    pose_payload: Mapping[str, Any],
    ball_world: Any,
) -> dict[str, Any] | None:
    joint_names = pose_payload.get("joint_names")
    if not isinstance(joint_names, Sequence) or isinstance(joint_names, (str, bytes)):
        return None
    name_to_index = {_normalize_name(name): index for index, name in enumerate(joint_names)}
    left_wrist_index = name_to_index.get("leftwrist")
    right_wrist_index = name_to_index.get("rightwrist")
    if left_wrist_index is None or right_wrist_index is None:
        return None

    frame_info = _nearest_pose_frame(
        pose_payload=pose_payload,
        contact_t=float(contact["t"]),
        player_id=contact.get("player_id"),
    )
    if frame_info is None:
        return None
    player_id, frame, frame_dt = frame_info
    joints = frame.get("joints_world")
    if not isinstance(joints, Sequence) or len(joints) <= max(left_wrist_index, right_wrist_index):
        return None
    left_wrist = _vector(joints[left_wrist_index], length=3)
    right_wrist = _vector(joints[right_wrist_index], length=3)
    if left_wrist is None or right_wrist is None:
        return None

    conf = _joint_confidence(frame, [left_wrist_index, right_wrist_index])
    if _is_vector(ball_world, length=3):
        ball = _vector(ball_world, length=3)
        if ball is not None:
            left_dist = _xy_distance(left_wrist, ball)
            right_dist = _xy_distance(right_wrist, ball)
            label = "fh_drive" if right_dist <= left_dist else "bh_drive"
            return {
                "type": label,
                "confidence": max(0.40, 0.72 * conf),
                "source": "semantic_wrist_to_ball",
                "player_id": player_id,
                "pose_frame_dt_s": round(frame_dt, 6),
                "semantic_joint_source": str(pose_payload.get("semantic_joint_source", "already_semantic")),
                "left_wrist_xy": [left_wrist[0], left_wrist[1]],
                "right_wrist_xy": [right_wrist[0], right_wrist[1]],
            }

    shoulder_mid = _shoulder_midpoint(joints, name_to_index)
    if shoulder_mid is not None:
        left_extension = abs(left_wrist[0] - shoulder_mid[0])
        right_extension = abs(right_wrist[0] - shoulder_mid[0])
    else:
        left_extension = abs(left_wrist[0])
        right_extension = abs(right_wrist[0])
    label = "fh_drive" if right_extension >= left_extension else "bh_drive"
    return {
        "type": label,
        "confidence": max(0.38, 0.62 * conf),
        "source": "semantic_wrist_extension",
        "player_id": player_id,
        "pose_frame_dt_s": round(frame_dt, 6),
        "semantic_joint_source": str(pose_payload.get("semantic_joint_source", "already_semantic")),
        "left_extension_m": round(left_extension, 6),
        "right_extension_m": round(right_extension, 6),
    }


def _nearest_pose_frame(
    *,
    pose_payload: Mapping[str, Any],
    contact_t: float,
    player_id: Any,
) -> tuple[int | None, Mapping[str, Any], float] | None:
    players = pose_payload.get("players", [])
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return None
    best: tuple[int | None, Mapping[str, Any], float] | None = None
    for player in players:
        if not isinstance(player, Mapping):
            continue
        current_id = player.get("id")
        if player_id is not None and current_id != player_id:
            continue
        frames = player.get("frames", [])
        if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
            continue
        for frame in frames:
            if not isinstance(frame, Mapping):
                continue
            dt = abs(_finite_or_zero(frame.get("t")) - contact_t)
            if best is None or dt < best[2]:
                best = (int(current_id) if current_id is not None else None, frame, dt)
    return best


def _track_ball_side_hint(
    *,
    contact: Mapping[str, Any],
    tracks: Sequence[dict[str, Any]],
    ball_frame: Mapping[str, Any],
    max_dt_s: float,
) -> dict[str, Any] | None:
    ball_xy = ball_frame.get("xy")
    if not _is_vector(ball_xy, length=2):
        return None
    best: dict[str, Any] | None = None
    for track in tracks:
        if contact.get("player_id") is not None and track["id"] != contact.get("player_id"):
            continue
        frame = _nearest_track_frame(float(contact["t"]), track["frames"], max_dt_s=max_dt_s)
        if frame is None:
            continue
        bbox = frame["bbox"]
        center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        distance = math.hypot(float(ball_xy[0]) - center[0], float(ball_xy[1]) - center[1])
        candidate = {
            "type": "fh_drive" if float(ball_xy[0]) >= center[0] else "bh_drive",
            "confidence": max(0.32, 0.52 * float(frame["conf"]) * float(ball_frame["conf"])),
            "source": "ball_track_bbox_side",
            "player_id": track["id"],
            "ball_track_dt_s": ball_frame["dt_s"],
            "track_frame_dt_s": frame["dt_s"],
            "ball_xy": [float(ball_xy[0]), float(ball_xy[1])],
            "bbox_center_xy": [round(center[0], 6), round(center[1], 6)],
            "distance_px": round(distance, 6),
        }
        if best is None or distance < float(best["distance_px"]):
            best = candidate
    return best


def _ball_track_image_side_hint(*, ball_frame: Mapping[str, Any], max_dt_s: float) -> dict[str, Any] | None:
    ball_xy = ball_frame.get("xy")
    if not _is_vector(ball_xy, length=2):
        return None
    image_center_x = _finite_or_zero(ball_frame.get("image_center_x"))
    dt_s = float(ball_frame["dt_s"])
    dt_penalty = min(0.18, dt_s / max(max_dt_s, 1e-6) * 0.18)
    confidence = max(0.20, min(0.36, (0.38 * float(ball_frame["conf"])) - dt_penalty))
    return {
        "type": "fh_drive" if float(ball_xy[0]) >= image_center_x else "bh_drive",
        "confidence": confidence,
        "source": "ball_track_image_side",
        "player_id": None,
        "ball_track_dt_s": ball_frame["dt_s"],
        "ball_xy": [float(ball_xy[0]), float(ball_xy[1])],
        "image_center_x": round(image_center_x, 6),
    }


def _track_players(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    players = payload.get("players", [])
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return []
    parsed = []
    for player in players:
        if not isinstance(player, Mapping):
            continue
        frames = []
        for frame in player.get("frames", []):
            if not isinstance(frame, Mapping):
                continue
            bbox = _vector(frame.get("bbox"), length=4)
            if bbox is None:
                continue
            frames.append(
                {
                    "t": _finite_or_zero(frame.get("t")),
                    "bbox": bbox,
                    "conf": _unit_interval(frame.get("conf", 0.0), "track.conf"),
                }
            )
        if frames:
            parsed.append({"id": player.get("id"), "frames": sorted(frames, key=lambda item: item["t"])})
    return parsed


def _ball_track_frames(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    frames = payload.get("frames", [])
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        return []
    parsed = []
    for frame in frames:
        if not isinstance(frame, Mapping) or not frame.get("visible", False):
            continue
        xy = _vector(frame.get("xy"), length=2)
        if xy is None:
            continue
        parsed.append(
            {
                "t": _require_non_negative(frame.get("t"), "ball_track.t"),
                "xy": xy,
                "conf": _unit_interval(frame.get("conf", 0.0), "ball_track.conf"),
            }
        )
    image_center_x = _image_center_x(payload, parsed)
    return sorted(({**frame, "image_center_x": image_center_x} for frame in parsed), key=lambda item: item["t"])


def _image_center_x(payload: Mapping[str, Any], frames: Sequence[Mapping[str, Any]]) -> float:
    for key in ("frame_width", "video_width", "width"):
        value = payload.get(key)
        if value is None:
            continue
        width = _finite_or_zero(value)
        if width > 0:
            return width / 2.0
    xs = [float(frame["xy"][0]) for frame in frames if _is_vector(frame.get("xy"), length=2)]
    if not xs:
        return 0.0
    return (min(xs) + max(xs)) / 2.0


def _nearest_ball_frame(contact_t: float, frames: Sequence[dict[str, Any]], *, max_dt_s: float) -> dict[str, Any] | None:
    best = None
    for frame in frames:
        dt = abs(float(frame["t"]) - contact_t)
        if dt > max_dt_s:
            continue
        candidate = {**frame, "dt_s": round(dt, 6)}
        if best is None or dt < float(best["dt_s"]):
            best = candidate
    return best


def _nearest_track_frame(contact_t: float, frames: Sequence[dict[str, Any]], *, max_dt_s: float) -> dict[str, Any] | None:
    best = None
    for frame in frames:
        dt = abs(float(frame["t"]) - contact_t)
        if dt > max_dt_s:
            continue
        candidate = {**frame, "dt_s": round(dt, 6)}
        if best is None or dt < float(best["dt_s"]):
            best = candidate
    return best


def _contact_evidence(contact: Mapping[str, Any]) -> dict[str, object]:
    return {
        "t": _round_time(contact["t"]),
        "frame": int(contact["frame"]),
        "confidence": float(contact["confidence"]),
        "sources": contact.get("sources", {}),
    }


def _inflection_evidence(match: Mapping[str, Any]) -> dict[str, object]:
    return {
        "time_s": _round_time(match["time_s"]),
        "frame": int(match.get("frame", 0)),
        "dt_s": float(match["dt_s"]),
        "ball_world_xyz": [round(float(value), 6) for value in match["ball_world_xyz"]],
        "confidence": float(match["confidence"]),
        "speed_before_mps": float(match["speed_before_mps"]),
        "speed_after_mps": float(match["speed_after_mps"]),
        "turn_angle_deg": float(match["turn_angle_deg"]),
        "approx": bool(match.get("approx", False)),
    }


def _fallback_evidence(fallback: Mapping[str, Any] | None) -> dict[str, object] | None:
    if fallback is None:
        return None
    return {
        key: value
        for key, value in fallback.items()
        if key not in {"type", "confidence"}
    }


def _rule_notes(match: Mapping[str, Any]) -> list[str]:
    xyz = match.get("ball_world_xyz") or [0.0, 0.0, 0.0]
    speed = max(float(match.get("speed_before_mps", 0.0)), float(match.get("speed_after_mps", 0.0)))
    notes = [f"max_ball_speed_mps={speed:.3f}", f"abs_court_y_m={abs(float(xyz[1])):.3f}"]
    if abs(float(xyz[1])) <= NVZ_HALF_DEPTH_M:
        notes.append("contact_near_nvz")
    return notes


def _shoulder_midpoint(joints: Sequence[Any], name_to_index: Mapping[str, int]) -> tuple[float, float, float] | None:
    left_index = name_to_index.get("leftshoulder")
    right_index = name_to_index.get("rightshoulder")
    if left_index is None or right_index is None or len(joints) <= max(left_index, right_index):
        return None
    left = _vector(joints[left_index], length=3)
    right = _vector(joints[right_index], length=3)
    if left is None or right is None:
        return None
    return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0, (left[2] + right[2]) / 2.0)


def _joint_confidence(frame: Mapping[str, Any], indexes: Sequence[int]) -> float:
    values = frame.get("joint_conf")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return 0.75
    confidences = []
    for index in indexes:
        if index < len(values):
            try:
                confidences.append(_unit_interval(values[index], "joint_conf"))
            except ValueError:
                continue
    if not confidences:
        return 0.75
    return sum(confidences) / len(confidences)


def _xy_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _normalize_name(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _is_vector(value: Any, *, length: int) -> bool:
    return _vector(value, length=length) is not None


def _vector(value: Any, *, length: int) -> tuple[float, ...] | None:
    if isinstance(value, (str, bytes)):
        return None
    try:
        vector = tuple(value)
    except TypeError:
        return None
    if len(vector) != length:
        return None
    try:
        return tuple(_finite(item, "vector") for item in vector)
    except ValueError:
        return None


def _require_non_negative(value: Any, name: str) -> float:
    number = _finite(value, name)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _require_non_negative_int(value: Any, name: str) -> int:
    number = _require_non_negative(value, name)
    if int(number) != number:
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _unit_interval(value: Any, name: str) -> float:
    number = _finite(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _finite_or_zero(value: Any) -> float:
    if value is None:
        return 0.0
    return _finite(value, "value")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _round_time(value: Any) -> float:
    return round(float(value), 6)


__all__ = ["CLASSIFIER_NAME", "classify_shots_from_payloads"]
