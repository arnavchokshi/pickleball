"""CPU-only wrist velocity cue generation from existing world-joint artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from threed.racketsport.skeleton3d import semanticize_skeleton_payload

ARTIFACT_TYPE = "racketsport_wrist_velocity_peaks"
DEFAULT_MIN_SPEED_MPS = 4.0
DEFAULT_MIN_CONFIDENCE = 0.25
DEFAULT_MIN_SEPARATION_S = 0.10
WRIST_ALIASES = {
    "left_wrist": {"left_wrist", "lwrist", "l_wrist", "left_hand", "lhand"},
    "right_wrist": {"right_wrist", "rwrist", "r_wrist", "right_hand", "rhand"},
}


def build_wrist_velocity_peaks_from_file(
    skeleton3d_path: str | Path,
    *,
    min_speed_mps: float = DEFAULT_MIN_SPEED_MPS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_separation_s: float = DEFAULT_MIN_SEPARATION_S,
    left_wrist_index: int | None = None,
    right_wrist_index: int | None = None,
) -> dict[str, Any]:
    path = Path(skeleton3d_path)
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("skeleton3d.json must contain an object")
    return build_wrist_velocity_peaks_from_skeleton(
        payload,
        source_path=path,
        min_speed_mps=min_speed_mps,
        min_confidence=min_confidence,
        min_separation_s=min_separation_s,
        left_wrist_index=left_wrist_index,
        right_wrist_index=right_wrist_index,
    )


def build_blocked_wrist_velocity_peaks(
    *,
    source_path: str | Path,
    blocker: str = "missing_skeleton3d",
    min_speed_mps: float = DEFAULT_MIN_SPEED_MPS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_separation_s: float = DEFAULT_MIN_SEPARATION_S,
) -> dict[str, Any]:
    return _artifact(
        status="blocked",
        source_path=source_path,
        joint_mapping={},
        player_count=0,
        usable_sample_count=0,
        raw_peak_count=0,
        peaks=[],
        blockers=[blocker],
        warnings=[blocker],
        min_speed_mps=_require_non_negative(min_speed_mps, "min_speed_mps"),
        min_confidence=_require_unit(min_confidence, "min_confidence"),
        min_separation_s=_require_non_negative(min_separation_s, "min_separation_s"),
    )


def build_wrist_velocity_peaks_from_skeleton(
    skeleton: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
    min_speed_mps: float = DEFAULT_MIN_SPEED_MPS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_separation_s: float = DEFAULT_MIN_SEPARATION_S,
    left_wrist_index: int | None = None,
    right_wrist_index: int | None = None,
) -> dict[str, Any]:
    """Build review-only wrist-velocity peaks from timestamped world joints."""

    min_speed_mps = _require_non_negative(min_speed_mps, "min_speed_mps")
    min_confidence = _require_unit(min_confidence, "min_confidence")
    min_separation_s = _require_non_negative(min_separation_s, "min_separation_s")
    if left_wrist_index is None and right_wrist_index is None:
        semantic_skeleton = semanticize_skeleton_payload(skeleton)
        if semantic_skeleton is not None:
            skeleton = semantic_skeleton
    players = skeleton.get("players")
    if not isinstance(players, list):
        raise ValueError("skeleton3d players must be a list")
    joint_mapping = _joint_mapping(
        skeleton.get("joint_names"),
        left_wrist_index=left_wrist_index,
        right_wrist_index=right_wrist_index,
    )
    if not joint_mapping:
        return _artifact(
            status="blocked",
            source_path=source_path,
            joint_mapping={},
            player_count=len(players),
            usable_sample_count=0,
            raw_peak_count=0,
            peaks=[],
            blockers=["missing_wrist_joint_mapping"],
            warnings=["missing_wrist_joint_mapping"],
            min_speed_mps=min_speed_mps,
            min_confidence=min_confidence,
            min_separation_s=min_separation_s,
        )

    raw_peaks: list[dict[str, Any]] = []
    usable_sample_count = 0
    for player in players:
        if not isinstance(player, Mapping):
            continue
        player_id = int(player.get("id", 0))
        frames = player.get("frames")
        if not isinstance(frames, list):
            continue
        for joint_name, joint_index in joint_mapping.items():
            side = "left" if joint_name == "left_wrist" else "right"
            samples = _usable_wrist_samples(
                frames,
                joint_index=joint_index,
                min_confidence=min_confidence,
            )
            usable_sample_count += len(samples)
            for previous, current, following in zip(samples, samples[1:], samples[2:]):
                dt_before = current["time_s"] - previous["time_s"]
                dt_after = following["time_s"] - current["time_s"]
                if dt_before <= 0.0 or dt_after <= 0.0:
                    continue
                speed_before = _distance(previous["point"], current["point"]) / dt_before
                speed_after = _distance(current["point"], following["point"]) / dt_after
                speed = max(speed_before, speed_after)
                if speed < min_speed_mps:
                    continue
                confidence = min(float(previous["confidence"]), float(current["confidence"]), float(following["confidence"]))
                raw_peaks.append(
                    {
                        "time_s": current["time_s"],
                        "frame": current.get("frame"),
                        "player_id": player_id,
                        "wrist_side": side,
                        "wrist_world_xyz": [round(float(value), 9) for value in current["point"]],
                        "speed_mps": round(speed, 6),
                        "confidence": round(confidence, 6),
                        "source": "neighbor_segment_world_joints",
                    }
                )

    peaks = _suppress_nearby_peaks(raw_peaks, min_separation_s=min_separation_s)
    blockers = [] if peaks else ["insufficient_wrist_velocity_peaks"]
    warnings = ["review_only_not_gate_verified"]
    if blockers:
        warnings.extend(blockers)
    return _artifact(
        status="review_only" if peaks else "blocked",
        source_path=source_path,
        joint_mapping=joint_mapping,
        player_count=len(players),
        usable_sample_count=usable_sample_count,
        raw_peak_count=len(raw_peaks),
        peaks=peaks,
        blockers=blockers,
        warnings=warnings,
        min_speed_mps=min_speed_mps,
        min_confidence=min_confidence,
        min_separation_s=min_separation_s,
    )


def write_wrist_velocity_peaks(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(
    *,
    status: str,
    source_path: str | Path | None,
    joint_mapping: Mapping[str, int],
    player_count: int,
    usable_sample_count: int,
    raw_peak_count: int,
    peaks: list[dict[str, Any]],
    blockers: list[str],
    warnings: list[str],
    min_speed_mps: float,
    min_confidence: float,
    min_separation_s: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "status": status,
        "source": "skeleton3d_world_joints",
        "source_path": str(source_path) if source_path is not None else "",
        "not_gate_verified": True,
        "trusted_for_contact": False,
        "joint_mapping": dict(joint_mapping),
        "blockers": blockers,
        "warnings": warnings,
        "summary": {
            "player_count": player_count,
            "usable_sample_count": usable_sample_count,
            "raw_peak_count": raw_peak_count,
            "peak_count": len(peaks),
            "min_speed_mps": min_speed_mps,
            "min_confidence": min_confidence,
            "min_separation_s": min_separation_s,
        },
        "peaks": peaks,
    }


def _joint_mapping(
    joint_names: Any,
    *,
    left_wrist_index: int | None,
    right_wrist_index: int | None,
) -> dict[str, int]:
    mapping: dict[str, int] = {}
    if left_wrist_index is not None:
        mapping["left_wrist"] = _non_negative_index(left_wrist_index, "left_wrist_index")
    if right_wrist_index is not None:
        mapping["right_wrist"] = _non_negative_index(right_wrist_index, "right_wrist_index")
    if mapping:
        return mapping

    if not isinstance(joint_names, list):
        return {}
    normalized_names = [_normalize_joint_name(name) for name in joint_names]
    for target, aliases in WRIST_ALIASES.items():
        for index, normalized in enumerate(normalized_names):
            if normalized in aliases:
                mapping[target] = index
                break
    return mapping


def _usable_wrist_samples(
    frames: Sequence[Any],
    *,
    joint_index: int,
    min_confidence: float,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            continue
        joints = frame.get("joints_world")
        joint_conf = frame.get("joint_conf")
        if not isinstance(joints, list) or joint_index >= len(joints):
            continue
        if not isinstance(joint_conf, list) or joint_index >= len(joint_conf):
            continue
        try:
            confidence = _require_unit(joint_conf[joint_index], "joint_conf")
            if confidence < min_confidence:
                continue
            point = _vector3(joints[joint_index], "joints_world")
            time_s = _require_non_negative(frame.get("t", frame.get("time_s")), "t")
        except ValueError:
            continue
        samples.append(
            {
                "time_s": time_s,
                "frame": frame.get("frame", frame_index),
                "point": point,
                "confidence": confidence,
            }
        )
    return sorted(samples, key=lambda sample: (sample["time_s"], sample["frame"]))


def _suppress_nearby_peaks(candidates: list[dict[str, Any]], *, min_separation_s: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (float(item["time_s"]), int(item["player_id"]))):
        if not kept or float(candidate["time_s"]) - float(kept[-1]["time_s"]) >= min_separation_s:
            kept.append(candidate)
            continue
        if _rank(candidate) > _rank(kept[-1]):
            kept[-1] = candidate
    return kept


def _rank(candidate: Mapping[str, Any]) -> tuple[float, float]:
    return float(candidate.get("speed_mps", 0.0)), float(candidate.get("confidence", 0.0))


def _normalize_joint_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _vector3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-vector")
    return tuple(_require_finite(component, name) for component in value)


def _non_negative_index(value: int, name: str) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_unit(value: Any, name: str) -> float:
    value = _require_finite(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _require_non_negative(value: Any, name: str) -> float:
    value = _require_finite(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_finite(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


__all__ = [
    "ARTIFACT_TYPE",
    "build_blocked_wrist_velocity_peaks",
    "build_wrist_velocity_peaks_from_file",
    "build_wrist_velocity_peaks_from_skeleton",
    "write_wrist_velocity_peaks",
]
