#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.doubles_id import assign_doubles_roles
from threed.racketsport.court_templates import get_court_template
from threed.racketsport.person_fast import PersonDetection, court_polygon_filter, person_detection_from_bbox
from threed.racketsport.schemas import CourtCalibration, PlayerTrack, TrackFrame, Tracks, validate_artifact_file
from threed.racketsport.track_lock import TrackCandidate, ground_step_plausible


IdStrategy = Literal["auto", "raw_track", "role_lock"]
RESOLVED_ID_STRATEGIES = {"raw_track", "role_lock"}


@dataclass(frozen=True)
class _FramePerson:
    frame_idx: int
    track_id: int
    person: PersonDetection


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_calibration(path: Path) -> CourtCalibration:
    parsed = validate_artifact_file("court_calibration", path)
    if not isinstance(parsed, CourtCalibration):
        raise ValueError("calibration artifact did not parse as CourtCalibration")
    return parsed


def _bbox_xyxy(detection: dict[str, Any]) -> tuple[float, float, float, float]:
    raw = detection.get("bbox") or detection.get("bbox_xyxy")
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        raise ValueError("detection bbox must contain four xyxy values")
    return tuple(float(value) for value in raw)  # type: ignore[return-value]


def _track_key(detection: dict[str, Any], fallback: int) -> int | str:
    for field in ("player_id", "track_id", "temp_track_id", "temp_id", "id"):
        value = detection.get(field)
        if value is not None:
            return int(value) if isinstance(value, int) or (isinstance(value, str) and value.isdigit()) else str(value)
    return fallback


def _int_track_id(key: int | str, mapping: dict[str, int], used_ids: set[int]) -> int:
    if isinstance(key, int):
        used_ids.add(key)
        return key
    if key not in mapping:
        next_id = 1
        while next_id in used_ids or next_id in mapping.values():
            next_id += 1
        mapping[key] = next_id
    return mapping[key]


def _frame_index(frame_entry: dict[str, Any], default: int) -> int:
    value = frame_entry.get("frame", frame_entry.get("frame_index", default))
    return int(value)


def _is_person_detection(detection: dict[str, Any]) -> bool:
    value = detection.get("class", "person")
    if value == 0:
        return True
    return str(value).lower() in {"person", "player", "0"}


def build_tracks(
    detections_payload: dict[str, Any],
    calibration: CourtCalibration,
    *,
    max_step_m: float,
    max_players: int = 4,
    court_margin_m: float = 0.0,
    id_strategy: IdStrategy = "raw_track",
) -> tuple[Tracks, dict[str, int]]:
    _validate_max_players(max_players)
    if court_margin_m < 0.0:
        raise ValueError("court_margin_m must be non-negative")
    if id_strategy not in {"auto", "raw_track", "role_lock"}:
        raise ValueError("id_strategy must be auto, raw_track, or role_lock")
    resolved_id_strategy = _resolve_id_strategy(detections_payload, id_strategy=id_strategy)
    fps = float(detections_payload["fps"])
    frames = detections_payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("detections payload must contain a frames list")
    if resolved_id_strategy == "role_lock":
        return _build_role_locked_tracks(
            frames,
            calibration,
            fps=fps,
            max_players=max_players,
            court_margin_m=court_margin_m,
            requested_id_strategy=id_strategy,
        )

    by_player: dict[int, list[TrackFrame]] = defaultdict(list)
    last_world_xy: dict[int, list[float]] = {}
    string_ids: dict[str, int] = {}
    used_ids: set[int] = set()
    counts = {
        "accepted": 0,
        "accepted_before_player_cap": 0,
        "outside_court": 0,
        "implausible_step": 0,
        "non_person": 0,
        "candidate_players": 0,
        "output_players": 0,
        "max_players": max_players,
        "court_margin_m": court_margin_m,
        "id_strategy": resolved_id_strategy,
        "requested_id_strategy": id_strategy,
        "extra_players_dropped": 0,
        "extra_player_frames_dropped": 0,
    }

    for default_frame_idx, frame_entry in enumerate(frames):
        if not isinstance(frame_entry, dict):
            raise ValueError("each frame entry must be an object")
        frame_idx = _frame_index(frame_entry, default_frame_idx)
        frame_detections = frame_entry.get("detections", [])
        if not isinstance(frame_detections, list):
            raise ValueError("frame detections must be a list")

        for det_idx, detection in enumerate(frame_detections):
            if not isinstance(detection, dict):
                raise ValueError("each detection must be an object")
            if not _is_person_detection(detection):
                counts["non_person"] += 1
                continue

            bbox = _bbox_xyxy(detection)
            confidence = float(detection.get("conf", detection.get("confidence", 1.0)))
            person = person_detection_from_bbox(calibration, bbox_xyxy=bbox, confidence=confidence)
            if not court_polygon_filter([person], sport=calibration.sport, margin_m=court_margin_m):
                counts["outside_court"] += 1
                continue

            player_id = _int_track_id(_track_key(detection, det_idx + 1), string_ids, used_ids)
            previous = last_world_xy.get(player_id)
            if previous is not None and not ground_step_plausible(
                previous,
                person.foot_world_xy,
                max_step_m=max_step_m * max(1, frame_idx - int(round(by_player[player_id][-1].t * fps))),
            ):
                counts["implausible_step"] += 1
                continue

            by_player[player_id].append(
                TrackFrame(
                    t=frame_idx / fps,
                    bbox=person.bbox_xyxy,
                    world_xy=person.foot_world_xy,
                    conf=person.confidence,
                )
            )
            last_world_xy[player_id] = person.foot_world_xy
            counts["accepted_before_player_cap"] += 1

    selected_ids = _select_stable_player_ids(by_player, max_players=max_players)
    selected_id_set = set(selected_ids)
    counts["candidate_players"] = len(by_player)
    counts["output_players"] = len(selected_ids)
    counts["extra_players_dropped"] = max(0, len(by_player) - len(selected_ids))
    counts["extra_player_frames_dropped"] = sum(
        len(frames) for player_id, frames in by_player.items() if player_id not in selected_id_set
    )
    counts["accepted"] = sum(len(by_player[player_id]) for player_id in selected_ids)

    identities = _identity_labels(by_player, selected_ids=selected_ids, max_players=max_players)
    players = []
    for player_id in sorted(selected_ids):
        frames = by_player[player_id]
        identity = identities[player_id]
        players.append(
            PlayerTrack(
                id=player_id,
                side=identity["side"],
                role=identity["role"],
                frames=frames,
            )
        )
    return Tracks(schema_version=1, fps=fps, players=players, rally_spans=[]), counts


def _build_role_locked_tracks(
    frame_entries: list[Any],
    calibration: CourtCalibration,
    *,
    fps: float,
    max_players: int,
    court_margin_m: float,
    requested_id_strategy: IdStrategy = "role_lock",
) -> tuple[Tracks, dict[str, Any]]:
    counts: dict[str, Any] = {
        "accepted": 0,
        "accepted_before_player_cap": 0,
        "outside_court": 0,
        "implausible_step": 0,
        "non_person": 0,
        "candidate_players": 0,
        "output_players": 0,
        "max_players": max_players,
        "court_margin_m": court_margin_m,
        "id_strategy": "role_lock",
        "requested_id_strategy": requested_id_strategy,
        "extra_players_dropped": 0,
        "extra_player_frames_dropped": 0,
    }
    by_role: dict[int, list[TrackFrame]] = defaultdict(list)
    raw_ids: set[int] = set()

    for default_frame_idx, frame_entry in enumerate(frame_entries):
        frame_people = _accepted_frame_people(
            frame_entry,
            default_frame_idx=default_frame_idx,
            calibration=calibration,
            court_margin_m=court_margin_m,
            raw_ids=raw_ids,
            counts=counts,
        )
        frame_people = _nms_frame_people(frame_people, iou_threshold=0.85)
        assignments = _assign_role_locked_frame(
            frame_people,
            calibration=calibration,
            max_players=max_players,
        )
        for role_id, frame_person in assignments:
            person = frame_person.person
            by_role[role_id].append(
                TrackFrame(
                    t=frame_person.frame_idx / fps,
                    bbox=person.bbox_xyxy,
                    world_xy=person.foot_world_xy,
                    conf=person.confidence,
                )
            )

    identities = _role_lock_identities(max_players=max_players)
    players = [
        PlayerTrack(id=role_id, side=identities[role_id]["side"], role=identities[role_id]["role"], frames=frames)
        for role_id, frames in sorted(by_role.items())
        if frames
    ]
    counts["candidate_players"] = len(raw_ids)
    counts["output_players"] = len(players)
    counts["accepted"] = sum(len(player.frames) for player in players)
    counts["extra_players_dropped"] = max(0, counts["candidate_players"] - len(players))
    counts["extra_player_frames_dropped"] = max(0, counts["accepted_before_player_cap"] - counts["accepted"])
    return Tracks(schema_version=1, fps=fps, players=players, rally_spans=[]), counts


def _accepted_frame_people(
    frame_entry: Any,
    *,
    default_frame_idx: int,
    calibration: CourtCalibration,
    court_margin_m: float,
    raw_ids: set[int],
    counts: dict[str, Any],
) -> list[_FramePerson]:
    if not isinstance(frame_entry, dict):
        raise ValueError("each frame entry must be an object")
    frame_idx = _frame_index(frame_entry, default_frame_idx)
    frame_detections = frame_entry.get("detections", [])
    if not isinstance(frame_detections, list):
        raise ValueError("frame detections must be a list")

    accepted: list[_FramePerson] = []
    string_ids: dict[str, int] = {}
    used_ids: set[int] = set()
    for det_idx, detection in enumerate(frame_detections):
        if not isinstance(detection, dict):
            raise ValueError("each detection must be an object")
        if not _is_person_detection(detection):
            counts["non_person"] += 1
            continue
        bbox = _bbox_xyxy(detection)
        confidence = float(detection.get("conf", detection.get("confidence", 1.0)))
        person = person_detection_from_bbox(calibration, bbox_xyxy=bbox, confidence=confidence)
        if not court_polygon_filter([person], sport=calibration.sport, margin_m=court_margin_m):
            counts["outside_court"] += 1
            continue
        track_id = _int_track_id(_track_key(detection, det_idx + 1), string_ids, used_ids)
        raw_ids.add(track_id)
        counts["accepted_before_player_cap"] += 1
        accepted.append(_FramePerson(frame_idx=frame_idx, track_id=track_id, person=person))
    return accepted


def _resolve_id_strategy(detections_payload: dict[str, Any], *, id_strategy: IdStrategy) -> Literal["raw_track", "role_lock"]:
    if id_strategy in RESOLVED_ID_STRATEGIES:
        return id_strategy
    if detections_payload.get("source") == "player_labels" and not _payload_has_tracker_ids(detections_payload):
        return "role_lock"
    return "raw_track"


def _payload_has_tracker_ids(detections_payload: dict[str, Any]) -> bool:
    frames = detections_payload.get("frames")
    if not isinstance(frames, list):
        return False
    for frame_entry in frames:
        if not isinstance(frame_entry, dict):
            continue
        detections = frame_entry.get("detections")
        if not isinstance(detections, list):
            continue
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            if any(detection.get(field) is not None for field in ("player_id", "track_id", "temp_track_id", "temp_id", "id")):
                return True
    return False


def _assign_role_locked_frame(
    frame_people: list[_FramePerson],
    *,
    calibration: CourtCalibration,
    max_players: int,
) -> list[tuple[int, _FramePerson]]:
    if not frame_people:
        return []
    slots = _role_slots(calibration, max_players=max_players)
    candidates = sorted(frame_people, key=lambda item: item.person.confidence, reverse=True)[:8]
    if len(candidates) >= len(slots):
        selected = candidates[: len(slots)]
        best_slots: tuple[float, tuple[dict[str, Any], ...]] | None = None
        for slot_order in itertools.permutations(slots, len(selected)):
            cost = sum(_assignment_cost(slot["xy"], item) for slot, item in zip(slot_order, selected, strict=True))
            if best_slots is None or cost < best_slots[0]:
                best_slots = (cost, slot_order)
        assert best_slots is not None
        return sorted(
            [(int(slot["id"]), item) for slot, item in zip(best_slots[1], selected, strict=True)],
            key=lambda item: item[0],
        )

    best_partial: tuple[float, tuple[dict[str, Any], ...]] | None = None
    for slot_order in itertools.permutations(slots, len(candidates)):
        cost = sum(_assignment_cost(slot["xy"], item) for slot, item in zip(slot_order, candidates, strict=True))
        if best_partial is None or cost < best_partial[0]:
            best_partial = (cost, slot_order)
    assert best_partial is not None
    return sorted(
        [(int(slot["id"]), item) for slot, item in zip(best_partial[1], candidates, strict=True)],
        key=lambda item: item[0],
    )


def _role_slots(calibration: CourtCalibration, *, max_players: int) -> list[dict[str, Any]]:
    template = get_court_template(calibration.sport)
    if max_players == 2:
        return [
            {"id": 1, "side": "near", "role": "singles", "xy": [0.0, -0.45 * template.length_m / 2.0]},
            {"id": 2, "side": "far", "role": "singles", "xy": [0.0, 0.45 * template.length_m / 2.0]},
        ]
    x_anchor = 0.45 * template.width_m / 2.0
    y_anchor = 0.45 * template.length_m / 2.0
    return [
        {"id": 1, "side": "near", "role": "left", "xy": [-x_anchor, -y_anchor]},
        {"id": 2, "side": "near", "role": "right", "xy": [x_anchor, -y_anchor]},
        {"id": 3, "side": "far", "role": "left", "xy": [-x_anchor, y_anchor]},
        {"id": 4, "side": "far", "role": "right", "xy": [x_anchor, y_anchor]},
    ]


def _role_lock_identities(*, max_players: int) -> dict[int, dict[str, str]]:
    return {
        int(slot["id"]): {"side": str(slot["side"]), "role": str(slot["role"])}
        for slot in _role_slots_for_identity(max_players=max_players)
    }


def _role_slots_for_identity(*, max_players: int) -> list[dict[str, Any]]:
    if max_players == 2:
        return [
            {"id": 1, "side": "near", "role": "singles"},
            {"id": 2, "side": "far", "role": "singles"},
        ]
    return [
        {"id": 1, "side": "near", "role": "left"},
        {"id": 2, "side": "near", "role": "right"},
        {"id": 3, "side": "far", "role": "left"},
        {"id": 4, "side": "far", "role": "right"},
    ]


def _assignment_cost(slot_xy: list[float], frame_person: _FramePerson) -> float:
    x, y = frame_person.person.foot_world_xy
    sx, sy = slot_xy
    distance = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
    return distance - (0.35 * frame_person.person.confidence)


def _nms_frame_people(frame_people: list[_FramePerson], *, iou_threshold: float) -> list[_FramePerson]:
    kept: list[_FramePerson] = []
    for candidate in sorted(frame_people, key=lambda item: item.person.confidence, reverse=True):
        if all(_bbox_iou(candidate.person.bbox_xyxy, other.person.bbox_xyxy) <= iou_threshold for other in kept):
            kept.append(candidate)
    return kept


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def _validate_max_players(max_players: int) -> None:
    if max_players <= 0:
        raise ValueError("max_players must be positive")
    if max_players > 4:
        raise ValueError("max_players must be <= 4")


def _select_stable_player_ids(by_player: dict[int, list[TrackFrame]], *, max_players: int) -> list[int]:
    ranked = sorted(
        ((player_id, frames) for player_id, frames in by_player.items() if frames),
        key=lambda item: (
            -len(item[1]),
            -sum(frame.conf for frame in item[1]) / len(item[1]),
            item[1][0].t,
            item[0],
        ),
    )
    return [player_id for player_id, _frames in ranked[:max_players]]


def _identity_labels(
    by_player: dict[int, list[TrackFrame]],
    *,
    selected_ids: list[int],
    max_players: int,
) -> dict[int, dict[str, str]]:
    candidates = [
        TrackCandidate(
            track_id=player_id,
            world_xy=list(by_player[player_id][0].world_xy),
            confidence=sum(frame.conf for frame in by_player[player_id]) / len(by_player[player_id]),
        )
        for player_id in selected_ids
    ]
    if max_players == 2:
        return {
            candidate.track_id: {
                "side": "near" if candidate.world_xy[1] <= 0.0 else "far",
                "role": "singles",
            }
            for candidate in candidates
        }

    doubles = assign_doubles_roles(candidates)
    return {
        track_id: {
            "side": identity.side,
            "role": identity.role,
        }
        for track_id, identity in doubles.items()
    }


def _write_tracks(path: Path, tracks: Tracks) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tracks.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_inputs(detections_path: Path, calibration_path: Path) -> list[str]:
    errors: list[str] = []
    if not detections_path.exists():
        errors.append(f"missing detections file: {detections_path}")
    if not calibration_path.exists():
        errors.append(f"missing calibration file: {calibration_path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert precomputed person bbox detections into tracks.json.")
    parser.add_argument("--detections", type=Path, required=True, help="Input JSON with fps and per-frame detections.")
    parser.add_argument("--calibration", type=Path, required=True, help="court_calibration.json artifact.")
    parser.add_argument("--out", type=Path, required=True, help="Output tracks.json path.")
    parser.add_argument("--max-step-m", type=float, default=2.0, help="Maximum plausible ground step per frame.")
    parser.add_argument(
        "--court-margin-m",
        type=float,
        default=0.0,
        help="Runoff margin around the regulation court footprint for accepting player footpoints.",
    )
    parser.add_argument(
        "--id-strategy",
        choices=("auto", "raw_track", "role_lock"),
        default="auto",
        help=(
            "auto role-locks prototype player-label detections without tracker IDs and otherwise keeps raw tracker IDs; "
            "raw_track keeps tracker IDs; role_lock assigns stable logical near/far left/right player IDs per frame."
        ),
    )
    parser.add_argument(
        "--max-players",
        type=int,
        choices=(2, 4),
        default=4,
        help="Maximum on-court player identities to keep: 2 for singles, 4 for doubles.",
    )
    args = parser.parse_args()

    input_errors = _validate_inputs(args.detections, args.calibration)
    if input_errors:
        print("; ".join(input_errors), file=sys.stderr)
        return 2

    try:
        calibration = _load_calibration(args.calibration)
        tracks, counts = build_tracks(
            _read_json(args.detections),
            calibration,
            max_step_m=args.max_step_m,
            max_players=args.max_players,
            court_margin_m=args.court_margin_m,
            id_strategy=args.id_strategy,
        )
        _write_tracks(args.out, tracks)
    except Exception as exc:
        print(f"track conversion failed: {exc}", file=sys.stderr)
        return 1

    print(
        "track conversion: "
        f"accepted={counts['accepted']} "
        f"outside_court={counts['outside_court']} "
        f"implausible_step={counts['implausible_step']} "
        f"non_person={counts['non_person']} "
        f"max_players={counts['max_players']} "
        f"court_margin_m={counts['court_margin_m']} "
        f"id_strategy={counts['id_strategy']} "
        f"output_players={counts['output_players']} "
        f"extra_players_dropped={counts['extra_players_dropped']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
