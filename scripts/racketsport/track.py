#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threed.racketsport.person_fast import court_polygon_filter, person_detection_from_bbox
from threed.racketsport.schemas import CourtCalibration, PlayerTrack, TrackFrame, Tracks, validate_artifact_file
from threed.racketsport.track_lock import ground_step_plausible


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
) -> tuple[Tracks, dict[str, int]]:
    fps = float(detections_payload["fps"])
    by_player: dict[int, list[TrackFrame]] = defaultdict(list)
    last_world_xy: dict[int, list[float]] = {}
    string_ids: dict[str, int] = {}
    used_ids: set[int] = set()
    counts = {"accepted": 0, "outside_court": 0, "implausible_step": 0, "non_person": 0}

    frames = detections_payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("detections payload must contain a frames list")

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
            if not court_polygon_filter([person], sport=calibration.sport):
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
            counts["accepted"] += 1

    players = [
        PlayerTrack(
            id=player_id,
            side="near" if frames[0].world_xy[1] <= 0.0 else "far",
            role="tracked",
            frames=frames,
        )
        for player_id, frames in sorted(by_player.items())
        if frames
    ]
    return Tracks(schema_version=1, fps=fps, players=players, rally_spans=[]), counts


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
    args = parser.parse_args()

    input_errors = _validate_inputs(args.detections, args.calibration)
    if input_errors:
        print("; ".join(input_errors), file=sys.stderr)
        return 2

    try:
        calibration = _load_calibration(args.calibration)
        tracks, counts = build_tracks(_read_json(args.detections), calibration, max_step_m=args.max_step_m)
        _write_tracks(args.out, tracks)
    except Exception as exc:
        print(f"track conversion failed: {exc}", file=sys.stderr)
        return 1

    print(
        "track conversion: "
        f"accepted={counts['accepted']} "
        f"outside_court={counts['outside_court']} "
        f"implausible_step={counts['implausible_step']} "
        f"non_person={counts['non_person']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
