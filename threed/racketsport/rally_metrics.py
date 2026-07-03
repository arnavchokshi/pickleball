"""Per-rally positional facts for scrubber coaching cards."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import ijson
except ImportError:  # pragma: no cover - local env has ijson; fallback is for small fixtures.
    ijson = None  # type: ignore[assignment]


ARTIFACT_TYPE = "rally_metrics"
FACTS_ARTIFACT_TYPE = "coaching_card_facts"
BASELINE_BAND_M = 1.5
KITCHEN_PROXIMITY_M = 0.5
MIN_OK_COVERAGE = 0.8


@dataclass(frozen=True)
class PlayerFrame:
    player_id: str
    frame_index: int
    t: float
    track_world_xy: tuple[float, float]
    estimated_input: bool = False


@dataclass(frozen=True)
class PlayerTrack:
    player_id: str
    frames: tuple[PlayerFrame, ...]


@dataclass(frozen=True)
class WorldTrackData:
    fps: float
    ball_frame_count: int | None
    players: tuple[PlayerTrack, ...]


@dataclass(frozen=True)
class RallySpan:
    rally_id: str
    t0: float
    t1: float
    scope: str
    notes: tuple[str, ...] = ()


@dataclass
class _FrameBuilder:
    frame_index: int
    t: float | None = None
    track_world_xy: list[float] = field(default_factory=list)
    trust_tokens: list[str] = field(default_factory=list)


@dataclass
class _PlayerBuilder:
    ordinal: int
    player_id: str | None = None
    frames: list[_FrameBuilder] = field(default_factory=list)


def build_rally_metrics(run_dir: Path | str) -> dict[str, Any]:
    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"run dir does not exist: {run_path}")
    if not run_path.is_dir():
        raise NotADirectoryError(f"run dir is not a directory: {run_path}")

    world = read_virtual_world_tracks(run_path / "virtual_world.json")
    court_zones = _load_court_zones(run_path / "court_zones.json")
    spans, rally_scope = _load_rally_spans(run_path / "rally_spans.json", world=world)
    contacts = _load_contact_events(run_path / "contact_windows.json")

    rally_payloads = []
    for span in spans:
        rally_payloads.append(_rally_payload(span, world=world, court_zones=court_zones, contacts=contacts))

    facts = _coaching_card_facts(rally_payloads, source_run_dir=str(run_path), rally_scope=rally_scope)
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "source_run_dir": str(run_path),
        "rally_scope": rally_scope,
        "policy": {
            "position_based_only": True,
            "pose_biomechanics_used": False,
            "protected_eval_labels_read": False,
            "gap_aware_speed": True,
            "max_integrated_gap_s": _max_gap_s(world.fps),
        },
        "inputs": {
            "virtual_world": str(run_path / "virtual_world.json"),
            "rally_spans": str(run_path / "rally_spans.json") if (run_path / "rally_spans.json").exists() else None,
            "court_zones": str(run_path / "court_zones.json"),
            "contact_windows": str(run_path / "contact_windows.json") if (run_path / "contact_windows.json").exists() else None,
            "foot_contact_phases": str(run_path / "foot_contact_phases.json")
            if (run_path / "foot_contact_phases.json").exists()
            else None,
        },
        "fps": world.fps,
        "player_count": len(world.players),
        "rallies": rally_payloads,
        "coaching_card_facts": facts,
    }


def read_virtual_world_tracks(path: Path) -> WorldTrackData:
    if not path.exists():
        raise FileNotFoundError(f"virtual_world.json not found: {path}")
    if ijson is None:
        return _read_virtual_world_tracks_small_json(path)
    return _read_virtual_world_tracks_stream(path)


def _read_virtual_world_tracks_stream(path: Path) -> WorldTrackData:
    fps = 30.0
    ball_frame_count: int | None = None
    ball_frame_counter = 0
    players: list[PlayerTrack] = []
    current_player: _PlayerBuilder | None = None
    current_frame: _FrameBuilder | None = None
    player_ordinal = 0

    with path.open("rb") as handle:
        for prefix, event, value in ijson.parse(handle):
            if prefix == "fps" and event == "number":
                fps = float(value)
            elif prefix == "summary.ball_frame_count" and event == "number":
                ball_frame_count = int(value)
            elif prefix == "ball.frames.item" and event == "start_map":
                ball_frame_counter += 1
            elif prefix == "players.item" and event == "start_map":
                current_player = _PlayerBuilder(ordinal=player_ordinal)
                player_ordinal += 1
            elif prefix == "players.item" and event == "end_map":
                if current_player is not None:
                    players.append(_finalize_player(current_player))
                current_player = None
            elif current_player is not None and prefix == "players.item.id" and event in {"number", "string"}:
                current_player.player_id = _id_string(value)
            elif current_player is not None and prefix == "players.item.frames.item" and event == "start_map":
                current_frame = _FrameBuilder(frame_index=len(current_player.frames))
            elif (
                current_player is not None
                and current_frame is not None
                and prefix == "players.item.frames.item"
                and event == "end_map"
            ):
                current_player.frames.append(current_frame)
                current_frame = None
            elif current_frame is not None and prefix == "players.item.frames.item.t" and event == "number":
                current_frame.t = float(value)
            elif (
                current_frame is not None
                and prefix == "players.item.frames.item.track_world_xy.item"
                and event == "number"
            ):
                if len(current_frame.track_world_xy) < 2:
                    current_frame.track_world_xy.append(float(value))
            elif current_frame is not None and _is_frame_trust_prefix(prefix) and event in {
                "string",
                "number",
                "boolean",
            }:
                current_frame.trust_tokens.append(str(value))

    if ball_frame_count is None and ball_frame_counter:
        ball_frame_count = ball_frame_counter
    return WorldTrackData(fps=fps, ball_frame_count=ball_frame_count, players=tuple(players))


def _read_virtual_world_tracks_small_json(path: Path) -> WorldTrackData:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fps = float(payload.get("fps", 30.0))
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    ball_frame_count = _optional_int(summary.get("ball_frame_count"))
    players = []
    for ordinal, player in enumerate(payload.get("players", [])):
        if not isinstance(player, Mapping):
            continue
        builder = _PlayerBuilder(ordinal=ordinal, player_id=_id_string(player.get("id", ordinal)))
        for index, frame in enumerate(player.get("frames", [])):
            if not isinstance(frame, Mapping):
                continue
            frame_builder = _FrameBuilder(frame_index=index)
            if frame.get("t") is not None:
                frame_builder.t = float(frame["t"])
            xy = frame.get("track_world_xy")
            if isinstance(xy, Sequence) and len(xy) >= 2:
                frame_builder.track_world_xy = [float(xy[0]), float(xy[1])]
            frame_builder.trust_tokens.extend(_trust_tokens_from_mapping(frame))
            builder.frames.append(frame_builder)
        players.append(_finalize_player(builder))
    return WorldTrackData(fps=fps, ball_frame_count=ball_frame_count, players=tuple(players))


def _finalize_player(builder: _PlayerBuilder) -> PlayerTrack:
    player_id = builder.player_id if builder.player_id is not None else str(builder.ordinal)
    frames: list[PlayerFrame] = []
    for frame in builder.frames:
        if len(frame.track_world_xy) < 2:
            raise ValueError(f"missing required field players[{player_id}].frames[{frame.frame_index}].track_world_xy")
        if frame.t is None:
            raise ValueError(f"missing required field players[{player_id}].frames[{frame.frame_index}].t")
        frames.append(
            PlayerFrame(
                player_id=player_id,
                frame_index=frame.frame_index,
                t=frame.t,
                track_world_xy=(frame.track_world_xy[0], frame.track_world_xy[1]),
                estimated_input=_tokens_are_estimated(frame.trust_tokens),
            )
        )
    return PlayerTrack(player_id=player_id, frames=tuple(frames))


def _is_frame_trust_prefix(prefix: str) -> bool:
    return prefix.startswith("players.item.frames.item.trust_band.") or prefix.startswith(
        "players.item.frames.item.confidence_provenance."
    )


def _trust_tokens_from_mapping(frame: Mapping[str, Any]) -> list[str]:
    tokens: list[str] = []
    for key in ("trust_band", "confidence_provenance"):
        value = frame.get(key)
        if isinstance(value, Mapping):
            tokens.extend(str(item) for item in _walk_scalars(value))
    return tokens


def _walk_scalars(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_scalars(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_scalars(item)
    elif value is not None:
        yield value


def _tokens_are_estimated(tokens: Sequence[str]) -> bool:
    joined = " ".join(token.lower() for token in tokens)
    return "interpolated" in joined or "predicted" in joined or "low_confidence" in joined


def _load_court_zones(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"court_zones.json not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    zones = payload.get("zones")
    if not isinstance(zones, Mapping):
        raise ValueError(f"{path} missing zones object")
    return {str(name): value for name, value in zones.items() if isinstance(value, list)}


def _load_rally_spans(path: Path, *, world: WorldTrackData) -> tuple[list[RallySpan], str]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_spans = payload.get("spans") if isinstance(payload, Mapping) else payload
        spans = []
        if isinstance(raw_spans, list):
            for index, raw in enumerate(raw_spans):
                if not isinstance(raw, Mapping):
                    continue
                t0 = _optional_float(raw.get("t0"))
                t1 = _optional_float(raw.get("t1"))
                if t0 is None or t1 is None or t1 <= t0:
                    continue
                spans.append(
                    RallySpan(
                        rally_id=str(raw.get("id") or f"rally_{index:03d}"),
                        t0=t0,
                        t1=t1,
                        scope="rally_spans",
                    )
                )
        if spans:
            return spans, "rally_spans"

    t0, t1 = _clip_bounds(world)
    notes = ("rally_spans absent or degenerate; metrics computed over the whole clip",)
    return [RallySpan(rally_id="clip", t0=t0, t1=t1, scope="clip_fallback", notes=notes)], "clip_fallback"


def _clip_bounds(world: WorldTrackData) -> tuple[float, float]:
    if world.ball_frame_count is not None and world.ball_frame_count > 0:
        return 0.0, world.ball_frame_count / world.fps
    times = [frame.t for player in world.players for frame in player.frames]
    if not times:
        return 0.0, 1.0 / world.fps
    return min(times), max(times) + (1.0 / world.fps)


def _load_contact_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events") if isinstance(payload, Mapping) else payload
    if not isinstance(events, list):
        return []
    return [dict(event) for event in events if isinstance(event, Mapping) and event.get("type", "contact") == "contact"]


def _rally_payload(
    span: RallySpan,
    *,
    world: WorldTrackData,
    court_zones: Mapping[str, Any],
    contacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frames_total = _frames_total(span, world=world)
    player_payloads = []
    for player in world.players:
        frames = [frame for frame in player.frames if _time_in_span(frame.t, span)]
        player_contacts = [
            contact
            for contact in contacts
            if _contact_matches_player(contact, player.player_id) and _time_in_span(_contact_time(contact), span)
        ]
        player_payloads.append(
            {
                "player_id": player.player_id,
                "frames_used": len(frames),
                "frames_total": frames_total,
                "coverage_fraction": _coverage(len(frames), frames_total),
                "metrics": _player_metrics(
                    frames,
                    frames_total=frames_total,
                    fps=world.fps,
                    court_zones=court_zones,
                    contacts=player_contacts,
                ),
            }
        )
    return {
        "id": span.rally_id,
        "t0": span.t0,
        "t1": span.t1,
        "rally_scope": span.scope,
        "notes": list(span.notes),
        "players": player_payloads,
    }


def _player_metrics(
    frames: Sequence[PlayerFrame],
    *,
    frames_total: int,
    fps: float,
    court_zones: Mapping[str, Any],
    contacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frames_used = len(frames)
    coverage = _coverage(frames_used, frames_total)
    position_trust = _position_trust(frames, coverage)
    distance, speeds = _distance_and_speeds(frames, fps=fps)
    zone_counts = {"kitchen": 0, "transition": 0, "baseline": 0, "out_of_court": 0}
    proximity_count = 0
    for frame in frames:
        zone = _zone_for_point(frame.track_world_xy, court_zones)
        zone_counts[zone] += 1
        if _near_kitchen_line(frame.track_world_xy, court_zones):
            proximity_count += 1
    zone_value = {
        zone: (count / frames_used if frames_used else 0.0)
        for zone, count in zone_counts.items()
    }
    contact_trust = _contact_trust(contacts, fallback=position_trust)
    contact_positions = [
        _contact_position_payload(contact, frames=frames, fallback_trust=contact_trust)
        for contact in contacts
    ]

    return {
        "distance_covered_m": _metric(round(distance, 6), frames_used, frames_total, position_trust, unit="m"),
        "avg_speed_mps": _metric(
            round(distance / sum(_valid_segment_dts(frames, fps=fps)), 6)
            if speeds and sum(_valid_segment_dts(frames, fps=fps)) > 0
            else 0.0,
            frames_used,
            frames_total,
            position_trust,
            unit="mps",
        ),
        "p95_speed_mps": _metric(round(_percentile(speeds, 95.0), 6), frames_used, frames_total, position_trust, unit="mps"),
        "zone_occupancy": _metric(zone_value, frames_used, frames_total, position_trust, unit="fraction"),
        "kitchen_proximity_s": _metric(round(proximity_count / fps, 6), frames_used, frames_total, position_trust, unit="s"),
        "contact_count": _metric(len(contacts), frames_used, frames_total, contact_trust, unit="count"),
        "contact_positions_world": _metric(contact_positions, frames_used, frames_total, contact_trust, unit="m"),
    }


def _metric(value: Any, frames_used: int, frames_total: int, trust: str, *, unit: str) -> dict[str, Any]:
    return {
        "value": value,
        "unit": unit,
        "frames_used": frames_used,
        "frames_total": frames_total,
        "coverage_fraction": _coverage(frames_used, frames_total),
        "trust": trust,
    }


def _position_trust(frames: Sequence[PlayerFrame], coverage: float) -> str:
    if coverage < MIN_OK_COVERAGE:
        return "estimated"
    if any(frame.estimated_input for frame in frames):
        return "estimated"
    return "ok"


def _contact_trust(contacts: Sequence[Mapping[str, Any]], *, fallback: str) -> str:
    if any(_is_wrist_cue_only(contact) for contact in contacts):
        return "unverified_cue"
    return fallback


def _is_wrist_cue_only(contact: Mapping[str, Any]) -> bool:
    note = str(contact.get("trust_band_note", "")).lower()
    if "wrist-cue-only" in note or "wrist cue only" in note:
        return True
    sources = contact.get("sources")
    if isinstance(sources, Mapping) and "wrist_vel" in sources and not sources.get("ball_inflection"):
        return True
    return False


def _contact_position_payload(
    contact: Mapping[str, Any],
    *,
    frames: Sequence[PlayerFrame],
    fallback_trust: str,
) -> dict[str, Any]:
    trust = "unverified_cue" if _is_wrist_cue_only(contact) else fallback_trust
    contact_t = _contact_time(contact)
    nearest = _nearest_frame(frames, contact_t)
    return {
        "frame": _optional_int(contact.get("frame")),
        "t": contact_t,
        "position_world_xy": list(nearest.track_world_xy) if nearest is not None else None,
        "trust": trust,
        "trust_note": contact.get("trust_band_note"),
    }


def _nearest_frame(frames: Sequence[PlayerFrame], t: float) -> PlayerFrame | None:
    if not frames:
        return None
    return min(frames, key=lambda frame: abs(frame.t - t))


def _distance_and_speeds(frames: Sequence[PlayerFrame], *, fps: float) -> tuple[float, list[float]]:
    distance = 0.0
    speeds: list[float] = []
    sorted_frames = sorted(frames, key=lambda frame: frame.t)
    for previous, current in zip(sorted_frames, sorted_frames[1:]):
        dt = current.t - previous.t
        if dt <= 0 or dt > _max_gap_s(fps):
            continue
        step = math.dist(previous.track_world_xy, current.track_world_xy)
        distance += step
        speeds.append(step / dt)
    return distance, speeds


def _valid_segment_dts(frames: Sequence[PlayerFrame], *, fps: float) -> list[float]:
    dts: list[float] = []
    sorted_frames = sorted(frames, key=lambda frame: frame.t)
    for previous, current in zip(sorted_frames, sorted_frames[1:]):
        dt = current.t - previous.t
        if 0 < dt <= _max_gap_s(fps):
            dts.append(dt)
    return dts


def _max_gap_s(fps: float) -> float:
    return 1.5 / fps


def _zone_for_point(point: tuple[float, float], court_zones: Mapping[str, Any]) -> str:
    if not _point_in_named_zone(point, court_zones, "court"):
        return "out_of_court"
    if _point_in_named_zone(point, court_zones, "near_nvz") or _point_in_named_zone(point, court_zones, "far_nvz"):
        return "kitchen"
    bounds = _court_bounds(court_zones)
    if bounds is not None:
        _min_x, _max_x, min_y, max_y = bounds
        if point[1] <= min_y + BASELINE_BAND_M or point[1] >= max_y - BASELINE_BAND_M:
            return "baseline"
    return "transition"


def _near_kitchen_line(point: tuple[float, float], court_zones: Mapping[str, Any]) -> bool:
    if not _point_in_named_zone(point, court_zones, "court"):
        return False
    kitchen_y = _kitchen_line_abs_y(court_zones)
    if kitchen_y is None:
        return False
    return abs(abs(point[1]) - kitchen_y) <= KITCHEN_PROXIMITY_M


def _kitchen_line_abs_y(court_zones: Mapping[str, Any]) -> float | None:
    values: list[float] = []
    for name in ("near_nvz", "far_nvz"):
        polygon = court_zones.get(name)
        if isinstance(polygon, list):
            for point in polygon:
                if isinstance(point, list) and len(point) >= 2 and abs(float(point[1])) > 1e-9:
                    values.append(abs(float(point[1])))
    return min(values) if values else None


def _point_in_named_zone(point: tuple[float, float], court_zones: Mapping[str, Any], name: str) -> bool:
    polygon = court_zones.get(name)
    return isinstance(polygon, list) and _point_in_polygon(point, polygon)


def _point_in_polygon(point: tuple[float, float], polygon: Sequence[Sequence[float]]) -> bool:
    x, y = point
    inside = False
    count = len(polygon)
    for idx in range(count):
        x1, y1 = float(polygon[idx][0]), float(polygon[idx][1])
        x2, y2 = float(polygon[(idx + 1) % count][0]), float(polygon[(idx + 1) % count][1])
        if _point_on_segment(x, y, x1, y1, x2, y2):
            return True
        intersects = (y1 > y) != (y2 > y)
        if intersects:
            x_intersection = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_intersection:
                inside = not inside
    return inside


def _point_on_segment(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> bool:
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > 1e-9:
        return False
    return min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9


def _court_bounds(court_zones: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    polygon = court_zones.get("court")
    if not isinstance(polygon, list) or not polygon:
        return None
    xs = [float(point[0]) for point in polygon if isinstance(point, list) and len(point) >= 2]
    ys = [float(point[1]) for point in polygon if isinstance(point, list) and len(point) >= 2]
    if not xs or not ys:
        return None
    return min(xs), max(xs), min(ys), max(ys)


def _coaching_card_facts(
    rallies: Sequence[Mapping[str, Any]],
    *,
    source_run_dir: str,
    rally_scope: str,
) -> dict[str, Any]:
    facts = []
    for rally in rallies:
        for player in rally["players"]:
            fact = _select_fact(rally_id=str(rally["id"]), rally_scope=str(rally["rally_scope"]), player=player)
            facts.append(fact)
    return {
        "schema_version": 1,
        "artifact_type": FACTS_ARTIFACT_TYPE,
        "source_run_dir": source_run_dir,
        "rally_scope": rally_scope,
        "priority_rule": [
            "contact_count_when_present",
            "kitchen_proximity_when_positive",
            "distance_covered_when_positive",
            "p95_speed_when_positive",
            "dominant_zone_occupancy",
        ],
        "facts": facts,
    }


def _select_fact(*, rally_id: str, rally_scope: str, player: Mapping[str, Any]) -> dict[str, Any]:
    metrics = player["metrics"]
    candidates = [
        ("contact_count", metrics["contact_count"], lambda metric: metric["value"] > 0),
        ("kitchen_proximity_s", metrics["kitchen_proximity_s"], lambda metric: metric["value"] > 0),
        ("distance_covered_m", metrics["distance_covered_m"], lambda metric: metric["value"] > 0),
        ("p95_speed_mps", metrics["p95_speed_mps"], lambda metric: metric["value"] > 0),
        ("zone_occupancy", metrics["zone_occupancy"], lambda _metric: True),
    ]
    metric_name, metric = next(
        ((name, payload) for name, payload, predicate in candidates if predicate(payload)),
        ("zone_occupancy", metrics["zone_occupancy"]),
    )
    value = metric["value"]
    if metric_name == "zone_occupancy":
        zone, fraction = max(value.items(), key=lambda item: (item[1], item[0]))
        value = {"zone": zone, "fraction": fraction}
    return {
        "rally_id": rally_id,
        "rally_scope": rally_scope,
        "player_id": player["player_id"],
        "metric": metric_name,
        "value": value,
        "unit": metric["unit"],
        "trust": metric["trust"],
        "frames_used": metric["frames_used"],
        "frames_total": metric["frames_total"],
        "coverage_fraction": metric["coverage_fraction"],
    }


def _frames_total(span: RallySpan, *, world: WorldTrackData) -> int:
    if span.scope == "clip_fallback" and world.ball_frame_count is not None:
        return max(1, world.ball_frame_count)
    return max(1, int(round((span.t1 - span.t0) * world.fps)))


def _coverage(frames_used: int, frames_total: int) -> float:
    if frames_total <= 0:
        return 0.0
    return frames_used / frames_total


def _time_in_span(t: float, span: RallySpan) -> bool:
    return span.t0 <= t < span.t1


def _contact_matches_player(contact: Mapping[str, Any], player_id: str) -> bool:
    return _id_string(contact.get("player_id")) == player_id


def _contact_time(contact: Mapping[str, Any]) -> float:
    value = contact.get("t")
    if value is not None:
        return float(value)
    window = contact.get("window")
    if isinstance(window, Mapping) and window.get("t0") is not None and window.get("t1") is not None:
        return (float(window["t0"]) + float(window["t1"])) / 2.0
    return 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _id_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
